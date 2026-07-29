"""Benchmark real GX16 setjs command frequency with a constant zero command."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from libgex2 import Hand16  # noqa: E402


JOINT_COUNT = 16
DEFAULT_SERIAL_NUMBER = "FTAKRP3A"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def latency_timer_path(port: str | None) -> Path | None:
    if not port:
        return None
    name = Path(port).name
    path = Path("/sys/bus/usb-serial/devices") / name / "latency_timer"
    return path if path.exists() else None


def read_latency_timer(port: str | None) -> str | None:
    path = latency_timer_path(port)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def summarize_ms(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    ordered = sorted(samples_ms)
    return {
        "count": float(len(ordered)),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p90_ms": ordered[min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)],
        "p99_ms": ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)],
    }


def print_latency_summary(title: str, samples_ms: Sequence[float]) -> None:
    stats = summarize_ms(samples_ms)
    count = int(stats["count"])
    mean_hz = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else float("nan")
    median_hz = 1000.0 / stats["median_ms"] if stats["median_ms"] > 0 else float("nan")
    print()
    print(title)
    print(f"  samples:        {count}")
    print(f"  mean rate:      {mean_hz:.2f} Hz")
    print(f"  median rate:    {median_hz:.2f} Hz")
    print(f"  mean latency:   {stats['mean_ms']:.2f} ms")
    print(f"  median latency: {stats['median_ms']:.2f} ms")
    print(f"  min latency:    {stats['min_ms']:.2f} ms")
    print(f"  p90 latency:    {stats['p90_ms']:.2f} ms")
    print(f"  p99 latency:    {stats['p99_ms']:.2f} ms")
    print(f"  max latency:    {stats['max_ms']:.2f} ms")


def print_summary(
    setjs_ms: Sequence[float],
    loop_ms: Sequence[float],
    elapsed_s: float,
    getjs_ms: Sequence[float] | None = None,
) -> None:
    if not loop_ms:
        raise ValueError("loop_ms must not be empty")
    actual_hz = len(loop_ms) / elapsed_s if elapsed_s > 0 else float("nan")
    print()
    print("GX16 control-loop benchmark summary")
    print(f"  samples:        {len(loop_ms)}")
    print(f"  elapsed:        {elapsed_s:.3f} s")
    print(f"  actual rate:    {actual_hz:.2f} Hz")
    print_latency_summary("setjs latency", setjs_ms)
    if getjs_ms is not None:
        print_latency_summary("getjs latency", getjs_ms)
    print_latency_summary("total loop latency", loop_ms)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure real GX16 control frequency by repeatedly sending all-zero setjs commands."
    )
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--port", help="Serial port, for example /dev/ttyUSB0")
    connection.add_argument(
        "--serial-number",
        default=DEFAULT_SERIAL_NUMBER,
        help="USB serial number used when --port is not set.",
    )
    parser.add_argument("--samples", type=positive_int, default=100)
    parser.add_argument("--warmup", type=non_negative_int, default=5)
    parser.add_argument(
        "--period",
        type=finite_float,
        default=0.0,
        help="Optional command period in seconds. Default 0 sends as fast as setjs returns.",
    )
    parser.add_argument("--curr-limit", type=int, default=1000)
    parser.add_argument("--goal-current", type=int, default=600)
    parser.add_argument("--goal-pwm", type=int, default=200)
    parser.add_argument(
        "--read-position",
        action="store_true",
        help="After each setjs command, also call getjs and include read time in the loop frequency.",
    )
    parser.add_argument(
        "--keep-torque-on-exit",
        action="store_true",
        help="Do not torque off the hand after the benchmark.",
    )
    args = parser.parse_args(argv)
    if args.period < 0:
        parser.error("--period must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = [0.0] * JOINT_COUNT
    hand = Hand16(port=args.port, serial_number=args.serial_number)
    print("Connecting GX16...")
    hand.connect(
        curr_limit=args.curr_limit,
        goal_current=args.goal_current,
        goal_pwm=args.goal_pwm,
    )
    print(f"Connected: {hand.port}")
    latency = read_latency_timer(hand.port)
    if latency is not None:
        print(f"USB latency_timer: {latency} ms ({latency_timer_path(hand.port)})")
    print(
        f"Command: all-zero setjs, warmup={args.warmup}, "
        f"samples={args.samples}, period={args.period:g}s, "
        f"read_position={args.read_position}"
    )

    setjs_ms: list[float] = []
    getjs_ms: list[float] = []
    loop_ms: list[float] = []
    started_measurement = None
    try:
        total = args.warmup + args.samples
        for index in range(total):
            loop_started = time.perf_counter()
            hand.setjs(command)
            after_setjs = time.perf_counter()
            read_duration_ms = None
            if args.read_position:
                positions = hand.getjs()
                if len(positions) != JOINT_COUNT:
                    raise RuntimeError(
                        f"Hand16.getjs returned {len(positions)} joints; "
                        f"expected {JOINT_COUNT}"
                    )
                after_read = time.perf_counter()
                read_duration_ms = (after_read - after_setjs) * 1000.0
            else:
                after_read = after_setjs
            write_duration_ms = (after_setjs - loop_started) * 1000.0
            total_duration_ms = (after_read - loop_started) * 1000.0
            if index >= args.warmup:
                if started_measurement is None:
                    started_measurement = loop_started
                setjs_ms.append(write_duration_ms)
                if read_duration_ms is not None:
                    getjs_ms.append(read_duration_ms)
                loop_ms.append(total_duration_ms)
            if args.period > 0:
                remaining = args.period - (time.perf_counter() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
            if (index + 1) % 10 == 0 or index + 1 == total:
                phase = "warmup" if index < args.warmup else "measure"
                read_text = (
                    ""
                    if read_duration_ms is None
                    else f", getjs={read_duration_ms:.2f} ms"
                )
                print(
                    f"{index + 1:4d}/{total} {phase}: "
                    f"setjs={write_duration_ms:.2f} ms"
                    f"{read_text}, loop={total_duration_ms:.2f} ms",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nInterrupted; summarizing collected samples.")
    finally:
        if not args.keep_torque_on_exit:
            try:
                hand.off()
                print("GX16 torque disabled.")
            except Exception as exc:
                print(f"Warning: failed to disable GX16 torque: {exc}", file=sys.stderr)

    if not loop_ms:
        print("No measured samples collected.", file=sys.stderr)
        return 1
    elapsed_s = time.perf_counter() - (started_measurement or time.perf_counter())
    print_summary(
        setjs_ms,
        loop_ms,
        elapsed_s,
        getjs_ms if args.read_position else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
