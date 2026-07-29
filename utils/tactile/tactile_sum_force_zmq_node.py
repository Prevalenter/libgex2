#!/usr/bin/env python3
"""Publish the measured Fx/Fy/Fz resultant force for each tactile sensor."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.tactile.hand import FingerConfig, load_finger_configs
    from utils.tactile.tactile_zmq_node import ordered_ports_for_fingers
    from utils.tactile.uart import DEFAULT_BAUDRATE, DEFAULT_DEVICE_ID, L3530
else:
    from .hand import FingerConfig, load_finger_configs
    from .tactile_zmq_node import ordered_ports_for_fingers
    from .uart import DEFAULT_BAUDRATE, DEFAULT_DEVICE_ID, L3530


DEFAULT_ENDPOINT = "tcp://127.0.0.1:5561"
DEFAULT_PUBLISH_RATE = 30.0
DEFAULT_FORCE_RESOLUTION_N = 0.1
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_FINGER = "thumb"
FORCE_AXES = ("x", "y", "z")


def parse_int(value: str) -> int:
    return int(value, 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish tactile-sensor Fx/Fy/Fz resultant forces over ZeroMQ."
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help="Single-sensor serial port; bypasses USB-serial-number discovery.",
    )
    parser.add_argument(
        "--finger",
        default=DEFAULT_FINGER,
        help="Name used for a sensor selected with --port.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--publish-rate", type=float, default=DEFAULT_PUBLISH_RATE)
    parser.add_argument("--config-json", default="config/config.json")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--device-id", type=parse_int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument(
        "--force-resolution-n",
        type=float,
        default=DEFAULT_FORCE_RESOLUTION_N,
        help="Newtons represented by one sensor LSB (default: 0.1 N).",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.publish_rate) or args.publish_rate <= 0:
        parser.error("--publish-rate must be a finite number greater than zero")
    if not math.isfinite(args.force_resolution_n) or args.force_resolution_n <= 0:
        parser.error("--force-resolution-n must be a finite number greater than zero")
    return args


def decode_sum_force(raw_force: Sequence[int] | bytes, resolution_n: float) -> list[float]:
    """Decode signed Fx/Fy and unsigned Fz sensor bytes into newtons."""
    raw = np.asarray(
        list(raw_force) if isinstance(raw_force, bytes) else raw_force,
        dtype=np.uint8,
    )
    if raw.shape != (3,):
        raise ValueError(f"expected three sum-force bytes [Fx, Fy, Fz], got shape {raw.shape}")

    fx = int(raw[0]) - 256 if raw[0] >= 128 else int(raw[0])
    fy = int(raw[1]) - 256 if raw[1] >= 128 else int(raw[1])
    fz = int(raw[2])
    return [value * resolution_n for value in (fx, fy, fz)]


def build_payload(
    fingers: Sequence[FingerConfig],
    ports: Sequence[str],
    raw_forces: Sequence[Sequence[int] | bytes],
    resolution_n: float = DEFAULT_FORCE_RESOLUTION_N,
    now_ns: int | None = None,
) -> dict[str, Any]:
    if len(raw_forces) != len(fingers):
        raise ValueError(
            f"expected {len(fingers)} sum-force samples, got {len(raw_forces)}"
        )
    force_xyz_n = [decode_sum_force(raw, resolution_n) for raw in raw_forces]
    stamp_ns = time.time_ns() if now_ns is None else int(now_ns)
    return {
        "stamp_sec": stamp_ns // 1_000_000_000,
        "stamp_nanosec": stamp_ns % 1_000_000_000,
        "fingers": [finger.finger for finger in fingers],
        "serials": [finger.serial for finger in fingers],
        "ports": list(ports),
        "axes": list(FORCE_AXES),
        "unit": "N",
        "force_xyz_n": force_xyz_n,
    }


class TactileSumForceZmqNode:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("missing pyzmq, install with: pip install pyzmq") from exc

        self.zmq = zmq
        self.args = args
        self.context: Any = None
        self.pub_socket: Any = None
        self.fingers: tuple[FingerConfig, ...] = ()
        self.ports: list[str] = []
        self.sensors: list[L3530] = []
        self.running = False

    def start(self) -> None:
        if self.args.port:
            self.fingers = (FingerConfig(self.args.finger, ""),)
            self.ports = [self.args.port]
        else:
            self.fingers = load_finger_configs(self.args.config_json)
            self.ports = ordered_ports_for_fingers(self.fingers)
        self.sensors = [
            L3530(
                port,
                baudrate=self.args.baud,
                device_id=self.args.device_id,
                timeout=self.args.timeout,
            )
            for port in self.ports
        ]

        self.context = self.zmq.Context()
        self.pub_socket = self.context.socket(self.zmq.PUB)
        self.pub_socket.setsockopt(self.zmq.LINGER, 0)
        self.pub_socket.bind(self.args.endpoint)
        self.running = True
        print(
            "Tactile sum-force ZeroMQ node started. "
            f"endpoint={self.args.endpoint}, publish_rate={self.args.publish_rate}, "
            f"fingers={[finger.finger for finger in self.fingers]}, ports={self.ports}",
            flush=True,
        )

    def run(self) -> None:
        interval = 1.0 / self.args.publish_rate
        next_publish = time.monotonic()
        while self.running:
            started = time.monotonic()
            self.update_and_publish()
            next_publish = max(next_publish + interval, started + interval)
            sleep_sec = next_publish - time.monotonic()
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    def update_and_publish(self) -> None:
        if self.pub_socket is None:
            return
        try:
            raw_forces = [sensor.read_sum_force() for sensor in self.sensors]
            payload = build_payload(
                self.fingers,
                self.ports,
                raw_forces,
                resolution_n=self.args.force_resolution_n,
            )
        except Exception as exc:
            print(f"tactile sum-force read/publish warning: {exc}", file=sys.stderr)
            return
        self.pub_socket.send_json(payload)

    def stop(self) -> None:
        self.running = False
        for sensor in self.sensors:
            try:
                sensor.close()
            except Exception as exc:
                print(f"tactile close warning: {exc}", file=sys.stderr)
        self.sensors = []

        if self.pub_socket is not None:
            self.pub_socket.close(0)
            self.pub_socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        node = TactileSumForceZmqNode(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def request_stop(_signum: int, _frame: Any) -> None:
        node.running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node.start()
        node.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        node.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
