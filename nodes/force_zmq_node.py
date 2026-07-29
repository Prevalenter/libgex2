"""Read a serial force sensor and publish each value directly over ZMQ."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 2400
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5577"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force sensor ZMQ publisher.")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Force sensor serial port.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="ZMQ PUB endpoint to bind.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.01,
        help="Seconds to wait when the serial receive buffer has no complete value.",
    )
    args = parser.parse_args(argv)
    if args.baudrate <= 0:
        parser.error("--baudrate must be greater than zero")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be a finite non-negative number")
    if not math.isfinite(args.poll_interval) or args.poll_interval < 0:
        parser.error("--poll-interval must be a finite non-negative number")
    return args


def _load_dependencies():
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

    try:
        from utils.force_measurement.base import ForceSensor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pyserial is required: python -m pip install pyserial") from exc
    return zmq, ForceSensor


def publish_values(sensor, publisher, poll_interval: float, running) -> None:
    """Publish every complete sensor value as a plain UTF-8 floating-point number."""
    while running():
        values = sensor.read_buf_data()
        if values:
            for value in values:
                publisher.send_string(str(float(value)))
        elif poll_interval:
            time.sleep(poll_interval)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    zmq, force_sensor_type = _load_dependencies()
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.linger = 0
    sensor = None
    started = False
    keep_running = True

    def request_stop(*_args) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        sensor = force_sensor_type(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
        publisher.bind(args.endpoint)
        sensor.start()
        started = True
        print(
            f"Force sensor PUB: {args.endpoint}, device={args.port}, "
            f"baudrate={args.baudrate}",
            flush=True,
        )
        publish_values(sensor, publisher, args.poll_interval, lambda: keep_running)
    except KeyboardInterrupt:
        request_stop()
    finally:
        if sensor is not None:
            if started:
                try:
                    sensor.stop()
                except Exception as exc:
                    print(f"Warning: failed to stop force sensor: {exc}", flush=True)
            try:
                sensor.ser.close()
            except Exception as exc:
                print(f"Warning: failed to close force sensor: {exc}", flush=True)
        publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
