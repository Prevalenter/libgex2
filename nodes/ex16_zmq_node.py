"""Publish EX16 joint positions over ZMQ."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libgex import Glove16  # noqa: E402


JOINT_COUNT = 16
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5567"
STATE_TOPIC = "ex16/state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EX16 ZMQ joint-state publisher.")
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--port", help="Serial port, for example COM7")
    connection.add_argument("--serial-number", help="USB serial number")
    parser.add_argument("--left", action="store_true", help="Use left-hand directions")
    parser.add_argument("--state-endpoint", default=DEFAULT_STATE_ENDPOINT)
    parser.add_argument("--state-hz", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_hz <= 0:
        raise ValueError("--state-hz must be greater than zero")

    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

    glove = Glove16(
        port=args.port,
        serial_number=args.serial_number,
        left=args.left,
    )
    glove.connect()

    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.linger = 0
    publisher.bind(args.state_endpoint)
    print(
        f"EX16 state: {args.state_endpoint}, topic={STATE_TOPIC}, "
        f"device={glove.port}",
        flush=True,
    )

    period = 1.0 / args.state_hz
    sequence = 0
    try:
        while True:
            started = time.monotonic()
            positions = [float(value) for value in glove.getjs()]
            if len(positions) != JOINT_COUNT:
                raise RuntimeError(
                    f"Glove16.getjs returned {len(positions)} joints; "
                    f"expected {JOINT_COUNT}"
                )
            payload = {
                "name": "ex16",
                "sequence": sequence,
                "timestamp": time.time(),
                "urdf_deg": positions,
            }
            publisher.send_string(
                f"{STATE_TOPIC} {json.dumps(payload, separators=(',', ':'))}"
            )
            sequence += 1
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("Stopping EX16 joint-state node...", flush=True)
    finally:
        glove.off()
        publisher.close()


if __name__ == "__main__":
    main()
