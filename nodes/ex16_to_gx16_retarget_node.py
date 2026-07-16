"""Subscribe to EX16 states, retarget them, and optionally command GX16."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libgex import Hand16  # noqa: E402
from libgex.gx16.libgx16 import JOINT_MOTOR_DIRECTIONS  # noqa: E402
from libgex.gx16.retargeting import EX16ToGX16Retargeting  # noqa: E402


JOINT_COUNT = 16
EX16_TOPIC = "ex16/state"
GX16_TOPIC = "gx16/retarget_state"
DEFAULT_EX16_ENDPOINT = "tcp://127.0.0.1:5567"
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5568"
GX16_CONFIG = REPO_ROOT / "libgex" / "gx16" / "config.yaml"


def parse_args() -> argparse.Namespace:
    with GX16_CONFIG.open("r", encoding="utf-8") as file:
        default_serial = yaml.safe_load(file)["BASIC"].get("SERIAL_NUMBER")

    parser = argparse.ArgumentParser(description="EX16-to-GX16 retargeting node.")
    parser.add_argument("--ex16-endpoint", default=DEFAULT_EX16_ENDPOINT)
    parser.add_argument("--state-endpoint", default=DEFAULT_STATE_ENDPOINT)
    parser.add_argument("--command-hz", type=float, default=30.0)
    parser.add_argument("--max-step-deg", type=float, default=3.0)
    parser.add_argument(
        "--enable-output",
        action="store_true",
        help="Actually connect to and command GX16; otherwise only publish results.",
    )
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--gx16-port")
    connection.add_argument("--gx16-serial-number", default=default_serial)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command_hz <= 0:
        raise ValueError("--command-hz must be greater than zero")
    if args.max_step_deg <= 0:
        raise ValueError("--max-step-deg must be greater than zero")
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

    retargeter = EX16ToGX16Retargeting()
    directions = np.asarray(JOINT_MOTOR_DIRECTIONS, dtype=np.float64)
    if directions.shape != (JOINT_COUNT,) or not np.all(np.isin(directions, [-1, 1])):
        raise ValueError("GX16 JOINT_MOTOR_DIRECTIONS must contain 16 values of 1/-1")

    hand = None
    previous_urdf_deg = np.zeros(JOINT_COUNT)
    if args.enable_output:
        hand = Hand16(
            port=args.gx16_port,
            serial_number=None if args.gx16_port else args.gx16_serial_number,
        )
        hand.connect()
        previous_urdf_deg = np.asarray(hand.getjs(), dtype=float) * directions
        print(f"GX16 output enabled on {hand.port}.", flush=True)
    else:
        print("Preview mode: GX16 output is disabled.", flush=True)

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    publisher = context.socket(zmq.PUB)
    subscriber.linger = 0
    publisher.linger = 0
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, EX16_TOPIC)
    subscriber.connect(args.ex16_endpoint)
    publisher.bind(args.state_endpoint)
    print(f"EX16 input: {args.ex16_endpoint}, topic={EX16_TOPIC}", flush=True)
    print(f"GX16 state: {args.state_endpoint}, topic={GX16_TOPIC}", flush=True)

    period = 1.0 / args.command_hz
    next_command_time = time.monotonic()
    sequence = 0
    try:
        while True:
            message = subscriber.recv_string()
            now = time.monotonic()
            if now < next_command_time:
                continue
            next_command_time = now + period

            topic, encoded = message.split(" ", 1)
            if topic != EX16_TOPIC:
                continue
            ex16_state = json.loads(encoded)
            ex16_deg = np.asarray(ex16_state["urdf_deg"], dtype=np.float64)
            desired_urdf_deg = np.rad2deg(retargeter.retarget(ex16_deg))

            delta = np.clip(
                desired_urdf_deg - previous_urdf_deg,
                -args.max_step_deg,
                args.max_step_deg,
            )
            commanded_urdf_deg = previous_urdf_deg + delta
            previous_urdf_deg = commanded_urdf_deg
            motor_deg = commanded_urdf_deg * directions

            if hand is not None:
                hand.setjs(motor_deg.tolist())

            payload = {
                "name": "gx16_retarget",
                "sequence": sequence,
                "timestamp": time.time(),
                "source_timestamp": ex16_state.get("timestamp"),
                "ex16_urdf_deg": ex16_deg.tolist(),
                "desired_urdf_deg": desired_urdf_deg.tolist(),
                "commanded_urdf_deg": commanded_urdf_deg.tolist(),
                "motor_deg": motor_deg.tolist(),
                "output_enabled": hand is not None,
            }
            publisher.send_string(
                f"{GX16_TOPIC} {json.dumps(payload, separators=(',', ':'))}"
            )
            sequence += 1
            if sequence % max(1, round(args.command_hz)) == 0:
                print(np.round(commanded_urdf_deg, 1).tolist(), flush=True)
    except KeyboardInterrupt:
        print("Stopping EX16-to-GX16 retargeting node...", flush=True)
    finally:
        subscriber.close()
        publisher.close()
        if hand is not None:
            hand.off()


if __name__ == "__main__":
    main()
