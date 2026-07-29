#!/usr/bin/env python3

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.tactile.hand import FingerConfig, PortInfo, list_port_infos, load_finger_configs, split_connected_serials
    from utils.tactile.uart import DEFAULT_BAUDRATE, DEFAULT_DEVICE_ID, L3530, TACTILE_AXES, TACTILE_POINTS
else:
    from .hand import FingerConfig, PortInfo, list_port_infos, load_finger_configs, split_connected_serials
    from .uart import DEFAULT_BAUDRATE, DEFAULT_DEVICE_ID, L3530, TACTILE_AXES, TACTILE_POINTS


DEFAULT_ENDPOINT = "tcp://127.0.0.1:5560"
DEFAULT_PUBLISH_RATE = 30.0


def parse_int(value: str) -> int:
    return int(value, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish thumb/index/middle tactile frames over ZeroMQ.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--publish-rate", type=float, default=DEFAULT_PUBLISH_RATE)
    parser.add_argument("--config-json", default="config/config.json")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--device-id", type=parse_int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--timeout", type=float, default=0.05)
    return parser.parse_args()


def port_info_payload(port_infos: Sequence[PortInfo]) -> list[dict[str, Any]]:
    return [
        {
            "device": info.device,
            "serial": info.serial_number,
            "vid": info.vid,
            "pid": info.pid,
            "hwid": info.hwid,
        }
        for info in port_infos
    ]


def ordered_ports_for_fingers(
    fingers: Sequence[FingerConfig],
    port_infos: Sequence[PortInfo] | None = None,
) -> list[str]:
    infos = list(list_port_infos() if port_infos is None else port_infos)
    serials = tuple(finger.serial for finger in fingers)
    connected, missing = split_connected_serials(serials, port_infos=infos)
    if missing:
        missing_fingers = [
            {"finger": finger.finger, "serial": finger.serial}
            for finger in fingers
            if finger.serial in missing
        ]
        raise RuntimeError(
            "missing configured tactile sensors: "
            f"{missing_fingers}; available_ports={port_info_payload(infos)}"
        )

    port_by_serial = dict(connected)
    return [port_by_serial[finger.serial] for finger in fingers]


def build_payload(
    fingers: Sequence[FingerConfig],
    ports: Sequence[str],
    force_frames,
    now_ns: int | None = None,
) -> dict[str, Any]:
    frames = np.asarray(force_frames, dtype=np.float32)
    expected_shape = (len(fingers), TACTILE_POINTS, TACTILE_AXES)
    if frames.shape != expected_shape:
        raise ValueError(f"expected force frame shape {expected_shape}, got {frames.shape}")

    stamp_ns = time.time_ns() if now_ns is None else int(now_ns)
    return {
        "stamp_sec": stamp_ns // 1_000_000_000,
        "stamp_nanosec": stamp_ns % 1_000_000_000,
        "fingers": [finger.finger for finger in fingers],
        "serials": [finger.serial for finger in fingers],
        "ports": list(ports),
        "shape": list(frames.shape),
        "force_frames": frames.tolist(),
    }


class TactileZmqNode:
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
            "Tactile ZeroMQ node started. "
            f"endpoint={self.args.endpoint}, publish_rate={self.args.publish_rate}, "
            f"fingers={[finger.finger for finger in self.fingers]}, "
            f"serials={[finger.serial for finger in self.fingers]}, ports={self.ports}",
            flush=True,
        )

    def run(self) -> None:
        interval = 1.0 / self.args.publish_rate if self.args.publish_rate > 0 else 0.01
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
            force_frames = [sensor.read_force_frame() for sensor in self.sensors]
            payload = build_payload(self.fingers, self.ports, force_frames)
        except Exception as exc:
            print(f"tactile read/publish warning: {exc}", file=sys.stderr)
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
            try:
                self.pub_socket.close(0)
            except Exception:
                pass
            self.pub_socket = None

        if self.context is not None:
            try:
                self.context.term()
            except Exception:
                pass
            self.context = None


def main() -> int:
    args = parse_args()
    try:
        node = TactileZmqNode(args)
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
    sys.exit(main())
