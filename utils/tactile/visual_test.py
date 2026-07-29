from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.tactile.hand import list_port_infos, load_finger_configs, split_connected_serials
    from utils.tactile.uart import DEFAULT_BAUDRATE, DEFAULT_DEVICE_ID, L3530
    from utils.tactile.viz import load_sensor_xy, show_live
else:
    from .hand import list_port_infos, load_finger_configs, split_connected_serials
    from .uart import DEFAULT_BAUDRATE, DEFAULT_DEVICE_ID, L3530
    from .viz import load_sensor_xy, show_live


def parse_int(value: str) -> int:
    return int(value, 0)


def print_ports() -> None:
    for info in list_port_infos():
        print(f"{info.device}\n  hwid: {info.hwid}\n  vid/pid: {info.vid}/{info.pid}\n  serial: {info.serial_number}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual test for thumb, index, and middle tactile sensors.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--record", help="optional .npy output path, relative to utils/tactile")
    parser.add_argument("--config-json", default="config/config.json")
    parser.add_argument("--sensor-config", default="config/L3530.xlsx")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--device-id", type=parse_int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--timeout", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fingers = load_finger_configs(args.config_json)
    serials = tuple(finger.serial for finger in fingers)
    connected, missing = split_connected_serials(serials)
    port_by_serial = dict(connected)

    print("configured:", [{"finger": finger.finger, "serial": finger.serial} for finger in fingers])
    print(
        "connected:",
        [
            {"finger": finger.finger, "serial": finger.serial, "port": port_by_serial[finger.serial]}
            for finger in fingers
            if finger.serial in port_by_serial
        ],
    )
    if missing:
        print("missing:", [{"finger": finger.finger, "serial": finger.serial} for finger in fingers if finger.serial in missing])
    if not connected:
        print("no configured tactile devices are connected; showing three gray placeholders")
        print_ports()

    xy = load_sensor_xy(args.sensor_config)
    sensors = []
    inactive = []
    titles = []

    for finger in fingers:
        port = port_by_serial.get(finger.serial)
        inactive.append(port is None)
        status = port if port is not None else "missing"
        titles.append(f"{finger.finger.capitalize()} | SN={finger.serial} | {status}")
        if port is None:
            sensors.append(None)
        else:
            sensors.append(
                L3530(
                    port,
                    baudrate=args.baud,
                    device_id=args.device_id,
                    timeout=args.timeout,
                )
            )

    zero_frame = np.zeros((xy.shape[0], 3), dtype=np.float32)

    def read_frames():
        frames = []
        for sensor in sensors:
            frames.append(zero_frame.copy() if sensor is None else sensor.read_force_frame())
        return frames

    try:
        show_live(
            frame_reader=read_frames,
            xy=xy,
            n_sensors=len(fingers),
            fps=args.fps,
            record_path=args.record,
            titles=titles,
            inactive=inactive,
        )
    finally:
        for sensor in sensors:
            if sensor is not None:
                sensor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
