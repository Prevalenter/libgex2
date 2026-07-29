from __future__ import annotations

import argparse
import time

from .hand import DEFAULT_SERIALS, HandTactile, list_port_infos
from .uart import (
    DEFAULT_BAUDRATE,
    DEFAULT_DEVICE_ID,
    L3530,
    TACTILE_ADDR,
    hexdump,
    scan_device_ids,
)


def parse_int(value: str) -> int:
    return int(value, 0)


def add_serial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--serials", nargs="*", default=None, help="USB serial numbers")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--device-id", type=parse_int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--timeout", type=float, default=0.05)


def cmd_ports(_args: argparse.Namespace) -> int:
    for info in list_port_infos():
        print(f"{info.device}\n  hwid: {info.hwid}\n  vid/pid: {info.vid}/{info.pid}\n  serial: {info.serial_number}")
    return 0


def cmd_sniff(args: argparse.Namespace) -> int:
    results = scan_device_ids(
        args.port,
        start_addr=args.start_addr,
        read_len=args.read_len,
        baudrate=args.baud,
        timeout_s=args.timeout_s,
        send_setup=not args.no_setup,
    )
    found = False
    for device_id, data, error in results:
        if data is None:
            print(f"dev=0x{device_id:02X}: no response ({error})")
            continue
        found = True
        print(f"dev=0x{device_id:02X}: {hexdump(data)}")
    return 0 if found else 1


def cmd_live(args: argparse.Namespace) -> int:
    from .viz import load_sensor_xy, show_live

    xy = load_sensor_xy(args.config)
    if args.port:
        sensor = L3530(
            args.port,
            baudrate=args.baud,
            device_id=args.device_id,
            timeout=args.timeout,
        )
        try:
            show_live(
                frame_reader=lambda: [sensor.read_force_frame()],
                xy=xy,
                n_sensors=1,
                fps=args.fps,
                record_path=args.record,
                titles=[args.port],
                vmin=args.vmin,
                vmax=args.vmax,
            )
        finally:
            sensor.close()
        return 0

    serials = tuple(args.serials or DEFAULT_SERIALS)
    with HandTactile(
        serials,
        baudrate=args.baud,
        device_id=args.device_id,
        timeout=args.timeout,
    ) as hand:
        show_live(
            frame_reader=hand.read_frames,
            xy=xy,
            n_sensors=len(serials),
            fps=args.fps,
            record_path=args.record,
            titles=[f"SN={serial}" for serial in serials],
            vmin=args.vmin,
            vmax=args.vmax,
        )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .viz import load_sensor_xy, show_replay

    xy = load_sensor_xy(args.config)
    show_replay(args.input, xy=xy, fps=args.fps, save_mp4=args.save_mp4, vmin=args.vmin, vmax=args.vmax)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from .viz import load_sensor_xy, save_plot

    xy = load_sensor_xy(args.config)
    save_plot(args.input, args.output, xy=xy, vmin=args.vmin, vmax=args.vmax)
    return 0


def cmd_force(args: argparse.Namespace) -> int:
    serials = tuple(args.serials or DEFAULT_SERIALS)
    count = args.count
    with HandTactile(
        serials,
        baudrate=args.baud,
        device_id=args.device_id,
        timeout=args.timeout,
    ) as hand:
        index = 0
        while count <= 0 or index < count:
            values = hand.read_sum_forces()
            print(index, [value.tolist() for value in values])
            index += 1
            if count <= 0 or index < count:
                time.sleep(args.interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m utils.tactile")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ports = subparsers.add_parser("ports", help="list serial ports")
    ports.set_defaults(func=cmd_ports)

    sniff = subparsers.add_parser("sniff", help="scan device IDs on one port")
    sniff.add_argument("--port", required=True)
    sniff.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    sniff.add_argument("--start-addr", type=parse_int, default=TACTILE_ADDR)
    sniff.add_argument("--read-len", type=parse_int, default=0x20)
    sniff.add_argument("--timeout-s", type=float, default=0.8)
    sniff.add_argument("--no-setup", action="store_true", help="skip the setup write used by the old probe script")
    sniff.set_defaults(func=cmd_sniff)

    live = subparsers.add_parser("live", help="show live tactile frames")
    add_serial_args(live)
    live.add_argument("--port", help="single serial port; overrides --serials")
    live.add_argument("--fps", type=int, default=10)
    live.add_argument("--record", help="optional .npy output path, relative to utils/tactile")
    live.add_argument("--config", default="config/L3530.xlsx")
    live.add_argument("--vmin", type=float, default=0.0)
    live.add_argument("--vmax", type=float, default=200.0)
    live.set_defaults(func=cmd_live)

    replay = subparsers.add_parser("replay", help="replay a recorded .npy file")
    replay.add_argument("--input", required=True)
    replay.add_argument("--fps", type=int, default=10)
    replay.add_argument("--save-mp4")
    replay.add_argument("--config", default="config/L3530.xlsx")
    replay.add_argument("--vmin", type=float, default=0.0)
    replay.add_argument("--vmax", type=float, default=300.0)
    replay.set_defaults(func=cmd_replay)

    plot = subparsers.add_parser("plot", help="save a static first-frame plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", default="data/L3530.png")
    plot.add_argument("--config", default="config/L3530.xlsx")
    plot.add_argument("--vmin", type=float, default=0.0)
    plot.add_argument("--vmax", type=float, default=200.0)
    plot.set_defaults(func=cmd_plot)

    force = subparsers.add_parser("force", help="read summed force values")
    add_serial_args(force)
    force.add_argument("--count", type=int, default=1, help="<=0 means loop forever")
    force.add_argument("--interval", type=float, default=0.01)
    force.set_defaults(func=cmd_force)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
