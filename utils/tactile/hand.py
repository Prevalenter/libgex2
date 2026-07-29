from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from serial.tools.list_ports import comports

from .uart import DEFAULT_BAUDRATE, DEFAULT_DEVICE_ID, L3530

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_JSON = PACKAGE_DIR / "config" / "config.json"
DEFAULT_FINGER_ORDER = ("thumb", "index", "middle")


@dataclass(frozen=True)
class PortInfo:
    device: str
    hwid: str
    vid: int | None
    pid: int | None
    serial_number: str | None


@dataclass(frozen=True)
class FingerConfig:
    finger: str
    serial: str


def resolve_config_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PACKAGE_DIR / value


def _read_json_config(config_path: str | Path) -> dict:
    with resolve_config_path(config_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_finger_configs(config_path: str | Path = DEFAULT_CONFIG_JSON) -> tuple[FingerConfig, ...]:
    data = _read_json_config(config_path)

    fingers = data.get("fingers")
    if fingers is None:
        serials = data.get("default_serials")
        if not isinstance(serials, list) or len(serials) != len(DEFAULT_FINGER_ORDER):
            raise ValueError("config.json must contain fingers or three default_serials")
        if not all(isinstance(item, str) and item for item in serials):
            raise ValueError("default_serials must contain non-empty strings")
        return tuple(FingerConfig(finger, serial) for finger, serial in zip(DEFAULT_FINGER_ORDER, serials))

    if not isinstance(fingers, list) or len(fingers) != len(DEFAULT_FINGER_ORDER):
        raise ValueError("fingers must bind exactly thumb, index, and middle")

    configs = []
    for expected_finger, item in zip(DEFAULT_FINGER_ORDER, fingers):
        if not isinstance(item, dict):
            raise ValueError("each finger config must be an object")
        finger = item.get("finger")
        serial = item.get("serial")
        if finger != expected_finger:
            raise ValueError(f"expected finger {expected_finger!r}, got {finger!r}")
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"finger {finger!r} must have a non-empty serial")
        configs.append(FingerConfig(finger=finger, serial=serial))
    return tuple(configs)


def load_default_serials(config_path: str | Path = DEFAULT_CONFIG_JSON) -> tuple[str, ...]:
    return tuple(config.serial for config in load_finger_configs(config_path))


DEFAULT_SERIALS = load_default_serials()


def list_port_infos() -> list[PortInfo]:
    return [
        PortInfo(
            device=p.device,
            hwid=p.hwid,
            vid=p.vid,
            pid=p.pid,
            serial_number=p.serial_number,
        )
        for p in comports()
    ]


def split_connected_serials(
    serial_numbers: list[str] | tuple[str, ...],
    port_infos: list[PortInfo] | None = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    infos = list_port_infos() if port_infos is None else port_infos
    devices_by_serial: dict[str, list[str]] = {}
    for info in infos:
        if info.serial_number:
            devices_by_serial.setdefault(info.serial_number, []).append(info.device)

    connected = []
    missing = []
    for serial_number in serial_numbers:
        devices = devices_by_serial.get(serial_number, [])
        if len(devices) == 1:
            connected.append((serial_number, devices[0]))
        else:
            missing.append(serial_number)
    return connected, missing


def port_name_from_serial(serial_number: str) -> str:
    if not serial_number or not isinstance(serial_number, str):
        raise ValueError("serial_number must be a non-empty string")

    matches = [p.device for p in comports() if p.serial_number == serial_number]
    if not matches:
        raise RuntimeError(f"no serial port found for serial_number={serial_number!r}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple ports match serial_number={serial_number!r}: {matches}")
    return matches[0]


class HandTactile:
    def __init__(
        self,
        serial_numbers: list[str] | tuple[str, ...] = DEFAULT_SERIALS,
        baudrate: int = DEFAULT_BAUDRATE,
        device_id: int = DEFAULT_DEVICE_ID,
        timeout: float = 0.05,
        boot_delay: float = 0.8,
    ):
        self.serial_numbers = tuple(serial_numbers)
        self.ports = [port_name_from_serial(serial_number) for serial_number in self.serial_numbers]
        self.sensors = [
            L3530(
                port,
                baudrate=baudrate,
                device_id=device_id,
                timeout=timeout,
                boot_delay=boot_delay,
            )
            for port in self.ports
        ]

    def __enter__(self) -> "HandTactile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for sensor in self.sensors:
            sensor.close()

    def read_raw(self):
        return [sensor.read_tactile_raw() for sensor in self.sensors]

    def read_frames(self):
        return [sensor.read_force_frame() for sensor in self.sensors]

    def read_sum_forces(self):
        return [sensor.read_sum_force() for sensor in self.sensors]
