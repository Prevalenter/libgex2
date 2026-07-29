#!/usr/bin/env python3
"""Own and expose one robot/exoskeleton finger Dynamixel bus over ZMQ."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libgex.dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler  # noqa: E402


ROBOT_IDS = (1, 2, 3, 4)
EXOSKELETON_IDS = (21, 22, 23, 24)
ALL_IDS = ROBOT_IDS + EXOSKELETON_IDS
DEFAULT_PORT = "/dev/ttyUSB1"
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_COMMAND_ENDPOINT = "tcp://127.0.0.1:5580"
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5581"
PROTOCOL_VERSION = 2.0
POSITION_UNIT_DEG = 0.087891
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132


def finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_motor_id(value: Any) -> int:
    try:
        motor_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid motor ID: {value!r}") from exc
    if motor_id not in ALL_IDS:
        raise ValueError(f"motor ID must be one of {list(ALL_IDS)}, got {motor_id}")
    return motor_id


def raw_to_degrees(raw: int) -> float:
    signed = raw - 2**32 if raw >= 2**31 else raw
    return signed * POSITION_UNIT_DEG


def degrees_to_raw(degrees: float) -> int:
    return int(round(degrees / POSITION_UNIT_DEG)) & 0xFFFFFFFF


def ok(result: Any = None) -> dict[str, Any]:
    return {"ok": True, "result": {} if result is None else result, "error": None}


def error(message: Any) -> dict[str, Any]:
    return {"ok": False, "result": None, "error": str(message)}


class FingerPairBus:
    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        read_only: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.read_only = read_only
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.lock = threading.Lock()
        self.connected = False

    def connect(self) -> None:
        if not self.port_handler.openPort():
            raise RuntimeError(f"failed to open motor port {self.port}")
        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise RuntimeError(f"failed to set motor baudrate {self.baudrate}")
        missing = []
        with self.lock:
            for motor_id in ALL_IDS:
                _model, comm_result, packet_error = self.packet_handler.ping(
                    self.port_handler, motor_id
                )
                if comm_result != COMM_SUCCESS or packet_error:
                    missing.append(motor_id)
        if missing:
            self.port_handler.closePort()
            raise RuntimeError(f"motor IDs not responding: {missing}")
        self.connected = True

    def _check(self, motor_id: int, comm_result: int, packet_error: int) -> None:
        if comm_result != COMM_SUCCESS:
            detail = self.packet_handler.getTxRxResult(comm_result)
            raise RuntimeError(f"motor {motor_id}: {detail}")
        if packet_error:
            detail = self.packet_handler.getRxPacketError(packet_error)
            raise RuntimeError(f"motor {motor_id}: {detail}")

    def _read_position_unlocked(self, motor_id: int) -> float:
        raw, comm_result, packet_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, ADDR_PRESENT_POSITION
        )
        self._check(motor_id, comm_result, packet_error)
        return raw_to_degrees(raw)

    def read_positions(self) -> dict[int, float]:
        with self.lock:
            return {motor_id: self._read_position_unlocked(motor_id) for motor_id in ALL_IDS}

    def _read_torque_unlocked(self, motor_id: int) -> bool:
        value, comm_result, packet_error = self.packet_handler.read1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE
        )
        self._check(motor_id, comm_result, packet_error)
        return bool(value)

    def read_torque_states(self) -> dict[int, bool]:
        with self.lock:
            return {motor_id: self._read_torque_unlocked(motor_id) for motor_id in ALL_IDS}

    def _write_position_unlocked(self, motor_id: int, position_deg: float) -> None:
        comm_result, packet_error = self.packet_handler.write4ByteTxRx(
            self.port_handler,
            motor_id,
            ADDR_GOAL_POSITION,
            degrees_to_raw(position_deg),
        )
        self._check(motor_id, comm_result, packet_error)

    def set_position(self, motor_id: int, position_deg: float, max_step_deg: float) -> float:
        if self.read_only:
            raise RuntimeError("motor node is in read-only mode")
        with self.lock:
            current = self._read_position_unlocked(motor_id)
            if abs(position_deg - current) > max_step_deg:
                raise ValueError(
                    f"motor {motor_id} target differs from current position by "
                    f"{abs(position_deg - current):.2f} deg; limit is {max_step_deg:.2f} deg"
                )
            self._write_position_unlocked(motor_id, position_deg)
        return current

    def set_torque(self, enabled: bool) -> None:
        if self.read_only:
            raise RuntimeError("motor node is in read-only mode")
        with self.lock:
            if enabled:
                # Hold the measured pose before enabling torque so stale goal
                # registers cannot make either finger jump unexpectedly.
                positions = {
                    motor_id: self._read_position_unlocked(motor_id)
                    for motor_id in ALL_IDS
                }
                for motor_id, position in positions.items():
                    self._write_position_unlocked(motor_id, position)
            for motor_id in ALL_IDS:
                comm_result, packet_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, ADDR_TORQUE_ENABLE, int(enabled)
                )
                self._check(motor_id, comm_result, packet_error)

    def close(self, disable_torque: bool = True) -> None:
        if self.connected and disable_torque and not self.read_only:
            try:
                self.set_torque(False)
            except Exception as exc:
                print(f"Warning: failed to disable motor torque: {exc}", file=sys.stderr)
        if self.port_handler.ser is not None:
            self.port_handler.closePort()
        self.connected = False


class FingerPairNode:
    def __init__(self, args: argparse.Namespace, bus: FingerPairBus | None = None) -> None:
        self.args = args
        self.bus = bus or FingerPairBus(args.port, args.baudrate, args.read_only)
        self.running = True
        self.last_error: str | None = None
        self.last_positions: dict[int, float] = {}
        self.last_torque: dict[int, bool] = {}

    def state_payload(self, refresh: bool = True) -> dict[str, Any]:
        if refresh:
            self.last_positions = self.bus.read_positions()
            self.last_torque = self.bus.read_torque_states()
        return {
            "name": "finger_pair",
            "timestamp": time.time(),
            "port": self.args.port,
            "baudrate": self.args.baudrate,
            "read_only": self.args.read_only,
            "robot_ids": list(ROBOT_IDS),
            "exoskeleton_ids": list(EXOSKELETON_IDS),
            "positions_deg": {str(key): value for key, value in self.last_positions.items()},
            "torque_enabled": {str(key): value for key, value in self.last_torque.items()},
            "last_error": self.last_error,
        }

    def handle_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return error("request must be a JSON object")
        try:
            command = request.get("cmd")
            if command == "ping":
                return ok({"name": "finger_pair_zmq_node", "timestamp": time.time()})
            if command in ("status", "get_positions"):
                return ok(self.state_payload(refresh=True))
            if command == "set_position":
                motor_id = validate_motor_id(request.get("id"))
                position = finite_float(request.get("position_deg"), "position_deg")
                if not self.args.min_position_deg <= position <= self.args.max_position_deg:
                    raise ValueError(
                        f"position must be in [{self.args.min_position_deg}, "
                        f"{self.args.max_position_deg}] deg"
                    )
                previous = self.bus.set_position(motor_id, position, self.args.max_step_deg)
                return ok(
                    {
                        "id": motor_id,
                        "previous_position_deg": previous,
                        "target_position_deg": position,
                    }
                )
            if command == "torque_on":
                self.bus.set_torque(True)
                return ok({"torque_enabled": True})
            if command == "torque_off":
                self.bus.set_torque(False)
                return ok({"torque_enabled": False})
            if command == "shutdown":
                self.running = False
                return ok({"shutdown": True})
            raise ValueError(f"unknown command: {command!r}")
        except Exception as exc:
            self.last_error = str(exc)
            return error(exc)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot/exoskeleton finger ZMQ node.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--command-endpoint", default=DEFAULT_COMMAND_ENDPOINT)
    parser.add_argument("--state-endpoint", default=DEFAULT_STATE_ENDPOINT)
    parser.add_argument("--state-rate", type=float, default=10.0)
    parser.add_argument("--min-position-deg", type=float, default=-360.0)
    parser.add_argument("--max-position-deg", type=float, default=360.0)
    parser.add_argument("--max-step-deg", type=float, default=10.0)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args(argv)
    for name in ("state_rate", "max_step_deg"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    if args.min_position_deg >= args.max_position_deg:
        parser.error("--min-position-deg must be less than --max-position-deg")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

    node = FingerPairNode(args)
    context = zmq.Context.instance()
    command_socket = context.socket(zmq.REP)
    state_socket = context.socket(zmq.PUB)
    command_socket.linger = 0
    state_socket.linger = 0

    def request_stop(*_args) -> None:
        node.running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node.bus.connect()
        command_socket.bind(args.command_endpoint)
        state_socket.bind(args.state_endpoint)
        print(
            f"Finger pair connected: port={args.port}, IDs={list(ALL_IDS)}, "
            f"command={args.command_endpoint}, state={args.state_endpoint}, "
            f"read_only={args.read_only}",
            flush=True,
        )
        poller = zmq.Poller()
        poller.register(command_socket, zmq.POLLIN)
        next_state = time.monotonic()
        while node.running:
            events = dict(poller.poll(20))
            if command_socket in events:
                try:
                    request = command_socket.recv_json()
                    response = node.handle_request(request)
                except Exception as exc:
                    response = error(exc)
                command_socket.send_json(response)
            now = time.monotonic()
            if now >= next_state:
                try:
                    state_socket.send_json(node.state_payload(refresh=True))
                except Exception as exc:
                    node.last_error = str(exc)
                    print(f"Motor state warning: {exc}", file=sys.stderr, flush=True)
                next_state = now + 1.0 / args.state_rate
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Finger pair node failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        node.bus.close(disable_torque=True)
        command_socket.close()
        state_socket.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
