import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from libgex2 import Hand16  # noqa: E402
from libgex2.libgex.gx16.libgx16 import JOINT_MOTOR_DIRECTIONS  # noqa: E402


JOINT_COUNT = 16
DEFAULT_SERIAL_NUMBER = "FTAKRP3AA"
DEFAULT_CMD_ENDPOINT = "tcp://127.0.0.1:5556"
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5557"
STATE_TOPIC = "gx16/state"
STATE_HZ = 10.0
VALID_UNITS = {"urdf_deg", "motor_deg"}


def _load_zmq():
    try:
        import zmq
    except ImportError:
        print(
            "pyzmq is required. Install it with: python -m pip install pyzmq",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return zmq


def _joint_directions():
    directions = [int(value) for value in JOINT_MOTOR_DIRECTIONS]
    if len(directions) != JOINT_COUNT:
        raise ValueError(
            f"JOINT_MOTOR_DIRECTIONS must contain {JOINT_COUNT} values, "
            f"got {len(directions)}"
        )
    if any(value not in (-1, 1) for value in directions):
        raise ValueError("JOINT_MOTOR_DIRECTIONS values must be 1 or -1.")
    return directions


JOINT_DIRECTIONS = _joint_directions()


def _now():
    return time.time()


def _ok(result=None):
    return {"ok": True, "result": result if result is not None else {}, "error": None}


def _error(message):
    return {"ok": False, "result": None, "error": str(message)}


def _validate_units(units):
    units = units or "urdf_deg"
    if units not in VALID_UNITS:
        raise ValueError(f"units must be one of {sorted(VALID_UNITS)}, got {units!r}")
    return units


def _validate_positions(values, expected_count):
    if not isinstance(values, list):
        raise ValueError("positions must be a JSON list")
    if len(values) != expected_count:
        raise ValueError(f"positions must contain {expected_count} values, got {len(values)}")

    positions = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"positions[{index}] is not a number: {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"positions[{index}] must be finite, got {value!r}")
        positions.append(number)
    return positions


def _validate_position(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"position is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"position must be finite, got {value!r}")
    return number


def _urdf_to_motor(urdf_deg):
    return [angle * direction for angle, direction in zip(urdf_deg, JOINT_DIRECTIONS)]


def _motor_to_urdf(motor_deg):
    return [angle * direction for angle, direction in zip(motor_deg, JOINT_DIRECTIONS)]


class GX16ZmqNode:
    def __init__(self, args):
        self.args = args
        self.hand = None
        self.connected = False
        self.running = True
        self.last_command = None
        self.last_error = None
        self.last_motor_deg = [0.0] * JOINT_COUNT
        self.hardware_lock = threading.Lock()
        self.state_lock = threading.Lock()

    @property
    def port(self):
        if self.hand is not None:
            return getattr(self.hand, "port", None)
        return self.args.port

    def connect_hand(self):
        if self.args.dry_run:
            self.connected = False
            print("Dry-run mode: not connecting to GX16 hardware.", flush=True)
            return

        try:
            self.hand = Hand16(port=self.args.port, serial_number=self.args.serial_number)
            self.hand.connect(
                curr_limit=self.args.curr_limit,
                goal_current=self.args.goal_current,
                goal_pwm=self.args.goal_pwm,
            )
        except SystemExit as exc:
            raise RuntimeError(f"failed to connect GX16: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"failed to connect GX16: {exc}") from exc

        self.connected = True
        self.last_motor_deg = self._read_motor_deg(update_error=False)
        print(f"GX16 connected on {self.port}.", flush=True)

    def _read_motor_deg(self, update_error=True):
        if self.args.dry_run:
            with self.state_lock:
                return list(self.last_motor_deg)
        if self.hand is None or not self.connected:
            raise RuntimeError("GX16 is not connected")
        try:
            with self.hardware_lock:
                values = [float(value) for value in self.hand.getjs()]
        except Exception as exc:
            if update_error:
                self.set_last_error(str(exc))
            raise
        if len(values) != JOINT_COUNT:
            raise RuntimeError(f"Hand16.getjs returned {len(values)} values, expected {JOINT_COUNT}")
        with self.state_lock:
            self.last_motor_deg = values
        return values

    def _send_setjs(self, motor_deg):
        if self.args.dry_run:
            print(
                "Dry-run setjs motor_deg: "
                + ", ".join(f"{value:.3f}" for value in motor_deg),
                flush=True,
            )
        else:
            if self.hand is None or not self.connected:
                raise RuntimeError("GX16 is not connected")
            with self.hardware_lock:
                self.hand.setjs(motor_deg)
        with self.state_lock:
            self.last_motor_deg = list(motor_deg)

    def _send_setj(self, joint, motor_deg):
        if self.args.dry_run:
            print(f"Dry-run setj joint={joint} motor_deg={motor_deg:.3f}", flush=True)
        else:
            if self.hand is None or not self.connected:
                raise RuntimeError("GX16 is not connected")
            with self.hardware_lock:
                self.hand.setj(joint, motor_deg)
        with self.state_lock:
            self.last_motor_deg[joint - 1] = motor_deg

    def set_last_error(self, error):
        with self.state_lock:
            self.last_error = error

    def set_running(self, running):
        with self.state_lock:
            self.running = running

    def is_running(self):
        with self.state_lock:
            return self.running

    def status_payload(self, read_position=True):
        with self.state_lock:
            motor_deg = list(self.last_motor_deg)
            last_command = self.last_command
            last_error = self.last_error
        if read_position and (self.args.dry_run or self.connected):
            try:
                motor_deg = self._read_motor_deg()
            except Exception:
                with self.state_lock:
                    motor_deg = list(self.last_motor_deg)
                    last_error = self.last_error

        return {
            "connected": bool(self.connected),
            "dry_run": bool(self.args.dry_run),
            "port": self.port,
            "last_command": last_command,
            "last_error": last_error,
            "motor_deg": motor_deg,
            "urdf_deg": _motor_to_urdf(motor_deg),
            "timestamp": _now(),
        }

    def handle_request(self, request):
        if not isinstance(request, dict):
            return _error("request must be a JSON object")

        cmd = request.get("cmd")
        with self.state_lock:
            self.last_command = cmd

        try:
            if cmd == "ping":
                return _ok(
                    {
                        "name": "gx16_zmq_node",
                        "connected": self.connected,
                        "dry_run": self.args.dry_run,
                        "timestamp": _now(),
                    }
                )

            if cmd == "status":
                return _ok(self.status_payload(read_position=True))

            if cmd == "getjs":
                units = _validate_units(request.get("units", "urdf_deg"))
                motor_deg = self._read_motor_deg()
                urdf_deg = _motor_to_urdf(motor_deg)
                positions = urdf_deg if units == "urdf_deg" else motor_deg
                return _ok(
                    {
                        "units": units,
                        "positions": positions,
                        "motor_deg": motor_deg,
                        "urdf_deg": urdf_deg,
                        "timestamp": _now(),
                    }
                )

            if cmd == "setjs":
                units = _validate_units(request.get("units", "urdf_deg"))
                positions = _validate_positions(request.get("positions"), JOINT_COUNT)
                motor_deg = _urdf_to_motor(positions) if units == "urdf_deg" else positions
                self._send_setjs(motor_deg)
                return _ok(
                    {
                        "units": units,
                        "motor_deg": list(motor_deg),
                        "urdf_deg": _motor_to_urdf(motor_deg),
                        "timestamp": _now(),
                    }
                )

            if cmd == "setj":
                units = _validate_units(request.get("units", "urdf_deg"))
                joint = int(request.get("joint"))
                if joint < 1 or joint > JOINT_COUNT:
                    raise ValueError(f"joint must be in [1, {JOINT_COUNT}], got {joint}")
                position = _validate_position(request.get("position"))
                motor_position = (
                    position * JOINT_DIRECTIONS[joint - 1] if units == "urdf_deg" else position
                )
                self._send_setj(joint, motor_position)
                return _ok(
                    {
                        "units": units,
                        "joint": joint,
                        "motor_deg": list(self.last_motor_deg),
                        "urdf_deg": _motor_to_urdf(self.last_motor_deg),
                        "timestamp": _now(),
                    }
                )

            if cmd == "home":
                if self.args.dry_run:
                    print("Dry-run home.", flush=True)
                else:
                    if self.hand is None or not self.connected:
                        raise RuntimeError("GX16 is not connected")
                    with self.hardware_lock:
                        self.hand.home()
                with self.state_lock:
                    self.last_motor_deg = [0.0] * JOINT_COUNT
                return _ok(self.status_payload(read_position=False))

            if cmd == "torque_on":
                if self.args.dry_run:
                    print("Dry-run torque_on.", flush=True)
                else:
                    if self.hand is None or not self.connected:
                        raise RuntimeError("GX16 is not connected")
                    with self.hardware_lock:
                        self.hand.on()
                return _ok(self.status_payload(read_position=False))

            if cmd == "torque_off":
                if self.args.dry_run:
                    print("Dry-run torque_off.", flush=True)
                else:
                    if self.hand is None or not self.connected:
                        raise RuntimeError("GX16 is not connected")
                    with self.hardware_lock:
                        self.hand.off()
                return _ok(self.status_payload(read_position=False))

            if cmd == "shutdown":
                self.set_running(False)
                return _ok({"shutdown": True, "timestamp": _now()})

            raise ValueError(f"unknown cmd: {cmd!r}")
        except Exception as exc:
            self.set_last_error(str(exc))
            return _error(exc)

    def publish_state(self, pub_socket, read_position=False):
        payload = self.status_payload(read_position=read_position)
        pub_socket.send_string(f"{STATE_TOPIC} {json.dumps(payload, separators=(',', ':'))}")


def _state_publisher_loop(node, pub_socket, interval, stop_event):
    while not stop_event.is_set() and node.is_running():
        try:
            node.publish_state(pub_socket, read_position=node.args.state_read_position)
        except Exception as exc:
            node.set_last_error(str(exc))
        stop_event.wait(interval)


def parse_args():
    parser = argparse.ArgumentParser(description="GX16 ZMQ control node.")
    parser.add_argument("--port", help="Serial port, for example COM6.")
    parser.add_argument(
        "--serial-number",
        default=DEFAULT_SERIAL_NUMBER,
        help="USB serial number used when --port is not set.",
    )
    parser.add_argument("--cmd-endpoint", default=DEFAULT_CMD_ENDPOINT)
    parser.add_argument("--state-endpoint", default=DEFAULT_STATE_ENDPOINT)
    parser.add_argument("--state-hz", type=float, default=STATE_HZ)
    parser.add_argument(
        "--state-read-position",
        action="store_true",
        help="Read real motor positions before each state publish. Disabled by default for smoother command streaming.",
    )
    parser.add_argument("--curr-limit", type=int, default=1000)
    parser.add_argument("--goal-current", type=int, default=600)
    parser.add_argument("--goal-pwm", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Do not connect to hardware.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.state_hz <= 0:
        raise SystemExit("--state-hz must be > 0")

    zmq = _load_zmq()
    context = zmq.Context.instance()
    rep_socket = context.socket(zmq.REP)
    pub_socket = context.socket(zmq.PUB)
    rep_socket.linger = 0
    pub_socket.linger = 0

    try:
        rep_socket.bind(args.cmd_endpoint)
        pub_socket.bind(args.state_endpoint)
        print(f"Command REP endpoint: {args.cmd_endpoint}", flush=True)
        print(f"State PUB endpoint: {args.state_endpoint} topic={STATE_TOPIC}", flush=True)

        node = GX16ZmqNode(args)
        node.connect_hand()

        stop_event = threading.Event()
        state_thread = threading.Thread(
            target=_state_publisher_loop,
            args=(node, pub_socket, 1.0 / args.state_hz, stop_event),
            name="gx16-state-publisher",
            daemon=True,
        )
        state_thread.start()

        poller = zmq.Poller()
        poller.register(rep_socket, zmq.POLLIN)

        while node.is_running():
            events = dict(poller.poll(100))

            if rep_socket in events:
                try:
                    request = rep_socket.recv_json()
                    response = node.handle_request(request)
                except Exception as exc:
                    response = _error(exc)
                rep_socket.send_json(response)

    except KeyboardInterrupt:
        print("Interrupted. Stopping GX16 ZMQ node without torque off.", flush=True)
    finally:
        if "node" in locals():
            node.set_running(False)
        if "stop_event" in locals():
            stop_event.set()
        if "state_thread" in locals():
            state_thread.join(timeout=1.0)
        rep_socket.close()
        pub_socket.close()


if __name__ == "__main__":
    main()
