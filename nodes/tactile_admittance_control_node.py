#!/usr/bin/env python3
"""Three-axis tactile admittance control using a URDF geometric Jacobian."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "libgex" / "gx16" / "urdf" / "gx16_thumb.urdf"
DEFAULT_TACTILE_ENDPOINT = "tcp://127.0.0.1:5561"
DEFAULT_MOTOR_COMMAND_ENDPOINT = "tcp://127.0.0.1:5580"
DEFAULT_MOTOR_STATE_ENDPOINT = "tcp://127.0.0.1:5581"
DEFAULT_COMMAND_ENDPOINT = "tcp://127.0.0.1:5590"
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5591"
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")
ACTIVE_INDICES = (0, 2, 3)
ACTIVE_MOTOR_IDS = (1, 3, 4)
ROBOT_MOTOR_IDS = (1, 2, 3, 4)
JOINT_DIRECTIONS = np.asarray((1.0, 1.0, -1.0, 1.0))


def finite_vector(values: Any, length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return array


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


def parse_triplet(element: ET.Element | None, attribute: str) -> np.ndarray:
    if element is None:
        return np.zeros(3)
    raw = element.get(attribute, "0 0 0")
    return finite_vector(raw.split(), 3, attribute)


class ThumbKinematics:
    """Minimal URDF chain evaluator for joint1..joint4 and link18."""

    def __init__(self, urdf_path: Path, contact_offset_m: Sequence[float] = (0, 0, 0)):
        root = ET.parse(urdf_path).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        self.origins = []
        self.axes = []
        self.lower = []
        self.upper = []
        for name in JOINT_NAMES:
            joint = joints.get(name)
            if joint is None:
                raise ValueError(f"URDF is missing {name}")
            origin = joint.find("origin")
            self.origins.append(
                transform(parse_triplet(origin, "xyz"), parse_triplet(origin, "rpy"))
            )
            axis = parse_triplet(joint.find("axis"), "xyz")
            if np.linalg.norm(axis) == 0:
                raise ValueError(f"URDF {name} has a zero axis")
            self.axes.append(axis / np.linalg.norm(axis))
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"URDF {name} has no limit")
            self.lower.append(float(limit.get("lower")))
            self.upper.append(float(limit.get("upper")))
        fixed = joints.get("joint18")
        if fixed is None:
            raise ValueError("URDF is missing fixed fingertip joint18")
        fixed_origin = fixed.find("origin")
        self.tip_origin = transform(
            parse_triplet(fixed_origin, "xyz"), parse_triplet(fixed_origin, "rpy")
        )
        self.contact_offset_m = finite_vector(contact_offset_m, 3, "contact_offset_m")
        self.lower = np.asarray(self.lower)
        self.upper = np.asarray(self.upper)

    def pose_and_jacobian(self, q_rad: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        q = finite_vector(q_rad, 4, "q_rad")
        current = np.eye(4)
        joint_points = []
        joint_axes = []
        for origin, local_axis, angle in zip(self.origins, self.axes, q):
            at_joint = current @ origin
            joint_points.append(at_joint[:3, 3].copy())
            joint_axes.append(at_joint[:3, :3] @ local_axis)
            rotation = np.eye(4)
            rotation[:3, :3] = axis_angle_matrix(local_axis, float(angle))
            current = at_joint @ rotation
        tip = current @ self.tip_origin
        contact = tip.copy()
        contact[:3, 3] += tip[:3, :3] @ self.contact_offset_m
        point = contact[:3, 3]
        jacobian = np.column_stack(
            [np.cross(axis, point - origin) for origin, axis in zip(joint_points, joint_axes)]
        )
        return contact, jacobian


def motor_to_urdf_rad(
    positions_deg: Sequence[float], motor_zero_deg: Sequence[float]
) -> np.ndarray:
    positions = finite_vector(positions_deg, 4, "positions_deg")
    zeros = finite_vector(motor_zero_deg, 4, "motor_zero_deg")
    return np.deg2rad(JOINT_DIRECTIONS * (positions - zeros))


def urdf_rad_to_motor_deg(
    q_rad: Sequence[float], motor_zero_deg: Sequence[float]
) -> np.ndarray:
    q = finite_vector(q_rad, 4, "q_rad")
    zeros = finite_vector(motor_zero_deg, 4, "motor_zero_deg")
    return zeros + JOINT_DIRECTIONS * np.rad2deg(q)


def admittance_step(
    jacobian_active: np.ndarray,
    contact_rotation: np.ndarray,
    force_error_sensor_n: Sequence[float],
    gains_m_per_n: Sequence[float],
    motion_signs: Sequence[float],
    damping: float,
    max_cartesian_step_m: float,
    max_joint_step_rad: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    jacobian = np.asarray(jacobian_active, dtype=float)
    rotation = np.asarray(contact_rotation, dtype=float)
    if jacobian.shape != (3, 3) or rotation.shape != (3, 3):
        raise ValueError("active Jacobian and contact rotation must both be 3x3")
    error = finite_vector(force_error_sensor_n, 3, "force_error_sensor_n")
    gains = finite_vector(gains_m_per_n, 3, "gains_m_per_n")
    signs = finite_vector(motion_signs, 3, "motion_signs")
    delta_sensor = gains * signs * error
    delta_sensor = np.clip(
        delta_sensor, -max_cartesian_step_m, max_cartesian_step_m
    )
    delta_base = rotation @ delta_sensor
    regularized = jacobian @ jacobian.T + (damping**2) * np.eye(3)
    delta_q = jacobian.T @ np.linalg.solve(regularized, delta_base)
    delta_q = np.clip(delta_q, -max_joint_step_rad, max_joint_step_rad)
    condition = float(np.linalg.cond(jacobian))
    return delta_q, condition, delta_sensor


class AdmittanceController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.kinematics = ThumbKinematics(args.urdf, args.contact_offset_m)
        self.motor_zero_deg = np.asarray(args.motor_zero_deg, dtype=float)
        self.sensor_rotation = rpy_matrix(np.deg2rad(args.sensor_rpy_deg))
        self.target_force = np.asarray(args.target_force_xyz_n, dtype=float)
        self.gains_m_per_n = np.asarray(args.gains_mm_per_n, dtype=float) / 1000.0
        self.motion_signs = np.asarray(args.motion_signs, dtype=float)
        self.latest_raw_force: np.ndarray | None = None
        self.filtered_raw_force: np.ndarray | None = None
        self.force_zero: np.ndarray | None = None
        self.latest_force_time: float | None = None
        self.motor_positions: dict[int, float] = {}
        self.motor_torque: dict[int, bool] = {}
        self.latest_motor_time: float | None = None
        self.active = False
        self.mode = "idle"
        self.fault: str | None = None
        self.zero_samples: list[np.ndarray] = []
        self.zero_deadline: float | None = None
        self.last_condition: float | None = None
        self.last_delta_q_deg = np.zeros(3)
        self.last_delta_x_mm = np.zeros(3)
        self.last_commanded_motor_deg: dict[int, float] = {}
        self.last_tracking_error_deg: dict[int, float] = {}
        self.commanded_q: np.ndarray | None = None
        self.command_count = 0

    @property
    def measured_force(self) -> np.ndarray | None:
        if self.filtered_raw_force is None:
            return None
        zero = np.zeros(3) if self.force_zero is None else self.force_zero
        return self.filtered_raw_force - zero

    def update_force(self, raw_force: Sequence[float], now: float) -> None:
        raw = finite_vector(raw_force, 3, "raw_force")
        self.latest_raw_force = raw
        if self.filtered_raw_force is None:
            self.filtered_raw_force = raw.copy()
        else:
            alpha = self.args.filter_alpha
            self.filtered_raw_force = alpha * raw + (1.0 - alpha) * self.filtered_raw_force
        self.latest_force_time = now
        if self.zero_deadline is not None:
            self.zero_samples.append(raw.copy())

    def update_motor_state(
        self, positions: dict[int, float], torque: dict[int, bool], now: float
    ) -> None:
        self.motor_positions = dict(positions)
        self.motor_torque = dict(torque)
        self.latest_motor_time = now

    def begin_zeroing(self, now: float) -> None:
        self.active = False
        self.commanded_q = None
        self.mode = "zeroing"
        self.fault = None
        self.zero_samples.clear()
        self.zero_deadline = now + self.args.zero_duration

    def finish_zeroing_if_ready(self, now: float) -> None:
        if self.zero_deadline is None or now < self.zero_deadline:
            return
        if len(self.zero_samples) < self.args.zero_min_samples:
            self.set_fault(
                f"force zeroing received only {len(self.zero_samples)} samples"
            )
        else:
            self.force_zero = np.median(np.asarray(self.zero_samples), axis=0)
            self.mode = "idle"
        self.zero_deadline = None
        self.zero_samples.clear()

    def set_fault(self, message: str) -> None:
        self.active = False
        self.commanded_q = None
        self.mode = "fault"
        self.fault = str(message)

    def data_ready(self, now: float) -> tuple[bool, str | None]:
        if self.force_zero is None:
            return False, "force sensor has not been zeroed"
        if self.latest_force_time is None or now - self.latest_force_time > self.args.data_timeout:
            return False, "tactile force data is stale"
        if self.latest_motor_time is None or now - self.latest_motor_time > self.args.data_timeout:
            return False, "motor state data is stale"
        missing = [motor_id for motor_id in ROBOT_MOTOR_IDS if motor_id not in self.motor_positions]
        if missing:
            return False, f"motor state is missing IDs {missing}"
        if not all(self.motor_torque.get(motor_id, False) for motor_id in ROBOT_MOTOR_IDS):
            return False, "robot finger motors 1-4 must all be torque enabled"
        return True, None

    def start(self, now: float) -> None:
        ready, reason = self.data_ready(now)
        if not ready:
            raise RuntimeError(reason)
        positions = [self.motor_positions[motor_id] for motor_id in ROBOT_MOTOR_IDS]
        self.commanded_q = motor_to_urdf_rad(positions, self.motor_zero_deg)
        self.last_commanded_motor_deg = {
            motor_id: float(positions[index])
            for index, motor_id in zip(ACTIVE_INDICES, ACTIVE_MOTOR_IDS)
        }
        self.last_tracking_error_deg = {motor_id: 0.0 for motor_id in ACTIVE_MOTOR_IDS}
        self.active = True
        self.mode = "holding"
        self.fault = None

    def stop(self) -> None:
        self.active = False
        self.commanded_q = None
        if self.mode != "fault":
            self.mode = "idle"

    def compute_command(self, now: float) -> dict[int, float] | None:
        if not self.active:
            return None
        ready, reason = self.data_ready(now)
        if not ready:
            self.set_fault(reason or "controller data is not ready")
            return None
        force = self.measured_force
        assert force is not None
        if np.linalg.norm(force) > self.args.force_norm_limit_n:
            self.set_fault(
                f"force norm {np.linalg.norm(force):.2f} N exceeds "
                f"{self.args.force_norm_limit_n:.2f} N"
            )
            return None
        positions = [self.motor_positions[motor_id] for motor_id in ROBOT_MOTOR_IDS]
        q = motor_to_urdf_rad(positions, self.motor_zero_deg)
        tolerance = math.radians(self.args.joint_limit_tolerance_deg)
        if np.any(q < self.kinematics.lower - tolerance) or np.any(
            q > self.kinematics.upper + tolerance
        ):
            self.set_fault(f"URDF joint position is outside limits: {np.rad2deg(q).tolist()}")
            return None
        contact, full_jacobian = self.kinematics.pose_and_jacobian(q)
        active_jacobian = full_jacobian[:, ACTIVE_INDICES]
        base_sensor_rotation = contact[:3, :3] @ self.sensor_rotation
        delta_q, condition, delta_sensor = admittance_step(
            active_jacobian,
            base_sensor_rotation,
            self.target_force - force,
            self.gains_m_per_n,
            self.motion_signs,
            self.args.damping,
            self.args.max_cartesian_step_mm / 1000.0,
            math.radians(self.args.max_joint_step_deg),
        )
        self.last_condition = condition
        self.last_delta_q_deg = np.rad2deg(delta_q)
        self.last_delta_x_mm = delta_sensor * 1000.0
        if not math.isfinite(condition) or condition > self.args.max_condition:
            self.set_fault(
                f"active Jacobian condition {condition:.1f} exceeds "
                f"{self.args.max_condition:.1f}"
            )
            return None
        if self.commanded_q is None:
            self.commanded_q = q.copy()
        # Integrate the admittance increment into the previous command, while
        # inactive joint 2 continues to follow its measured position.
        target_q = q.copy()
        target_q[list(ACTIVE_INDICES)] = self.commanded_q[list(ACTIVE_INDICES)]
        target_q[list(ACTIVE_INDICES)] += delta_q
        tracking_error_urdf_deg = np.rad2deg(
            target_q[list(ACTIVE_INDICES)] - q[list(ACTIVE_INDICES)]
        )
        tracking_error_motor_deg = (
            JOINT_DIRECTIONS[list(ACTIVE_INDICES)] * tracking_error_urdf_deg
        )
        self.last_tracking_error_deg = {
            motor_id: float(error_deg)
            for motor_id, error_deg in zip(ACTIVE_MOTOR_IDS, tracking_error_motor_deg)
        }
        max_tracking_error = float(np.max(np.abs(tracking_error_urdf_deg)))
        if max_tracking_error > self.args.max_tracking_error_deg:
            self.set_fault(
                f"joint target tracking error {max_tracking_error:.2f} deg exceeds "
                f"{self.args.max_tracking_error_deg:.2f} deg; errors="
                f"{self.last_tracking_error_deg}"
            )
            return None
        lower = self.kinematics.lower - tolerance
        upper = self.kinematics.upper + tolerance
        if np.any(target_q < lower) or np.any(target_q > upper):
            self.set_fault("admittance command would cross a URDF joint limit")
            return None
        target_motor = urdf_rad_to_motor_deg(target_q, self.motor_zero_deg)
        command = {
            motor_id: float(target_motor[index])
            for index, motor_id in zip(ACTIVE_INDICES, ACTIVE_MOTOR_IDS)
        }
        self.commanded_q = target_q
        self.last_commanded_motor_deg = command
        self.command_count += 1
        return command

    def status_payload(self) -> dict[str, Any]:
        force = self.measured_force
        positions = [self.motor_positions.get(motor_id) for motor_id in ROBOT_MOTOR_IDS]
        q_deg = None
        if all(value is not None for value in positions):
            q_deg = np.rad2deg(
                motor_to_urdf_rad(positions, self.motor_zero_deg)
            ).tolist()
        return {
            "name": "tactile_admittance_control",
            "timestamp": time.time(),
            "mode": self.mode,
            "active": self.active,
            "fault": self.fault,
            "target_force_xyz_n": self.target_force.tolist(),
            "measured_force_xyz_n": None if force is None else force.tolist(),
            "force_zero_xyz_n": None if self.force_zero is None else self.force_zero.tolist(),
            "motor_zero_deg": self.motor_zero_deg.tolist(),
            "urdf_q_deg": q_deg,
            "active_joints": [1, 3, 4],
            "jacobian_condition": self.last_condition,
            "delta_q_deg": self.last_delta_q_deg.tolist(),
            "delta_x_sensor_mm": self.last_delta_x_mm.tolist(),
            "last_commanded_motor_deg": {
                str(key): value for key, value in self.last_commanded_motor_deg.items()
            },
            "tracking_error_deg": {
                str(key): value for key, value in self.last_tracking_error_deg.items()
            },
            "command_count": self.command_count,
        }


def decode_tactile(payload: Any, finger: str) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError("tactile payload must be an object")
    fingers = payload.get("fingers")
    forces = payload.get("force_xyz_n")
    if not isinstance(fingers, list) or not isinstance(forces, list):
        raise ValueError("invalid tactile payload")
    try:
        index = fingers.index(finger)
    except ValueError as exc:
        raise ValueError(f"tactile payload does not contain finger {finger!r}") from exc
    return finite_vector(forces[index], 3, "force_xyz_n")


def decode_motor_state(payload: Any) -> tuple[dict[int, float], dict[int, bool]]:
    if not isinstance(payload, dict):
        raise ValueError("motor state must be an object")
    positions_raw = payload.get("positions_deg")
    torque_raw = payload.get("torque_enabled")
    if not isinstance(positions_raw, dict) or not isinstance(torque_raw, dict):
        raise ValueError("invalid motor state payload")
    positions = {
        motor_id: float(positions_raw[str(motor_id)]) for motor_id in ROBOT_MOTOR_IDS
    }
    torque = {
        motor_id: bool(torque_raw[str(motor_id)]) for motor_id in ROBOT_MOTOR_IDS
    }
    return positions, torque


def ok(result: Any = None) -> dict[str, Any]:
    return {"ok": True, "result": {} if result is None else result, "error": None}


def error(message: Any) -> dict[str, Any]:
    return {"ok": False, "result": None, "error": str(message)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URDF-Jacobian tactile admittance controller.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--finger", default="thumb")
    parser.add_argument("--tactile-endpoint", default=DEFAULT_TACTILE_ENDPOINT)
    parser.add_argument("--motor-command-endpoint", default=DEFAULT_MOTOR_COMMAND_ENDPOINT)
    parser.add_argument("--motor-state-endpoint", default=DEFAULT_MOTOR_STATE_ENDPOINT)
    parser.add_argument("--command-endpoint", default=DEFAULT_COMMAND_ENDPOINT)
    parser.add_argument("--state-endpoint", default=DEFAULT_STATE_ENDPOINT)
    parser.add_argument("--control-rate", type=float, default=10.0)
    parser.add_argument("--motor-zero-deg", nargs=4, type=float, default=(90, 90, 90, 90))
    parser.add_argument("--target-force-xyz-n", nargs=3, type=float, default=(0, 0, 1))
    parser.add_argument("--gains-mm-per-n", nargs=3, type=float, default=(0.02, 0.02, 0.05))
    parser.add_argument("--motion-signs", nargs=3, type=float, default=(1, 1, 1))
    parser.add_argument("--sensor-rpy-deg", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--contact-offset-m", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--damping", type=float, default=0.005)
    parser.add_argument("--filter-alpha", type=float, default=0.2)
    parser.add_argument("--max-cartesian-step-mm", type=float, default=0.1)
    parser.add_argument("--max-joint-step-deg", type=float, default=0.2)
    parser.add_argument("--max-tracking-error-deg", type=float, default=2.0)
    parser.add_argument("--max-condition", type=float, default=1000.0)
    parser.add_argument("--force-norm-limit-n", type=float, default=5.0)
    parser.add_argument("--joint-limit-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--data-timeout", type=float, default=0.3)
    parser.add_argument("--motor-command-timeout-ms", type=int, default=1000)
    parser.add_argument("--zero-duration", type=float, default=1.0)
    parser.add_argument("--zero-min-samples", type=int, default=10)
    args = parser.parse_args(argv)
    positive = (
        "control_rate", "damping", "filter_alpha", "max_cartesian_step_mm",
        "max_joint_step_deg", "max_tracking_error_deg", "max_condition", "force_norm_limit_n",
        "data_timeout", "zero_duration",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    if args.filter_alpha > 1:
        parser.error("--filter-alpha must be at most 1")
    if args.joint_limit_tolerance_deg < 0:
        parser.error("--joint-limit-tolerance-deg must be non-negative")
    if args.zero_min_samples <= 0:
        parser.error("--zero-min-samples must be greater than zero")
    if args.motor_command_timeout_ms <= 0:
        parser.error("--motor-command-timeout-ms must be greater than zero")
    vector_arguments = (
        ("motor-zero-deg", args.motor_zero_deg),
        ("target-force-xyz-n", args.target_force_xyz_n),
        ("sensor-rpy-deg", args.sensor_rpy_deg),
        ("contact-offset-m", args.contact_offset_m),
    )
    for name, values in vector_arguments:
        if not all(math.isfinite(value) for value in values):
            parser.error(f"--{name} values must be finite")
    if not all(math.isfinite(value) and value >= 0 for value in args.gains_mm_per_n):
        parser.error("--gains-mm-per-n values must be finite and non-negative")
    if any(value not in (-1, 1) for value in args.motion_signs):
        parser.error("--motion-signs values must each be -1 or 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc
    controller = AdmittanceController(args)
    context = zmq.Context.instance()
    tactile_socket = context.socket(zmq.SUB)
    motor_state_socket = context.socket(zmq.SUB)
    for socket, endpoint in (
        (tactile_socket, args.tactile_endpoint),
        (motor_state_socket, args.motor_state_endpoint),
    ):
        socket.linger = 0
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(endpoint)
    command_socket = context.socket(zmq.REP)
    state_socket = context.socket(zmq.PUB)
    command_socket.linger = 0
    state_socket.linger = 0
    command_socket.bind(args.command_endpoint)
    state_socket.bind(args.state_endpoint)
    motor_command_socket = None

    def new_motor_command_socket():
        nonlocal motor_command_socket
        if motor_command_socket is not None:
            motor_command_socket.close(0)
        motor_command_socket = context.socket(zmq.REQ)
        motor_command_socket.linger = 0
        motor_command_socket.setsockopt(zmq.SNDTIMEO, args.motor_command_timeout_ms)
        motor_command_socket.setsockopt(zmq.RCVTIMEO, args.motor_command_timeout_ms)
        motor_command_socket.connect(args.motor_command_endpoint)

    new_motor_command_socket()
    running = True
    disabled_fault: str | None = None
    last_logged_mode = controller.mode
    last_logged_fault = controller.fault

    def log_event(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def request_stop(*_args) -> None:
        nonlocal running
        running = False

    def motor_request(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal motor_command_socket
        try:
            motor_command_socket.send_json(payload)
            response = motor_command_socket.recv_json()
        except Exception:
            new_motor_command_socket()
            raise
        if not isinstance(response, dict) or not response.get("ok"):
            raise RuntimeError(response.get("error", "motor command failed"))
        return response

    def disable_motor_torque(reason: str) -> None:
        log_event(f"TORQUE_OFF requested: reason={reason}")
        try:
            motor_request({"cmd": "torque_off"})
        except Exception as exc:
            log_event(f"TORQUE_OFF failed: reason={reason}; error={exc}")
            raise
        log_event(f"TORQUE_OFF succeeded: reason={reason}")

    def fault_and_disable(message: str) -> None:
        nonlocal disabled_fault
        controller.set_fault(message)
        if disabled_fault == controller.fault:
            return
        try:
            disable_motor_torque(f"safety fault: {message}")
            disabled_fault = controller.fault
        except Exception as exc:
            controller.fault = f"{message}; torque-off failed: {exc}"
            # A failed torque-off is logged and exposed in the state, but must
            # not be retried every control cycle: doing so can starve recovery
            # commands and recursively grow the fault message.
            disabled_fault = controller.fault

    def log_controller_state_changes() -> None:
        nonlocal last_logged_mode, last_logged_fault
        if controller.mode != last_logged_mode:
            log_event(f"MODE {last_logged_mode} -> {controller.mode}")
            last_logged_mode = controller.mode
        if controller.fault != last_logged_fault:
            if controller.fault:
                log_event(f"FAULT: {controller.fault}")
            elif last_logged_fault:
                log_event("FAULT cleared")
            last_logged_fault = controller.fault

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    poller = zmq.Poller()
    poller.register(tactile_socket, zmq.POLLIN)
    poller.register(motor_state_socket, zmq.POLLIN)
    poller.register(command_socket, zmq.POLLIN)
    next_control = time.monotonic()
    next_state = time.monotonic()
    log_event(
        f"Admittance controller: command={args.command_endpoint}, state={args.state_endpoint}, "
        "active joints=[1, 3, 4], starts idle"
    )
    try:
        while running:
            events = dict(poller.poll(10))
            now = time.monotonic()
            if tactile_socket in events:
                latest = None
                while True:
                    try:
                        latest = tactile_socket.recv_json(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                if latest is not None:
                    try:
                        controller.update_force(decode_tactile(latest, args.finger), now)
                    except Exception as exc:
                        if controller.active:
                            fault_and_disable(f"invalid tactile data: {exc}")
            if motor_state_socket in events:
                latest = None
                while True:
                    try:
                        latest = motor_state_socket.recv_json(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                if latest is not None:
                    try:
                        positions, torque = decode_motor_state(latest)
                        controller.update_motor_state(positions, torque, now)
                    except Exception as exc:
                        if controller.active:
                            fault_and_disable(f"invalid motor state: {exc}")
            if command_socket in events:
                request = command_socket.recv_json()
                command = request.get("cmd") if isinstance(request, dict) else None
                if command not in ("ping", "status"):
                    log_event(f"COMMAND received: cmd={command!r}, request={request!r}")
                try:
                    if command in ("ping", "status"):
                        response = ok(controller.status_payload())
                    elif command == "zero":
                        disabled_fault = None
                        controller.begin_zeroing(now)
                        response = ok({"zeroing": True})
                    elif command == "set_target":
                        controller.target_force = finite_vector(
                            request.get("target_force_xyz_n"), 3, "target_force_xyz_n"
                        )
                        response = ok({"target_force_xyz_n": controller.target_force.tolist()})
                    elif command == "set_gains":
                        gains = finite_vector(request.get("gains_mm_per_n"), 3, "gains_mm_per_n")
                        if np.any(gains < 0):
                            raise ValueError("admittance gains must be non-negative")
                        controller.gains_m_per_n = gains / 1000.0
                        signs = finite_vector(request.get("motion_signs"), 3, "motion_signs")
                        if np.any(np.abs(signs) != 1):
                            raise ValueError("motion signs must each be -1 or 1")
                        controller.motion_signs = signs
                        response = ok({"gains_mm_per_n": gains.tolist(), "motion_signs": signs.tolist()})
                    elif command == "start":
                        disabled_fault = None
                        controller.start(now)
                        response = ok({"active": True})
                    elif command == "stop":
                        controller.stop()
                        response = ok({"active": False})
                    elif command == "emergency_stop":
                        controller.stop()
                        disable_motor_torque("emergency stop command")
                        response = ok({"active": False, "torque_enabled": False})
                    elif command == "shutdown":
                        controller.stop()
                        running = False
                        response = ok({"shutdown": True})
                    else:
                        raise ValueError(f"unknown command: {command!r}")
                except Exception as exc:
                    response = error(exc)
                command_socket.send_json(response)
                if command not in ("ping", "status"):
                    if response.get("ok"):
                        log_event(f"COMMAND accepted: cmd={command!r}, result={response['result']!r}")
                    else:
                        log_event(f"COMMAND rejected: cmd={command!r}, error={response['error']}")
            controller.finish_zeroing_if_ready(now)
            if now >= next_control:
                command = controller.compute_command(now)
                if command is not None:
                    try:
                        for motor_id, target in command.items():
                            motor_request(
                                {"cmd": "set_position", "id": motor_id, "position_deg": target}
                            )
                    except Exception as exc:
                        fault_and_disable(f"motor command failed: {exc}")
                elif (
                    controller.mode == "fault"
                    and controller.fault
                    and disabled_fault != controller.fault
                ):
                    # compute_command has already made the controller inactive;
                    # ensure a newly detected safety fault also drops torque.
                    try:
                        fault_reason = controller.fault
                        disable_motor_torque(f"safety fault: {fault_reason}")
                        disabled_fault = fault_reason
                    except Exception as exc:
                        controller.fault = (
                            f"{fault_reason}; torque-off failed: {exc}"
                        )
                        disabled_fault = controller.fault
                next_control = now + 1.0 / args.control_rate
            log_controller_state_changes()
            if now >= next_state:
                state_socket.send_json(controller.status_payload())
                next_state = now + 0.1
    finally:
        controller.stop()
        try:
            # The controller owns the closed-loop session. Dropping torque on
            # every exit also makes a standalone SIGTERM fail safe.
            disable_motor_torque("controller process exit")
        except Exception:
            pass
        for socket in (
            tactile_socket, motor_state_socket, command_socket, state_socket,
            motor_command_socket,
        ):
            socket.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
