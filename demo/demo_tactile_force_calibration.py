#!/usr/bin/env python3
"""Qt viewer for comparing tactile XYZ force with a reference force sensor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import zmq
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyQt5 is required: python -m pip install PyQt5") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TACTILE_ENDPOINT = "tcp://127.0.0.1:5561"
DEFAULT_FORCE_ENDPOINT = "tcp://127.0.0.1:5577"
DEFAULT_MOTOR_COMMAND_ENDPOINT = "tcp://127.0.0.1:5580"
DEFAULT_MOTOR_STATE_ENDPOINT = "tcp://127.0.0.1:5581"
DEFAULT_CONTROL_COMMAND_ENDPOINT = "tcp://127.0.0.1:5590"
DEFAULT_CONTROL_STATE_ENDPOINT = "tcp://127.0.0.1:5591"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiment" / "data"
TACTILE_NODE = REPO_ROOT / "utils" / "tactile" / "tactile_sum_force_zmq_node.py"
FORCE_NODE = REPO_ROOT / "nodes" / "force_zmq_node.py"
MOTOR_NODE = REPO_ROOT / "nodes" / "finger_pair_zmq_node.py"
CONTROL_NODE = REPO_ROOT / "nodes" / "tactile_admittance_control_node.py"
ROBOT_MOTOR_IDS = (1, 2, 3, 4)
EXOSKELETON_MOTOR_IDS = (21, 22, 23, 24)
ALL_MOTOR_IDS = ROBOT_MOTOR_IDS + EXOSKELETON_MOTOR_IDS


@dataclass(frozen=True)
class TactileReading:
    finger: str
    fx_n: float
    fy_n: float
    fz_n: float
    source_timestamp: float

    @property
    def magnitude_n(self) -> float:
        return math.sqrt(self.fx_n**2 + self.fy_n**2 + self.fz_n**2)


@dataclass(frozen=True)
class CalibrationSample:
    captured_at: float
    finger: str
    tactile_timestamp: float
    fx_n: float
    fy_n: float
    fz_n: float
    magnitude_n: float
    reference_force: float


def finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def decode_tactile_payload(payload: Any, finger: str | None = None) -> TactileReading:
    if not isinstance(payload, dict):
        raise ValueError("tactile payload must be a JSON object")
    fingers = payload.get("fingers")
    forces = payload.get("force_xyz_n")
    if not isinstance(fingers, list) or not isinstance(forces, list):
        raise ValueError("tactile payload must contain fingers and force_xyz_n lists")
    if len(fingers) != len(forces) or not fingers:
        raise ValueError("tactile fingers and force_xyz_n must have the same non-zero length")

    index = 0
    if finger is not None:
        try:
            index = fingers.index(finger)
        except ValueError as exc:
            raise ValueError(f"finger {finger!r} is absent from tactile payload") from exc
    values = forces[index]
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("each force_xyz_n entry must contain [Fx, Fy, Fz]")
    sec = finite_float(payload.get("stamp_sec"), "stamp_sec")
    nanosec = finite_float(payload.get("stamp_nanosec"), "stamp_nanosec")
    return TactileReading(
        finger=str(fingers[index]),
        fx_n=finite_float(values[0], "Fx"),
        fy_n=finite_float(values[1], "Fy"),
        fz_n=finite_float(values[2], "Fz"),
        source_timestamp=sec + nanosec / 1_000_000_000.0,
    )


def subtract_tactile_zero(
    reading: TactileReading, zero_xyz_n: Sequence[float] | None
) -> TactileReading:
    if zero_xyz_n is None:
        return reading
    zero = np.asarray(zero_xyz_n, dtype=float)
    if zero.shape != (3,) or not np.isfinite(zero).all():
        raise ValueError("tactile zero must contain three finite values")
    return TactileReading(
        finger=reading.finger,
        fx_n=reading.fx_n - float(zero[0]),
        fy_n=reading.fy_n - float(zero[1]),
        fz_n=reading.fz_n - float(zero[2]),
        source_timestamp=reading.source_timestamp,
    )


def decode_reference_force(message: str) -> float:
    return finite_float(message.strip(), "reference force")


def decode_motor_state(payload: Any) -> tuple[dict[int, float], dict[int, bool]]:
    if not isinstance(payload, dict):
        raise ValueError("motor state must be a JSON object")
    raw_positions = payload.get("positions_deg")
    raw_torque = payload.get("torque_enabled")
    if not isinstance(raw_positions, dict) or not isinstance(raw_torque, dict):
        raise ValueError("motor state must contain positions_deg and torque_enabled")
    positions = {}
    torque = {}
    for motor_id in ALL_MOTOR_IDS:
        key = str(motor_id)
        if key not in raw_positions or key not in raw_torque:
            raise ValueError(f"motor state is missing ID {motor_id}")
        positions[motor_id] = finite_float(raw_positions[key], f"motor {motor_id} position")
        torque[motor_id] = bool(raw_torque[key])
    return positions, torque


def decode_control_state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("control state must be a JSON object")
    required = ("mode", "active", "target_force_xyz_n", "measured_force_xyz_n")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"control state is missing {missing}")
    if payload["measured_force_xyz_n"] is not None:
        finite_vector = np.asarray(payload["measured_force_xyz_n"], dtype=float)
        if finite_vector.shape != (3,) or not np.isfinite(finite_vector).all():
            raise ValueError("measured_force_xyz_n must contain three finite values")
    return payload


def save_samples_csv(path: Path, samples: Sequence[CalibrationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CalibrationSample.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)


def parse_int(value: str) -> int:
    return int(value, 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tactile/reference-force calibration UI.")
    parser.add_argument("--tactile-endpoint", default=DEFAULT_TACTILE_ENDPOINT)
    parser.add_argument("--force-endpoint", default=DEFAULT_FORCE_ENDPOINT)
    parser.add_argument("--motor-command-endpoint", default=DEFAULT_MOTOR_COMMAND_ENDPOINT)
    parser.add_argument("--motor-state-endpoint", default=DEFAULT_MOTOR_STATE_ENDPOINT)
    parser.add_argument("--control-command-endpoint", default=DEFAULT_CONTROL_COMMAND_ENDPOINT)
    parser.add_argument("--control-state-endpoint", default=DEFAULT_CONTROL_STATE_ENDPOINT)
    parser.add_argument("--tactile-port", default="/dev/ttyACM0")
    parser.add_argument("--force-port", default="/dev/ttyUSB0")
    parser.add_argument("--motor-port", default="/dev/ttyUSB1")
    parser.add_argument("--motor-baudrate", type=int, default=1_000_000)
    parser.add_argument("--motor-state-rate", type=float, default=10.0)
    parser.add_argument("--motor-min-position", type=float, default=-360.0)
    parser.add_argument("--motor-max-position", type=float, default=360.0)
    parser.add_argument("--motor-max-step", type=float, default=10.0)
    parser.add_argument("--control-rate", type=float, default=10.0)
    parser.add_argument("--command-timeout-ms", type=int, default=1500)
    parser.add_argument("--max-tracking-error-deg", type=float, default=2.0)
    parser.add_argument("--motor-zero-deg", nargs=4, type=float, default=(90, 90, 90, 90))
    parser.add_argument("--target-force-xyz-n", nargs=3, type=float, default=(0, 0, 1))
    parser.add_argument("--gains-mm-per-n", nargs=3, type=float, default=(0.02, 0.02, 0.05))
    parser.add_argument("--motion-signs", nargs=3, type=float, default=(1, 1, 1))
    parser.add_argument("--sensor-rpy-deg", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--contact-offset-m", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--finger", default="thumb")
    parser.add_argument("--device-id", type=parse_int, default=0x03)
    parser.add_argument("--tactile-rate", type=float, default=30.0)
    parser.add_argument("--force-baudrate", type=int, default=2400)
    parser.add_argument("--stale-timeout", type=float, default=1.0)
    parser.add_argument(
        "--no-start-nodes",
        action="store_true",
        help="Only subscribe; do not launch the hardware node processes.",
    )
    args = parser.parse_args(argv)
    for name in (
        "tactile_rate", "stale_timeout", "motor_state_rate", "motor_max_step",
        "control_rate", "max_tracking_error_deg",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a finite number greater than zero")
    if args.force_baudrate <= 0:
        parser.error("--force-baudrate must be greater than zero")
    if args.motor_baudrate <= 0:
        parser.error("--motor-baudrate must be greater than zero")
    if args.command_timeout_ms <= 0:
        parser.error("--command-timeout-ms must be greater than zero")
    if args.motor_min_position >= args.motor_max_position:
        parser.error("--motor-min-position must be less than --motor-max-position")
    if not all(math.isfinite(value) for value in args.target_force_xyz_n):
        parser.error("--target-force-xyz-n values must be finite")
    if not all(math.isfinite(value) and value >= 0 for value in args.gains_mm_per_n):
        parser.error("--gains-mm-per-n values must be finite and non-negative")
    if any(value not in (-1, 1) for value in args.motion_signs):
        parser.error("--motion-signs values must each be -1 or 1")
    for name in ("motor_zero_deg", "sensor_rpy_deg", "contact_offset_m"):
        if not all(math.isfinite(value) for value in getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} values must be finite")
    return args


class CalibrationWindow(QtWidgets.QWidget):
    def __init__(self, args: argparse.Namespace, parent=None) -> None:
        super().__init__(parent)
        self.args = args
        self.latest_tactile: TactileReading | None = None
        self.latest_reference: float | None = None
        self.tactile_received_at: float | None = None
        self.reference_received_at: float | None = None
        self.motor_received_at: float | None = None
        self.control_received_at: float | None = None
        self.control_state: dict[str, Any] = {}
        self.tactile_zero_xyz_n: list[float] | None = None
        self.motor_positions: dict[int, float] = {}
        self.motor_torque: dict[int, bool] = {}
        self.motor_targets_initialized = False
        self.samples: list[CalibrationSample] = []
        self.processes: dict[str, QtCore.QProcess] = {}

        self.setWindowTitle("触觉传感器接触力标定")
        self.resize(1080, 900)
        self._build_ui()
        self._setup_subscribers()
        if args.no_start_nodes:
            self.tactile_process_status.setText("外部进程（订阅中）")
            self.force_process_status.setText("外部进程（订阅中）")
            self.motor_process_status.setText("外部进程（订阅中）")
            self.control_process_status.setText("外部进程（订阅中）")
        else:
            self._start_nodes()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll_messages)
        self.poll_timer.start(20)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.update_freshness)
        self.status_timer.start(200)

    def _value_label(self) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel("--")
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        label.setFont(font)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumHeight(70)
        return label

    def _build_ui(self) -> None:
        self.tactile_status = QtWidgets.QLabel("等待数据")
        self.reference_status = QtWidgets.QLabel("等待数据")
        self.motor_status = QtWidgets.QLabel("等待电机状态")
        self.tactile_process_status = QtWidgets.QLabel("未启动")
        self.force_process_status = QtWidgets.QLabel("未启动")
        self.motor_process_status = QtWidgets.QLabel("未启动")
        self.control_process_status = QtWidgets.QLabel("未启动")
        self.fx_label = self._value_label()
        self.fy_label = self._value_label()
        self.fz_label = self._value_label()
        self.magnitude_label = self._value_label()
        self.reference_label = self._value_label()

        process_group = QtWidgets.QGroupBox("节点状态")
        process_layout = QtWidgets.QGridLayout(process_group)
        process_layout.addWidget(QtWidgets.QLabel("触觉合力节点"), 0, 0)
        process_layout.addWidget(self.tactile_process_status, 0, 1)
        process_layout.addWidget(QtWidgets.QLabel(self.args.tactile_endpoint), 0, 2)
        process_layout.addWidget(QtWidgets.QLabel("参考力节点"), 1, 0)
        process_layout.addWidget(self.force_process_status, 1, 1)
        process_layout.addWidget(QtWidgets.QLabel(self.args.force_endpoint), 1, 2)
        process_layout.addWidget(QtWidgets.QLabel("手指电机节点"), 2, 0)
        process_layout.addWidget(self.motor_process_status, 2, 1)
        process_layout.addWidget(QtWidgets.QLabel(self.args.motor_state_endpoint), 2, 2)
        process_layout.addWidget(QtWidgets.QLabel("三轴导纳控制节点"), 3, 0)
        process_layout.addWidget(self.control_process_status, 3, 1)
        process_layout.addWidget(QtWidgets.QLabel(self.args.control_state_endpoint), 3, 2)

        tactile_group = QtWidgets.QGroupBox(
            f"触觉接触合力（{self.args.finger}，N；未归零时显示原始值）"
        )
        tactile_layout = QtWidgets.QGridLayout(tactile_group)
        for column, (name, label) in enumerate(
            (("Fx", self.fx_label), ("Fy", self.fy_label), ("Fz", self.fz_label), ("|F|", self.magnitude_label))
        ):
            heading = QtWidgets.QLabel(name)
            heading.setAlignment(QtCore.Qt.AlignCenter)
            tactile_layout.addWidget(heading, 0, column)
            tactile_layout.addWidget(label, 1, column)
        tactile_layout.addWidget(self.tactile_status, 2, 0, 1, 4)

        reference_group = QtWidgets.QGroupBox("标准力传感器")
        reference_layout = QtWidgets.QVBoxLayout(reference_group)
        reference_layout.addWidget(self.reference_label)
        reference_layout.addWidget(self.reference_status)

        self.motor_position_labels: dict[int, QtWidgets.QLabel] = {}
        self.motor_target_boxes: dict[int, QtWidgets.QDoubleSpinBox] = {}
        self.motor_send_buttons: dict[int, QtWidgets.QPushButton] = {}
        robot_group = self._build_finger_controls("机械手指", ROBOT_MOTOR_IDS)
        exoskeleton_group = self._build_finger_controls(
            "外骨骼手指", EXOSKELETON_MOTOR_IDS
        )
        self.copy_positions_button = QtWidgets.QPushButton("目标值同步到当前位置")
        self.copy_positions_button.setEnabled(False)
        self.copy_positions_button.clicked.connect(self.copy_current_positions)
        self.torque_on_button = QtWidgets.QPushButton("使能全部电机")
        self.torque_on_button.setEnabled(False)
        self.torque_on_button.clicked.connect(self.torque_on)
        self.torque_off_button = QtWidgets.QPushButton("失能全部电机")
        self.torque_off_button.setEnabled(False)
        self.torque_off_button.clicked.connect(self.torque_off)
        motor_buttons = QtWidgets.QHBoxLayout()
        motor_buttons.addWidget(self.copy_positions_button)
        motor_buttons.addWidget(self.torque_on_button)
        motor_buttons.addWidget(self.torque_off_button)
        motor_buttons.addStretch(1)
        motor_group = QtWidgets.QGroupBox("关节位置读取与控制（绝对电机角度，deg）")
        motor_layout = QtWidgets.QVBoxLayout(motor_group)
        finger_layout = QtWidgets.QHBoxLayout()
        finger_layout.addWidget(robot_group)
        finger_layout.addWidget(exoskeleton_group)
        motor_layout.addLayout(finger_layout)
        motor_layout.addLayout(motor_buttons)
        motor_layout.addWidget(self.motor_status)

        control_group = self._build_admittance_controls()

        self.capture_button = QtWidgets.QPushButton("记录当前标定点")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_sample)
        self.clear_button = QtWidgets.QPushButton("清空")
        self.clear_button.clicked.connect(self.clear_samples)
        self.save_button = QtWidgets.QPushButton("保存 CSV")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_samples)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch(1)

        self.sample_table = QtWidgets.QTableWidget(0, 7)
        self.sample_table.setHorizontalHeaderLabels(
            ["序号", "Fx (N)", "Fy (N)", "Fz (N)", "|F| (N)", "标准力", "采集时间"]
        )
        self.sample_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.sample_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        log_group = QtWidgets.QGroupBox("节点日志")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.addWidget(self.log_view)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(process_group)
        sensor_layout = QtWidgets.QHBoxLayout()
        sensor_layout.addWidget(tactile_group, 4)
        sensor_layout.addWidget(reference_group, 1)
        main_layout.addLayout(sensor_layout)
        main_layout.addWidget(motor_group)
        main_layout.addWidget(control_group)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.sample_table, 2)
        main_layout.addWidget(log_group, 1)

    def _build_finger_controls(
        self, title: str, motor_ids: Sequence[int]
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QGridLayout(group)
        for column, heading in enumerate(("ID", "当前位置", "目标位置", "控制")):
            layout.addWidget(QtWidgets.QLabel(heading), 0, column)
        for row, motor_id in enumerate(motor_ids, start=1):
            position_label = QtWidgets.QLabel("--")
            position_label.setMinimumWidth(80)
            target_box = QtWidgets.QDoubleSpinBox()
            target_box.setRange(
                self.args.motor_min_position, self.args.motor_max_position
            )
            target_box.setDecimals(2)
            target_box.setSingleStep(1.0)
            target_box.setSuffix("°")
            target_box.setMinimumWidth(110)
            send_button = QtWidgets.QPushButton("发送")
            send_button.setEnabled(False)
            send_button.clicked.connect(
                lambda _checked=False, selected_id=motor_id: self.send_motor_target(
                    selected_id
                )
            )
            self.motor_position_labels[motor_id] = position_label
            self.motor_target_boxes[motor_id] = target_box
            self.motor_send_buttons[motor_id] = send_button
            layout.addWidget(QtWidgets.QLabel(str(motor_id)), row, 0)
            layout.addWidget(position_label, row, 1)
            layout.addWidget(target_box, row, 2)
            layout.addWidget(send_button, row, 3)
        return group

    def _build_admittance_controls(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("URDF雅可比 + XYZ导纳力控")
        layout = QtWidgets.QGridLayout(group)
        self.force_target_boxes: list[QtWidgets.QDoubleSpinBox] = []
        self.admittance_gain_boxes: list[QtWidgets.QDoubleSpinBox] = []
        self.motion_sign_boxes: list[QtWidgets.QComboBox] = []
        for column, heading in enumerate(("轴", "目标力 (N)", "增益 (mm/N·周期)", "运动符号")):
            layout.addWidget(QtWidgets.QLabel(heading), 0, column)
        for row, (axis, target, gain, sign_value) in enumerate(
            zip(
                ("X", "Y", "Z"),
                self.args.target_force_xyz_n,
                self.args.gains_mm_per_n,
                self.args.motion_signs,
            ),
            start=1,
        ):
            target_box = QtWidgets.QDoubleSpinBox()
            target_box.setRange(-20.0, 20.0)
            target_box.setDecimals(3)
            target_box.setSingleStep(0.1)
            target_box.setValue(target)
            gain_box = QtWidgets.QDoubleSpinBox()
            gain_box.setRange(0.0, 2.0)
            gain_box.setDecimals(4)
            gain_box.setSingleStep(0.01)
            gain_box.setValue(gain)
            sign_box = QtWidgets.QComboBox()
            sign_box.addItems(("+1", "-1"))
            sign_box.setCurrentIndex(0 if sign_value == 1 else 1)
            self.force_target_boxes.append(target_box)
            self.admittance_gain_boxes.append(gain_box)
            self.motion_sign_boxes.append(sign_box)
            layout.addWidget(QtWidgets.QLabel(axis), row, 0)
            layout.addWidget(target_box, row, 1)
            layout.addWidget(gain_box, row, 2)
            layout.addWidget(sign_box, row, 3)

        self.control_mode_label = QtWidgets.QLabel("未连接")
        self.control_mode_label.setWordWrap(True)
        self.control_mode_label.setMinimumWidth(260)
        self.control_mode_label.setMaximumWidth(520)
        self.control_mode_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        self.control_force_label = QtWidgets.QLabel("测量力：--")
        self.control_urdf_label = QtWidgets.QLabel("URDF关节：--")
        self.control_jacobian_label = QtWidgets.QLabel("雅可比条件数：--")
        self.control_joint_target_label = QtWidgets.QLabel(
            "闭环关节目标：ID1=--, ID3=--, ID4=--"
        )
        self.zero_force_button = QtWidgets.QPushButton("无接触归零")
        self.zero_force_button.setEnabled(False)
        self.zero_force_button.clicked.connect(self.zero_control_force)
        self.apply_control_button = QtWidgets.QPushButton("应用目标和增益")
        self.apply_control_button.setEnabled(False)
        self.apply_control_button.clicked.connect(self.apply_control_settings)
        self.start_control_button = QtWidgets.QPushButton("启动XYZ力控")
        self.start_control_button.setEnabled(False)
        self.start_control_button.clicked.connect(self.start_force_control)
        self.stop_control_button = QtWidgets.QPushButton("停止力控")
        self.stop_control_button.setEnabled(False)
        self.stop_control_button.clicked.connect(self.stop_force_control)
        self.emergency_stop_button = QtWidgets.QPushButton("急停并失能")
        self.emergency_stop_button.setEnabled(False)
        self.emergency_stop_button.setStyleSheet(
            "QPushButton { background-color: #b71c1c; color: white; font-weight: bold; }"
        )
        self.emergency_stop_button.clicked.connect(self.emergency_stop)
        buttons = QtWidgets.QHBoxLayout()
        for button in (
            self.zero_force_button,
            self.apply_control_button,
            self.start_control_button,
            self.stop_control_button,
            self.emergency_stop_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addWidget(self.control_mode_label, 0, 4, 1, 2)
        layout.addWidget(self.control_force_label, 1, 4, 1, 2)
        layout.addWidget(self.control_urdf_label, 2, 4, 1, 2)
        layout.addWidget(self.control_jacobian_label, 3, 4, 1, 2)
        layout.addWidget(self.control_joint_target_label, 4, 0, 1, 6)
        layout.addLayout(buttons, 5, 0, 1, 6)
        return group

    def _new_subscriber(self, endpoint: str):
        socket = self.context.socket(zmq.SUB)
        socket.linger = 0
        socket.setsockopt(zmq.RCVHWM, 10)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(endpoint)
        return socket

    def _setup_subscribers(self) -> None:
        self.context = zmq.Context.instance()
        self.tactile_socket = self._new_subscriber(self.args.tactile_endpoint)
        self.force_socket = self._new_subscriber(self.args.force_endpoint)
        self.motor_state_socket = self._new_subscriber(self.args.motor_state_endpoint)
        self.control_state_socket = self._new_subscriber(self.args.control_state_endpoint)
        self._create_motor_command_socket()
        self._create_control_command_socket()

    def _create_motor_command_socket(self) -> None:
        existing = getattr(self, "motor_command_socket", None)
        if existing is not None:
            existing.close(0)
        self.motor_command_socket = self.context.socket(zmq.REQ)
        self.motor_command_socket.linger = 0
        self.motor_command_socket.setsockopt(zmq.SNDTIMEO, self.args.command_timeout_ms)
        self.motor_command_socket.setsockopt(zmq.RCVTIMEO, self.args.command_timeout_ms)
        self.motor_command_socket.connect(self.args.motor_command_endpoint)

    def _create_control_command_socket(self) -> None:
        existing = getattr(self, "control_command_socket", None)
        if existing is not None:
            existing.close(0)
        self.control_command_socket = self.context.socket(zmq.REQ)
        self.control_command_socket.linger = 0
        self.control_command_socket.setsockopt(zmq.SNDTIMEO, self.args.command_timeout_ms)
        self.control_command_socket.setsockopt(zmq.RCVTIMEO, self.args.command_timeout_ms)
        self.control_command_socket.connect(self.args.control_command_endpoint)

    def _start_nodes(self) -> None:
        tactile_args = [
            str(TACTILE_NODE),
            "--port", self.args.tactile_port,
            "--finger", self.args.finger,
            "--device-id", hex(self.args.device_id),
            "--publish-rate", str(self.args.tactile_rate),
            "--endpoint", self.args.tactile_endpoint,
        ]
        force_args = [
            str(FORCE_NODE),
            "--port", self.args.force_port,
            "--baudrate", str(self.args.force_baudrate),
            "--endpoint", self.args.force_endpoint,
        ]
        motor_args = [
            str(MOTOR_NODE),
            "--port", self.args.motor_port,
            "--baudrate", str(self.args.motor_baudrate),
            "--state-rate", str(self.args.motor_state_rate),
            "--min-position-deg", str(self.args.motor_min_position),
            "--max-position-deg", str(self.args.motor_max_position),
            "--max-step-deg", str(self.args.motor_max_step),
            "--command-endpoint", self.args.motor_command_endpoint,
            "--state-endpoint", self.args.motor_state_endpoint,
        ]
        control_args = [
            str(CONTROL_NODE),
            "--finger", self.args.finger,
            "--tactile-endpoint", self.args.tactile_endpoint,
            "--motor-command-endpoint", self.args.motor_command_endpoint,
            "--motor-state-endpoint", self.args.motor_state_endpoint,
            "--command-endpoint", self.args.control_command_endpoint,
            "--state-endpoint", self.args.control_state_endpoint,
            "--control-rate", str(self.args.control_rate),
            "--motor-zero-deg", *[str(value) for value in self.args.motor_zero_deg],
            "--target-force-xyz-n", *[str(value) for value in self.args.target_force_xyz_n],
            "--gains-mm-per-n", *[str(value) for value in self.args.gains_mm_per_n],
            "--motion-signs", *[str(value) for value in self.args.motion_signs],
            "--max-tracking-error-deg", str(self.args.max_tracking_error_deg),
            "--sensor-rpy-deg", *[str(value) for value in self.args.sensor_rpy_deg],
            "--contact-offset-m", *[str(value) for value in self.args.contact_offset_m],
        ]
        self._start_process("tactile", tactile_args, self.tactile_process_status)
        self._start_process("force", force_args, self.force_process_status)
        self._start_process("motor", motor_args, self.motor_process_status)
        self._start_process("control", control_args, self.control_process_status)

    def _start_process(
        self, name: str, arguments: list[str], status_label: QtWidgets.QLabel
    ) -> None:
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProgram(sys.executable)
        process.setArguments(arguments)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.started.connect(lambda: status_label.setText("运行中"))
        process.finished.connect(
            lambda code, _status: status_label.setText(f"已退出（{code}）")
        )
        process.errorOccurred.connect(
            lambda _error: status_label.setText(f"启动失败：{process.errorString()}")
        )
        process.readyReadStandardOutput.connect(
            lambda: self._append_process_output(name, process)
        )
        self.processes[name] = process
        process.start()
        status_label.setText("正在启动…")

    def _append_process_output(self, name: str, process: QtCore.QProcess) -> None:
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in output.rstrip().splitlines():
            self.log_view.appendPlainText(f"[{name}] {line}")

    def poll_messages(self) -> None:
        self._poll_tactile()
        self._poll_reference()
        self._poll_motor_state()
        self._poll_control_state()

    def _poll_tactile(self) -> None:
        latest = None
        while True:
            try:
                latest = self.tactile_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                self.tactile_status.setText(f"消息错误：{exc}")
                return
        if latest is None:
            return
        try:
            raw_reading = decode_tactile_payload(latest, self.args.finger)
            reading = subtract_tactile_zero(raw_reading, self.tactile_zero_xyz_n)
        except ValueError as exc:
            self.tactile_status.setText(f"消息错误：{exc}")
            return
        self.latest_tactile = reading
        self.tactile_received_at = time.monotonic()
        self.fx_label.setText(f"{reading.fx_n:.2f}")
        self.fy_label.setText(f"{reading.fy_n:.2f}")
        self.fz_label.setText(f"{reading.fz_n:.2f}")
        self.magnitude_label.setText(f"{reading.magnitude_n:.2f}")

    def _poll_reference(self) -> None:
        latest = None
        while True:
            try:
                latest = self.force_socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                self.reference_status.setText(f"消息错误：{exc}")
                return
        if latest is None:
            return
        try:
            self.latest_reference = decode_reference_force(latest)
        except ValueError as exc:
            self.reference_status.setText(f"消息错误：{exc}")
            return
        self.reference_received_at = time.monotonic()
        self.reference_label.setText(f"{self.latest_reference:.3f}")

    def _poll_motor_state(self) -> None:
        latest = None
        while True:
            try:
                latest = self.motor_state_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                self.motor_status.setText(f"电机状态消息错误：{exc}")
                return
        if latest is None:
            return
        try:
            positions, torque = decode_motor_state(latest)
        except ValueError as exc:
            self.motor_status.setText(f"电机状态消息错误：{exc}")
            return
        self.motor_positions = positions
        self.motor_torque = torque
        self.motor_received_at = time.monotonic()
        for motor_id, position in positions.items():
            self.motor_position_labels[motor_id].setText(f"{position:.2f}°")
        if not self.motor_targets_initialized:
            self.copy_current_positions()
            self.motor_targets_initialized = True

    def _motor_request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            self.motor_command_socket.send_json(payload)
            response = self.motor_command_socket.recv_json()
        except Exception as exc:
            self.motor_status.setText(f"电机命令失败：{exc}")
            self._create_motor_command_socket()
            return None
        if not isinstance(response, dict) or not response.get("ok"):
            message = response.get("error", "invalid response") if isinstance(response, dict) else response
            self.motor_status.setText(f"电机命令被拒绝：{message}")
            return None
        return response

    def copy_current_positions(self) -> None:
        for motor_id, position in self.motor_positions.items():
            self.motor_target_boxes[motor_id].setValue(position)

    def send_motor_target(self, motor_id: int) -> None:
        target = self.motor_target_boxes[motor_id].value()
        response = self._motor_request(
            {"cmd": "set_position", "id": motor_id, "position_deg": target}
        )
        if response is not None:
            self.motor_status.setText(f"已发送：ID {motor_id} → {target:.2f}°")

    def torque_on(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "使能电机",
            "将先锁定全部电机的当前位置，再使能力矩。确认机械结构安全？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        if self._motor_request({"cmd": "torque_on"}) is not None:
            self.motor_status.setText("全部电机已使能，目标已锁定为当前位置")

    def torque_off(self) -> None:
        if self._motor_request({"cmd": "torque_off"}) is not None:
            self.motor_status.setText("全部电机已失能")

    def _poll_control_state(self) -> None:
        latest = None
        while True:
            try:
                latest = self.control_state_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:
                self.control_mode_label.setText(f"控制状态消息错误：{exc}")
                return
        if latest is None:
            return
        try:
            state = decode_control_state(latest)
        except ValueError as exc:
            self.control_mode_label.setText(f"控制状态消息错误：{exc}")
            return
        self.control_state = state
        self.control_received_at = time.monotonic()
        zero = state.get("force_zero_xyz_n")
        self.tactile_zero_xyz_n = (
            None if zero is None else [float(value) for value in zero]
        )
        mode = state.get("mode", "unknown")
        fault = state.get("fault")
        self.control_mode_label.setText(
            f"状态：{mode}" + (f" · 故障：{fault}" if fault else "")
        )
        measured = state.get("measured_force_xyz_n")
        target = state.get("target_force_xyz_n")
        if measured is not None:
            self.control_force_label.setText(
                "测量/目标力："
                + ", ".join(
                    f"{axis}={float(value):.3f}/{float(goal):.3f}N"
                    for axis, value, goal in zip("XYZ", measured, target)
                )
            )
        q_deg = state.get("urdf_q_deg")
        if q_deg is not None:
            self.control_urdf_label.setText(
                "URDF关节：" + ", ".join(
                    f"J{index}={float(value):.2f}°"
                    for index, value in enumerate(q_deg, start=1)
                )
            )
        condition = state.get("jacobian_condition")
        self.control_jacobian_label.setText(
            "雅可比条件数：--" if condition is None else f"雅可比条件数：{float(condition):.2f}"
        )
        commanded = state.get("last_commanded_motor_deg", {})
        tracking = state.get("tracking_error_deg", {})
        if commanded:
            values = []
            for motor_id in (1, 3, 4):
                key = str(motor_id)
                target_value = commanded.get(key)
                error_value = tracking.get(key)
                if target_value is None:
                    values.append(f"ID{motor_id}=--")
                elif error_value is None:
                    values.append(f"ID{motor_id}={float(target_value):.3f}°")
                else:
                    values.append(
                        f"ID{motor_id}={float(target_value):.3f}° "
                        f"(误差{float(error_value):+.3f}°)"
                    )
            self.control_joint_target_label.setText(
                "闭环关节目标：" + ", ".join(values)
            )

    def _control_request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            self.control_command_socket.send_json(payload)
            response = self.control_command_socket.recv_json()
        except Exception as exc:
            self.control_mode_label.setText(f"控制命令失败：{exc}")
            self._create_control_command_socket()
            return None
        if not isinstance(response, dict) or not response.get("ok"):
            message = response.get("error", "invalid response") if isinstance(response, dict) else response
            self.control_mode_label.setText(f"控制命令被拒绝：{message}")
            return None
        return response

    def _target_values(self) -> list[float]:
        return [box.value() for box in self.force_target_boxes]

    def _gain_values(self) -> list[float]:
        return [box.value() for box in self.admittance_gain_boxes]

    def _motion_sign_values(self) -> list[int]:
        return [1 if box.currentIndex() == 0 else -1 for box in self.motion_sign_boxes]

    def apply_control_settings(self) -> bool:
        if self._control_request(
            {"cmd": "set_target", "target_force_xyz_n": self._target_values()}
        ) is None:
            return False
        if self._control_request(
            {
                "cmd": "set_gains",
                "gains_mm_per_n": self._gain_values(),
                "motion_signs": self._motion_sign_values(),
            }
        ) is None:
            return False
        self.control_mode_label.setText("目标力、导纳增益和运动符号已应用")
        return True

    def zero_control_force(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "触觉归零",
            "请确保触觉传感器完全无接触。开始采集零偏？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            if self._control_request({"cmd": "zero"}) is not None:
                self.control_mode_label.setText("状态：zeroing · 正在采集无接触零偏…")

    def start_force_control(self) -> None:
        if not self.apply_control_settings():
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "启动XYZ力控",
            "控制器将调节机械手指 joint1、joint3、joint4。确认已轻接触且运动符号正确？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self._control_request({"cmd": "start"})

    def stop_force_control(self) -> None:
        self._control_request({"cmd": "stop"})

    def emergency_stop(self) -> None:
        if self._is_fresh(self.control_received_at):
            self._control_request({"cmd": "emergency_stop"})
        else:
            self._motor_request({"cmd": "torque_off"})

    def _is_fresh(self, received_at: float | None) -> bool:
        return received_at is not None and time.monotonic() - received_at <= self.args.stale_timeout

    def update_freshness(self) -> None:
        tactile_fresh = self._is_fresh(self.tactile_received_at)
        reference_fresh = self._is_fresh(self.reference_received_at)
        motor_fresh = self._is_fresh(self.motor_received_at)
        control_fresh = self._is_fresh(self.control_received_at)
        control_active = bool(self.control_state.get("active")) if control_fresh else False
        if tactile_fresh:
            zero_status = (
                "已软件归零"
                if self.tactile_zero_xyz_n is not None
                else "未归零（原始值）"
            )
            self.tactile_status.setText(f"数据正常 · {zero_status}")
        else:
            self.tactile_status.setText("等待数据或数据超时")
        self.reference_status.setText("数据正常" if reference_fresh else "等待数据或数据超时")
        if motor_fresh:
            enabled_count = sum(self.motor_torque.values())
            self.motor_status.setText(
                f"电机状态正常 · 已使能 {enabled_count}/{len(ALL_MOTOR_IDS)}"
            )
        else:
            self.motor_status.setText("等待电机状态或数据超时")
        self.copy_positions_button.setEnabled(motor_fresh and not control_active)
        self.torque_on_button.setEnabled(
            motor_fresh and not all(self.motor_torque.values()) and not control_active
        )
        self.torque_off_button.setEnabled(
            motor_fresh and any(self.motor_torque.values())
        )
        for motor_id, button in self.motor_send_buttons.items():
            button.setEnabled(
                motor_fresh
                and self.motor_torque.get(motor_id, False)
                and not control_active
            )
        self.zero_force_button.setEnabled(control_fresh and not control_active)
        self.apply_control_button.setEnabled(control_fresh and not control_active)
        self.start_control_button.setEnabled(control_fresh and not control_active)
        self.stop_control_button.setEnabled(control_fresh and control_active)
        self.emergency_stop_button.setEnabled(control_fresh or motor_fresh)
        self.capture_button.setEnabled(tactile_fresh and reference_fresh)

    def capture_sample(self) -> None:
        if self.latest_tactile is None or self.latest_reference is None:
            return
        reading = self.latest_tactile
        sample = CalibrationSample(
            captured_at=time.time(),
            finger=reading.finger,
            tactile_timestamp=reading.source_timestamp,
            fx_n=reading.fx_n,
            fy_n=reading.fy_n,
            fz_n=reading.fz_n,
            magnitude_n=reading.magnitude_n,
            reference_force=self.latest_reference,
        )
        self.samples.append(sample)
        row = self.sample_table.rowCount()
        self.sample_table.insertRow(row)
        values = (
            str(row + 1),
            f"{sample.fx_n:.3f}",
            f"{sample.fy_n:.3f}",
            f"{sample.fz_n:.3f}",
            f"{sample.magnitude_n:.3f}",
            f"{sample.reference_force:.3f}",
            datetime.fromtimestamp(sample.captured_at).strftime("%H:%M:%S.%f")[:-3],
        )
        for column, value in enumerate(values):
            self.sample_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.save_button.setEnabled(True)

    def clear_samples(self) -> None:
        self.samples.clear()
        self.sample_table.setRowCount(0)
        self.save_button.setEnabled(False)

    def save_samples(self) -> None:
        if not self.samples:
            return
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        suggested = DEFAULT_OUTPUT_DIR / datetime.now().strftime(
            "tactile_force_calibration_%Y%m%d_%H%M%S.csv"
        )
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存标定数据", str(suggested), "CSV files (*.csv)"
        )
        if not selected:
            return
        try:
            save_samples_csv(Path(selected), self.samples)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.log_view.appendPlainText(f"[ui] 已保存 {len(self.samples)} 个标定点：{selected}")

    def _stop_processes(self) -> None:
        for process in self.processes.values():
            if process.state() != QtCore.QProcess.NotRunning:
                process.terminate()
        for process in self.processes.values():
            if process.state() != QtCore.QProcess.NotRunning and not process.waitForFinished(1500):
                process.kill()
                process.waitForFinished(500)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.poll_timer.stop()
        self.status_timer.stop()
        self._stop_processes()
        self.tactile_socket.close(0)
        self.force_socket.close(0)
        self.motor_state_socket.close(0)
        self.motor_command_socket.close(0)
        self.control_state_socket.close(0)
        self.control_command_socket.close(0)
        event.accept()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    window = CalibrationWindow(args)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    signal_timer = QtCore.QTimer()
    signal_timer.start(200)
    signal_timer.timeout.connect(lambda: None)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
