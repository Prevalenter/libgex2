"""Publish UI-controlled fake EX16 joint positions over ZMQ."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import zmq
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyQt5 is required: python -m pip install PyQt5") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_COUNT = 16
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5567"
DEFAULT_POSE_DIR = REPO_ROOT / "utils" / "GeoRT" / "data" / "pose" / "ex16"
STATE_TOPIC = "ex16/state"
POSE_FORMAT = "libgex.ex16_pose"
POSE_VERSION = 1
POSE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-\u3400-\u4dbf\u4e00-\u9fff]+$")


def validate_pose_name(name: str) -> str:
    """Return a safe pose name or raise ValueError."""
    name = name.strip()
    if not name:
        raise ValueError("姿态名称不能为空")
    if len(name) > 80:
        raise ValueError("姿态名称不能超过 80 个字符")
    if not POSE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "姿态名称只能包含中文、字母、数字、下划线和连字符"
        )
    return name


def validate_positions(values: Any, angle_min: int, angle_max: int) -> list[int]:
    """Validate and normalize a saved EX16 pose."""
    if not isinstance(values, list) or len(values) != JOINT_COUNT:
        count = len(values) if isinstance(values, list) else "非列表"
        raise ValueError(f"urdf_deg 必须包含 {JOINT_COUNT} 个关节值，当前为 {count}")

    positions: list[int] = []
    for index, value in enumerate(values, start=1):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"J{index:02d} 不是有效数值")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"J{index:02d} 必须是有限数值")
        if not numeric.is_integer():
            raise ValueError(f"J{index:02d} 必须是整数角度")
        position = int(numeric)
        if not angle_min <= position <= angle_max:
            raise ValueError(
                f"J{index:02d}={position} 超出范围 [{angle_min}, {angle_max}]"
            )
        positions.append(position)
    return positions


def pose_path(pose_dir: Path, name: str) -> Path:
    return pose_dir / f"{validate_pose_name(name)}.json"


def save_pose_file(
    pose_dir: Path,
    name: str,
    positions: Sequence[int],
    angle_min: int,
    angle_max: int,
) -> Path:
    """Atomically save one EX16 pose and return its path."""
    safe_name = validate_pose_name(name)
    normalized = validate_positions(list(positions), angle_min, angle_max)
    pose_dir.mkdir(parents=True, exist_ok=True)
    destination = pose_path(pose_dir, safe_name)
    payload = {
        "format": POSE_FORMAT,
        "version": POSE_VERSION,
        "name": safe_name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "urdf_deg": normalized,
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=pose_dir,
            prefix=f".{safe_name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_pose_file(path: Path, angle_min: int, angle_max: int) -> list[int]:
    """Load and validate one versioned EX16 pose file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("姿态文件顶层必须是 JSON 对象")
    if payload.get("format") != POSE_FORMAT:
        raise ValueError(f"不支持的姿态格式，应为 {POSE_FORMAT}")
    if payload.get("version") != POSE_VERSION:
        raise ValueError(f"不支持的姿态版本，应为 {POSE_VERSION}")
    return validate_positions(payload.get("urdf_deg"), angle_min, angle_max)


def list_pose_names(pose_dir: Path) -> list[str]:
    """List valid pose filenames without loading their contents."""
    if not pose_dir.exists():
        return []
    return sorted(
        path.stem
        for path in pose_dir.glob("*.json")
        if path.is_file() and POSE_NAME_PATTERN.fullmatch(path.stem)
    )


def build_state_payload(sequence: int, positions: Sequence[int]) -> dict[str, Any]:
    return {
        "name": "ex16",
        "sequence": sequence,
        "timestamp": time.time(),
        "urdf_deg": [float(value) for value in positions],
    }


class JointSlider(QtWidgets.QWidget):
    """Synchronized slider and integer spin box for one joint."""

    def __init__(
        self, joint_index: int, angle_min: int, angle_max: int, parent=None
    ) -> None:
        super().__init__(parent)
        self.name_label = QtWidgets.QLabel(f"J{joint_index:02d}")
        self.name_label.setMinimumWidth(38)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(angle_min, angle_max)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setMinimumWidth(300)

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setRange(angle_min, angle_max)
        self.spinbox.setValue(0)
        self.spinbox.setSuffix("°")
        self.spinbox.setMinimumWidth(90)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spinbox)

        self.slider.valueChanged.connect(self._sync_from_slider)
        self.spinbox.valueChanged.connect(self._sync_from_spinbox)

    def _sync_from_slider(self, value: int) -> None:
        blocker = QtCore.QSignalBlocker(self.spinbox)
        self.spinbox.setValue(value)
        del blocker

    def _sync_from_spinbox(self, value: int) -> None:
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(value)
        del blocker

    def value(self) -> int:
        return self.spinbox.value()

    def set_value(self, value: int) -> None:
        slider_blocker = QtCore.QSignalBlocker(self.slider)
        spinbox_blocker = QtCore.QSignalBlocker(self.spinbox)
        self.slider.setValue(value)
        self.spinbox.setValue(value)
        del slider_blocker
        del spinbox_blocker


class FakeEX16Window(QtWidgets.QWidget):
    def __init__(self, args: argparse.Namespace, parent=None) -> None:
        super().__init__(parent)
        self.args = args
        self.pose_dir = Path(args.pose_dir).expanduser().resolve()
        self.pose_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = 0

        self.context = zmq.Context.instance()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.linger = 0
        self.publisher.sndhwm = 1
        try:
            self.publisher.bind(args.state_endpoint)
        except Exception:
            self.publisher.close()
            raise

        self.setWindowTitle("Fake EX16 State Publisher")
        self.resize(760, 820)
        self.joint_sliders = [
            JointSlider(index, args.angle_min, args.angle_max)
            for index in range(1, JOINT_COUNT + 1)
        ]
        self.pose_name_edit = QtWidgets.QLineEdit()
        self.pose_name_edit.setPlaceholderText("输入姿态名称")
        self.pose_combo = QtWidgets.QComboBox()
        self.zero_button = QtWidgets.QPushButton("全部归零")
        self.save_button = QtWidgets.QPushButton("保存姿态")
        self.load_button = QtWidgets.QPushButton("读取姿态")
        self.refresh_button = QtWidgets.QPushButton("刷新列表")
        self.status_label = QtWidgets.QLabel()

        self._build_layout()
        self._connect_signals()
        self.refresh_pose_list()

        self.publish_timer = QtCore.QTimer(self)
        self.publish_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.publish_timer.setInterval(max(1, round(1000.0 / args.state_hz)))
        self.publish_timer.timeout.connect(self.publish_state)
        self.publish_timer.start()
        self.publish_state()
        self._set_status("正在发布零位姿态")

    def _build_layout(self) -> None:
        publisher_group = QtWidgets.QGroupBox("发布状态")
        publisher_layout = QtWidgets.QGridLayout(publisher_group)
        publisher_layout.addWidget(QtWidgets.QLabel("Endpoint"), 0, 0)
        publisher_layout.addWidget(QtWidgets.QLabel(self.args.state_endpoint), 0, 1)
        publisher_layout.addWidget(QtWidgets.QLabel("Topic"), 1, 0)
        publisher_layout.addWidget(QtWidgets.QLabel(STATE_TOPIC), 1, 1)
        publisher_layout.addWidget(QtWidgets.QLabel("频率"), 2, 0)
        publisher_layout.addWidget(QtWidgets.QLabel(f"{self.args.state_hz:g} Hz"), 2, 1)

        sliders_widget = QtWidgets.QWidget()
        sliders_layout = QtWidgets.QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(10, 10, 10, 10)
        for slider in self.joint_sliders:
            sliders_layout.addWidget(slider)
        sliders_layout.addStretch(1)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(sliders_widget)

        pose_group = QtWidgets.QGroupBox("姿态库")
        pose_layout = QtWidgets.QGridLayout(pose_group)
        pose_layout.addWidget(QtWidgets.QLabel("保存名称"), 0, 0)
        pose_layout.addWidget(self.pose_name_edit, 0, 1, 1, 3)
        pose_layout.addWidget(QtWidgets.QLabel("已保存姿态"), 1, 0)
        pose_layout.addWidget(self.pose_combo, 1, 1, 1, 3)
        pose_layout.addWidget(self.zero_button, 2, 0)
        pose_layout.addWidget(self.save_button, 2, 1)
        pose_layout.addWidget(self.load_button, 2, 2)
        pose_layout.addWidget(self.refresh_button, 2, 3)
        pose_layout.addWidget(QtWidgets.QLabel(f"目录：{self.pose_dir}"), 3, 0, 1, 4)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(publisher_group)
        main_layout.addWidget(scroll_area, 1)
        main_layout.addWidget(pose_group)
        main_layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        self.zero_button.clicked.connect(self.zero_positions)
        self.save_button.clicked.connect(self.save_pose)
        self.load_button.clicked.connect(self.load_pose)
        self.refresh_button.clicked.connect(self.refresh_pose_list)
        self.pose_combo.currentTextChanged.connect(self.pose_name_edit.setText)

    def positions(self) -> list[int]:
        return [slider.value() for slider in self.joint_sliders]

    def set_positions(self, positions: Sequence[int]) -> None:
        for slider, value in zip(self.joint_sliders, positions):
            slider.set_value(int(value))

    def publish_state(self) -> None:
        payload = build_state_payload(self.sequence, self.positions())
        try:
            self.publisher.send_string(
                f"{STATE_TOPIC} {json.dumps(payload, separators=(',', ':'))}",
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            return
        self.sequence += 1
        if self.sequence % max(1, round(self.args.state_hz)) == 0:
            self._set_status(f"正在发布，sequence={self.sequence - 1}")

    def zero_positions(self) -> None:
        self.set_positions([0] * JOINT_COUNT)
        self._set_status("已全部归零")

    def refresh_pose_list(self, selected_name: str | None = None) -> None:
        selected_name = selected_name or self.pose_combo.currentText()
        names = list_pose_names(self.pose_dir)
        blocker = QtCore.QSignalBlocker(self.pose_combo)
        self.pose_combo.clear()
        self.pose_combo.addItems(names)
        if selected_name in names:
            self.pose_combo.setCurrentText(selected_name)
        del blocker
        self.load_button.setEnabled(bool(names))
        self._set_status(f"姿态库中有 {len(names)} 个文件")

    def save_pose(self) -> None:
        try:
            name = validate_pose_name(self.pose_name_edit.text())
            destination = pose_path(self.pose_dir, name)
        except ValueError as exc:
            self._show_error("无法保存姿态", exc)
            return

        if destination.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "覆盖姿态",
                f"姿态“{name}”已经存在，是否覆盖？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                self._set_status("已取消覆盖")
                return
        try:
            save_pose_file(
                self.pose_dir,
                name,
                self.positions(),
                self.args.angle_min,
                self.args.angle_max,
            )
        except (OSError, ValueError) as exc:
            self._show_error("无法保存姿态", exc)
            return
        self.refresh_pose_list(name)
        self.pose_name_edit.setText(name)
        self._set_status(f"已保存姿态：{name}")

    def load_pose(self) -> None:
        name = self.pose_combo.currentText()
        if not name:
            self._show_error("无法读取姿态", ValueError("没有可读取的姿态"))
            return
        try:
            positions = load_pose_file(
                pose_path(self.pose_dir, name),
                self.args.angle_min,
                self.args.angle_max,
            )
        except (OSError, ValueError) as exc:
            self._show_error("无法读取姿态", exc)
            return
        self.set_positions(positions)
        self.pose_name_edit.setText(name)
        self.publish_state()
        self._set_status(f"已读取并发布姿态：{name}")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _show_error(self, title: str, error: Exception) -> None:
        self._set_status(f"{title}：{error}")
        QtWidgets.QMessageBox.critical(self, title, str(error))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.publish_timer.stop()
        self.publisher.close()
        event.accept()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qt fake EX16 ZMQ state publisher.")
    parser.add_argument("--state-endpoint", default=DEFAULT_STATE_ENDPOINT)
    parser.add_argument("--state-hz", type=float, default=100.0)
    parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    parser.add_argument("--angle-min", type=int, default=-180)
    parser.add_argument("--angle-max", type=int, default=180)
    args = parser.parse_args(argv)
    if not math.isfinite(args.state_hz) or args.state_hz <= 0:
        parser.error("--state-hz must be a finite number greater than zero")
    if args.angle_min >= args.angle_max:
        parser.error("--angle-min must be less than --angle-max")
    if not args.angle_min <= 0 <= args.angle_max:
        parser.error("the angle range must include zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        window = FakeEX16Window(args)
    except (OSError, zmq.ZMQError) as exc:
        message = f"Failed to start fake EX16 publisher: {exc}"
        print(message, file=sys.stderr)
        QtWidgets.QMessageBox.critical(None, "Fake EX16", message)
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
