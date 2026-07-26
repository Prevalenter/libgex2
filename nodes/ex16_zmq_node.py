"""Publish real EX16 joint positions over ZMQ and save pose snapshots."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import zmq
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyQt5 is required: python -m pip install PyQt5") from exc

from libgex import Glove16  # noqa: E402
from nodes.fake_ex16_zmq_node import (  # noqa: E402
    DEFAULT_POSE_DIR,
    pose_path,
    save_pose_file,
    validate_pose_name,
)


JOINT_COUNT = 16
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5567"
STATE_TOPIC = "ex16/state"
POSE_ANGLE_MIN = -180
POSE_ANGLE_MAX = 180


class EX16PublisherWorker(QtCore.QObject):
    """Read the glove and publish states without blocking the Qt UI."""

    positions_updated = QtCore.pyqtSignal(object)
    status_updated = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.stop_event = threading.Event()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        glove = None
        publisher = None
        try:
            glove = Glove16(
                port=self.args.port,
                serial_number=self.args.serial_number,
                left=self.args.left,
            )
            glove.connect()

            publisher = zmq.Context.instance().socket(zmq.PUB)
            publisher.linger = 0
            publisher.bind(self.args.state_endpoint)
            status = f"正在发布：{glove.port} → {self.args.state_endpoint}"
            self.status_updated.emit(status)
            print(
                f"EX16 state: {self.args.state_endpoint}, topic={STATE_TOPIC}, "
                f"device={glove.port}",
                flush=True,
            )

            period = 1.0 / self.args.state_hz
            sequence = 0
            while not self.stop_event.is_set():
                started = time.monotonic()
                positions = [float(value) for value in glove.getjs()]
                if len(positions) != JOINT_COUNT:
                    raise RuntimeError(
                        f"Glove16.getjs returned {len(positions)} joints; "
                        f"expected {JOINT_COUNT}"
                    )
                payload = {
                    "name": "ex16",
                    "sequence": sequence,
                    "timestamp": time.time(),
                    "urdf_deg": positions,
                }
                publisher.send_string(
                    f"{STATE_TOPIC} {json.dumps(payload, separators=(',', ':'))}"
                )
                self.positions_updated.emit(positions)
                sequence += 1
                self.stop_event.wait(
                    max(0.0, period - (time.monotonic() - started))
                )
        except Exception as exc:
            print(f"EX16 node failed: {exc}", file=sys.stderr, flush=True)
            self.failed.emit(str(exc))
        finally:
            if publisher is not None:
                publisher.close()
            if glove is not None and glove.is_connected:
                try:
                    glove.off()
                except Exception as exc:
                    self.failed.emit(f"关闭 EX16 失败：{exc}")
            print("Stopping EX16 joint-state node...", flush=True)
            self.stopped.emit()

    def request_stop(self) -> None:
        self.stop_event.set()


class EX16StateWindow(QtWidgets.QWidget):
    """Display the latest measured pose and save snapshots."""

    def __init__(self, args: argparse.Namespace, parent=None) -> None:
        super().__init__(parent)
        self.args = args
        self.pose_dir = Path(args.pose_dir).expanduser().resolve()
        self.pose_dir.mkdir(parents=True, exist_ok=True)
        self.latest_positions: list[float] | None = None
        self.close_after_worker_stops = False

        self.setWindowTitle("EX16 State Publisher")
        self.resize(560, 680)
        self.value_labels = [QtWidgets.QLabel("--") for _ in range(JOINT_COUNT)]
        self.pose_name_edit = QtWidgets.QLineEdit()
        self.pose_name_edit.setPlaceholderText(
            "留空则自动使用 ex16_YYYYMMDD_HHMMSS"
        )
        self.save_button = QtWidgets.QPushButton("保存当前位置")
        self.save_button.setEnabled(False)
        self.status_label = QtWidgets.QLabel("正在连接 EX16…")
        self._build_layout()
        self.save_button.clicked.connect(self.save_current_pose)

        self.worker_thread = QtCore.QThread(self)
        self.worker = EX16PublisherWorker(args)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.positions_updated.connect(self.update_positions)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.failed.connect(self.handle_worker_error)
        self.worker.stopped.connect(
            self.worker_thread.quit, QtCore.Qt.DirectConnection
        )
        self.worker_thread.finished.connect(self.handle_worker_stopped)
        self.worker_thread.start()

    def _build_layout(self) -> None:
        state_group = QtWidgets.QGroupBox("当前位置（URDF degree）")
        state_layout = QtWidgets.QGridLayout(state_group)
        for index, value_label in enumerate(self.value_labels):
            row = index % 8
            column = (index // 8) * 2
            joint_label = QtWidgets.QLabel(f"J{index + 1:02d}")
            joint_label.setMinimumWidth(36)
            value_label.setMinimumWidth(90)
            value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            state_layout.addWidget(joint_label, row, column)
            state_layout.addWidget(value_label, row, column + 1)

        save_group = QtWidgets.QGroupBox("保存姿态")
        save_layout = QtWidgets.QGridLayout(save_group)
        save_layout.addWidget(QtWidgets.QLabel("姿态名称"), 0, 0)
        save_layout.addWidget(self.pose_name_edit, 0, 1)
        save_layout.addWidget(self.save_button, 1, 0, 1, 2)
        save_layout.addWidget(QtWidgets.QLabel(f"目录：{self.pose_dir}"), 2, 0, 1, 2)

        info_group = QtWidgets.QGroupBox("发布设置")
        info_layout = QtWidgets.QFormLayout(info_group)
        info_layout.addRow("Endpoint", QtWidgets.QLabel(self.args.state_endpoint))
        info_layout.addRow("Topic", QtWidgets.QLabel(STATE_TOPIC))
        info_layout.addRow("频率", QtWidgets.QLabel(f"{self.args.state_hz:g} Hz"))

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(info_group)
        main_layout.addWidget(state_group)
        main_layout.addWidget(save_group)
        main_layout.addStretch(1)
        main_layout.addWidget(self.status_label)

    @QtCore.pyqtSlot(object)
    def update_positions(self, positions: Sequence[float]) -> None:
        self.latest_positions = [float(value) for value in positions]
        for label, value in zip(self.value_labels, self.latest_positions):
            label.setText(f"{value:.2f}°")
        self.save_button.setEnabled(True)

    def save_current_pose(self) -> None:
        if self.latest_positions is None:
            return
        entered_name = self.pose_name_edit.text().strip()
        name = entered_name or datetime.now().strftime("ex16_%Y%m%d_%H%M%S")
        try:
            name = validate_pose_name(name)
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
                self.status_label.setText("已取消覆盖")
                return

        positions = [int(round(value)) for value in self.latest_positions]
        try:
            save_pose_file(
                self.pose_dir,
                name,
                positions,
                POSE_ANGLE_MIN,
                POSE_ANGLE_MAX,
            )
        except (OSError, ValueError) as exc:
            self._show_error("无法保存姿态", exc)
            return
        self.pose_name_edit.setText(name)
        self.status_label.setText(f"已保存当前位置：{destination}")

    def _show_error(self, title: str, error: Exception) -> None:
        self.status_label.setText(f"{title}：{error}")
        QtWidgets.QMessageBox.critical(self, title, str(error))

    @QtCore.pyqtSlot(str)
    def handle_worker_error(self, message: str) -> None:
        self.save_button.setEnabled(self.latest_positions is not None)
        self._show_error("EX16 节点错误", RuntimeError(message))

    @QtCore.pyqtSlot()
    def handle_worker_stopped(self) -> None:
        if self.close_after_worker_stops:
            self.close()

    def request_worker_stop(self) -> None:
        if self.worker_thread.isRunning():
            self.worker.request_stop()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self.worker_thread.isRunning():
            self.close_after_worker_stops = True
            self.worker.request_stop()
            self.status_label.setText("正在停止 EX16 节点…")
            event.ignore()
            return
        event.accept()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EX16 ZMQ joint-state publisher.")
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--port", help="Serial port, for example COM7")
    connection.add_argument("--serial-number", help="USB serial number")
    parser.add_argument("--left", action="store_true", help="Use left-hand directions")
    parser.add_argument("--state-endpoint", default=DEFAULT_STATE_ENDPOINT)
    parser.add_argument("--state-hz", type=float, default=100.0)
    parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    args = parser.parse_args(argv)
    if not math.isfinite(args.state_hz) or args.state_hz <= 0:
        parser.error("--state-hz must be a finite number greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    window = EX16StateWindow(args)
    app.aboutToQuit.connect(window.request_worker_stop)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal_timer = QtCore.QTimer()
    signal_timer.start(200)
    signal_timer.timeout.connect(lambda: None)
    window.show()
    exit_code = app.exec_()
    window.request_worker_stop()
    window.worker_thread.wait()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
