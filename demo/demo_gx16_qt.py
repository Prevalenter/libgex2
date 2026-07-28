import argparse
import errno
import os
import platform
import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
from libgex2 import Hand16
from libgex2.libgex.gx16.libgx16 import JOINT_MOTOR_DIRECTIONS
from serial.tools import list_ports


JOINT_COUNT = 16
ANGLE_MIN = -90
ANGLE_MAX = 90
DEFAULT_SERIAL_NUMBER = "FTAKRP3A"
FTDI_VENDOR_ID = 0x0403
JOINT_DIRECTIONS = [int(direction) for direction in JOINT_MOTOR_DIRECTIONS]

if len(JOINT_DIRECTIONS) != JOINT_COUNT:
    raise ValueError(
        f"JOINT_MOTOR_DIRECTIONS must contain {JOINT_COUNT} values, "
        f"got {len(JOINT_DIRECTIONS)}"
    )

if any(direction not in (-1, 1) for direction in JOINT_DIRECTIONS):
    raise ValueError("JOINT_MOTOR_DIRECTIONS values must be 1 or -1")


def host_system():
    name = platform.system() or sys.platform
    return name


def port_placeholder(system=None):
    system = system or host_system()
    if system == "Windows":
        return "COM6"
    if system == "Darwin":
        return "/dev/cu.usbserial-*"
    return "/dev/ttyUSB0 or /dev/ttyACM0"


def serial_devices():
    """Return serial metadata without collapsing missing/duplicate serial IDs."""
    devices = []
    for info in list_ports.comports():
        devices.append(
            {
                "device": str(info.device),
                "serial_number": info.serial_number,
                "description": info.description or "",
                "manufacturer": info.manufacturer or "",
                "vid": info.vid,
                "pid": info.pid,
            }
        )
    return devices


def _platform_candidates(devices, system):
    if system == "Windows":
        return [item for item in devices if item["device"].upper().startswith("COM")]
    if system == "Darwin":
        return [
            item
            for item in devices
            if item["device"].startswith(("/dev/cu.usb", "/dev/tty.usb"))
        ]
    return [
        item
        for item in devices
        if item["device"].startswith(("/dev/ttyUSB", "/dev/ttyACM"))
    ]


def choose_serial_device(devices, requested_serial=None, system=None):
    """Choose an unambiguous GX16 adapter and explain the selection."""
    system = system or host_system()
    if requested_serial:
        exact = [
            item for item in devices if item["serial_number"] == requested_serial
        ]
        if len(exact) == 1:
            return exact[0], f"matched USB serial {requested_serial}"

    candidates = _platform_candidates(devices, system)
    ftdi = [item for item in candidates if item["vid"] == FTDI_VENDOR_ID]
    if len(ftdi) == 1:
        item = ftdi[0]
        serial = item["serial_number"] or "unknown"
        return item, f"unique FTDI adapter (USB serial {serial})"
    if len(candidates) == 1:
        item = candidates[0]
        serial = item["serial_number"] or "unknown"
        return item, f"only compatible serial adapter (USB serial {serial})"
    return None, "no unique compatible adapter"


def device_summary(device):
    serial = device["serial_number"] or "no serial"
    description = device["description"] or device["manufacturer"] or "serial device"
    return f"{device['device']} — {description} — {serial}"


def linux_port_problem(port):
    if host_system() != "Linux" or not port.startswith("/dev/"):
        return None
    path = Path(port)
    if not path.exists():
        return f"Serial device does not exist: {port}"
    if not os.access(port, os.R_OK | os.W_OK):
        return (
            f"No read/write permission for {port}. Add the user to dialout with:\n"
            "sudo usermod -aG dialout $USER\n"
            "Then log out and log back in."
        )
    return None


def connection_error_message(error, port, devices):
    details = str(error).strip() or repr(error)
    error_number = getattr(error, "errno", None)
    if error_number == errno.EBUSY or "busy" in details.lower():
        details += f"\n{port} is busy. Check it with: fuser -v {port}"
    permission_problem = linux_port_problem(port)
    if permission_problem:
        details += f"\n{permission_problem}"
    if devices:
        details += "\nDetected ports:\n" + "\n".join(
            f"  {device_summary(item)}" for item in devices
        )
    else:
        details += (
            "\nNo serial adapters were detected. Check USB power/cable and run:\n"
            "python -m serial.tools.list_ports -v"
        )
    return details


def qt_environment_problem():
    if host_system() != "Linux":
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return None
    return (
        "No Linux graphical session was detected (DISPLAY and WAYLAND_DISPLAY "
        "are empty). Run this demo from the Ubuntu desktop session, or verify "
        "SSH X11 forwarding with: echo $DISPLAY"
    )


def urdf_to_motor_angle(joint_index, angle):
    return angle * JOINT_DIRECTIONS[joint_index - 1]


def motor_to_urdf_angles(angles):
    return [
        angle * direction
        for angle, direction in zip(angles[:JOINT_COUNT], JOINT_DIRECTIONS)
    ]


def import_qt():
    candidates = (
        ("PyQt5", "PyQt5"),
        ("PySide6", "PySide6"),
        ("PyQt6", "PyQt6"),
        ("PySide2", "PySide2"),
    )
    for module_name, api_name in candidates:
        try:
            module = __import__(module_name, fromlist=["QtCore", "QtWidgets"])
            return module.QtCore, module.QtWidgets, api_name
        except ImportError:
            continue
    raise ImportError(
        "No Qt binding found. Install one first, for example: pip install PyQt5"
    )


try:
    QtCore, QtWidgets, QT_API = import_qt()
except ImportError as exc:
    print(exc)
    raise SystemExit(1)

try:
    QT_HORIZONTAL = QtCore.Qt.Horizontal
except AttributeError:
    QT_HORIZONTAL = QtCore.Qt.Orientation.Horizontal

try:
    QT_WAIT_CURSOR = QtCore.Qt.WaitCursor
except AttributeError:
    QT_WAIT_CURSOR = QtCore.Qt.CursorShape.WaitCursor


class JointSlider(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(int, int) if QT_API.startswith("PyQt") else QtCore.Signal(int, int)
    released = QtCore.pyqtSignal() if QT_API.startswith("PyQt") else QtCore.Signal()

    def __init__(self, joint_index, parent=None):
        super().__init__(parent)
        self.joint_index = joint_index

        self.name_label = QtWidgets.QLabel(f"J{joint_index:02d}")
        self.name_label.setMinimumWidth(36)

        self.slider = QtWidgets.QSlider(QT_HORIZONTAL)
        self.slider.setRange(ANGLE_MIN, ANGLE_MAX)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setMinimumWidth(300)
        try:
            self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        except AttributeError:
            self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(30)

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setRange(ANGLE_MIN, ANGLE_MAX)
        self.spinbox.setValue(0)
        self.spinbox.setSuffix(" deg")
        self.spinbox.setMinimumWidth(92)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spinbox)

        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self.slider.sliderReleased.connect(self.released)
        self.spinbox.valueChanged.connect(self._on_spinbox_value_changed)
        self.spinbox.editingFinished.connect(self.released)

    def set_value(self, value):
        value = max(ANGLE_MIN, min(ANGLE_MAX, int(round(value))))
        slider_blocker = QtCore.QSignalBlocker(self.slider)
        spinbox_blocker = QtCore.QSignalBlocker(self.spinbox)
        self.slider.setValue(value)
        self.spinbox.setValue(value)
        del slider_blocker
        del spinbox_blocker

    def value(self):
        return self.slider.value()

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)

    def _on_slider_value_changed(self, value):
        if self.spinbox.value() != value:
            blocker = QtCore.QSignalBlocker(self.spinbox)
            self.spinbox.setValue(value)
            del blocker
        self.value_changed.emit(self.joint_index, value)

    def _on_spinbox_value_changed(self, value):
        if self.slider.value() != value:
            blocker = QtCore.QSignalBlocker(self.slider)
            self.slider.setValue(value)
            del blocker
        self.value_changed.emit(self.joint_index, value)


class GX16ControlWindow(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hand = None
        self.pending_commands = {}
        self.is_connected = False
        self.is_updating_controls = False
        self.detected_devices = []
        self.selection_reason = ""

        self.command_timer = QtCore.QTimer(self)
        self.command_timer.setInterval(80)
        self.command_timer.timeout.connect(self.flush_pending_commands)

        self.setWindowTitle(f"GX16 Joint Control ({QT_API})")
        self.resize(720, 760)

        self.system_label = QtWidgets.QLabel(
            f"{host_system()} · Qt {QT_API}"
        )
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.lineEdit().setPlaceholderText(port_placeholder())
        if args.port:
            self.port_combo.addItem(args.port)
            self.port_combo.setCurrentText(args.port)
        self.serial_edit = QtWidgets.QLineEdit(args.serial_number or "")
        self.serial_edit.setPlaceholderText(DEFAULT_SERIAL_NUMBER)
        self.refresh_button = QtWidgets.QPushButton("Refresh")

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.home_button = QtWidgets.QPushButton("Home")
        self.read_button = QtWidgets.QPushButton("Read")
        self.torque_on_button = QtWidgets.QPushButton("Torque On")
        self.torque_off_button = QtWidgets.QPushButton("Torque Off")

        self.status_label = QtWidgets.QLabel("Disconnected")

        self.joint_sliders = [JointSlider(i + 1) for i in range(JOINT_COUNT)]
        for joint_slider in self.joint_sliders:
            joint_slider.setEnabled(False)
            joint_slider.value_changed.connect(self.queue_joint_command)
            joint_slider.released.connect(self.flush_pending_commands)

        self._build_layout()
        self._connect_signals()
        self._set_connected_state(False)
        self.refresh_serial_ports()

    def _build_layout(self):
        device_group = QtWidgets.QGroupBox("Device")
        device_layout = QtWidgets.QGridLayout(device_group)
        device_layout.addWidget(self.system_label, 0, 0, 1, 5)
        device_layout.addWidget(QtWidgets.QLabel("Port"), 1, 0)
        device_layout.addWidget(self.port_combo, 1, 1)
        device_layout.addWidget(self.refresh_button, 1, 2)
        device_layout.addWidget(QtWidgets.QLabel("USB Serial"), 2, 0)
        device_layout.addWidget(self.serial_edit, 2, 1, 1, 2)
        device_layout.addWidget(self.connect_button, 1, 3, 2, 2)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.home_button)
        button_layout.addWidget(self.read_button)
        button_layout.addWidget(self.torque_on_button)
        button_layout.addWidget(self.torque_off_button)
        button_layout.addStretch(1)

        sliders_widget = QtWidgets.QWidget()
        sliders_layout = QtWidgets.QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(10, 10, 10, 10)
        for joint_slider in self.joint_sliders:
            sliders_layout.addWidget(joint_slider)
        sliders_layout.addStretch(1)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(sliders_widget)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(device_group)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(scroll_area, 1)
        main_layout.addWidget(self.status_label)

    def _connect_signals(self):
        self.connect_button.clicked.connect(self.connect_hand)
        self.refresh_button.clicked.connect(self.refresh_serial_ports)
        self.home_button.clicked.connect(self.home)
        self.read_button.clicked.connect(self.read_current_positions)
        self.torque_on_button.clicked.connect(self.torque_on)
        self.torque_off_button.clicked.connect(self.torque_off)

    def _set_connected_state(self, connected):
        self.is_connected = connected
        for joint_slider in self.joint_sliders:
            joint_slider.setEnabled(connected)
        self.home_button.setEnabled(connected)
        self.read_button.setEnabled(connected)
        self.torque_on_button.setEnabled(connected)
        self.torque_off_button.setEnabled(connected)
        self.connect_button.setEnabled(not connected)
        self.port_combo.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)
        self.serial_edit.setEnabled(not connected)

    def refresh_serial_ports(self):
        requested_port = self.port_combo.currentText().strip() or self.args.port
        try:
            devices = serial_devices()
        except Exception as exc:
            self.detected_devices = []
            self.status_label.setText(f"Serial scan failed: {exc}")
            return

        self.detected_devices = devices
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for device in devices:
            self.port_combo.addItem(device["device"], device)

        reason = ""
        selected_port = requested_port
        if not selected_port:
            selected, reason = choose_serial_device(
                devices, self.serial_edit.text().strip() or None
            )
            if selected is not None:
                selected_port = selected["device"]
                if selected["serial_number"]:
                    self.serial_edit.setText(selected["serial_number"])
        if selected_port:
            self.port_combo.setCurrentText(selected_port)
        self.port_combo.blockSignals(False)
        self.selection_reason = reason

        if selected_port:
            suffix = f" ({reason})" if reason else ""
            self.status_label.setText(f"Detected {selected_port}{suffix}")
        elif devices:
            self.status_label.setText(
                f"Detected {len(devices)} ports; select the GX16 adapter manually"
            )
        else:
            self.status_label.setText(
                "No serial adapters detected; check USB power and cable"
            )

    def connect_hand(self):
        port = self.port_combo.currentText().strip() or None
        serial_number = self.serial_edit.text().strip() or None

        if not port:
            self.refresh_serial_ports()
            port = self.port_combo.currentText().strip() or None
        if not port:
            message = connection_error_message(
                RuntimeError(
                    f"Could not uniquely select the GX16 adapter for serial "
                    f"{serial_number or '(not set)'}"
                ),
                port_placeholder(),
                self.detected_devices,
            )
            self.status_label.setText("Connect failed: no unique serial port")
            QtWidgets.QMessageBox.critical(self, "GX16", message)
            return

        permission_problem = linux_port_problem(port)
        if permission_problem:
            self.status_label.setText(f"Connect failed: no permission for {port}")
            QtWidgets.QMessageBox.critical(self, "GX16", permission_problem)
            return

        self.status_label.setText("Connecting...")
        QtWidgets.QApplication.setOverrideCursor(QT_WAIT_CURSOR)
        try:
            # Resolve to an explicit platform port before entering Hand16. This
            # avoids OS-dependent serial-number formatting differences.
            self.hand = Hand16(port=port, serial_number=None)
            self.hand.connect(
                curr_limit=self.args.curr_limit,
                goal_current=self.args.goal_current,
                goal_pwm=self.args.goal_pwm,
            )
        except (Exception, SystemExit) as exc:
            self.hand = None
            self._set_connected_state(False)
            details = connection_error_message(exc, port, self.detected_devices)
            self.status_label.setText(f"Connect failed: {details.splitlines()[0]}")
            QtWidgets.QMessageBox.critical(self, "GX16", f"Connect failed:\n{details}")
        else:
            self._set_connected_state(True)
            selected = next(
                (item for item in self.detected_devices if item["device"] == port),
                None,
            )
            suffix = f" · {selected['serial_number']}" if selected else ""
            self.status_label.setText(f"Connected: {self.hand.port}{suffix}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def queue_joint_command(self, joint_index, angle):
        if self.is_updating_controls:
            return
        self.pending_commands[joint_index] = angle
        if self.is_connected and not self.command_timer.isActive():
            self.command_timer.start()

    def flush_pending_commands(self):
        self.command_timer.stop()
        if not self.is_connected or self.hand is None or not self.pending_commands:
            self.pending_commands.clear()
            return

        commands = sorted(self.pending_commands.items())
        self.pending_commands.clear()
        try:
            for joint_index, angle in commands:
                self.hand.setj(joint_index, urdf_to_motor_angle(joint_index, angle))
        except Exception as exc:
            self.status_label.setText(f"Command failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "GX16", f"Command failed:\n{exc}")
        else:
            summary = ", ".join(f"J{i:02d}={a}" for i, a in commands[-4:])
            self.status_label.setText(f"Sent {len(commands)} command(s): {summary}")

    def home(self):
        if not self.is_connected or self.hand is None:
            return
        self.flush_pending_commands()
        self.status_label.setText("Moving home...")
        QtWidgets.QApplication.setOverrideCursor(QT_WAIT_CURSOR)
        try:
            self.hand.home()
        except Exception as exc:
            self.status_label.setText(f"Home failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "GX16", f"Home failed:\n{exc}")
        else:
            self._set_all_slider_values([0] * JOINT_COUNT)
            self.status_label.setText("Home done")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def read_current_positions(self):
        if not self.is_connected or self.hand is None:
            return
        self.flush_pending_commands()
        self.status_label.setText("Reading...")
        QtWidgets.QApplication.setOverrideCursor(QT_WAIT_CURSOR)
        try:
            positions = self.hand.getjs()
        except Exception as exc:
            self.status_label.setText(f"Read failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "GX16", f"Read failed:\n{exc}")
        else:
            self._set_all_slider_values(motor_to_urdf_angles(positions))
            self.status_label.setText("Read current joint positions")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def torque_on(self):
        if not self.is_connected or self.hand is None:
            return
        try:
            self.hand.on()
        except Exception as exc:
            self.status_label.setText(f"Torque on failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "GX16", f"Torque on failed:\n{exc}")
        else:
            self.status_label.setText("Torque on")

    def torque_off(self):
        if not self.is_connected or self.hand is None:
            return
        self.flush_pending_commands()
        try:
            self.hand.off()
        except Exception as exc:
            self.status_label.setText(f"Torque off failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "GX16", f"Torque off failed:\n{exc}")
        else:
            self.status_label.setText("Torque off")

    def _set_all_slider_values(self, values):
        self.is_updating_controls = True
        try:
            for joint_slider, value in zip(self.joint_sliders, values):
                joint_slider.set_value(value)
        finally:
            self.is_updating_controls = False

    def closeEvent(self, event):
        self.command_timer.stop()
        self.flush_pending_commands()
        if self.hand is not None:
            try:
                self.hand.off()
            except Exception:
                pass
            port_handler = getattr(self.hand, "portHandler", None)
            if port_handler is not None:
                try:
                    port_handler.closePort()
                except Exception:
                    pass
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description="Qt slider control for GX16 joints.")
    parser.add_argument("--port", help="Serial port, for example COM6.")
    parser.add_argument(
        "--serial-number",
        default=DEFAULT_SERIAL_NUMBER,
        help="USB serial number used when --port is not set.",
    )
    parser.add_argument("--curr-limit", type=int, default=1000)
    parser.add_argument("--goal-current", type=int, default=600)
    parser.add_argument("--goal-pwm", type=int, default=200)
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Print detected serial adapters and exit without opening Qt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_ports:
        devices = serial_devices()
        print(f"System: {host_system()}")
        if not devices:
            print("No serial adapters detected.")
            return 1
        selected, reason = choose_serial_device(devices, args.serial_number)
        for device in devices:
            marker = "*" if selected is device else " "
            print(f"{marker} {device_summary(device)}")
        if selected is not None:
            print(f"Auto-selected: {selected['device']} ({reason})")
        else:
            print("Auto-selection is ambiguous; pass --port explicitly.")
        return 0

    environment_problem = qt_environment_problem()
    if environment_problem:
        raise RuntimeError(environment_problem)
    # Do not pass application-specific options such as --port to Qt itself.
    app = QtWidgets.QApplication([sys.argv[0]])
    window = GX16ControlWindow(args)
    window.show()
    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, RuntimeError) as exc:
        print(exc)
        raise SystemExit(1)
