import os
import unittest
from unittest import mock

from demo import demo_gx16_qt as viewer
from libgex2.libgex import utils as serial_utils


def device(name, serial=None, vid=None):
    return {
        "device": name,
        "serial_number": serial,
        "description": "USB adapter",
        "manufacturer": "",
        "vid": vid,
        "pid": None,
    }


class PlatformPortSelectionTest(unittest.TestCase):
    def test_platform_specific_placeholders(self):
        self.assertEqual(viewer.port_placeholder("Windows"), "COM6")
        self.assertIn("ttyUSB", viewer.port_placeholder("Linux"))
        self.assertIn("cu.usbserial", viewer.port_placeholder("Darwin"))

    def test_exact_serial_wins(self):
        devices = [
            device("/dev/ttyUSB0", "other", viewer.FTDI_VENDOR_ID),
            device("/dev/ttyUSB1", "wanted", viewer.FTDI_VENDOR_ID),
        ]
        selected, reason = viewer.choose_serial_device(devices, "wanted", "Linux")
        self.assertEqual(selected["device"], "/dev/ttyUSB1")
        self.assertIn("matched", reason)

    def test_unique_ftdi_is_used_when_serial_format_differs(self):
        devices = [device("/dev/ttyUSB0", "FTAKRP3A", viewer.FTDI_VENDOR_ID)]
        selected, reason = viewer.choose_serial_device(
            devices, "FTAKRP3AA", "Linux"
        )
        self.assertEqual(selected["device"], "/dev/ttyUSB0")
        self.assertIn("unique FTDI", reason)

    def test_multiple_ftdi_devices_are_not_guessed(self):
        devices = [
            device("/dev/ttyUSB0", "a", viewer.FTDI_VENDOR_ID),
            device("/dev/ttyUSB1", "b", viewer.FTDI_VENDOR_ID),
        ]
        selected, _ = viewer.choose_serial_device(devices, "missing", "Linux")
        self.assertIsNone(selected)

    def test_linux_missing_device_has_clear_diagnostic(self):
        with mock.patch.object(viewer, "host_system", return_value="Linux"):
            problem = viewer.linux_port_problem("/dev/does-not-exist-gx16")
        self.assertIn("does not exist", problem)


class QtEnvironmentTest(unittest.TestCase):
    def test_headless_linux_is_reported(self):
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"}
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            viewer, "host_system", return_value="Linux"
        ):
            self.assertIn("graphical session", viewer.qt_environment_problem())

    def test_windows_does_not_require_linux_display_variables(self):
        with mock.patch.object(viewer, "host_system", return_value="Windows"):
            self.assertIsNone(viewer.qt_environment_problem())


class CoreSerialDiscoveryTest(unittest.TestCase):
    def test_no_ports_returns_empty_mapping(self):
        with mock.patch.object(
            serial_utils.serial.tools.list_ports, "comports", return_value=[]
        ):
            self.assertEqual(serial_utils.search_ports(), {})


if __name__ == "__main__":
    unittest.main()
