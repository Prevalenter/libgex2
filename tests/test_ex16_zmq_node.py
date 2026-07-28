import json
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets  # noqa: E402

from nodes import ex16_zmq_node  # noqa: E402


class FakeGlove16:
    def __init__(self, port=None, serial_number=None, left=False):
        self.port = port or "FAKE_EX16"
        self.is_connected = False
        self.positions = [index + 0.49 for index in range(16)]

    def connect(self):
        self.is_connected = True

    def getjs(self):
        return self.positions

    def off(self):
        self.is_connected = False


class EX16StateWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_save_button_saves_latest_measured_pose(self):
        endpoint = f"inproc://real-ex16-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory() as directory:
            args = ex16_zmq_node.parse_args(
                [
                    "--port",
                    "FAKE_EX16",
                    "--state-endpoint",
                    endpoint,
                    "--state-hz",
                    "100",
                    "--pose-dir",
                    directory,
                ]
            )
            with mock.patch.object(ex16_zmq_node, "Glove16", FakeGlove16):
                window = ex16_zmq_node.EX16StateWindow(args)
                try:
                    deadline = time.monotonic() + 1.0
                    while window.latest_positions is None and time.monotonic() < deadline:
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.005)
                    self.assertIsNotNone(window.latest_positions)
                    self.assertTrue(window.save_button.isEnabled())
                    self.assertEqual(window.value_labels[0].text(), "0.49°")
                    self.assertEqual(window.value_labels[15].text(), "15.49°")
                    deadline = time.monotonic() + 1.0
                    while window.publish_frequency_label.text() == "--" and time.monotonic() < deadline:
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.005)
                    self.assertIn("Hz", window.publish_frequency_label.text())

                    window.pose_name_edit.setText("measured_pose")
                    window.save_current_pose()
                    saved_path = Path(directory, "measured_pose.json")
                    payload = json.loads(saved_path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["urdf_deg"], list(range(16)))
                finally:
                    window.request_worker_stop()
                    self.assertTrue(window.worker_thread.wait(2000))
                    window.close()


class ControlStatusTest(unittest.TestCase):
    def test_frequency_estimate_uses_ewma(self):
        hz, last_time = ex16_zmq_node.update_frequency_estimate(None, None, 10.0)
        self.assertIsNone(hz)
        self.assertEqual(last_time, 10.0)
        hz, last_time = ex16_zmq_node.update_frequency_estimate(last_time, hz, 10.1)
        self.assertAlmostEqual(hz, 10.0)
        hz, last_time = ex16_zmq_node.update_frequency_estimate(
            last_time, hz, 10.3, alpha=0.5
        )
        self.assertAlmostEqual(hz, 7.5)
        self.assertEqual(last_time, 10.3)

    def test_decodes_geort_control_status_message(self):
        message = (
            f"{ex16_zmq_node.CONTROL_STATUS_TOPIC} "
            '{"sequence":7,"control_hz":9.8,'
            '"gx16_command_hz":9.6,"gx16_command_ms":103.0}'
        )
        payload = ex16_zmq_node.decode_control_status_message(message)
        self.assertEqual(payload["sequence"], 7)
        self.assertEqual(payload["control_hz"], 9.8)
        self.assertEqual(payload["gx16_command_hz"], 9.6)
        self.assertEqual(payload["gx16_command_ms"], 103.0)


if __name__ == "__main__":
    unittest.main()
