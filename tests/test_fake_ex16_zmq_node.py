import json
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import zmq  # noqa: E402
from PyQt5 import QtTest, QtWidgets  # noqa: E402

from nodes.fake_ex16_zmq_node import (  # noqa: E402
    JOINT_COUNT,
    POSE_FORMAT,
    STATE_TOPIC,
    FakeEX16Window,
    list_pose_names,
    load_pose_file,
    parse_args,
    save_pose_file,
    validate_pose_name,
)


class PoseFileTest(unittest.TestCase):
    def test_pose_round_trip(self):
        positions = list(range(-8, 8))
        with tempfile.TemporaryDirectory() as directory:
            pose_dir = Path(directory)
            path = save_pose_file(pose_dir, "测试_pose-1", positions, -180, 180)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], POSE_FORMAT)
            self.assertEqual(payload["version"], 1)
            self.assertEqual(payload["urdf_deg"], positions)
            self.assertEqual(list_pose_names(pose_dir), ["测试_pose-1"])
            self.assertEqual(load_pose_file(path, -180, 180), positions)

    def test_invalid_pose_files_are_rejected(self):
        invalid_payloads = [
            {"format": POSE_FORMAT, "version": 1, "urdf_deg": [0] * 15},
            {"format": POSE_FORMAT, "version": 1, "urdf_deg": [0] * 15 + ["x"]},
            {"format": POSE_FORMAT, "version": 1, "urdf_deg": [0] * 15 + [181]},
            {"format": "other", "version": 1, "urdf_deg": [0] * 16},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for payload in invalid_payloads:
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_pose_file(path, -180, 180)

            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pose_file(path, -180, 180)

    def test_unsafe_pose_names_are_rejected(self):
        for name in ("", "../pose", "has space", "pose.json", "a/b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_pose_name(name)


class FakeEX16WindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_controls_pose_loading_and_zmq_payload(self):
        context = zmq.Context.instance()
        endpoint = f"inproc://fake-ex16-{uuid.uuid4().hex}"
        subscriber = context.socket(zmq.SUB)
        subscriber.linger = 0
        subscriber.setsockopt_string(zmq.SUBSCRIBE, STATE_TOPIC)
        subscriber.connect(endpoint)

        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--state-endpoint",
                    endpoint,
                    "--state-hz",
                    "100",
                    "--pose-dir",
                    directory,
                ]
            )
            window = FakeEX16Window(args)
            try:
                self.assertEqual(len(window.joint_sliders), JOINT_COUNT)
                self.assertEqual(window.positions(), [0] * JOINT_COUNT)

                expected = [0] * JOINT_COUNT
                expected[0], expected[7], expected[15] = -30, 45, 120
                window.set_positions(expected)
                window.pose_name_edit.setText("three_joints")
                window.save_pose()
                self.assertTrue(Path(directory, "three_joints.json").is_file())

                window.zero_positions()
                self.assertEqual(window.positions(), [0] * JOINT_COUNT)
                window.pose_combo.setCurrentText("three_joints")
                window.load_pose()
                self.assertEqual(window.positions(), expected)

                window.set_positions([1] * JOINT_COUNT)
                window.pose_name_edit.setText("three_joints")
                with mock.patch.object(
                    QtWidgets.QMessageBox,
                    "question",
                    return_value=QtWidgets.QMessageBox.No,
                ):
                    window.save_pose()
                self.assertEqual(
                    load_pose_file(Path(directory, "three_joints.json"), -180, 180),
                    expected,
                )

                Path(directory, "broken.json").write_text("{", encoding="utf-8")
                window.refresh_pose_list("broken")
                window.set_positions([7] * JOINT_COUNT)
                with mock.patch.object(QtWidgets.QMessageBox, "critical"):
                    window.load_pose()
                self.assertEqual(window.positions(), [7] * JOINT_COUNT)

                window.pose_combo.setCurrentText("three_joints")
                window.load_pose()

                QtTest.QTest.qWait(100)
                window.publish_state()
                deadline = time.monotonic() + 1.0
                payload = None
                topic = None
                observed_sequences = []
                while time.monotonic() < deadline:
                    QtWidgets.QApplication.processEvents()
                    if not subscriber.poll(20, zmq.POLLIN):
                        continue
                    topic, encoded = subscriber.recv_string().split(" ", 1)
                    candidate = json.loads(encoded)
                    observed_sequences.append(candidate["sequence"])
                    if candidate["urdf_deg"] == [float(value) for value in expected]:
                        payload = candidate
                        break
                self.assertIsNotNone(payload, "did not receive the loaded pose")
                self.assertEqual(topic, STATE_TOPIC)
                self.assertEqual(payload["name"], "ex16")
                self.assertEqual(len(payload["urdf_deg"]), JOINT_COUNT)
                self.assertEqual(payload["urdf_deg"], [float(value) for value in expected])
                self.assertIsInstance(payload["sequence"], int)
                self.assertIsInstance(payload["timestamp"], float)
                self.assertEqual(observed_sequences, sorted(set(observed_sequences)))
            finally:
                window.close()
                subscriber.close()


if __name__ == "__main__":
    unittest.main()
