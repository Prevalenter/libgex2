import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo_ex16_recording_viser import load_recording, qpos_for_urdf, resolve_recording
from demo_ex16_viser import load_urdf


class RecordingLoadTest(unittest.TestCase):
    def test_loads_valid_non_pickled_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(
                path,
                qpos_deg=np.zeros((20, 16)),
                metadata_json=np.asarray(json.dumps({"fps": 10.0})),
            )
            qpos, metadata = load_recording(path)
            self.assertEqual(qpos.shape, (20, 16))
            self.assertEqual(metadata["fps"], 10.0)

    def test_rejects_invalid_joint_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, qpos_deg=np.zeros((20, 15)))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_recording(path)

    def test_direct_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, qpos_deg=np.zeros((1, 16)))
            self.assertEqual(resolve_recording(path), path.resolve())


class UrdfConfigurationTest(unittest.TestCase):
    def test_joint_order_and_degree_conversion(self):
        urdf = load_urdf()
        degrees = np.arange(1.0, 17.0)
        actual = qpos_for_urdf(urdf, degrees)
        expected_by_name = {
            f"joint{index}": np.deg2rad(value)
            for index, value in enumerate(degrees, start=1)
        }
        expected = np.asarray(
            [expected_by_name[name] for name in urdf.actuated_joint_names]
        )
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
