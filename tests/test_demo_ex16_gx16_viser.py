import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo import demo_ex16_gx16_viser as viewer


class BaseTransformStorageTest(unittest.TestCase):
    def test_save_load_round_trip_and_atomic_cleanup(self):
        transforms = viewer.default_base_transforms()
        transforms["ex16"]["position"] = (0.1, -0.2, 0.3)
        transforms["ex16"]["wxyz"] = viewer.rpy_degrees_to_wxyz((10, 20, 30))
        transforms["gx16"]["position"] = (-0.4, 0.5, -0.6)
        transforms["gx16"]["wxyz"] = viewer.rpy_degrees_to_wxyz((-20, 5, 80))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "layout.json"
            viewer.save_base_transforms(path, transforms)
            loaded = viewer.load_base_transforms(path)

            self.assertEqual(loaded["ex16"]["position"], (0.1, -0.2, 0.3))
            self.assertEqual(loaded["gx16"]["position"], (-0.4, 0.5, -0.6))
            np.testing.assert_allclose(
                loaded["ex16"]["wxyz"], transforms["ex16"]["wxyz"]
            )
            np.testing.assert_allclose(
                loaded["gx16"]["wxyz"], transforms["gx16"]["wxyz"]
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], viewer.BASE_TRANSFORM_FORMAT)
            self.assertEqual(payload["version"], viewer.BASE_TRANSFORM_VERSION)

    def test_quaternion_is_normalized_when_loading(self):
        payload = viewer.base_transform_document(viewer.default_base_transforms())
        payload["ex16"]["wxyz"] = [2.0, 0.0, 0.0, 0.0]
        loaded = viewer.validate_base_transforms(payload)
        self.assertEqual(loaded["ex16"]["wxyz"], (1.0, 0.0, 0.0, 0.0))

    def test_invalid_documents_are_rejected(self):
        valid = viewer.base_transform_document(viewer.default_base_transforms())
        invalid_documents = []

        wrong_format = copy.deepcopy(valid)
        wrong_format["format"] = "other"
        invalid_documents.append(wrong_format)

        wrong_version = copy.deepcopy(valid)
        wrong_version["version"] = 2
        invalid_documents.append(wrong_version)

        missing_model = copy.deepcopy(valid)
        del missing_model["gx16"]
        invalid_documents.append(missing_model)

        wrong_position_size = copy.deepcopy(valid)
        wrong_position_size["ex16"]["position"] = [0.0, 0.0]
        invalid_documents.append(wrong_position_size)

        non_numeric = copy.deepcopy(valid)
        non_numeric["gx16"]["position"] = [0.0, "x", 0.0]
        invalid_documents.append(non_numeric)

        non_finite = copy.deepcopy(valid)
        non_finite["gx16"]["wxyz"] = [float("nan"), 0.0, 0.0, 0.0]
        invalid_documents.append(non_finite)

        zero_quaternion = copy.deepcopy(valid)
        zero_quaternion["ex16"]["wxyz"] = [0.0, 0.0, 0.0, 0.0]
        invalid_documents.append(zero_quaternion)

        for payload in [None, [], *invalid_documents]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                viewer.validate_base_transforms(payload)

    def test_invalid_json_and_missing_file_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            with self.assertRaises(FileNotFoundError):
                viewer.load_base_transforms(path)
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                viewer.load_base_transforms(path)


class BaseTransformConversionTest(unittest.TestCase):
    def test_rpy_quaternion_round_trip(self):
        expected = np.asarray((25.0, -35.0, 120.0))
        quaternion = viewer.rpy_degrees_to_wxyz(expected)
        actual = viewer.wxyz_to_rpy_degrees(quaternion)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0)

    def test_default_layout_returns_independent_copies(self):
        first = viewer.default_base_transforms()
        second = viewer.default_base_transforms()
        first["ex16"]["position"] = (9.0, 9.0, 9.0)
        self.assertEqual(second["ex16"]["position"], (-0.23, 0.0, 0.0))


class UrdfLinkFrameTest(unittest.TestCase):
    def test_plam_link_poses_use_each_urdf_fixed_transform(self):
        cases = (
            (
                viewer.EX16_URDF_PATH,
                viewer.EX16_MESH_DIR,
                (-0.04, 0.0, -0.03),
                (0.0, 3.14, -1.57),
            ),
            (
                viewer.GX16_URDF_PATH,
                viewer.GX16_MESH_DIR,
                (0.0, 0.0, 0.0),
                (1.57, 0.0, 1.57),
            ),
        )
        for urdf_path, mesh_dir, expected_position, expected_rpy in cases:
            with self.subTest(urdf=urdf_path):
                urdf = viewer.load_urdf(urdf_path, mesh_dir)
                position, wxyz = viewer.link_frame_pose(urdf, "plam_link")

                np.testing.assert_allclose(position, expected_position, atol=1e-12)
                expected_wxyz = viewer.rpy_degrees_to_wxyz(
                    np.rad2deg(expected_rpy)
                )
                np.testing.assert_allclose(wxyz, expected_wxyz, atol=1e-12)

    def test_missing_link_is_rejected(self):
        urdf = viewer.load_urdf(viewer.GX16_URDF_PATH, viewer.GX16_MESH_DIR)
        with self.assertRaisesRegex(ValueError, "missing link"):
            viewer.link_frame_pose(urdf, "not_a_link")


if __name__ == "__main__":
    unittest.main()
