import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "utils" / "GeoRT" / "calibration" / "set_robot_hand"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from set_robot_hand import (  # noqa: E402
    DEFAULT_EX16_POSE_DIR,
    DEFAULT_GX16_POSE_DIR,
    DEFAULT_LAYOUT_FILE,
    EX16_POSE_FORMAT,
    FINGER_SPECS,
    GX16Kinematics,
    GX16_POSE_FORMAT,
    JOINT_NAMES,
    list_pose_names,
    default_layout,
    load_layout,
    load_ex16_pose,
    load_gx16_pose,
    matrix_to_wxyz,
    parse_args,
    save_gx16_pose,
    save_layout,
)


def fingertip_targets(kinematics):
    targets = {}
    for finger in FINGER_SPECS:
        transform = kinematics.tip_transform(finger)
        targets[finger] = {
            "position": transform[:3, 3],
            "wxyz": matrix_to_wxyz(transform[:3, :3]),
        }
    return targets


class PoseIOTest(unittest.TestCase):
    def test_defaults_use_device_subdirectories(self):
        args = parse_args([])
        self.assertEqual(args.ex16_pose_dir, DEFAULT_EX16_POSE_DIR)
        self.assertEqual(args.gx16_pose_dir, DEFAULT_GX16_POSE_DIR)
        self.assertEqual(DEFAULT_EX16_POSE_DIR.parts[-2:], ("pose", "ex16"))
        self.assertEqual(DEFAULT_GX16_POSE_DIR.parts[-2:], ("pose", "gx16"))

    def test_layout_round_trip_contains_bases_and_camera(self):
        layout = default_layout()
        with tempfile.TemporaryDirectory() as directory:
            path = save_layout(Path(directory) / "layout.json", layout["base_transforms"], layout["camera"])
            loaded = load_layout(path)
            self.assertEqual(set(loaded["base_transforms"]), {"ex16", "gx16"})
            np.testing.assert_allclose(
                loaded["camera"]["position"], layout["camera"]["position"]
            )
        self.assertEqual(DEFAULT_LAYOUT_FILE.name, "ex16_gx16_calibration.json")

    def test_ex16_load_and_natural_pose_order(self):
        with tempfile.TemporaryDirectory() as directory:
            pose_dir = Path(directory)
            for name in ("10", "2", "1"):
                (pose_dir / f"{name}.json").write_text(
                    json.dumps(
                        {
                            "format": EX16_POSE_FORMAT,
                            "version": 1,
                            "name": name,
                            "urdf_deg": list(range(16)),
                        }
                    ),
                    encoding="utf-8",
                )
            self.assertEqual(list_pose_names(pose_dir), ("1", "2", "10"))
            np.testing.assert_array_equal(load_ex16_pose(pose_dir, "2"), range(16))

    def test_gx16_round_trip_and_overwrite_guard(self):
        kinematics = GX16Kinematics()
        qpos_deg = np.rad2deg(kinematics.qpos_rad)
        with tempfile.TemporaryDirectory() as directory:
            pose_dir = Path(directory)
            path = save_gx16_pose(
                pose_dir,
                "pinch_01",
                qpos_deg,
                "2",
                fingertip_targets(kinematics),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], GX16_POSE_FORMAT)
            self.assertEqual(payload["source_ex16_pose"], "2")
            loaded, _ = load_gx16_pose(pose_dir, "pinch_01")
            np.testing.assert_allclose(loaded, qpos_deg)
            with self.assertRaises(FileExistsError):
                save_gx16_pose(
                    pose_dir,
                    "pinch_01",
                    qpos_deg,
                    "2",
                    fingertip_targets(kinematics),
                )


class IKTest(unittest.TestCase):
    def test_reaches_known_thumb_pose_within_joint_limits(self):
        kinematics = GX16Kinematics()
        target_qpos = kinematics.qpos_rad.copy()
        target_qpos[:4] = (0.45, 0.9, 0.65, 0.35)
        kinematics.set_qpos(target_qpos)
        target = kinematics.tip_transform("thumb")
        kinematics.set_qpos(np.zeros(16))

        result = kinematics.solve_finger(
            "thumb",
            target[:3, 3],
            matrix_to_wxyz(target[:3, :3]),
            use_orientation=True,
        )
        self.assertLess(result.position_error_m, 1e-5)
        self.assertLess(result.orientation_error_rad, 1e-4)
        self.assertTrue(np.all(result.qpos_rad >= kinematics.lower_rad - 1e-9))
        self.assertTrue(np.all(result.qpos_rad <= kinematics.upper_rad + 1e-9))
        self.assertEqual(len(JOINT_NAMES), 16)


if __name__ == "__main__":
    unittest.main()
