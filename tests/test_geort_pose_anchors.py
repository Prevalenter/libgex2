import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GEORT_ROOT = ROOT / "utils" / "GeoRT"
CALIBRATION_DIR = GEORT_ROOT / "calibration"
for path in (ROOT, GEORT_ROOT, CALIBRATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_pose_anchors import load_pose_pairs  # noqa: E402
from geort.anchor import load_anchor_dataset  # noqa: E402
from geort.utils.path import get_human_data  # noqa: E402


def write_pair(root: Path, name: str, source: str) -> None:
    (root / "ex16").mkdir(parents=True, exist_ok=True)
    (root / "gx16").mkdir(parents=True, exist_ok=True)
    (root / "ex16" / f"{name}.json").write_text(
        json.dumps(
            {
                "format": "libgex.ex16_pose",
                "version": 1,
                "name": name,
                "urdf_deg": [0] * 16,
            }
        ),
        encoding="utf-8",
    )
    (root / "gx16" / f"{name}.json").write_text(
        json.dumps(
            {
                "format": "libgex.gx16_pose",
                "version": 1,
                "name": name,
                "source_ex16_pose": source,
                "joint_names": [f"joint{index}" for index in range(1, 17)],
                "urdf_deg": [0] * 16,
                "urdf_rad": [0] * 16,
            }
        ),
        encoding="utf-8",
    )


class PosePairValidationTest(unittest.TestCase):
    def test_human_data_prefers_exact_npy_over_raw_npz_prefix(self):
        path = get_human_data("human_ex16_pinch_01")
        self.assertEqual(path.name, "human_ex16_pinch_01.npy")

    def test_loads_naturally_sorted_strict_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("10", "2", "1"):
                write_pair(root, name, name)
            pairs = load_pose_pairs(root)
            self.assertEqual([pair["name"] for pair in pairs], ["1", "2", "10"])

    def test_rejects_source_pose_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pair(root, "2", "3")
            with self.assertRaisesRegex(ValueError, "source_ex16_pose"):
                load_pose_pairs(root)

    def test_generated_dataset_is_training_ready(self):
        path = GEORT_ROOT / "data" / "pose" / "paired_anchors.npz"
        dataset = load_anchor_dataset(path)
        self.assertEqual(dataset["anchor_names"].tolist(), [str(i) for i in range(1, 9)])
        self.assertEqual(dataset["human_keypoints"].shape, (8, 21, 3))
        self.assertEqual(dataset["gx16_qpos_rad"].shape, (8, 16))
        self.assertTrue(np.isfinite(dataset["human_keypoints"]).all())
        self.assertTrue(
            dataset["metadata"]["calibration_path"].endswith(
                "human_ex16_pinch_01_ex16_raw.npz"
            )
        )


if __name__ == "__main__":
    unittest.main()
