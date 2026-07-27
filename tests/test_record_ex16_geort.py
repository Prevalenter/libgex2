import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodes.record_ex16_geort import (
    DEFAULT_URDF_PATH,
    EX16HumanProjector,
    TIP_IDS,
    atomic_save_npy,
    atomic_save_npz,
    decode_state_message,
    fit_origin_similarity,
    parse_args,
    select_open_reference_frame,
)


class SimilarityFitTest(unittest.TestCase):
    def test_recovers_origin_fixed_scale_and_rotation(self):
        source = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.0, 0.0, 3.0),
                (1.0, 1.0, 1.0),
            )
        )
        rotation = np.asarray(
            ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        target = source @ rotation * 0.4

        scale, actual_rotation = fit_origin_similarity(source, target)

        self.assertAlmostEqual(scale, 0.4)
        np.testing.assert_allclose(actual_rotation, rotation, atol=1e-12)

    def test_open_reference_uses_four_tracked_tips(self):
        reference = np.zeros((3, 21, 3))
        reference[0, TIP_IDS, 0] = 1.0
        reference[1, TIP_IDS, 0] = 2.0
        reference[2, TIP_IDS, 0] = 1.5
        self.assertEqual(select_open_reference_frame(reference), 1)


class StateMessageTest(unittest.TestCase):
    def test_default_recording_rate_and_length(self):
        args = parse_args([])
        self.assertEqual(args.fps, 10.0)
        self.assertEqual(args.frames, 3498)

    def test_valid_message(self):
        payload = {"sequence": 7, "timestamp": 12.5, "urdf_deg": list(range(16))}
        result = decode_state_message("ex16/state " + json.dumps(payload))
        self.assertEqual(result["sequence"], 7)
        self.assertEqual(result["qpos_deg"].shape, (16,))

    def test_rejects_wrong_shape_and_topic(self):
        payload = {"timestamp": 12.5, "urdf_deg": [0] * 15}
        with self.assertRaises(ValueError):
            decode_state_message("ex16/state " + json.dumps(payload))
        with self.assertRaises(ValueError):
            decode_state_message("other " + json.dumps(payload))


class ProjectorTest(unittest.TestCase):
    def test_real_urdf_projects_finite_21_point_frame(self):
        projector = EX16HumanProjector(DEFAULT_URDF_PATH)
        source_open = projector.source_keypoints(np.zeros(16))
        reference = np.repeat(source_open[None], 2, axis=0)
        reference[:, 17:21] = source_open[13:17]
        projector.calibrate(np.zeros((3, 16)), reference)

        projected = projector.project(np.zeros(16))

        self.assertEqual(projected.shape, (21, 3))
        self.assertEqual(projected.dtype, np.float32)
        self.assertTrue(np.isfinite(projected).all())
        np.testing.assert_array_equal(projected[0], np.zeros(3))
        np.testing.assert_allclose(projected[TIP_IDS], source_open[TIP_IDS], atol=1e-6)

    def test_saved_calibration_can_be_restored_exactly(self):
        first = EX16HumanProjector(DEFAULT_URDF_PATH)
        source_open = first.source_keypoints(np.zeros(16))
        reference = np.repeat(source_open[None], 2, axis=0)
        reference[:, 17:21] = source_open[13:17]
        first.calibrate(np.zeros((3, 16)), reference)

        restored = EX16HumanProjector(DEFAULT_URDF_PATH)
        restored.restore_calibration(
            first.calibration_qpos_deg,
            first.scale,
            first.rotation,
            reference,
            first.reference_frame_index,
        )

        pose = np.arange(16, dtype=np.float64)
        np.testing.assert_allclose(restored.project(pose), first.project(pose))

    def test_atomic_outputs_round_trip_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npy_path = root / "hand.npy"
            npz_path = root / "raw.npz"
            hand = np.zeros((4, 21, 3), dtype=np.float32)
            atomic_save_npy(npy_path, hand)
            atomic_save_npz(npz_path, qpos_deg=np.zeros((4, 16)))

            np.testing.assert_array_equal(np.load(npy_path, allow_pickle=False), hand)
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertEqual(archive["qpos_deg"].shape, (4, 16))


if __name__ == "__main__":
    unittest.main()
