import unittest

import numpy as np

from demo.demo_ex16_geort_gx16_viser import (
    DEFAULT_CALIBRATION,
    control_status_payload,
    finite_or_none,
    gx16_configuration,
    parse_args,
    restore_projector,
    smooth_gx16_command,
    update_control_frequency,
)
from demo.demo_ex16_gx16_viser import GX16_MESH_DIR, GX16_URDF_PATH, load_urdf


class SavedCalibrationTest(unittest.TestCase):
    def test_restored_projector_reproduces_recorded_human_frame(self):
        projector, _ = restore_projector(DEFAULT_CALIBRATION)
        with np.load(DEFAULT_CALIBRATION, allow_pickle=False) as raw:
            qpos = raw["qpos_deg"][0]
        expected = np.load(
            DEFAULT_CALIBRATION.parent / "human_ex16.npy", allow_pickle=False
        )[0]
        np.testing.assert_allclose(projector.project(qpos), expected, atol=1e-7)


class GX16ConfigurationTest(unittest.TestCase):
    def test_configures_all_joints_in_urdf_order(self):
        urdf = load_urdf(GX16_URDF_PATH, GX16_MESH_DIR)
        qpos = np.arange(16, dtype=np.float64) / 10.0
        actual = gx16_configuration(urdf, qpos)
        by_name = dict(zip((f"joint{i}" for i in range(1, 17)), qpos))
        expected = np.asarray([by_name[name] for name in urdf.actuated_joint_names])
        np.testing.assert_allclose(actual, expected)


class GX16HardwareSafetyTest(unittest.TestCase):
    def test_command_is_clamped_and_ema_smoothed_without_step_limit(self):
        lower = np.zeros(16)
        upper = np.ones(16)
        previous = np.full(16, 0.5)
        desired = np.asarray([-2.0] + [2.0] * 15)
        actual = smooth_gx16_command(desired, previous, lower, upper, 0.5)
        self.assertAlmostEqual(actual[0], 0.25)
        np.testing.assert_allclose(actual[1:], 0.75)

    def test_command_rejects_non_finite_values(self):
        desired = np.zeros(16)
        desired[3] = np.nan
        with self.assertRaisesRegex(ValueError, "desired_rad"):
            smooth_gx16_command(
                desired, np.zeros(16), np.zeros(16), np.ones(16), 0.35
            )

    def test_simple_hardware_output_can_skip_collision_display(self):
        args = parse_args(["--enable-gx16-output", "--no-collision-check"])
        self.assertTrue(args.enable_gx16_output)
        self.assertTrue(args.no_collision_check)

    def test_smoothing_alpha_is_configurable(self):
        args = parse_args(["--smoothing-alpha", "0.25"])
        self.assertEqual(args.smoothing_alpha, 0.25)

    def test_control_frequency_estimate_uses_ewma(self):
        hz, last_time = update_control_frequency(None, None, 10.0)
        self.assertIsNone(hz)
        self.assertEqual(last_time, 10.0)
        hz, last_time = update_control_frequency(last_time, hz, 10.1)
        self.assertAlmostEqual(hz, 10.0)
        hz, last_time = update_control_frequency(last_time, hz, 10.3, alpha=0.5)
        self.assertAlmostEqual(hz, 7.5)
        self.assertEqual(last_time, 10.3)

    def test_control_status_payload_contains_qt_display_fields(self):
        payload = control_status_payload(
            sequence=3,
            control_hz=9.5,
            target_hz=10.0,
            inference_ms=1.2,
            source_age_ms=4.5,
            hardware_enabled=True,
            hardware_status="commanding real GX16",
            smoothing_alpha=0.35,
            gx16_command_hz=9.4,
            gx16_command_ms=96.0,
        )
        self.assertEqual(payload["sequence"], 3)
        self.assertEqual(payload["control_hz"], 9.5)
        self.assertEqual(payload["target_hz"], 10.0)
        self.assertTrue(payload["hardware_enabled"])
        self.assertEqual(payload["gx16_command_hz"], 9.4)
        self.assertEqual(payload["gx16_command_ms"], 96.0)

    def test_finite_or_none_filters_missing_and_non_finite_values(self):
        self.assertEqual(finite_or_none("3.5"), 3.5)
        self.assertIsNone(finite_or_none(None))
        self.assertIsNone(finite_or_none(float("nan")))


if __name__ == "__main__":
    unittest.main()
