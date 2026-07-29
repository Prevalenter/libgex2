from __future__ import annotations

import unittest

import numpy as np

from nodes.tactile_admittance_control_node import (
    ACTIVE_INDICES,
    AdmittanceController,
    ThumbKinematics,
    admittance_step,
    motor_to_urdf_rad,
    parse_args,
    urdf_rad_to_motor_deg,
)


class TactileAdmittanceControlTests(unittest.TestCase):
    def ready_controller(self, extra_args=None):
        args = parse_args([] if extra_args is None else extra_args)
        controller = AdmittanceController(args)
        q = np.deg2rad([30.0, 40.0, 35.0, 25.0])
        motor = urdf_rad_to_motor_deg(q, args.motor_zero_deg)
        positions = {index: float(value) for index, value in enumerate(motor, start=1)}
        controller.force_zero = np.zeros(3)
        controller.update_force([0, 0, 0], now=1.0)
        controller.update_motor_state(
            positions,
            {motor_id: True for motor_id in (1, 2, 3, 4)},
            now=1.0,
        )
        controller.start(now=1.0)
        return controller, positions

    def test_motor_command_timeout_default(self):
        self.assertEqual(parse_args([]).motor_command_timeout_ms, 1000)

    def test_motor_urdf_round_trip_with_joint3_reversed(self):
        motor = np.asarray([100.0, 110.0, 80.0, 120.0])
        q = motor_to_urdf_rad(motor, [90, 90, 90, 90])
        np.testing.assert_allclose(np.rad2deg(q), [10, 20, 10, 30])
        np.testing.assert_allclose(urdf_rad_to_motor_deg(q, [90] * 4), motor)

    def test_analytic_jacobian_matches_finite_difference(self):
        args = parse_args([])
        kinematics = ThumbKinematics(args.urdf)
        q = np.deg2rad([30.0, 40.0, 35.0, 25.0])
        pose, jacobian = kinematics.pose_and_jacobian(q)
        epsilon = 1e-7
        numeric = np.zeros((3, 4))
        for index in range(4):
            shifted = q.copy()
            shifted[index] += epsilon
            shifted_pose, _ = kinematics.pose_and_jacobian(shifted)
            numeric[:, index] = (shifted_pose[:3, 3] - pose[:3, 3]) / epsilon
        np.testing.assert_allclose(jacobian, numeric, rtol=1e-5, atol=1e-7)

    def test_admittance_step_is_finite_and_bounded(self):
        args = parse_args([])
        kinematics = ThumbKinematics(args.urdf)
        pose, jacobian = kinematics.pose_and_jacobian(np.deg2rad([30, 40, 35, 25]))
        delta_q, condition, delta_x = admittance_step(
            jacobian[:, ACTIVE_INDICES],
            pose[:3, :3],
            [2, -3, 4],
            np.asarray([0.02, 0.02, 0.05]) / 1000,
            [1, 1, 1],
            damping=0.005,
            max_cartesian_step_m=0.0001,
            max_joint_step_rad=np.deg2rad(0.2),
        )
        self.assertTrue(np.isfinite(delta_q).all())
        self.assertTrue(np.isfinite(condition))
        self.assertLessEqual(np.max(np.abs(np.rad2deg(delta_q))), 0.2 + 1e-9)
        self.assertLessEqual(np.max(np.abs(delta_x)), 0.0001 + 1e-12)

    def test_joint_command_accumulates_when_measured_position_is_static(self):
        controller, positions = self.ready_controller()
        first = controller.compute_command(now=1.0)
        second = controller.compute_command(now=1.0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        for motor_id in (1, 3, 4):
            first_offset = first[motor_id] - positions[motor_id]
            second_offset = second[motor_id] - positions[motor_id]
            self.assertAlmostEqual(second_offset, 2.0 * first_offset, places=9)
        self.assertEqual(controller.command_count, 2)

    def test_tracking_error_limit_faults_before_sending_unbounded_target(self):
        controller, _positions = self.ready_controller(
            ["--max-tracking-error-deg", "0.001"]
        )
        self.assertIsNone(controller.compute_command(now=1.0))
        self.assertEqual(controller.mode, "fault")
        self.assertIn("tracking error", controller.fault)
        self.assertEqual(controller.command_count, 0)


if __name__ == "__main__":
    unittest.main()
