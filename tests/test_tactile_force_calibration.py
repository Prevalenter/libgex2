from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from demo.demo_tactile_force_calibration import (
    CalibrationSample,
    decode_control_state,
    decode_motor_state,
    decode_reference_force,
    decode_tactile_payload,
    parse_args,
    save_samples_csv,
    subtract_tactile_zero,
)


class TactileForceCalibrationTests(unittest.TestCase):
    def test_decode_tactile_payload_selects_finger(self):
        reading = decode_tactile_payload(
            {
                "stamp_sec": 10,
                "stamp_nanosec": 500_000_000,
                "fingers": ["thumb", "index"],
                "force_xyz_n": [[1, 2, 3], [-1, -2, 4]],
            },
            "index",
        )
        self.assertEqual(reading.finger, "index")
        self.assertEqual((reading.fx_n, reading.fy_n, reading.fz_n), (-1.0, -2.0, 4.0))
        self.assertAlmostEqual(reading.magnitude_n, 21**0.5)
        self.assertEqual(reading.source_timestamp, 10.5)

    def test_decode_reference_force(self):
        self.assertEqual(decode_reference_force("-1.250\n"), -1.25)
        with self.assertRaises(ValueError):
            decode_reference_force("nan")

    def test_subtract_tactile_zero(self):
        raw = decode_tactile_payload(
            {
                "stamp_sec": 1,
                "stamp_nanosec": 0,
                "fingers": ["thumb"],
                "force_xyz_n": [[1.2, -0.4, 3.0]],
            },
            "thumb",
        )
        corrected = subtract_tactile_zero(raw, [1.0, -0.5, 2.8])
        self.assertAlmostEqual(corrected.fx_n, 0.2)
        self.assertAlmostEqual(corrected.fy_n, 0.1)
        self.assertAlmostEqual(corrected.fz_n, 0.2)
        self.assertAlmostEqual(corrected.magnitude_n, 0.3)

    def test_decode_motor_state(self):
        positions, torque = decode_motor_state(
            {
                "positions_deg": {
                    str(motor_id): motor_id + 0.5
                    for motor_id in (1, 2, 3, 4, 21, 22, 23, 24)
                },
                "torque_enabled": {
                    str(motor_id): motor_id == 1
                    for motor_id in (1, 2, 3, 4, 21, 22, 23, 24)
                },
            }
        )
        self.assertEqual(positions[21], 21.5)
        self.assertTrue(torque[1])
        self.assertFalse(torque[24])

    def test_decode_control_state(self):
        state = decode_control_state(
            {
                "mode": "idle",
                "active": False,
                "target_force_xyz_n": [0, 0, 1],
                "measured_force_xyz_n": [0.1, -0.2, 0.3],
            }
        )
        self.assertEqual(state["mode"], "idle")
        with self.assertRaises(ValueError):
            decode_control_state(
                {
                    "mode": "idle",
                    "active": False,
                    "target_force_xyz_n": [0, 0, 1],
                    "measured_force_xyz_n": [float("nan"), 0, 0],
                }
            )

    def test_default_hardware_arguments(self):
        args = parse_args([])
        self.assertEqual(args.tactile_port, "/dev/ttyACM0")
        self.assertEqual(args.force_port, "/dev/ttyUSB0")
        self.assertEqual(args.finger, "thumb")
        self.assertEqual(args.device_id, 0x03)
        self.assertEqual(args.tactile_rate, 30.0)
        self.assertEqual(args.motor_port, "/dev/ttyUSB1")
        self.assertEqual(args.motor_command_endpoint, "tcp://127.0.0.1:5580")
        self.assertEqual(args.command_timeout_ms, 1500)
        self.assertEqual(args.sensor_rpy_deg, (0, 0, 0))
        self.assertEqual(args.contact_offset_m, (0, 0, 0))

    def test_save_samples_csv(self):
        sample = CalibrationSample(1.0, "thumb", 2.0, 0.1, 0.2, 0.3, 0.4, 5.0)
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "samples.csv"
            save_samples_csv(path, [sample])
            with path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["finger"], "thumb")
        self.assertEqual(rows[0]["reference_force"], "5.0")


if __name__ == "__main__":
    unittest.main()
