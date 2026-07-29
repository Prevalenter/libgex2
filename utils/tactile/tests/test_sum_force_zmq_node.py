from __future__ import annotations

import unittest

from utils.tactile.hand import FingerConfig
from utils.tactile.tactile_sum_force_zmq_node import (
    build_payload,
    decode_sum_force,
    parse_args,
)


class TactileSumForceZmqNodeTests(unittest.TestCase):
    def test_decode_signed_xy_and_unsigned_z_in_newtons(self):
        self.assertEqual(decode_sum_force(bytes([255, 128, 10]), 0.1), [-0.1, -12.8, 1.0])

    def test_build_payload(self):
        fingers = (
            FingerConfig("thumb", "A"),
            FingerConfig("index", "B"),
            FingerConfig("middle", "C"),
        )
        payload = build_payload(
            fingers,
            ["/dev/tty0", "/dev/tty1", "/dev/tty2"],
            [bytes([10, 246, 20]), bytes([0, 0, 1]), bytes([128, 127, 255])],
            now_ns=1_700_000_000_123_456_789,
        )
        self.assertEqual(payload["stamp_sec"], 1_700_000_000)
        self.assertEqual(payload["stamp_nanosec"], 123_456_789)
        self.assertEqual(payload["axes"], ["x", "y", "z"])
        self.assertEqual(payload["unit"], "N")
        self.assertEqual(payload["force_xyz_n"][0], [1.0, -1.0, 2.0])
        for actual, expected in zip(
            payload["force_xyz_n"][2], [-12.8, 12.7, 25.5]
        ):
            self.assertAlmostEqual(actual, expected)

    def test_parse_args_defaults(self):
        args = parse_args([])
        self.assertEqual(args.endpoint, "tcp://127.0.0.1:5561")
        self.assertEqual(args.force_resolution_n, 0.1)
        self.assertEqual(args.port, "/dev/ttyACM0")
        self.assertEqual(args.finger, "thumb")
        self.assertEqual(args.device_id, 0x03)
        self.assertEqual(args.publish_rate, 30.0)

    def test_parse_args_accepts_single_port(self):
        args = parse_args(["--port", "/dev/ttyUSB0", "--finger", "index"])
        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertEqual(args.finger, "index")


if __name__ == "__main__":
    unittest.main()
