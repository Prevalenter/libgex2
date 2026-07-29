from __future__ import annotations

import argparse
import unittest

from nodes.finger_pair_zmq_node import (
    ALL_IDS,
    FingerPairBus,
    FingerPairNode,
    degrees_to_raw,
    parse_args,
    raw_to_degrees,
)


class FakeBus:
    def __init__(self):
        self.positions = {motor_id: float(motor_id) for motor_id in ALL_IDS}
        self.torque = {motor_id: False for motor_id in ALL_IDS}
        self.set_calls = []

    def read_positions(self):
        return dict(self.positions)

    def read_torque_states(self):
        return dict(self.torque)

    def set_position(self, motor_id, position, max_step):
        self.set_calls.append((motor_id, position, max_step))
        return self.positions[motor_id]

    def set_torque(self, enabled):
        self.torque = {motor_id: enabled for motor_id in ALL_IDS}


class FingerPairNodeTests(unittest.TestCase):
    def args(self):
        return parse_args([])

    def test_position_conversion_handles_negative_extended_position(self):
        for value in (-90.0, 0.0, 180.0):
            self.assertAlmostEqual(raw_to_degrees(degrees_to_raw(value)), value, delta=0.05)

    def test_state_contains_both_fingers(self):
        payload = FingerPairNode(self.args(), FakeBus()).state_payload()
        self.assertEqual(payload["robot_ids"], [1, 2, 3, 4])
        self.assertEqual(payload["exoskeleton_ids"], [21, 22, 23, 24])
        self.assertEqual(payload["positions_deg"]["24"], 24.0)

    def test_set_position_validates_id_and_forwards_command(self):
        bus = FakeBus()
        node = FingerPairNode(self.args(), bus)
        response = node.handle_request(
            {"cmd": "set_position", "id": 21, "position_deg": 22.5}
        )
        self.assertTrue(response["ok"])
        self.assertEqual(bus.set_calls, [(21, 22.5, 10.0)])
        self.assertFalse(
            node.handle_request({"cmd": "set_position", "id": 9, "position_deg": 0})["ok"]
        )

    def test_default_hardware_arguments(self):
        args = self.args()
        self.assertEqual(args.port, "/dev/ttyUSB1")
        self.assertEqual(args.baudrate, 1_000_000)
        self.assertEqual(args.max_step_deg, 10.0)

    def test_close_is_safe_when_serial_port_never_opened(self):
        bus = FingerPairBus("/dev/does-not-exist")
        bus.close(disable_torque=True)
        self.assertFalse(bus.connected)


if __name__ == "__main__":
    unittest.main()
