import time
import unittest
from types import SimpleNamespace

from nodes import gx16_zmq_node


def dry_run_args():
    return SimpleNamespace(
        dry_run=True,
        port=None,
        serial_number="FAKE_GX16",
        curr_limit=1000,
        goal_current=600,
        goal_pwm=200,
        state_read_position=False,
    )


class GX16CommandStatsTest(unittest.TestCase):
    def test_dry_run_setjs_records_command_frequency(self):
        node = gx16_zmq_node.GX16ZmqNode(dry_run_args())
        request = {
            "cmd": "setjs",
            "units": "urdf_deg",
            "positions": [0.0] * gx16_zmq_node.JOINT_COUNT,
        }
        first = node.handle_request(request)
        self.assertTrue(first["ok"])
        self.assertEqual(first["result"]["command_count"], 1)
        self.assertIsNone(first["result"]["command_hz"])
        self.assertIsNotNone(first["result"]["last_command_duration_ms"])

        time.sleep(0.01)
        second = node.handle_request(request)
        self.assertTrue(second["ok"])
        self.assertEqual(second["result"]["command_count"], 2)
        self.assertGreater(second["result"]["command_hz"], 0.0)
        self.assertIsNotNone(second["result"]["last_command_duration_ms"])

        status = node.status_payload(read_position=False)
        self.assertEqual(status["command_count"], 2)
        self.assertGreater(status["command_hz"], 0.0)

    def test_frequency_estimate_uses_ewma(self):
        hz, last_time = gx16_zmq_node.update_frequency_estimate(None, None, 10.0)
        self.assertIsNone(hz)
        self.assertEqual(last_time, 10.0)
        hz, last_time = gx16_zmq_node.update_frequency_estimate(last_time, hz, 10.1)
        self.assertAlmostEqual(hz, 10.0)
        hz, last_time = gx16_zmq_node.update_frequency_estimate(
            last_time, hz, 10.3, alpha=0.5
        )
        self.assertAlmostEqual(hz, 7.5)
        self.assertEqual(last_time, 10.3)


if __name__ == "__main__":
    unittest.main()
