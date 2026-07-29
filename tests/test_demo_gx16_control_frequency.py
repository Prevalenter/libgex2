import tempfile
import unittest
from pathlib import Path
from unittest import mock

from demo import demo_gx16_control_frequency as benchmark


class GX16ControlFrequencyBenchmarkTest(unittest.TestCase):
    def test_default_args_send_zero_commands_fast(self):
        args = benchmark.parse_args([])
        self.assertEqual(args.serial_number, benchmark.DEFAULT_SERIAL_NUMBER)
        self.assertEqual(args.samples, 100)
        self.assertEqual(args.warmup, 5)
        self.assertEqual(args.period, 0.0)
        self.assertFalse(args.read_position)

    def test_read_position_flag_is_optional(self):
        args = benchmark.parse_args(["--read-position"])
        self.assertTrue(args.read_position)

    def test_rejects_negative_period(self):
        with self.assertRaises(SystemExit):
            benchmark.parse_args(["--period", "-0.1"])

    def test_latency_timer_is_read_from_usb_serial_sysfs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timer = root / "ttyUSB0" / "latency_timer"
            timer.parent.mkdir(parents=True)
            timer.write_text("16\n", encoding="utf-8")
            with mock.patch.object(
                benchmark,
                "latency_timer_path",
                return_value=timer,
            ):
                self.assertEqual(benchmark.read_latency_timer("/dev/ttyUSB0"), "16")

    def test_summary_statistics_are_reported_in_ms(self):
        stats = benchmark.summarize_ms([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(stats["count"], 4.0)
        self.assertEqual(stats["mean_ms"], 25.0)
        self.assertEqual(stats["median_ms"], 25.0)
        self.assertEqual(stats["min_ms"], 10.0)
        self.assertEqual(stats["max_ms"], 40.0)

    def test_print_summary_accepts_optional_getjs_samples(self):
        benchmark.print_summary(
            setjs_ms=[10.0, 10.0],
            getjs_ms=[20.0, 20.0],
            loop_ms=[30.0, 30.0],
            elapsed_s=0.06,
        )


if __name__ == "__main__":
    unittest.main()
