from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.tactile.hand import FingerConfig, PortInfo, load_default_serials, load_finger_configs, split_connected_serials
from utils.tactile.tactile_zmq_node import build_payload, ordered_ports_for_fingers
from utils.tactile.uart import (
    READ_FUNC,
    RX_HEAD,
    TACTILE_ADDR,
    TACTILE_POINTS,
    build_read_request,
    lrc_cal,
    parse_frame,
    raw_to_force_frame,
)
from utils.tactile.viz import load_recording


def build_response(dev_id: int, addr: int, data: bytes) -> bytes:
    func = (READ_FUNC | 0x80) & 0xFF
    payload = b"\x00" + data
    length_field = 9 + len(payload)
    frame = bytearray()
    frame += RX_HEAD
    frame += struct.pack("<H", length_field)
    frame += bytes([dev_id, 0x00, func])
    frame += struct.pack("<I", addr)
    frame += struct.pack("<H", len(data))
    frame += payload
    frame += bytes([lrc_cal(frame)])
    return bytes(frame)


class UARTTests(unittest.TestCase):
    def test_lrc_balances_request_bytes(self):
        request = build_read_request(0x03, READ_FUNC, TACTILE_ADDR, 0x20)
        self.assertEqual(sum(request) & 0xFF, 0)

    def test_parse_synthetic_response(self):
        data = bytes(range(8))
        frame = parse_frame(build_response(0x03, TACTILE_ADDR, data))
        self.assertEqual(frame.dev_id, 0x03)
        self.assertEqual(frame.func, READ_FUNC | 0x80)
        self.assertEqual(frame.addr, TACTILE_ADDR)
        self.assertEqual(frame.data_len, len(data))
        self.assertEqual(frame.payload, b"\x00" + data)

    def test_parse_rejects_bad_lrc(self):
        raw = bytearray(build_response(0x03, TACTILE_ADDR, b"\x01\x02"))
        raw[-1] ^= 0xFF
        with self.assertRaises(ValueError):
            parse_frame(bytes(raw))

    def test_raw_to_force_frame(self):
        raw = np.arange(TACTILE_POINTS * 3, dtype=np.uint8)
        frame = raw_to_force_frame(raw)
        self.assertEqual(frame.shape, (TACTILE_POINTS, 3))
        self.assertEqual(frame.dtype, np.float32)
        np.testing.assert_array_equal(frame[0], np.array([0, 1, 2], dtype=np.float32))

    def test_raw_to_force_frame_accepts_bytes(self):
        raw = bytes(np.arange(TACTILE_POINTS * 3, dtype=np.uint8))
        frame = raw_to_force_frame(raw)
        self.assertEqual(frame.shape, (TACTILE_POINTS, 3))
        np.testing.assert_array_equal(frame[0], np.array([0, 1, 2], dtype=np.float32))


class RecordingTests(unittest.TestCase):
    def save_and_load(self, array: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.npy"
            np.save(path, array)
            return load_recording(path)

    def test_load_multi_sensor_raw_recording(self):
        raw = np.zeros((2, 3, TACTILE_POINTS * 3), dtype=np.uint8)
        loaded = self.save_and_load(raw)
        self.assertEqual(loaded.shape, (2, 3, TACTILE_POINTS, 3))

    def test_load_single_sensor_frame_recording(self):
        frames = np.zeros((2, TACTILE_POINTS, 3), dtype=np.float32)
        loaded = self.save_and_load(frames)
        self.assertEqual(loaded.shape, (2, 1, TACTILE_POINTS, 3))

    def test_load_single_raw_frame(self):
        raw = np.zeros((TACTILE_POINTS * 3,), dtype=np.uint8)
        loaded = self.save_and_load(raw)
        self.assertEqual(loaded.shape, (1, 1, TACTILE_POINTS, 3))


class HandConfigTests(unittest.TestCase):
    def test_load_finger_configs_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "fingers": [
                            {"finger": "thumb", "serial": "A"},
                            {"finger": "index", "serial": "B"},
                            {"finger": "middle", "serial": "C"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_finger_configs(path),
                (
                    FingerConfig("thumb", "A"),
                    FingerConfig("index", "B"),
                    FingerConfig("middle", "C"),
                ),
            )
            self.assertEqual(load_default_serials(path), ("A", "B", "C"))

    def test_load_default_serials_from_legacy_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"default_serials": ["A", "B", "C"]}), encoding="utf-8")
            self.assertEqual(
                load_finger_configs(path),
                (
                    FingerConfig("thumb", "A"),
                    FingerConfig("index", "B"),
                    FingerConfig("middle", "C"),
                ),
            )

    def test_split_connected_serials(self):
        ports = [
            PortInfo("/dev/ttyUSB0", "hw0", None, None, "B"),
            PortInfo("/dev/ttyACM0", "hw1", None, None, "C"),
        ]
        connected, missing = split_connected_serials(("A", "B", "C"), port_infos=ports)
        self.assertEqual(connected, [("B", "/dev/ttyUSB0"), ("C", "/dev/ttyACM0")])
        self.assertEqual(missing, ["A"])

    def test_split_connected_serials_none_available(self):
        connected, missing = split_connected_serials(("A", "B"), port_infos=[])
        self.assertEqual(connected, [])
        self.assertEqual(missing, ["A", "B"])


class TactileZmqNodeTests(unittest.TestCase):
    def fingers(self):
        return (
            FingerConfig("thumb", "A"),
            FingerConfig("index", "B"),
            FingerConfig("middle", "C"),
        )

    def test_ordered_ports_for_fingers(self):
        ports = [
            PortInfo("/dev/ttyACM2", "hw2", None, None, "C"),
            PortInfo("/dev/ttyACM0", "hw0", None, None, "A"),
            PortInfo("/dev/ttyACM1", "hw1", None, None, "B"),
        ]
        self.assertEqual(
            ordered_ports_for_fingers(self.fingers(), port_infos=ports),
            ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"],
        )

    def test_ordered_ports_for_fingers_missing_fails(self):
        ports = [PortInfo("/dev/ttyACM0", "hw0", None, None, "A")]
        with self.assertRaises(RuntimeError) as cm:
            ordered_ports_for_fingers(self.fingers(), port_infos=ports)
        self.assertIn("missing configured tactile sensors", str(cm.exception))
        self.assertIn("index", str(cm.exception))
        self.assertIn("middle", str(cm.exception))

    def test_build_payload(self):
        frames = np.zeros((3, TACTILE_POINTS, 3), dtype=np.float32)
        frames[1, 0] = np.array([1, 2, 3], dtype=np.float32)
        payload = build_payload(
            self.fingers(),
            ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"],
            frames,
            now_ns=1_700_000_000_123_456_789,
        )
        self.assertEqual(payload["stamp_sec"], 1_700_000_000)
        self.assertEqual(payload["stamp_nanosec"], 123_456_789)
        self.assertEqual(payload["fingers"], ["thumb", "index", "middle"])
        self.assertEqual(payload["serials"], ["A", "B", "C"])
        self.assertEqual(payload["shape"], [3, TACTILE_POINTS, 3])
        self.assertEqual(payload["force_frames"][1][0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
