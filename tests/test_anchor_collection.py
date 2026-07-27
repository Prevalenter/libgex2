import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_ROOT = ROOT / "anchor_collection"
if str(ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(ANCHOR_ROOT))

from anchor_io import (  # noqa: E402
    build_anchor_document,
    delete_anchor,
    rebuild_paired_dataset,
    save_anchor,
    validate_anchor_name,
)
from collect_ex16_anchors import decode_ex16_message, window_median  # noqa: E402
from collect_gx16_anchors import GX16ReadClient, parse_getjs_response  # noqa: E402


class AnchorStorageTest(unittest.TestCase):
    def test_name_validation(self):
        self.assertEqual(validate_anchor_name("pinch_thumb-index_01"), "pinch_thumb-index_01")
        for invalid in ("", "has space", "../escape", "中文"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_anchor_name(invalid)

    def test_pair_export_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            ex16 = build_anchor_document(
                device="ex16",
                name="open_01",
                qpos_urdf_deg=np.arange(16),
                captured_at=1.0,
                human_keypoints=np.arange(63).reshape(21, 3) / 1000.0,
            )
            gx16 = build_anchor_document(
                device="gx16",
                name="open_01",
                qpos_urdf_deg=np.arange(16) + 10,
                captured_at=2.0,
                collision=False,
            )
            save_anchor(data_dir, ex16)
            first_report = rebuild_paired_dataset(data_dir)
            self.assertEqual(first_report.ex16_only, ("open_01",))
            save_anchor(data_dir, gx16)
            report = rebuild_paired_dataset(data_dir)
            self.assertEqual(report.exported_names, ("open_01",))
            with np.load(report.output_path, allow_pickle=False) as archive:
                self.assertEqual(archive["anchor_names"].tolist(), ["open_01"])
                self.assertEqual(archive["ex16_qpos_deg"].shape, (1, 16))
                self.assertEqual(archive["gx16_qpos_rad"].shape, (1, 16))
                self.assertEqual(archive["human_keypoints"].shape, (1, 21, 3))
                np.testing.assert_allclose(
                    archive["gx16_qpos_rad"][0], np.deg2rad(np.arange(16) + 10)
                )
            delete_anchor(data_dir, "gx16", "open_01")
            report = rebuild_paired_dataset(data_dir)
            self.assertEqual(report.exported_names, ())

    def test_uncalibrated_pair_is_not_training_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            save_anchor(
                data_dir,
                build_anchor_document(
                    device="ex16",
                    name="raw",
                    qpos_urdf_deg=np.zeros(16),
                    captured_at=1.0,
                ),
            )
            save_anchor(
                data_dir,
                build_anchor_document(
                    device="gx16",
                    name="raw",
                    qpos_urdf_deg=np.zeros(16),
                    captured_at=1.0,
                ),
            )
            report = rebuild_paired_dataset(data_dir)
            self.assertEqual(report.paired_names, ("raw",))
            self.assertEqual(report.exported_names, ())
            self.assertEqual(report.missing_human_keypoints, ("raw",))


class CaptureParsingTest(unittest.TestCase):
    def test_ex16_window_median(self):
        samples = [
            (9.4, np.full(16, 100.0)),
            (9.6, np.full(16, 1.0)),
            (9.8, np.full(16, 3.0)),
            (10.0, np.full(16, 2.0)),
        ]
        np.testing.assert_allclose(window_median(samples, 10.0, 0.5), 2.0)

    def test_ex16_message_validation(self):
        message = "ex16/state " + json.dumps(
            {"urdf_deg": list(range(16)), "timestamp": 3.0, "sequence": 7}
        )
        state = decode_ex16_message(message, "ex16/state")
        self.assertEqual(state["qpos_deg"].shape, (16,))
        with self.assertRaises(ValueError):
            decode_ex16_message("ex16/state " + json.dumps({"urdf_deg": [0]}), "ex16/state")

    def test_gx16_getjs_response_validation(self):
        qpos, timestamp = parse_getjs_response(
            {
                "ok": True,
                "result": {"urdf_deg": list(range(16)), "timestamp": 4.0},
                "error": None,
            }
        )
        self.assertEqual(qpos.shape, (16,))
        self.assertEqual(timestamp, 4.0)
        with self.assertRaises(RuntimeError):
            parse_getjs_response({"ok": False, "result": None, "error": "offline"})

    def test_gx16_client_sends_only_read_command(self):
        import zmq

        context = zmq.Context.instance()
        server = context.socket(zmq.REP)
        server.linger = 0
        port = server.bind_to_random_port("tcp://127.0.0.1")
        received = []

        def reply():
            request = server.recv_json()
            received.append(request)
            server.send_json(
                {
                    "ok": True,
                    "result": {"urdf_deg": [0.0] * 16, "timestamp": 5.0},
                    "error": None,
                }
            )

        thread = threading.Thread(target=reply)
        thread.start()
        client = GX16ReadClient(context, f"tcp://127.0.0.1:{port}", 1000)
        try:
            qpos, timestamp = client.read()
        finally:
            client.close()
            thread.join(timeout=2.0)
            server.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(received, [{"cmd": "getjs", "units": "urdf_deg"}])
        np.testing.assert_allclose(qpos, 0.0)
        self.assertEqual(timestamp, 5.0)


if __name__ == "__main__":
    unittest.main()
