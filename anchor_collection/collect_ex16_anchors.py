"""Collect named EX16 anchors from the live ZMQ state stream in Viser."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import viser
from viser.extras import ViserUrdf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GEORT_ROOT = REPO_ROOT / "utils" / "GeoRT"
for import_path in (SCRIPT_DIR, REPO_ROOT, GEORT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from anchor_io import (  # noqa: E402
    build_anchor_document,
    delete_anchor,
    list_anchor_names,
    load_anchor,
    rebuild_paired_dataset,
    save_anchor,
    validate_anchor_name,
)
from viewer_utils import (  # noqa: E402
    configure_camera,
    dropdown_options,
    load_visual_urdf,
    report_markdown,
    urdf_configuration,
)
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5567"
DEFAULT_TOPIC = "ex16/state"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_URDF = REPO_ROOT / "libgex" / "ex16" / "urdf" / "glove4.urdf"
DEFAULT_MESH_DIR = REPO_ROOT / "libgex" / "ex16" / "meshes"
DEFAULT_CALIBRATION = GEORT_ROOT / "data" / "human_ex16_ex16_raw.npz"
DEFAULT_REFERENCE = GEORT_ROOT / "data" / "human_alex.npy"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect named EX16 anchor poses in Viser.")
    parser.add_argument("--state-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sample-window", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port-viser", type=int, default=8081)
    args = parser.parse_args(argv)
    for name in ("sample_window", "timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0 <= args.port_viser <= 65535:
        parser.error("--port-viser must be between 0 and 65535")
    return args


def decode_ex16_message(message: str, topic: str) -> dict[str, Any]:
    try:
        received_topic, encoded = message.split(" ", 1)
        payload = json.loads(encoded)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid EX16 state message: {exc}") from exc
    if received_topic != topic or not isinstance(payload, dict):
        raise ValueError(f"expected topic {topic!r}, got {received_topic!r}")
    qpos = np.asarray(payload.get("urdf_deg"), dtype=np.float64)
    if qpos.shape != (16,) or not np.isfinite(qpos).all():
        raise ValueError("EX16 urdf_deg must contain 16 finite values")
    timestamp = float(payload.get("timestamp", time.time()))
    if not math.isfinite(timestamp):
        raise ValueError("EX16 timestamp must be finite")
    return {
        "qpos_deg": qpos,
        "timestamp": timestamp,
        "sequence": payload.get("sequence"),
    }


def window_median(samples: Sequence[tuple[float, np.ndarray]], now: float, window: float) -> np.ndarray:
    rows = [qpos for received_at, qpos in samples if now - received_at <= window]
    if not rows:
        raise RuntimeError("no EX16 samples are available in the capture window")
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 16 or not np.isfinite(array).all():
        raise ValueError("EX16 capture window contains invalid joint values")
    return np.median(array, axis=0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc
    # Keep GeoRT's PyTorch dependency out of lightweight parsing/storage tests.
    from demo_ex16_geort_gx16_viser import restore_projector

    data_dir = args.data_dir.expanduser().resolve()
    urdf = load_visual_urdf(args.urdf, args.mesh_dir)
    projector = None
    calibration_metadata: dict[str, Any] | None = None
    calibration_error = None
    try:
        projector, calibration_metadata = restore_projector(args.calibration, args.reference)
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        calibration_error = str(exc)
        print(f"Warning: EX16 calibration unavailable: {exc}", file=sys.stderr, flush=True)

    server = viser.ViserServer(host=args.host, port=args.port_viser)
    server.scene.set_up_direction("+z")
    configure_camera(server)
    robot = ViserUrdf(server, urdf, root_node_name="/ex16")
    robot.update_cfg(urdf_configuration(urdf, np.zeros(16)))

    with server.gui.add_folder("EX16 live state"):
        live_updates = server.gui.add_checkbox("Live updates", initial_value=True)
        stream_status = server.gui.add_markdown("Waiting for EX16 state...")
        joint_status = server.gui.add_markdown("")
    with server.gui.add_folder("Anchor capture"):
        anchor_name = server.gui.add_text("Anchor name", initial_value="")
        notes = server.gui.add_text("Notes", initial_value="", multiline=True)
        allow_replace = server.gui.add_checkbox("Confirm overwrite / delete", initial_value=False)
        save_button = server.gui.add_button("Save current EX16 anchor")
        saved_dropdown = server.gui.add_dropdown(
            "Saved EX16 anchors", dropdown_options(list_anchor_names(data_dir, "ex16"))
        )
        load_button = server.gui.add_button("Load selected anchor")
        delete_button = server.gui.add_button("Delete selected anchor")
        action_status = server.gui.add_markdown("Ready.")
        dataset_status = server.gui.add_markdown(report_markdown(rebuild_paired_dataset(data_dir)))

    lock = threading.RLock()
    samples: deque[tuple[float, np.ndarray]] = deque()
    latest: dict[str, Any] | None = None
    last_received: float | None = None

    def refresh_dataset() -> None:
        names = list_anchor_names(data_dir, "ex16")
        saved_dropdown.options = dropdown_options(names)
        if saved_dropdown.value not in saved_dropdown.options:
            saved_dropdown.value = saved_dropdown.options[0]
        dataset_status.content = report_markdown(rebuild_paired_dataset(data_dir))

    @save_button.on_click
    def save_current(_event: Any) -> None:
        nonlocal latest
        try:
            name = validate_anchor_name(anchor_name.value)
            now = time.monotonic()
            with lock:
                if latest is None or last_received is None or now - last_received > args.timeout:
                    raise RuntimeError("EX16 stream is missing or stale")
                qpos = window_median(tuple(samples), now, args.sample_window)
                source_timestamp = float(latest["timestamp"])
                sequence = latest.get("sequence")
                sample_count = sum(
                    now - timestamp <= args.sample_window for timestamp, _ in samples
                )
            points = None if projector is None else projector.project(qpos)
            calibration = None
            if projector is not None:
                calibration = {
                    "path": str(args.calibration.expanduser().resolve()),
                    "reference_path": str(args.reference.expanduser().resolve()),
                    "reference_frame": int(calibration_metadata["reference_frame"]),
                    "scale": float(projector.scale),
                    "rotation": np.asarray(projector.rotation, dtype=np.float64).tolist(),
                    "calibration_qpos_deg": np.asarray(
                        projector.calibration_qpos_deg, dtype=np.float64
                    ).tolist(),
                    "four_tip_rmse_m": float(projector.calibration_rmse_m),
                }
            document = build_anchor_document(
                device="ex16",
                name=name,
                qpos_urdf_deg=qpos,
                captured_at=time.time(),
                notes=notes.value,
                source={
                    "mode": "zmq_stream_median",
                    "endpoint": args.state_endpoint,
                    "topic": args.topic,
                    "source_timestamp": source_timestamp,
                    "sequence": sequence,
                    "sample_window_s": args.sample_window,
                    "sample_count": sample_count,
                },
                human_keypoints=points,
                calibration=calibration,
            )
            path = save_anchor(data_dir, document, overwrite=bool(allow_replace.value))
            refresh_dataset()
            saved_dropdown.value = name
            action_status.content = f"Saved `{name}` to `{path}`"
        except Exception as exc:
            action_status.content = f"**Save failed:** {exc}"

    @load_button.on_click
    def load_selected(_event: Any) -> None:
        if saved_dropdown.value == "(none)":
            action_status.content = "**Load failed:** no EX16 anchors saved"
            return
        try:
            document = load_anchor(data_dir, "ex16", saved_dropdown.value)
            qpos = np.asarray(document["qpos_urdf_deg"], dtype=np.float64)
            live_updates.value = False
            robot.update_cfg(urdf_configuration(urdf, qpos))
            anchor_name.value = document["name"]
            notes.value = document.get("notes", "")
            action_status.content = f"Loaded `{document['name']}`; live updates paused."
        except Exception as exc:
            action_status.content = f"**Load failed:** {exc}"

    @delete_button.on_click
    def delete_selected(_event: Any) -> None:
        try:
            if saved_dropdown.value == "(none)":
                raise RuntimeError("no EX16 anchors saved")
            if not allow_replace.value:
                raise RuntimeError("enable 'Confirm overwrite / delete' first")
            name = saved_dropdown.value
            delete_anchor(data_dir, "ex16", name)
            refresh_dataset()
            action_status.content = f"Deleted EX16 anchor `{name}`"
        except Exception as exc:
            action_status.content = f"**Delete failed:** {exc}"

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.linger = 0
    subscriber.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    subscriber.connect(args.state_endpoint)
    print(f"EX16 collector: http://127.0.0.1:{args.port_viser}", flush=True)
    print(f"Subscribing to {args.state_endpoint}, topic={args.topic}", flush=True)
    if calibration_error:
        action_status.content = f"Calibration unavailable; raw anchors only: `{calibration_error}`"

    last_gui_update = 0.0
    try:
        while True:
            if subscriber.poll(20):
                try:
                    state = decode_ex16_message(subscriber.recv_string(), args.topic)
                except (TypeError, ValueError) as exc:
                    print(f"Warning: ignoring invalid EX16 state: {exc}", file=sys.stderr)
                else:
                    received_at = time.monotonic()
                    with lock:
                        latest = state
                        last_received = received_at
                        samples.append((received_at, state["qpos_deg"].copy()))
                        keep_after = received_at - max(args.sample_window * 3.0, args.timeout)
                        while samples and samples[0][0] < keep_after:
                            samples.popleft()
                    if live_updates.value:
                        robot.update_cfg(urdf_configuration(urdf, state["qpos_deg"]))
            now = time.monotonic()
            if now - last_gui_update >= 0.1:
                with lock:
                    current = latest
                    received_at = last_received
                if current is None or received_at is None:
                    stream_status.content = "Waiting for EX16 state..."
                else:
                    age = now - received_at
                    label = "connected" if age <= args.timeout else "⚠️ stale"
                    stream_status.content = (
                        f"Status: **{label}**  \nSequence: `{current.get('sequence')}`  \n"
                        f"Receive age: `{age * 1000.0:.0f} ms`  \n"
                        f"Source age: `{max(0.0, time.time() - current['timestamp']) * 1000.0:.0f} ms`"
                    )
                    joint_status.content = "Joint angles (deg):  \n```text\n" + np.array2string(
                        current["qpos_deg"], precision=1, separator=", ", max_line_width=88
                    ) + "\n```"
                last_gui_update = now
    except KeyboardInterrupt:
        print("Stopping EX16 anchor collector...", flush=True)
    finally:
        subscriber.close()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
