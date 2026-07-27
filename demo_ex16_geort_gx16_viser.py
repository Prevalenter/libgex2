"""Run a trained GeoRT model on live EX16 states and visualize GX16 in Viser."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import viser
from viser.extras import ViserUrdf


PROJECT_ROOT = Path(__file__).resolve().parent
GEORT_ROOT = PROJECT_ROOT / "utils" / "GeoRT"
if str(GEORT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEORT_ROOT))

from geort import get_config, load_model  # noqa: E402
from geort.env.hand import HandKinematicModel  # noqa: E402
from nodes.record_ex16_geort import (  # noqa: E402
    DEFAULT_REFERENCE_PATH,
    DEFAULT_TOPIC,
    EX16HumanProjector,
    TIP_IDS,
    decode_state_message,
    load_reference_hand,
    wait_for_state,
)
from demo_ex16_gx16_viser import (  # noqa: E402
    GX16_MESH_DIR,
    GX16_URDF_PATH,
    configure_camera,
    load_urdf,
)


DEFAULT_ENDPOINT = "tcp://127.0.0.1:5567"
DEFAULT_CALIBRATION = GEORT_ROOT / "data" / "human_ex16_ex16_raw.npz"
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 17))
TIP_COLORS = np.asarray(
    (
        (245, 158, 11),  # index
        (34, 197, 94),   # middle
        (59, 130, 246),  # ring
        (239, 68, 68),   # thumb
    ),
    dtype=np.uint8,
)


def restore_projector(
    calibration_path: Path,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
) -> tuple[EX16HumanProjector, dict[str, Any]]:
    """Restore the exact EX16-to-human transform used to create training data."""
    calibration_path = Path(calibration_path).expanduser().resolve()
    reference_path = Path(reference_path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    reference = load_reference_hand(reference_path)
    try:
        with np.load(calibration_path, allow_pickle=False) as archive:
            required = {
                "calibration_qpos_deg",
                "calibration_scale",
                "calibration_rotation",
                "metadata_json",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"missing calibration arrays: {sorted(missing)}")
            qpos_deg = np.asarray(archive["calibration_qpos_deg"], dtype=np.float64)
            scale = float(archive["calibration_scale"].item())
            rotation = np.asarray(archive["calibration_rotation"], dtype=np.float64)
            metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load calibration from {calibration_path}: {exc}") from exc
    if not isinstance(metadata, dict) or "reference_frame" not in metadata:
        raise ValueError("Calibration metadata must contain reference_frame")

    projector = EX16HumanProjector()
    projector.restore_calibration(
        qpos_deg,
        scale,
        rotation,
        reference,
        int(metadata["reference_frame"]),
    )
    return projector, metadata


def gx16_configuration(urdf: Any, qpos: Sequence[float]) -> np.ndarray:
    """Reorder GeoRT joint1..joint16 radians into the URDF joint order."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (16,) or not np.isfinite(qpos).all():
        raise ValueError(f"Expected 16 finite GX16 joint angles, got {qpos.shape}")
    by_name = dict(zip(JOINT_NAMES, qpos))
    missing = set(JOINT_NAMES) - set(urdf.actuated_joint_names)
    if missing:
        raise ValueError(f"GX16 URDF is missing joints: {sorted(missing)}")
    return np.asarray([by_name[name] for name in urdf.actuated_joint_names])


class LiveGX16Viewer:
    """Viser scene and status panel for live GeoRT inference."""

    def __init__(
        self,
        server: viser.ViserServer,
        urdf: Any,
        checkpoint_tag: str,
        calibration_path: Path,
        update_hz: float,
    ) -> None:
        self.server = server
        self.enabled = True
        server.scene.set_up_direction("+z")
        server.scene.add_frame("/coordinates", axes_length=0.06, axes_radius=0.002)
        self.robot = ViserUrdf(server, urdf, root_node_name="/gx16")
        self.human_tips = server.scene.add_point_cloud(
            "/human_targets",
            points=np.zeros((4, 3), dtype=np.float32),
            colors=TIP_COLORS,
            point_size=0.006,
            point_shape="circle",
            precision="float32",
        )
        configure_camera(server)

        with server.gui.add_folder("Live GeoRT"):
            server.gui.add_markdown(
                f"Checkpoint: `{checkpoint_tag}`  \n"
                f"Calibration: `{calibration_path.name}`  \n"
                f"Update rate: `{update_hz:g} Hz`"
            )
            self.enabled_checkbox = server.gui.add_checkbox(
                "Enable live updates", initial_value=True
            )
            self.targets_checkbox = server.gui.add_checkbox(
                "Show human fingertip targets", initial_value=True
            )
            self.status = server.gui.add_markdown("Waiting for EX16 state...")
            self.joints = server.gui.add_markdown("")

        @self.enabled_checkbox.on_update
        def update_enabled(event: Any) -> None:
            self.enabled = bool(event.target.value)

        @self.targets_checkbox.on_update
        def update_targets(event: Any) -> None:
            self.human_tips.visible = bool(event.target.value)

    def update(
        self,
        urdf: Any,
        human_frame: np.ndarray,
        qpos: np.ndarray,
        collision: bool | None,
        inference_ms: float,
        source_age_ms: float,
        sequence: int,
    ) -> None:
        self.robot.update_cfg(gx16_configuration(urdf, qpos))
        self.human_tips.points = np.asarray(human_frame[TIP_IDS], dtype=np.float32)
        if collision is None:
            collision_text = "not checked"
        elif collision:
            collision_text = "⚠️ **SELF-COLLISION**"
        else:
            collision_text = "collision-free"
        self.status.content = (
            f"Sequence: `{sequence}`  \n"
            f"Inference: `{inference_ms:.2f} ms`  \n"
            f"Source age: `{source_age_ms:.1f} ms`  \n"
            f"Safety: {collision_text}"
        )
        qpos_deg = np.rad2deg(qpos)
        self.joints.content = (
            "GX16 joint angles (degree):  \n```text\n"
            + np.array2string(qpos_deg, precision=1, separator=", ", max_line_width=88)
            + "\n```"
        )

    def show_stale(self, seconds: float) -> None:
        self.status.content = f"⚠️ EX16 stream stale for `{seconds:.1f} s`"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize trained GeoRT GX16 output from live EX16 states."
    )
    parser.add_argument("--ex16-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--ckpt-tag", default="gx16_last")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--update-hz", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--collision-threshold-mm", type=float, default=0.5)
    parser.add_argument("--no-collision-check", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port-viser", type=int, default=8080)
    args = parser.parse_args(argv)
    for name in ("update_hz", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.smoothing_alpha) or not 0 < args.smoothing_alpha <= 1:
        parser.error("--smoothing-alpha must be in (0, 1]")
    if (
        not math.isfinite(args.collision_threshold_mm)
        or args.collision_threshold_mm < 0
    ):
        parser.error("--collision-threshold-mm must be finite and non-negative")
    if not 0 <= args.port_viser <= 65535:
        parser.error("--port-viser must be between 0 and 65535")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

    calibration_path = args.calibration.expanduser().resolve()
    projector, calibration_metadata = restore_projector(
        calibration_path, args.reference
    )
    model = load_model(args.ckpt_tag)
    gx16_urdf = load_urdf(GX16_URDF_PATH, GX16_MESH_DIR)
    collision_model = None
    if not args.no_collision_check:
        collision_config = dict(get_config("gx16"))
        collision_config["urdf_path"] = str(
            (GEORT_ROOT / collision_config["urdf_path"]).resolve()
        )
        collision_model = HandKinematicModel.build_from_config(collision_config)

    server = viser.ViserServer(host=args.host, port=args.port_viser)
    viewer = LiveGX16Viewer(
        server,
        gx16_urdf,
        args.ckpt_tag,
        calibration_path,
        args.update_hz,
    )

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.linger = 0
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    subscriber.connect(args.ex16_endpoint)
    print(
        f"Calibration restored: scale={projector.scale:.5f}, "
        f"RMSE={projector.calibration_rmse_m * 1000:.1f} mm, "
        f"reference frame={calibration_metadata['reference_frame']}",
        flush=True,
    )
    print(
        f"Waiting for {args.topic} at {args.ex16_endpoint}. "
        f"Open http://127.0.0.1:{args.port_viser}",
        flush=True,
    )
    latest = wait_for_state(subscriber, args.topic, args.timeout)
    last_received = time.monotonic()
    next_update = time.monotonic()
    smoothed_qpos = None

    try:
        while True:
            if subscriber.poll(10):
                try:
                    latest = decode_state_message(subscriber.recv_string(), args.topic)
                except (TypeError, ValueError) as exc:
                    print(f"Warning: ignoring invalid EX16 state: {exc}", file=sys.stderr)
                else:
                    last_received = time.monotonic()

            now = time.monotonic()
            stale_s = now - last_received
            if stale_s > args.timeout:
                viewer.show_stale(stale_s)
                time.sleep(0.02)
                continue
            if not viewer.enabled or now < next_update:
                time.sleep(0.002)
                continue
            next_update = now + 1.0 / args.update_hz

            human_frame = projector.project(latest["qpos_deg"])
            inference_started = time.perf_counter()
            qpos = np.asarray(model.forward(human_frame), dtype=np.float64)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            if smoothed_qpos is None:
                smoothed_qpos = qpos
            else:
                alpha = args.smoothing_alpha
                smoothed_qpos = alpha * qpos + (1.0 - alpha) * smoothed_qpos

            collision = None
            if collision_model is not None:
                collision = collision_model.has_self_collision(
                    smoothed_qpos,
                    penetration_threshold=args.collision_threshold_mm / 1000.0,
                )
            source_age_ms = max(0.0, (time.time() - latest["timestamp"]) * 1000.0)
            viewer.update(
                gx16_urdf,
                human_frame,
                smoothed_qpos,
                collision,
                inference_ms,
                source_age_ms,
                latest["sequence"],
            )
    except KeyboardInterrupt:
        print("Stopping live EX16 → GeoRT → GX16 viewer...", flush=True)
    finally:
        subscriber.close()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
