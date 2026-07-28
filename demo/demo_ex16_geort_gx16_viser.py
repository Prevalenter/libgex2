"""Run a trained GeoRT model on live EX16 states and visualize GX16 in Viser."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import viser
from viser.extras import ViserUrdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
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
from demo.demo_ex16_gx16_viser import (  # noqa: E402
    GX16_MESH_DIR,
    GX16_URDF_PATH,
    configure_camera,
    load_urdf,
)


DEFAULT_ENDPOINT = "tcp://127.0.0.1:5567"
DEFAULT_CALIBRATION = GEORT_ROOT / "data" / "human_ex16_ex16_raw.npz"
CONTROL_STATUS_TOPIC = "geort/control_status"
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


def _handle_termination(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


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


def smooth_gx16_command(
    desired_rad: Sequence[float],
    previous_rad: Sequence[float],
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Validate, clamp, and EMA-smooth one GX16 command in URDF joint order."""
    desired = np.asarray(desired_rad, dtype=np.float64)
    previous = np.asarray(previous_rad, dtype=np.float64)
    lower = np.asarray(lower_rad, dtype=np.float64)
    upper = np.asarray(upper_rad, dtype=np.float64)
    for name, value in (
        ("desired_rad", desired),
        ("previous_rad", previous),
        ("lower_rad", lower),
        ("upper_rad", upper),
    ):
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain 16 finite values")
    if np.any(lower > upper):
        raise ValueError("GX16 lower joint limits exceed upper limits")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    limited_target = np.clip(desired, lower, upper)
    limited_previous = np.clip(previous, lower, upper)
    command = alpha * limited_target + (1.0 - alpha) * limited_previous
    return np.clip(command, lower, upper)


def update_control_frequency(
    previous_time: float | None,
    smoothed_hz: float | None,
    current_time: float,
    alpha: float = 0.2,
) -> tuple[float | None, float]:
    """Update an EWMA estimate of the actual control-loop frequency."""
    if not math.isfinite(current_time):
        raise ValueError("current_time must be finite")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if previous_time is None:
        return smoothed_hz, current_time
    if not math.isfinite(previous_time):
        raise ValueError("previous_time must be finite")
    elapsed = current_time - previous_time
    if elapsed <= 0:
        return smoothed_hz, current_time
    instant_hz = 1.0 / elapsed
    if smoothed_hz is None:
        return instant_hz, current_time
    return alpha * instant_hz + (1.0 - alpha) * smoothed_hz, current_time


def control_status_payload(
    sequence: int,
    control_hz: float | None,
    target_hz: float,
    inference_ms: float,
    source_age_ms: float,
    hardware_enabled: bool,
    hardware_status: str,
    smoothing_alpha: float,
    gx16_command_hz: float | None = None,
    gx16_command_ms: float | None = None,
) -> dict[str, Any]:
    """Build the live status payload consumed by the optional Qt display."""
    return {
        "sequence": int(sequence),
        "timestamp": time.time(),
        "control_hz": None if control_hz is None else float(control_hz),
        "target_hz": float(target_hz),
        "inference_ms": float(inference_ms),
        "source_age_ms": float(source_age_ms),
        "hardware_enabled": bool(hardware_enabled),
        "hardware_status": str(hardware_status),
        "smoothing_alpha": float(smoothing_alpha),
        "gx16_command_hz": None
        if gx16_command_hz is None
        else float(gx16_command_hz),
        "gx16_command_ms": None
        if gx16_command_ms is None
        else float(gx16_command_ms),
    }


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class GX16CommandClient:
    """Small resilient REQ client for the GX16 hardware-owner node."""

    def __init__(self, endpoint: str, timeout_ms: int) -> None:
        import zmq

        self.zmq = zmq
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self.context = zmq.Context.instance()
        self.socket = None
        self._reset_socket()

    def _reset_socket(self) -> None:
        if self.socket is not None:
            self.socket.close()
        self.socket = self.context.socket(self.zmq.REQ)
        self.socket.linger = 0
        self.socket.sndtimeo = self.timeout_ms
        self.socket.rcvtimeo = self.timeout_ms
        self.socket.connect(self.endpoint)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self.socket.send_json(payload)
            response = self.socket.recv_json()
        except Exception:
            # A timed-out REQ socket cannot issue another request until its
            # missing reply arrives, so rebuild it before returning control.
            self._reset_socket()
            raise
        if not isinstance(response, dict) or not response.get("ok"):
            error = response.get("error") if isinstance(response, dict) else response
            raise RuntimeError(f"GX16 command failed: {error}")
        return response

    def get_qpos(self) -> np.ndarray:
        response = self.request({"cmd": "getjs", "units": "urdf_deg"})
        positions = np.asarray(response["result"]["positions"], dtype=np.float64)
        if positions.shape != (16,) or not np.isfinite(positions).all():
            raise ValueError("GX16 getjs returned invalid joint positions")
        return np.deg2rad(positions)

    def set_qpos(self, qpos_rad: Sequence[float]) -> dict[str, Any]:
        qpos = np.asarray(qpos_rad, dtype=np.float64)
        if qpos.shape != (16,) or not np.isfinite(qpos).all():
            raise ValueError("GX16 command must contain 16 finite values")
        return self.request(
            {
                "cmd": "setjs",
                "units": "urdf_deg",
                "positions": np.rad2deg(qpos).tolist(),
            }
        )

    def torque_off(self) -> None:
        self.request({"cmd": "torque_off"})

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None


class LiveGX16Viewer:
    """Viser scene and status panel for live GeoRT inference."""

    def __init__(
        self,
        server: viser.ViserServer,
        urdf: Any,
        checkpoint_tag: str,
        calibration_path: Path,
        update_hz: float,
        smoothing_alpha: float,
        hardware_output: bool = False,
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
                f"Update rate: `{update_hz:g} Hz`  \n"
                f"EMA alpha: `{smoothing_alpha:g}`  \n"
                f"GX16 hardware: `{'ARMED' if hardware_output else 'preview only'}`"
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
        control_hz: float | None,
        gx16_command_hz: float | None,
        gx16_command_ms: float | None,
        source_age_ms: float,
        sequence: int,
        hardware_status: str = "preview only",
    ) -> None:
        self.robot.update_cfg(gx16_configuration(urdf, qpos))
        self.human_tips.points = np.asarray(human_frame[TIP_IDS], dtype=np.float32)
        if collision is None:
            collision_text = "not checked"
        elif collision:
            collision_text = "⚠️ **SELF-COLLISION**"
        else:
            collision_text = "collision-free"
        control_rate = "warming up" if control_hz is None else f"{control_hz:.1f} Hz"
        gx16_rate = (
            "n/a" if gx16_command_hz is None else f"{gx16_command_hz:.1f} Hz"
        )
        gx16_duration = (
            "n/a" if gx16_command_ms is None else f"{gx16_command_ms:.1f} ms"
        )
        self.status.content = (
            f"Sequence: `{sequence}`  \n"
            f"Control rate: `{control_rate}`  \n"
            f"GX16 command rate: `{gx16_rate}`  \n"
            f"GX16 command time: `{gx16_duration}`  \n"
            f"Inference: `{inference_ms:.2f} ms`  \n"
            f"Source age: `{source_age_ms:.1f} ms`  \n"
            f"Safety: {collision_text}  \n"
            f"Hardware: {hardware_status}"
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
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--collision-threshold-mm", type=float, default=0.5)
    parser.add_argument("--no-collision-check", action="store_true")
    parser.add_argument(
        "--enable-gx16-output",
        action="store_true",
        help="Send safety-filtered commands to a real GX16 through its ZMQ node.",
    )
    parser.add_argument(
        "--gx16-command-endpoint", default="tcp://127.0.0.1:5556"
    )
    parser.add_argument("--gx16-command-timeout-ms", type=int, default=500)
    parser.add_argument(
        "--control-status-endpoint",
        default="",
        help="Optional ZMQ PUB endpoint for live control status.",
    )
    parser.add_argument("--control-status-topic", default=CONTROL_STATUS_TOPIC)
    parser.add_argument(
        "--keep-torque-on-exit",
        action="store_true",
        help="Do not request GX16 torque-off when this process exits.",
    )
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
    if args.gx16_command_timeout_ms <= 0:
        parser.error("--gx16-command-timeout-ms must be positive")
    if args.control_status_endpoint and not args.control_status_topic:
        parser.error("--control-status-topic must not be empty")
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
    robot_config = dict(get_config("gx16"))
    # Use the limits embedded in the selected checkpoint, rather than the
    # mutable hand config used for dataset generation.
    joint_lower = np.asarray(
        model.qpos_normalizer.joint_lower_limit, dtype=np.float64
    )
    joint_upper = np.asarray(
        model.qpos_normalizer.joint_upper_limit, dtype=np.float64
    )
    collision_model = None
    if not args.no_collision_check:
        collision_config = dict(robot_config)
        collision_config["urdf_path"] = str(
            (GEORT_ROOT / collision_config["urdf_path"]).resolve()
        )
        collision_model = HandKinematicModel.build_from_config(collision_config)

    command_client = None
    initial_hardware_qpos = None
    if args.enable_gx16_output:
        command_client = GX16CommandClient(
            args.gx16_command_endpoint, args.gx16_command_timeout_ms
        )
        initial_hardware_qpos = command_client.get_qpos()
        print(
            f"GX16 hardware output ARMED via {args.gx16_command_endpoint}; "
            f"EMA smoothing alpha={args.smoothing_alpha:g}.",
            flush=True,
        )

    def has_unsafe_collision(qpos: np.ndarray) -> bool | None:
        if collision_model is None:
            return None
        return collision_model.has_self_collision(
            qpos,
            penetration_threshold=args.collision_threshold_mm / 1000.0,
        )

    server = viser.ViserServer(host=args.host, port=args.port_viser)
    viewer = LiveGX16Viewer(
        server,
        gx16_urdf,
        args.ckpt_tag,
        calibration_path,
        args.update_hz,
        args.smoothing_alpha,
        hardware_output=args.enable_gx16_output,
    )

    context = zmq.Context.instance()
    status_publisher = None
    if args.control_status_endpoint:
        status_publisher = context.socket(zmq.PUB)
        status_publisher.linger = 0
        status_publisher.bind(args.control_status_endpoint)
        print(
            f"Control status: {args.control_status_endpoint}, "
            f"topic={args.control_status_topic}",
            flush=True,
        )

    subscriber = context.socket(zmq.SUB)
    subscriber.linger = 0
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    subscriber.connect(args.ex16_endpoint)
    signal.signal(signal.SIGTERM, _handle_termination)
    try:
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
        smoothed_qpos = (
            None if initial_hardware_qpos is None else initial_hardware_qpos.copy()
        )
        control_hz = None
        last_control_time = None
        gx16_command_hz = None
        gx16_command_ms = None

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
                control_hz = None
                last_control_time = None
                time.sleep(0.02)
                continue
            if not viewer.enabled:
                control_hz = None
                last_control_time = None
                time.sleep(0.01)
                continue
            if now < next_update:
                time.sleep(0.002)
                continue
            control_hz, last_control_time = update_control_frequency(
                last_control_time, control_hz, now
            )
            next_update = now + 1.0 / args.update_hz

            human_frame = projector.project(latest["qpos_deg"])
            inference_started = time.perf_counter()
            qpos = np.asarray(model.forward(human_frame), dtype=np.float64)
            if qpos.shape != (16,) or not np.isfinite(qpos).all():
                raise ValueError("GeoRT produced an invalid GX16 command")
            qpos = np.clip(qpos, joint_lower, joint_upper)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            if smoothed_qpos is None:
                smoothed_qpos = qpos
            else:
                smoothed_qpos = smooth_gx16_command(
                    qpos,
                    smoothed_qpos,
                    joint_lower,
                    joint_upper,
                    args.smoothing_alpha,
                )

            collision = has_unsafe_collision(smoothed_qpos)
            displayed_qpos = smoothed_qpos
            hardware_status = "preview only"
            if command_client is not None:
                command = smoothed_qpos
                try:
                    response = command_client.set_qpos(command)
                except Exception as exc:
                    hardware_status = f"⚠️ command error: {exc}"
                    print(f"GX16 command error: {exc}", file=sys.stderr, flush=True)
                else:
                    result = response.get("result", {})
                    if isinstance(result, dict):
                        gx16_command_hz = finite_or_none(result.get("command_hz"))
                        gx16_command_ms = finite_or_none(
                            result.get("last_command_duration_ms")
                        )
                    displayed_qpos = command
                    if collision:
                        hardware_status = "commanding real GX16 (collision shown only)"
                    else:
                        hardware_status = "**commanding real GX16**"
            source_age_ms = max(0.0, (time.time() - latest["timestamp"]) * 1000.0)
            viewer.update(
                gx16_urdf,
                human_frame,
                displayed_qpos,
                collision,
                inference_ms,
                control_hz,
                gx16_command_hz,
                gx16_command_ms,
                source_age_ms,
                latest["sequence"],
                hardware_status,
            )
            if status_publisher is not None:
                payload = control_status_payload(
                    latest["sequence"],
                    control_hz,
                    args.update_hz,
                    inference_ms,
                    source_age_ms,
                    command_client is not None,
                    hardware_status,
                    args.smoothing_alpha,
                    gx16_command_hz,
                    gx16_command_ms,
                )
                status_publisher.send_string(
                    f"{args.control_status_topic} "
                    f"{json.dumps(payload, separators=(',', ':'))}"
                )
    except KeyboardInterrupt:
        print("Stopping live EX16 → GeoRT → GX16 viewer...", flush=True)
    finally:
        subscriber.close()
        if status_publisher is not None:
            status_publisher.close()
        if command_client is not None:
            if not args.keep_torque_on_exit:
                try:
                    command_client.torque_off()
                except Exception as exc:
                    print(
                        f"Warning: GX16 torque-off on exit failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            command_client.close()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
