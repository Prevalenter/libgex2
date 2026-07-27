"""Record EX16 states as a GeoRT-compatible human hand trajectory.

The recorder subscribes to ``nodes/ex16_zmq_node.py``, resamples the latest
state at a fixed rate, and saves both a ``[T, 21, 3]`` GeoRT array and the raw
16-DoF EX16 readings.  EX16 has no pinky sensing; pinky points in the GeoRT
array are explicitly synthetic and are ignored by the Allegro/GX16 configs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5567"
DEFAULT_TOPIC = "ex16/state"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "utils" / "GeoRT" / "data"
DEFAULT_REFERENCE_PATH = DEFAULT_OUTPUT_DIR / "human_alex.npy"
DEFAULT_URDF_PATH = REPO_ROOT / "libgex" / "ex16" / "urdf" / "glove4.urdf"
DEFAULT_FPS = 10.0
DEFAULT_FRAMES = 3498
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 17))
TIP_IDS = np.asarray((4, 8, 12, 16), dtype=np.int64)

# Four EX16 mechanisms correspond to thumb, index, middle and ring.  There
# are five URDF link frames per mechanism after the wrist; the fourth
# mechanical link frame is omitted so each chain follows MediaPipe's four
# post-wrist landmarks and still terminates at the physical fingertip link.
SOURCE_CHAINS = (
    ("Link1", "Link2", "Link3", "Link17"),
    ("Link5", "Link6", "Link7", "Link18"),
    ("Link9", "Link10", "Link11", "Link19"),
    ("link13", "link14", "link15", "link20"),
)
HUMAN_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
)


def load_reference_hand(path: Path) -> np.ndarray:
    """Load a finite GeoRT ``[T, 21, 3]`` reference trajectory."""
    path = Path(path).expanduser().resolve()
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to load reference hand data from {path}: {exc}") from exc
    if data.ndim != 3 or data.shape[1:] != (21, 3) or data.shape[0] == 0:
        raise ValueError(
            f"Expected reference shape [T, 21, 3], got {data.shape} from {path}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"Reference hand data contains NaN or Inf: {path}")
    return np.asarray(data, dtype=np.float64)


def select_open_reference_frame(reference: np.ndarray) -> int:
    """Choose the frame with the largest combined four-finger wrist reach."""
    wrist_relative = reference[:, TIP_IDS] - reference[:, [0]]
    reach = np.linalg.norm(wrist_relative, axis=-1).sum(axis=-1)
    return int(np.argmax(reach))


def fit_origin_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit positive scale and proper row-vector rotation about the wrist."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must have matching [N, 3] shapes, got "
            f"{source.shape} and {target.shape}"
        )
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target must contain only finite values")
    source_energy = float(np.sum(source * source))
    if source_energy <= 1e-12:
        raise ValueError("source calibration points are degenerate")

    u, _, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    rotated = source @ rotation
    scale = float(np.sum(rotated * target) / source_energy)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"calibration produced an invalid scale: {scale}")
    return scale, rotation


class EX16HumanProjector:
    """Project EX16 URDF states into the GeoRT wrist-local hand convention."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF_PATH) -> None:
        try:
            import yourdfpy
        except ImportError as exc:
            raise ImportError("yourdfpy is required: python -m pip install yourdfpy") from exc

        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.urdf = yourdfpy.URDF.load(
            self.urdf_path,
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        if "plam_link" not in {link.name for link in self.urdf.robot.links}:
            raise ValueError(f"EX16 URDF is missing plam_link: {self.urdf_path}")
        available_joints = set(self.urdf.actuated_joint_names)
        missing_joints = set(JOINT_NAMES) - available_joints
        if missing_joints:
            raise ValueError(f"EX16 URDF is missing joints: {sorted(missing_joints)}")

        self.scale: float | None = None
        self.rotation: np.ndarray | None = None
        self.reference_frame: np.ndarray | None = None
        self.reference_frame_index: int | None = None
        self.open_projected: np.ndarray | None = None
        self.calibration_qpos_deg: np.ndarray | None = None
        self.calibration_rmse_m: float | None = None

    @staticmethod
    def _validate_qpos(qpos_deg: Sequence[float]) -> np.ndarray:
        qpos = np.asarray(qpos_deg, dtype=np.float64)
        if qpos.shape != (16,):
            raise ValueError(f"Expected 16 EX16 joint angles, got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError("EX16 joint angles contain NaN or Inf")
        return qpos

    def source_keypoints(self, qpos_deg: Sequence[float]) -> np.ndarray:
        """Return four EX16 chains relative to ``plam_link`` in metres."""
        qpos = self._validate_qpos(qpos_deg)
        self.urdf.update_cfg(
            {name: np.deg2rad(value) for name, value in zip(JOINT_NAMES, qpos)}
        )
        points = np.zeros((21, 3), dtype=np.float64)
        for source_chain, human_chain in zip(SOURCE_CHAINS, HUMAN_CHAINS):
            for link_name, human_id in zip(source_chain, human_chain):
                points[human_id] = self.urdf.get_transform(
                    link_name, "plam_link"
                )[:3, 3]
        return points

    def calibrate(self, open_qpos_deg: np.ndarray, reference: np.ndarray) -> None:
        """Align a held-open EX16 pose with an open frame from a human dataset."""
        samples = np.asarray(open_qpos_deg, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 16 or samples.shape[0] == 0:
            raise ValueError(
                f"Expected calibration joint angles with shape [N, 16], got {samples.shape}"
            )
        if not np.isfinite(samples).all():
            raise ValueError("Calibration joint angles contain NaN or Inf")

        frame_index = select_open_reference_frame(reference)
        reference_frame = np.asarray(reference[frame_index], dtype=np.float64)
        reference_frame = reference_frame - reference_frame[0]
        calibration_qpos = np.median(samples, axis=0)
        source_open = self.source_keypoints(calibration_qpos)
        scale, rotation = fit_origin_similarity(
            source_open[TIP_IDS], reference_frame[TIP_IDS]
        )
        open_projected = source_open @ rotation * scale
        errors = open_projected[TIP_IDS] - reference_frame[TIP_IDS]

        self.scale = scale
        self.rotation = rotation
        self.reference_frame = reference_frame
        self.reference_frame_index = frame_index
        self.open_projected = open_projected
        self.calibration_qpos_deg = calibration_qpos
        self.calibration_rmse_m = float(np.sqrt(np.mean(errors * errors)))

    def restore_calibration(
        self,
        calibration_qpos_deg: Sequence[float],
        scale: float,
        rotation: np.ndarray,
        reference: np.ndarray,
        reference_frame_index: int,
    ) -> None:
        """Restore a calibration saved in an EX16 raw recording."""
        calibration_qpos = self._validate_qpos(calibration_qpos_deg)
        scale = float(scale)
        rotation = np.asarray(rotation, dtype=np.float64)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Calibration scale must be positive and finite, got {scale}")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("Calibration rotation must be a finite [3, 3] matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Calibration rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("Calibration rotation must be a proper rotation")
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 3 or reference.shape[1:] != (21, 3):
            raise ValueError(f"Expected reference shape [T, 21, 3], got {reference.shape}")
        if not 0 <= reference_frame_index < len(reference):
            raise ValueError(
                f"Reference frame {reference_frame_index} is outside [0, {len(reference)})"
            )

        reference_frame = reference[reference_frame_index] - reference[reference_frame_index, 0]
        source_open = self.source_keypoints(calibration_qpos)
        open_projected = source_open @ rotation * scale
        errors = open_projected[TIP_IDS] - reference_frame[TIP_IDS]

        self.scale = scale
        self.rotation = rotation
        self.reference_frame = reference_frame
        self.reference_frame_index = int(reference_frame_index)
        self.open_projected = open_projected
        self.calibration_qpos_deg = calibration_qpos
        self.calibration_rmse_m = float(np.sqrt(np.mean(errors * errors)))

    def project(self, qpos_deg: Sequence[float]) -> np.ndarray:
        """Return one finite ``[21, 3]`` frame in GeoRT coordinates."""
        if self.scale is None or self.rotation is None:
            raise RuntimeError("EX16HumanProjector must be calibrated before projection")
        points = self.source_keypoints(qpos_deg) @ self.rotation * self.scale
        points[0] = 0.0

        # EX16 does not measure a pinky.  Move the reference pinky by the ring
        # chain displacement so standard 21-point tools can still load/display
        # the file. Allegro and GX16 training only select IDs 4/8/12/16.
        ring_delta = points[13:17] - self.open_projected[13:17]
        points[17:21] = self.reference_frame[17:21] + ring_delta
        return points.astype(np.float32)


def decode_state_message(message: str, topic: str = DEFAULT_TOPIC) -> dict[str, Any]:
    """Decode and validate one EX16 ZMQ state message."""
    try:
        received_topic, encoded = message.split(" ", 1)
    except ValueError as exc:
        raise ValueError("EX16 message must contain '<topic> <json>'") from exc
    if received_topic != topic:
        raise ValueError(f"Expected topic {topic!r}, got {received_topic!r}")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid EX16 state JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("EX16 state payload must be a JSON object")
    qpos = np.asarray(payload.get("urdf_deg"), dtype=np.float64)
    if qpos.shape != (16,) or not np.isfinite(qpos).all():
        raise ValueError("EX16 state urdf_deg must contain 16 finite values")
    timestamp = float(payload.get("timestamp", math.nan))
    if not math.isfinite(timestamp):
        raise ValueError("EX16 state timestamp must be finite")
    return {
        "qpos_deg": qpos,
        "timestamp": timestamp,
        "sequence": int(payload.get("sequence", -1)),
    }


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as file:
            temporary = Path(file.name)
            np.save(file, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
            temporary = Path(file.name)
            np.savez(file, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def wait_for_state(socket: Any, topic: str, timeout_s: float) -> dict[str, Any]:
    """Wait for a valid state, ignoring malformed messages until timeout."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        remaining_ms = max(1, math.ceil((deadline - time.monotonic()) * 1000))
        if not socket.poll(min(remaining_ms, 200)):
            continue
        try:
            return decode_state_message(socket.recv_string(), topic)
        except (TypeError, ValueError) as exc:
            last_error = exc
    detail = f" Last invalid message: {last_error}" if last_error else ""
    raise TimeoutError(f"No valid EX16 state received within {timeout_s:g} seconds.{detail}")


def collect_for_duration(socket: Any, topic: str, duration_s: float, timeout_s: float) -> list[dict[str, Any]]:
    samples = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            sample = wait_for_state(socket, topic, min(timeout_s, remaining))
        except TimeoutError:
            if samples and time.monotonic() >= deadline:
                break
            raise
        samples.append(sample)
    return samples


def countdown(seconds: float, label: str) -> None:
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    shown = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        value = math.ceil(remaining)
        if value != shown:
            print(f"{label}: {value}", flush=True)
            shown = value
        time.sleep(min(0.05, remaining))


def record_fixed_rate(
    socket: Any,
    projector: EX16HumanProjector,
    topic: str,
    fps: float,
    frames: int,
    timeout_s: float,
    initial_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Record exactly ``frames`` latest-value samples at ``fps``."""
    human = np.empty((frames, 21, 3), dtype=np.float32)
    qpos = np.empty((frames, 16), dtype=np.float64)
    source_timestamps = np.empty(frames, dtype=np.float64)
    receive_timestamps = np.empty(frames, dtype=np.float64)
    sequences = np.empty(frames, dtype=np.int64)
    latest = initial_state
    last_received = time.monotonic()
    started = time.monotonic()

    for index in range(frames):
        deadline = started + index / fps
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            poll_ms = max(1, min(20, math.ceil(remaining * 1000)))
            if socket.poll(poll_ms):
                try:
                    latest = decode_state_message(socket.recv_string(), topic)
                except (TypeError, ValueError) as exc:
                    print(f"Warning: ignoring invalid EX16 message: {exc}", file=sys.stderr)
                else:
                    last_received = time.monotonic()
            if time.monotonic() - last_received > timeout_s:
                raise TimeoutError(f"EX16 stream was silent for more than {timeout_s:g} seconds")

        qpos[index] = latest["qpos_deg"]
        source_timestamps[index] = latest["timestamp"]
        receive_timestamps[index] = time.time()
        sequences[index] = latest["sequence"]
        human[index] = projector.project(latest["qpos_deg"])
        if (index + 1) % max(1, round(fps * 5)) == 0 or index + 1 == frames:
            print(f"Recorded {index + 1}/{frames} frames", flush=True)

    return human, qpos, source_timestamps, receive_timestamps, sequences


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record EX16 as GeoRT-compatible human hand keypoints."
    )
    parser.add_argument("--name", default="human_ex16", help="Output dataset stem.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-prompt", action="store_true", help="Start without waiting for Enter."
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and greater than zero")
    if args.frames <= 0:
        parser.error("--frames must be greater than zero")
    for name in ("calibration_seconds", "countdown"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and greater than zero")
    stem = Path(args.name).name
    if stem.endswith(".npy"):
        stem = stem[:-4]
    if not stem or stem in {".", ".."} or any(char in stem for char in "/\\"):
        parser.error("--name must be a filename stem without directories")
    args.name = stem
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

    reference_path = args.reference.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    human_path = output_dir / f"{args.name}.npy"
    raw_path = output_dir / f"{args.name}_ex16_raw.npz"
    if human_path.exists() or raw_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing recording: {human_path} or {raw_path}"
        )

    reference = load_reference_hand(reference_path)
    projector = EX16HumanProjector(args.urdf)
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.linger = 0
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    socket.connect(args.endpoint)

    duration = args.frames / args.fps
    print(
        f"Target: {args.frames} frames at {args.fps:g} FPS "
        f"(~{duration:.1f} s), output {human_path}",
        flush=True,
    )
    print(f"Waiting for {args.topic} at {args.endpoint} ...", flush=True)
    first_state = wait_for_state(socket, args.topic, args.timeout)

    try:
        if not args.no_prompt:
            input("Wear EX16, fully open and relax the hand, then press Enter to calibrate: ")
        countdown(args.countdown, "Open-hand calibration starts in")
        print("Hold the hand fully open and still...", flush=True)
        calibration_states = collect_for_duration(
            socket, args.topic, args.calibration_seconds, args.timeout
        )
        if not calibration_states:
            calibration_states = [first_state]
        calibration_qpos = np.asarray(
            [state["qpos_deg"] for state in calibration_states], dtype=np.float64
        )
        projector.calibrate(calibration_qpos, reference)
        print(
            f"Calibration: scale={projector.scale:.5f}, "
            f"four-tip RMSE={projector.calibration_rmse_m * 1000:.1f} mm, "
            f"reference frame={projector.reference_frame_index}",
            flush=True,
        )

        countdown(args.countdown, "Recording starts in")
        initial_state = wait_for_state(socket, args.topic, args.timeout)
        print("Recording. Move through the intended hand workspace...", flush=True)
        recorded = record_fixed_rate(
            socket,
            projector,
            args.topic,
            args.fps,
            args.frames,
            args.timeout,
            initial_state,
        )
    finally:
        socket.close()

    human, qpos, source_timestamps, receive_timestamps, sequences = recorded
    if human.shape != (args.frames, 21, 3) or not np.isfinite(human).all():
        raise RuntimeError(f"Invalid projected recording: shape={human.shape}")

    metadata = {
        "format": "libgex.ex16_geort_recording",
        "version": 1,
        "fps": args.fps,
        "frames": args.frames,
        "duration_seconds": duration,
        "units": {"qpos": "degree", "keypoints": "metre"},
        "keypoint_order": "MediaPipe 21",
        "tracked_tip_ids": TIP_IDS.tolist(),
        "synthetic_pinky_ids": [17, 18, 19, 20],
        "reference_path": str(reference_path),
        "reference_frame": projector.reference_frame_index,
        "urdf_path": str(projector.urdf_path),
        "calibration_rmse_m": projector.calibration_rmse_m,
    }
    atomic_save_npy(human_path, human)
    atomic_save_npz(
        raw_path,
        qpos_deg=qpos,
        source_timestamp=source_timestamps,
        receive_timestamp=receive_timestamps,
        sequence=sequences,
        calibration_qpos_deg=projector.calibration_qpos_deg,
        calibration_scale=np.asarray(projector.scale),
        calibration_rotation=projector.rotation,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"Saved GeoRT data: {human_path} {human.shape} {human.dtype}")
    print(f"Saved raw EX16 data: {raw_path} {qpos.shape}")
    print(
        "Visualize with: cd utils/GeoRT && "
        f"python geort/mocap/visualize_human_data.py --data {args.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
