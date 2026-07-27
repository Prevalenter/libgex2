"""Replay a recorded EX16 joint trajectory in a Viser web viewer."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import viser
from viser.extras import ViserUrdf

from demo_ex16_viser import JOINT_COUNT, load_urdf


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "utils" / "GeoRT" / "data"


def resolve_recording(recording: str | Path | None) -> Path:
    """Resolve a direct path, dataset name, or the newest EX16 recording."""
    if recording is None:
        candidates = list(DEFAULT_DATA_DIR.glob("*_ex16_raw.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No *_ex16_raw.npz recordings found in {DEFAULT_DATA_DIR}"
            )
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    requested = Path(recording).expanduser()
    if requested.is_file():
        return requested.resolve()

    filename = requested.name
    if not filename.endswith(".npz"):
        if not filename.endswith("_ex16_raw"):
            filename += "_ex16_raw"
        filename += ".npz"
    candidate = DEFAULT_DATA_DIR / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            f"EX16 recording not found: {recording!r}; expected a file or {candidate}"
        )
    return candidate.resolve()


def load_recording(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load finite ``[T, 16]`` joint angles and non-pickled metadata."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "qpos_deg" not in archive.files:
                raise ValueError("missing qpos_deg")
            qpos_deg = np.asarray(archive["qpos_deg"], dtype=np.float64)
            metadata = {}
            if "metadata_json" in archive.files:
                metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load EX16 recording from {path}: {exc}") from exc

    if qpos_deg.ndim != 2 or qpos_deg.shape[1] != JOINT_COUNT or len(qpos_deg) == 0:
        raise ValueError(
            f"Expected qpos_deg with shape [T, {JOINT_COUNT}], got {qpos_deg.shape}"
        )
    if not np.isfinite(qpos_deg).all():
        raise ValueError(f"qpos_deg contains NaN or Inf in {path}")
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata_json must decode to an object in {path}")
    return qpos_deg, metadata


def qpos_for_urdf(urdf: Any, degrees: np.ndarray) -> np.ndarray:
    """Reorder joint1..joint16 degrees into the URDF actuated-joint order."""
    degrees = np.asarray(degrees, dtype=np.float64)
    if degrees.shape != (JOINT_COUNT,) or not np.isfinite(degrees).all():
        raise ValueError(f"Expected {JOINT_COUNT} finite joint values, got {degrees.shape}")
    by_name = {
        f"joint{index}": np.deg2rad(value)
        for index, value in enumerate(degrees, start=1)
    }
    missing = set(by_name) - set(urdf.actuated_joint_names)
    if missing:
        raise ValueError(f"EX16 URDF is missing joints: {sorted(missing)}")
    return np.asarray([by_name[name] for name in urdf.actuated_joint_names])


def configure_camera(server: viser.ViserServer, urdf: Any) -> None:
    center = np.asarray(urdf.scene.bounds).mean(axis=0)
    size = max(float(np.linalg.norm(urdf.scene.extents)), 0.1)
    direction = np.asarray((1.0, -1.0, 0.7))
    direction /= np.linalg.norm(direction)
    look_at = tuple(float(value) for value in center)
    position = tuple(float(value) for value in center + direction * size * 1.15)
    up = (0.0, 0.0, 1.0)
    fov = np.deg2rad(50.0)
    if hasattr(server, "initial_camera"):
        server.initial_camera.look_at = look_at
        server.initial_camera.position = position
        server.initial_camera.up = up
        server.initial_camera.fov = fov
    else:
        @server.on_client_connect
        def set_client_camera(client: viser.ClientHandle) -> None:
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = up
            client.camera.fov = fov


class EX16RecordingViewer:
    """Interactive EX16 mesh playback controls backed by a recorded qpos array."""

    def __init__(
        self,
        server: viser.ViserServer,
        urdf: Any,
        recording_path: Path,
        qpos_deg: np.ndarray,
        fps: float,
    ) -> None:
        self.server = server
        self.urdf = urdf
        self.recording_path = recording_path
        self.qpos_deg = qpos_deg
        self.fps = fps
        self.frame_index = 0
        self.playing = True
        self.loop = True
        self._updating_slider = False
        self.next_frame_time = time.monotonic() + 1.0 / fps

        server.scene.set_up_direction("+z")
        server.scene.add_frame("/coordinates", axes_length=0.06, axes_radius=0.002)
        self.robot = ViserUrdf(server, urdf, root_node_name="/ex16")
        configure_camera(server, urdf)

        with server.gui.add_folder("Recording"):
            server.gui.add_markdown(
                f"**{recording_path.name}**  \n"
                f"Frames: `{len(qpos_deg):,}`  \n"
                f"Playback rate: `{fps:g} FPS`"
            )
            self.frame_slider = server.gui.add_slider(
                "Frame",
                min=0,
                max=len(qpos_deg) - 1,
                step=1,
                initial_value=0,
            )
            self.play_button = server.gui.add_button("Play / pause")
            self.previous_button = server.gui.add_button("Previous frame")
            self.next_button = server.gui.add_button("Next frame")
            self.reset_button = server.gui.add_button("Reset")
            self.loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)
            self.fps_input = server.gui.add_number(
                "Playback FPS", initial_value=fps, min=0.1, max=120.0, step=0.1
            )
            self.status = server.gui.add_markdown("")

        @self.frame_slider.on_update
        def update_from_slider(event: Any) -> None:
            if self._updating_slider:
                return
            self.playing = False
            self.set_frame(int(event.target.value), update_slider=False)

        @self.play_button.on_click
        def toggle_playback(_event: Any) -> None:
            self.playing = not self.playing
            self.next_frame_time = time.monotonic() + 1.0 / self.fps
            self.update_status()

        @self.previous_button.on_click
        def previous_frame(_event: Any) -> None:
            self.playing = False
            self.set_frame(self.frame_index - 1)

        @self.next_button.on_click
        def next_frame(_event: Any) -> None:
            self.playing = False
            self.set_frame(self.frame_index + 1)

        @self.reset_button.on_click
        def reset(_event: Any) -> None:
            self.playing = False
            self.set_frame(0)

        @self.loop_checkbox.on_update
        def update_loop(event: Any) -> None:
            self.loop = bool(event.target.value)

        @self.fps_input.on_update
        def update_fps(event: Any) -> None:
            self.fps = float(event.target.value)
            self.next_frame_time = time.monotonic() + 1.0 / self.fps
            self.update_status()

        self.set_frame(0)

    def update_status(self) -> None:
        state = "Playing" if self.playing else "Paused"
        joint_text = np.array2string(
            self.qpos_deg[self.frame_index],
            precision=1,
            separator=", ",
            max_line_width=88,
        )
        self.status.content = (
            f"**{state}** — frame `{self.frame_index + 1}/{len(self.qpos_deg)}`  \n"
            f"Joint angles (degree):  \n```text\n{joint_text}\n```"
        )

    def set_frame(self, frame_index: int, update_slider: bool = True) -> None:
        if self.loop:
            frame_index %= len(self.qpos_deg)
        else:
            frame_index = min(max(frame_index, 0), len(self.qpos_deg) - 1)
        self.frame_index = int(frame_index)
        self.robot.update_cfg(qpos_for_urdf(self.urdf, self.qpos_deg[self.frame_index]))
        if update_slider:
            self._updating_slider = True
            self.frame_slider.value = self.frame_index
            self._updating_slider = False
        self.update_status()

    def tick(self) -> None:
        if not self.playing:
            return
        now = time.monotonic()
        if now < self.next_frame_time:
            return
        if self.frame_index == len(self.qpos_deg) - 1 and not self.loop:
            self.playing = False
            self.update_status()
            return
        self.set_frame(self.frame_index + 1)
        self.next_frame_time = now + 1.0 / self.fps


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay recorded EX16 states in Viser.")
    parser.add_argument(
        "--recording",
        help="Raw NPZ path/name; defaults to the newest *_ex16_raw.npz recording.",
    )
    parser.add_argument("--fps", type=float, help="Override metadata playback FPS.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port-viser", type=int, default=8080)
    args = parser.parse_args(argv)
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and greater than zero")
    if not 0 <= args.port_viser <= 65535:
        parser.error("--port-viser must be between 0 and 65535")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    recording_path = resolve_recording(args.recording)
    qpos_deg, metadata = load_recording(recording_path)
    fps = args.fps if args.fps is not None else float(metadata.get("fps", 10.0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Recording metadata contains invalid FPS: {fps}")

    urdf = load_urdf()
    server = viser.ViserServer(host=args.host, port=args.port_viser)
    viewer = EX16RecordingViewer(server, urdf, recording_path, qpos_deg, fps)
    print(
        f"Loaded {recording_path}: {len(qpos_deg)} frames at {fps:g} FPS\n"
        f"Open http://127.0.0.1:{args.port_viser} (Ctrl+C to stop)",
        flush=True,
    )
    try:
        while True:
            viewer.tick()
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Stopping EX16 recording viewer...", flush=True)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
