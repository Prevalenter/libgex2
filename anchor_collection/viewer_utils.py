"""Small Viser/URDF helpers shared by both anchor collectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yourdfpy

from anchor_io import JOINT_NAMES, ExportReport


def load_visual_urdf(path: Path, mesh_dir: Path) -> yourdfpy.URDF:
    path = Path(path).expanduser().resolve()
    mesh_dir = Path(mesh_dir).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    def resolve_mesh(fname: str) -> str:
        candidate = Path(fname)
        if candidate.is_file():
            return str(candidate)
        return str(mesh_dir / candidate.name)

    urdf = yourdfpy.URDF.load(
        path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        filename_handler=resolve_mesh,
    )
    missing = set(JOINT_NAMES) - set(urdf.actuated_joint_names)
    if missing:
        raise ValueError(f"URDF is missing joints: {sorted(missing)}")
    return urdf


def urdf_configuration(urdf: Any, qpos_deg: Sequence[float]) -> np.ndarray:
    qpos = np.asarray(qpos_deg, dtype=np.float64)
    if qpos.shape != (16,) or not np.isfinite(qpos).all():
        raise ValueError("qpos_deg must contain 16 finite values")
    by_name = dict(zip(JOINT_NAMES, np.deg2rad(qpos)))
    return np.asarray([by_name[name] for name in urdf.actuated_joint_names])


def joint_limits_deg(urdf: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for name in JOINT_NAMES:
        joint = urdf.joint_map[name]
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise ValueError(f"URDF joint {name} must have finite lower and upper limits")
        lower.append(float(joint.limit.lower))
        upper.append(float(joint.limit.upper))
    lower_array = np.rad2deg(np.asarray(lower, dtype=np.float64))
    upper_array = np.rad2deg(np.asarray(upper, dtype=np.float64))
    if not np.isfinite(lower_array).all() or not np.isfinite(upper_array).all():
        raise ValueError("URDF joint limits must be finite")
    return lower_array, upper_array


def configure_camera(server: Any) -> None:
    look_at = (0.0, 0.0, 0.08)
    position = (0.5, -0.6, 0.42)
    up = (0.0, 0.0, 1.0)
    fov = np.deg2rad(48.0)
    if hasattr(server, "initial_camera"):
        server.initial_camera.look_at = look_at
        server.initial_camera.position = position
        server.initial_camera.up = up
        server.initial_camera.fov = fov
    else:
        @server.on_client_connect
        def set_camera(client: Any) -> None:
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = up
            client.camera.fov = fov


def report_markdown(report: ExportReport) -> str:
    def names(values: Sequence[str]) -> str:
        return ", ".join(values) if values else "none"

    return (
        f"Training-ready pairs: `{len(report.exported_names)}`  \n"
        f"EX16 only: `{names(report.ex16_only)}`  \n"
        f"GX16 only: `{names(report.gx16_only)}`  \n"
        f"Pairs without calibration: `{names(report.missing_human_keypoints)}`  \n"
        f"Dataset: `{report.output_path}`"
    )


def dropdown_options(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(names) if names else ("(none)",)
