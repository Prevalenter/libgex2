"""Persistent storage shared by the EX16 and GX16 anchor collectors."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT = "libgex.ex16_gx16_anchor"
VERSION = 1
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 17))
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ExportReport:
    paired_names: tuple[str, ...]
    exported_names: tuple[str, ...]
    ex16_only: tuple[str, ...]
    gx16_only: tuple[str, ...]
    missing_human_keypoints: tuple[str, ...]
    output_path: Path


def validate_anchor_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError("anchor name must not be empty")
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "anchor name may contain only letters, numbers, '_' and '-'"
        )
    return name


def validate_qpos(values: Sequence[float], field: str = "qpos_urdf_deg") -> np.ndarray:
    try:
        qpos = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only numbers") from exc
    if qpos.shape != (16,) or not np.isfinite(qpos).all():
        raise ValueError(f"{field} must contain 16 finite values")
    return qpos


def validate_human_keypoints(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("human_keypoints must contain only numbers") from exc
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("human_keypoints must have shape [21, 3] and be finite")
    return points


def device_directory(data_dir: Path, device: str) -> Path:
    if device not in {"ex16", "gx16"}:
        raise ValueError(f"unsupported anchor device: {device!r}")
    return Path(data_dir).expanduser().resolve() / device


def anchor_path(data_dir: Path, device: str, name: str) -> Path:
    return device_directory(data_dir, device) / f"{validate_anchor_name(name)}.json"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_anchor_document(
    *,
    device: str,
    name: str,
    qpos_urdf_deg: Sequence[float],
    captured_at: float,
    notes: str = "",
    source: Mapping[str, Any] | None = None,
    human_keypoints: Any = None,
    calibration: Mapping[str, Any] | None = None,
    collision: bool | None = None,
) -> dict[str, Any]:
    if device not in {"ex16", "gx16"}:
        raise ValueError(f"unsupported anchor device: {device!r}")
    name = validate_anchor_name(name)
    qpos = validate_qpos(qpos_urdf_deg)
    timestamp = float(captured_at)
    if not np.isfinite(timestamp):
        raise ValueError("captured_at must be finite")
    points = validate_human_keypoints(human_keypoints)
    if device == "gx16" and points is not None:
        raise ValueError("human_keypoints are only valid for EX16 anchors")
    document: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "device": device,
        "name": name,
        "captured_at": timestamp,
        "joint_names": list(JOINT_NAMES),
        "qpos_urdf_deg": qpos.tolist(),
        "notes": str(notes),
        "source": dict(source or {}),
    }
    if device == "ex16":
        document["human_keypoints"] = None if points is None else points.tolist()
        document["calibration"] = None if calibration is None else dict(calibration)
    else:
        document["qpos_urdf_rad"] = np.deg2rad(qpos).tolist()
        document["self_collision"] = collision
    return document


def validate_anchor_document(payload: Any, expected_device: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("anchor file must contain a JSON object")
    if payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError(f"unsupported anchor format; expected {FORMAT} version {VERSION}")
    device = payload.get("device")
    if device not in {"ex16", "gx16"}:
        raise ValueError(f"invalid anchor device: {device!r}")
    if expected_device is not None and device != expected_device:
        raise ValueError(f"expected a {expected_device} anchor, got {device!r}")
    name = validate_anchor_name(payload.get("name", ""))
    if payload.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("joint_names must be joint1 through joint16 in order")
    qpos = validate_qpos(payload.get("qpos_urdf_deg"))
    captured_at = float(payload.get("captured_at"))
    if not np.isfinite(captured_at):
        raise ValueError("captured_at must be finite")
    normalized = dict(payload)
    normalized["name"] = name
    normalized["qpos_urdf_deg"] = qpos.tolist()
    if device == "ex16":
        points = validate_human_keypoints(payload.get("human_keypoints"))
        normalized["human_keypoints"] = None if points is None else points.tolist()
    return normalized


def save_anchor(data_dir: Path, document: Mapping[str, Any], overwrite: bool = False) -> Path:
    normalized = validate_anchor_document(dict(document))
    path = anchor_path(data_dir, normalized["device"], normalized["name"])
    if path.exists() and not overwrite:
        raise FileExistsError(f"anchor already exists: {path}")
    _atomic_json(path, normalized)
    return path


def load_anchor(data_dir: Path, device: str, name: str) -> dict[str, Any]:
    path = anchor_path(data_dir, device, name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid anchor JSON in {path}: {exc}") from exc
    normalized = validate_anchor_document(payload, device)
    if normalized["name"] != path.stem:
        raise ValueError(f"anchor name {normalized['name']!r} does not match {path.name}")
    return normalized


def list_anchor_names(data_dir: Path, device: str) -> tuple[str, ...]:
    directory = device_directory(data_dir, device)
    if not directory.exists():
        return ()
    names = []
    for path in directory.glob("*.json"):
        try:
            validate_anchor_name(path.stem)
        except ValueError:
            continue
        names.append(path.stem)
    return tuple(sorted(names))


def delete_anchor(data_dir: Path, device: str, name: str) -> Path:
    path = anchor_path(data_dir, device, name)
    path.unlink()
    return path


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as file:
            temporary = Path(file.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def rebuild_paired_dataset(data_dir: Path) -> ExportReport:
    """Rebuild the training-ready NPZ from valid, same-name device anchors."""
    data_dir = Path(data_dir).expanduser().resolve()
    ex16_names = set(list_anchor_names(data_dir, "ex16"))
    gx16_names = set(list_anchor_names(data_dir, "gx16"))
    paired_names = tuple(sorted(ex16_names & gx16_names))
    exported: list[str] = []
    missing_points: list[str] = []
    ex16_rows: list[np.ndarray] = []
    gx16_rows: list[np.ndarray] = []
    point_rows: list[np.ndarray] = []
    for name in paired_names:
        ex16 = load_anchor(data_dir, "ex16", name)
        gx16 = load_anchor(data_dir, "gx16", name)
        points = validate_human_keypoints(ex16.get("human_keypoints"))
        if points is None:
            missing_points.append(name)
            continue
        exported.append(name)
        ex16_rows.append(validate_qpos(ex16["qpos_urdf_deg"]))
        gx16_rows.append(validate_qpos(gx16["qpos_urdf_deg"]))
        point_rows.append(points)

    count = len(exported)
    ex16_array = np.asarray(ex16_rows, dtype=np.float64).reshape(count, 16)
    gx16_deg = np.asarray(gx16_rows, dtype=np.float64).reshape(count, 16)
    human_points = np.asarray(point_rows, dtype=np.float64).reshape(count, 21, 3)
    output_path = data_dir / "paired_anchors.npz"
    metadata = {
        "format": FORMAT,
        "version": VERSION,
        "joint_names": list(JOINT_NAMES),
        "angle_units": {"ex16_qpos_deg": "degree", "gx16_qpos_deg": "degree", "gx16_qpos_rad": "radian"},
        "position_units": {"human_keypoints": "metre"},
    }
    _atomic_npz(
        output_path,
        anchor_names=np.asarray(exported, dtype=str),
        ex16_qpos_deg=ex16_array,
        gx16_qpos_deg=gx16_deg,
        gx16_qpos_rad=np.deg2rad(gx16_deg),
        human_keypoints=human_points,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )
    return ExportReport(
        paired_names=paired_names,
        exported_names=tuple(exported),
        ex16_only=tuple(sorted(ex16_names - gx16_names)),
        gx16_only=tuple(sorted(gx16_names - ex16_names)),
        missing_human_keypoints=tuple(missing_points),
        output_path=output_path,
    )
