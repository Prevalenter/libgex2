"""Visualize ZMQ-published EX16 and retargeted GX16 states side by side."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import viser
import viser.transforms as tf
import yourdfpy
from viser.extras import ViserUrdf


PROJECT_ROOT = Path(__file__).resolve().parent
EX16_URDF_PATH = PROJECT_ROOT / "libgex" / "ex16" / "urdf" / "glove4.urdf"
EX16_MESH_DIR = PROJECT_ROOT / "libgex" / "ex16" / "meshes"
GX16_URDF_PATH = PROJECT_ROOT / "libgex" / "gx16" / "urdf" / "gx4m.urdf"
GX16_MESH_DIR = PROJECT_ROOT / "libgex" / "gx16" / "meshes"
DEFAULT_EX16_ENDPOINT = "tcp://127.0.0.1:5567"
DEFAULT_GX16_ENDPOINT = "tcp://127.0.0.1:5568"
DEFAULT_BASE_TRANSFORM_FILE = PROJECT_ROOT / "viewer_layouts" / "ex16_gx16.json"
EX16_TOPIC = "ex16/state"
GX16_TOPIC = "gx16/retarget_state"
JOINT_COUNT = 16
BASE_TRANSFORM_FORMAT = "libgex.ex16_gx16_base_transforms"
BASE_TRANSFORM_VERSION = 1
DEFAULT_BASE_TRANSFORMS = {
    "ex16": {"position": (-0.23, 0.0, 0.0), "wxyz": (1.0, 0.0, 0.0, 0.0)},
    "gx16": {"position": (0.23, 0.0, 0.0), "wxyz": (1.0, 0.0, 0.0, 0.0)},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display published EX16 and virtual GX16 states in Viser."
    )
    parser.add_argument("--ex16-endpoint", default=DEFAULT_EX16_ENDPOINT)
    parser.add_argument("--gx16-endpoint", default=DEFAULT_GX16_ENDPOINT)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--base-transform-file",
        type=Path,
        default=DEFAULT_BASE_TRANSFORM_FILE,
        help="JSON file used to save and restore EX16/GX16 base transforms",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port-viser", type=int, default=8080)
    return parser.parse_args(argv)


def default_base_transforms() -> dict[str, dict[str, tuple[float, ...]]]:
    """Return a fresh copy of the built-in viewer layout."""
    return {
        name: {
            "position": tuple(transform["position"]),
            "wxyz": tuple(transform["wxyz"]),
        }
        for name, transform in DEFAULT_BASE_TRANSFORMS.items()
    }


def _finite_vector(value: Any, length: int, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only numbers") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must contain {length} finite numbers")
    return vector


def validate_base_transforms(payload: Any) -> dict[str, dict[str, tuple[float, ...]]]:
    """Validate a versioned base-transform document and normalize quaternions."""
    if not isinstance(payload, dict):
        raise ValueError("base transform file must contain a JSON object")
    if payload.get("format") != BASE_TRANSFORM_FORMAT:
        raise ValueError(f"unsupported format; expected {BASE_TRANSFORM_FORMAT}")
    if payload.get("version") != BASE_TRANSFORM_VERSION:
        raise ValueError(f"unsupported version; expected {BASE_TRANSFORM_VERSION}")

    transforms: dict[str, dict[str, tuple[float, ...]]] = {}
    for name in ("ex16", "gx16"):
        transform = payload.get(name)
        if not isinstance(transform, dict):
            raise ValueError(f"{name} must be a JSON object")
        position = _finite_vector(transform.get("position"), 3, f"{name}.position")
        wxyz = _finite_vector(transform.get("wxyz"), 4, f"{name}.wxyz")
        norm = float(np.linalg.norm(wxyz))
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"{name}.wxyz must be a non-zero quaternion")
        transforms[name] = {
            "position": tuple(float(value) for value in position),
            "wxyz": tuple(float(value) for value in wxyz / norm),
        }
    return transforms


def base_transform_document(
    transforms: dict[str, dict[str, Sequence[float]]],
) -> dict[str, Any]:
    if not isinstance(transforms, dict):
        raise ValueError("transforms must be a mapping")
    payload: dict[str, Any] = {
        "format": BASE_TRANSFORM_FORMAT,
        "version": BASE_TRANSFORM_VERSION,
    }
    for name in ("ex16", "gx16"):
        if not isinstance(transforms.get(name), dict):
            raise ValueError(f"transforms must contain {name}")
        payload[name] = {
            "position": transforms[name].get("position"),
            "wxyz": transforms[name].get("wxyz"),
        }
    normalized = validate_base_transforms(payload)
    for name in ("ex16", "gx16"):
        payload[name] = {
            "position": list(normalized[name]["position"]),
            "wxyz": list(normalized[name]["wxyz"]),
        }
    return payload


def load_base_transforms(path: Path) -> dict[str, dict[str, tuple[float, ...]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    return validate_base_transforms(payload)


def save_base_transforms(
    path: Path, transforms: dict[str, dict[str, Sequence[float]]]
) -> None:
    """Atomically persist both base transforms."""
    payload = base_transform_document(transforms)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def rpy_degrees_to_wxyz(rpy_degrees: Sequence[float]) -> tuple[float, ...]:
    rpy = _finite_vector(rpy_degrees, 3, "rpy_degrees")
    rotation = tf.SO3.from_rpy_radians(*np.deg2rad(rpy))
    return tuple(float(value) for value in rotation.wxyz)


def wxyz_to_rpy_degrees(wxyz: Sequence[float]) -> tuple[float, ...]:
    quaternion = _finite_vector(wxyz, 4, "wxyz")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("wxyz must be a non-zero quaternion")
    rpy = tf.SO3(quaternion / norm).as_rpy_radians()
    return tuple(float(value) for value in np.rad2deg(rpy))


def load_urdf(path: Path, mesh_dir: Path) -> yourdfpy.URDF:
    def resolve_mesh(fname: str) -> str:
        return str(mesh_dir / Path(fname).name)

    return yourdfpy.URDF.load(
        path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        filename_handler=resolve_mesh,
    )


def configuration(urdf: yourdfpy.URDF, degrees: np.ndarray) -> np.ndarray:
    if degrees.shape != (JOINT_COUNT,):
        raise ValueError(f"Expected 16 joint values, got shape {degrees.shape}")
    values = {
        f"joint{index}": np.deg2rad(value)
        for index, value in enumerate(degrees, start=1)
    }
    missing = set(values) - set(urdf.actuated_joint_names)
    if missing:
        raise RuntimeError(f"URDF is missing joints: {sorted(missing)}")
    return np.asarray(
        [values[name] for name in urdf.actuated_joint_names], dtype=np.float64
    )


def link_frame_pose(
    urdf: yourdfpy.URDF, link_name: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a URDF link pose relative to the model's base link."""
    if link_name not in urdf.link_map:
        raise ValueError(f"URDF is missing link: {link_name}")
    transform = urdf.get_transform(link_name, urdf.base_link)
    position = tuple(float(value) for value in transform[:3, 3])
    wxyz = tuple(
        float(value) for value in tf.SO3.from_matrix(transform[:3, :3]).wxyz
    )
    return position, wxyz


def configure_camera(server: viser.ViserServer) -> None:
    look_at = (0.0, 0.0, 0.08)
    position = (0.66, -0.78, 0.55)
    up = (0.0, 0.0, 1.0)
    fov = np.deg2rad(48.0)
    if hasattr(server, "initial_camera"):
        server.initial_camera.look_at = look_at
        server.initial_camera.position = position
        server.initial_camera.up = up
        server.initial_camera.fov = fov
    else:
        @server.on_client_connect
        def set_client_camera(client: viser.ClientHandle) -> None:
            client.camera.position = position
            # Viser 0.2.x shifts look_at when position changes, so set the
            # final target only after setting the camera position.
            client.camera.look_at = look_at
            client.camera.up_direction = up
            client.camera.fov = fov


def parse_message(message: str, expected_topic: str) -> dict:
    topic, encoded = message.split(" ", 1)
    if topic != expected_topic:
        raise ValueError(f"Expected topic {expected_topic!r}, got {topic!r}")
    return json.loads(encoded)


def add_base_transform_control(
    server: viser.ViserServer,
    scene_name: str,
    label: str,
    initial: dict[str, Sequence[float]],
) -> dict[str, Any]:
    """Add a 6-DoF gizmo and synchronized numeric inputs for one model root."""
    transform = server.scene.add_transform_controls(
        scene_name,
        scale=0.08,
        line_width=3.0,
        depth_test=False,
        position=initial["position"],
        wxyz=initial["wxyz"],
    )
    with server.gui.add_folder(f"{label} Base"):
        position_input = server.gui.add_vector3(
            "Position (m)",
            initial_value=initial["position"],
            step=0.001,
        )
        rpy_input = server.gui.add_vector3(
            "RPY (deg)",
            initial_value=wxyz_to_rpy_degrees(initial["wxyz"]),
            step=0.1,
        )

    @transform.on_update
    def sync_inputs_from_gizmo(_event) -> None:
        position_input.value = tuple(float(value) for value in transform.position)
        rpy_input.value = wxyz_to_rpy_degrees(transform.wxyz)

    @position_input.on_update
    def sync_position_from_input(event) -> None:
        transform.position = event.target.value

    @rpy_input.on_update
    def sync_rotation_from_input(event) -> None:
        transform.wxyz = rpy_degrees_to_wxyz(event.target.value)

    return {
        "transform": transform,
        "position_input": position_input,
        "rpy_input": rpy_input,
    }


def apply_base_transform(
    control: dict[str, Any], transform: dict[str, Sequence[float]]
) -> None:
    control["transform"].position = transform["position"]
    control["transform"].wxyz = transform["wxyz"]
    control["position_input"].value = tuple(transform["position"])
    control["rpy_input"].value = wxyz_to_rpy_degrees(transform["wxyz"])


def current_base_transforms(
    controls: dict[str, dict[str, Any]],
) -> dict[str, dict[str, tuple[float, ...]]]:
    return {
        name: {
            "position": tuple(
                float(value) for value in controls[name]["transform"].position
            ),
            "wxyz": tuple(
                float(value) for value in controls[name]["transform"].wxyz
            ),
        }
        for name in ("ex16", "gx16")
    }


def main() -> None:
    args = parse_args()
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be greater than zero")
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc

    base_transform_file = args.base_transform_file.expanduser().resolve()
    base_transforms = default_base_transforms()
    startup_status = "Using built-in base transforms; no saved layout found."
    if base_transform_file.exists():
        try:
            base_transforms = load_base_transforms(base_transform_file)
        except (OSError, ValueError) as exc:
            startup_status = f"Saved layout is invalid; using defaults: {exc}"
            print(f"Warning: {startup_status}", flush=True)
        else:
            startup_status = f"Loaded base transforms from {base_transform_file}"
            print(startup_status, flush=True)

    ex16_urdf = load_urdf(EX16_URDF_PATH, EX16_MESH_DIR)
    gx16_urdf = load_urdf(GX16_URDF_PATH, GX16_MESH_DIR)

    server = viser.ViserServer(host=args.host, port=args.port_viser)
    server.scene.set_up_direction("+z")
    configure_camera(server)

    with server.gui.add_folder("Base Coordinate Transforms"):
        controls = {
            "ex16": add_base_transform_control(
                server, "/ex16", "EX16", base_transforms["ex16"]
            ),
            "gx16": add_base_transform_control(
                server, "/gx16", "GX16", base_transforms["gx16"]
            ),
        }
        save_button = server.gui.add_button("Save Base Transforms")
        reload_button = server.gui.add_button("Reload Saved")
        reset_button = server.gui.add_button("Reset Defaults")
        transform_status = server.gui.add_markdown(startup_status)

    @save_button.on_click
    def save_transform_layout(_event) -> None:
        try:
            save_base_transforms(
                base_transform_file, current_base_transforms(controls)
            )
        except (OSError, ValueError) as exc:
            transform_status.content = f"**Save failed:** {exc}"
            print(f"Failed to save base transforms: {exc}", flush=True)
        else:
            transform_status.content = f"Saved to `{base_transform_file}`"
            print(f"Saved base transforms to {base_transform_file}", flush=True)

    @reload_button.on_click
    def reload_transform_layout(_event) -> None:
        try:
            loaded = load_base_transforms(base_transform_file)
        except (OSError, ValueError) as exc:
            transform_status.content = f"**Reload failed:** {exc}"
            print(f"Failed to reload base transforms: {exc}", flush=True)
            return
        for name in ("ex16", "gx16"):
            apply_base_transform(controls[name], loaded[name])
        transform_status.content = f"Reloaded from `{base_transform_file}`"
        print(f"Reloaded base transforms from {base_transform_file}", flush=True)

    @reset_button.on_click
    def reset_transform_layout(_event) -> None:
        defaults = default_base_transforms()
        for name in ("ex16", "gx16"):
            apply_base_transform(controls[name], defaults[name])
        transform_status.content = "Reset to built-in defaults (not saved)."

    ex16_model = ViserUrdf(server, ex16_urdf, root_node_name="/ex16/model")
    gx16_model = ViserUrdf(server, gx16_urdf, root_node_name="/gx16/model")
    for model_name, urdf in (("ex16", ex16_urdf), ("gx16", gx16_urdf)):
        plam_position, plam_wxyz = link_frame_pose(urdf, "plam_link")
        server.scene.add_frame(
            f"/{model_name}/plam_link",
            axes_length=0.05,
            axes_radius=0.002,
            position=plam_position,
            wxyz=plam_wxyz,
        )
    ex16_model.update_cfg(configuration(ex16_urdf, np.zeros(JOINT_COUNT)))
    gx16_model.update_cfg(configuration(gx16_urdf, np.zeros(JOINT_COUNT)))

    context = zmq.Context.instance()
    ex16_subscriber = context.socket(zmq.SUB)
    gx16_subscriber = context.socket(zmq.SUB)
    for socket in (ex16_subscriber, gx16_subscriber):
        socket.linger = 0
        socket.setsockopt(zmq.CONFLATE, 1)
    ex16_subscriber.setsockopt_string(zmq.SUBSCRIBE, EX16_TOPIC)
    gx16_subscriber.setsockopt_string(zmq.SUBSCRIBE, GX16_TOPIC)
    ex16_subscriber.connect(args.ex16_endpoint)
    gx16_subscriber.connect(args.gx16_endpoint)

    poller = zmq.Poller()
    poller.register(ex16_subscriber, zmq.POLLIN)
    poller.register(gx16_subscriber, zmq.POLLIN)

    print(f"Viser: http://127.0.0.1:{args.port_viser}", flush=True)
    print(f"EX16: {args.ex16_endpoint}, topic={EX16_TOPIC}", flush=True)
    print(f"GX16: {args.gx16_endpoint}, topic={GX16_TOPIC}", flush=True)
    print(f"Base transforms: {base_transform_file}", flush=True)

    last_received = {EX16_TOPIC: None, GX16_TOPIC: None}
    last_sequence = {EX16_TOPIC: None, GX16_TOPIC: None}
    last_degrees = {EX16_TOPIC: None, GX16_TOPIC: None}
    last_warning = 0.0
    last_status = 0.0
    try:
        while True:
            events = dict(poller.poll(100))
            now = time.monotonic()

            if ex16_subscriber in events:
                payload = parse_message(ex16_subscriber.recv_string(), EX16_TOPIC)
                degrees = np.asarray(payload["urdf_deg"], dtype=np.float64)
                ex16_model.update_cfg(configuration(ex16_urdf, degrees))
                last_received[EX16_TOPIC] = now
                last_sequence[EX16_TOPIC] = payload.get("sequence")
                last_degrees[EX16_TOPIC] = degrees

            if gx16_subscriber in events:
                payload = parse_message(gx16_subscriber.recv_string(), GX16_TOPIC)
                field = next(
                    (
                        name
                        for name in ("commanded_urdf_deg", "desired_urdf_deg")
                        if name in payload
                    ),
                    None,
                )
                if field is None:
                    raise ValueError(
                        "GX16 payload has no commanded_urdf_deg or desired_urdf_deg"
                    )
                degrees = np.asarray(payload[field], dtype=np.float64)
                gx16_model.update_cfg(configuration(gx16_urdf, degrees))
                last_received[GX16_TOPIC] = now
                last_sequence[GX16_TOPIC] = payload.get("sequence")
                last_degrees[GX16_TOPIC] = degrees

            stale = [
                topic
                for topic, received_at in last_received.items()
                if received_at is None or now - received_at > args.timeout_s
            ]
            if stale and now - last_warning >= args.timeout_s:
                print(f"Waiting for: {', '.join(stale)}", flush=True)
                last_warning = now
            if not stale and now - last_status >= 1.0:
                ex16_first = np.round(last_degrees[EX16_TOPIC][:4], 1).tolist()
                gx16_first = np.round(last_degrees[GX16_TOPIC][:4], 1).tolist()
                print(
                    f"Live: EX16 seq={last_sequence[EX16_TOPIC]} "
                    f"j1-4={ex16_first}; GX16 seq={last_sequence[GX16_TOPIC]} "
                    f"j1-4={gx16_first}",
                    flush=True,
                )
                last_status = now
    except KeyboardInterrupt:
        print("Stopping EX16 + virtual GX16 viewer...", flush=True)
    finally:
        ex16_subscriber.close()
        gx16_subscriber.close()


if __name__ == "__main__":
    main()
