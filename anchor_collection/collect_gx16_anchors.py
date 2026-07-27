"""Collect named real or virtual GX16 target anchors in Viser."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
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
    JOINT_NAMES,
    build_anchor_document,
    delete_anchor,
    list_anchor_names,
    load_anchor,
    rebuild_paired_dataset,
    save_anchor,
    validate_anchor_name,
    validate_qpos,
)
from viewer_utils import (  # noqa: E402
    configure_camera,
    dropdown_options,
    joint_limits_deg,
    load_visual_urdf,
    report_markdown,
    urdf_configuration,
)
REAL_MODE = "Real hardware (read-only)"
VIRTUAL_MODE = "Virtual sliders"
DEFAULT_CMD_ENDPOINT = "tcp://127.0.0.1:5556"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_URDF = REPO_ROOT / "libgex" / "gx16" / "urdf" / "gx4m.urdf"
DEFAULT_MESH_DIR = REPO_ROOT / "libgex" / "gx16" / "meshes"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect named real or virtual GX16 target anchors in Viser."
    )
    parser.add_argument("--cmd-endpoint", default=DEFAULT_CMD_ENDPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--poll-hz", type=float, default=10.0)
    parser.add_argument("--request-timeout-ms", type=int, default=500)
    parser.add_argument("--stale-timeout", type=float, default=2.0)
    parser.add_argument("--collision-threshold-mm", type=float, default=0.5)
    parser.add_argument("--no-collision-check", action="store_true")
    parser.add_argument("--start-virtual", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port-viser", type=int, default=8082)
    args = parser.parse_args(argv)
    for name in ("poll_hz", "stale_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.request_timeout_ms <= 0:
        parser.error("--request-timeout-ms must be positive")
    if not math.isfinite(args.collision_threshold_mm) or args.collision_threshold_mm < 0:
        parser.error("--collision-threshold-mm must be finite and non-negative")
    if not 0 <= args.port_viser <= 65535:
        parser.error("--port-viser must be between 0 and 65535")
    return args


def parse_getjs_response(response: Any) -> tuple[np.ndarray, float]:
    if not isinstance(response, dict):
        raise ValueError("GX16 response must be a JSON object")
    if response.get("ok") is not True:
        raise RuntimeError(str(response.get("error") or "GX16 getjs failed"))
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("GX16 response result must be a JSON object")
    qpos = validate_qpos(result.get("urdf_deg"), "GX16 result.urdf_deg")
    timestamp = float(result.get("timestamp", time.time()))
    if not math.isfinite(timestamp):
        raise ValueError("GX16 result.timestamp must be finite")
    return qpos, timestamp


class GX16ReadClient:
    """A REQ client whose only request is the read-only ``getjs`` command."""

    def __init__(self, context: Any, endpoint: str, timeout_ms: int) -> None:
        self.context = context
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.socket: Any | None = None
        self._reset_socket()

    def _reset_socket(self) -> None:
        import zmq

        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.connect(self.endpoint)

    def read(self) -> tuple[np.ndarray, float]:
        # Deliberately keep this literal request here: this collector must never
        # issue setjs, setj, home or torque commands.
        self.socket.send_json({"cmd": "getjs", "units": "urdf_deg"})
        if not self.socket.poll(self.timeout_ms):
            self._reset_socket()
            raise TimeoutError(
                f"no GX16 getjs response from {self.endpoint} within {self.timeout_ms} ms"
            )
        return parse_getjs_response(self.socket.recv_json())

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None


def collision_label(collision: bool | None) -> str:
    if collision is None:
        return "not checked"
    if collision:
        return "⚠️ **SELF-COLLISION**"
    return "collision-free"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required: python -m pip install pyzmq") from exc
    # SAPIEN/GeoRT are only required when starting the actual viewer, not for
    # response/schema validation or ``--help``.
    from geort import get_config
    from geort.env.hand import HandKinematicModel

    data_dir = args.data_dir.expanduser().resolve()
    urdf = load_visual_urdf(args.urdf, args.mesh_dir)
    lower_deg, upper_deg = joint_limits_deg(urdf)
    virtual_qpos = np.clip(np.zeros(16), lower_deg, upper_deg)

    collision_model = None
    if not args.no_collision_check:
        collision_config = dict(get_config("gx16"))
        collision_config["urdf_path"] = str(
            (GEORT_ROOT / collision_config["urdf_path"]).resolve()
        )
        collision_model = HandKinematicModel.build_from_config(collision_config)

    server = viser.ViserServer(host=args.host, port=args.port_viser)
    server.scene.set_up_direction("+z")
    configure_camera(server)
    robot = ViserUrdf(server, urdf, root_node_name="/gx16")
    robot.update_cfg(urdf_configuration(urdf, virtual_qpos))

    initial_mode = VIRTUAL_MODE if args.start_virtual else REAL_MODE
    with server.gui.add_folder("GX16 pose source"):
        mode = server.gui.add_dropdown(
            "Mode", (REAL_MODE, VIRTUAL_MODE), initial_value=initial_mode
        )
        source_status = server.gui.add_markdown("Waiting for GX16 hardware...")
        joint_status = server.gui.add_markdown("")
    sliders = []
    with server.gui.add_folder("Virtual GX16 joints"):
        for index, name in enumerate(JOINT_NAMES):
            sliders.append(
                server.gui.add_slider(
                    f"{name} (deg)",
                    min=float(lower_deg[index]),
                    max=float(upper_deg[index]),
                    step=0.1,
                    initial_value=float(virtual_qpos[index]),
                    disabled=initial_mode == REAL_MODE,
                )
            )
    with server.gui.add_folder("Anchor capture"):
        anchor_name = server.gui.add_text("Anchor name", initial_value="")
        notes = server.gui.add_text("Notes", initial_value="", multiline=True)
        allow_collision = server.gui.add_checkbox(
            "Allow saving self-collision", initial_value=False
        )
        allow_replace = server.gui.add_checkbox("Confirm overwrite / delete", initial_value=False)
        save_button = server.gui.add_button("Save current GX16 anchor")
        saved_dropdown = server.gui.add_dropdown(
            "Saved GX16 anchors", dropdown_options(list_anchor_names(data_dir, "gx16"))
        )
        load_button = server.gui.add_button("Load selected anchor into virtual mode")
        delete_button = server.gui.add_button("Delete selected anchor")
        action_status = server.gui.add_markdown("Ready.")
        dataset_status = server.gui.add_markdown(report_markdown(rebuild_paired_dataset(data_dir)))

    lock = threading.RLock()
    latest_real: np.ndarray | None = None
    latest_source_timestamp: float | None = None
    last_received: float | None = None
    current_collision: bool | None = None

    def set_sliders_enabled(enabled: bool) -> None:
        for slider in sliders:
            slider.disabled = not enabled

    def set_virtual_pose(qpos_deg: Sequence[float]) -> None:
        nonlocal virtual_qpos
        qpos = np.clip(validate_qpos(qpos_deg), lower_deg, upper_deg)
        with lock:
            virtual_qpos = qpos.copy()
        for slider, value in zip(sliders, qpos):
            slider.value = float(value)
        robot.update_cfg(urdf_configuration(urdf, qpos))

    def selected_pose() -> tuple[np.ndarray, str, float | None]:
        with lock:
            if mode.value == REAL_MODE:
                if latest_real is None or last_received is None:
                    raise RuntimeError("GX16 hardware has no valid getjs reading")
                if time.monotonic() - last_received > args.stale_timeout:
                    raise RuntimeError("GX16 hardware reading is stale")
                return latest_real.copy(), "hardware_getjs", latest_source_timestamp
            return virtual_qpos.copy(), "virtual_sliders", None

    def check_collision(qpos_deg: Sequence[float]) -> bool | None:
        if collision_model is None:
            return None
        return bool(
            collision_model.has_self_collision(
                np.deg2rad(validate_qpos(qpos_deg)),
                penetration_threshold=args.collision_threshold_mm / 1000.0,
            )
        )

    def refresh_dataset() -> None:
        names = list_anchor_names(data_dir, "gx16")
        saved_dropdown.options = dropdown_options(names)
        if saved_dropdown.value not in saved_dropdown.options:
            saved_dropdown.value = saved_dropdown.options[0]
        dataset_status.content = report_markdown(rebuild_paired_dataset(data_dir))

    @mode.on_update
    def change_mode(event: Any) -> None:
        nonlocal virtual_qpos
        is_virtual = event.target.value == VIRTUAL_MODE
        if is_virtual:
            with lock:
                if latest_real is not None:
                    virtual_qpos = np.clip(latest_real.copy(), lower_deg, upper_deg)
            set_virtual_pose(virtual_qpos)
        set_sliders_enabled(is_virtual)

    for slider_index, slider in enumerate(sliders):
        @slider.on_update
        def update_virtual(event: Any, index: int = slider_index) -> None:
            nonlocal virtual_qpos
            if mode.value != VIRTUAL_MODE:
                return
            with lock:
                virtual_qpos[index] = float(event.target.value)
                qpos = virtual_qpos.copy()
            robot.update_cfg(urdf_configuration(urdf, qpos))

    @save_button.on_click
    def save_current(_event: Any) -> None:
        try:
            name = validate_anchor_name(anchor_name.value)
            qpos, source_mode, source_timestamp = selected_pose()
            collision = check_collision(qpos)
            if collision and not allow_collision.value:
                raise RuntimeError(
                    "pose has self-collision; adjust it or explicitly enable collision override"
                )
            document = build_anchor_document(
                device="gx16",
                name=name,
                qpos_urdf_deg=qpos,
                captured_at=time.time(),
                notes=notes.value,
                source={
                    "mode": source_mode,
                    "endpoint": args.cmd_endpoint if source_mode == "hardware_getjs" else None,
                    "source_timestamp": source_timestamp,
                },
                collision=collision,
            )
            path = save_anchor(data_dir, document, overwrite=bool(allow_replace.value))
            refresh_dataset()
            saved_dropdown.value = name
            action_status.content = f"Saved `{name}` to `{path}`; safety: {collision_label(collision)}"
        except Exception as exc:
            action_status.content = f"**Save failed:** {exc}"

    @load_button.on_click
    def load_selected(_event: Any) -> None:
        try:
            if saved_dropdown.value == "(none)":
                raise RuntimeError("no GX16 anchors saved")
            document = load_anchor(data_dir, "gx16", saved_dropdown.value)
            mode.value = VIRTUAL_MODE
            set_sliders_enabled(True)
            set_virtual_pose(document["qpos_urdf_deg"])
            anchor_name.value = document["name"]
            notes.value = document.get("notes", "")
            action_status.content = f"Loaded `{document['name']}` into virtual mode."
        except Exception as exc:
            action_status.content = f"**Load failed:** {exc}"

    @delete_button.on_click
    def delete_selected(_event: Any) -> None:
        try:
            if saved_dropdown.value == "(none)":
                raise RuntimeError("no GX16 anchors saved")
            if not allow_replace.value:
                raise RuntimeError("enable 'Confirm overwrite / delete' first")
            name = saved_dropdown.value
            delete_anchor(data_dir, "gx16", name)
            refresh_dataset()
            action_status.content = f"Deleted GX16 anchor `{name}`"
        except Exception as exc:
            action_status.content = f"**Delete failed:** {exc}"

    context = zmq.Context.instance()
    client = GX16ReadClient(context, args.cmd_endpoint, args.request_timeout_ms)
    print(f"GX16 collector: http://127.0.0.1:{args.port_viser}", flush=True)
    print(
        f"Read-only hardware endpoint: {args.cmd_endpoint}; virtual mode never sends commands",
        flush=True,
    )
    next_hardware_poll = 0.0
    last_collision_check = 0.0
    last_gui_update = 0.0
    last_error: str | None = None
    try:
        while True:
            now = time.monotonic()
            if mode.value == REAL_MODE and now >= next_hardware_poll:
                next_hardware_poll = now + 1.0 / args.poll_hz
                try:
                    qpos, source_timestamp = client.read()
                except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                    last_error = str(exc)
                else:
                    received_at = time.monotonic()
                    with lock:
                        latest_real = qpos
                        latest_source_timestamp = source_timestamp
                        last_received = received_at
                    last_error = None
                    robot.update_cfg(urdf_configuration(urdf, qpos))

            if now - last_collision_check >= 0.1:
                try:
                    displayed, _, _ = selected_pose()
                    current_collision = check_collision(displayed)
                except RuntimeError:
                    current_collision = None
                last_collision_check = now

            if now - last_gui_update >= 0.1:
                try:
                    displayed, _, source_timestamp = selected_pose()
                except RuntimeError:
                    displayed = virtual_qpos if mode.value == VIRTUAL_MODE else None
                    source_timestamp = None
                if mode.value == VIRTUAL_MODE:
                    source_status.content = f"Mode: **virtual**  \nSafety: {collision_label(current_collision)}"
                elif displayed is None:
                    source_status.content = f"Mode: **real, read-only**  \n⚠️ `{last_error or 'waiting for getjs'}`"
                else:
                    age = time.monotonic() - last_received
                    source_status.content = (
                        f"Mode: **real, read-only**  \nReceive age: `{age * 1000.0:.0f} ms`  \n"
                        f"Source age: `{max(0.0, time.time() - source_timestamp) * 1000.0:.0f} ms`  \n"
                        f"Safety: {collision_label(current_collision)}"
                    )
                if displayed is not None:
                    joint_status.content = "Joint angles (deg):  \n```text\n" + np.array2string(
                        displayed, precision=1, separator=", ", max_line_width=88
                    ) + "\n```"
                last_gui_update = now
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("Stopping GX16 anchor collector...", flush=True)
    finally:
        client.close()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
