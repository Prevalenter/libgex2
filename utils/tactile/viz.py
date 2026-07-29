from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from .uart import TACTILE_AXES, TACTILE_POINTS, raw_to_force_frame

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "L3530.xlsx"
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"

ARROW_SCALE = 0.01
ALPHA_BASE = 0.6
ALPHA_STRONG = 1.0
FORCE_ALPHA_THRESH = 100.0
INACTIVE_RGBA = np.array([0.45, 0.45, 0.45, 0.85], dtype=np.float32)


def resolve_package_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PACKAGE_DIR / value


def load_sensor_xy(config_path: str | Path = DEFAULT_CONFIG) -> np.ndarray:
    df = pd.read_excel(resolve_package_path(config_path))
    return np.array([df["X"].to_list(), df["Y"].to_list()]).T.astype(np.float32)


def ensure_force_frame(data, n_points: int = TACTILE_POINTS) -> np.ndarray:
    arr = np.asarray(data)
    if arr.shape == (n_points, TACTILE_AXES):
        return arr.astype(np.float32, copy=False)
    return raw_to_force_frame(arr.reshape(-1), n_points=n_points)


def load_recording(path: str | Path, n_points: int = TACTILE_POINTS) -> np.ndarray:
    data = np.load(resolve_package_path(path), allow_pickle=False)
    arr = np.asarray(data)

    if arr.ndim == 1:
        return ensure_force_frame(arr, n_points)[None, None, ...]
    if arr.ndim == 2:
        if arr.shape == (n_points, TACTILE_AXES):
            return ensure_force_frame(arr, n_points)[None, None, ...]
        if arr.shape[1] == n_points * TACTILE_AXES:
            return arr.reshape(arr.shape[0], 1, n_points, TACTILE_AXES).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-2:] == (n_points, TACTILE_AXES):
            return arr[:, None, ...].astype(np.float32)
        if arr.shape[-1] == n_points * TACTILE_AXES:
            return arr.reshape(arr.shape[0], arr.shape[1], n_points, TACTILE_AXES).astype(np.float32)
    if arr.ndim == 4 and arr.shape[-2:] == (n_points, TACTILE_AXES):
        return arr.astype(np.float32)

    raise ValueError(f"unsupported recording shape: {arr.shape}")


def rgba_from_values(values: np.ndarray, norm: Normalize, cmap_name: str = "magma", alpha=None):
    rgba = cm.get_cmap(cmap_name)(norm(values))
    if alpha is not None:
        rgba[:, 3] = alpha
    return rgba


class TactilePlot:
    def __init__(
        self,
        xy: np.ndarray,
        n_sensors: int,
        vmin: float = 0.0,
        vmax: float = 200.0,
        cmap_name: str = "magma",
        titles: Sequence[str] | None = None,
        dark: bool = False,
        inactive: Sequence[bool] | None = None,
    ):
        if dark:
            plt.style.use("dark_background")
        self.xy = xy
        self.n_sensors = n_sensors
        self.norm = Normalize(vmin=vmin, vmax=vmax)
        self.cmap_name = cmap_name
        self.inactive = list([False] * n_sensors if inactive is None else inactive)
        if len(self.inactive) != n_sensors:
            raise ValueError("inactive length must match n_sensors")
        self.fig, axes = plt.subplots(
            1,
            n_sensors,
            figsize=(6 * n_sensors, 6),
            constrained_layout=True,
        )
        self.axes = [axes] if n_sensors == 1 else list(axes)
        if dark:
            self.fig.patch.set_facecolor("black")

        pad = 0.05 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-6)
        xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        for index, ax in enumerate(self.axes):
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if titles:
                ax.set_title(titles[index])

        zeros = np.zeros((xy.shape[0], TACTILE_AXES), dtype=np.float32)
        self.scatters = []
        self.quivers = []
        for index, ax in enumerate(self.axes):
            colors = self._colors(zeros, inactive=self.inactive[index])
            self.scatters.append(
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=100,
                    marker="o",
                    facecolors=colors,
                    edgecolors="none",
                )
            )
            self.quivers.append(
                ax.quiver(
                    xy[:, 0],
                    xy[:, 1],
                    zeros[:, 0],
                    zeros[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.006,
                    headwidth=4,
                    headlength=6,
                    alpha=0.8,
                )
            )

        self.mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap_name)
        self.mappable.set_array(np.array([vmin, vmax], dtype=np.float32))
        self.fig.colorbar(self.mappable, ax=self.axes, shrink=0.9, pad=0.02)

    def _colors(self, frame: np.ndarray, inactive: bool = False):
        if inactive:
            return np.tile(INACTIVE_RGBA, (frame.shape[0], 1))
        force_norm = np.linalg.norm(frame, axis=1)
        alpha = np.full_like(force_norm, ALPHA_BASE, dtype=np.float32)
        alpha[force_norm > FORCE_ALPHA_THRESH] = ALPHA_STRONG
        return rgba_from_values(force_norm, self.norm, self.cmap_name, alpha=alpha)

    def update(self, frames: Sequence[np.ndarray]):
        all_norms = []
        for index, frame in enumerate(frames):
            frame = ensure_force_frame(frame, n_points=self.xy.shape[0])
            inactive = self.inactive[index]
            self.scatters[index].set_facecolors(self._colors(frame, inactive=inactive))
            if inactive:
                zeros = np.zeros(frame.shape[0], dtype=np.float32)
                self.quivers[index].set_UVC(zeros, zeros)
                all_norms.append(zeros)
            else:
                self.quivers[index].set_UVC(frame[:, 0] * ARROW_SCALE, frame[:, 1] * ARROW_SCALE)
                all_norms.append(np.linalg.norm(frame, axis=1))
        self.mappable.set_array(np.concatenate(all_norms, axis=0))
        return self.scatters + self.quivers


def show_live(
    frame_reader: Callable[[], Sequence[np.ndarray]],
    xy: np.ndarray,
    n_sensors: int,
    fps: int,
    record_path: str | Path | None = None,
    titles: Sequence[str] | None = None,
    vmin: float = 0.0,
    vmax: float = 200.0,
    inactive: Sequence[bool] | None = None,
) -> None:
    plotter = TactilePlot(xy, n_sensors=n_sensors, titles=titles, vmin=vmin, vmax=vmax, inactive=inactive)
    records = []

    def update(_):
        try:
            frames = list(frame_reader())
            if record_path is not None:
                records.append(np.asarray(frames))
        except Exception as exc:
            print("[WARN] read failed:", exc)
            return plotter.scatters + plotter.quivers
        return plotter.update(frames)

    interval_ms = int(1000 / fps)
    animation = FuncAnimation(plotter.fig, update, interval=interval_ms, blit=False)
    plt.show()
    _ = animation

    if record_path is not None:
        output = resolve_package_path(record_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.asarray(records))
        print("saved:", output)


def show_replay(
    input_path: str | Path,
    xy: np.ndarray,
    fps: int,
    save_mp4: str | Path | None = None,
    vmin: float = 0.0,
    vmax: float = 300.0,
) -> None:
    frames = load_recording(input_path, n_points=xy.shape[0])
    plotter = TactilePlot(xy, n_sensors=frames.shape[1], vmin=vmin, vmax=vmax, cmap_name="Reds", dark=True)

    def update(index):
        return plotter.update(frames[index])

    interval_ms = int(1000 / fps)
    animation = FuncAnimation(
        plotter.fig,
        update,
        frames=range(frames.shape[0]),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_mp4 is not None:
        output = resolve_package_path(save_mp4)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output, fps=fps)
        print("saved:", output)
    else:
        plt.show()


def save_plot(
    input_path: str | Path,
    output_path: str | Path,
    xy: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 200.0,
) -> None:
    frames = load_recording(input_path, n_points=xy.shape[0])
    plotter = TactilePlot(xy, n_sensors=frames.shape[1], vmin=vmin, vmax=vmax)
    plotter.update(frames[0])
    output = resolve_package_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.fig.savefig(output)
    plt.close(plotter.fig)
    print("saved:", output)
