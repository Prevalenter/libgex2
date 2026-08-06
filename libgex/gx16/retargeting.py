"""EX16-to-GX16 retargeting powered by dex-retargeting."""

from pathlib import Path

import numpy as np
import yaml


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "retargeting.yaml"
JOINT_NAMES = [f"joint{i}" for i in range(1, 17)]
DEXPILOT_ORIGIN_INDICES = np.asarray([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], dtype=int)
DEXPILOT_TASK_INDICES = np.asarray([1, 1, 1, 2, 2, 3, 1, 2, 3, 4], dtype=int)


class EX16ToGX16Retargeting:
    """Convert EX16 joint angles in degrees to GX16 URDF angles in radians."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        try:
            import yourdfpy
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise ImportError(
                f"Retargeting dependency {exc.name!r} is missing. Install it in "
                "the active Python environment together with dex-retargeting and "
                "yourdfpy."
            ) from exc

        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)

        source_config = raw_config["source"]
        source_urdf_path = self._resolve_path(source_config["urdf_path"])
        target_urdf_path = self._resolve_path(raw_config["retargeting"]["urdf_path"])
        self.source_base_link = source_config["base_link_name"]
        self.source_tip_links = source_config["finger_tip_link_names"]

        self.source_urdf = yourdfpy.URDF.load(
            source_urdf_path,
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        target_urdf = yourdfpy.URDF.load(
            target_urdf_path,
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )

        target_config = dict(raw_config["retargeting"])
        target_config["urdf_path"] = str(target_urdf_path)
        self.retargeting = RetargetingConfig.from_dict(target_config).build()
        self.target_joint_names = list(self.retargeting.joint_names)
        missing = set(JOINT_NAMES) - set(self.target_joint_names)
        if missing:
            raise ValueError(f"GX16 retargeting model is missing joints: {sorted(missing)}")
        self.output_indices = np.asarray(
            [self.target_joint_names.index(name) for name in JOINT_NAMES], dtype=int
        )

        source_neutral = self._source_vectors(np.zeros(16))
        retargeting_type = self.retargeting.optimizer.retargeting_type
        if retargeting_type == "DEXPILOT":
            origin_links = [raw_config["retargeting"]["wrist_link_name"]] * 4
            task_links = raw_config["retargeting"]["finger_tip_link_names"]
        elif retargeting_type == "VECTOR":
            origin_links = raw_config["retargeting"]["target_origin_link_names"]
            task_links = raw_config["retargeting"]["target_task_link_names"]
        else:
            raise ValueError(f"Unsupported EX16-to-GX16 retargeting type: {retargeting_type}")
        target_neutral = self._tip_vectors(
            target_urdf,
            origin_links,
            task_links,
        )
        self.source_to_target_rotation = self._fit_rotation(
            source_neutral, target_neutral
        )

        # Start from the GX16 open-hand pose instead of joint-limit midpoints.
        self._reset_to_open_hand()

    def _reset_to_open_hand(self):
        """Reset optimizer history to the zero-angle GX16 pose.

        dex-retargeting 0.4.0 exposes ``last_qpos`` directly, while newer
        releases provide ``set_qpos``. Support both APIs because this project
        is used with the Python 3.8-compatible 0.4.0 release as well.
        """
        initial_qpos = np.zeros(len(self.target_joint_names), dtype=np.float32)
        self.retargeting.reset()
        if hasattr(self.retargeting, "set_qpos"):
            self.retargeting.set_qpos(initial_qpos)
        elif hasattr(self.retargeting, "last_qpos"):
            if self.retargeting.last_qpos.shape != initial_qpos.shape:
                raise ValueError(
                    "dex-retargeting optimizer state has shape "
                    f"{self.retargeting.last_qpos.shape}, expected "
                    f"{initial_qpos.shape}"
                )
            self.retargeting.last_qpos = initial_qpos
        else:
            raise AttributeError(
                "Unsupported dex-retargeting API: expected set_qpos or last_qpos"
            )

        retarget_filter = getattr(self.retargeting, "filter", None)
        if retarget_filter is not None:
            retarget_filter.reset()
        projected = getattr(self.retargeting.optimizer, "projected", None)
        if projected is not None:
            projected.fill(False)

    def _resolve_path(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.config_path.parent / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _tip_vectors(urdf, origin_links, task_links):
        """Return target link vectors in the URDF root/world coordinates.

        dex-retargeting's vector optimizer subtracts the world positions of
        each origin/task pair. Computing ``get_transform(task, origin)`` would
        instead express the vector in the origin link's rotated coordinates,
        which is different when a palm frame such as ``plam_link`` is used.
        """
        zero = {name: np.float64(0.0) for name in JOINT_NAMES}
        urdf.update_cfg(zero)
        root_link = urdf.base_link
        return np.asarray(
            [
                urdf.get_transform(task, root_link)[:3, 3]
                - urdf.get_transform(origin, root_link)[:3, 3]
                for origin, task in zip(origin_links, task_links)
            ]
        )

    @staticmethod
    def _fit_rotation(source_vectors, target_vectors):
        source = source_vectors / np.linalg.norm(source_vectors, axis=1, keepdims=True)
        target = target_vectors / np.linalg.norm(target_vectors, axis=1, keepdims=True)
        u, _, vt = np.linalg.svd(source.T @ target)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return rotation

    def _source_vectors(self, joint_radians):
        configuration = {
            name: np.float64(value)
            for name, value in zip(JOINT_NAMES, joint_radians)
        }
        self.source_urdf.update_cfg(configuration)
        return np.asarray(
            [
                self.source_urdf.get_transform(tip, self.source_base_link)[:3, 3]
                for tip in self.source_tip_links
            ]
        )

    def retarget(self, ex16_joint_degrees):
        """Return GX16 angles in radians, ordered joint1 through joint16."""
        ex16_joint_degrees = np.asarray(ex16_joint_degrees, dtype=np.float64)
        if ex16_joint_degrees.shape != (16,):
            raise ValueError(
                f"Expected 16 EX16 joint angles, got {ex16_joint_degrees.shape}"
            )
        source_vectors = self._source_vectors(np.deg2rad(ex16_joint_degrees))
        aligned_vectors = source_vectors @ self.source_to_target_rotation
        if self.retargeting.optimizer.retargeting_type == "DEXPILOT":
            aligned_points = np.vstack([np.zeros((1, 3)), aligned_vectors])
            reference_vectors = (
                aligned_points[DEXPILOT_TASK_INDICES]
                - aligned_points[DEXPILOT_ORIGIN_INDICES]
            )
        else:
            reference_vectors = aligned_vectors
        qpos = self.retargeting.retarget(reference_vectors)
        return np.asarray(qpos, dtype=np.float64)[self.output_indices]

    def reset(self):
        self._reset_to_open_hand()
