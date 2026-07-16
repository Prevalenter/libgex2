"""EX16-to-GX16 vector retargeting powered by dex-retargeting."""

from pathlib import Path

import numpy as np
import yaml


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "retargeting.yaml"
JOINT_NAMES = [f"joint{i}" for i in range(1, 17)]


class EX16ToGX16Retargeting:
    """Convert EX16 joint angles in degrees to GX16 URDF angles in radians."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        try:
            import yourdfpy
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise ImportError(
                "Retargeting dependencies are missing. Activate gx_hand and install "
                "dex-retargeting==0.4.6 and yourdfpy."
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
        target_neutral = self._tip_vectors(
            target_urdf,
            raw_config["retargeting"]["target_origin_link_names"],
            raw_config["retargeting"]["target_task_link_names"],
        )
        self.source_to_target_rotation = self._fit_rotation(
            source_neutral, target_neutral
        )

        # Start from the GX16 open-hand pose instead of joint-limit midpoints.
        self.retargeting.set_qpos(np.zeros(len(self.target_joint_names)))

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
        zero = {name: np.float64(0.0) for name in JOINT_NAMES}
        urdf.update_cfg(zero)
        return np.asarray(
            [
                urdf.get_transform(task, origin)[:3, 3]
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
        qpos = self.retargeting.retarget(aligned_vectors)
        return np.asarray(qpos, dtype=np.float64)[self.output_indices]

    def reset(self):
        self.retargeting.reset()
        self.retargeting.set_qpos(np.zeros(len(self.target_joint_names)))

