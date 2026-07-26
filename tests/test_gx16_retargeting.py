import unittest
from pathlib import Path

import numpy as np
import yourdfpy

from libgex.gx16.retargeting import EX16ToGX16Retargeting, JOINT_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GX16_URDF_PATH = PROJECT_ROOT / "libgex" / "gx16" / "urdf" / "gx4m.urdf"


class TargetVectorCoordinatesTest(unittest.TestCase):
    def test_target_vectors_match_dex_world_position_subtraction(self):
        urdf = yourdfpy.URDF.load(
            GX16_URDF_PATH,
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        origins = ["plam_link"] * 4
        tasks = ["link18", "link19", "link20", "link21"]

        actual = EX16ToGX16Retargeting._tip_vectors(urdf, origins, tasks)

        urdf.update_cfg({name: 0.0 for name in JOINT_NAMES})
        root_link = urdf.base_link
        expected = np.asarray(
            [
                urdf.get_transform(task, root_link)[:3, 3]
                - urdf.get_transform(origin, root_link)[:3, 3]
                for origin, task in zip(origins, tasks)
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

        origin_relative = np.asarray(
            [
                urdf.get_transform(task, origin)[:3, 3]
                for origin, task in zip(origins, tasks)
            ]
        )
        self.assertFalse(np.allclose(actual, origin_relative))


if __name__ == "__main__":
    unittest.main()
