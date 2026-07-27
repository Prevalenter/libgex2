import unittest

import numpy as np

from demo_ex16_geort_gx16_viser import (
    DEFAULT_CALIBRATION,
    gx16_configuration,
    restore_projector,
)
from demo_ex16_gx16_viser import GX16_MESH_DIR, GX16_URDF_PATH, load_urdf


class SavedCalibrationTest(unittest.TestCase):
    def test_restored_projector_reproduces_recorded_human_frame(self):
        projector, _ = restore_projector(DEFAULT_CALIBRATION)
        with np.load(DEFAULT_CALIBRATION, allow_pickle=False) as raw:
            qpos = raw["qpos_deg"][0]
        expected = np.load(
            DEFAULT_CALIBRATION.parent / "human_ex16.npy", allow_pickle=False
        )[0]
        np.testing.assert_allclose(projector.project(qpos), expected, atol=1e-7)


class GX16ConfigurationTest(unittest.TestCase):
    def test_configures_all_joints_in_urdf_order(self):
        urdf = load_urdf(GX16_URDF_PATH, GX16_MESH_DIR)
        qpos = np.arange(16, dtype=np.float64) / 10.0
        actual = gx16_configuration(urdf, qpos)
        by_name = dict(zip((f"joint{i}" for i in range(1, 17)), qpos))
        expected = np.asarray([by_name[name] for name in urdf.actuated_joint_names])
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
