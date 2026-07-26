import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yourdfpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GX16_URDF_PATH = PROJECT_ROOT / "libgex" / "gx16" / "urdf" / "gx4m.urdf"
FIXED_TIP_JOINTS = {
    "joint18": ("link4", "link18"),
    "joint19": ("link8", "link19"),
    "joint20": ("link12", "link20"),
}


class GX16UrdfTest(unittest.TestCase):
    def test_fixed_tip_joint_definitions(self):
        root = ET.parse(GX16_URDF_PATH).getroot()
        links = {link.attrib["name"]: link for link in root.findall("link")}
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

        for joint_name, (parent_name, child_name) in FIXED_TIP_JOINTS.items():
            with self.subTest(joint=joint_name):
                joint = joints[joint_name]
                self.assertEqual(joint.attrib["type"], "fixed")
                self.assertEqual(joint.find("parent").attrib["link"], parent_name)
                self.assertEqual(joint.find("child").attrib["link"], child_name)
                self.assertEqual(list(links[child_name]), [])

    def test_fixed_tip_frames_follow_their_parent_links(self):
        urdf = yourdfpy.URDF.load(
            GX16_URDF_PATH,
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        self.assertEqual(
            urdf.actuated_joint_names,
            [f"joint{index}" for index in range(1, 17)],
        )

        urdf.update_cfg({f"joint{index}": 0.0 for index in range(1, 17)})
        expected_transforms = {
            child_name: urdf.get_transform(child_name, parent_name)
            for parent_name, child_name in FIXED_TIP_JOINTS.values()
        }
        urdf.update_cfg(
            {f"joint{index}": 0.03 * index for index in range(1, 17)}
        )
        for parent_name, child_name in FIXED_TIP_JOINTS.values():
            with self.subTest(child=child_name):
                np.testing.assert_allclose(
                    urdf.get_transform(child_name, parent_name),
                    expected_transforms[child_name],
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
