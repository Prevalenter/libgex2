"""Forward kinematics for the four-finger EX16 glove."""

import os.path as osp

import numpy as np

try:
    import pybullet as p
except ImportError:  # Keep importing libgex possible when FK is not used.
    p = None


abs_path = osp.dirname(osp.abspath(__file__))


class KinEX16:
    """PyBullet-backed forward kinematics model for EX16.

    Public angles are expressed in degrees, matching :class:`Glove`.
    """

    FINGER_JOINT_NAMES = (
        ("joint1", "joint2", "joint3", "joint4"),
        ("joint5", "joint6", "joint7", "joint8"),
        ("joint9", "joint10", "joint11", "joint12"),
        ("joint13", "joint14", "joint15", "joint16"),
    )
    TIP_LINK_NAMES = ("Link17", "Link18", "Link19", "link20")

    def __init__(self, vis=False) -> None:
        if p is None:
            raise ImportError("KinEX16 requires pybullet; install it with 'pip install pybullet'")

        self.name = "EX16"
        self.client_id = p.connect(p.GUI if vis else p.DIRECT)
        self.offset_z = 1.0
        flags = getattr(p, "URDF_IGNORE_VISUAL_SHAPES", 0) | getattr(
            p, "URDF_IGNORE_COLLISION_SHAPES", 0
        )
        self.bullet_hand = p.loadURDF(
            osp.join(abs_path, "urdf", "glove4.urdf"),
            useFixedBase=True,
            basePosition=[0, 0, self.offset_z],
            flags=flags,
            physicsClientId=self.client_id,
        )

        joints = {}
        links = {}
        for joint_id in range(p.getNumJoints(self.bullet_hand, self.client_id)):
            info = p.getJointInfo(self.bullet_hand, joint_id, self.client_id)
            joints[info[1].decode()] = joint_id
            links[info[12].decode()] = joint_id

        self.finger_joint_ids = [
            [joints[name] for name in names] for names in self.FINGER_JOINT_NAMES
        ]
        self.finger_link_ids = [links[name] for name in self.TIP_LINK_NAMES]

        # Compatibility attributes used by existing EX12-style client code.
        (
            self.thumb_joint_ids,
            self.index_joint_ids,
            self.middle_joint_ids,
            self.ring_joint_ids,
        ) = self.finger_joint_ids
        (
            self.thumb_link_id,
            self.index_link_id,
            self.middle_link_id,
            self.ring_link_id,
        ) = self.finger_link_ids

    def fk_finger(self, finger_index, q=None):
        """Return one fingertip XYZ position for four joint angles in degrees."""
        if finger_index not in range(4):
            raise ValueError("finger_index must be between 0 and 3")
        q = [0.0] * 4 if q is None else list(q)
        if len(q) != 4:
            raise ValueError("each EX16 finger requires exactly four joint angles")

        for joint_id, angle in zip(self.finger_joint_ids[finger_index], np.deg2rad(q)):
            p.resetJointState(
                self.bullet_hand,
                joint_id,
                float(angle),
                physicsClientId=self.client_id,
            )

        position = list(
            p.getLinkState(
                self.bullet_hand,
                self.finger_link_ids[finger_index],
                computeForwardKinematics=True,
                physicsClientId=self.client_id,
            )[4]
        )
        position[2] -= self.offset_z
        return position

    def fk_finger1(self, q=None):
        return self.fk_finger(0, q)

    def fk_finger2(self, q=None):
        return self.fk_finger(1, q)

    def fk_finger3(self, q=None):
        return self.fk_finger(2, q)

    def fk_finger4(self, q=None):
        return self.fk_finger(3, q)

    def close(self):
        if p is not None and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
