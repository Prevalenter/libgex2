"""Driver for the 16-DoF EX16 exoskeleton glove."""

import os

import numpy as np
import yaml

from ..dynamixel_sdk import PacketHandler, PortHandler
from ..motor import Motor
from ..utils import search_ports
from .kinematics import KinEX16


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
_CONFIG = load_config(CONFIG_FILE)
PROTOCOL_VERSION = _CONFIG["BASIC"]["PROTOCOL_VERSION"]
BAUDRATE = _CONFIG["BASIC"]["BAUDRATE"]
DEFAULT_SERIAL_NUMBER = _CONFIG["BASIC"].get("SERIAL_NUMBER")
NAME = _CONFIG["HAND"]["NAME"]
NUM_MOTORS = _CONFIG["HAND"]["NUM"]
THUMB_IDS = _CONFIG["HAND"]["THUMB_IDS"]
INDEX_IDS = _CONFIG["HAND"]["INDEX_IDS"]
MID_IDS = _CONFIG["HAND"]["MID_IDS"]
RING_IDS = _CONFIG["HAND"]["RING_IDS"]
JOINT_MOTOR_DIRECTIONS = _CONFIG["HAND"]["JOINT_MOTOR_DIRECTIONS"]


def load_joint_motor_directions():
    """Load and validate the live motor-to-URDF direction mapping."""
    directions = load_config(CONFIG_FILE)["HAND"]["JOINT_MOTOR_DIRECTIONS"]
    if len(directions) != NUM_MOTORS:
        raise ValueError(
            f"JOINT_MOTOR_DIRECTIONS must contain {NUM_MOTORS} values, "
            f"got {len(directions)}"
        )
    if any(direction not in (-1, 1) for direction in directions):
        raise ValueError("JOINT_MOTOR_DIRECTIONS values must be either 1 or -1")
    return np.asarray(directions, dtype=float)


class Glove:
    """Read joint angles and fingertip positions from an EX16 glove."""

    def __init__(self, port=None, serial_number=None, left=False) -> None:
        if port is None and serial_number is None:
            serial_number = DEFAULT_SERIAL_NUMBER
        if port is None and serial_number is None:
            raise ValueError(
                "port or serial_number is required; it can also be set as "
                "BASIC.SERIAL_NUMBER in config.yaml"
            )

        self.left_directions = np.array(
            [-1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1]
        )
        self.right_directions = np.ones(NUM_MOTORS)
        self.hand_directions = (
            self.left_directions if left else self.right_directions
        )
        self._config_mtime_ns = None
        self._joint_motor_directions = load_joint_motor_directions()
        self._refresh_joint_motor_directions()
        self.directions = self.hand_directions * self._joint_motor_directions
        self.is_connected = False

        if port is not None:
            self.port = port
        else:
            ports_info = search_ports()
            if serial_number not in ports_info:
                raise RuntimeError(f"Serial number {serial_number!r} is not available")
            self.port = ports_info[serial_number]

        self.name = NAME
        # Constructed lazily so joint reading does not require pybullet.
        self.kin = None

    def connect(self):
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        if not (self.portHandler.openPort() and self.portHandler.setBaudRate(BAUDRATE)):
            self.is_connected = False
            raise RuntimeError(f"Failed to open {self.port}")

        self.is_connected = True
        self.motors = [
            Motor(motor_id, self.portHandler, self.packetHandler)
            for motor_id in range(1, NUM_MOTORS + 1)
        ]
        init_js = [motor.get_pos() for motor in self.motors]
        self.init_offsets = np.array([0 if angle < 270 else 360 for angle in init_js])
        self.off()
        print(f"{self.name} connect done...")
        print("init joint positions:", init_js)
        print("joint offsets:", self.init_offsets.tolist())

    def off(self):
        for motor in self.motors:
            motor.torq_off()

    def getjs(self):
        """Return all 16 URDF joint angles in degrees."""
        if not self.is_connected:
            raise RuntimeError("EX16 is not connected")
        self._refresh_joint_motor_directions()
        angles = np.array([motor.get_pos() for motor in self.motors], dtype=float)
        return (
            (angles - 90 - self.init_offsets)
            * self.hand_directions
            * self._joint_motor_directions
        )

    def _refresh_joint_motor_directions(self):
        """Hot-reload direction changes without restarting the reader."""
        mtime_ns = os.stat(CONFIG_FILE).st_mtime_ns
        if mtime_ns != self._config_mtime_ns:
            self._joint_motor_directions = load_joint_motor_directions()
            self.directions = self.hand_directions * self._joint_motor_directions
            self._config_mtime_ns = mtime_ns

    def fk(self):
        """Return thumb, index, middle and ring fingertip XYZ positions in metres."""
        if self.kin is None:
            self.kin = KinEX16()
        js = self.getjs()
        return tuple(
            np.asarray(self.kin.fk_finger(finger, js[finger * 4 : finger * 4 + 4]))
            for finger in range(4)
        )


Glove16 = Glove
