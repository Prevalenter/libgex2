## Libgex2

Python library for GX10 and EX12.
### Usage

Basic requirements: x64 Ubuntu 22.04 and Python (>=3.8).

Python requirements:

```
pip install pyserial
pip install pybullet
pip install numpy
```

### EX16 ZMQ + Viser

The EX16 publisher and browser-based viewer have a separate reproducible
environment. Create it from the repository root:

```bash
conda env create -f environment-ex16.yml
conda activate libgex2-ex16
```

Alternatively, install the dependencies into an existing Python 3.10
environment:

```bash
python -m pip install -r requirements-ex16.txt
```

Connect the EX16 and start the publisher in one terminal (replace the port if
needed):

```bash
python nodes/ex16_zmq_node.py --port /dev/ttyACM0
```

The publisher opens a Qt status window showing the latest 16 measured joint
angles. Enter a pose name and click **Save Current Pose** to store it in
`poses/ex16/`; when the name is empty, a timestamped name is generated. Use
`--pose-dir PATH` to choose a different pose library.

Then start the viewer in another terminal:

```bash
python demo_ex16_viser.py
```

Open <http://127.0.0.1:8080> in a browser. If the device is selected by USB
serial number instead, use `--serial-number SERIAL_NUMBER`; pass `--left` for
a left-hand glove. On Linux, the current user must have permission to access
the serial device (normally through the `dialout` group described below).

For development without a physical EX16, start the Qt fake-state publisher:

```bash
python nodes/fake_ex16_zmq_node.py
```

Its 16 sliders publish the same `ex16/state` messages as the real node. The
pose controls save and load versioned JSON files under `poses/ex16/` by default;
use `--pose-dir PATH` to select another pose library. The angle range defaults
to -180 through 180 degrees and can be changed with `--angle-min` and
`--angle-max`.

The real and fake EX16 publishers both bind to `tcp://127.0.0.1:5567` by
default, so do not run them simultaneously on that endpoint. Use
`--state-endpoint` when separate endpoints are needed.

To view EX16 and the retargeted GX16 together, run:

```bash
python demo_ex16_gx16_viser.py
```

The scene gizmos and the **Base Coordinate Transforms** sidebar adjust each
model's XYZ position and RPY rotation. **Save Base Transforms** stores the
layout in `viewer_layouts/ex16_gx16.json`, which is loaded automatically on the
next start. Use `--base-transform-file PATH` to choose another layout file.

Add the current user to the `dialout` group so that it can access the serial devices (no need to `chmod 777 /dev/ttyUSB* or /dev/ttyACM*`):

OpenRB150 will be recognized as `/dev/ttyACM*` (usually `/dev/ttyACM0`). U2D2 will be recognized as `/dev/ttyUSB*` (usually `/dev/ttyUSB0`).

```bash
sudo usermod -a -G dialout $USER
```

Change the latency timer of the GX10 U2D2 to `1` (default is `16`), otherwise the latency will be `16` ms:

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

or changing all `ttyUSB*` by:
```bash
for dev in /sys/bus/usb-serial/devices/ttyUSB*/latency_timer; do
    echo 1 | sudo tee "$dev"
done
```

#### GX10

Connect 5V DC power and usb to the GX10, the device will be recognized as `/dev/ttyUSB*` (usually `/dev/ttyUSB0`), you can control the GX10 by: 

```python
import sys
sys.path.append('<path_to_libgex2>') # replace with the actual path to libgex2

from libgex2 import Hand
import numpy as np


hand = Hand("/dev/ttyUSB0")  # or using serial_number='XXXX'
hand.connect(curr_limit=1000, goal_current=600, goal_pwm=200)

hand.home() # move to home position
print(hand.getjs()) # get joint positions, unit: degree

hand.setjs([0]*10) # equal to hand.home()

hand.setj(10, 60) # set joint 10 (starting from 1) to 60 degree
```

The joint order of the GX10 is:

```
1: thumb 1 (bottom)
2: thumb 2
3: thumb 3
4: thumb 4 (tip)
5: index 1 (bottom)
6: index 2
7: index 3 (tip)
8: middle 1 (bottom)
9: middle 2
10: middle 3 (tip)
```


### EX12
Connect type-c USB to EX12, the device will be recognized as `/dev/ttyACM*` (usually `/dev/ttyACM0`), you can control the EX12 by:

```python
import sys
sys.path.append('<path_to_libgex2>') # replace with the actual path to libgex2

from libgex2 import Glove
import numpy as np


glove = Glove("/dev/ttyUSB0", left=False)  # or using serial_number='XXXX', left=True for left hand
glove.connect()

print(glove.getjs()) # get joint positions, unit: degree

thumb_tip_xyz, index_tip_xyz, mid_tip_xyz = glove.fk() # get finger XYZ coordinates, unit: m
```

The joint order of the EX12 is:

```
1: thumb 1 (bottom)
2: thumb 2
3: thumb 3
4: thumb 4 (tip)
5: index 1 (bottom)
6: index 2
7: index 3 
8: index 4 (tip)
9: middle 1 (bottom)
10: middle 2 
11: middle 3 
12: middle 4 (tip)
```

Detail of the API can be found [here (Chinese)](libgex/api.md).

### Retargeting from EX12 to GX10 (Dexterous Teleoperation)

The retargeting code is in [gex_retargeting_sim](https://github.com/Democratizing-Dexterous/gex_retargeting_sim). Put folder `gex_retargeting_sim` in the same directory level as `libgex2`.

```python
import sys
sys.path.append('<path_to_libgex2>') # replace with the actual path to libgex2, gex_retargeting_sim is in the same directory level as libgex2


import numpy as np
import time

from libgex import Hand, Glove
from gex_retargeting_sim import GexRetarget

gex_retarget = GexRetarget()


hand = Hand(port="/dev/ttyUSB0")  
hand.connect()


glove = Glove(port="/dev/ttyACM0") 
glove.connect()

print("start retargeting...")

while True:

    glove_base_pose = np.array([0, 0, 0])

    glove_finger1_pos, glove_finger2_pos, glove_finger3_pos = glove.fk()

    glove_fingers_pos = np.concatenate(
        [
            glove_base_pose[None, :],
            glove_finger1_pos[None, :],
            glove_finger2_pos[None, :],
            glove_finger3_pos[None, :],
        ],
        axis=0,
    )

    qpos = gex_retarget.retarget(glove_fingers_pos)

    qpos_degree = qpos * 180 / np.pi

    hand.setjs(qpos_degree)
```

By running the code above, you can move the glove to control the dexterous hand as following:

<img src='assets/retarget_real.png' width=45%>


### Main Updates

#### No Need to Set Zero

No need to set zero of the motors. The zero position of the GX10 or EX12 should be installed as following (The red rectangle should be aligned with the 'D'):

<img src='assets/motor_zero.png' width=45%>

In this way, when the GX10 or EX12 is in machanical zero positions, all the motors should have 90 degree position reading from the sensor.

#### GX10

GX10 is a brand new tri-finger 10 DoF dexterous hand (zero positions viewed by [URDFly](https://github.com/Democratizing-Dexterous/URDFly)):

<img src='assets/gx10.png' width=45%>

<img src='assets/gx10_real.png' width=45%>

GX10 has the same size of human and and optimized wiring (nearly no wire exposed).

The urdf file of GX10 is [here](libgex/gx10/urdf/gx10.urdf).

#### EX12

EX12 is a brand new tri-finger 12 DoF exoskeleton glove (zero positions viewed by [URDFly](https://github.com/Democratizing-Dexterous/URDFly)):

<img src='assets/ex12.png' width=45%>

<img src='assets/ex12_real.png' width=45%>

EX12 is fully optimized for wearable purpose (customized finger tip and wearable glove).

The urdf file of EX12 is [here](libgex/ex12/urdf/ex12.urdf).
