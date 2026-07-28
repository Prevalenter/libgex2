import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
from libgex2 import Hand16
import numpy as np


hand = Hand16(serial_number="FTAKRP3A")
hand.connect()

hand.home()
