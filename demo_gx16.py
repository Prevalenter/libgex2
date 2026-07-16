import sys

sys.path.append("../")  # replace with the actual path to libgex2
from libgex2 import Hand16
import numpy as np


hand = Hand16(serial_number="FTAKRP3AA")
hand.connect()

hand.home()
