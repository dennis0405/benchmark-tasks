"""Fixed reference geometry, in metres. World coordinates use Z up."""

import numpy as np

MOUTH = 0.72
OVEN_Y = -0.26
STONE_Z = 1.0
PARK = np.array([MOUTH - 0.10, OVEN_Y - 0.09, STONE_Z + 0.00085])
GRIP_DISTANCE = 0.465
HANDLE_Z = 0.042
