"""Dimensions of the rigid foot and the measured short-stroke two-speed screw."""

LOW_LEAD = 0.0254 / 32
HIGH_LEAD = 0.0254 / 3.5
LOW_SWEEP = 40
HIGH_SWEEP = 45
STROKE = (LOW_LEAD * LOW_SWEEP + HIGH_LEAD * HIGH_SWEEP) / 360
PAD_TOP = 0.0075
FOOT_HALF_WIDTHS = (0.115, 0.105, 0.013)
FOOT_CENTER_Z = -0.5395
# A counterbalanced benchtop carriage preloads its foot through the screw.
CARRIAGE_MASS = 40.0
OUTPUT_STIFFNESS = 1_200_000.0
OUTPUT_DAMPING = 4_800.0
PRELOAD = 0.000028
# Nominal friction coefficient and effective radius of the loaded thrust surface.
THRUST_FRICTION = 0.35
THRUST_RADIUS = 0.055
CLEAR_MM = 0.035
