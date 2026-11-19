"""Dimensions of the rigid foot and the measured short-stroke two-speed screw."""

LOW_LEAD = 0.0254 / 32
HIGH_LEAD = 0.0254 / 3.5
LOW_SWEEP = 40
HIGH_SWEEP = 45
STROKE = (LOW_LEAD * LOW_SWEEP + HIGH_LEAD * HIGH_SWEEP) / 360
PAD_TOP = 0.0075
FOOT_HALF_WIDTHS = (0.115, 0.105, 0.013)
FOOT_CENTER_Z = -0.5395
PRELOAD = 0.000015
CLEAR_MM = 0.035
