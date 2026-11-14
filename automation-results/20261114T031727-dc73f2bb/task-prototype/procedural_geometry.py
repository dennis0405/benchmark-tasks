"""Frozen SI geometry shared by the YAM scene and native component diagnostics."""

TABLE_Z = 0.76
INSERT_XY = ((0.285, -0.032), (0.315, 0.032), (0.365, -0.032), (0.405, 0.032))

# The shell is an open container assembled from convex primitives.  The centre
# is deliberately empty: detached tangs remain visible and recoverable.
GEOMETRY = {
    "shell_base": {"kind": "elongated_five_box_open_shell", "dimensions_m": [0.180, 0.090, 0.024]},
    "boss": {"kind": "hollow_boss_four_boxes", "dimensions_m": [0.014, 0.014, 0.018]},
    "retained_insert": {"kind": "hollow_insert_four_boxes", "dimensions_m": [0.0048, 0.0048, 0.00375]},
    "tang": {"kind": "box", "dimensions_m": [0.0032, 0.00065, 0.00045]},
    "mandrel_shaft": {"kind": "cylinder", "dimensions_m": [0.0020, 0.0020, 0.070]},
    "mandrel_handle": {"kind": "cylinder", "dimensions_m": [0.014, 0.014, 0.040]},
    "hammer_head": {"kind": "box", "dimensions_m": [0.030, 0.018, 0.016]},
    "hammer_handle": {"kind": "cylinder", "dimensions_m": [0.012, 0.012, 0.105]},
    "tray": {"kind": "adjacent_five_box_open_container", "dimensions_m": [0.080, 0.125, 0.040]},
}

# Prototype material-test assumptions, frozen by artifacts/contract.json.
NOTCH_BREAK_FORCE_N = 7.0
NOTCH_BREAK_TORQUE_NM = 0.018
SEAT_BREAK_FORCE_N = 90.0
SHELL_BREAK_FORCE_N = 250.0
