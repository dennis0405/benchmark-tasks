"""Check nominal wrench and valve paths with stock YAM IK, without contacts."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik, station
from service_paths import lever_grasp, wrench_flat
from yam_arena.constants import READY, PAD_TIP_PAST_TCP
from yam_arena.demo import grip_rotation

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
base = station.rig().bases["right"]
rack = np.array([0.24, -0.17, 0.79])
grip_offset = np.array([-0.085, 0, PAD_TIP_PAST_TCP])
nut = np.array([0.36, -0.018, 0.842])
pivot = np.array([0.29, -0.025, 0.82])


def fit(path, start, arm_base=base):
    q = start.copy()
    max_pos = max_angle = 0.0
    for pos, rot in path:
        q, ep, ea = ik.solve(
            pos, ik.matrix_to_quat_xyzw(rot), q_init=q, base_pos=arm_base, iters=150
        )
        max_pos = max(ep, max_pos)
        max_angle = max(ea, max_angle)
    return q, dict(
        position_m=float(max_pos),
        rotation_rad=float(max_angle),
        reachable=bool(max_pos <= 0.002 and max_angle <= 0.04),
    )


racks = []
for yaw in (0, math.pi):
    orient = grip_rotation(yaw)
    q, errors = fit(
        [(rack + grip_offset + [0, 0, h], orient) for h in (0.05, 0, 0.05)],
        np.array(READY),
    )
    racks.append((errors["position_m"] + 0.05 * errors["rotation_rad"], yaw, q, orient))
_, yaw, q, orient = min(racks, key=lambda r: r[0])
sweeps = []
for deg in range(0, 360, 60):
    rot = Rotation.from_euler("z", deg, degrees=True).as_matrix()
    align = [
        (nut + rot @ (grip_offset + [-d, 0, 0]) + [0, 0, h], rot @ orient)
        for d, h in ((0.036, 0.04), (0.036, 0), (0.018, 0), (0, 0))
    ]
    turn = []
    for delta in range(5, 66, 5):
        r = Rotation.from_euler("z", deg - delta, degrees=True).as_matrix()
        turn.append((nut + r @ grip_offset, r @ orient))
    _, m = fit(align + turn, q)
    sweeps.append(dict(initial_flat_deg=deg, **m))
lever = []
for angle, h in [
    (0, 0.035),
    (0, 0),
    *[(d, 0) for d in range(0, 31, 5)],
    *[(d, 0) for d in range(25, -31, -5)],
    *[(d, 0) for d in range(-25, 31, 5)],
]:
    axis = Rotation.from_euler("y", angle, degrees=True).apply([0, 0, 1])
    lever.append(
        (pivot + 0.070 * axis + [0, 0, PAD_TIP_PAST_TCP + h], grip_rotation(0))
    )
_, lv = fit(lever, np.array(READY))
starts = [np.array(READY), np.array(READY)]
residual, hand, r, tilt, new_yaw = lever_grasp(pivot, starts, station.rig().bases)
sw, angle, _ = wrench_flat(nut, 0.0, grip_offset, orient, q, base)
report = dict(
    scope="Nominal IK with stock YAM limits; no physical contact or score.",
    rack_yaw_deg=math.degrees(yaw),
    wrench=sweeps,
    lever_fixed_orientation=lv,
    lever_side_on=dict(
        hand=("left", "right")[hand],
        tilt_deg=tilt,
        yaw_deg=new_yaw,
        weighted_ik_residual_m=residual,
    ),
    selected_wrench_flat_deg=math.degrees(angle),
    wrench_weighted_ik_residual_m=sw,
)
a.out.write_text(json.dumps(report, indent=2) + "\n")
print(report)
