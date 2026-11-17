"""Screen rack approach angles and uninterrupted clockwise arcs with the pinned YAM IK."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik, station
from yam_arena.constants import READY, PAD_TIP_PAST_TCP
from yam_arena.demo import grip_rotation

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
rig = station.rig()
base = rig.bases["left"]
rack = np.array([0.24, 0.11, 0.793])
axis = Rotation.from_euler("x", -90, degrees=True).as_matrix()
grasp = rack + np.array([0, 0.100, 0])
mouth = np.array([0.36, -0.018, 0.8517])
out = []
for tilt in (0, 35, 60, 80):
    for az in (90, -90):
        rr = Rotation.from_euler("x", -tilt, degrees=True).as_matrix() @ grip_rotation(
            math.radians(az)
        )
        center = grasp - PAD_TIP_PAST_TCP * rr[:, 2]
        q = np.array(READY)
        rack_error = 0
        for pos in (center - 0.050 * rr[:, 2], center, center + [0, 0, 0.050]):
            q, pe, re = ik.solve(
                pos, ik.matrix_to_quat_xyzw(rr), q_init=q, base_pos=base, iters=110
            )
            rack_error = max(rack_error, pe + 0.05 * re)
        entry = dict(tilt_deg=tilt, yaw_deg=az, rack_error_m=rack_error, arcs=[])
        if rack_error < 0.002:
            offset = axis.T @ (center - rack)
            orient = axis.T @ rr
            for idx in range(6):
                warm = q.copy()
                span = -1
                for a in [None, *range(0, 281, 10)]:
                    r = Rotation.from_euler(
                        "z", 60 * idx - (a or 0), degrees=True
                    ).as_matrix()
                    pos = mouth + r @ offset + [0, 0, 0.050 if a is None else -0.0026]
                    qq, pe, re = ik.solve(
                        pos,
                        ik.matrix_to_quat_xyzw(r @ orient),
                        q_init=warm,
                        base_pos=base,
                        iters=75,
                    )
                    if (
                        pe > 0.002
                        or re > 0.035
                        or (a is not None and a > 0 and max(abs(qq - warm)) > 1.25)
                    ):
                        break
                    if a is not None:
                        span = a
                    warm = qq
                entry["arcs"].append(span)
        print(entry, flush=True)
        out.append(entry)
args.out.write_text(json.dumps(out, indent=2) + "\n")
