"""Screen a fixture translation with the real rack-grasp transform and original robot limits."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik, station

from service_paths import wrench_flat

p = argparse.ArgumentParser()
p.add_argument("--trace", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
rows = json.loads(args.trace.read_text())
grasp = [r for r in rows if r["phase"] == "hold"][-1]
starts = [np.array(q) for q in grasp["measured_arm_joints_rad"]]
bases = station.rig().bases
offsets = []
for i, side in enumerate(("left", "right")):
    g = Rotation.from_quat(grasp["tool_quats_xyzw"][i]).as_matrix()
    pos, orientation = ik.tcp_pose(starts[i], bases[side])
    offsets.append((g.T @ (pos - grasp["tools"][i]), g.T @ orientation))


def screen_site(x, y):
    mouth = np.array([x, y - 0.018, 0.8517])
    off, grasp_r = offsets[0]
    choices = []
    for index in range(6):
        q = starts[0].copy()
        span = -1
        origin = index * math.pi / 3
        for degree, depth in [(0, 0.05), (0, -0.004)] + [
            (-a, -0.004) for a in range(10, 271, 10)
        ]:
            rot = Rotation.from_euler("z", origin + math.radians(degree)).as_matrix()
            target = mouth + [0, 0, depth] + rot @ off
            nxt, pe, re = ik.solve(
                target,
                ik.matrix_to_quat_xyzw(rot @ grasp_r),
                q_init=q,
                base_pos=bases["left"],
                iters=65,
            )
            margin = min(np.min(nxt - ik.LOWER), np.min(ik.UPPER - nxt))
            if (
                pe > 0.0012
                or re > 0.028
                or margin < math.radians(10)
                or (degree < 0 and max(abs(nxt - q)) > 1.20)
            ):
                break
            q = nxt
            span = -degree
        choices.append(span)
    return dict(x=x, y=y, spans_deg=choices, max_arc=max(choices))


result = [
    screen_site(x, y)
    for x in (0.26, 0.30, 0.34, 0.36, 0.38, 0.40)
    for y in (-0.08, -0.04, 0, 0.04, 0.08)
]
for r in result:
    if r["max_arc"] < 70:
        continue
    try:
        residual, yaw, _ = wrench_flat(
            np.array([r["x"], r["y"] - 0.018, 0.842]),
            0,
            *offsets[1],
            starts[1],
            bases["right"],
        )
        r.update(
            wrench_reachable=True,
            wrench_yaw_deg=math.degrees(yaw),
            wrench_error_m=residual,
        )
    except ValueError:
        r.update(wrench_reachable=False)
args.out.write_text(
    json.dumps(
        dict(
            scope="Offline IK only. No fixture moved and no YAM service score.",
            grasp_source=args.trace.name,
            placements=result,
        ),
        indent=2,
    )
    + "\n"
)
print(
    "screened",
    len(result),
    "placements",
    sorted((r["max_arc"], r["x"], r["y"]) for r in result if r.get("wrench_reachable"))[
        -5:
    ],
)
