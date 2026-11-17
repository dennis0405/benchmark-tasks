"""Replay measured native poses and independently move each nut off its guide."""

import argparse, json
from pathlib import Path
import numpy as np
from nut_retention import nut_retention

p = argparse.ArgumentParser()
p.add_argument("trace", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
rows = json.loads(a.trace.read_text())


def measure(poses):
    return nut_retention(*[np.array(poses[k]) for k in ("screw", "nut", "retainer")])


samples = [row["poses"] for row in rows if "poses" in row]
reports = [measure(poses) for poses in samples]
assert all(r["nuts_retained"] for r in reports)
controls = []
for name, axis, delta in (
    ("nut", 2, -0.023),
    ("nut", 0, 0.004),
    ("retainer", 0, 0.0021),
    ("retainer", 2, 0.0031),
):
    pose = {k: v.copy() for k, v in rows[0]["poses"].items()}
    pose[name][axis] += delta
    r = measure(pose)
    assert not r["nuts_retained"]
    controls.append(dict(body=name, axis=axis, delta_m=delta, **r))
result = dict(
    trace_ticks=len(rows),
    pose_samples=len(reports),
    minimum_axial_overlaps_m=np.min(
        [r["nut_guide_overlap_m"] for r in reports], axis=0
    ).tolist(),
    maximum_retainer_offset_m=max(r["retainer_installed_error_m"] for r in reports),
    negative_controls=controls,
)
a.out.write_text(json.dumps(result, indent=2) + "\n")
print(result["pose_samples"], result["minimum_axial_overlaps_m"])
