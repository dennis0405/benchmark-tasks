"""Audit cradle containment in a saved robot trajectory using its actual tool poses."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tool_bounds import ToolBounds

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--trace", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
rack = json.loads((args.assets / "fixture-geometry-contract.json").read_text())["rack"]
names = ("hex_driver", "open_wrench")
keys = ("D", "W")
geometry = [
    ToolBounds(
        args.trace.with_name(args.trace.name.split(".")[0] + "_" + name + ".usda")
    )
    for name in names
]
bounds = [np.array(rack[k + "_cradle_bounds_world_m"]) for k in keys]
rows = json.loads(args.trace.read_text())
samples = []
previous = None
for tick, row in enumerate(rows, 1):
    values = []
    for i, body in enumerate(geometry):
        xyz = np.array(row["tools"][i])
        rotation = Rotation.from_quat(row["tool_quats_xyzw"][i]).as_matrix()
        values.append(body.in_cradle(xyz, rotation, bounds[i]))
    if values != previous:
        samples.append(
            dict(action=tick, phase=row["phase"], driver=values[0], wrench=values[1])
        )
        previous = values
# A control with its origin inside the cradle but its long handle protruding must fail.
for i, body in enumerate(geometry):
    mid = np.mean(bounds[i], axis=0)
    rotation = (
        np.eye(3) if i == 0 else Rotation.from_euler("z", 90, degrees=True).as_matrix()
    )
    if body.in_cradle(mid, rotation, bounds[i]):
        raise AssertionError(f"{names[i]} accepted a protruding handle")
args.out.write_text(
    json.dumps(
        dict(
            scope="Recorded 20 Hz control samples",
            actions=len(rows),
            transitions=samples,
            protrusion_controls=2,
        ),
        indent=2,
    )
    + "\n"
)
print(samples)
