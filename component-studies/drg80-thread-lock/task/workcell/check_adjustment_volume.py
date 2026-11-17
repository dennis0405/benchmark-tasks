"""Check compound tool geometry against known containment and separation cases."""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tool_bounds import ToolBounds
from adjustment_volume import AdjustmentVolume

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
rack = json.loads((args.assets / "fixture-geometry-contract.json").read_text())["rack"]
frame = np.array([0.36, 0.0, 0.75])
mouth = frame + [0, -0.018, 0.1017]
rows = []
for name, key, reset, rot in (
    (
        "hex_driver",
        "D",
        "D_reset_tip_world_m",
        Rotation.from_euler("x", -90, degrees=True).as_matrix(),
    ),
    ("open_wrench", "W", "W_reset_nut_axis_world_m", np.eye(3)),
):
    volume = AdjustmentVolume(ToolBounds(args.assets / "tools" / (name + ".usda")))
    # For the wrench, move the reference beyond the far edge while leaving
    # part of its handle in the finite cylinder. A centre-only test misses it.
    cases = [
        ("rack", np.array(rack[reset]), rot, True),
        ("engaged", mouth, np.eye(3), False),
        ("above", mouth + [0, 0, 0.20], np.eye(3), True),
    ]
    if key == "W":
        cases.append(("handle_crosses", mouth + [0.080, 0, 0], np.eye(3), False))
    # The same relative rigid transforms must give the same signed distance.
    common = Rotation.from_euler("zx", [71, -19], degrees=True).as_matrix()
    shift = np.array([0.15, 0.27, -0.34])
    for label, pos, r, expected in cases:
        d = volume.distance(pos, r, frame, np.eye(3))
        d2 = volume.distance(
            common @ pos + shift, common @ r, common @ frame + shift, common
        )
        assert (d > 0) == expected and abs(d - d2) < 2.0e-6, (name, label, d, d2)
        rows.append(dict(tool=name, pose=label, distance_m=d))
args.out.write_text(json.dumps(rows, indent=2) + "\n")
print(rows)
