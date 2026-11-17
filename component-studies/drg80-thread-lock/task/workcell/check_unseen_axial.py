"""Apply a bidirectional native load after the thread predictor has run.

A contact partner can supply an axial impulse the predictor does not yet know.
This component experiment replaces that impulse with a known external load.
It does not involve a robot or issue task points.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--assets", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--hz", type=int, choices=(240, 480), default=240)
parser.add_argument("--solver", choices=("tgs", "pgs"), default="tgs")
parser.add_argument(
    "--external-mount",
    action="store_true",
    help="Attach the workcell with a finite-strength external fixed joint.",
)
parser.add_argument("--implicit-axial", action="store_true")
parser.add_argument("--transmitted-reaction", action="store_true")
args = parser.parse_args()
app = AppLauncher(headless=True).app

import numpy as np
import warp as wp
from pxr import UsdGeom
from isaaclab_physx.physics import PhysxCfg
import isaaclab.sim as sim_utils
from build_bench import build
from native_bench import Bench


def main():
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asset = args.out.with_suffix(".usda").resolve()
    build(
        asset,
        args.assets,
        continuous_seat=True,
        implicit_axial=args.implicit_axial,
        external_mount=args.external_mount,
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / args.hz,
            device="cuda:0",
            physics=PhysxCfg(
                solver_type=int(args.solver == "tgs"),
                enable_external_forces_every_iteration=args.solver == "tgs",
            ),
        )
    )
    UsdGeom.Xform.Define(
        sim.stage, "/World/Bench"
    ).GetPrim().GetReferences().AddReference(str(asset))
    sim.reset()
    bench = Bench(
        "/World/Bench",
        args.hz,
        native_friction=True,
        transmitted_reaction=args.transmitted_reaction,
        implicit_axial=args.implicit_axial,
    )
    result = {
        "rate_hz": args.hz,
        "solver": args.solver,
        "implicit_axial": args.implicit_axial,
        "transmitted_reaction": args.transmitted_reaction,
        "trials": [],
    }
    trace = []
    lead = 0.001 / math.tau
    try:
        for part, axial, rotary, k in (("screw", 0, 1, 1e6), ("nut", 2, 3, 1e5)):
            for sign in (-1, 1):
                stats = []
                for phase, load in (
                    ("before", 0.0),
                    ("load", sign * 5.0),
                    ("released", 0.0),
                ):
                    values = []
                    for i in range(args.hz):
                        bench.step()
                        # A genuine external generalized force. Nut force here is
                        # relative to the screw, so this is a differential nut load.
                        actuated = bench.art.get_dof_actuation_forces().numpy().copy()
                        actuated[0, bench.index[axial]] += load * min(
                            1.0, (i + 1) / (0.20 * args.hz)
                        )
                        bench.art.set_dof_actuation_forces(
                            wp.array(actuated, device=bench.device), bench.ids
                        )
                        sim.step(render=False)
                        q, v = bench.state()
                        e = q[axial] - lead * q[rotary]
                        if (
                            not np.isfinite(q).all()
                            or abs(q[0]) > 0.0048
                            or 0.001 + q[0] + q[2] < -0.0002
                        ):
                            raise ValueError(
                                f"invalid native guide/seat under {part} {load} N"
                            )
                        values.append(e)
                        trace.append([part, sign, phase, e, *q[:4]])
                    stats.append(median(values[-args.hz // 4 :]))
                expected = sign * 5 / k
                actual = float(stats[1] - stats[0])
                back = float(stats[2] - stats[0])
                result["trials"].append(
                    dict(
                        body=part,
                        force_n=sign * 5,
                        expected_m=expected,
                        measured_m=actual,
                        recovered_m=back,
                        pass_static=abs(actual / expected - 1) <= 0.10,
                        pass_return=abs(back) <= 1e-6,
                    )
                )
    except Exception as e:
        result["failure"] = str(e)
    result["pass"] = len(result["trials"]) == 4 and all(
        t["pass_static"] and t["pass_return"] for t in result["trials"]
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".trace.json").write_text(json.dumps(trace))
    print(json.dumps(result), flush=True)


try:
    main()
finally:
    app.close()
