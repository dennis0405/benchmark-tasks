#!/usr/bin/env python3
"""Native callback/timestep preflight; this is not the aggregate contract report."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="../measurements/api-preflight.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import IsaacEvents, PhysxManager


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    counts = {"pre": 0, "post": 0}
    dts = []

    def pre(dt):
        counts["pre"] += 1
        dts.append(float(dt))

    def post(_dt):
        counts["post"] += 1

    h0 = PhysxManager.register_callback(pre, IsaacEvents.PRE_PHYSICS_STEP, name="drg80_preflight_pre", wrap_weak_ref=False)
    h1 = PhysxManager.register_callback(post, IsaacEvents.POST_PHYSICS_STEP, name="drg80_preflight_post", wrap_weak_ref=False)
    sim.reset()
    for _ in range(24):
        sim.step(render=False)
    measured = {
        "callback_pre_count": counts["pre"],
        "callback_post_count": counts["post"],
        "callback_count_difference": abs(counts["pre"] - counts["post"]),
        "physics_dt_max_error": max(abs(x - 1.0 / 240.0) for x in dts),
    }
    checks = [
        {"name": "callback_pre_count", "measured": measured["callback_pre_count"], "operator": "==", "threshold": 24, "unit": "samples"},
        {"name": "callback_post_count", "measured": measured["callback_post_count"], "operator": "==", "threshold": 24, "unit": "samples"},
        {"name": "callback_count_difference", "measured": measured["callback_count_difference"], "operator": "==", "threshold": 0, "unit": "samples"},
        {"name": "physics_dt_max_error", "measured": measured["physics_dt_max_error"], "operator": "<=", "threshold": 1e-9, "unit": "s"},
    ]
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checks": checks, "artifacts": []}, indent=2) + "\n")
    for handle in (h0, h1):
        handle.deregister()
    return out


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
