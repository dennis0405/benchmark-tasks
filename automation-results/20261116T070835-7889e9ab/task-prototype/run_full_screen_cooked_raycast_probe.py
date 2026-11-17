"""Bounded native scene-query probe of the cooked full-screen floor.

This is not a dynamics or production qualification.  It asks one question:
after PhysX cooking, do vertical rays pass through every authored aperture and
hit representative rib/annulus material in the 3 mm floor slab?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import carb
import omni.physx
import isaaclab.sim as sim_utils
from pxr import UsdPhysics

from full_screen_scene import (APERTURE, FLOOR_T, HOLES, INNER_R, PITCH,
                               SCREEN_ROOT_Z, SCREEN_X, SCREEN_Y,
                               spawn_screen)


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / 240.0, device=args.device, render_interval=12,
        enable_scene_query_support=True))
    spawn_screen("/World/Screen", None,
                 translation=(SCREEN_X, SCREEN_Y, SCREEN_ROOT_Z))
    UsdPhysics.RigidBodyAPI.Get(
        sim_utils.get_current_stage(), "/World/Screen").CreateKinematicEnabledAttr(True)
    sim.reset()
    # One completed interval ensures that scene-query acceleration structures
    # contain the cooked collision descendants.
    sim.step(render=False)
    query = omni.physx.get_physx_scene_query_interface()
    z0 = SCREEN_ROOT_Z + 0.010
    ray_length = 0.020

    def floor_hits(x: float, y: float) -> list[dict]:
        hits = []
        def callback(hit):
            path = str(hit.collision)
            if path.startswith("/World/Screen/Floor_"):
                hits.append({"distance_m": float(hit.distance), "collision": path})
            return True
        query.raycast_all(carb.Float3(SCREEN_X + x, SCREEN_Y + y, z0),
                          carb.Float3(0.0, 0.0, -1.0), ray_length, callback)
        return hits

    aperture_rows = []
    for index, (x, y) in enumerate(HOLES):
        hits = floor_hits(x, y)
        aperture_rows.append({"index": index, "xy_m": [x, y],
                              "cooked_floor_hit_count": len(hits), "hits": hits})

    # Midpoints between axial neighbor holes sample the narrowest nominal ribs.
    hole_set = {(round(x, 8), round(y, 8)) for x, y in HOLES}
    rib_points = set()
    for x, y in HOLES:
        for dx, dy in ((PITCH, 0.0), (0.0, PITCH)):
            if (round(x + dx, 8), round(y + dy, 8)) in hole_set:
                rib_points.add((round(x + dx / 2, 8), round(y + dy / 2, 8)))
    rib_rows = [{"xy_m": [x, y], "hits": floor_hits(x, y)}
                for x, y in sorted(rib_points)]
    # The solid annulus is sampled away from the wall at 24 angles.
    annulus_r = INNER_R - 0.003
    annulus_rows = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        x, y = annulus_r * math.cos(a), annulus_r * math.sin(a)
        annulus_rows.append({"xy_m": [x, y], "hits": floor_hits(x, y)})

    aperture_blocked = sum(bool(row["hits"]) for row in aperture_rows)
    rib_missed = sum(not row["hits"] for row in rib_rows)
    annulus_missed = sum(not row["hits"] for row in annulus_rows)
    report_path = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(__file__).resolve().parent.parent
    trace_path = report_path.with_name(report_path.stem + "-trace.json")
    trace_path.write_text(json.dumps({
        "scope": "cooked_scene_query_path_discovery_only",
        "physics_sample_rate_hz": 240,
        "apertures": aperture_rows,
        "rib_samples": rib_rows,
        "annulus_samples": annulus_rows,
    }, indent=2))
    report = {"checks": [
        {"name": "cooked_aperture_center_ray_blocked_count", "measured": aperture_blocked,
         "operator": "==", "threshold": 0, "unit": "of_261_rays"},
        {"name": "cooked_rib_center_ray_miss_count", "measured": rib_missed,
         "operator": "==", "threshold": 0, "unit": f"of_{len(rib_rows)}_rays"},
        {"name": "cooked_annulus_ray_miss_count", "measured": annulus_missed,
         "operator": "==", "threshold": 0, "unit": "of_24_rays"},
        {"name": "native_scene_query_aperture_count", "measured": len(aperture_rows),
         "operator": "==", "threshold": 261, "unit": "rays"},
        {"name": "native_scene_query_rib_sample_count", "measured": len(rib_rows),
         "operator": ">", "threshold": 0, "unit": "rays"},
        {"name": "physics_dt_absolute_error", "measured": abs(float(sim.get_physics_dt()) - 1/240),
         "operator": "<=", "threshold": 1e-12, "unit": "s"},
    ], "artifacts": [trace_path.relative_to(workspace).as_posix()]}
    report_path.write_text(json.dumps(report, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
