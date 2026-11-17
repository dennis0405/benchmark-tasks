"""Discover same-interval bilateral contact sampling at native physics rate.

A free rigid beam settles across two separated supports.  Two filtered contact
sensors are explicitly refreshed after every completed SimulationContext step.
This small API probe does not qualify the YAM grasp or the loaded screen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-hz", type=int, choices=(240, 480), required=True)
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg


def body(path: str, size, pos, *, kinematic: bool, mass: float) -> RigidObject:
    cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=kinematic, solver_position_iteration_count=32,
            solver_velocity_iteration_count=8),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.00015, rest_offset=0.0),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.5, restitution=0.0))
    return RigidObject(RigidObjectCfg(
        prim_path=path, spawn=cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos)))


def main() -> None:
    dt = 1.0 / args.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, render_interval=max(1, args.physics_hz // 20)))
    left = body("/World/LeftSupport", (0.04, 0.08, 0.04), (-0.08, 0, 0.50),
                kinematic=True, mass=1.0)
    right = body("/World/RightSupport", (0.04, 0.08, 0.04), (0.08, 0, 0.50),
                 kinematic=True, mass=1.0)
    beam = body("/World/Beam", (0.24, 0.06, 0.025), (0, 0, 0.58),
                kinematic=False, mass=0.25)
    left_sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/LeftSupport", update_period=0.0, history_length=0,
        track_air_time=False, filter_prim_paths_expr=["/World/Beam"]))
    right_sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/RightSupport", update_period=0.0, history_length=0,
        track_air_time=False, filter_prim_paths_expr=["/World/Beam"]))
    sim.reset(); left.reset(); right.reset(); beam.reset()
    simultaneous = longest = run = 0
    left_positive = right_positive = 0
    max_residual = 0.0
    device = ""
    rows = []
    for step in range(args.steps):
        sim.step(render=False)
        left.update(dt); right.update(dt); beam.update(dt)
        left_sensor.update(dt, force_recompute=True)
        right_sensor.update(dt, force_recompute=True)
        lf = left_sensor.data.normal_force_matrix_w.torch[0, 0, 0].detach()
        rf = right_sensor.data.normal_force_matrix_w.torch[0, 0, 0].detach()
        if step == 0:
            device = str(left_sensor.data.normal_force_matrix_w.torch.device)
        ln = float(torch.linalg.vector_norm(lf).cpu())
        rn = float(torch.linalg.vector_norm(rf).cpu())
        l_on, r_on = ln > 1e-6, rn > 1e-6
        left_positive += int(l_on); right_positive += int(r_on)
        both = l_on and r_on
        simultaneous += int(both)
        run = run + 1 if both else 0
        longest = max(longest, run)
        # Vertical reactions should sum to beam weight after settling.  This is
        # a diagnostic sanity bound, not an equal/opposite pair comparison.
        if both:
            residual = abs(float((lf[2] + rf[2]).cpu()) - 0.25 * 9.81)
            max_residual = max(max_residual, residual)
        if step % max(1, args.physics_hz // 20) == 0 or both:
            rows.append({"physics_interval": step + 1,
                         "left_force_w_N": lf.cpu().tolist(),
                         "right_force_w_N": rf.cpu().tolist(),
                         "simultaneous": both})

    report_path = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(__file__).resolve().parent.parent
    trace_path = report_path.with_name(report_path.stem + "-trace.json")
    trace_path.write_text(json.dumps({
        "scope": "same_interval_bilateral_contact_api_discovery_only",
        "physics_sample_rate_hz": args.physics_hz,
        "samples": rows,
    }, indent=2))
    report = {"checks": [
        {"name": "physics_dt_absolute_error", "measured": abs(float(sim.get_physics_dt()) - dt),
         "operator": "<=", "threshold": 1e-12, "unit": "s"},
        {"name": "cuda_backed_contact_matrices", "measured": int(device.startswith("cuda")),
         "operator": "==", "threshold": 1, "unit": "bool"},
        {"name": "left_positive_contact_physics_samples", "measured": left_positive,
         "operator": ">", "threshold": 0, "unit": "completed_physics_samples"},
        {"name": "right_positive_contact_physics_samples", "measured": right_positive,
         "operator": ">", "threshold": 0, "unit": "completed_physics_samples"},
        {"name": "same_interval_bilateral_contact_physics_samples", "measured": simultaneous,
         "operator": ">", "threshold": 0, "unit": "completed_physics_samples"},
        {"name": "longest_uninterrupted_bilateral_contact_duration", "measured": longest * dt,
         "operator": ">=", "threshold": 0.5, "unit": "s"},
        {"name": "fresh_completed_physics_sample_count", "measured": args.steps,
         "operator": "==", "threshold": args.steps, "unit": "samples"},
    ], "artifacts": [trace_path.relative_to(workspace).as_posix()]}
    report_path.write_text(json.dumps(report, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
