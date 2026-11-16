"""Bounded native probe for filtered PhysX contact-force observations.

This is API discovery, not apparatus qualification.  It drops one dynamic
collision-enabled cube onto another and reads both directions of the native
filtered normal-force matrix at every completed physics interval.  The result
answers whether later bridge and two-beam support tests can use measured
contact instead of pose heuristics.
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


def cube(path: str, z: float, *, kinematic: bool) -> RigidObject:
    spawn = sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=kinematic,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.00015, rest_offset=0.0
        ),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.35, restitution=0.0
        ),
    )
    return RigidObject(RigidObjectCfg(
        prim_path=path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z)),
    ))


def main() -> None:
    dt = 1.0 / args.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, render_interval=max(1, args.physics_hz // 20)
    ))
    lower = cube("/World/Lower", 0.50, kinematic=True)
    upper = cube("/World/Upper", 0.60, kinematic=False)
    lower_sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/Lower",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        filter_prim_paths_expr=["/World/Upper"],
    ))
    upper_sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/Upper",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        filter_prim_paths_expr=["/World/Lower"],
    ))
    sim.reset()
    lower.reset()
    upper.reset()

    samples = []
    contact_samples = 0
    max_pair_residual = 0.0
    max_force = 0.0
    consecutive = 0
    max_consecutive = 0
    matrix_shape = None
    matrix_device = None
    for step in range(args.steps):
        sim.step(render=False)
        lower.update(dt)
        upper.update(dt)
        lower_sensor.update(dt, force_recompute=True)
        upper_sensor.update(dt, force_recompute=True)
        fl = lower_sensor.data.normal_force_matrix_w.torch[0, 0, 0].detach()
        fu = upper_sensor.data.normal_force_matrix_w.torch[0, 0, 0].detach()
        if matrix_shape is None:
            matrix_shape = list(lower_sensor.data.normal_force_matrix_w.torch.shape)
            matrix_device = str(lower_sensor.data.normal_force_matrix_w.torch.device)
        nl = float(torch.linalg.vector_norm(fl).cpu())
        nu = float(torch.linalg.vector_norm(fu).cpu())
        residual = float(torch.linalg.vector_norm(fl + fu).cpu())
        max_force = max(max_force, nl, nu)
        max_pair_residual = max(max_pair_residual, residual)
        touching = nl > 1.0e-6 and nu > 1.0e-6
        if touching:
            contact_samples += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
        if step % max(1, args.physics_hz // 20) == 0 or touching:
            samples.append({
                "step": step + 1,
                "lower_to_upper_normal_force_w_N": fl.cpu().tolist(),
                "upper_to_lower_normal_force_w_N": fu.cpu().tolist(),
                "pair_residual_N": residual,
            })

    report_path = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = report_path.with_name(report_path.stem + "-trace.json")
    workspace = Path(__file__).resolve().parent.parent
    trace_path.write_text(json.dumps({
        "scope": "filtered_contact_api_discovery_only",
        "physics_sample_rate_hz": args.physics_hz,
        "force_matrix_shape": matrix_shape,
        "force_matrix_device": matrix_device,
        "samples": samples,
    }, indent=2))
    report = {
        "checks": [
            {"name": "physics_dt_absolute_error", "measured": abs(float(sim.get_physics_dt()) - dt), "operator": "<=", "threshold": 1e-9, "unit": "s"},
            {"name": "cuda_backed_contact_matrix", "measured": int(str(matrix_device).startswith("cuda")), "operator": "==", "threshold": 1, "unit": "bool"},
            {"name": "filtered_contact_sample_count", "measured": contact_samples, "operator": ">", "threshold": 0, "unit": "completed_physics_samples"},
            {"name": "maximum_consecutive_contact_samples", "measured": max_consecutive, "operator": ">=", "threshold": 12, "unit": "completed_physics_samples"},
            {"name": "maximum_observed_pair_force", "measured": max_force, "operator": ">", "threshold": 0.0, "unit": "N"},
            {"name": "maximum_equal_opposite_force_residual", "measured": max_pair_residual, "operator": "<=", "threshold": 1e-3, "unit": "N"},
            {"name": "fresh_physics_sample_count", "measured": args.steps, "operator": "==", "threshold": args.steps, "unit": "samples"},
        ],
        "artifacts": [trace_path.resolve().relative_to(workspace).as_posix()],
    }
    report_path.write_text(json.dumps(report, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
