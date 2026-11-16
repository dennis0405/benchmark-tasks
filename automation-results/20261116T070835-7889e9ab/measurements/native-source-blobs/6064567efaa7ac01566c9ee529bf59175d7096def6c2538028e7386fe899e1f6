"""Bounded native probe for the separating-screen revolute drive.

This diagnostic asks only whether a standalone, full-scale rigid screen with
the frozen mass can be moved through a world-to-body revolute joint by normal
joint-drive commands, whether its native rigid pose follows both commanded
directions, and whether it returns to neutral.  It does not model apertures,
supports, media, washers, a robot action, or aggregate task success.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-hz", type=int, choices=(240, 480), required=True)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.utils import get_current_stage
import numpy as np
from pxr import Gf, Sdf, UsdPhysics


SCREEN_OUTER_DIAMETER_M = 0.3683
SCREEN_FLOOR_THICKNESS_M = 0.003
SCREEN_MASS_KG = 0.40
SCREEN_CENTER = np.array((0.0, 0.0, 0.35), dtype=float)
HINGE_OFFSET_BODY = np.array((0.0, -0.10, 0.0), dtype=float)
HINGE_WORLD = SCREEN_CENTER + HINGE_OFFSET_BODY


def rotation_xyzw(quaternion):
    """Return a rotation matrix for the runtime pose's verified XYZW order."""
    x, y, z, w = map(float, quaternion)
    n = x*x + y*y + z*z + w*w
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


def read_native(body):
    position = body.data.root_pos_w.torch[0].detach().cpu().numpy().astype(float)
    quaternion = body.data.root_quat_w.torch[0].detach().cpu().numpy().astype(float)
    velocity = body.data.root_vel_w.torch[0].detach().cpu().numpy().astype(float)
    rotation = rotation_xyzw(quaternion)
    roll_deg = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    pivot = position + rotation @ HINGE_OFFSET_BODY
    return position, quaternion, velocity, roll_deg, pivot


def main():
    dt = 1.0 / args.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, render_interval=max(1, args.physics_hz // 20)))

    screen_cfg = RigidObjectCfg(
        prim_path="/World/DiagnosticScreen",
        spawn=sim_utils.CylinderCfg(
            radius=SCREEN_OUTER_DIAMETER_M / 2.0,
            height=SCREEN_FLOOR_THICKNESS_M,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
                max_depenetration_velocity=2.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=SCREEN_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.00015, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.30, 0.38, 0.45))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(SCREEN_CENTER), rot=(0.0, 0.0, 0.0, 1.0)))
    screen = RigidObject(screen_cfg)

    stage = get_current_stage()
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/DiagnosticScreenHinge")
    joint.CreateAxisAttr("X")
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/DiagnosticScreen")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, HINGE_WORLD)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, HINGE_OFFSET_BODY)))
    joint.CreateLowerLimitAttr(-8.0)
    joint.CreateUpperLimitAttr(8.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(500.0)
    drive.CreateDampingAttr(25.0)
    drive.CreateMaxForceAttr(50.0)
    drive.CreateTargetPositionAttr(0.0)

    sim.reset()
    screen.reset()
    native_physics_dt = float(sim.get_physics_dt())
    native_screen_mass = float(screen.data.default_mass.torch[0, 0].detach().cpu())
    trace = []
    steps = int(round(args.seconds * args.physics_hz))
    for step in range(steps):
        elapsed = step * dt
        if elapsed < 1.0:
            target_deg = 0.0
            phase = "settle"
        elif elapsed < 2.0:
            target_deg = 5.0
            phase = "positive"
        elif elapsed < 3.0:
            target_deg = -5.0
            phase = "negative"
        else:
            target_deg = 0.0
            phase = "return"
        drive.GetTargetPositionAttr().Set(target_deg)
        sim.step(render=False)
        screen.update(dt)
        position, quaternion, velocity, roll_deg, pivot = read_native(screen)
        trace.append({
            "completed_physics_step": step + 1,
            "time_s": (step + 1) * dt,
            "phase": phase,
            "drive_target_deg": target_deg,
            "root_position_m": position.tolist(),
            "root_quaternion_xyzw": quaternion.tolist(),
            "root_velocity_mps_radps": velocity.tolist(),
            "observed_roll_deg": roll_deg,
            "hinge_pivot_world_m": pivot.tolist(),
            "hinge_pivot_residual_m": float(np.linalg.norm(pivot - HINGE_WORLD)),
        })

    positive = [x["observed_roll_deg"] for x in trace if x["phase"] == "positive"]
    negative = [x["observed_roll_deg"] for x in trace if x["phase"] == "negative"]
    returned = trace[-max(1, args.physics_hz // 2):]
    finite = bool(np.isfinite(np.asarray([
        x["root_position_m"] + x["root_quaternion_xyzw"] +
        x["root_velocity_mps_radps"] for x in trace])).all())
    maximum_pivot_residual = max(x["hinge_pivot_residual_m"] for x in trace)
    final_half_second_max_abs_roll = max(abs(x["observed_roll_deg"]) for x in returned)
    final_half_second_max_speed = max(
        float(np.linalg.norm(x["root_velocity_mps_radps"][:3])) for x in returned)
    final_half_second_max_angular_speed = max(
        float(np.linalg.norm(x["root_velocity_mps_radps"][3:6])) for x in returned)

    report_path = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = report_path.with_name(report_path.stem + "-trace.json.gz")
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
        json.dump({
            "scope": "standalone revolute-drive/readback discovery only",
            "physics_hz": args.physics_hz,
            "screen_geometry": {
                "diagnostic_shape": "solid cylinder; apertures intentionally out of scope",
                "outer_diameter_m": SCREEN_OUTER_DIAMETER_M,
                "thickness_m": SCREEN_FLOOR_THICKNESS_M,
                "mass_kg": SCREEN_MASS_KG,
                "native_mass_readback_kg": native_screen_mass,
                "native_physics_dt_s": native_physics_dt,
                "hinge_offset_body_m": HINGE_OFFSET_BODY.tolist(),
            },
            "samples": trace,
        }, stream, separators=(",", ":"))

    workspace = Path(__file__).resolve().parent.parent
    trace_relative = trace_path.relative_to(workspace).as_posix()
    report = {
        "checks": [
            {"name": "physics_dt_absolute_error", "measured": abs(native_physics_dt - dt),
             "operator": "<=", "threshold": 1e-9, "unit": "s"},
            {"name": "native_screen_mass_relative_error",
             "measured": abs(native_screen_mass - SCREEN_MASS_KG) / SCREEN_MASS_KG,
             "operator": "<=", "threshold": 1e-5, "unit": "ratio"},
            {"name": "finite_native_pose_velocity_samples", "measured": int(finite),
             "operator": "==", "threshold": 1, "unit": "bool"},
            {"name": "fresh_physics_sample_count", "measured": len(trace),
             "operator": "==", "threshold": steps, "unit": "completed_physics_samples"},
            {"name": "positive_drive_maximum_observed_roll", "measured": max(positive),
             "operator": ">", "threshold": 2.0, "unit": "deg"},
            {"name": "negative_drive_minimum_observed_roll", "measured": min(negative),
             "operator": "<", "threshold": -2.0, "unit": "deg"},
            {"name": "maximum_hinge_pivot_residual", "measured": maximum_pivot_residual,
             "operator": "<=", "threshold": 0.0005, "unit": "m"},
            {"name": "final_half_second_maximum_absolute_roll", "measured": final_half_second_max_abs_roll,
             "operator": "<=", "threshold": 1.0, "unit": "deg"},
            {"name": "final_half_second_maximum_linear_speed", "measured": final_half_second_max_speed,
             "operator": "<=", "threshold": 0.01, "unit": "m/s"},
            {"name": "final_half_second_maximum_angular_speed",
             "measured": final_half_second_max_angular_speed,
             "operator": "<=", "threshold": 0.05, "unit": "rad/s"},
        ],
        "artifacts": [trace_relative],
    }
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    simulation_app.close()


if __name__ == "__main__":
    main()
