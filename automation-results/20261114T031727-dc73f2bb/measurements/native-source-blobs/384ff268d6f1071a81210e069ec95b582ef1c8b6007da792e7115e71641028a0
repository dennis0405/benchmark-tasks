"""UNVERIFIED native GPU mimic discovery; not a gripper or environment pass.

The trusted rcauto controller must review and execute this script. An isolated,
gravity-free articulation tests the installed deprecated PhysxMimicJointAPI,
its mixed angular/linear units and motion across one complete revolution.
No contact, friction, preload, constitutive spring or YAM claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
import uuid
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--trace", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output, trace = Path(args.output).resolve(), Path(args.trace).resolve()
for path in (output, trace):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
trial = str(uuid.uuid4())
source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
import numpy as np
import warp as wp
from isaaclab_physx.physics import PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

DT, STEPS, LEAD = 1 / 240, 1800, 0.001


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def body(stage, name, mass):
    prim = UsdGeom.Cube.Define(stage, "/World/Helix/" + name)
    prim.CreateSizeAttr(0.01)
    prim.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
    m = UsdPhysics.MassAPI.Apply(prim.GetPrim())
    m.CreateMassAttr(mass)
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim.GetPrim()).CreateDisableGravityAttr(True)
    return prim.GetPath()


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    stage = sim.stage
    root = UsdGeom.Xform.Define(stage, "/World/Helix")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    frame, carrier, rotor = (
        body(stage, n, m) for n, m in [("frame", 1.0), ("carrier", 0.01), ("rotor", 0.01)]
    )
    mount = UsdPhysics.FixedJoint.Define(stage, "/World/Helix/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.5))
    prism = UsdPhysics.PrismaticJoint.Define(stage, "/World/Helix/axial")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Helix/rotation")
    for joint, parent, child in [(prism, frame, carrier), (hinge, carrier, rotor)]:
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateAxisAttr("Z")
    prism.CreateLowerLimitAttr(-0.005)
    prism.CreateUpperLimitAttr(0.005)
    hinge.CreateLowerLimitAttr(-1440.0)
    hinge.CreateUpperLimitAttr(1440.0)
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(hinge.GetPrim(), "rotZ")
    mimic.CreateReferenceJointRel().SetTargets([prism.GetPath()])
    mimic.CreateReferenceJointAxisAttr("rotX")  # one-DOF prismatic reference
    # PhysxMimicJointAPI evaluates the USD revolute coordinate in degrees while
    # the prismatic coordinate is in metres.  With q_rot + G*q_linear = 0,
    # negative 360 deg/lead produces positive one-turn rotation per +LEAD.
    # Native parser/solver support and the sign remain outcomes of this probe.
    mimic.CreateGearingAttr(-360.0 / LEAD)
    mimic.CreateOffsetAttr(0.0)
    mimic.CreateNaturalFrequencyAttr(0.0)  # ideal constraint discovery only
    mimic.CreateDampingRatioAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(prism.GetPrim(), "linear")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(100.0)
    drive.CreateMaxForceAttr(10.0)
    drive.CreateTargetVelocityAttr(0.0)
    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    physics.set_subspace_roots("/")
    articulation = physics.create_articulation_view("/World/Helix")
    if articulation.count != 1:
        raise RuntimeError(f"expected one native articulation, got {articulation.count}")
    metatype = articulation.get_metatype(0)
    dof_names = list(metatype.dof_names)
    try:
        axial_dof = dof_names.index("axial")
    except ValueError as exc:
        raise RuntimeError(f"native axial DOF missing; observed DOFs: {dof_names}") from exc
    all_indices = wp.array([0], dtype=wp.uint32, device=args.device)
    views = [physics.create_rigid_body_view([str(path)]) for path in (frame, carrier, rotor)]
    samples = []
    for phase, velocity in [("forward", LEAD / (2 * math.pi)), ("reverse", -LEAD / (2 * math.pi))]:
        # Direct live USD drive mutation is illegal with eENABLE_DIRECT_GPU_API.
        # Use the installed native tensor target setter and preserve every other
        # DOF target.  This is a diagnostic actuator command, not a state write.
        targets = wp.to_torch(articulation.get_dof_velocity_targets()).clone()
        targets[0, axial_dof] = velocity
        articulation.set_dof_velocity_targets(wp.from_torch(targets), all_indices)
        for step in range(STEPS):
            sim.step(render=False)
            PhysxManager.raise_callback_exception_if_any()
            poses = [
                wp.to_torch(v.get_transforms()).detach().cpu().clone().tolist()[0] for v in views
            ]
            dof_positions = (
                wp.to_torch(articulation.get_dof_positions()).detach().cpu().clone().tolist()[0]
            )
            dof_velocities = (
                wp.to_torch(articulation.get_dof_velocities()).detach().cpu().clone().tolist()[0]
            )
            samples.append(
                {
                    "phase": phase,
                    "step": step,
                    "poses_xyzw": poses,
                    "dof_positions": dof_positions,
                    "dof_velocities": dof_velocities,
                }
            )
    a = np.asarray([s["poses_xyzw"] for s in samples], dtype=np.float64)
    # Joint axes are world Z in this isolated fixed-frame coupon.
    q = a[:, 2, 3:7]
    angle = np.unwrap(2 * np.arctan2(q[:, 2], q[:, 3]))
    axial = a[:, 1, 2] - a[:, 0, 2]
    forward_angle = float(angle[STEPS - 1] - angle[0])
    reverse_angle = float(angle[-1] - angle[STEPS - 1])
    forward_z = float(axial[STEPS - 1] - axial[0])
    reverse_z = float(axial[-1] - axial[STEPS - 1])
    save(
        trace,
        {
            "native_trial_id": trial,
            "source_sha256": source_hash,
            "scope": "isolated ideal mimic API discovery only",
            "dof_names": dof_names,
            "axial_dof_index": axial_dof,
            "samples": samples,
        },
    )
    if abs(forward_angle) <= 1e-12:
        raise RuntimeError("native rotor did not move; thread lead is undefined; see raw trace")
    actual_lead = 2 * math.pi * forward_z / forward_angle
    helical_residual = (angle - angle[0]) - 2 * math.pi / LEAD * (axial - axial[0])
    readings = [
        ("native_link_count", sum(v.count for v in views), "==", 3, "links"),
        ("native_articulation_count", articulation.count, "==", 1, "articulations"),
        ("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
        ("finite_pose_fraction", np.isfinite(a).mean(), "==", 1.0, "fraction"),
        ("actual_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
        ("all_sample_helical_residual", np.max(np.abs(helical_residual)), "<=", 0.02, "rad"),
        ("forward_rotation", forward_angle, ">=", 2 * math.pi, "rad"),
        ("reverse_rotation", reverse_angle, "<=", -2 * math.pi, "rad"),
        ("forward_axial_travel", forward_z, ">=", LEAD, "m"),
        ("reverse_axial_travel", reverse_z, "<=", -LEAD, "m"),
        ("measured_lead_error", abs(actual_lead - LEAD), "<=", LEAD * 0.05, "m/revolution"),
        ("carrier_rotor_axial_gap", np.max(np.abs(a[:, 2, 2] - a[:, 1, 2])), "<=", 1e-5, "m"),
        ("off_axis_quaternion_component", np.max(np.abs(q[:, :2])), "<=", 1e-5, "dimensionless"),
    ]
    checks = []
    for name, value, operator, threshold, unit in readings:
        value = float(value)
        passed = (
            math.isfinite(value)
            and {"==": value == threshold, ">=": value >= threshold, "<=": value <= threshold}[
                operator
            ]
        )
        checks.append(
            {
                "name": name,
                "measured": value,
                "operator": operator,
                "threshold": threshold,
                "unit": unit,
                "passed": passed,
            }
        )
    save(
        trace,
        {
            "native_trial_id": trial,
            "source_sha256": source_hash,
            "scope": "isolated ideal mimic API discovery only",
            "dof_names": dof_names,
            "axial_dof_index": axial_dof,
            "samples": samples,
            "measured_lead_m_per_revolution": actual_lead,
        },
    )
    save(
        output,
        {
            "native_trial_id": trial,
            "source_sha256": source_hash,
            "qualification": False,
            "checks": checks,
            "overall_pass": all(c["passed"] for c in checks),
            "artifacts": [str(trace.relative_to(output.parent.parent))],
        },
    )
    if not all(c["passed"] for c in checks):
        raise RuntimeError(
            "native mimic discovery failed: "
            + ", ".join(c["name"] for c in checks if not c["passed"])
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if not output.exists():
            save(
                output,
                {
                    "native_trial_id": trial,
                    "source_sha256": source_hash,
                    "qualification": False,
                    "error": detail,
                    "checks": [
                        {
                            "name": "native_probe_completed",
                            "measured": 0,
                            "operator": "==",
                            "threshold": 1,
                            "unit": "boolean",
                        }
                    ],
                    "artifacts": [str(trace.relative_to(output.parent.parent))]
                    if trace.exists()
                    else [],
                },
            )
        raise
    finally:
        app.close()
