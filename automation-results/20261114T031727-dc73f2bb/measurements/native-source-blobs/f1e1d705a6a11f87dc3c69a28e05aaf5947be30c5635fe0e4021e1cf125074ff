#!/usr/bin/env python3
"""Bounded native GPU mimic reaction discovery; not task qualification.

The earlier motion coupon established that the pinned PhysX mimic API couples a
revolute and prismatic DOF with the authored one-millimetre lead.  This separate
gravity-free, non-contact coupon asks the next question: does a signed physical
torque applied to the revolute DOF produce a reciprocal native projected force
at a finitely restrained prismatic DOF?  It records every 240 Hz native sample.

This probe does not represent the DRG80 fixture, thread friction, nut contact,
clamp preload, robot interaction, scoring, or a YAM demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import uuid

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--trace", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

output_path = Path(args.output).resolve()
trace_path = Path(args.trace).resolve()
for product in (output_path, trace_path):
    product.parent.mkdir(parents=True, exist_ok=True)
    product.unlink(missing_ok=True)

trial_id = str(uuid.uuid4())
source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
app = AppLauncher(args).app


import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


DT = 1.0 / 240.0
LEAD_M_PER_REV = 0.001
APPLIED_TORQUE_NM = 0.005
SETTLE_STEPS = 240
LOADED_STEPS = 480
RELEASE_STEPS = 240
AXIAL_STIFFNESS_N_M = 1.0e6
AXIAL_DAMPING_N_S_M = 2500.0
AXIAL_MAX_FORCE_N = 100.0


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check(name: str, measured: float, operator: str, threshold: float, unit: str) -> dict:
    measured = float(measured)
    comparisons = {
        "==": lambda a, b: a == b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
    }
    passed = math.isfinite(measured) and comparisons[operator](measured, float(threshold))
    return {
        "name": name,
        "measured": measured,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(passed),
    }


def create_body(stage, name: str, mass_kg: float):
    cube = UsdGeom.Cube.Define(stage, f"/World/Helix/{name}")
    cube.CreateSizeAttr(0.01)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-6, 1.0e-6, 1.0e-6))
    PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr(True)
    return cube.GetPath()


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    stage = sim.stage
    root = UsdGeom.Xform.Define(stage, "/World/Helix")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    frame = create_body(stage, "frame", 1.0)
    carrier = create_body(stage, "carrier", 0.01)
    rotor = create_body(stage, "rotor", 0.01)

    mount = UsdPhysics.FixedJoint.Define(stage, "/World/Helix/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    axial = UsdPhysics.PrismaticJoint.Define(stage, "/World/Helix/axial")
    rotation = UsdPhysics.RevoluteJoint.Define(stage, "/World/Helix/rotation")
    for joint, parent, child in (
        (axial, frame, carrier),
        (rotation, carrier, rotor),
    ):
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateAxisAttr("Z")
    axial.CreateLowerLimitAttr(-0.005)
    axial.CreateUpperLimitAttr(0.005)
    rotation.CreateLowerLimitAttr(-1440.0)
    rotation.CreateUpperLimitAttr(1440.0)

    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(rotation.GetPrim(), "rotZ")
    mimic.CreateReferenceJointRel().SetTargets([axial.GetPath()])
    mimic.CreateReferenceJointAxisAttr("rotX")
    mimic.CreateGearingAttr(-360.0 / LEAD_M_PER_REV)
    mimic.CreateOffsetAttr(0.0)
    mimic.CreateNaturalFrequencyAttr(0.0)
    mimic.CreateDampingRatioAttr(0.0)

    # A finite native position drive holds the axial reference at its reset
    # coordinate.  The drive is intentionally capped above the expected
    # 2*pi*tau/lead reaction, and no pose, velocity, or joint state is written.
    drive = UsdPhysics.DriveAPI.Apply(axial.GetPrim(), "linear")
    drive.CreateStiffnessAttr(AXIAL_STIFFNESS_N_M)
    drive.CreateDampingAttr(AXIAL_DAMPING_N_S_M)
    drive.CreateMaxForceAttr(AXIAL_MAX_FORCE_N)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)

    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    if physics is None:
        raise RuntimeError("native PhysX simulation view is unavailable after reset")
    physics.set_subspace_roots("/")
    articulation = physics.create_articulation_view("/World/Helix")
    if articulation.count != 1:
        raise RuntimeError(f"expected one articulation, observed {articulation.count}")
    dof_names = list(articulation.get_metatype(0).dof_names)
    try:
        axial_index = dof_names.index("axial")
        rotation_index = dof_names.index("rotation")
    except ValueError as exc:
        raise RuntimeError(f"required native DOFs missing; observed {dof_names}") from exc

    all_indices = wp.array([0], dtype=wp.uint32, device=args.device)
    body_views = {
        name: physics.create_rigid_body_view([str(path)])
        for name, path in (("frame", frame), ("carrier", carrier), ("rotor", rotor))
    }
    if any(view.count != 1 for view in body_views.values()):
        raise RuntimeError(
            "native body view count mismatch: "
            + repr({name: int(view.count) for name, view in body_views.items()})
        )

    samples: list[dict] = []

    def run_phase(name: str, steps: int, rotor_torque_nm: float) -> None:
        efforts = torch.zeros(
            (articulation.count, articulation.max_dofs),
            dtype=torch.float32,
            device=args.device,
        )
        efforts[0, rotation_index] = rotor_torque_nm
        effort_buffer = wp.from_torch(efforts.contiguous(), dtype=wp.float32)
        for step in range(steps):
            # This is a physical generalized torque input, not a state write.
            articulation.set_dof_actuation_forces(effort_buffer, all_indices)
            sim.step(render=False)
            PhysxManager.raise_callback_exception_if_any()
            positions = wp.to_torch(articulation.get_dof_positions()).detach().cpu().clone()[0]
            velocities = wp.to_torch(articulation.get_dof_velocities()).detach().cpu().clone()[0]
            applied = wp.to_torch(articulation.get_dof_actuation_forces()).detach().cpu().clone()[0]
            projected = (
                wp.to_torch(articulation.get_dof_projected_joint_forces())
                .detach()
                .cpu()
                .clone()[0]
            )
            transforms = {
                body_name: wp.to_torch(view.get_transforms()).detach().cpu().clone()[0].tolist()
                for body_name, view in body_views.items()
            }
            samples.append(
                {
                    "phase": name,
                    "phase_step": step,
                    "dof_positions": positions.tolist(),
                    "dof_velocities": velocities.tolist(),
                    "dof_actuation_forces": applied.tolist(),
                    "dof_projected_joint_forces": projected.tolist(),
                    "body_transforms_xyzw": transforms,
                }
            )

    run_phase("settle", SETTLE_STEPS, 0.0)
    run_phase("positive_torque", LOADED_STEPS, APPLIED_TORQUE_NM)
    run_phase("negative_torque", LOADED_STEPS, -APPLIED_TORQUE_NM)
    run_phase("released", RELEASE_STEPS, 0.0)

    position_array = np.asarray([sample["dof_positions"] for sample in samples], dtype=np.float64)
    velocity_array = np.asarray([sample["dof_velocities"] for sample in samples], dtype=np.float64)
    applied_array = np.asarray(
        [sample["dof_actuation_forces"] for sample in samples], dtype=np.float64
    )
    projected_array = np.asarray(
        [sample["dof_projected_joint_forces"] for sample in samples], dtype=np.float64
    )
    all_native = np.concatenate(
        (position_array.ravel(), velocity_array.ravel(), applied_array.ravel(), projected_array.ravel())
    )

    positive_slice = slice(SETTLE_STEPS + LOADED_STEPS // 2, SETTLE_STEPS + LOADED_STEPS)
    negative_start = SETTLE_STEPS + LOADED_STEPS
    negative_slice = slice(negative_start + LOADED_STEPS // 2, negative_start + LOADED_STEPS)
    positive_axial_force = float(np.median(projected_array[positive_slice, axial_index]))
    negative_axial_force = float(np.median(projected_array[negative_slice, axial_index]))
    expected_force = 2.0 * math.pi * APPLIED_TORQUE_NM / LEAD_M_PER_REV
    positive_ratio = abs(positive_axial_force) / expected_force
    negative_ratio = abs(negative_axial_force) / expected_force
    sign_product = positive_axial_force * negative_axial_force
    maximum_axial_motion = float(np.max(np.abs(position_array[:, axial_index])))
    maximum_helix_residual = float(
        np.max(
            np.abs(
                position_array[:, rotation_index]
                - position_array[0, rotation_index]
                - (2.0 * math.pi / LEAD_M_PER_REV)
                * (position_array[:, axial_index] - position_array[0, axial_index])
            )
        )
    )

    checks = [
        check("native_articulation_count", articulation.count, "==", 1, "articulations"),
        check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
        check("native_body_count", sum(view.count for view in body_views.values()), "==", 3, "bodies"),
        check("actual_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
        check("native_state_finite_fraction", np.isfinite(all_native).mean(), "==", 1.0, "fraction"),
        check("fresh_native_sample_count", len(samples), "==", 1440, "physics steps"),
        check("axial_restraint_motion_max", maximum_axial_motion, "<=", 0.0005, "m"),
        check("loaded_helix_residual_max", maximum_helix_residual, "<=", 0.02, "rad"),
        check("positive_reaction_force_min", abs(positive_axial_force), ">=", 1.0, "N"),
        check("negative_reaction_force_min", abs(negative_axial_force), ">=", 1.0, "N"),
        check("reaction_force_sign_reversal", sign_product, "<", 0.0, "N^2"),
        check("positive_reciprocal_effort_ratio_min", positive_ratio, ">=", 0.5, "ratio"),
        check("positive_reciprocal_effort_ratio_max", positive_ratio, "<=", 1.5, "ratio"),
        check("negative_reciprocal_effort_ratio_min", negative_ratio, ">=", 0.5, "ratio"),
        check("negative_reciprocal_effort_ratio_max", negative_ratio, "<=", 1.5, "ratio"),
    ]

    write_json(
        trace_path,
        {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "scope": "isolated ideal mimic reciprocal-reaction discovery only",
            "dof_names": dof_names,
            "axial_dof_index": axial_index,
            "rotation_dof_index": rotation_index,
            "applied_rotor_torque_nm": APPLIED_TORQUE_NM,
            "expected_ideal_axial_reaction_n": expected_force,
            "measured_positive_axial_projected_force_n": positive_axial_force,
            "measured_negative_axial_projected_force_n": negative_axial_force,
            "samples": samples,
        },
    )
    write_json(
        output_path,
        {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "qualification": False,
            "checks": checks,
            "overall_pass": all(item["passed"] for item in checks),
            "artifacts": [str(trace_path.relative_to(output_path.parent.parent))],
        },
    )
    if not all(item["passed"] for item in checks):
        raise RuntimeError(
            "native mimic reaction discovery failed: "
            + ", ".join(item["name"] for item in checks if not item["passed"])
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if not output_path.exists():
            write_json(
                output_path,
                {
                    "native_trial_id": trial_id,
                    "source_sha256": source_sha256,
                    "qualification": False,
                    "error": detail,
                    "checks": [
                        {
                            "name": "native_probe_completed",
                            "measured": 0.0,
                            "operator": "==",
                            "threshold": 1.0,
                            "unit": "boolean",
                            "passed": False,
                        }
                    ],
                    "overall_pass": False,
                    "artifacts": [str(trace_path.relative_to(output_path.parent.parent))]
                    if trace_path.exists()
                    else [],
                },
            )
        raise
    finally:
        app.close()
