#!/usr/bin/env python3
"""Native serial-bearing/helical-coupling discovery for DRG80; not qualification.

The positive-compliance mimic failed its loaded material checks under both TGS
and the final bounded PGS comparison.  This probe asks one different mechanical
question: can the installed GPU solver combine the already-supported hard
helical mimic with a separate, ordinary prismatic bearing spring/damper and
transmit the frozen 0.005 N*m test torque into signed bilateral plate contact?

The bearing drive retains K=1e6 N/m and D=4000 N*s/m.  The one-mm/revolution
lead, existing carrier/rotor masses, 240 Hz clock, torque, and loaded
helix-error threshold are unchanged.  The new 0.01 kg support is disclosed as
part of this serial topology.  This is a proposed fidelity revision, not the
currently frozen constitutive coordinate, so even a pass cannot qualify the
fixture.  No pose, velocity, or joint-state write occurs after reset.
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
K_N_PER_M = 1.0e6
D_NS_PER_M = 4000.0
TEST_TORQUE_NM = 0.005
POSITION_ITERATIONS = 128
VELOCITY_ITERATIONS = 4
SETTLE_STEPS = 240
LOAD_STEPS = 960
RELEASE_STEPS = 240
EXPECTED_FORCE_N = 2.0 * math.pi * TEST_TORQUE_NM / LEAD_M_PER_REV


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check(name: str, measured: float, operator: str, threshold: float, unit: str) -> dict:
    value = float(measured)
    comparisons = {
        "==": lambda a, b: a == b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
    }
    passed = math.isfinite(value) and comparisons[operator](value, float(threshold))
    return {
        "name": name,
        "measured": value,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(passed),
    }


def as_torch(value) -> torch.Tensor:
    return wp.to_torch(value).detach().cpu().clone()


def create_link(stage, name: str, mass_kg: float, inertia: float, *, collision: bool = False):
    cube = UsdGeom.Cube.Define(stage, f"/World/Helix/{name}")
    cube.CreateSizeAttr(0.01)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
    PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr(True)
    return cube.GetPath()


def create_plate(stage, name: str, z_m: float):
    plate = UsdGeom.Cube.Define(stage, f"/World/{name}")
    plate.CreateSizeAttr(1.0)
    plate.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_m))
    plate.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.004))
    rigid = UsdPhysics.RigidBodyAPI.Apply(plate.GetPrim())
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    UsdPhysics.MassAPI.Apply(plate.GetPrim()).CreateMassAttr(100.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(plate.GetPrim()).CreateDisableGravityAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(plate.GetPrim()).CreateThresholdAttr(0.0)
    return plate.GetPath()


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    stage = sim.stage
    root = UsdGeom.Xform.Define(stage, "/World/Helix")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    articulation_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)

    frame = create_link(stage, "frame", 1.0, 1.0e-3)
    support = create_link(stage, "bearing_support", 0.01, 1.0e-6)
    carrier = create_link(stage, "carrier", 0.01, 1.0e-6, collision=True)
    rotor = create_link(stage, "rotor", 0.01, 1.0e-5)

    contact_offset = 0.005 + 0.002
    lower_plate = create_plate(stage, "LowerPlate", 0.5 - contact_offset)
    upper_plate = create_plate(stage, "UpperPlate", 0.5 + contact_offset)

    mount = UsdPhysics.FixedJoint.Define(stage, "/World/Helix/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    bearing = UsdPhysics.PrismaticJoint.Define(stage, "/World/Helix/bearing")
    bearing.CreateBody0Rel().SetTargets([frame])
    bearing.CreateBody1Rel().SetTargets([support])
    bearing.CreateAxisAttr("Z")
    bearing.CreateLowerLimitAttr(-0.001)
    bearing.CreateUpperLimitAttr(0.001)
    drive = UsdPhysics.DriveAPI.Apply(bearing.GetPrim(), "linear")
    drive.CreateStiffnessAttr(K_N_PER_M)
    drive.CreateDampingAttr(D_NS_PER_M)
    drive.CreateMaxForceAttr(1000.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)

    axial = UsdPhysics.PrismaticJoint.Define(stage, "/World/Helix/axial")
    axial.CreateBody0Rel().SetTargets([support])
    axial.CreateBody1Rel().SetTargets([carrier])
    axial.CreateAxisAttr("Z")
    axial.CreateLowerLimitAttr(-0.005)
    axial.CreateUpperLimitAttr(0.005)

    rotation = UsdPhysics.RevoluteJoint.Define(stage, "/World/Helix/rotation")
    rotation.CreateBody0Rel().SetTargets([carrier])
    rotation.CreateBody1Rel().SetTargets([rotor])
    rotation.CreateAxisAttr("Z")
    rotation.CreateLowerLimitAttr(-1440.0)
    rotation.CreateUpperLimitAttr(1440.0)

    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(rotation.GetPrim(), "rotZ")
    mimic.CreateReferenceJointRel().SetTargets([axial.GetPath()])
    mimic.CreateReferenceJointAxisAttr("rotX")
    mimic.CreateGearingAttr(-360.0 / LEAD_M_PER_REV)
    mimic.CreateOffsetAttr(0.0)
    mimic.CreateNaturalFrequencyAttr(0.0)
    mimic.CreateDampingRatioAttr(0.0)

    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    if physics is None:
        raise RuntimeError("native PhysX simulation view unavailable after reset")
    physics.set_subspace_roots("/")
    articulation = physics.create_articulation_view("/World/Helix")
    if articulation.count != 1 or articulation.max_dofs != 3:
        raise RuntimeError(
            f"unexpected articulation shape {articulation.count}x{articulation.max_dofs}"
        )
    dof_names = list(articulation.get_metatype(0).dof_names)
    bearing_index = dof_names.index("bearing")
    axial_index = dof_names.index("axial")
    rotation_index = dof_names.index("rotation")
    contact_view = physics.create_rigid_contact_view(
        [str(carrier)],
        filter_patterns=[[str(lower_plate), str(upper_plate)]],
        max_contact_data_count=32,
    )
    if contact_view.sensor_count != 1 or contact_view.filter_count != 2:
        raise RuntimeError(
            f"unexpected contact view {contact_view.sensor_count}x{contact_view.filter_count}"
        )

    all_indices = wp.array([0], dtype=wp.uint32, device=args.device)
    samples: list[dict] = []

    def run_phase(name: str, steps: int, torque_nm: float) -> None:
        efforts = torch.zeros(
            (articulation.count, articulation.max_dofs),
            dtype=torch.float32,
            device=args.device,
        )
        efforts[0, rotation_index] = torque_nm
        effort_buffer = wp.from_torch(efforts.contiguous(), dtype=wp.float32)
        for phase_step in range(steps):
            articulation.set_dof_actuation_forces(effort_buffer, all_indices)
            sim.step(render=False)
            PhysxManager.raise_callback_exception_if_any()
            q = as_torch(articulation.get_dof_positions())[0]
            qd = as_torch(articulation.get_dof_velocities())[0]
            applied = as_torch(articulation.get_dof_actuation_forces())[0]
            contact = as_torch(contact_view.get_contact_force_matrix(dt=DT)).reshape(1, 2, 3)[0]
            samples.append(
                {
                    "phase": name,
                    "phase_step": phase_step,
                    "dof_positions": q.tolist(),
                    "dof_velocities": qd.tolist(),
                    "dof_actuation_forces": applied.tolist(),
                    "contact_force_matrix_n": contact.tolist(),
                }
            )

    run_phase("settle", SETTLE_STEPS, 0.0)
    run_phase("positive_torque", LOAD_STEPS, TEST_TORQUE_NM)
    run_phase("released_mid", RELEASE_STEPS, 0.0)
    run_phase("negative_torque", 2 * LOAD_STEPS, -TEST_TORQUE_NM)
    run_phase("released_final", RELEASE_STEPS, 0.0)

    positions = np.asarray([sample["dof_positions"] for sample in samples], dtype=np.float64)
    velocities = np.asarray([sample["dof_velocities"] for sample in samples], dtype=np.float64)
    efforts = np.asarray([sample["dof_actuation_forces"] for sample in samples], dtype=np.float64)
    contacts = np.asarray([sample["contact_force_matrix_n"] for sample in samples], dtype=np.float64)
    signed_contact_z = contacts[:, :, 2].sum(axis=1)
    finite_values = np.concatenate((positions.ravel(), velocities.ravel(), efforts.ravel(), contacts.ravel()))

    positive_start = SETTLE_STEPS
    positive_slice = slice(positive_start + LOAD_STEPS // 2, positive_start + LOAD_STEPS)
    negative_start = SETTLE_STEPS + LOAD_STEPS + RELEASE_STEPS
    negative_slice = slice(negative_start + LOAD_STEPS, negative_start + 2 * LOAD_STEPS)
    positive_force = float(np.median(signed_contact_z[positive_slice]))
    negative_force = float(np.median(signed_contact_z[negative_slice]))
    positive_bearing = float(np.median(positions[positive_slice, bearing_index]))
    negative_bearing = float(np.median(positions[negative_slice, bearing_index]))
    positive_rate = float(np.max(np.abs(velocities[positive_slice, bearing_index])))
    negative_rate = float(np.max(np.abs(velocities[negative_slice, bearing_index])))
    inferred_k_positive = abs(positive_force / positive_bearing) if positive_bearing != 0.0 else math.inf
    inferred_k_negative = abs(negative_force / negative_bearing) if negative_bearing != 0.0 else math.inf
    stiffness_error = max(
        abs(inferred_k_positive - K_N_PER_M) / K_N_PER_M,
        abs(inferred_k_negative - K_N_PER_M) / K_N_PER_M,
    )
    helix_residual = (
        positions[:, rotation_index]
        - positions[0, rotation_index]
        - (2.0 * math.pi / LEAD_M_PER_REV)
        * (positions[:, axial_index] - positions[0, axial_index])
    )
    loaded_indices = np.r_[
        np.arange(*positive_slice.indices(len(samples))),
        np.arange(*negative_slice.indices(len(samples))),
    ]
    loaded_residual_max = float(np.max(np.abs(helix_residual[loaded_indices])))
    effort_error = max(
        float(np.max(np.abs(efforts[positive_slice, rotation_index] - TEST_TORQUE_NM))),
        float(np.max(np.abs(efforts[negative_slice, rotation_index] + TEST_TORQUE_NM))),
    )
    released_final_bearing = float(np.max(np.abs(positions[-120:, bearing_index])))
    expected_steps = SETTLE_STEPS + 3 * LOAD_STEPS + 2 * RELEASE_STEPS

    checks = [
        check("native_articulation_count", articulation.count, "==", 1, "articulations"),
        check("native_dof_count", articulation.max_dofs, "==", 3, "dofs"),
        check("native_contact_sensor_count", contact_view.sensor_count, "==", 1, "bodies"),
        check("native_contact_filter_count", contact_view.filter_count, "==", 2, "bodies"),
        check("actual_physics_dt_absolute_error", abs(sim.get_physics_dt() - DT), "<=", 1e-9, "s"),
        check("fresh_native_sample_count", len(samples), "==", expected_steps, "physics steps"),
        check("native_state_finite_fraction", np.isfinite(finite_values).mean(), "==", 1.0, "fraction"),
        check("applied_rotor_effort_error_max", effort_error, "<=", 5e-6, "N*m"),
        check("positive_contact_reaction_magnitude_min", abs(positive_force), ">=", 1.0, "N"),
        check("negative_contact_reaction_magnitude_min", abs(negative_force), ">=", 1.0, "N"),
        check("bidirectional_contact_reaction_sign_product", positive_force * negative_force, "<", 0.0, "N^2"),
        check("positive_contact_to_ideal_force_relative_error", abs(abs(positive_force) - EXPECTED_FORCE_N) / EXPECTED_FORCE_N, "<=", 0.1, "fraction"),
        check("negative_contact_to_ideal_force_relative_error", abs(abs(negative_force) - EXPECTED_FORCE_N) / EXPECTED_FORCE_N, "<=", 0.1, "fraction"),
        check("settled_bearing_rate_max", max(positive_rate, negative_rate), "<=", 1e-6, "m/s"),
        check("native_inferred_bearing_stiffness_relative_error", stiffness_error, "<=", 0.1, "fraction"),
        check("loaded_helix_residual_max", loaded_residual_max, "<=", 0.02, "rad"),
        check("released_final_bearing_displacement_max", released_final_bearing, "<=", 5e-6, "m"),
    ]

    write_json(
        trace_path,
        {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "scope": "serial native bearing spring plus hard helical coupling discovery only",
            "qualification": False,
            "dof_names": dof_names,
            "lead_m_per_revolution": LEAD_M_PER_REV,
            "bearing_stiffness_n_per_m": K_N_PER_M,
            "bearing_damping_ns_per_m": D_NS_PER_M,
            "added_bearing_support_mass_kg": 0.01,
            "test_torque_nm": TEST_TORQUE_NM,
            "expected_ideal_contact_force_n": EXPECTED_FORCE_N,
            "positive_contact_force_z_median_n": positive_force,
            "negative_contact_force_z_median_n": negative_force,
            "positive_bearing_displacement_median_m": positive_bearing,
            "negative_bearing_displacement_median_m": negative_bearing,
            "inferred_positive_bearing_stiffness_n_per_m": inferred_k_positive,
            "inferred_negative_bearing_stiffness_n_per_m": inferred_k_negative,
            "loaded_helix_residual_max_rad": loaded_residual_max,
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
            "serial bearing/helical discovery failed: "
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
                    "artifacts": [],
                },
            )
        raise
    finally:
        app.close()
