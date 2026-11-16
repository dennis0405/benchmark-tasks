#!/usr/bin/env python3
"""Bounded native loaded-mimic/contact-tensor discovery; not qualification.

The previously passed coupon established unloaded helical motion.  This probe
retains the failed coupon's full +/-0.005 N*m rotor load, but replaces the
ambiguous projected-joint-force proxy with collision against two immobile,
individually filtered load-cell plates.  It records native DOF, body, pairwise
contact-force, and per-patch contact data after every 240 Hz integration step.

No DRG fixture, thread friction, nut preload, robot, scorer, or YAM success is
represented or claimed here.
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
# The prior failed coupon used 1e-6 kg*m^2 for every diagnostic link.  Its
# measured steady loaded mimic error (0.0595967 rad) was of the same order as
# one step of unconstrained rotor acceleration, tau*dt^2/I = 0.0868 rad, even
# though the independently filtered plate force matched 2*pi*tau/lead.  The
# production adjustment is driven with a seated socket/driver, not a bare
# massless coordinate.  This bounded repair therefore represents the combined
# screw-and-seated-tool rotational inertia explicitly while retaining the same
# torque, timestep, contact load, gearing, and acceptance threshold.  It is a
# numerical-conditioning discovery input, not a claimed SCHUNK mass property.
LINK_INERTIA_KG_M2 = 1.0e-6
ROTOR_WITH_TOOL_INERTIA_KG_M2 = 1.0e-5
SETTLE_STEPS = 240
LOAD_STEPS = 960
RELEASE_STEPS = 240
# Start at geometric touch.  The failed native run established that the full
# 0.005 N*m load moves the carrier only about 31 um before the hard mimic
# accumulates error, so the former 0.5 mm clearance made contact impossible
# and did not test the intended contact-loaded mechanism at all.
PLATE_GAP_M = 0.0
POSITION_ITERATIONS = 128
# The installed TGS runtime warns that values above four changed semantics.
# This coupon relies on position iterations for constraint convergence and
# keeps the velocity count at the runtime's non-warning ceiling.
VELOCITY_ITERATIONS = 4


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


def create_dynamic_cube(
    stage,
    name: str,
    mass_kg: float,
    *,
    diagonal_inertia_kg_m2: float = LINK_INERTIA_KG_M2,
    collision: bool = False,
    contact_reporter: bool = False,
):
    cube = UsdGeom.Cube.Define(stage, f"/World/Helix/{name}")
    cube.CreateSizeAttr(0.01)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(
        Gf.Vec3f(
            diagonal_inertia_kg_m2,
            diagonal_inertia_kg_m2,
            diagonal_inertia_kg_m2,
        )
    )
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim())
    body.CreateDisableGravityAttr(True)
    if contact_reporter:
        if not collision:
            raise ValueError("a contact-reporting link must have collision enabled")
        PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr(0.0)
    return cube.GetPath()


def create_plate(stage, name: str, z_m: float):
    plate = UsdGeom.Cube.Define(stage, f"/World/{name}")
    plate.CreateSizeAttr(1.0)
    plate.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_m))
    plate.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.004))
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(plate.GetPrim())
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    UsdPhysics.MassAPI.Apply(plate.GetPrim()).CreateMassAttr(100.0)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(plate.GetPrim())
    body.CreateDisableGravityAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(plate.GetPrim()).CreateThresholdAttr(0.0)
    return plate.GetPath()


def as_torch(value) -> torch.Tensor:
    return wp.to_torch(value).detach().cpu().clone()


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    stage = sim.stage
    root = UsdGeom.Xform.Define(stage, "/World/Helix")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    articulation_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)

    # Frame and rotor are inertial articulation links only.  Giving these
    # co-located diagnostic links collision geometry creates parallel hidden
    # load paths to the plates.  The carrier is the sole load-bearing platen,
    # so every plate reaction is both intended and observed by contact_view.
    frame = create_dynamic_cube(stage, "frame", 1.0)
    carrier = create_dynamic_cube(
        stage,
        "carrier",
        0.01,
        collision=True,
        contact_reporter=True,
    )
    rotor = create_dynamic_cube(
        stage,
        "rotor",
        0.01,
        diagonal_inertia_kg_m2=ROTOR_WITH_TOOL_INERTIA_KG_M2,
    )

    half_carrier = 0.005
    half_plate = 0.002
    offset = half_carrier + half_plate + PLATE_GAP_M
    lower_plate = create_plate(stage, "LowerPlate", 0.5 - offset)
    upper_plate = create_plate(stage, "UpperPlate", 0.5 + offset)

    mount = UsdPhysics.FixedJoint.Define(stage, "/World/Helix/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    axial = UsdPhysics.PrismaticJoint.Define(stage, "/World/Helix/axial")
    rotation = UsdPhysics.RevoluteJoint.Define(stage, "/World/Helix/rotation")
    for joint, parent, child in ((axial, frame, carrier), (rotation, carrier, rotor)):
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

    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    if physics is None:
        raise RuntimeError("native PhysX simulation view is unavailable after reset")
    physics.set_subspace_roots("/")
    articulation = physics.create_articulation_view("/World/Helix")
    dof_names = list(articulation.get_metatype(0).dof_names)
    if articulation.count != 1 or articulation.max_dofs != 2:
        raise RuntimeError(
            f"unexpected articulation shape: count={articulation.count}, dofs={articulation.max_dofs}"
        )
    axial_index = dof_names.index("axial")
    rotation_index = dof_names.index("rotation")
    carrier_view = physics.create_rigid_body_view([str(carrier)])
    plate_view = physics.create_rigid_body_view([str(lower_plate), str(upper_plate)])
    contact_view = physics.create_rigid_contact_view(
        [str(carrier)],
        filter_patterns=[[str(lower_plate), str(upper_plate)]],
        max_contact_data_count=32,
    )
    if (
        carrier_view.count != 1
        or plate_view.count != 2
        or contact_view.sensor_count != 1
        or contact_view.filter_count != 2
    ):
        raise RuntimeError(
            "unexpected native views: "
            f"carrier={carrier_view.count}, plates={plate_view.count}, "
            f"contact=({contact_view.sensor_count},{contact_view.filter_count})"
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
            positions = as_torch(articulation.get_dof_positions())[0]
            velocities = as_torch(articulation.get_dof_velocities())[0]
            applied_efforts = as_torch(articulation.get_dof_actuation_forces())[0]
            matrix = as_torch(contact_view.get_contact_force_matrix(dt=DT)).reshape(1, 2, 3)
            net = as_torch(contact_view.get_net_contact_forces(dt=DT)).reshape(1, 3)
            contact_data = contact_view.get_contact_data(dt=DT)
            if len(contact_data) != 6:
                raise RuntimeError(f"unexpected contact-data tuple length {len(contact_data)}")
            forces, points, normals, separations, counts, starts = map(as_torch, contact_data)
            impulse_forces = as_torch(contact_view.get_contact_data(dt=1.0)[0])
            count_array = counts.to(torch.int64).reshape(1, 2)
            start_array = starts.to(torch.int64).reshape(1, 2)
            patch_force_sums = []
            patch_impulse_sums = []
            for filter_index in range(2):
                count = int(count_array[0, filter_index].item())
                start = int(start_array[0, filter_index].item()) if count else 0
                patch_force_sums.append(float(forces.flatten()[start : start + count].sum().item()))
                patch_impulse_sums.append(
                    float(impulse_forces.flatten()[start : start + count].sum().item())
                )
            samples.append(
                {
                    "phase": name,
                    "phase_step": phase_step,
                    "dof_positions": positions.tolist(),
                    "dof_velocities": velocities.tolist(),
                    "dof_actuation_forces": applied_efforts.tolist(),
                    "carrier_transform_xyzw": as_torch(carrier_view.get_transforms())[0].tolist(),
                    "plate_transforms_xyzw": as_torch(plate_view.get_transforms()).tolist(),
                    "contact_force_matrix_n": matrix[0].tolist(),
                    "net_contact_force_n": net[0].tolist(),
                    "patch_force_sums_n": patch_force_sums,
                    "patch_impulse_sums_ns": patch_impulse_sums,
                    "contact_counts": count_array[0].tolist(),
                    "contact_points_w": points.tolist(),
                    "contact_normals_w": normals.tolist(),
                    "contact_separations_m": separations.tolist(),
                }
            )

    run_phase("settle", SETTLE_STEPS, 0.0)
    run_phase("positive_torque", LOAD_STEPS, APPLIED_TORQUE_NM)
    run_phase("released_mid", RELEASE_STEPS, 0.0)
    run_phase("negative_torque", 2 * LOAD_STEPS, -APPLIED_TORQUE_NM)
    run_phase("released_final", RELEASE_STEPS, 0.0)

    positions = np.asarray([sample["dof_positions"] for sample in samples], dtype=np.float64)
    velocities = np.asarray([sample["dof_velocities"] for sample in samples], dtype=np.float64)
    applied_efforts = np.asarray(
        [sample["dof_actuation_forces"] for sample in samples], dtype=np.float64
    )
    matrices = np.asarray([sample["contact_force_matrix_n"] for sample in samples], dtype=np.float64)
    net_forces = np.asarray([sample["net_contact_force_n"] for sample in samples], dtype=np.float64)
    patch_forces = np.asarray([sample["patch_force_sums_n"] for sample in samples], dtype=np.float64)
    patch_impulses = np.asarray(
        [sample["patch_impulse_sums_ns"] for sample in samples], dtype=np.float64
    )
    finite_values = np.concatenate(
        (
            positions.ravel(),
            velocities.ravel(),
            applied_efforts.ravel(),
            matrices.ravel(),
            net_forces.ravel(),
            patch_forces.ravel(),
        )
    )
    residual = (
        positions[:, rotation_index]
        - positions[0, rotation_index]
        - (2.0 * math.pi / LEAD_M_PER_REV)
        * (positions[:, axial_index] - positions[0, axial_index])
    )

    positive_start = SETTLE_STEPS
    positive_slice = slice(positive_start + LOAD_STEPS // 2, positive_start + LOAD_STEPS)
    negative_start = SETTLE_STEPS + LOAD_STEPS + RELEASE_STEPS
    negative_slice = slice(negative_start + LOAD_STEPS, negative_start + 2 * LOAD_STEPS)
    positive_contact = np.linalg.norm(matrices[positive_slice], axis=2).sum(axis=1)
    negative_contact = np.linalg.norm(matrices[negative_slice], axis=2).sum(axis=1)
    positive_force_median = float(np.median(positive_contact))
    negative_force_median = float(np.median(negative_contact))
    # The signed load observable is exclusively the sum of the two explicitly
    # filtered carrier/plate pairs.  The unfiltered net getter is retained only
    # as an independent check that no third-body carrier contact contributed.
    filtered_net = matrices.sum(axis=1)
    positive_z_median = float(np.median(filtered_net[positive_slice, 2]))
    negative_z_median = float(np.median(filtered_net[negative_slice, 2]))
    filtered_net_error = float(np.max(np.abs(filtered_net - net_forces)))
    positive_applied = applied_efforts[positive_slice, rotation_index]
    negative_applied = applied_efforts[negative_slice, rotation_index]
    positive_effort_min = float(np.min(positive_applied))
    negative_effort_max = float(np.max(negative_applied))
    positive_effort_median = float(np.median(positive_applied))
    negative_effort_median = float(np.median(negative_applied))
    positive_effort_error = float(np.max(np.abs(positive_applied - APPLIED_TORQUE_NM)))
    negative_effort_error = float(np.max(np.abs(negative_applied + APPLIED_TORQUE_NM)))
    loaded_indices = np.r_[
        np.arange(*positive_slice.indices(len(samples))),
        np.arange(*negative_slice.indices(len(samples))),
    ]
    loaded_residual_max = float(np.max(np.abs(residual[loaded_indices])))
    contact_samples = np.where(patch_forces.sum(axis=1) > 0.0)[0]
    # Evaluate the scaling identity over every fresh sample.  Zero-contact
    # samples contribute measured zero on both sides; importantly, a run with
    # no contact now reaches the numerical checks and emits its complete trace
    # instead of collapsing to a one-bit completion sentinel.
    force_impulse_error = float(
        np.max(np.abs(patch_forces * DT - patch_impulses))
    )
    expected_positive_force = 2.0 * math.pi * abs(positive_effort_median) / LEAD_M_PER_REV
    expected_negative_force = 2.0 * math.pi * abs(negative_effort_median) / LEAD_M_PER_REV
    positive_force_ratio = positive_force_median / expected_positive_force
    negative_force_ratio = negative_force_median / expected_negative_force
    total_steps = SETTLE_STEPS + 3 * LOAD_STEPS + 2 * RELEASE_STEPS

    checks = [
        check("native_articulation_count", articulation.count, "==", 1, "articulations"),
        check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
        check("filtered_contact_sensor_count", contact_view.sensor_count, "==", 1, "bodies"),
        check("filtered_contact_plate_count", contact_view.filter_count, "==", 2, "bodies"),
        check("actual_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
        check("fresh_native_sample_count", len(samples), "==", total_steps, "physics steps"),
        check("native_state_finite_fraction", np.isfinite(finite_values).mean(), "==", 1.0, "fraction"),
        check("positive_load_contact_force_median", positive_force_median, ">=", 1.0, "N"),
        check("negative_load_contact_force_median", negative_force_median, ">=", 1.0, "N"),
        check("positive_applied_rotor_effort_min", positive_effort_min, ">=", 0.00495, "N*m"),
        check("negative_applied_rotor_effort_max", negative_effort_max, "<=", -0.00495, "N*m"),
        check(
            "positive_applied_rotor_effort_error_max",
            positive_effort_error,
            "<=",
            5.0e-5,
            "N*m",
        ),
        check(
            "negative_applied_rotor_effort_error_max",
            negative_effort_error,
            "<=",
            5.0e-5,
            "N*m",
        ),
        check(
            "filtered_plate_force_sign_reversal",
            positive_z_median * negative_z_median,
            "<",
            0.0,
            "N^2",
        ),
        check("filtered_to_net_force_error_max", filtered_net_error, "<=", 1.0e-4, "N"),
        check("force_impulse_dt_scaling_error", force_impulse_error, "<=", 1.0e-5, "N*s"),
        check("positive_contact_to_ideal_force_ratio_min", positive_force_ratio, ">=", 0.5, "ratio"),
        check("positive_contact_to_ideal_force_ratio_max", positive_force_ratio, "<=", 1.5, "ratio"),
        check("negative_contact_to_ideal_force_ratio_min", negative_force_ratio, ">=", 0.5, "ratio"),
        check("negative_contact_to_ideal_force_ratio_max", negative_force_ratio, "<=", 1.5, "ratio"),
        check("loaded_helix_residual_max", loaded_residual_max, "<=", 0.02, "rad"),
    ]

    write_json(
        trace_path,
        {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "scope": "isolated two-sided contact-loaded native mimic discovery only",
            "dof_names": dof_names,
            "filter_paths": list(contact_view.filter_paths),
            "applied_rotor_torque_nm": APPLIED_TORQUE_NM,
            "link_diagonal_inertia_kg_m2": LINK_INERTIA_KG_M2,
            "rotor_with_seated_tool_diagonal_inertia_kg_m2": (
                ROTOR_WITH_TOOL_INERTIA_KG_M2
            ),
            "measured_positive_rotor_effort_median_nm": positive_effort_median,
            "measured_negative_rotor_effort_median_nm": negative_effort_median,
            "expected_positive_ideal_axial_force_n": expected_positive_force,
            "expected_negative_ideal_axial_force_n": expected_negative_force,
            "solver_position_iterations": POSITION_ITERATIONS,
            "solver_velocity_iterations": VELOCITY_ITERATIONS,
            "positive_contact_force_median_n": positive_force_median,
            "negative_contact_force_median_n": negative_force_median,
            "positive_filtered_plate_force_z_median_n": positive_z_median,
            "negative_filtered_plate_force_z_median_n": negative_z_median,
            "filtered_to_net_force_error_max_n": filtered_net_error,
            "loaded_helix_residual_max_rad": loaded_residual_max,
            "contact_sample_count": int(contact_samples.size),
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
            "native loaded mimic/contact discovery failed: "
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
