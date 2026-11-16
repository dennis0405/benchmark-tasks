#!/usr/bin/env python3
"""Native free-release stability probe for the frozen DRG80 thread law.

This is a bounded component preflight, not environment qualification.  It uses
the already source-verified PRE_PHYSICS_STEP callback and articulation effort
API to apply the frozen conservative screw/frame law

    e = z - p*theta,  F = K*e + D*e_dot,
    K = 1e6 N/m, D = 4000 N*s/m, p = 0.001/(2*pi) m/rad.

A small rotor torque is ramped through the native DOF and then removed.  Both
DOFs remain dynamic during the release.  The probe records native positions,
velocities, applied efforts, the native generalized mass matrix, elastic
energy, kinetic energy, and damping power at every 240 Hz physics step.  It
performs no post-reset pose, velocity, or joint-state writes.  Released growth
is measured against the last fresh hold sample immediately before torque
removal, so a first-release-step spike cannot become the denominator.  The
purpose is to determine whether the explicit frozen law is numerically bounded
for a genuinely free low-mass mechanism; a failure is evidence to use a
supported native implicit realization, not permission to retune frozen K or D.
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
SAMPLES: list[dict] = []
METADATA: dict = {}


import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import IsaacEvents, PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


DT = 1.0 / 240.0
LEAD_M_PER_REV = 0.001
P_M_PER_RAD = LEAD_M_PER_REV / (2.0 * math.pi)
K_N_PER_M = 1.0e6
D_NS_PER_M = 4000.0
CARRIER_MASS_KG = 0.01
ROTOR_MASS_KG = 0.01
ROTOR_INERTIA_KG_M2 = 1.0e-6
# A 25 ms ramp and 25 ms hold excite the elastic mode without accelerating the
# free neutral screw mode into the authored joint limits during the 2 s release.
RAMP_STEPS = 6
HOLD_STEPS = 6
RELEASE_STEPS = 480
TEST_TORQUE_NM = 2.0e-4
AXIAL_LIMIT_M = 0.005
ROTATION_LIMIT_RAD = math.radians(1440.0)
AXIAL_LIMIT_CLEARANCE_MIN_M = 5.0e-4
ROTATION_LIMIT_CLEARANCE_MIN_RAD = 1.0
POSITION_ITERATIONS = 128
VELOCITY_ITERATIONS = 4


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_number(value: float) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0.0 else "-Infinity"
    return number


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return safe_number(value)
    return value


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
        "measured": safe_number(value),
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(passed),
    }


def create_link(stage, name: str, mass_kg: float, inertia: float):
    cube = UsdGeom.Cube.Define(stage, f"/World/Thread/{name}")
    cube.CreateSizeAttr(0.01)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
    PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr(True)
    return cube.GetPath()


def as_torch(value) -> torch.Tensor:
    return wp.to_torch(value).detach().cpu().clone()


def main() -> None:
    global METADATA
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    stage = sim.stage

    root = UsdGeom.Xform.Define(stage, "/World/Thread")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    articulation_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)

    frame = create_link(stage, "frame", 1.0, 1.0e-3)
    carrier = create_link(stage, "carrier", CARRIER_MASS_KG, 1.0e-6)
    rotor = create_link(stage, "rotor", ROTOR_MASS_KG, ROTOR_INERTIA_KG_M2)

    mount = UsdPhysics.FixedJoint.Define(stage, "/World/Thread/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    axial = UsdPhysics.PrismaticJoint.Define(stage, "/World/Thread/axial")
    axial.CreateBody0Rel().SetTargets([frame])
    axial.CreateBody1Rel().SetTargets([carrier])
    axial.CreateAxisAttr("Z")
    axial.CreateLowerLimitAttr(-AXIAL_LIMIT_M)
    axial.CreateUpperLimitAttr(AXIAL_LIMIT_M)

    rotation = UsdPhysics.RevoluteJoint.Define(stage, "/World/Thread/rotation")
    rotation.CreateBody0Rel().SetTargets([carrier])
    rotation.CreateBody1Rel().SetTargets([rotor])
    rotation.CreateAxisAttr("Z")
    rotation.CreateLowerLimitAttr(-math.degrees(ROTATION_LIMIT_RAD))
    rotation.CreateUpperLimitAttr(math.degrees(ROTATION_LIMIT_RAD))

    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    if physics is None:
        raise RuntimeError("native PhysX simulation view unavailable after reset")
    physics.set_subspace_roots("/")
    articulation = physics.create_articulation_view("/World/Thread")
    if articulation.count != 1 or articulation.max_dofs != 2:
        raise RuntimeError(
            f"unexpected articulation shape {articulation.count}x{articulation.max_dofs}"
        )
    dof_names = list(articulation.get_metatype(0).dof_names)
    axial_index = dof_names.index("axial")
    rotation_index = dof_names.index("rotation")
    all_indices = wp.array([0], dtype=wp.uint32, device=args.device)

    callback = {
        "count": 0,
        "dt_values": set(),
        "external_torque_nm": 0.0,
        "law_force_n": 0.0,
    }
    METADATA = {
        "native_trial_id": trial_id,
        "source_sha256": source_sha256,
        "scope": "free-release explicit frozen-thread-law component preflight only",
        "dof_names": dof_names,
        "physics_dt_s": DT,
        "thread_stiffness_n_per_m": K_N_PER_M,
        "thread_damping_ns_per_m": D_NS_PER_M,
        "lead_m_per_revolution": LEAD_M_PER_REV,
        "test_torque_nm": TEST_TORQUE_NM,
    }

    def apply_law(dt: float) -> None:
        try:
            q = as_torch(articulation.get_dof_positions())[0]
            qd = as_torch(articulation.get_dof_velocities())[0]
            extension = float(q[axial_index] - P_M_PER_RAD * q[rotation_index])
            extension_rate = float(
                qd[axial_index] - P_M_PER_RAD * qd[rotation_index]
            )
            law_force = K_N_PER_M * extension + D_NS_PER_M * extension_rate
            efforts = torch.zeros(
                (articulation.count, articulation.max_dofs),
                dtype=torch.float32,
                device=args.device,
            )
            efforts[0, axial_index] = -law_force
            efforts[0, rotation_index] = (
                callback["external_torque_nm"] + P_M_PER_RAD * law_force
            )
            articulation.set_dof_actuation_forces(
                wp.from_torch(efforts.contiguous(), dtype=wp.float32), all_indices
            )
            callback["law_force_n"] = law_force
            callback["count"] += 1
            callback["dt_values"].add(float(dt))
        except Exception as exc:
            PhysxManager.store_callback_exception(exc)

    handle = PhysxManager.register_callback(
        apply_law,
        IsaacEvents.PRE_PHYSICS_STEP,
        order=0,
        name="drg80_free_thread_stability_probe",
        wrap_weak_ref=False,
    )

    def capture(phase: str, phase_step: int) -> bool:
        q = as_torch(articulation.get_dof_positions())[0]
        qd = as_torch(articulation.get_dof_velocities())[0]
        efforts = as_torch(articulation.get_dof_actuation_forces())[0]
        mass_matrix = as_torch(articulation.get_generalized_mass_matrices())[0]
        expected_mass_shape = (articulation.max_dofs, articulation.max_dofs)
        if tuple(mass_matrix.shape) != expected_mass_shape:
            raise RuntimeError(
                "unexpected fixed-base generalized mass-matrix shape "
                f"{tuple(mass_matrix.shape)}"
            )
        z = float(q[axial_index])
        theta = float(q[rotation_index])
        zd = float(qd[axial_index])
        thetad = float(qd[rotation_index])
        extension = z - P_M_PER_RAD * theta
        extension_rate = zd - P_M_PER_RAD * thetad
        elastic_energy = 0.5 * K_N_PER_M * extension * extension
        # The serial rotor translates with the carrier, and the generalized
        # mass matrix also captures any other coupled-link contribution.  A
        # hand-written diagonal expression would omit that motion.
        kinetic_energy = 0.5 * float(qd @ mass_matrix @ qd)
        values = np.asarray(
            [
                z,
                theta,
                zd,
                thetad,
                extension,
                extension_rate,
                elastic_energy,
                kinetic_energy,
                callback["law_force_n"],
                *efforts.tolist(),
                *mass_matrix.flatten().tolist(),
            ],
            dtype=np.float64,
        )
        SAMPLES.append(
            {
                "phase": phase,
                "phase_step": phase_step,
                "dof_positions": q.tolist(),
                "dof_velocities": qd.tolist(),
                "dof_actuation_forces": efforts.tolist(),
                "generalized_mass_matrix": mass_matrix.tolist(),
                "commanded_external_torque_nm": callback["external_torque_nm"],
                "applied_thread_law_force_n": callback["law_force_n"],
                "extension_m": extension,
                "extension_rate_m_s": extension_rate,
                "elastic_energy_j": elastic_energy,
                "kinetic_energy_j": kinetic_energy,
                "mechanical_energy_j": elastic_energy + kinetic_energy,
                "damping_power_w": D_NS_PER_M * extension_rate * extension_rate,
            }
        )
        return bool(np.all(np.isfinite(values)))

    phases = (
        ("ramp", RAMP_STEPS),
        ("hold", HOLD_STEPS),
        ("released", RELEASE_STEPS),
    )
    all_finite = True
    for phase, steps in phases:
        for phase_step in range(steps):
            if phase == "ramp":
                callback["external_torque_nm"] = TEST_TORQUE_NM * (
                    (phase_step + 1) / RAMP_STEPS
                )
            elif phase == "hold":
                callback["external_torque_nm"] = TEST_TORQUE_NM
            else:
                callback["external_torque_nm"] = 0.0
            sim.step(render=False)
            PhysxManager.raise_callback_exception_if_any()
            all_finite = capture(phase, phase_step) and all_finite
            if not all_finite:
                break
        if not all_finite:
            break
    handle.deregister()

    q = np.asarray([sample["dof_positions"] for sample in SAMPLES], dtype=np.float64)
    qd = np.asarray([sample["dof_velocities"] for sample in SAMPLES], dtype=np.float64)
    extension = np.asarray([sample["extension_m"] for sample in SAMPLES], dtype=np.float64)
    energy = np.asarray([sample["mechanical_energy_j"] for sample in SAMPLES], dtype=np.float64)
    mass_matrices = np.asarray(
        [sample["generalized_mass_matrix"] for sample in SAMPLES], dtype=np.float64
    )
    applied_efforts = np.asarray(
        [sample["dof_actuation_forces"] for sample in SAMPLES], dtype=np.float64
    )
    law_force = np.asarray(
        [sample["applied_thread_law_force_n"] for sample in SAMPLES], dtype=np.float64
    )
    commanded = np.asarray(
        [sample["commanded_external_torque_nm"] for sample in SAMPLES], dtype=np.float64
    )
    finite_mask = np.isfinite(
        np.column_stack(
            (
                q,
                qd,
                extension,
                energy,
                applied_efforts,
                law_force,
                mass_matrices.reshape(len(SAMPLES), -1),
            )
        )
    )
    finite_fraction = float(finite_mask.mean()) if finite_mask.size else 0.0
    if not all_finite:
        checks = [
            check(
                "native_state_finite_fraction",
                finite_fraction,
                "==",
                1.0,
                "fraction",
            )
        ]
        write_json(
            trace_path,
            json_safe(
                {
                    **METADATA,
                    "status": "failed_nonfinite_native_state",
                    "pre_physics_callback_count": callback["count"],
                    "pre_physics_callback_dt_values_s": sorted(callback["dt_values"]),
                    "samples": SAMPLES,
                }
            ),
        )
        write_json(
            output_path,
            {
                "native_trial_id": trial_id,
                "source_sha256": source_sha256,
                "qualification": False,
                "checks": checks,
                "overall_pass": False,
                "artifacts": [str(trace_path.relative_to(output_path.parent.parent))],
            },
        )
        raise RuntimeError("non-finite native state during free thread-law release")
    release_start = RAMP_STEPS + HOLD_STEPS
    have_full_release = len(SAMPLES) == release_start + RELEASE_STEPS
    if have_full_release:
        release_extension = extension[release_start:]
        release_energy = energy[release_start:]
        # Baseline strictly precedes torque removal.  Every released sample,
        # including release_extension[0]/release_energy[0], is in the maxima.
        baseline_extension = abs(float(extension[release_start - 1]))
        baseline_energy = float(energy[release_start - 1])
        final_extension_ratio = abs(float(release_extension[-1])) / max(
            baseline_extension, 1.0e-15
        )
        peak_extension_ratio = float(np.max(np.abs(release_extension))) / max(
            baseline_extension, 1.0e-15
        )
        peak_energy_ratio = float(np.max(release_energy)) / max(
            baseline_energy, 1.0e-15
        )
    else:
        baseline_extension = 0.0
        baseline_energy = 0.0
        final_extension_ratio = math.inf
        peak_extension_ratio = math.inf
        peak_energy_ratio = math.inf

    expected_axial = -law_force
    expected_rotor = commanded + P_M_PER_RAD * law_force
    axial_effort_error = float(
        np.max(np.abs(applied_efforts[:, axial_index] - expected_axial))
    )
    rotor_effort_error = float(
        np.max(np.abs(applied_efforts[:, rotation_index] - expected_rotor))
    )
    total_steps = RAMP_STEPS + HOLD_STEPS + RELEASE_STEPS
    dt_error = (
        abs(next(iter(callback["dt_values"])) - DT)
        if len(callback["dt_values"]) == 1
        else math.inf
    )
    actual_physics_dt = float(sim.get_physics_dt())
    actual_physics_dt_error = abs(actual_physics_dt - DT)
    mass_matrix_symmetry_error = float(
        np.max(np.abs(mass_matrices - np.swapaxes(mass_matrices, 1, 2)))
    )
    minimum_mass_matrix_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(
                0.5 * (mass_matrices + np.swapaxes(mass_matrices, 1, 2))
            )
        )
    )
    axial_limit_clearance = float(np.min(AXIAL_LIMIT_M - np.abs(q[:, axial_index])))
    rotation_limit_clearance = float(
        np.min(ROTATION_LIMIT_RAD - np.abs(q[:, rotation_index]))
    )
    checks = [
        check("native_articulation_count", articulation.count, "==", 1, "articulations"),
        check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
        check(
            "actual_physics_dt_absolute_error",
            actual_physics_dt_error,
            "<=",
            1.0e-9,
            "s",
        ),
        check(
            "pre_physics_callback_count",
            callback["count"],
            "==",
            total_steps,
            "callbacks",
        ),
        check(
            "pre_physics_callback_dt_value_count",
            len(callback["dt_values"]),
            "==",
            1,
            "distinct values",
        ),
        check(
            "pre_physics_callback_dt_absolute_error",
            dt_error,
            "<=",
            1.0e-9,
            "s",
        ),
        check(
            "fresh_native_sample_count",
            len(SAMPLES),
            "==",
            total_steps,
            "physics steps",
        ),
        check("native_state_finite_fraction", finite_fraction, "==", 1.0, "fraction"),
        check(
            "generalized_mass_matrix_symmetry_error_max",
            mass_matrix_symmetry_error,
            "<=",
            1.0e-7,
            "mixed SI generalized inertia",
        ),
        check(
            "generalized_mass_matrix_eigenvalue_min",
            minimum_mass_matrix_eigenvalue,
            ">",
            0.0,
            "mixed SI generalized inertia",
        ),
        check(
            "axial_joint_limit_clearance_min",
            axial_limit_clearance,
            ">=",
            AXIAL_LIMIT_CLEARANCE_MIN_M,
            "m",
        ),
        check(
            "rotation_joint_limit_clearance_min",
            rotation_limit_clearance,
            ">=",
            ROTATION_LIMIT_CLEARANCE_MIN_RAD,
            "rad",
        ),
        check(
            "native_applied_axial_law_effort_error_max",
            axial_effort_error,
            "<=",
            5.0e-4,
            "N",
        ),
        check(
            "native_applied_rotor_law_effort_error_max",
            rotor_effort_error,
            "<=",
            5.0e-6,
            "N*m",
        ),
        check(
            "pre_release_hold_extension_magnitude_min",
            baseline_extension,
            ">=",
            1.0e-7,
            "m",
        ),
        check(
            "pre_release_hold_mechanical_energy_min",
            baseline_energy,
            ">",
            0.0,
            "J",
        ),
        check("released_peak_extension_ratio_max", peak_extension_ratio, "<=", 2.0, "ratio"),
        check("released_peak_mechanical_energy_ratio_max", peak_energy_ratio, "<=", 1.05, "ratio"),
        check("released_final_extension_ratio_max", final_extension_ratio, "<=", 0.1, "ratio"),
    ]
    trace_payload = {
        **METADATA,
        "status": "validation_complete",
        "pre_physics_callback_count": callback["count"],
        "pre_physics_callback_dt_values_s": sorted(callback["dt_values"]),
        "actual_physics_dt_s": actual_physics_dt,
        "release_baseline_sample_index": release_start - 1,
        "samples": SAMPLES,
    }
    write_json(trace_path, json_safe(trace_payload))
    write_json(
        output_path,
        {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "qualification": False,
            "actual_physics_dt_s": actual_physics_dt,
            "checks": checks,
            "overall_pass": all(item["passed"] for item in checks),
            "artifacts": [str(trace_path.relative_to(output_path.parent.parent))],
        },
    )
    if not all(item["passed"] for item in checks):
        raise RuntimeError(
            "free thread-law stability preflight failed: "
            + ", ".join(item["name"] for item in checks if not item["passed"])
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if not trace_path.exists():
            write_json(
                trace_path,
                json_safe(
                    {
                        **METADATA,
                        "native_trial_id": trial_id,
                        "source_sha256": source_sha256,
                        "scope": "free-release explicit frozen-thread-law component preflight only",
                        "status": "runtime_exception",
                        "error": detail,
                        "samples": SAMPLES,
                    }
                ),
            )
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
                    "artifacts": [str(trace_path.relative_to(output_path.parent.parent))],
                },
            )
        raise
    finally:
        app.close()
