#!/usr/bin/env python3
"""Native compliant-mimic probe for the frozen DRG80 thread law.

This bounded preflight follows honestly failed explicit-callback, fixed-tendon,
and post-step-velocity energy-quadrature baselines.  It does not qualify the
gripper environment.
A positive-compliance native PhysX articulation mimic realizes

    e = z - p*theta, U = 0.5*K*e**2,
    K = 1e6 N/m, D = 4000 N*s/m, p = 0.001/(2*pi) m/rad.

The probe applies a persistently exciting generalized load parallel to grad(e),
holds a static load, then releases both dynamic DOFs.  It infers K and D from
fresh native acceleration/reaction balance and uses the measured K and D for
release energy and dissipation.  Release dissipation is approximated from the
measured extension increment between consecutive fresh post-step positions.
That interval-mean-rate-squared quadrature is a lower bound on unresolved
within-step viscous heat, not an exact heat measurement.  The previous baseline
incorrectly used only the projected endpoint velocity, which does not include
the compliant solver's within-step position correction.  Both quadratures are
retained in the trace and no residual is filled with a constant or proxy.
The mimic uses zero velocity iterations: the
pinned shared PhysX solver's velocity-iteration branch hard-projects relative
velocity even for positive compliance, which made D unobservable in the failed
four-velocity-iteration trial.  No pose, velocity, or joint-state writes occur
after reset.
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
from isaaclab_physx.physics import PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


DT = 1.0 / 240.0
LEAD_M_PER_REV = 0.001
P_M_PER_RAD = LEAD_M_PER_REV / (2.0 * math.pi)
K_N_PER_M = 1.0e6
D_NS_PER_M = 4000.0
CARRIER_MASS_KG = 0.01
ROTOR_MASS_KG = 0.01
ROTOR_INERTIA_KG_M2 = 1.0e-6
RAMP_STEPS = 6
IDENTIFY_STEPS = 960
HOLD_STEPS = 120
RELEASE_STEPS = 480
TEST_EXTENSION_FORCE_N = 1.0
AXIAL_LIMIT_M = 0.005
ROTATION_LIMIT_RAD = math.radians(1440.0)
AXIAL_LIMIT_CLEARANCE_MIN_M = 5.0e-4
ROTATION_LIMIT_CLEARANCE_MIN_RAD = 1.0
POSITION_ITERATIONS = 128
VELOCITY_ITERATIONS = 0
MIMIC_AXIS = "rotZ"
CONSTITUTIVE_RELATIVE_ERROR_MAX = 0.10
RELEASE_ENERGY_BALANCE_RELATIVE_ERROR_MAX = 0.25
# For native C=theta-z/p, kp_C=K*p^2 and kd_C=D*p^2.  The shared
# CPU/GPU PhysX mimic solver uses kp=mu^2/r and kd=2*mu*zeta/r.  These
# values follow from the declared bare-mechanism mass matrix, not tuning.
ANALYTIC_M_AXIAL = CARRIER_MASS_KG + ROTOR_MASS_KG
ANALYTIC_M_ROTATION = ROTOR_INERTIA_KG_M2
ANALYTIC_R = (1.0 / P_M_PER_RAD**2) / ANALYTIC_M_AXIAL + 1.0 / ANALYTIC_M_ROTATION
KP_C = K_N_PER_M * P_M_PER_RAD**2
KD_C = D_NS_PER_M * P_M_PER_RAD**2
MIMIC_NATURAL_FREQUENCY_S_INV = math.sqrt(KP_C * ANALYTIC_R)
MIMIC_DAMPING_RATIO = KD_C * ANALYTIC_R / (
    2.0 * MIMIC_NATURAL_FREQUENCY_S_INV
)


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

    # Installed native discovery established that the revolute coordinate is
    # authored in degrees for this schema and that -360000 degree/metre gives
    # the required one-millimetre lead.  Positive compliance is derived from
    # the frozen physical K,D and bare-mechanism inverse mass response.
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(rotation.GetPrim(), MIMIC_AXIS)
    mimic.CreateReferenceJointRel().SetTargets([axial.GetPath()])
    mimic.CreateReferenceJointAxisAttr("rotX")
    mimic.CreateGearingAttr(-360.0 / LEAD_M_PER_REV)
    mimic.CreateOffsetAttr(0.0)
    mimic.CreateNaturalFrequencyAttr(MIMIC_NATURAL_FREQUENCY_S_INV)
    mimic.CreateDampingRatioAttr(MIMIC_DAMPING_RATIO)

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

    authored_frequency = float(mimic.GetNaturalFrequencyAttr().Get())
    authored_damping_ratio = float(mimic.GetDampingRatioAttr().Get())
    authored_gearing = float(mimic.GetGearingAttr().Get())
    authored_position_iterations = int(
        articulation_api.GetSolverPositionIterationCountAttr().Get()
    )
    authored_velocity_iterations = int(
        articulation_api.GetSolverVelocityIterationCountAttr().Get()
    )
    METADATA = {
        "native_trial_id": trial_id,
        "source_sha256": source_sha256,
        "scope": "positive-compliance mimic free-release component preflight only",
        "dof_names": dof_names,
        "physics_dt_s": DT,
        "thread_stiffness_n_per_m": K_N_PER_M,
        "thread_damping_ns_per_m": D_NS_PER_M,
        "lead_m_per_revolution": LEAD_M_PER_REV,
        "test_extension_force_n": TEST_EXTENSION_FORCE_N,
        "analytic_inverse_mass_response": ANALYTIC_R,
        "kp_constraint_coordinate": KP_C,
        "kd_constraint_coordinate": KD_C,
        "authored_mimic_natural_frequency_s_inverse": authored_frequency,
        "authored_mimic_damping_ratio": authored_damping_ratio,
        "authored_mimic_gearing_degree_per_metre": authored_gearing,
        "authored_solver_position_iterations": authored_position_iterations,
        "authored_solver_velocity_iterations": authored_velocity_iterations,
    }

    def set_effort(extension_force_n: float) -> None:
        efforts = torch.zeros(
            (articulation.count, articulation.max_dofs),
            dtype=torch.float32,
            device=args.device,
        )
        efforts[0, axial_index] = extension_force_n
        efforts[0, rotation_index] = -P_M_PER_RAD * extension_force_n
        articulation.set_dof_actuation_forces(
            wp.from_torch(efforts.contiguous(), dtype=wp.float32), all_indices
        )

    def capture(phase: str, phase_step: int, commanded_force_n: float) -> bool:
        q = as_torch(articulation.get_dof_positions())[0]
        qd = as_torch(articulation.get_dof_velocities())[0]
        efforts = as_torch(articulation.get_dof_actuation_forces())[0]
        mass_matrix = as_torch(articulation.get_generalized_mass_matrices())[0]
        expected_shape = (articulation.max_dofs, articulation.max_dofs)
        if tuple(mass_matrix.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected generalized mass-matrix shape {tuple(mass_matrix.shape)}"
            )
        z = float(q[axial_index])
        theta = float(q[rotation_index])
        zd = float(qd[axial_index])
        thetad = float(qd[rotation_index])
        extension = z - P_M_PER_RAD * theta
        extension_rate = zd - P_M_PER_RAD * thetad
        kinetic_energy = 0.5 * float(qd @ mass_matrix @ qd)
        values = np.asarray(
            [
                *q.tolist(),
                *qd.tolist(),
                *efforts.tolist(),
                *mass_matrix.flatten().tolist(),
                extension,
                extension_rate,
                kinetic_energy,
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
                "commanded_extension_force_n": commanded_force_n,
                "extension_m": extension,
                "extension_rate_m_s": extension_rate,
                "kinetic_energy_j": kinetic_energy,
            }
        )
        return bool(np.all(np.isfinite(values)))

    all_finite = True
    for phase, steps in (
        ("ramp", RAMP_STEPS),
        ("identify", IDENTIFY_STEPS),
        ("hold", HOLD_STEPS),
        ("released", RELEASE_STEPS),
    ):
        for phase_step in range(steps):
            if phase == "ramp":
                force_n = TEST_EXTENSION_FORCE_N * (phase_step + 1) / RAMP_STEPS
            elif phase == "identify":
                # Three non-harmonic components make extension and extension
                # rate independently observable.  The zero-mean bounded load
                # remains parallel to grad(e); it is not a pose/velocity drive.
                time_s = phase_step * DT
                force_n = TEST_EXTENSION_FORCE_N * (
                    0.60 * math.sin(2.0 * math.pi * 5.0 * time_s)
                    + 0.30 * math.sin(2.0 * math.pi * 17.0 * time_s)
                    + 0.10 * math.sin(2.0 * math.pi * 31.0 * time_s)
                )
            elif phase == "hold":
                force_n = TEST_EXTENSION_FORCE_N
            else:
                force_n = 0.0
            set_effort(force_n)
            sim.step(render=False)
            all_finite = capture(phase, phase_step, force_n) and all_finite
            if not all_finite:
                break
        if not all_finite:
            break

    q = np.asarray([sample["dof_positions"] for sample in SAMPLES], dtype=np.float64)
    qd = np.asarray([sample["dof_velocities"] for sample in SAMPLES], dtype=np.float64)
    efforts = np.asarray(
        [sample["dof_actuation_forces"] for sample in SAMPLES], dtype=np.float64
    )
    mass_matrices = np.asarray(
        [sample["generalized_mass_matrix"] for sample in SAMPLES], dtype=np.float64
    )
    extension = np.asarray([sample["extension_m"] for sample in SAMPLES], dtype=np.float64)
    commanded = np.asarray(
        [sample["commanded_extension_force_n"] for sample in SAMPLES], dtype=np.float64
    )
    finite_values = np.column_stack(
        (q, qd, efforts, extension, mass_matrices.reshape(len(SAMPLES), -1))
    )
    finite_fraction = float(np.isfinite(finite_values).mean()) if finite_values.size else 0.0
    identify_start = RAMP_STEPS
    hold_start = identify_start + IDENTIFY_STEPS
    release_start = hold_start + HOLD_STEPS
    expected_sample_count = release_start + RELEASE_STEPS
    have_complete_trace = len(SAMPLES) == expected_sample_count

    # For the post-step sample k, backward velocity difference is the native
    # acceleration across the just-completed integration interval.  With
    # gravity disabled, no shapes/contacts, and limits checked separately,
    # Q_native = M(q_k)*(v_k-v_{k-1})/dt - Q_user is the measured passive
    # generalized reaction.  The frozen law predicts
    # Q_native = -(K*e + D*edot)*[1,-p].
    acceleration = np.full_like(qd, np.nan)
    generalized_reaction = np.full_like(qd, np.nan)
    acceleration[1:] = (qd[1:] - qd[:-1]) / DT
    generalized_reaction[1:] = np.einsum(
        "nij,nj->ni", mass_matrices[1:], acceleration[1:]
    ) - efforts[1:]
    measured_force_n = -generalized_reaction[:, axial_index]

    identify_indices = np.arange(identify_start + 1, hold_start, dtype=np.int64)
    hold_indices = np.arange(release_start - 60, release_start, dtype=np.int64)
    extension_rate = qd[:, axial_index] - P_M_PER_RAD * qd[:, rotation_index]
    scaled_design = np.column_stack(
        (
            K_N_PER_M * extension[identify_indices],
            D_NS_PER_M * extension_rate[identify_indices],
        )
    )
    response = measured_force_n[identify_indices]
    singular_values = np.linalg.svd(scaled_design, compute_uv=False)
    design_condition = float(singular_values[0] / max(singular_values[-1], 1e-30))
    # K is measured in a settled constant-load window where the damping term
    # is negligible.  Holding that measured K fixed, D is identified from the
    # independent multi-frequency transient window.  This prevents a single
    # ill-conditioned two-parameter fit from trading stiffness for damping.
    settled_force = measured_force_n[hold_indices]
    settled_extension = extension[hold_indices]
    inferred_k = float(
        np.dot(settled_extension, settled_force)
        / max(float(np.dot(settled_extension, settled_extension)), 1e-30)
    )
    damping_response = response - inferred_k * extension[identify_indices]
    transient_rate = extension_rate[identify_indices]
    inferred_d = float(
        np.dot(transient_rate, damping_response)
        / max(float(np.dot(transient_rate, transient_rate)), 1e-30)
    )
    settled_extension_rate_max = float(np.max(np.abs(extension_rate[hold_indices])))
    settled_acceleration_max = float(
        np.max(np.abs(acceleration[hold_indices, axial_index]))
    )
    transient_extension_rate_rms = float(np.sqrt(np.mean(transient_rate**2)))
    constitutive_prediction = (
        inferred_k * extension + inferred_d
        * (qd[:, axial_index] - P_M_PER_RAD * qd[:, rotation_index])
    )
    fit_residual = response - constitutive_prediction[identify_indices]
    fit_relative_rmse = float(np.sqrt(np.mean(fit_residual**2))) / max(
        float(np.sqrt(np.mean(response**2))), 1e-12
    )
    stiffness_relative_error = abs(inferred_k - K_N_PER_M) / K_N_PER_M
    damping_relative_error = abs(inferred_d - D_NS_PER_M) / D_NS_PER_M

    # Check the equal-and-opposite generalized torque component independently.
    # Scaling by p converts torque residual to the equivalent axial-force unit.
    reaction_torque_force_error = float(
        np.sqrt(
            np.mean(
                (
                    generalized_reaction[identify_indices, rotation_index]
                    / (-P_M_PER_RAD)
                    - generalized_reaction[identify_indices, axial_index]
                )
                ** 2
            )
        )
    )

    measured_elastic_energy = 0.5 * inferred_k * extension**2
    measured_kinetic_energy = np.asarray(
        [sample["kinetic_energy_j"] for sample in SAMPLES], dtype=np.float64
    )
    measured_energy = measured_elastic_energy + measured_kinetic_energy
    measured_damping_power = inferred_d * extension_rate**2
    for index, sample in enumerate(SAMPLES):
        sample["measured_constitutive_force_n"] = float(constitutive_prediction[index])
        sample["measured_elastic_energy_j"] = float(measured_elastic_energy[index])
        sample["measured_mechanical_energy_j"] = float(measured_energy[index])
        sample["measured_damping_power_w"] = float(measured_damping_power[index])
        if index > 0:
            sample["measured_generalized_reaction"] = generalized_reaction[index].tolist()

    if have_complete_trace:
        baseline_extension = abs(float(extension[release_start - 1]))
        baseline_energy = float(measured_energy[release_start - 1])
        release_extension = np.abs(extension[release_start:])
        release_energy = measured_energy[release_start:]
        peak_extension_ratio = float(np.max(release_extension)) / max(baseline_extension, 1e-15)
        final_extension_ratio = float(release_extension[-1]) / max(baseline_extension, 1e-15)
        peak_energy_ratio = float(np.max(release_energy)) / max(baseline_energy, 1e-15)
        # Each SAMPLES[k] entry is a fresh state after one native physics step.
        # For the release interval ending at k, the measured generalized
        # extension increment is e[k]-e[k-1].  Its quotient by the actual DT
        # is the only rate directly consistent with those two measured native
        # positions over that completed interval.  The endpoint qdot getter is
        # also recorded, but the compliant mimic's implicit position solve can
        # correct position without reflecting that correction in the projected
        # post-solve velocity.  Using only qdot[k] therefore omitted real
        # measured displacement and caused the retained failed 0.7006 result.
        released_interval_extension_rate = np.diff(
            extension[release_start - 1:]
        ) / DT
        released_damping_loss = float(
            np.sum(inferred_d * released_interval_extension_rate**2) * DT
        )
        released_endpoint_velocity_damping_loss = float(
            np.sum(measured_damping_power[release_start:]) * DT
        )
        released_interval_rate_finite_fraction = float(
            np.isfinite(released_interval_extension_rate).mean()
        )
        observed_release_energy_loss = baseline_energy - float(release_energy[-1])
        release_energy_balance_relative_error = abs(
            released_damping_loss - observed_release_energy_loss
        ) / max(baseline_energy, 1e-15)
    else:
        baseline_extension = 0.0
        baseline_energy = 0.0
        peak_extension_ratio = math.inf
        final_extension_ratio = math.inf
        peak_energy_ratio = math.inf
        released_damping_loss = 0.0
        released_endpoint_velocity_damping_loss = 0.0
        released_interval_rate_finite_fraction = 0.0
        observed_release_energy_loss = 0.0
        release_energy_balance_relative_error = math.inf

    expected_axial = commanded
    expected_rotor = -P_M_PER_RAD * commanded
    axial_effort_error = float(np.max(np.abs(efforts[:, axial_index] - expected_axial)))
    rotor_effort_error = float(np.max(np.abs(efforts[:, rotation_index] - expected_rotor)))
    mass_symmetry_error = float(
        np.max(np.abs(mass_matrices - np.swapaxes(mass_matrices, 1, 2)))
    )
    min_mass_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(
                0.5 * (mass_matrices + np.swapaxes(mass_matrices, 1, 2))
            )
        )
    )
    axial_clearance = float(np.min(AXIAL_LIMIT_M - np.abs(q[:, axial_index])))
    rotation_clearance = float(np.min(ROTATION_LIMIT_RAD - np.abs(q[:, rotation_index])))
    actual_dt = float(sim.get_physics_dt())
    initial_mass_matrix = mass_matrices[0]
    constraint_jacobian = np.asarray([-1.0 / P_M_PER_RAD, 1.0])
    native_inverse_mass_response = float(
        constraint_jacobian @ np.linalg.inv(initial_mass_matrix) @ constraint_jacobian
    )
    native_mapped_frequency = math.sqrt(KP_C * native_inverse_mass_response)
    native_mapped_damping_ratio = (
        KD_C
        * native_inverse_mass_response
        / (2.0 * native_mapped_frequency)
    )
    analytic_mass_matrix = np.diag([ANALYTIC_M_AXIAL, ANALYTIC_M_ROTATION])
    initial_mass_matrix_error = float(
        np.max(np.abs(initial_mass_matrix - analytic_mass_matrix))
    )
    frequency_mapping_relative_error = abs(
        authored_frequency - native_mapped_frequency
    ) / native_mapped_frequency
    damping_mapping_relative_error = abs(
        authored_damping_ratio - native_mapped_damping_ratio
    ) / native_mapped_damping_ratio
    gearing_error = abs(authored_gearing + 360.0 / LEAD_M_PER_REV)

    checks = [
        check("native_articulation_count", articulation.count, "==", 1, "articulations"),
        check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
        check("native_collision_shape_count", articulation.max_shapes, "==", 0, "shapes"),
        check("authored_solver_position_iterations", authored_position_iterations, "==", POSITION_ITERATIONS, "iterations"),
        check("authored_solver_velocity_iterations", authored_velocity_iterations, "==", 0, "iterations"),
        check("actual_physics_dt_absolute_error", abs(actual_dt - DT), "<=", 1e-9, "s"),
        check("fresh_native_sample_count", len(SAMPLES), "==", expected_sample_count, "physics steps"),
        check("native_state_finite_fraction", finite_fraction, "==", 1.0, "fraction"),
        check("initial_native_mass_matrix_error_max", initial_mass_matrix_error, "<=", 1e-7, "mixed SI generalized inertia"),
        check("mimic_gearing_readback_error", gearing_error, "<=", 1e-3, "degree/m"),
        check("native_mass_mapped_frequency_relative_error", frequency_mapping_relative_error, "<=", 1e-3, "fraction"),
        check("native_mass_mapped_damping_ratio_relative_error", damping_mapping_relative_error, "<=", 1e-3, "fraction"),
        check("generalized_mass_matrix_symmetry_error_max", mass_symmetry_error, "<=", 1e-7, "mixed SI generalized inertia"),
        check("generalized_mass_matrix_eigenvalue_min", min_mass_eigenvalue, ">", 0.0, "mixed SI generalized inertia"),
        check("axial_joint_limit_clearance_min", axial_clearance, ">=", AXIAL_LIMIT_CLEARANCE_MIN_M, "m"),
        check("rotation_joint_limit_clearance_min", rotation_clearance, ">=", ROTATION_LIMIT_CLEARANCE_MIN_RAD, "rad"),
        check("native_applied_axial_excitation_error_max", axial_effort_error, "<=", 5e-6, "N"),
        check("native_applied_rotor_excitation_error_max", rotor_effort_error, "<=", 5e-9, "N*m"),
        check("constitutive_identification_design_condition_max", design_condition, "<=", 100.0, "condition number"),
        check("settled_extension_rate_max", settled_extension_rate_max, "<=", 1e-6, "m/s"),
        check("settled_axial_acceleration_max", settled_acceleration_max, "<=", 1e-4, "m/s^2"),
        check("transient_extension_rate_rms_min", transient_extension_rate_rms, ">=", 1e-6, "m/s"),
        check("native_inferred_stiffness_relative_error", stiffness_relative_error, "<=", CONSTITUTIVE_RELATIVE_ERROR_MAX, "fraction"),
        check("native_inferred_damping_relative_error", damping_relative_error, "<=", CONSTITUTIVE_RELATIVE_ERROR_MAX, "fraction"),
        check("native_constitutive_fit_relative_rmse", fit_relative_rmse, "<=", CONSTITUTIVE_RELATIVE_ERROR_MAX, "fraction"),
        check("reaction_axial_torque_equivalent_rmse", reaction_torque_force_error, "<=", 0.10, "N equivalent"),
        check("pre_release_hold_extension_magnitude_min", baseline_extension, ">=", 1e-7, "m"),
        check("pre_release_hold_mechanical_energy_min", baseline_energy, ">", 0.0, "J"),
        check("released_peak_extension_ratio_max", peak_extension_ratio, "<=", 2.0, "ratio"),
        check("released_peak_mechanical_energy_ratio_max", peak_energy_ratio, "<=", 1.05, "ratio"),
        check("released_final_extension_ratio_max", final_extension_ratio, "<=", 0.1, "ratio"),
        check("released_interval_rate_finite_fraction", released_interval_rate_finite_fraction, "==", 1.0, "fraction"),
        check("released_measured_energy_balance_relative_error", release_energy_balance_relative_error, "<=", RELEASE_ENERGY_BALANCE_RELATIVE_ERROR_MAX, "fraction"),
    ]
    write_json(
        trace_path,
        json_safe(
            {
                **METADATA,
                "status": "validation_complete",
                "actual_physics_dt_s": actual_dt,
                "release_baseline_sample_index": release_start - 1,
                "constitutive_identification": {
                    "inferred_stiffness_n_per_m": inferred_k,
                    "inferred_damping_ns_per_m": inferred_d,
                    "fit_relative_rmse": fit_relative_rmse,
                    "design_condition": design_condition,
                    "settled_window_sample_indices": hold_indices.tolist(),
                    "transient_window_sample_indices": identify_indices.tolist(),
                    "settled_extension_rate_max_m_s": settled_extension_rate_max,
                    "settled_axial_acceleration_max_m_s2": settled_acceleration_max,
                    "transient_extension_rate_rms_m_s": transient_extension_rate_rms,
                    "released_damping_loss_j": released_damping_loss,
                    "released_endpoint_velocity_damping_loss_j": released_endpoint_velocity_damping_loss,
                    "released_interval_rate_finite_fraction": released_interval_rate_finite_fraction,
                    "released_damping_quadrature_method": "D times squared measured interval-mean extension rate times dt; lower bound on unresolved within-step viscous heat",
                    "observed_release_energy_loss_j": observed_release_energy_loss,
                    "release_energy_balance_relative_error": release_energy_balance_relative_error,
                },
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
            "actual_physics_dt_s": actual_dt,
            "checks": checks,
            "overall_pass": all(item["passed"] for item in checks),
            "artifacts": [str(trace_path.relative_to(output_path.parent.parent))],
        },
    )
    if not all(item["passed"] for item in checks):
        raise RuntimeError(
            "positive-compliance mimic stability preflight failed: "
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
                        "scope": "positive-compliance mimic free-release component preflight only",
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
