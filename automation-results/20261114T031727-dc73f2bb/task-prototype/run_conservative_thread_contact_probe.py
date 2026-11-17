#!/usr/bin/env python3
"""Bounded native conservative thread-law/contact discovery; not qualification.

This fresh coupon keeps the previously measured two-sided contact load path and
the full +/-0.005 N*m rotor effort, but replaces the failed ideal hard mimic
with either finite constitutive profile frozen in artifacts/contract.json:

    e = z - p*theta, U = 0.5*K*e**2,
    (K,D)=(1e6 N/m,4000 N*s/m) or (1e5 N/m,2500 N*s/m).

The installed draft fixed-tendon axis API was rejected by an earlier native
trial because its observed generalized reactions did not implement that
gradient.  This probe instead evaluates the selected frozen law from fresh
native DOF state in PRE_PHYSICS_STEP and applies its two generalized forces
through the native articulation effort API.  Native plate contact remains the
independent reaction measurement.  The program does not treat K*e as a force
sensor and does not claim fixture qualification.  With the KN/DN pair this is
only an N73-relative-to-clamped-S constitutive/contact probe: it does not prove
the required equal/opposite reactions on a freely moving S body, released seat
preload, friction, or lock retention.
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
parser.add_argument(
    "--thread-stiffness",
    type=float,
    default=1.0e6,
    help="Frozen axial thread stiffness in N/m (KS=1e6 or KN=1e5).",
)
parser.add_argument(
    "--thread-damping",
    type=float,
    default=4000.0,
    help="Frozen axial thread damping in N*s/m (DS=4000 or DN=2500).",
)
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
DIAGNOSTIC_SAMPLES: list[dict] = []
DIAGNOSTIC_METADATA: dict = {}


import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import IsaacEvents, PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


DT = 1.0 / 240.0
LEAD_M_PER_REV = 0.001
P_M_PER_RAD = LEAD_M_PER_REV / (2.0 * math.pi)
P_M_PER_DEG = LEAD_M_PER_REV / 360.0
THREAD_STIFFNESS_N_PER_M = float(args.thread_stiffness)
THREAD_DAMPING_NS_PER_M = float(args.thread_damping)
APPLIED_TORQUE_NM = 0.005
LINK_INERTIA_KG_M2 = 1.0e-6
SETTLE_STEPS = 480
LOAD_STEPS = 1440
RELEASE_STEPS = 480
POSITION_ITERATIONS = 128
VELOCITY_ITERATIONS = 4


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_safe(value):
    """Keep diagnostic evidence JSON-serializable even after a non-finite sample."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0.0 else "-Infinity"
        return number
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
        "measured": value,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(passed),
    }


def create_link(stage, name: str, mass_kg: float, *, collision: bool = False):
    cube = UsdGeom.Cube.Define(stage, f"/World/Thread/{name}")
    cube.CreateSizeAttr(0.01)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(
        Gf.Vec3f(LINK_INERTIA_KG_M2, LINK_INERTIA_KG_M2, LINK_INERTIA_KG_M2)
    )
    PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr(True)
    return cube.GetPath()


def create_plate(stage, name: str, z_m: float):
    plate = UsdGeom.Cube.Define(stage, f"/World/{name}")
    plate.CreateSizeAttr(1.0)
    plate.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_m))
    plate.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.004))
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(plate.GetPrim())
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    UsdPhysics.MassAPI.Apply(plate.GetPrim()).CreateMassAttr(100.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(plate.GetPrim()).CreateDisableGravityAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(plate.GetPrim()).CreateThresholdAttr(0.0)
    return plate.GetPath()


def as_torch(value) -> torch.Tensor:
    return wp.to_torch(value).detach().cpu().clone()


def main() -> None:
    global DIAGNOSTIC_METADATA
    allowed_pairs = {(1.0e6, 4000.0), (1.0e5, 2500.0)}
    if (THREAD_STIFFNESS_N_PER_M, THREAD_DAMPING_NS_PER_M) not in allowed_pairs:
        raise ValueError(
            "thread stiffness/damping must be one frozen pair: "
            "KS=1e6,DS=4000 or KN=1e5,DN=2500"
        )
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    stage = sim.stage
    root = UsdGeom.Xform.Define(stage, "/World/Thread")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    articulation_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)

    frame = create_link(stage, "frame", 1.0)
    carrier = create_link(stage, "carrier", 0.01, collision=True)
    rotor = create_link(stage, "rotor", 0.01)

    half_carrier = 0.005
    half_plate = 0.002
    offset = half_carrier + half_plate
    lower_plate = create_plate(stage, "LowerPlate", 0.5 - offset)
    upper_plate = create_plate(stage, "UpperPlate", 0.5 + offset)

    mount = UsdPhysics.FixedJoint.Define(stage, "/World/Thread/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    axial = UsdPhysics.PrismaticJoint.Define(stage, "/World/Thread/axial")
    rotation = UsdPhysics.RevoluteJoint.Define(stage, "/World/Thread/rotation")
    for joint, parent, child in ((axial, frame, carrier), (rotation, carrier, rotor)):
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateAxisAttr("Z")
    axial.CreateLowerLimitAttr(-0.005)
    axial.CreateUpperLimitAttr(0.005)
    rotation.CreateLowerLimitAttr(-1440.0)
    rotation.CreateUpperLimitAttr(1440.0)

    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    if physics is None:
        raise RuntimeError("native PhysX simulation view is unavailable after reset")
    physics.set_subspace_roots("/")
    articulation = physics.create_articulation_view("/World/Thread")
    dof_names = list(articulation.get_metatype(0).dof_names)
    if articulation.count != 1 or articulation.max_dofs != 2:
        raise RuntimeError(
            f"unexpected articulation shape: count={articulation.count}, dofs={articulation.max_dofs}"
        )
    axial_index = dof_names.index("axial")
    rotation_index = dof_names.index("rotation")
    carrier_view = physics.create_rigid_body_view([str(carrier)])
    contact_view = physics.create_rigid_contact_view(
        [str(carrier)],
        filter_patterns=[[str(lower_plate), str(upper_plate)]],
        max_contact_data_count=32,
    )
    if (
        carrier_view.count != 1
        or contact_view.sensor_count != 1
        or contact_view.filter_count != 2
    ):
        raise RuntimeError(
            "unexpected native views: "
            f"carrier={carrier_view.count}, "
            f"contact=({contact_view.sensor_count},{contact_view.filter_count})"
        )

    all_indices = wp.array([0], dtype=wp.uint32, device=args.device)
    samples = DIAGNOSTIC_SAMPLES
    DIAGNOSTIC_METADATA = {
        "native_trial_id": trial_id,
        "source_sha256": source_sha256,
        "scope": "isolated PRE_PHYSICS_STEP conservative-law/contact discovery only",
        "dof_names": dof_names,
        "filter_paths": list(contact_view.filter_paths),
        "lead_m_per_revolution": LEAD_M_PER_REV,
        "p_m_per_radian": P_M_PER_RAD,
        "p_m_per_degree": P_M_PER_DEG,
        "thread_stiffness_n_per_m": THREAD_STIFFNESS_N_PER_M,
        "thread_damping_ns_per_m": THREAD_DAMPING_NS_PER_M,
        "applied_rotor_torque_nm": APPLIED_TORQUE_NM,
    }

    def persist_trace(status: str, error: str | None = None) -> None:
        payload = {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "scope": "isolated PRE_PHYSICS_STEP conservative-law/contact discovery only",
            "status": status,
            "error": error,
            "dof_names": dof_names,
            "filter_paths": list(contact_view.filter_paths),
            "lead_m_per_revolution": LEAD_M_PER_REV,
            "p_m_per_radian": P_M_PER_RAD,
            "p_m_per_degree": P_M_PER_DEG,
            "thread_stiffness_n_per_m": THREAD_STIFFNESS_N_PER_M,
            "thread_damping_ns_per_m": THREAD_DAMPING_NS_PER_M,
            "applied_rotor_torque_nm": APPLIED_TORQUE_NM,
            "pre_physics_callback_count": callback_state["count"],
            "pre_physics_callback_dt_values_s": sorted(callback_state["dt_values"]),
            "samples": samples,
        }
        write_json(trace_path, json_safe(payload))

    callback_state = {
        "commanded_external_torque_nm": 0.0,
        "count": 0,
        "dt_values": set(),
        "last_law_force_n": 0.0,
    }

    def apply_thread_law(dt: float) -> None:
        """Apply -grad(U)-D*grad(e)*e_dot plus the commanded test torque."""
        try:
            positions = as_torch(articulation.get_dof_positions())[0]
            velocities = as_torch(articulation.get_dof_velocities())[0]
            extension = float(
                positions[axial_index].item()
                - P_M_PER_RAD * positions[rotation_index].item()
            )
            extension_speed = float(
                velocities[axial_index].item()
                - P_M_PER_RAD * velocities[rotation_index].item()
            )
            law_force = (
                THREAD_STIFFNESS_N_PER_M * extension
                + THREAD_DAMPING_NS_PER_M * extension_speed
            )
            efforts = torch.zeros(
                (articulation.count, articulation.max_dofs),
                dtype=torch.float32,
                device=args.device,
            )
            efforts[0, axial_index] = -law_force
            efforts[0, rotation_index] = (
                callback_state["commanded_external_torque_nm"]
                + P_M_PER_RAD * law_force
            )
            articulation.set_dof_actuation_forces(
                wp.from_torch(efforts.contiguous(), dtype=wp.float32), all_indices
            )
            callback_state["last_law_force_n"] = law_force
            callback_state["count"] += 1
            callback_state["dt_values"].add(float(dt))
        except Exception as exc:
            PhysxManager.store_callback_exception(exc)

    callback_handle = PhysxManager.register_callback(
        apply_thread_law,
        IsaacEvents.PRE_PHYSICS_STEP,
        order=0,
        name="drg80_conservative_thread_law_probe",
        wrap_weak_ref=False,
    )

    def run_phase(name: str, steps: int, torque_nm: float) -> None:
        callback_state["commanded_external_torque_nm"] = torque_nm
        for phase_step in range(steps):
            sim.step(render=False)
            PhysxManager.raise_callback_exception_if_any()
            positions = as_torch(articulation.get_dof_positions())[0]
            velocities = as_torch(articulation.get_dof_velocities())[0]
            applied_efforts = as_torch(articulation.get_dof_actuation_forces())[0]
            matrix = as_torch(contact_view.get_contact_force_matrix(dt=DT)).reshape(1, 2, 3)
            impulse_matrix = as_torch(
                contact_view.get_contact_force_matrix(dt=1.0)
            ).reshape(1, 2, 3)
            net = as_torch(contact_view.get_net_contact_forces(dt=DT)).reshape(1, 3)
            force_contact_data = tuple(
                as_torch(value) for value in contact_view.get_contact_data(dt=DT)
            )
            impulse_contact_data = tuple(
                as_torch(value) for value in contact_view.get_contact_data(dt=1.0)
            )
            if len(force_contact_data) != 6 or len(impulse_contact_data) != 6:
                raise RuntimeError(
                    "unexpected contact-data tuple lengths "
                    f"{len(force_contact_data)}, {len(impulse_contact_data)}"
                )
            forces, points, normals, separations, counts, starts = force_contact_data
            (
                impulse_forces,
                _impulse_points,
                impulse_normals,
                _impulse_separations,
                impulse_counts,
                impulse_starts,
            ) = impulse_contact_data
            count_array = counts.to(torch.int64).reshape(1, 2)
            start_array = starts.to(torch.int64).reshape(1, 2)
            impulse_count_array = impulse_counts.to(torch.int64).reshape(1, 2)
            impulse_start_array = impulse_starts.to(torch.int64).reshape(1, 2)
            patch_force_sums = []
            patch_impulse_sums = []
            reconstructed_force_vectors = []
            reconstructed_impulse_vectors = []
            active_contact_entries = []
            active_impulse_scalars = []
            for filter_index in range(2):
                count = int(count_array[0, filter_index].item())
                start = int(start_array[0, filter_index].item()) if count else 0
                impulse_count = int(impulse_count_array[0, filter_index].item())
                impulse_start = (
                    int(impulse_start_array[0, filter_index].item())
                    if impulse_count
                    else 0
                )
                patch_force_sums.append(float(forces.flatten()[start : start + count].sum().item()))
                patch_impulse_sums.append(
                    float(
                        impulse_forces.flatten()[
                            impulse_start : impulse_start + impulse_count
                        ].sum().item()
                    )
                )
                force_scalars = forces.reshape(-1, 1)[start : start + count]
                force_normals = normals.reshape(-1, 3)[start : start + count]
                impulse_scalars = impulse_forces.reshape(-1, 1)[
                    impulse_start : impulse_start + impulse_count
                ]
                pair_impulse_normals = impulse_normals.reshape(-1, 3)[
                    impulse_start : impulse_start + impulse_count
                ]
                reconstructed_force_vectors.append(
                    (force_scalars * force_normals).sum(dim=0).tolist()
                )
                reconstructed_impulse_vectors.append(
                    (impulse_scalars * pair_impulse_normals).sum(dim=0).tolist()
                )
                active_impulse_scalars.extend(impulse_scalars.flatten().tolist())
                for contact_index in range(start, start + count):
                    active_contact_entries.append(
                        {
                            "filter_index": filter_index,
                            "buffer_index": contact_index,
                            "force_n": float(forces.flatten()[contact_index].item()),
                            "point_w_m": points.reshape(-1, 3)[contact_index].tolist(),
                            "normal_w": normals.reshape(-1, 3)[contact_index].tolist(),
                            "separation_m": float(separations.flatten()[contact_index].item()),
                        }
                    )
            extension_m = float(
                positions[axial_index].item()
                - P_M_PER_RAD * positions[rotation_index].item()
            )
            extension_rate_m_s = float(
                velocities[axial_index].item()
                - P_M_PER_RAD * velocities[rotation_index].item()
            )
            samples.append(
                {
                    "phase": name,
                    "phase_step": phase_step,
                    "dof_positions": positions.tolist(),
                    "dof_velocities": velocities.tolist(),
                    "dof_actuation_forces": applied_efforts.tolist(),
                    "commanded_external_torque_nm": torque_nm,
                    "applied_thread_law_force_n": callback_state["last_law_force_n"],
                    "carrier_transform_xyzw": as_torch(carrier_view.get_transforms())[0].tolist(),
                    "contact_force_matrix_n": matrix[0].tolist(),
                    "contact_impulse_matrix_ns": impulse_matrix[0].tolist(),
                    "net_contact_force_n": net[0].tolist(),
                    "patch_force_sums_n": patch_force_sums,
                    "patch_impulse_sums_ns": patch_impulse_sums,
                    "reconstructed_force_vectors_n": reconstructed_force_vectors,
                    "reconstructed_impulse_vectors_ns": reconstructed_impulse_vectors,
                    "contact_counts": count_array[0].tolist(),
                    "contact_starts": start_array[0].tolist(),
                    "impulse_contact_counts": impulse_count_array[0].tolist(),
                    "impulse_contact_starts": impulse_start_array[0].tolist(),
                    "active_impulse_scalars_ns": active_impulse_scalars,
                    "active_contact_entries": active_contact_entries,
                    "measured_tendon_extension_m": extension_m,
                    "measured_tendon_speed_m_s": extension_rate_m_s,
                }
            )
        # Phase-boundary persistence guarantees that a later SDK/runtime or
        # validation failure cannot discard already captured native samples.
        persist_trace(f"captured_{name}")

    run_phase("settle", SETTLE_STEPS, 0.0)
    run_phase("positive_torque", LOAD_STEPS, APPLIED_TORQUE_NM)
    run_phase("released_mid", RELEASE_STEPS, 0.0)
    run_phase("negative_torque", 2 * LOAD_STEPS, -APPLIED_TORQUE_NM)
    run_phase("released_final", RELEASE_STEPS, 0.0)
    callback_handle.deregister()

    positions = np.asarray([s["dof_positions"] for s in samples], dtype=np.float64)
    velocities = np.asarray([s["dof_velocities"] for s in samples], dtype=np.float64)
    applied = np.asarray(
        [s["commanded_external_torque_nm"] for s in samples], dtype=np.float64
    )
    native_applied_efforts = np.asarray(
        [s["dof_actuation_forces"] for s in samples], dtype=np.float64
    )
    applied_law_force = np.asarray(
        [s["applied_thread_law_force_n"] for s in samples], dtype=np.float64
    )
    matrices = np.asarray([s["contact_force_matrix_n"] for s in samples], dtype=np.float64)
    impulse_matrices = np.asarray(
        [s["contact_impulse_matrix_ns"] for s in samples], dtype=np.float64
    )
    net_forces = np.asarray([s["net_contact_force_n"] for s in samples], dtype=np.float64)
    patch_forces = np.asarray([s["patch_force_sums_n"] for s in samples], dtype=np.float64)
    patch_impulses = np.asarray([s["patch_impulse_sums_ns"] for s in samples], dtype=np.float64)
    reconstructed_forces = np.asarray(
        [s["reconstructed_force_vectors_n"] for s in samples], dtype=np.float64
    )
    reconstructed_impulses = np.asarray(
        [s["reconstructed_impulse_vectors_ns"] for s in samples], dtype=np.float64
    )
    force_counts = np.asarray([s["contact_counts"] for s in samples], dtype=np.int64)
    force_starts = np.asarray([s["contact_starts"] for s in samples], dtype=np.int64)
    impulse_counts = np.asarray(
        [s["impulse_contact_counts"] for s in samples], dtype=np.int64
    )
    impulse_starts = np.asarray(
        [s["impulse_contact_starts"] for s in samples], dtype=np.int64
    )
    active_impulse_scalars = np.asarray(
        [
            value
            for sample in samples
            for value in sample["active_impulse_scalars_ns"]
        ],
        dtype=np.float64,
    )
    extension = np.asarray([s["measured_tendon_extension_m"] for s in samples], dtype=np.float64)
    extension_speed = np.asarray([s["measured_tendon_speed_m_s"] for s in samples], dtype=np.float64)
    active_contact_values = []
    active_contact_entries = []
    for sample in samples:
        for entry in sample["active_contact_entries"]:
            active_contact_entries.append(entry)
            active_contact_values.extend(
                [
                    entry["force_n"],
                    *entry["point_w_m"],
                    *entry["normal_w"],
                    entry["separation_m"],
                ]
            )
    finite_values = np.concatenate(
        (
            positions.ravel(), velocities.ravel(), applied.ravel(),
            native_applied_efforts.ravel(), applied_law_force.ravel(), matrices.ravel(),
            impulse_matrices.ravel(), net_forces.ravel(), patch_forces.ravel(),
            patch_impulses.ravel(), reconstructed_forces.ravel(),
            reconstructed_impulses.ravel(), active_impulse_scalars.ravel(),
            extension, extension_speed,
            np.asarray(active_contact_values, dtype=np.float64),
        )
    )
    finite_fraction = float(np.isfinite(finite_values).mean())
    if finite_fraction != 1.0:
        persist_trace("failed_nonfinite_active_state")
        checks = [
            check("native_active_state_finite_fraction", finite_fraction, "==", 1.0, "fraction")
        ]
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
        raise RuntimeError("non-finite native state or active contact-buffer entry")

    # Validate the detailed native contact buffer entry-by-entry.  Each scalar
    # from get_contact_data is multiplied by that call's own normal, then the
    # vectors selected by that call's own counts/starts are summed separately
    # for every sample and filter.  The resulting tensors must reproduce the
    # corresponding get_contact_force_matrix result at both dt values.
    if not active_contact_entries or active_impulse_scalars.size == 0:
        persist_trace("failed_no_active_contact_entries")
        checks = [
            check(
                "native_active_contact_entry_count",
                len(active_contact_entries),
                ">=",
                1,
                "entries",
            ),
            check(
                "native_active_impulse_entry_count",
                active_impulse_scalars.size,
                ">=",
                1,
                "entries",
            ),
        ]
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
        raise RuntimeError("native contact view returned no active detailed entries")

    entry_forces = np.asarray(
        [entry["force_n"] for entry in active_contact_entries], dtype=np.float64
    )
    entry_points = np.asarray(
        [entry["point_w_m"] for entry in active_contact_entries], dtype=np.float64
    )
    entry_normals = np.asarray(
        [entry["normal_w"] for entry in active_contact_entries], dtype=np.float64
    )
    entry_separations = np.asarray(
        [entry["separation_m"] for entry in active_contact_entries], dtype=np.float64
    )
    entry_filter_indices = np.asarray(
        [entry["filter_index"] for entry in active_contact_entries], dtype=np.int64
    )
    expected_interface_z = np.where(
        entry_filter_indices == 0,
        0.5 - half_carrier,
        0.5 + half_carrier,
    )
    entry_normal_unit_error = np.abs(np.linalg.norm(entry_normals, axis=1) - 1.0)
    entry_normal_transverse = np.linalg.norm(entry_normals[:, :2], axis=1)
    entry_point_lateral_excess = np.maximum(
        np.maximum(np.abs(entry_points[:, 0]), np.abs(entry_points[:, 1])) - half_carrier,
        0.0,
    )
    entry_point_interface_error = np.abs(entry_points[:, 2] - expected_interface_z)
    detailed_force_matrix_error = np.abs(reconstructed_forces - matrices)
    detailed_impulse_matrix_error = np.abs(reconstructed_impulses - impulse_matrices)
    detailed_dt_scaling_error = np.abs(reconstructed_forces * DT - reconstructed_impulses)
    contact_layout_count_error = np.abs(force_counts - impulse_counts)
    # A start index has no meaning for an empty pair.  Compare it only where
    # either dt call reports an active pair.
    active_pair_mask = (force_counts > 0) | (impulse_counts > 0)
    contact_layout_start_error = np.where(
        active_pair_mask, np.abs(force_starts - impulse_starts), 0
    )

    positive_start = SETTLE_STEPS
    positive_slice = slice(positive_start + 3 * LOAD_STEPS // 4, positive_start + LOAD_STEPS)
    negative_start = SETTLE_STEPS + LOAD_STEPS + RELEASE_STEPS
    negative_slice = slice(negative_start + 3 * LOAD_STEPS // 2, negative_start + 2 * LOAD_STEPS)
    filtered_net = matrices.sum(axis=1)
    positive_z = float(np.median(filtered_net[positive_slice, 2]))
    negative_z = float(np.median(filtered_net[negative_slice, 2]))
    positive_plate_abs_z = np.median(np.abs(matrices[positive_slice, :, 2]), axis=0)
    negative_plate_abs_z = np.median(np.abs(matrices[negative_slice, :, 2]), axis=0)
    positive_active_plate = int(np.argmax(positive_plate_abs_z))
    negative_active_plate = int(np.argmax(negative_plate_abs_z))
    positive_inactive_abs_z = float(positive_plate_abs_z[1 - positive_active_plate])
    negative_inactive_abs_z = float(negative_plate_abs_z[1 - negative_active_plate])
    positive_inactive_ratio = positive_inactive_abs_z / max(abs(positive_z), 1.0e-12)
    negative_inactive_ratio = negative_inactive_abs_z / max(abs(negative_z), 1.0e-12)
    positive_transverse = float(
        np.median(np.linalg.norm(matrices[positive_slice, :, :2], axis=2).sum(axis=1))
    )
    negative_transverse = float(
        np.median(np.linalg.norm(matrices[negative_slice, :, :2], axis=2).sum(axis=1))
    )
    positive_transverse_ratio = positive_transverse / max(abs(positive_z), 1.0e-12)
    negative_transverse_ratio = negative_transverse / max(abs(negative_z), 1.0e-12)
    unloaded_slices = (
        slice(SETTLE_STEPS // 2, SETTLE_STEPS),
        slice(SETTLE_STEPS + LOAD_STEPS + RELEASE_STEPS // 2,
              SETTLE_STEPS + LOAD_STEPS + RELEASE_STEPS),
        slice(len(samples) - RELEASE_STEPS // 2, len(samples)),
    )
    unloaded_axial_baseline = max(
        float(np.median(np.abs(matrices[window, :, 2]).sum(axis=1)))
        for window in unloaded_slices
    )
    unloaded_baseline_ratio = unloaded_axial_baseline / max(
        min(abs(positive_z), abs(negative_z)), 1.0e-12
    )
    filtered_net_error = float(np.max(np.abs(filtered_net - net_forces)))
    force_impulse_error = float(np.max(np.abs(matrices * DT - impulse_matrices)))
    positive_effort = float(np.median(applied[positive_slice]))
    negative_effort = float(np.median(applied[negative_slice]))
    positive_effort_error = float(
        np.max(np.abs(applied[positive_slice] - APPLIED_TORQUE_NM))
    )
    negative_effort_error = float(
        np.max(np.abs(applied[negative_slice] + APPLIED_TORQUE_NM))
    )
    positive_extension = float(np.median(extension[positive_slice]))
    negative_extension = float(np.median(extension[negative_slice]))
    positive_speed = float(np.median(extension_speed[positive_slice]))
    negative_speed = float(np.median(extension_speed[negative_slice]))
    # Use the force actually submitted by PRE_PHYSICS_STEP for each integrated
    # sample.  Recomputing from the post-step state would introduce a one-step
    # phase error and would no longer validate the applied generalized force.
    law_force_samples = applied_law_force
    native_axial_effort_error = float(
        np.max(np.abs(native_applied_efforts[:, axial_index] + law_force_samples))
    )
    native_rotor_effort_error = float(
        np.max(
            np.abs(
                native_applied_efforts[:, rotation_index]
                - (applied + P_M_PER_RAD * law_force_samples)
            )
        )
    )
    positive_law_samples = law_force_samples[positive_slice]
    negative_law_samples = law_force_samples[negative_slice]
    positive_reaction_samples = filtered_net[positive_slice, 2]
    negative_reaction_samples = filtered_net[negative_slice, 2]
    positive_law_magnitude_min = float(np.min(np.abs(positive_law_samples)))
    negative_law_magnitude_min = float(np.min(np.abs(negative_law_samples)))
    denominator_checks = [
        check("positive_frozen_law_force_magnitude_min", positive_law_magnitude_min, ">", 1.0e-12, "N"),
        check("negative_frozen_law_force_magnitude_min", negative_law_magnitude_min, ">", 1.0e-12, "N"),
        check("positive_applied_rotor_effort_magnitude", abs(positive_effort), ">", 1.0e-12, "N*m"),
        check("negative_applied_rotor_effort_magnitude", abs(negative_effort), ">", 1.0e-12, "N*m"),
    ]
    if not all(item["passed"] for item in denominator_checks):
        persist_trace("failed_zero_ratio_denominator")
        write_json(
            output_path,
            {
                "native_trial_id": trial_id,
                "source_sha256": source_sha256,
                "qualification": False,
                "checks": denominator_checks,
                "overall_pass": False,
                "artifacts": [str(trace_path.relative_to(output_path.parent.parent))],
            },
        )
        raise RuntimeError("zero measured denominator prevents physical reaction ratio")
    positive_contact_law_ratios = positive_reaction_samples / positive_law_samples
    negative_contact_law_ratios = negative_reaction_samples / negative_law_samples
    positive_contact_law_ratio_min = float(np.min(positive_contact_law_ratios))
    positive_contact_law_ratio_max = float(np.max(positive_contact_law_ratios))
    negative_contact_law_ratio_min = float(np.min(negative_contact_law_ratios))
    negative_contact_law_ratio_max = float(np.max(negative_contact_law_ratios))
    positive_aligned_law_relative_error_max = float(
        np.max(np.abs(positive_reaction_samples - positive_law_samples)
               / np.abs(positive_law_samples))
    )
    negative_aligned_law_relative_error_max = float(
        np.max(np.abs(negative_reaction_samples - negative_law_samples)
               / np.abs(negative_law_samples))
    )
    positive_aligned_sign_product_min = float(
        np.min(positive_reaction_samples * positive_law_samples)
    )
    negative_aligned_sign_product_min = float(
        np.min(negative_reaction_samples * negative_law_samples)
    )
    positive_torque_ratio = -P_M_PER_RAD * positive_z / positive_effort
    negative_torque_ratio = -P_M_PER_RAD * negative_z / negative_effort
    axial_motion_max = float(np.max(np.abs(positions[:, axial_index] - positions[0, axial_index])))
    total_steps = SETTLE_STEPS + 3 * LOAD_STEPS + 2 * RELEASE_STEPS

    checks = denominator_checks + [
        check("native_articulation_count", articulation.count, "==", 1, "articulations"),
        check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
        check("pre_physics_callback_count", callback_state["count"], "==", total_steps, "callbacks"),
        check("pre_physics_callback_dt_value_count", len(callback_state["dt_values"]), "==", 1, "distinct values"),
        # PhysX delivers callback dt through a float32 ABI.  Compare the measured
        # callback value to the configured double-precision timestep by an
        # explicit absolute representation-error bound; exact binary equality
        # incorrectly rejects 1/240 returned as float32 (2.17e-10 s error).
        check(
            "pre_physics_callback_dt_absolute_error",
            abs(next(iter(callback_state["dt_values"])) - DT),
            "<=",
            1.0e-9,
            "s",
        ),
        check("native_applied_axial_law_effort_error_max", native_axial_effort_error, "<=", 5.0e-4, "N"),
        check("native_applied_rotor_law_effort_error_max", native_rotor_effort_error, "<=", 5.0e-6, "N*m"),
        check("filtered_contact_sensor_count", contact_view.sensor_count, "==", 1, "bodies"),
        check("filtered_contact_plate_count", contact_view.filter_count, "==", 2, "bodies"),
        check("actual_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
        check("fresh_native_sample_count", len(samples), "==", total_steps, "physics steps"),
        check("native_active_state_finite_fraction", finite_fraction, "==", 1.0, "fraction"),
        check("native_active_contact_entry_count", len(active_contact_entries), ">=", 1, "entries"),
        check("native_active_impulse_entry_count", active_impulse_scalars.size, ">=", 1, "entries"),
        check("active_contact_scalar_force_min", float(np.min(entry_forces)), ">=", 0.0, "N"),
        check("active_contact_scalar_impulse_min", float(np.min(active_impulse_scalars)), ">=", 0.0, "N*s"),
        check("active_contact_normal_unit_error_max", float(np.max(entry_normal_unit_error)), "<=", 1.0e-3, "dimensionless"),
        check("active_contact_normal_transverse_max", float(np.max(entry_normal_transverse)), "<=", 0.02, "dimensionless"),
        check("active_contact_point_lateral_excess_max", float(np.max(entry_point_lateral_excess)), "<=", 1.0e-3, "m"),
        check("active_contact_point_interface_error_max", float(np.max(entry_point_interface_error)), "<=", 1.0e-3, "m"),
        check("active_contact_separation_min", float(np.min(entry_separations)), ">=", -0.002, "m"),
        check("active_contact_separation_max", float(np.max(entry_separations)), "<=", 0.0005, "m"),
        check("contact_dt_layout_count_error_max", float(np.max(contact_layout_count_error)), "==", 0.0, "entries"),
        check("contact_dt_layout_active_start_error_max", float(np.max(contact_layout_start_error)), "==", 0.0, "buffer indices"),
        check("per_sample_filter_detailed_force_matrix_error_max", float(np.max(detailed_force_matrix_error)), "<=", 1.0e-4, "N"),
        check("per_sample_filter_detailed_impulse_matrix_error_max", float(np.max(detailed_impulse_matrix_error)), "<=", 1.0e-6, "N*s"),
        check("per_sample_filter_detailed_dt_scaling_error_max", float(np.max(detailed_dt_scaling_error)), "<=", 1.0e-6, "N*s"),
        check("axial_restraint_motion_max", axial_motion_max, "<=", 0.0005, "m"),
        check("positive_load_signed_axial_reaction_magnitude", abs(positive_z), ">=", 1.0, "N"),
        check("negative_load_signed_axial_reaction_magnitude", abs(negative_z), ">=", 1.0, "N"),
        check("positive_applied_rotor_effort_error_max", positive_effort_error, "<=", 5.0e-5, "N*m"),
        check("negative_applied_rotor_effort_error_max", negative_effort_error, "<=", 5.0e-5, "N*m"),
        check("filtered_plate_force_sign_reversal", positive_z * negative_z, "<", 0.0, "N^2"),
        check("positive_aligned_reaction_frozen_law_sign_min", positive_aligned_sign_product_min, ">", 0.0, "N^2"),
        check("negative_aligned_reaction_frozen_law_sign_min", negative_aligned_sign_product_min, ">", 0.0, "N^2"),
        check("positive_inactive_plate_axial_ratio_max", positive_inactive_ratio, "<=", 0.1, "ratio"),
        check("negative_inactive_plate_axial_ratio_max", negative_inactive_ratio, "<=", 0.1, "ratio"),
        check("positive_transverse_to_axial_ratio_max", positive_transverse_ratio, "<=", 0.1, "ratio"),
        check("negative_transverse_to_axial_ratio_max", negative_transverse_ratio, "<=", 0.1, "ratio"),
        check("unloaded_axial_baseline_to_load_ratio_max", unloaded_baseline_ratio, "<=", 0.1, "ratio"),
        check("measured_tendon_extension_sign_reversal", positive_extension * negative_extension, "<", 0.0, "m^2"),
        check("positive_tendon_extension_magnitude", abs(positive_extension), ">=", 1.0e-6, "m"),
        check("negative_tendon_extension_magnitude", abs(negative_extension), ">=", 1.0e-6, "m"),
        check("positive_sample_aligned_contact_to_law_ratio_min", positive_contact_law_ratio_min, ">=", 0.5, "ratio"),
        check("positive_sample_aligned_contact_to_law_ratio_max", positive_contact_law_ratio_max, "<=", 1.5, "ratio"),
        check("negative_sample_aligned_contact_to_law_ratio_min", negative_contact_law_ratio_min, ">=", 0.5, "ratio"),
        check("negative_sample_aligned_contact_to_law_ratio_max", negative_contact_law_ratio_max, "<=", 1.5, "ratio"),
        check("positive_sample_aligned_law_relative_error_max", positive_aligned_law_relative_error_max, "<=", 0.5, "fraction"),
        check("negative_sample_aligned_law_relative_error_max", negative_aligned_law_relative_error_max, "<=", 0.5, "fraction"),
        check("positive_contact_to_applied_torque_ratio_min", positive_torque_ratio, ">=", 0.5, "ratio"),
        check("positive_contact_to_applied_torque_ratio_max", positive_torque_ratio, "<=", 1.5, "ratio"),
        check("negative_contact_to_applied_torque_ratio_min", negative_torque_ratio, ">=", 0.5, "ratio"),
        check("negative_contact_to_applied_torque_ratio_max", negative_torque_ratio, "<=", 1.5, "ratio"),
        check("filtered_to_net_force_error_max", filtered_net_error, "<=", 1.0e-4, "N"),
        check("force_matrix_impulse_dt_scaling_error", force_impulse_error, "<=", 1.0e-5, "N*s"),
    ]

    persist_trace("validation_complete")
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
            "native conservative thread-law/contact discovery failed: "
            + ", ".join(item["name"] for item in checks if not item["passed"])
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        diagnostic = {
            **DIAGNOSTIC_METADATA,
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "scope": "isolated PRE_PHYSICS_STEP conservative-law/contact discovery only",
            "status": "runtime_exception",
            "error": detail,
            "samples": DIAGNOSTIC_SAMPLES,
        }
        write_json(trace_path, json_safe(diagnostic))
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
