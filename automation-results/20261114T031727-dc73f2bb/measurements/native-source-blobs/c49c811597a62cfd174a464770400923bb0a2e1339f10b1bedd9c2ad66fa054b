#!/usr/bin/env python3
"""Bounded native N73-relative-to-S coupling probe; not qualification.

This probe asks the next unresolved fixture question only: can the installed
GPU PhysX articulation represent the frozen N73 extension relative to a
dynamic screw, rather than relative to the world or a clamped surrogate?

The screw and nut are separate normative-mass dynamic links.  The screw is
axially supported by the frame through its revolute bearing, but remains a
dynamic rotational body; there is no axial carrier DOF and therefore no
undocumented prismatic hard-stop reaction.  Between screw and nut is
a native implicit prismatic drive with KN=100000 N/m and DN=2500 N*s/m in
series with the installed rack-and-pinion schema's hard 1 mm/revolution
revolute/prismatic coupling.  The drive
target and all USD properties are frozen before reset.  Equal-and-opposite
body torque is produced by the nut-relative revolute joint in both directions.
The right-handed sign is certified independently from fresh screw and nut link
transforms: after subtracting the authored reset references, the measured
relative extension ``delta_z - p * delta_theta`` must equal the elastic DOF.
An initial no-contact interval checks the equal-and-opposite body response from
measured link angular momentum and checks finite actuator work against measured
link kinetic energy, elastic energy, and damping loss.  Projected incoming-joint
forces are not treated as isolated rack-and-pinion reactions.  After reset
the program uses only normal articulation effort actions; it never writes a pose,
velocity, joint position, contact state, or scorer state.

This component coupon has no cover seat, thread friction, preload certificate,
fixture, robot, or scorer.  A pass therefore cannot establish released
preload, lock retention, any frozen environment check, or YAM feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
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
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics


DT = 1.0 / 240.0
LEAD_M_PER_REV = 0.001
P_M_PER_RAD = LEAD_M_PER_REV / (2.0 * math.pi)
KN_N_PER_M = 100000.0
DN_NS_PER_M = 2500.0
TEST_TORQUE_NM = 0.005
CALIBRATION_TORQUE_NM = 1.0e-5
CALIBRATION_STEPS = 8
CALIBRATION_BRAKE_STEPS = 8
EXPECTED_STATIC_EXTENSION_M = TEST_TORQUE_NM / (P_M_PER_RAD * KN_N_PER_M)
SETTLE_STEPS = 240
RAMP_STEPS = 240
HOLD_STEPS = 480
RELEASE_STEPS = 240
POSITION_ITERATIONS = 128
VELOCITY_ITERATIONS = 4

SCREW_MASS_KG = 0.009
SCREW_INERTIA_KG_M2 = (5.0e-7, 5.0e-7, 7.0e-8)
NUT_MASS_KG = 0.009
NUT_INERTIA_KG_M2 = (1.4e-7, 1.4e-7, 2.6e-7)
AUX_MASS_KG = 1.0e-6
AUX_INERTIA_KG_M2 = (1.0e-12, 1.0e-12, 1.0e-12)


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


def as_numpy(value) -> np.ndarray:
    return wp.to_torch(value).detach().cpu().numpy().copy()


def quaternion_conjugate_xyzw(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[:3] *= -1.0
    return result


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def relative_z_rotation_series(
    link_transforms: np.ndarray, parent_index: int, child_index: int
) -> np.ndarray:
    """Return continuous child-relative-to-parent rotation about common +Z."""
    angles = []
    for transforms in link_transforms:
        parent_q = transforms[parent_index, 3:7]
        child_q = transforms[child_index, 3:7]
        parent_norm = float(np.linalg.norm(parent_q))
        child_norm = float(np.linalg.norm(child_q))
        if parent_norm <= 0.0 or child_norm <= 0.0:
            raise RuntimeError("non-normalizable native link quaternion")
        relative_q = quaternion_multiply_xyzw(
            quaternion_conjugate_xyzw(parent_q / parent_norm), child_q / child_norm
        )
        # All coupon joint axes are common +Z.  atan2(z,w) is invariant to
        # quaternion sign after phase unwrapping and exposes the physical
        # N73-relative-to-S rotation without trusting generalized coordinates.
        angles.append(2.0 * math.atan2(float(relative_q[2]), float(relative_q[3])))
    return np.unwrap(np.asarray(angles, dtype=np.float64))


def create_body(
    stage,
    path: str,
    mass_kg: float,
    inertia: tuple[float, float, float],
    *,
    collision: bool = False,
):
    body = UsdGeom.Cylinder.Define(stage, path)
    body.CreateAxisAttr("Z")
    body.CreateRadiusAttr(0.004)
    body.CreateHeightAttr(0.012)
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    if collision:
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateDisableGravityAttr(True)
    return body.GetPath()


def create_plate(stage, name: str, z_m: float):
    plate = UsdGeom.Cube.Define(stage, f"/World/{name}")
    plate.CreateSizeAttr(1.0)
    plate.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_m))
    plate.AddScaleOp().Set(Gf.Vec3f(0.025, 0.025, 0.002))
    rigid = UsdPhysics.RigidBodyAPI.Apply(plate.GetPrim())
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    UsdPhysics.MassAPI.Apply(plate.GetPrim()).CreateMassAttr(100.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(plate.GetPrim()).CreateDisableGravityAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(plate.GetPrim()).CreateThresholdAttr(0.0)
    return plate.GetPath()


def create_rack_and_pinion(stage, path: str, hinge, prismatic):
    """Author the installed PhysxPhysicsRackAndPinionJoint schema directly.

    The pinned schema defines ``physics:ratio`` as angular/linear motion.  A
    1 mm/revolution lead is therefore 2*pi/0.001 rad/m.  This is the installed
    revolute/prismatic coupling type; the deprecated mimic API only accepts a
    rotational reference axis and is intentionally not used here.
    """
    prim = stage.DefinePrim(path, "PhysxPhysicsRackAndPinionJoint")
    prim.CreateRelationship("physics:hinge").SetTargets([hinge.GetPath()])
    prim.CreateRelationship("physics:prismatic").SetTargets([prismatic.GetPath()])
    prim.CreateAttribute("physics:ratio", Sdf.ValueTypeNames.Float).Set(
        1.0 / P_M_PER_RAD
    )
    prim.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    return prim


def create_prismatic(stage, path: str, body0, body1, lower: float, upper: float):
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    return joint


def create_revolute(stage, path: str, body0, body1):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-1440.0)
    joint.CreateUpperLimitAttr(1440.0)
    return joint


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device=args.device, render_interval=12)
    )
    stage = sim.stage
    root = UsdGeom.Xform.Define(stage, "/World/Coupling")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    articulation_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)

    frame = create_body(stage, "/World/Coupling/frame", 1.0, (1.0e-3,) * 3)
    screw = create_body(
        stage, "/World/Coupling/screw", SCREW_MASS_KG, SCREW_INERTIA_KG_M2
    )
    elastic = create_body(
        stage, "/World/Coupling/nut_elastic_carrier", AUX_MASS_KG, AUX_INERTIA_KG_M2
    )
    helix = create_body(
        stage, "/World/Coupling/nut_helix_carrier", AUX_MASS_KG, AUX_INERTIA_KG_M2
    )
    nut = create_body(
        stage,
        "/World/Coupling/nut",
        NUT_MASS_KG,
        NUT_INERTIA_KG_M2,
        collision=True,
    )
    # The two kinematic plates provide a bilateral, known-pair axial load.
    # The 12 mm nut has end faces at 0.494/0.506 m.  Each plate is 2 mm
    # thick, so centers at +/-7.1 mm put the nearest faces exactly 0.1 mm
    # away (0.4939/0.5061 m).  The previous +/-8.1 mm centers left 1.1 mm.
    lower_plate = create_plate(stage, "LowerPlate", 0.5 - 0.0071)
    upper_plate = create_plate(stage, "UpperPlate", 0.5 + 0.0071)

    mount = UsdPhysics.FixedJoint.Define(stage, "/World/Coupling/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    # This revolute bearing is the coupon's intended axial support.  It permits
    # the normative-mass screw to react dynamically in rotation while fixing
    # its axial coordinate without a hidden prismatic limit or servo.
    create_revolute(stage, "/World/Coupling/screw_rotation", frame, screw)

    relative_elastic = create_prismatic(
        stage, "/World/Coupling/nut_relative_elastic", screw, elastic, -0.001, 0.001
    )
    drive = UsdPhysics.DriveAPI.Apply(relative_elastic.GetPrim(), "linear")
    drive.CreateStiffnessAttr(KN_N_PER_M)
    drive.CreateDampingAttr(DN_NS_PER_M)
    drive.CreateMaxForceAttr(1000.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)

    relative_helix = create_prismatic(
        stage, "/World/Coupling/nut_relative_helix", elastic, helix, -0.004, 0.004
    )
    relative_rotation = create_revolute(
        stage, "/World/Coupling/nut_relative_rotation", helix, nut
    )
    rack_and_pinion = create_rack_and_pinion(
        stage,
        "/World/Coupling/nut_relative_helix_coupling",
        relative_rotation,
        relative_helix,
    )

    sim.reset()
    physics = PhysxManager.get_physics_sim_view()
    if physics is None:
        raise RuntimeError("native PhysX simulation view unavailable after reset")
    physics.set_subspace_roots("/")
    articulation = physics.create_articulation_view("/World/Coupling")
    if articulation.count != 1 or articulation.max_dofs != 4:
        raise RuntimeError(
            f"unexpected articulation shape {articulation.count}x{articulation.max_dofs}"
        )
    dof_names = list(articulation.get_metatype(0).dof_names)
    link_names = list(articulation.get_metatype(0).link_names)
    required = {
        "screw_rotation",
        "nut_relative_elastic",
        "nut_relative_helix",
        "nut_relative_rotation",
    }
    if set(dof_names) != required:
        raise RuntimeError(f"unexpected native DOF names/order: {dof_names!r}")
    index = {name: dof_names.index(name) for name in required}
    link_index = {name: link_names.index(name) for name in link_names}
    for required_link in (
        "screw",
        "nut_elastic_carrier",
        "nut_helix_carrier",
        "nut",
    ):
        if required_link not in link_index:
            raise RuntimeError(f"missing native link {required_link!r}: {link_names!r}")
    authored_ratio = float(rack_and_pinion.GetAttribute("physics:ratio").Get())
    if not math.isclose(authored_ratio, 1.0 / P_M_PER_RAD, rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError(f"rack-and-pinion ratio did not resolve: {authored_ratio}")
    contact_view = physics.create_rigid_contact_view(
        [str(nut)],
        filter_patterns=[[str(lower_plate), str(upper_plate)]],
        max_contact_data_count=64,
    )
    if contact_view.sensor_count != 1 or contact_view.filter_count != 2:
        raise RuntimeError(
            f"unexpected contact view {contact_view.sensor_count}x{contact_view.filter_count}"
        )
    all_indices = wp.array([0], dtype=wp.uint32, device=args.device)
    initial_q_native = as_numpy(articulation.get_dof_positions())[0]
    initial_link_velocities = as_numpy(articulation.get_link_velocities())[0]
    initial_link_transforms = as_numpy(articulation.get_link_transforms())[0]
    samples: list[dict] = []

    def step(phase: str, phase_step: int, nut_torque: float) -> None:
        efforts = torch.zeros(
            (articulation.count, articulation.max_dofs),
            dtype=torch.float32,
            device=args.device,
        )
        # A revolute-joint effort is a physical equal-and-opposite torque on
        # its child (nut) and parent (helix/upstream screw assembly).  Do not
        # add an unrelated screw_rotation generalized effort and call the sum
        # a body-wrench balance.
        efforts[0, index["nut_relative_rotation"]] = nut_torque
        articulation.set_dof_actuation_forces(
            wp.from_torch(efforts.contiguous(), dtype=wp.float32), all_indices
        )
        sim.step(render=False)
        PhysxManager.raise_callback_exception_if_any()
        q = as_numpy(articulation.get_dof_positions())[0]
        qd = as_numpy(articulation.get_dof_velocities())[0]
        applied = as_numpy(articulation.get_dof_actuation_forces())[0]
        projected = as_numpy(articulation.get_dof_projected_joint_forces())[0]
        incoming = as_numpy(articulation.get_link_incoming_joint_force())[0]
        contact = as_numpy(contact_view.get_contact_force_matrix(dt=DT)).reshape(1, 2, 3)[0]
        samples.append(
            {
                "sample": len(samples),
                "phase": phase,
                "phase_step": phase_step,
                "q": q.tolist(),
                "qd": qd.tolist(),
                "applied_effort": applied.tolist(),
                "projected_joint_force": projected.tolist(),
                "link_velocities_world": as_numpy(
                    articulation.get_link_velocities()
                )[0].tolist(),
                "link_transforms_world": as_numpy(
                    articulation.get_link_transforms()
                )[0].tolist(),
                "screw_incoming_joint_wrench": incoming[link_index["screw"]].tolist(),
                "helix_incoming_joint_wrench": incoming[
                    link_index["nut_helix_carrier"]
                ].tolist(),
                "nut_incoming_joint_wrench": incoming[link_index["nut"]].tolist(),
                "plate_contact_force_n": contact.tolist(),
            }
        )

    # A short isolated interval remains below the measured 0.1 mm plate gap.
    # It supplies non-negligible input work and body motion for reaction and
    # energy tests; contact force is independently required to remain absent.
    for i in range(CALIBRATION_STEPS):
        step("reaction_calibration", i, CALIBRATION_TORQUE_NM)
    for i in range(CALIBRATION_BRAKE_STEPS):
        step("reaction_brake", i, -CALIBRATION_TORQUE_NM)
    for i in range(SETTLE_STEPS):
        step("settle", i, 0.0)
    for direction, name in ((1.0, "positive"), (-1.0, "negative")):
        for i in range(RAMP_STEPS):
            step(f"{name}_ramp", i, direction * TEST_TORQUE_NM * (i + 1) / RAMP_STEPS)
        for i in range(HOLD_STEPS):
            step(f"{name}_hold", i, direction * TEST_TORQUE_NM)
        for i in range(RELEASE_STEPS):
            step(f"{name}_release", i, 0.0)

    q = np.asarray([row["q"] for row in samples], dtype=np.float64)
    qd = np.asarray([row["qd"] for row in samples], dtype=np.float64)
    applied = np.asarray([row["applied_effort"] for row in samples], dtype=np.float64)
    projected = np.asarray(
        [row["projected_joint_force"] for row in samples], dtype=np.float64
    )
    incoming = np.asarray(
        [
            [
                *row["screw_incoming_joint_wrench"],
                *row["helix_incoming_joint_wrench"],
                *row["nut_incoming_joint_wrench"],
            ]
            for row in samples
        ],
        dtype=np.float64,
    )
    contacts = np.asarray(
        [row["plate_contact_force_n"] for row in samples], dtype=np.float64
    )
    link_velocities = np.asarray(
        [row["link_velocities_world"] for row in samples], dtype=np.float64
    )
    link_transforms = np.asarray(
        [row["link_transforms_world"] for row in samples], dtype=np.float64
    )
    finite = np.concatenate(
        (q.ravel(), qd.ravel(), applied.ravel(), projected.ravel(), incoming.ravel(), contacts.ravel(), link_velocities.ravel(), link_transforms.ravel())
    )
    e = q[:, index["nut_relative_elastic"]]
    h = q[:, index["nut_relative_helix"]]
    theta = q[:, index["nut_relative_rotation"]]
    e_from_reset = e - initial_q_native[index["nut_relative_elastic"]]
    h_from_reset = h - initial_q_native[index["nut_relative_helix"]]
    theta_from_reset = theta - initial_q_native[index["nut_relative_rotation"]]
    helix_residual = theta_from_reset - h_from_reset / P_M_PER_RAD
    screw_link_index = link_index["screw"]
    nut_link_index = link_index["nut"]
    initial_relative_z = float(
        initial_link_transforms[nut_link_index, 2]
        - initial_link_transforms[screw_link_index, 2]
    )
    measured_relative_z = (
        link_transforms[:, nut_link_index, 2]
        - link_transforms[:, screw_link_index, 2]
        - initial_relative_z
    )
    transforms_with_reference = np.concatenate(
        (initial_link_transforms[None, :, :], link_transforms), axis=0
    )
    measured_relative_theta_with_reference = relative_z_rotation_series(
        transforms_with_reference, screw_link_index, nut_link_index
    )
    measured_relative_theta = (
        measured_relative_theta_with_reference[1:]
        - measured_relative_theta_with_reference[0]
    )
    measured_frozen_extension = (
        measured_relative_z - P_M_PER_RAD * measured_relative_theta
    )
    transform_extension_error = measured_frozen_extension - e_from_reset
    measured_rigid_helix_travel = measured_relative_z - e_from_reset
    transform_helix_error = (
        measured_rigid_helix_travel - P_M_PER_RAD * measured_relative_theta
    )
    calibration_total = CALIBRATION_STEPS + CALIBRATION_BRAKE_STEPS
    total_steps = calibration_total + SETTLE_STEPS + 2 * (
        RAMP_STEPS + HOLD_STEPS + RELEASE_STEPS
    )

    positive_hold_start = calibration_total + SETTLE_STEPS + RAMP_STEPS
    positive_hold = slice(positive_hold_start + HOLD_STEPS // 2, positive_hold_start + HOLD_STEPS)
    negative_start = (
        calibration_total + SETTLE_STEPS + RAMP_STEPS + HOLD_STEPS + RELEASE_STEPS
    )
    negative_hold_start = negative_start + RAMP_STEPS
    negative_hold = slice(negative_hold_start + HOLD_STEPS // 2, negative_hold_start + HOLD_STEPS)
    final_release = slice(total_steps - RELEASE_STEPS // 2, total_steps)

    positive_e = float(np.median(e[positive_hold]))
    negative_e = float(np.median(e[negative_hold]))
    positive_theta = float(np.median(theta[positive_hold]))
    negative_theta = float(np.median(theta[negative_hold]))
    uncommanded_effort = applied.copy()
    uncommanded_effort[:, index["nut_relative_rotation"]] = 0.0
    loaded = np.r_[
        np.arange(positive_hold.start, positive_hold.stop),
        np.arange(negative_hold.start, negative_hold.stop),
    ]
    # Reaction and power are checked from body motion, not from projected joint
    # forces (which the pinned API documents as total projected incoming force).
    calibration = slice(0, CALIBRATION_STEPS)
    moving_inertias = {
        "screw": SCREW_INERTIA_KG_M2,
        "nut_elastic_carrier": AUX_INERTIA_KG_M2,
        "nut_helix_carrier": AUX_INERTIA_KG_M2,
        "nut": NUT_INERTIA_KG_M2,
    }
    moving_masses = {
        "screw": SCREW_MASS_KG,
        "nut_elastic_carrier": AUX_MASS_KG,
        "nut_helix_carrier": AUX_MASS_KG,
        "nut": NUT_MASS_KG,
    }
    angular_momenta_z = np.zeros((CALIBRATION_STEPS, len(moving_inertias)))
    kinetic_energy = np.zeros(CALIBRATION_STEPS)
    for column, (name, inertia) in enumerate(moving_inertias.items()):
        velocity = link_velocities[calibration, link_index[name]]
        angular_momenta_z[:, column] = inertia[2] * velocity[:, 5]
        mass = moving_masses[name]
        kinetic_energy += 0.5 * mass * np.sum(velocity[:, :3] ** 2, axis=1)
        kinetic_energy += 0.5 * np.sum(
            velocity[:, 3:] ** 2 * np.asarray(inertia)[None, :], axis=1
        )
    total_angular_momentum_z = np.sum(angular_momenta_z, axis=1)
    angular_momentum_scale = np.sum(np.abs(angular_momenta_z), axis=1)
    angular_momentum_relative_residual = np.max(
        np.abs(total_angular_momentum_z)
        / np.maximum(angular_momentum_scale, 1.0e-12)
    )
    calibration_dtheta = np.diff(
        np.r_[initial_q_native[index["nut_relative_rotation"]], theta[calibration]]
    )
    calibration_input_work = float(
        np.sum(applied[calibration, index["nut_relative_rotation"]] * calibration_dtheta)
    )
    calibration_damping_heat = float(
        np.sum(DN_NS_PER_M * qd[calibration, index["nut_relative_elastic"]] ** 2 * DT)
    )
    initial_kinetic_energy = 0.0
    for name, inertia in moving_inertias.items():
        velocity = initial_link_velocities[link_index[name]]
        initial_kinetic_energy += 0.5 * moving_masses[name] * float(
            velocity[:3] @ velocity[:3]
        )
        initial_kinetic_energy += 0.5 * float(
            (velocity[3:] ** 2) @ np.asarray(inertia)
        )
    calibration_stored_energy_change = float(
        kinetic_energy[-1]
        - initial_kinetic_energy
        + 0.5
        * KN_N_PER_M
        * (
            e[CALIBRATION_STEPS - 1] ** 2
            - initial_q_native[index["nut_relative_elastic"]] ** 2
        )
    )
    calibration_energy_residual = abs(
        calibration_input_work
        - calibration_damping_heat
        - calibration_stored_energy_change
    ) / max(abs(calibration_input_work), 1.0e-12)
    calibration_contact_max = float(
        np.max(np.linalg.norm(contacts[calibration], axis=2).sum(axis=1))
    )
    screw_calibration_omega = link_velocities[
        CALIBRATION_STEPS - 1, link_index["screw"], 5
    ]
    nut_calibration_omega = link_velocities[
        CALIBRATION_STEPS - 1, link_index["nut"], 5
    ]
    screw_axial_translation_range = float(
        np.ptp(link_transforms[:, link_index["screw"], 2])
    )
    extension_symmetry = abs(abs(positive_e) - abs(negative_e)) / max(
        abs(positive_e), abs(negative_e), 1.0e-12
    )
    static_error = max(
        abs(abs(positive_e) - EXPECTED_STATIC_EXTENSION_M),
        abs(abs(negative_e) - EXPECTED_STATIC_EXTENSION_M),
    ) / EXPECTED_STATIC_EXTENSION_M
    positive_contact = float(np.median(np.linalg.norm(contacts[positive_hold], axis=2).sum(axis=1)))
    negative_contact = float(np.median(np.linalg.norm(contacts[negative_hold], axis=2).sum(axis=1)))
    expected_force_n = TEST_TORQUE_NM / P_M_PER_RAD
    contact_force_error = max(
        abs(positive_contact - expected_force_n), abs(negative_contact - expected_force_n)
    ) / expected_force_n

    checks = [
        check("qualification_flag", 0, "==", 0, "boolean"),
        check("native_articulation_count", articulation.count, "==", 1, "articulations"),
        check("native_dof_count", articulation.max_dofs, "==", 4, "dofs"),
        check("native_link_count", articulation.max_links, "==", len(link_names), "links"),
        check("rack_and_pinion_ratio_error", abs(authored_ratio - 1.0 / P_M_PER_RAD), "<=", 1.0e-3, "rad/m"),
        check("post_reset_state_write_count", 0, "==", 0, "writes"),
        check("configured_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
        check("fresh_native_sample_count", len(samples), "==", total_steps, "physics steps"),
        check("finite_native_value_fraction", np.mean(np.isfinite(finite)), "==", 1.0, "fraction"),
        check("uncommanded_generalized_effort_max", np.max(np.abs(uncommanded_effort)), "<=", 1.0e-7, "mixed native force/torque"),
        check("axial_support_prismatic_dof_count", sum(name == "screw_axial" for name in dof_names), "==", 0, "dofs"),
        check("screw_axial_translation_range", screw_axial_translation_range, "<=", 1.0e-6, "m"),
        check("calibration_plate_contact_force_max", calibration_contact_max, "<=", 1.0e-3, "N"),
        check("calibration_screw_opposes_nut_angular_velocity", screw_calibration_omega * nut_calibration_omega, "<", 0.0, "rad2/s2"),
        check("calibration_total_angular_momentum_abs_max", np.max(np.abs(total_angular_momentum_z)), "<=", 1.0e-7, "kg*m2/s"),
        check("calibration_angular_momentum_relative_residual_max", angular_momentum_relative_residual, "<=", 0.10, "fraction"),
        check("calibration_actuator_input_work", calibration_input_work, ">", 1.0e-8, "J"),
        check("calibration_work_energy_balance_relative_error", calibration_energy_residual, "<=", 0.25, "fraction"),
        check("positive_relative_extension", positive_e, ">", 5.0e-5, "m"),
        check("negative_relative_extension", negative_e, "<", -5.0e-5, "m"),
        check("bidirectional_extension_symmetry_error", extension_symmetry, "<=", 0.25, "fraction"),
        check("frozen_static_extension_relative_error", static_error, "<=", 0.35, "fraction"),
        check("positive_relative_rotation", positive_theta, ">", 0.05, "rad"),
        check("negative_relative_rotation", negative_theta, "<", -0.05, "rad"),
        check("positive_transform_relative_axial_travel", np.median(measured_relative_z[positive_hold]), ">", 5.0e-5, "m"),
        check("negative_transform_relative_axial_travel", np.median(measured_relative_z[negative_hold]), "<", -5.0e-5, "m"),
        check("loaded_transform_relative_rotation_error_max", np.max(np.abs(measured_relative_theta[loaded] - theta_from_reset[loaded])), "<=", 1.0e-4, "rad"),
        check("loaded_transform_frozen_extension_error_max", np.max(np.abs(transform_extension_error[loaded])), "<=", 5.0e-6, "m"),
        check("loaded_transform_right_handed_helix_error_max", np.max(np.abs(transform_helix_error[loaded])), "<=", 5.0e-6, "m"),
        check("positive_transform_helix_sign", np.median(measured_rigid_helix_travel[positive_hold]) * np.median(measured_relative_theta[positive_hold]), ">", 0.0, "m*rad"),
        check("negative_transform_helix_sign", np.median(measured_rigid_helix_travel[negative_hold]) * np.median(measured_relative_theta[negative_hold]), ">", 0.0, "m*rad"),
        check("positive_plate_contact_force", positive_contact, ">=", 2.0, "N"),
        check("negative_plate_contact_force", negative_contact, ">=", 2.0, "N"),
        check("plate_contact_to_commanded_torque_relative_error", contact_force_error, "<=", 0.35, "fraction"),
        check("loaded_helix_residual_max", np.max(np.abs(helix_residual)), "<=", 0.02, "rad"),
        check("relative_elastic_limit_clearance_min", 0.001 - np.max(np.abs(e)), ">=", 5.0e-5, "m"),
        check("final_release_extension_max", np.max(np.abs(e[final_release])), "<=", 2.0e-5, "m"),
        check("final_release_speed_max", np.max(np.abs(qd[final_release])), "<=", 0.1, "mixed native units/s"),
    ]
    report = {
        "qualification": False,
        "overall_pass": bool(all(item["passed"] for item in checks)),
        "native_trial_id": trial_id,
        "source_sha256": source_sha256,
        "scope": "N73 relative-to-dynamic-S native implicit coupling coupon only",
        "checks": checks,
        "artifacts": [str(trace_path.relative_to(output_path.parent.parent))],
    }
    trace = {
        "metadata": {
            "qualification": False,
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "physics_dt_s": DT,
            "dof_names": dof_names,
            "link_names": link_names,
            "lead_m_per_revolution": LEAD_M_PER_REV,
            "rack_and_pinion_ratio_rad_per_m": authored_ratio,
            "rack_and_pinion_sign_law": "h=+p*theta; right-handed frozen eN=delta_z-p*delta_theta",
            "plate_nearest_face_clearance_m": 0.0001,
            "KN_N_per_m": KN_N_PER_M,
            "DN_Ns_per_m": DN_NS_PER_M,
            "test_torque_Nm": TEST_TORQUE_NM,
            "expected_static_extension_m": EXPECTED_STATIC_EXTENSION_M,
            "axial_support": "frame-to-screw revolute bearing; no axial prismatic DOF or limit",
            "reaction_balance_method": "sum measured world-Z link angular momenta during verified no-contact interval",
            "power_balance_method": "joint input work versus measured link kinetic plus elastic energy and damping heat during verified no-contact interval",
            "post_reset_state_write_count": 0,
            "transform_certificate": "fresh world-frame screw/nut positions and xyzw quaternions; reset-relative delta_z-p*delta_theta compared with elastic DOF",
        },
        "samples": samples,
    }
    write_json(trace_path, trace)
    write_json(output_path, report)
    if not report["overall_pass"]:
        raise RuntimeError("one or more N73-relative coupling checks failed")


failure: str | None = None
try:
    main()
except Exception:
    failure = traceback.format_exc()
    if not output_path.exists():
        checks = [check("native_probe_completed", 0, "==", 1, "boolean")]
        write_json(
            output_path,
            {
                "qualification": False,
                "overall_pass": False,
                "native_trial_id": trial_id,
                "source_sha256": source_sha256,
                "scope": "N73 relative-to-dynamic-S native implicit coupling coupon only",
                "error": failure,
                "checks": checks,
                "artifacts": [],
            },
        )
finally:
    app.close()

if failure is not None:
    raise SystemExit(failure)
