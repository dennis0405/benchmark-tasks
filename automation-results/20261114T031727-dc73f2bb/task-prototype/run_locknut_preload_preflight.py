#!/usr/bin/env python3
"""Native M8x1 nut/cover detailed-contact measurement probe.

This is deliberately not fixture qualification or a robot demonstration.  It
uses the SHA-pinned Factory M8 collision meshes at the already-tested +36 deg
phase, tightens the nut against a fixed cover with a bounded native articulation
effort, removes the effort completely, and samples the physical contact for one
fresh second.  It asks only whether the native detailed-contact API can provide
the frozen scorer's sum of compressive normal impulses divided by physics dt,
and whether that reconstruction agrees with the filtered contact vector.  It
does not model or claim released preload, lock retention, or the frozen N73/S
constitutive coupling.
No body pose, velocity, joint state, collision flag, or scorer state is written
after reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--torque", type=float, default=0.012)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import IsaacEvents, PhysxCfg, PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "input/nvidia-factory-reference"
BOLT_OBJ = REFERENCE / "factory_bolt_m8_tight.obj"
NUT_OBJ = REFERENCE / "factory_nut_m8_tight_subdiv_3x.obj"
EXPECTED = {
    BOLT_OBJ: "b5affc6583fe2b6038e1b53c74102eabd0842ecc05fe614ee04dc7b32385c123",
    NUT_OBJ: "c4583cce9525314dcf7be900b6544cbcb4f7112ea54034786a2de2bf604947bb",
}
DT = 1.0 / 240.0
STEPS = 12 * 240
CONTACT_CAPACITY = 4096


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def obj(path: Path):
    points, counts, indices = [], [], []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.strip().split()
            if not fields or fields[0] == "#":
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise RuntimeError(f"malformed vertex {path}:{line_no}")
                xyz = tuple(float(fields[i]) for i in range(1, 4))
                if not all(math.isfinite(value) for value in xyz):
                    raise RuntimeError(f"non-finite vertex {path}:{line_no}")
                points.append(Gf.Vec3f(xyz[0], xyz[1], 0.8 * xyz[2]))
            elif fields[0] == "f":
                face = []
                for token in fields[1:]:
                    raw_index = int(token.split("/", 1)[0])
                    index = raw_index - 1 if raw_index > 0 else len(points) + raw_index
                    if not 0 <= index < len(points):
                        raise RuntimeError(f"face index out of range {path}:{line_no}")
                    face.append(index)
                for offset in range(1, len(face) - 1):
                    counts.append(3)
                    indices.extend((face[0], face[offset], face[offset + 1]))
    if not points or not counts:
        raise RuntimeError(f"empty OBJ: {path}")
    return points, counts, indices


def body(stage, path: str, z: float, mass: float, inertia=None, yaw=0.0):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
    if yaw:
        xform.AddRotateZOp().Set(yaw)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(mass)
    if inertia is not None:
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
    physx.CreateDisableGravityAttr(True)
    physx.CreateLinearDampingAttr(0.0)
    physx.CreateAngularDampingAttr(0.0)
    PhysxSchema.PhysxContactReportAPI.Apply(xform.GetPrim()).CreateThresholdAttr(0.0)
    return xform


def sdf_mesh(stage, path: str, source: Path, resolution: int, material):
    points, counts, indices = obj(source)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateContactOffsetAttr(0.005)
    collision.CreateRestOffsetAttr(0.0)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
    sdf.CreateSdfResolutionAttr(resolution)
    sdf.CreateSdfSubgridResolutionAttr(6)
    sdf.CreateSdfBitsPerSubgridPixelAttr("BitsPerPixel16")
    sdf.CreateSdfEnableRemeshingAttr(False)
    sdf.CreateSdfMarginAttr(0.01)
    sdf.CreateSdfNarrowBandThicknessAttr(0.01)
    sdf.CreateSdfTriangleCountReductionFactorAttr(1.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return mesh


def material(stage, path: str, static: float, dynamic: float):
    result = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(result.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return result


def check(name, measured, operator, threshold, unit):
    return {"name": name, "measured": float(measured), "operator": operator, "threshold": threshold, "unit": unit}


def run():
    trial_id = str(uuid.uuid4())
    source_sha256 = digest(Path(__file__).resolve())
    if not 0.0 < args.torque <= 0.02:
        raise ValueError("--torque must be in (0, 0.02] N*m")
    verified = sum(digest(path) == expected for path, expected in EXPECTED.items())
    if verified != 2:
        raise RuntimeError("pinned Factory M8 asset hash mismatch")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT,
        device=args.device,
        gravity=(0.0, 0.0, 0.0),
        physics=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,
            gpu_collision_stack_size=2**26,
            gpu_max_num_partitions=1,
        ),
    ))
    stage = sim.stage
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    bolt_mat = material(stage, "/World/BoltMaterial", 0.75, 0.75)
    nut_mat = material(stage, "/World/NutMaterial", 0.35, 0.28)
    cover_mat = material(stage, "/World/CoverMaterial", 0.08, 0.05)

    bolt = body(stage, "/World/Bolt", 0.5, 0.05)
    bolt_mesh = sdf_mesh(stage, "/World/Bolt/Collision", BOLT_OBJ, 512, bolt_mat)
    mount = UsdPhysics.FixedJoint.Define(stage, "/World/BoltMount")
    mount.CreateBody1Rel().SetTargets([bolt.GetPath()])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    root = UsdGeom.Xform.Define(stage, "/World/NutAssembly")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art_api.CreateSolverPositionIterationCountAttr(192)
    art_api.CreateSolverVelocityIterationCountAttr(1)
    frame = body(stage, "/World/NutAssembly/frame", 0.504, 1.0)
    carrier = body(stage, "/World/NutAssembly/carrier", 0.504, 0.001, (1e-7, 1e-7, 1e-7))
    nut = body(stage, "/World/NutAssembly/nut", 0.504, 0.009, (1.4e-7, 1.4e-7, 2.6e-7), 36.0)
    nut_mesh = sdf_mesh(stage, "/World/NutAssembly/nut/Collision", NUT_OBJ, 256, nut_mat)
    frame_mount = UsdPhysics.FixedJoint.Define(stage, "/World/NutAssembly/mount")
    frame_mount.CreateBody1Rel().SetTargets([frame.GetPath()])
    frame_mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.504))
    axial = UsdPhysics.PrismaticJoint.Define(stage, "/World/NutAssembly/axial")
    axial.CreateBody0Rel().SetTargets([frame.GetPath()])
    axial.CreateBody1Rel().SetTargets([carrier.GetPath()])
    axial.CreateAxisAttr("Z")
    axial.CreateLowerLimitAttr(-0.01)
    axial.CreateUpperLimitAttr(0.01)
    rotation = UsdPhysics.RevoluteJoint.Define(stage, "/World/NutAssembly/rotation")
    rotation.CreateBody0Rel().SetTargets([carrier.GetPath()])
    rotation.CreateBody1Rel().SetTargets([nut.GetPath()])
    rotation.CreateAxisAttr("Z")
    phase = math.radians(-18.0)
    rotation.CreateLocalRot1Attr(Gf.Quatf(math.cos(phase), Gf.Vec3f(0.0, 0.0, math.sin(phase))))
    rotation.CreateLowerLimitAttr(-7200.0)
    rotation.CreateUpperLimitAttr(7200.0)

    cover_z = 0.5101
    cover = body(stage, "/World/Cover", cover_z, 0.05, (5e-6, 5e-6, 5e-6))
    for index, (x, y) in enumerate(((0.006, 0.0), (-0.006, 0.0), (0.0, 0.006), (0.0, -0.006))):
        pad = UsdGeom.Cube.Define(stage, f"/World/Cover/Pad{index}")
        pad.CreateSizeAttr(1.0)
        pad.AddTranslateOp().Set(Gf.Vec3d(x, y, -0.0005))
        pad.AddScaleOp().Set(Gf.Vec3f(0.002, 0.002, 0.001))
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        pad_api = PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
        pad_api.CreateContactOffsetAttr(0.00015)
        pad_api.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(
            cover_mat,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
    cover_joint = UsdPhysics.FixedJoint.Define(stage, "/World/CoverMount")
    cover_joint.CreateBody1Rel().SetTargets([cover.GetPath()])
    cover_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, cover_z))

    callbacks = {"count": 0, "dts": [], "errors": []}
    def post(dt):
        try:
            callbacks["count"] += 1
            callbacks["dts"].append(float(dt))
        except Exception as exc:
            callbacks["errors"].append(f"{type(exc).__name__}: {exc}")
    handle = PhysxManager.register_callback(post, IsaacEvents.POST_PHYSICS_STEP, name="nut_cover_contact_measurement", wrap_weak_ref=False)
    try:
        sim.reset()
        physics = PhysxManager.get_physics_sim_view()
        if physics is None:
            raise RuntimeError("native PhysX tensor view unavailable")
        physics.set_subspace_roots("/")
        articulation = physics.create_articulation_view("/World/NutAssembly")
        dofs = list(articulation.get_metatype(0).dof_names)
        rotation_index, axial_index = dofs.index("rotation"), dofs.index("axial")
        nut_view = physics.create_rigid_body_view([str(nut.GetPath())])
        cover_view = physics.create_rigid_body_view([str(cover.GetPath())])
        contact = physics.create_rigid_contact_view(
            [str(nut.GetPath())], filter_patterns=[[str(cover.GetPath())]], max_contact_data_count=CONTACT_CAPACITY
        )
        bolt_sdf = physics.create_sdf_shape_view(str(bolt_mesh.GetPath()), 8)
        nut_sdf = physics.create_sdf_shape_view(str(nut_mesh.GetPath()), 8)
        if articulation.count != 1 or articulation.max_dofs != 2:
            raise RuntimeError("unexpected articulation dimensions")
        if contact.sensor_count != 1 or contact.filter_count != 1:
            raise RuntimeError("nut/cover filtered contact did not resolve")

        ids = wp.array([0], dtype=wp.uint32, device=args.device)
        samples = []
        finite_count = value_count = 0
        max_contacts = saturation_steps = 0
        minimum_separation = math.inf
        last_speed = 0.0
        for step in range(STEPS):
            time_s = step * DT
            phase_name = "settle" if time_s < 0.5 else "tighten" if time_s < 9.0 else "unpowered"
            applied = max(-args.torque, min(0.0, args.torque * (-3.0 - last_speed) / 3.0)) if phase_name == "tighten" else 0.0
            efforts = np.zeros((1, articulation.max_dofs), dtype=np.float32)
            efforts[0, rotation_index] = applied
            articulation.set_dof_actuation_forces(wp.array(efforts, device=args.device), ids)
            sim.step(render=False)
            q = articulation.get_dof_positions().numpy()[0]
            qd = articulation.get_dof_velocities().numpy()[0]
            native_effort = articulation.get_dof_actuation_forces().numpy()[0]
            nut_twist = nut_view.get_velocities().numpy()[0]
            cover_pose = cover_view.get_transforms().numpy()[0]
            force = contact.get_contact_force_matrix(dt=DT).numpy().reshape(1, 1, 3)[0, 0]
            impulse_data = tuple(value.numpy() for value in contact.get_contact_data(dt=1.0))
            if len(impulse_data) != 6:
                raise RuntimeError(f"unexpected detailed-contact tuple length {len(impulse_data)}")
            impulses, _, normals, separations, counts, starts = impulse_data
            count = int(counts.reshape(-1)[0])
            start = int(starts.reshape(-1)[0])
            max_contacts = max(max_contacts, count)
            saturation_steps += int(count >= CONTACT_CAPACITY)
            if count > CONTACT_CAPACITY or start < 0 or start + count > CONTACT_CAPACITY:
                raise RuntimeError("detailed contact range exceeds native buffer")
            if count:
                sep = separations.reshape(-1)[start:start + count]
                minimum_separation = min(minimum_separation, float(np.min(sep)))
            scalar_impulses = impulses.reshape(-1)[start:start + count]
            pair_normals = normals.reshape(-1, 3)[start:start + count]
            if np.any(scalar_impulses < -1e-12):
                raise RuntimeError("native normal impulse contains a negative scalar")
            compressive_normal_impulse = float(np.sum(scalar_impulses))
            reconstructed_impulse = (
                np.sum(scalar_impulses[:, None] * pair_normals, axis=0)
                if count else np.zeros(3, dtype=np.float64)
            )
            normal = compressive_normal_impulse / DT
            vector_error = float(np.max(np.abs(reconstructed_impulse - force * DT)))
            values = np.asarray([
                *q, *qd, *native_effort, *nut_twist, *cover_pose, *force,
                compressive_normal_impulse, *reconstructed_impulse, vector_error,
            ], dtype=np.float64)
            finite_count += int(np.count_nonzero(np.isfinite(values)))
            value_count += int(values.size)
            last_speed = float(qd[rotation_index])
            samples.append({
                "step": step + 1, "time_s": (step + 1) * DT, "phase": phase_name,
                "commanded_torque_nm": applied, "native_torque_nm": float(native_effort[rotation_index]),
                "angle_rad": float(q[rotation_index]), "omega_rad_s": last_speed,
                "axial_m": float(q[axial_index]), "axial_speed_m_s": float(qd[axial_index]),
                "cover_z_m": float(cover_pose[2]), "seat_gap_m": 0.5104 + float(q[axial_index]) - float(cover_pose[2]),
                "compressive_normal_impulse_ns": compressive_normal_impulse,
                "compressive_normal_force_n": normal,
                "filtered_force_vector_n": force.tolist(),
                "reconstructed_impulse_vector_ns": reconstructed_impulse.tolist(),
                "detailed_vector_impulse_error_ns": vector_error,
                "contact_count": count,
            })
        if callbacks["errors"]:
            raise RuntimeError(f"post-step callback failure: {callbacks['errors']}")
        unpowered = [row for row in samples if row["phase"] == "unpowered"]
        tightening = [row for row in samples if row["phase"] == "tighten"]
        if len(unpowered) != 720:
            raise RuntimeError("unpowered window is not exactly three seconds")
        certificate = unpowered[-240:]
        penetration = max(0.0, -minimum_separation) if math.isfinite(minimum_separation) else 0.0
        checks = [
            check("verified_original_obj_hash_count", verified, "==", 2, "assets"),
            check("native_sdf_shape_count", bolt_sdf.count + nut_sdf.count, "==", 2, "shapes"),
            check("native_sdf_view_valid_count", int(bool(bolt_sdf.check())) + int(bool(nut_sdf.check())), "==", 2, "views"),
            check("native_articulation_count", articulation.count, "==", 1, "articulations"),
            check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
            check("actual_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
            check("post_physics_callback_count", callbacks["count"], "==", STEPS, "callbacks"),
            check("post_physics_callback_dt_error_max", max(abs(value - DT) for value in callbacks["dts"]), "<=", 1e-9, "s"),
            check("fresh_native_sample_count", len(samples), "==", STEPS, "physics steps"),
            check("finite_native_value_fraction", finite_count / value_count, "==", 1.0, "fraction"),
            check("detailed_contact_buffer_saturation_steps", saturation_steps, "==", 0, "physics steps"),
            check("maximum_observed_contact_count", max_contacts, "<=", CONTACT_CAPACITY - 1, "entries per physics step"),
            check("maximum_contact_penetration", penetration, "<=", 0.00075, "m"),
            check("seat_pair_compressive_normal_force_peak", max(row["compressive_normal_force_n"] for row in samples), "<=", 250.0, "N"),
            check("nut_linear_speed_peak", max(abs(row["axial_speed_m_s"]) for row in samples), "<=", 0.25, "m/s"),
            check("nut_angular_speed_peak", max(abs(row["omega_rad_s"]) for row in samples), "<=", 10.0, "rad/s"),
            check("tightening_compressive_normal_force_peak", max(row["compressive_normal_force_n"] for row in tightening), ">=", 2.0, "N"),
            check("detailed_contact_impulse_entry_count", sum(row["contact_count"] for row in samples), ">=", 1, "entries"),
            check("detailed_to_filtered_vector_impulse_error_max", max(row["detailed_vector_impulse_error_ns"] for row in samples), "<=", 1e-5, "N*s"),
            check("unpowered_actuator_effort_max", max(abs(row["native_torque_nm"]) for row in certificate), "<=", 1e-7, "N*m"),
            check("unpowered_measurement_fresh_window", len(certificate), "==", 240, "physics steps"),
        ]
        trace = {
            "metadata": {
                "native_trial_id": trial_id,
                "source_sha256": source_sha256,
                "qualification": False,
                "scope": "task-pitch M8 nut/fixed-cover detailed normal-impulse measurement API discovery only; no N73/S coupling, preload, lock, fixture, or qualification claim",
                "thread_pitch_m_per_rev": 0.001,
                "authored_phase_deg": 36.0,
                "cover_support": "native fixed joint; measurement coupon only",
                "compressive_metric": "sum of detailed nonnegative normal impulses divided by actual physics dt",
                "asset_sha256": {path.name: digest(path) for path in EXPECTED},
            },
            "samples": samples,
        }
        trace_path = args.trace.resolve()
        output_path = args.output.resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
        report = {"checks": checks, "artifacts": [str(trace_path.relative_to(ROOT))]}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        handle.deregister()


try:
    run()
finally:
    app.close()
