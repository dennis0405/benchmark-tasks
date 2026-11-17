#!/usr/bin/env python3
"""Native unpowered M8 SDF initial-geometry probe; never qualification.

The preceding full-contact trial showed large nut/bolt contact forces and motion
on its first zero-command step.  This probe changes only authored pre-reset
geometry selected by ``--case``.  It applies zero generalized effort for two
seconds and measures fresh native states and the exact nut/bolt contact stream.
There are no post-reset state writes, drives, mimic constraints, or score state.

The original NVIDIA Factory OBJ bytes, scale, masses, material adaptation,
collision offsets, SDF resolutions, and solver settings remain unchanged.  A
2.5 mm lift is exactly two stock M8 pitches, so the ``lifted-original-cook``
case removes bolt-head seating while preserving the -90 degree helical phase.
The phase cases retain that lift and vary only authored yaw.  The remeshed case
retains lift and yaw and changes only the documented native SDF cook switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback
import uuid

from bounded_contact_archive import BoundedContactArchive
from isaaclab.app import AppLauncher


CASES = {
    "fit-36": {"yaw_deg": 36.0, "z_offset_m": 0.0025, "remesh": False},
    "lifted-original-cook": {"yaw_deg": -90.0, "z_offset_m": 0.0025, "remesh": False},
    "lifted-quarter-phase": {"yaw_deg": 0.0, "z_offset_m": 0.0025, "remesh": False},
    "lifted-half-phase": {"yaw_deg": 90.0, "z_offset_m": 0.0025, "remesh": False},
    "lifted-remeshed": {"yaw_deg": -90.0, "z_offset_m": 0.0025, "remesh": True},
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--trace", required=True)
parser.add_argument("--case", required=True, choices=tuple(CASES))
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
from isaaclab_physx.physics import IsaacEvents, PhysxCfg, PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "input" / "nvidia-factory-reference"
BOLT_OBJ = REFERENCE / "factory_bolt_m8_tight.obj"
NUT_OBJ = REFERENCE / "factory_nut_m8_tight_subdiv_3x.obj"
ARCHIVE_HELPER = Path(__file__).with_name("bounded_contact_archive.py")
EXPECTED_HASHES = {
    "bolt": "b5affc6583fe2b6038e1b53c74102eabd0842ecc05fe614ee04dc7b32385c123",
    "nut": "c4583cce9525314dcf7be900b6544cbcb4f7112ea54034786a2de2bf604947bb",
    "archive": "13e998f29f8055c55343f68abdb706314e4647d2bb9b6ac40605d782656a0dd5",
}
DT = 1.0 / 240.0
STEPS = 480
FINAL_WINDOW_STEPS = 120
CONTACT_CAPACITY = 4096
CONTACT_OFFSET_M = 0.005
REST_OFFSET_M = 0.0
NUT_MASS_KG = 0.029
CARRIER_MASS_KG = 0.001
POSITION_ITERATIONS = 192
VELOCITY_ITERATIONS = 1
SAMPLES: list[dict] = []
METADATA: dict = {}
ARCHIVE: BoundedContactArchive | None = None
ARCHIVE_INDEX: dict | None = None
RAW_DIGEST = None
RAW_COUNT = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def check(name: str, measured: float, operator: str, threshold: float, unit: str) -> dict:
    value = float(measured)
    predicates = {
        "==": lambda a, b: a == b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
    }
    return {
        "name": name,
        "measured": value,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(math.isfinite(value) and predicates[operator](value, float(threshold))),
    }


def as_cpu(value) -> torch.Tensor:
    return wp.to_torch(value).detach().cpu().clone()


def load_obj(path: Path) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    vertices: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.strip().split()
            if not fields or fields[0] == "#":
                continue
            if fields[0] == "v":
                xyz = [float(value) for value in fields[1:4]]
                if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
                    raise RuntimeError(f"invalid vertex at {path.name}:{line_no}")
                vertices.append(Gf.Vec3f(*xyz))
            elif fields[0] == "f":
                polygon = []
                for token in fields[1:]:
                    raw_index = int(token.split("/", 1)[0])
                    index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    if index < 0 or index >= len(vertices):
                        raise RuntimeError(f"invalid face index at {path.name}:{line_no}")
                    polygon.append(index)
                for offset in range(1, len(polygon) - 1):
                    counts.append(3)
                    indices.extend((polygon[0], polygon[offset], polygon[offset + 1]))
    if not vertices or not counts:
        raise RuntimeError(f"empty OBJ geometry: {path}")
    return vertices, counts, indices


def create_body(stage, path: str, position: Gf.Vec3d, mass: float, yaw: float | None = None):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(position)
    if yaw is not None:
        body.AddRotateZOp().Set(yaw)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    physx.CreateDisableGravityAttr(True)
    physx.CreateLinearDampingAttr(0.0)
    physx.CreateAngularDampingAttr(0.0)
    return body


def create_sdf(stage, path: str, source: Path, resolution: int, material, remesh: bool):
    points, counts, indices = load_obj(source)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateContactOffsetAttr(CONTACT_OFFSET_M)
    collision.CreateRestOffsetAttr(REST_OFFSET_M)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
    sdf.CreateSdfResolutionAttr(resolution)
    sdf.CreateSdfSubgridResolutionAttr(6)
    sdf.CreateSdfBitsPerSubgridPixelAttr("BitsPerPixel16")
    sdf.CreateSdfEnableRemeshingAttr(remesh)
    sdf.CreateSdfMarginAttr(0.01)
    sdf.CreateSdfNarrowBandThicknessAttr(0.01)
    sdf.CreateSdfTriangleCountReductionFactorAttr(1.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    PhysxSchema.PhysxContactReportAPI.Apply(mesh.GetPrim()).CreateThresholdAttr(0.0)
    return mesh


def finish_archive() -> dict | None:
    global ARCHIVE_INDEX
    if ARCHIVE is not None and ARCHIVE_INDEX is None:
        ARCHIVE_INDEX = ARCHIVE.finish()
    return ARCHIVE_INDEX


def main() -> None:
    global ARCHIVE, ARCHIVE_INDEX, METADATA, RAW_COUNT, RAW_DIGEST
    for label, path in (("bolt", BOLT_OBJ), ("nut", NUT_OBJ), ("archive", ARCHIVE_HELPER)):
        if not path.is_file() or sha256(path) != EXPECTED_HASHES[label]:
            raise RuntimeError(f"missing or hash-mismatched {label} source")
    case = CASES[args.case]
    ARCHIVE = BoundedContactArchive(ROOT, trial_id)
    RAW_DIGEST = hashlib.sha256()

    cfg = sim_utils.SimulationCfg(
        dt=DT,
        device=args.device,
        gravity=(0.0, 0.0, 0.0),
        physics=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=POSITION_ITERATIONS,
            max_velocity_iteration_count=VELOCITY_ITERATIONS,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,
            gpu_collision_stack_size=2**26,
            gpu_max_num_partitions=1,
        ),
    )
    sim = sim_utils.SimulationContext(cfg)
    stage = sim.stage
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    def material(path: str, friction: float):
        result = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(result.GetPrim())
        api.CreateStaticFrictionAttr(friction)
        api.CreateDynamicFrictionAttr(friction)
        api.CreateRestitutionAttr(0.0)
        return result

    bolt_material = material("/World/BoltMaterial", 0.75)
    nut_material = material("/World/NutMaterial", 0.01)
    bolt = create_body(stage, "/World/Bolt", Gf.Vec3d(0.0, 0.0, 0.5), 0.05)
    PhysxSchema.PhysxContactReportAPI.Apply(bolt.GetPrim()).CreateThresholdAttr(0.0)
    # The warning and CPU topology defect are specific to the nut.  Keep the
    # already-watertight bolt cook unchanged so the remeshing comparison has
    # exactly one geometry/cooking axis.
    bolt_mesh = create_sdf(
        stage, "/World/Bolt/Collision", BOLT_OBJ, 512, bolt_material, False
    )
    bolt_mount = UsdPhysics.FixedJoint.Define(stage, "/World/BoltMount")
    bolt_mount.CreateBody1Rel().SetTargets([bolt.GetPath()])
    bolt_mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    root = UsdGeom.Xform.Define(stage, "/World/NutAssembly")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    art_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)
    initial_z = 0.5 + float(case["z_offset_m"])
    frame = create_body(stage, "/World/NutAssembly/frame", Gf.Vec3d(0.0, 0.0, initial_z), 1.0)
    carrier = create_body(
        stage, "/World/NutAssembly/carrier", Gf.Vec3d(0.0, 0.0, initial_z), CARRIER_MASS_KG
    )
    UsdPhysics.MassAPI.Apply(carrier.GetPrim()).CreateDiagonalInertiaAttr(
        Gf.Vec3f(1.0e-7, 1.0e-7, 1.0e-7)
    )
    nut = create_body(
        stage,
        "/World/NutAssembly/nut",
        Gf.Vec3d(0.0, 0.0, initial_z),
        NUT_MASS_KG,
        float(case["yaw_deg"]),
    )
    PhysxSchema.PhysxContactReportAPI.Apply(nut.GetPrim()).CreateThresholdAttr(0.0)
    nut_mesh = create_sdf(
        stage,
        "/World/NutAssembly/nut/Collision",
        NUT_OBJ,
        256,
        nut_material,
        bool(case["remesh"]),
    )
    mount = UsdPhysics.FixedJoint.Define(stage, "/World/NutAssembly/mount")
    mount.CreateBody1Rel().SetTargets([frame.GetPath()])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, initial_z))
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
    half_frame = math.radians(-float(case["yaw_deg"]) / 2.0)
    rotation.CreateLocalRot1Attr(
        Gf.Quatf(math.cos(half_frame), Gf.Vec3f(0.0, 0.0, math.sin(half_frame)))
    )
    rotation.CreateLowerLimitAttr(-7200.0)
    rotation.CreateUpperLimitAttr(7200.0)

    callbacks = {"count": 0, "dts": [], "errors": []}

    def post_callback(dt_value):
        try:
            callbacks["count"] += 1
            callbacks["dts"].append(float(dt_value))
        except Exception as exc:
            callbacks["errors"].append(f"{type(exc).__name__}: {exc}")

    handle = PhysxManager.register_callback(
        post_callback,
        IsaacEvents.POST_PHYSICS_STEP,
        name=f"sdf_initial_geometry_{trial_id}",
        wrap_weak_ref=False,
    )
    try:
        sim.reset()
        physics = PhysxManager.get_physics_sim_view()
        if physics is None:
            raise RuntimeError("native PhysX tensor view unavailable")
        physics.set_subspace_roots("/")
        articulation = physics.create_articulation_view("/World/NutAssembly")
        dof_names = list(articulation.get_metatype(0).dof_names)
        axial_index = dof_names.index("axial")
        rotation_index = dof_names.index("rotation")
        all_indices = wp.array([0], dtype=wp.uint32, device=args.device)
        nut_view = physics.create_rigid_body_view([str(nut.GetPath())])
        contact_view = physics.create_rigid_contact_view(
            [str(nut.GetPath())],
            filter_patterns=[[str(bolt.GetPath())]],
            max_contact_data_count=CONTACT_CAPACITY,
        )
        bolt_sdf = physics.create_sdf_shape_view(str(bolt_mesh.GetPath()), 8)
        nut_sdf = physics.create_sdf_shape_view(str(nut_mesh.GetPath()), 8)
        if articulation.count != 1 or articulation.max_dofs != 2:
            raise RuntimeError("unexpected native articulation shape")
        if contact_view.sensor_count != 1 or contact_view.filter_count != 1:
            raise RuntimeError("native nut/bolt filtered contact view unresolved")

        METADATA = {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "qualification": False,
            "scope": "unpowered stock-M8 authored initial geometry/SDF cook comparison only",
            "case": args.case,
            "authored_initial_yaw_deg": case["yaw_deg"],
            "authored_initial_z_offset_m": case["z_offset_m"],
            "authored_nut_sdf_remeshing": case["remesh"],
            "authored_bolt_sdf_remeshing": False,
            "asset_sha256": {"bolt": sha256(BOLT_OBJ), "nut": sha256(NUT_OBJ)},
            "post_reset_state_writes": 0,
            "commanded_effort_n_nm": [0.0, 0.0],
            "dof_names": dof_names,
        }
        zero = np.zeros((1, articulation.max_dofs), dtype=np.float32)
        finite_values = 0
        numeric_values = 0
        peak_force = 0.0
        minimum_separation = math.inf
        maximum_count = 0
        saturated_steps = 0
        for step in range(1, STEPS + 1):
            articulation.set_dof_actuation_forces(
                wp.array(zero, dtype=wp.float32, device=args.device), all_indices
            )
            sim.step(render=False)
            q = as_cpu(articulation.get_dof_positions())[0].numpy()
            qd = as_cpu(articulation.get_dof_velocities())[0].numpy()
            effort = as_cpu(articulation.get_dof_actuation_forces())[0].numpy()
            twist = as_cpu(nut_view.get_velocities())[0].numpy()
            force_matrix = (
                as_cpu(contact_view.get_contact_force_matrix(dt=DT)).reshape(1, 1, 3)[0, 0].numpy()
            )
            forces, points, normals, separations, counts, starts = tuple(
                as_cpu(value) for value in contact_view.get_contact_data(dt=DT)
            )
            count = int(counts.reshape(-1)[0].item())
            start = int(starts.reshape(-1)[0].item())
            maximum_count = max(maximum_count, count)
            saturated_steps += int(count >= CONTACT_CAPACITY)
            if count > CONTACT_CAPACITY or start < 0 or start + count > CONTACT_CAPACITY:
                raise RuntimeError("detailed contact range exceeds native buffer")
            evidence = []
            for index in range(start, start + count):
                scalar_force = float(forces.reshape(-1)[index].item())
                point = [float(value) for value in points.reshape(-1, 3)[index].tolist()]
                normal = [float(value) for value in normals.reshape(-1, 3)[index].tolist()]
                separation = float(separations.reshape(-1)[index].item())
                payload = struct.pack(
                    "<qq8d", step, index, scalar_force, *point, *normal, separation
                )
                RAW_DIGEST.update(payload)
                ARCHIVE.append(payload)
                RAW_COUNT += 1
                peak_force = max(peak_force, scalar_force)
                minimum_separation = min(minimum_separation, separation)
                if len(evidence) < 8:
                    evidence.append(
                        {
                            "native_contact_index": index,
                            "normal_force_n": scalar_force,
                            "point_w_m": point,
                            "normal_w": normal,
                            "separation_m": separation,
                        }
                    )
            values = np.asarray([*q, *qd, *effort, *twist, *force_matrix], dtype=np.float64)
            finite_values += int(np.count_nonzero(np.isfinite(values)))
            numeric_values += int(values.size)
            SAMPLES.append(
                {
                    "step": step,
                    "q_axial_m": float(q[axial_index]),
                    "q_rotation_rad": float(q[rotation_index]),
                    "v_axial_m_s": float(qd[axial_index]),
                    "v_rotation_rad_s": float(qd[rotation_index]),
                    "native_effort": effort.tolist(),
                    "nut_twist": twist.tolist(),
                    "filtered_contact_force_n": force_matrix.tolist(),
                    "contact_count": count,
                    "raw_contact_evidence": evidence,
                }
            )
        if callbacks["errors"]:
            raise RuntimeError(f"post-step callback failure: {callbacks['errors']}")
        final_rows = SAMPLES[-FINAL_WINDOW_STEPS:]
        archive_index = finish_archive()
        if archive_index is None:
            raise RuntimeError("raw contact archive missing")
        # Keep the same small private-archive index in metadata as well as the
        # derived trace section.  The raw gzip chunks themselves remain outside
        # measurements and are never listed as public report artifacts.
        METADATA["raw_contact_archive"] = archive_index
        dt_error = max(abs(value - DT) for value in callbacks["dts"])
        penetration = max(0.0, -minimum_separation) if math.isfinite(minimum_separation) else 0.0
        checks = [
            check("qualification_flag", 0, "==", 0, "boolean"),
            check("verified_original_obj_hash_count", 2, "==", 2, "assets"),
            check("native_sdf_shape_count", bolt_sdf.count + nut_sdf.count, "==", 2, "shapes"),
            check("native_sdf_view_valid_count", int(bool(bolt_sdf.check())) + int(bool(nut_sdf.check())), "==", 2, "views"),
            check("native_articulation_count", articulation.count, "==", 1, "articulations"),
            check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
            check("post_reset_state_write_count", 0, "==", 0, "writes"),
            check("actual_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
            check("post_physics_callback_count", callbacks["count"], "==", STEPS, "callbacks"),
            check("post_physics_callback_dt_error_max", dt_error, "<=", 1.0e-9, "s"),
            check("fresh_native_sample_count", len(SAMPLES), "==", STEPS, "physics steps"),
            check("finite_native_value_fraction", finite_values / numeric_values, "==", 1.0, "fraction"),
            check("detailed_contact_buffer_saturation_steps", saturated_steps, "==", 0, "physics steps"),
            check("maximum_observed_contact_count", maximum_count, "<=", CONTACT_CAPACITY - 1, "entries per physics step"),
            check("filtered_contact_entry_count", RAW_COUNT, ">=", 1, "entries"),
            check("maximum_contact_penetration", penetration, "<=", 0.00075, "m"),
            check("unpowered_pair_force_peak", max(np.linalg.norm(row["filtered_contact_force_n"]) for row in SAMPLES), "<=", 250.0, "N"),
            check("unpowered_nut_linear_speed_peak", max(np.linalg.norm(row["nut_twist"][:3]) for row in SAMPLES), "<=", 0.25, "m/s"),
            check("unpowered_nut_angular_speed_peak", max(np.linalg.norm(row["nut_twist"][3:]) for row in SAMPLES), "<=", 10.0, "rad/s"),
            check("final_window_axial_speed_max", max(abs(row["v_axial_m_s"]) for row in final_rows), "<=", 0.02, "m/s"),
            check("final_window_angular_speed_max", max(abs(row["v_rotation_rad_s"]) for row in final_rows), "<=", 0.1, "rad/s"),
            check("private_raw_archive_count_matches", int(archive_index["records"] == RAW_COUNT), "==", 1, "boolean"),
            check("private_raw_archive_digest_matches", int(archive_index["stream_sha256"] == RAW_DIGEST.hexdigest()), "==", 1, "boolean"),
            check("private_raw_archive_round_trip_verified", int(archive_index["all_appended_records_round_trip_verified"]), "==", 1, "boolean"),
        ]
        trace = {
            "metadata": METADATA,
            "derived": {
                "raw_contact_record_count": RAW_COUNT,
                "raw_contact_stream_sha256": RAW_DIGEST.hexdigest(),
                "raw_contact_archive": archive_index,
                "raw_contact_force_peak_n": peak_force,
                "raw_contact_minimum_separation_m": (
                    minimum_separation if math.isfinite(minimum_separation) else None
                ),
                "maximum_observed_contact_count": maximum_count,
            },
            "samples": SAMPLES,
        }
        write_json(trace_path, trace)
        report = {
            "qualification": False,
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "checks": checks,
            "overall_pass": all(item["passed"] for item in checks),
            "artifacts": [str(trace_path.relative_to(ROOT))],
        }
        write_json(output_path, report)
        print(json.dumps(report, allow_nan=False))
        if not report["overall_pass"]:
            raise SystemExit(1)
    finally:
        PhysxManager.deregister_callback(handle)


def failure_report(exc: Exception) -> None:
    archive_error = None
    archive_index = None
    try:
        archive_index = finish_archive()
    except Exception as error:
        archive_error = f"{type(error).__name__}: {error}"
    raw_digest = RAW_DIGEST.hexdigest() if RAW_DIGEST is not None else None
    archive_count_matches = int(
        archive_index is not None and archive_index.get("records") == RAW_COUNT
    )
    archive_digest_matches = int(
        archive_index is not None
        and raw_digest is not None
        and archive_index.get("stream_sha256") == raw_digest
    )
    archive_round_trip_verified = int(
        archive_index is not None
        and bool(archive_index.get("all_appended_records_round_trip_verified"))
    )
    # Persist the finalized index and the independently accumulated count/hash
    # even for exceptions after a partial rollout.  These audit gates describe
    # storage integrity only; this path remains an unconditional physics fail.
    METADATA["raw_contact_archive"] = archive_index
    if not trace_path.exists():
        write_json(
            trace_path,
            {
                "metadata": METADATA,
                "error": f"{type(exc).__name__}: {exc}",
                "archive_error": archive_error,
                "traceback": traceback.format_exc(),
                "derived": {
                    "raw_contact_record_count": RAW_COUNT,
                    "raw_contact_stream_sha256": raw_digest,
                    "raw_contact_archive": archive_index,
                    "private_raw_archive_count_matches": archive_count_matches,
                    "private_raw_archive_digest_matches": archive_digest_matches,
                    "private_raw_archive_round_trip_verified": archive_round_trip_verified,
                },
                "samples": SAMPLES,
            },
        )
    if not output_path.exists():
        write_json(
            output_path,
            {
                "qualification": False,
                "native_trial_id": trial_id,
                "source_sha256": source_sha256,
                "checks": [
                    check("native_program_completed", 0, "==", 1, "boolean"),
                    check("fresh_native_sample_count", len(SAMPLES), ">=", 1, "physics steps"),
                    check(
                        "private_raw_archive_count_matches",
                        archive_count_matches,
                        "==",
                        1,
                        "boolean",
                    ),
                    check(
                        "private_raw_archive_digest_matches",
                        archive_digest_matches,
                        "==",
                        1,
                        "boolean",
                    ),
                    check(
                        "private_raw_archive_round_trip_verified",
                        archive_round_trip_verified,
                        "==",
                        1,
                        "boolean",
                    ),
                ],
                "overall_pass": False,
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": [str(trace_path.relative_to(ROOT))],
            },
        )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        failure_report(error)
        exit_code = 2
    finally:
        try:
            finish_archive()
        finally:
            app.close()
    raise SystemExit(exit_code)
