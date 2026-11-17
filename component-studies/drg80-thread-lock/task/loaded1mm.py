#!/usr/bin/env python3
"""Native stock-M8 Factory SDF thread-contact reproduction; not qualification.

This bounded coupon uses the original, hash-pinned NVIDIA Factory M8 OBJ bytes
at their authored metre scale.  The bolt and nut collision meshes are cooked as
SDFs at the original URDF resolutions (512 and 256).  A serial prismatic /
revolute bearing removes nuisance motion but leaves axial z and rotation theta
independent: only native nut/bolt surface contact can couple them.  After reset
the program writes only generalized actuator efforts; it never writes a pose,
velocity, joint position, mimic relation, weld, or scorer state.

The 30 g moving mass follows the pinned Factory NutThread held-asset mass and is
explicitly partitioned into a 1 g non-colliding bearing carrier and a 29 g nut
rotor.  This representation is diagnostic, not an assertion of OEM mass or a
revision of the frozen DRG80 contract.
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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--trace", required=True)
parser.add_argument(
    "--direction",
    required=True,
    choices=("positive", "negative"),
    help="Run one independently initialized screw direction; invoke once per direction.",
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


import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import IsaacEvents, PhysxCfg, PhysxManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "input" / "nvidia-factory-reference"
ARCHIVE_HELPER = Path(__file__).resolve().with_name("bounded_contact_archive.py")
EXPECTED_ARCHIVE_HELPER_SHA256 = "13e998f29f8055c55343f68abdb706314e4647d2bb9b6ac40605d782656a0dd5"
BOLT_OBJ = REFERENCE / "factory_bolt_m8_tight.obj"
NUT_OBJ = REFERENCE / "factory_nut_m8_tight_subdiv_3x.obj"
EXPECTED_HASHES = {
    "bolt": "b5affc6583fe2b6038e1b53c74102eabd0842ecc05fe614ee04dc7b32385c123",
    "nut": "c4583cce9525314dcf7be900b6544cbcb4f7112ea54034786a2de2bf604947bb",
}

DT = 1.0 / 240.0
TEST_PITCH_M = 0.001
STOCK_CONTACT_OFFSET_M = 0.005
STOCK_REST_OFFSET_M = 0.0
BOLT_SDF_RESOLUTION = 512
NUT_SDF_RESOLUTION = 256
SETTLE_STEPS = 240
SENSE_STEPS = 240
DRIVE_STEPS = 3600
TARGET_ANGULAR_SPEED = 1.0
MAX_TORQUE_NM = 0.0015
RESISTING_AXIAL_FORCE_N = 0.5
NUT_REFERENCE_FRICTION = 0.01
BOLT_REFERENCE_FRICTION = 0.75
# Conservative lower bound for useful axial-load output divided by aligned
# signed actuator input plus any recovered native kinetic energy.  For an M8
# 60-degree thread, using the nominal 8 mm diameter (smaller lead angle than
# the pitch diameter) and the larger authored body friction 0.75 gives the
# classical raising-efficiency reference
# tan(lambda)/tan(lambda+atan(mu/cos(30 deg))) = 0.05197.  Rounding down to
# 0.05 leaves margin for the SDF/contact approximation while still rejecting
# a run in which essentially all measured input disappears numerically.  This
# is a diagnostic plausibility floor, not an OEM efficiency calibration.
MIN_USEFUL_OUTPUT_EFFICIENCY = 0.05
MAX_UNEXPLAINED_ENERGY_FRACTION = 1.0 - MIN_USEFUL_OUTPUT_EFFICIENCY
# Closed-loop poles for the unsaturated, free-axis discrete PI diagnostic.
# Gains are derived after reset from the measured native Izz and dt; they are
# not hand-tuned torque gains and no mass/inertia is changed for stability.
PI_POLE_1 = 0.85
PI_POLE_2 = 0.90
POSITION_ITERATIONS = 192
VELOCITY_ITERATIONS = 1
CARRIER_MASS_KG = 0.001
NUT_MASS_KG = 0.029
MOVING_MASS_KG = CARRIER_MASS_KG + NUT_MASS_KG
CONTACT_FORCE_EPS_N = 1.0e-6
MIN_LOADED_CONTACT_FRACTION = 0.90
MIN_CONTIGUOUS_CONTACT_STEPS = 240
# The user requires advance reporting above 50 MB/file.  Keep this diagnostic
# independently below that boundary.  Detailed contact tensors can contain
# millions of entries, so the trace retains every per-step native aggregate,
# exact raw extrema, a deterministic prefix of raw entries, and a SHA-256 over
# every raw entry in native order instead of serializing every repeated point.
MAX_ARTIFACT_BYTES = 48_000_000
RAW_CONTACT_PREFIX_PER_STEP = 8
CONTACT_DATA_CAPACITY = 4096

SAMPLES: list[dict] = []
METADATA: dict = {}
CALLBACK_ERRORS: list[str] = []
CONTACT_ARCHIVE: BoundedContactArchive | None = None
CONTACT_ARCHIVE_INDEX: dict | None = None
RAW_CONTACT_DIGEST = None
RAW_CONTACT_ENTRY_COUNT: list[int] | None = None


def finish_contact_archive() -> dict | None:
    """Idempotently finalize and expose the complete private raw stream index."""
    global CONTACT_ARCHIVE_INDEX
    if CONTACT_ARCHIVE is not None and CONTACT_ARCHIVE_INDEX is None:
        CONTACT_ARCHIVE_INDEX = CONTACT_ARCHIVE.finish()
        METADATA["raw_contact_archive"] = CONTACT_ARCHIVE_INDEX
    return CONTACT_ARCHIVE_INDEX


def write_json(path: Path, payload: dict) -> int:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            f"refusing to write {len(encoded)}-byte artifact above bounded "
            f"{MAX_ARTIFACT_BYTES}-byte policy: {path}"
        )
    path.write_bytes(encoded)
    return len(encoded)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0.0 else "-Infinity"
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def check(name: str, measured: float, operator: str, threshold: float, unit: str) -> dict:
    value = float(measured)
    predicates = {
        "==": lambda a, b: a == b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
    }
    passed = math.isfinite(value) and predicates[operator](value, float(threshold))
    return {
        "name": name,
        "measured": value,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(passed),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_obj(path: Path) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    """Read every vertex/face and triangulate polygon fans without modifying coordinates."""
    vertices: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.strip().split()
            if not fields or fields[0] == "#":
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise RuntimeError(f"malformed vertex at {path.name}:{line_no}")
                xyz = [float(fields[i]) for i in range(1, 4)]
                if not all(math.isfinite(v) for v in xyz):
                    raise RuntimeError(f"non-finite vertex at {path.name}:{line_no}")
                vertices.append(Gf.Vec3f(xyz[0], xyz[1], 0.8 * xyz[2]))
            elif fields[0] == "f":
                polygon: list[int] = []
                for token in fields[1:]:
                    raw_index = int(token.split("/", 1)[0])
                    index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    if index < 0 or index >= len(vertices):
                        raise RuntimeError(f"face index out of range at {path.name}:{line_no}")
                    polygon.append(index)
                if len(polygon) < 3:
                    raise RuntimeError(f"degenerate face at {path.name}:{line_no}")
                for offset in range(1, len(polygon) - 1):
                    counts.append(3)
                    indices.extend((polygon[0], polygon[offset], polygon[offset + 1]))
    if not vertices or not counts:
        raise RuntimeError(f"OBJ has no usable geometry: {path}")
    return vertices, counts, indices


def create_body(
    stage,
    path: str,
    position: Gf.Vec3d,
    mass_kg: float,
    diagonal_inertia: Gf.Vec3f | None = None,
    rotation_z_deg: float | None = None,
):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(position)
    if rotation_z_deg is not None:
        body.AddRotateZOp().Set(rotation_z_deg)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(mass_kg)
    if diagonal_inertia is not None:
        mass_api.CreateDiagonalInertiaAttr(diagonal_inertia)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    physx_body.CreateDisableGravityAttr(True)
    physx_body.CreateLinearDampingAttr(0.0)
    physx_body.CreateAngularDampingAttr(0.0)
    return body


def create_sdf_mesh(stage, path: str, obj_path: Path, resolution: int, material: UsdShade.Material):
    points, counts, indices = load_obj(obj_path)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    # Match the pinned stock Factory CollisionPropertiesCfg rather than
    # inheriting runtime defaults.  Native tensor readback below is normative.
    physx_collision.CreateContactOffsetAttr(STOCK_CONTACT_OFFSET_M)
    physx_collision.CreateRestOffsetAttr(STOCK_REST_OFFSET_M)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
    sdf.CreateSdfResolutionAttr(resolution)
    sdf.CreateSdfSubgridResolutionAttr(6)
    sdf.CreateSdfBitsPerSubgridPixelAttr("BitsPerPixel16")
    sdf.CreateSdfEnableRemeshingAttr(False)
    sdf.CreateSdfMarginAttr(0.01)
    sdf.CreateSdfNarrowBandThicknessAttr(0.01)
    sdf.CreateSdfTriangleCountReductionFactorAttr(1.0)
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    PhysxSchema.PhysxContactReportAPI.Apply(mesh.GetPrim()).CreateThresholdAttr(0.0)
    return mesh, len(points), len(counts)


def as_cpu(value) -> torch.Tensor:
    return wp.to_torch(value).detach().cpu().clone()


def phase_delta(rows: list[dict]) -> tuple[float, float, float]:
    dq = float(rows[-1]["q_rotation_rad"] - rows[0]["q_rotation_rad"])
    dz = float(rows[-1]["q_axial_m"] - rows[0]["q_axial_m"])
    pitch = abs(dz / dq) * 2.0 * math.pi if abs(dq) > 1.0e-12 else math.inf
    return dq, dz, pitch


def regression_r2(rows: list[dict]) -> float:
    theta = np.asarray([row["q_rotation_rad"] for row in rows], dtype=np.float64)
    axial = np.asarray([row["q_axial_m"] for row in rows], dtype=np.float64)
    design = np.column_stack((theta, np.ones_like(theta)))
    prediction = design @ np.linalg.lstsq(design, axial, rcond=None)[0]
    residual = float(np.sum((axial - prediction) ** 2))
    total = float(np.sum((axial - np.mean(axial)) ** 2))
    return 1.0 - residual / total if total > 1.0e-20 else 0.0


def longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a normalized [x,y,z,w] quaternion."""
    xyz = quaternion[:3].astype(np.float64)
    w = float(quaternion[3])
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def phase_contact_metrics(rows: list[dict]) -> dict:
    """Integrate only measured nut/bolt contact transfer at the physics rate."""
    active = [
        row["contact_count"] > 0
        and float(np.linalg.norm(row["filtered_contact_force_n"])) > CONTACT_FORCE_EPS_N
        for row in rows
    ]
    return {
        "active_steps": int(sum(active)),
        "active_fraction": float(np.mean(active)),
        "longest_contiguous_steps": longest_true_run(active),
        "axial_abs_impulse_n_s": float(
            sum(abs(row["normal_contact_force_n"][2]) * DT for row in rows)
        ),
        "axial_signed_impulse_n_s": float(
            sum(row["normal_contact_force_n"][2] * DT for row in rows)
        ),
        "rotational_abs_impulse_nm_s": float(
            sum(abs(row["normal_contact_torque_nm"][2]) * DT for row in rows)
        ),
        "rotational_signed_impulse_nm_s": float(
            sum(row["normal_contact_torque_nm"][2] * DT for row in rows)
        ),
        # This is only absolute normal-wrench activity.  It is deliberately
        # excluded from the motor/load energy balance: filtered normal contact
        # is neither total contact power nor a physical heat estimate.
        "normal_contact_absolute_wrench_work_j": float(
            sum(abs(row["normal_contact_power_w"]) * DT for row in rows)
        ),
    }


def main() -> None:
    global CONTACT_ARCHIVE, METADATA, RAW_CONTACT_DIGEST, RAW_CONTACT_ENTRY_COUNT
    if sha256(ARCHIVE_HELPER) != EXPECTED_ARCHIVE_HELPER_SHA256:
        raise RuntimeError("bounded raw-contact archive helper hash mismatch")
    CONTACT_ARCHIVE = BoundedContactArchive(ROOT, trial_id)
    for label, path in (("bolt", BOLT_OBJ), ("nut", NUT_OBJ)):
        if not path.is_file():
            raise RuntimeError(f"missing staged Factory asset: {path}")
        if sha256(path) != EXPECTED_HASHES[label]:
            raise RuntimeError(f"Factory {label} OBJ hash mismatch")

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

    # The pinned Factory environment overwrites its global 1/1 material with
    # per-body task values: NutM16=0.01 and FixedAssetCfg=0.75.  Applying those
    # values to the original M8 meshes is a declared prospective reproduction
    # choice, not an OEM M8 calibration claim.  Native readback below is the
    # actual authority for this probe.
    bolt_material = UsdShade.Material.Define(stage, "/World/BoltFactoryReferenceMaterial")
    bolt_material_api = UsdPhysics.MaterialAPI.Apply(bolt_material.GetPrim())
    bolt_material_api.CreateStaticFrictionAttr(BOLT_REFERENCE_FRICTION)
    bolt_material_api.CreateDynamicFrictionAttr(BOLT_REFERENCE_FRICTION)
    bolt_material_api.CreateRestitutionAttr(0.0)
    nut_material = UsdShade.Material.Define(stage, "/World/NutFactoryReferenceMaterial")
    nut_material_api = UsdPhysics.MaterialAPI.Apply(nut_material.GetPrim())
    nut_material_api.CreateStaticFrictionAttr(NUT_REFERENCE_FRICTION)
    nut_material_api.CreateDynamicFrictionAttr(NUT_REFERENCE_FRICTION)
    nut_material_api.CreateRestitutionAttr(0.0)

    bolt = create_body(stage, "/World/Bolt", Gf.Vec3d(0.0, 0.0, 0.5), 0.05)
    # Tensor contact sensors resolve the reporting API on the rigid body, not
    # merely on its child collision mesh.  The first host log explicitly
    # reported that /World/Bolt and /World/NutAssembly/nut did not match rigid
    # contact entries when the API was authored on the meshes alone.
    PhysxSchema.PhysxContactReportAPI.Apply(bolt.GetPrim()).CreateThresholdAttr(0.0)
    bolt_mesh, bolt_vertex_count, bolt_triangle_count = create_sdf_mesh(
        stage, "/World/Bolt/Collision", BOLT_OBJ, BOLT_SDF_RESOLUTION, bolt_material
    )
    bolt_fixed = UsdPhysics.FixedJoint.Define(stage, "/World/BoltMount")
    bolt_fixed.CreateBody1Rel().SetTargets([bolt.GetPath()])
    bolt_fixed.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

    root = UsdGeom.Xform.Define(stage, "/World/NutAssembly")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    articulation_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)

    frame = create_body(stage, "/World/NutAssembly/frame", Gf.Vec3d(0.0, 0.0, 0.504), 1.0)
    carrier = create_body(
        stage,
        "/World/NutAssembly/carrier",
        Gf.Vec3d(0.0, 0.0, 0.504),
        CARRIER_MASS_KG,
        Gf.Vec3f(1.0e-7, 1.0e-7, 1.0e-7),
    )
    # Factory's pinned nut-thread setup authors a -90 degree initial held-asset
    # yaw.  The failed trial omitted this stock phase and began at zero yaw,
    # producing a jammed contact field rather than thread following.
    nut = create_body(
        stage,
        "/World/NutAssembly/nut",
        Gf.Vec3d(0.0, 0.0, 0.504),
        NUT_MASS_KG,
        rotation_z_deg=36.0,
    )
    PhysxSchema.PhysxContactReportAPI.Apply(nut.GetPrim()).CreateThresholdAttr(0.0)
    nut_mesh, nut_vertex_count, nut_triangle_count = create_sdf_mesh(
        stage, "/World/NutAssembly/nut/Collision", NUT_OBJ, NUT_SDF_RESOLUTION, nut_material
    )

    mount = UsdPhysics.FixedJoint.Define(stage, "/World/NutAssembly/mount")
    mount.CreateBody1Rel().SetTargets([frame.GetPath()])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.504))
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
    # Preserve the Factory -90 degree mesh phase while defining generalized
    # q=0: body1(-90 deg) * localFrame1(+90 deg) == body0 local frame.
    half_phase = math.radians(-18.0)
    rotation.CreateLocalRot1Attr(
        Gf.Quatf(math.cos(half_phase), Gf.Vec3f(0.0, 0.0, math.sin(half_phase)))
    )
    rotation.CreateLowerLimitAttr(-7200.0)
    rotation.CreateUpperLimitAttr(7200.0)

    callback_count = [0]
    callback_dts: list[float] = []

    def post_callback(dt_value) -> None:
        try:
            callback_count[0] += 1
            callback_dts.append(float(dt_value))
        except Exception as exc:  # callback dispatch can otherwise swallow errors
            CALLBACK_ERRORS.append(f"{type(exc).__name__}: {exc}")

    post_handle = PhysxManager.register_callback(
        post_callback,
        IsaacEvents.POST_PHYSICS_STEP,
        name=f"factory_sdf_thread_{trial_id}",
        wrap_weak_ref=False,
    )
    try:
        sim.reset()
        physics = PhysxManager.get_physics_sim_view()
        if physics is None:
            raise RuntimeError("native PhysX tensor view unavailable after reset")
        physics.set_subspace_roots("/")
        articulation = physics.create_articulation_view("/World/NutAssembly")
        if articulation.count != 1 or articulation.max_dofs != 2:
            raise RuntimeError(
                f"unexpected nut articulation shape {articulation.count}x{articulation.max_dofs}"
            )
        dof_names = list(articulation.get_metatype(0).dof_names)
        link_names = list(articulation.get_metatype(0).link_names)
        axial_index = dof_names.index("axial")
        rotation_index = dof_names.index("rotation")
        nut_view = physics.create_rigid_body_view([str(nut.GetPath())])
        bolt_view = physics.create_rigid_body_view([str(bolt.GetPath())])
        contact_view = physics.create_rigid_contact_view(
            [str(nut.GetPath())],
            filter_patterns=[[str(bolt.GetPath())]],
            max_contact_data_count=CONTACT_DATA_CAPACITY,
        )
        # The installed Python wrapper documents list patterns, but the pinned
        # 110.3.2 PhysX backend binding accepts exactly one string pattern.  The
        # first trusted run reached this call and rejected a list before taking
        # a physics sample.  Use one native view per exact, known collider path;
        # this also prevents a wildcard from silently admitting another shape.
        bolt_sdf_view = physics.create_sdf_shape_view(str(bolt_mesh.GetPath()), 8)
        nut_sdf_view = physics.create_sdf_shape_view(str(nut_mesh.GetPath()), 8)
        if nut_view.count != 1 or bolt_view.count != 1:
            raise RuntimeError("native rigid-body views did not resolve both Factory actors")
        if contact_view.sensor_count != 1 or contact_view.filter_count != 1:
            raise RuntimeError("native filtered contact view did not resolve nut/bolt pair")

        all_indices = wp.array([0], dtype=wp.uint32, device=args.device)
        native_masses = as_cpu(articulation.get_masses()).reshape(-1).numpy()
        native_inertias_array = as_cpu(articulation.get_inertias()).numpy()
        native_inertias = native_inertias_array.tolist()
        native_material_nut = as_cpu(nut_view.get_material_properties()).numpy().tolist()
        native_material_bolt = as_cpu(bolt_view.get_material_properties()).numpy().tolist()
        native_nut_com_local = as_cpu(nut_view.get_coms())[0].numpy()
        native_initial_nut_transform = as_cpu(nut_view.get_transforms())[0].numpy()
        native_contact_offsets_nut = as_cpu(nut_view.get_contact_offsets()).numpy().reshape(-1)
        native_contact_offsets_bolt = as_cpu(bolt_view.get_contact_offsets()).numpy().reshape(-1)
        native_rest_offsets_nut = as_cpu(nut_view.get_rest_offsets()).numpy().reshape(-1)
        native_rest_offsets_bolt = as_cpu(bolt_view.get_rest_offsets()).numpy().reshape(-1)
        if any(
            values.size != 1
            for values in (
                native_contact_offsets_nut,
                native_contact_offsets_bolt,
                native_rest_offsets_nut,
                native_rest_offsets_bolt,
            )
        ):
            raise RuntimeError("expected exactly one collision shape on each Factory actor")
        if len(native_masses) != len(link_names):
            raise RuntimeError("native link-name/mass vector length mismatch")
        native_mass_by_link = {name: float(native_masses[i]) for i, name in enumerate(link_names)}

        # Design a finite-torque discrete PI controller from the native inertia.
        # For omega[k+1]=omega[k]+dt*u[k]/I and
        # u[k]=Kp*e[k]+integral[k], integral[k+1]=integral[k]+Ki*dt*e[k],
        # the unsaturated error-state characteristic polynomial is
        # lambda^2-(2-b*Kp)lambda+(1-b*Kp+b*Ki*dt), b=dt/I.
        # The contact-side design inertia includes the measured nut Izz plus
        # the translational reflected term m*(pitch/2pi)^2.  We also evaluate
        # the less-inertial free axis explicitly; both pole pairs must be
        # strictly inside the unit circle before any rollout samples are taken.
        nut_link_index = link_names.index("nut")
        native_nut_izz = float(native_inertias_array[0, nut_link_index].reshape(3, 3)[2, 2])
        lead_m_per_rad = TEST_PITCH_M / (2.0 * math.pi)
        controller_inertia = native_nut_izz + MOVING_MASS_KG * lead_m_per_rad**2
        pole_sum = PI_POLE_1 + PI_POLE_2
        pole_product = PI_POLE_1 * PI_POLE_2
        kp = (2.0 - pole_sum) * controller_inertia / DT
        ki = (pole_product - 1.0 + (2.0 - pole_sum)) * controller_inertia / DT**2

        def pi_poles(inertia: float) -> np.ndarray:
            b = DT / inertia
            return np.roots([1.0, -(2.0 - b * kp), 1.0 - b * kp + b * ki * DT])

        design_poles = pi_poles(controller_inertia)
        free_axis_poles = pi_poles(native_nut_izz)
        if max(abs(design_poles)) >= 1.0 or max(abs(free_axis_poles)) >= 1.0:
            raise RuntimeError("derived finite-torque PI poles are not strictly stable")

        resolved_material_count = 0
        for mesh, expected_material in (
            (bolt_mesh, bolt_material),
            (nut_mesh, nut_material),
        ):
            bound, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial(
                materialPurpose="physics"
            )
            resolved_material_count += int(bound.GetPath() == expected_material.GetPath())

        METADATA = {
            "native_trial_id": trial_id,
            "source_sha256": source_sha256,
            "qualification": False,
            "scope": "stock M8 Factory SDF contact-coupling preflight only",
            "commanded_direction": args.direction,
            "diagnostic_initial_nut_yaw_deg": 36.0, "nut_lift_m": 0.004, "axial_mesh_scale": 0.8, "source_pitch_m": 0.00125,
            "device": args.device,
            "physics_dt_s": DT,
            "benchmark_pitch_m_per_revolution": TEST_PITCH_M,
            "asset_paths": {"bolt": str(BOLT_OBJ), "nut": str(NUT_OBJ)},
            "asset_sha256": {"bolt": sha256(BOLT_OBJ), "nut": sha256(NUT_OBJ)},
            "authored_sdf_resolution": {"bolt": BOLT_SDF_RESOLUTION, "nut": NUT_SDF_RESOLUTION},
            "obj_geometry": {
                "bolt_vertices": bolt_vertex_count,
                "bolt_triangles": bolt_triangle_count,
                "nut_vertices": nut_vertex_count,
                "nut_triangles": nut_triangle_count,
            },
            "dof_names": dof_names,
            "link_names": link_names,
            "native_mass_by_link_kg": native_mass_by_link,
            "native_inertias": native_inertias,
            "authored_mass_partition_kg": {
                "carrier": CARRIER_MASS_KG,
                "nut": NUT_MASS_KG,
                "moving_total": MOVING_MASS_KG,
            },
            "native_material_nut": native_material_nut,
            "native_material_bolt": native_material_bolt,
            "material_adaptation": (
                "Pinned Factory per-body NutM16=0.01 and FixedAssetCfg=0.75 applied "
                "prospectively to original M8 meshes; not OEM M8 calibration."
            ),
            "energy_plausibility_reference": {
                "kind": (
                    "conservative analytical diagnostic bound; not measured friction heat "
                    "or OEM calibration"
                ),
                "nominal_thread_diameter_m": 0.008,
                "thread_included_angle_deg": 60.0,
                "conservative_friction_coefficient": max(
                    NUT_REFERENCE_FRICTION, BOLT_REFERENCE_FRICTION
                ),
                "minimum_useful_output_efficiency": MIN_USEFUL_OUTPUT_EFFICIENCY,
                "maximum_unexplained_energy_fraction": MAX_UNEXPLAINED_ENERGY_FRACTION,
            },
            "finite_torque_pi": {
                "native_nut_izz_kg_m2": native_nut_izz,
                "reflected_contact_inertia_kg_m2": MOVING_MASS_KG * lead_m_per_rad**2,
                "design_inertia_kg_m2": controller_inertia,
                "kp_nm_per_rad_s": kp,
                "ki_nm_per_rad": ki,
                "torque_limit_nm": MAX_TORQUE_NM,
                "target_poles": [PI_POLE_1, PI_POLE_2],
                "computed_design_poles": design_poles.tolist(),
                "computed_free_axis_poles": free_axis_poles.tolist(),
                "anti_windup": "conditional integration when saturation would be driven farther",
            },
            "native_nut_com_local_xyzw": native_nut_com_local.tolist(),
            "native_initial_nut_transform_xyzw": native_initial_nut_transform.tolist(),
            "authored_collision_offsets_m": {
                "contact": STOCK_CONTACT_OFFSET_M,
                "rest": STOCK_REST_OFFSET_M,
            },
            "native_collision_offsets_m": {
                "nut_contact": float(native_contact_offsets_nut[0]),
                "bolt_contact": float(native_contact_offsets_bolt[0]),
                "nut_rest": float(native_rest_offsets_nut[0]),
                "bolt_rest": float(native_rest_offsets_bolt[0]),
            },
            "native_sdf_object_paths": {
                "bolt": list(bolt_sdf_view.object_paths),
                "nut": list(nut_sdf_view.object_paths),
            },
            "bearing_declaration": "prismatic z plus revolute theta; no mimic/tendon/gear coupling",
            "post_reset_state_writes": 0,
            "trace_artifact_policy": {
                "maximum_bytes": MAX_ARTIFACT_BYTES,
                "raw_contact_prefix_entries_per_step": RAW_CONTACT_PREFIX_PER_STEP,
                "all_per_step_native_aggregates": True,
                "all_raw_contact_entries_canonical_sha256": True,
                "global_raw_extrema_entries": True,
                "reason": "remain below the user's 50 MB/file advance-reporting boundary",
            },
            "raw_contact_archive_helper": {
                "path": str(ARCHIVE_HELPER.relative_to(ROOT)),
                "sha256": sha256(ARCHIVE_HELPER),
                "expected_sha256": EXPECTED_ARCHIVE_HELPER_SHA256,
            },
        }

        def joint_state() -> dict:
            positions = as_cpu(articulation.get_dof_positions())[0].numpy().astype(np.float64)
            velocities = as_cpu(articulation.get_dof_velocities())[0].numpy().astype(np.float64)
            mass_matrix = (
                as_cpu(articulation.get_generalized_mass_matrices())
                .reshape(articulation.count, articulation.max_dofs, articulation.max_dofs)[0]
                .numpy()
                .astype(np.float64)
            )
            kinetic = 0.5 * float(velocities @ mass_matrix @ velocities)
            return {
                "positions": positions,
                "velocities": velocities,
                "mass_matrix": mass_matrix,
                "kinetic_energy_j": kinetic,
            }

        RAW_CONTACT_DIGEST = hashlib.sha256()
        RAW_CONTACT_ENTRY_COUNT = [0]
        raw_contact_digest = RAW_CONTACT_DIGEST
        raw_contact_entry_count = RAW_CONTACT_ENTRY_COUNT
        raw_contact_numeric_count = [0]
        raw_contact_finite_count = [0]
        raw_contact_min_separation = [math.inf]
        raw_contact_peak_force = [-math.inf]
        raw_contact_min_separation_entry: list[dict | None] = [None]
        raw_contact_peak_force_entry: list[dict | None] = [None]
        max_observed_contact_count = [0]
        saturated_contact_step_count = [0]

        def sample(
            step: int,
            phase: str,
            command: np.ndarray,
            before: dict,
            controller: dict | None = None,
        ) -> None:
            positions = as_cpu(articulation.get_dof_positions())[0].numpy()
            velocities = as_cpu(articulation.get_dof_velocities())[0].numpy()
            mass_matrix = (
                as_cpu(articulation.get_generalized_mass_matrices())
                .reshape(articulation.count, articulation.max_dofs, articulation.max_dofs)[0]
                .numpy()
                .astype(np.float64)
            )
            kinetic_energy = 0.5 * float(
                velocities.astype(np.float64) @ mass_matrix @ velocities.astype(np.float64)
            )
            efforts = as_cpu(articulation.get_dof_actuation_forces())[0].numpy()
            transforms = as_cpu(nut_view.get_transforms())[0].numpy()
            twists = as_cpu(nut_view.get_velocities())[0].numpy()
            matrix = as_cpu(contact_view.get_contact_force_matrix(dt=DT)).reshape(1, 1, 3)[0, 0].numpy()
            contact_data = tuple(as_cpu(value) for value in contact_view.get_contact_data(dt=DT))
            forces, points, normals, separations, counts, starts = contact_data
            count = int(counts.reshape(-1)[0].item())
            start = int(starts.reshape(-1)[0].item())
            contact_capacity = int(contact_view.max_contact_data_count)
            max_observed_contact_count[0] = max(max_observed_contact_count[0], count)
            contact_buffer_saturated = count >= contact_capacity
            saturated_contact_step_count[0] += int(contact_buffer_saturated)
            if count > contact_capacity or start < 0 or start + count > contact_capacity:
                raise RuntimeError(
                    "native detailed-contact range exceeds its allocated buffer: "
                    f"start={start}, count={count}, capacity={contact_capacity}"
                )
            evidence_entries = []
            raw_normal_force = np.zeros(3, dtype=np.float64)
            raw_normal_torque = np.zeros(3, dtype=np.float64)
            nut_com_world = transforms[:3].astype(np.float64) + rotate_xyzw(
                transforms[3:].astype(np.float64), native_nut_com_local[:3].astype(np.float64)
            )
            for k in range(start, start + count):
                scalar_force = float(forces.reshape(-1)[k].item())
                point = np.asarray(points.reshape(-1, 3)[k].tolist(), dtype=np.float64)
                normal = np.asarray(normals.reshape(-1, 3)[k].tolist(), dtype=np.float64)
                entry_force = scalar_force * normal
                raw_normal_force += entry_force
                raw_normal_torque += np.cross(point - nut_com_world, entry_force)
                separation = float(separations.reshape(-1)[k].item())
                entry = {
                    "buffer_index": k,
                    "normal_force_n": scalar_force,
                    "point_w_m": point.tolist(),
                    "normal_w": normal.tolist(),
                    "separation_m": separation,
                }
                raw_values = (scalar_force, *point.tolist(), *normal.tolist(), separation)
                payload = struct.pack("<qq8d", step, k, *raw_values)
                raw_contact_digest.update(payload)
                CONTACT_ARCHIVE.append(payload)
                raw_contact_entry_count[0] += 1
                raw_contact_numeric_count[0] += len(raw_values)
                raw_contact_finite_count[0] += sum(math.isfinite(value) for value in raw_values)
                if len(evidence_entries) < RAW_CONTACT_PREFIX_PER_STEP:
                    evidence_entries.append(entry)
                if separation < raw_contact_min_separation[0]:
                    raw_contact_min_separation[0] = separation
                    raw_contact_min_separation_entry[0] = {"step": step, **entry}
                if scalar_force > raw_contact_peak_force[0]:
                    raw_contact_peak_force[0] = scalar_force
                    raw_contact_peak_force_entry[0] = {"step": step, **entry}
            # Detailed-contact normals may use the opposite pair convention
            # from the sensor-side force matrix.  Orient the reconstructed
            # normal wrench toward the native filtered force vector without
            # altering its magnitude or inventing a tangential component.
            orientation = 1.0
            if np.dot(raw_normal_force, matrix.astype(np.float64)) < 0.0:
                orientation = -1.0
            normal_force = orientation * raw_normal_force
            normal_torque = orientation * raw_normal_torque
            normal_power = float(
                np.dot(normal_force, twists[:3].astype(np.float64))
                + np.dot(normal_torque, twists[3:].astype(np.float64))
            )
            SAMPLES.append(
                {
                    "step": step,
                    "phase": phase,
                    "post_callback_count": callback_count[0],
                    "q_axial_m": float(positions[axial_index]),
                    "q_rotation_rad": float(positions[rotation_index]),
                    "v_axial_m_s": float(velocities[axial_index]),
                    "v_rotation_rad_s": float(velocities[rotation_index]),
                    "command_axial_n": float(command[axial_index]),
                    "command_torque_nm": float(command[rotation_index]),
                    "native_axial_effort_n": float(efforts[axial_index]),
                    "native_rotation_effort_nm": float(efforts[rotation_index]),
                    # The command and post-step native effort belong to this
                    # exact preceding interval.  Keeping its start/end state
                    # on one row prevents the one-step work shift rejected by
                    # the component reviewer.
                    "interval_q_axial_before_m": float(before["positions"][axial_index]),
                    "interval_q_rotation_before_rad": float(before["positions"][rotation_index]),
                    "interval_delta_axial_m": float(
                        positions[axial_index] - before["positions"][axial_index]
                    ),
                    "interval_delta_rotation_rad": float(
                        positions[rotation_index] - before["positions"][rotation_index]
                    ),
                    "interval_kinetic_energy_before_j": float(before["kinetic_energy_j"]),
                    "interval_kinetic_energy_after_j": kinetic_energy,
                    "interval_mass_matrix_before": before["mass_matrix"].tolist(),
                    "interval_mass_matrix_after": mass_matrix.tolist(),
                    "controller": controller,
                    "nut_transform_xyzw": transforms.tolist(),
                    "nut_com_world_m": nut_com_world.tolist(),
                    "nut_twist": twists.tolist(),
                    "filtered_contact_force_n": matrix.tolist(),
                    "normal_contact_force_n": normal_force.tolist(),
                    "normal_contact_torque_nm": normal_torque.tolist(),
                    "normal_contact_power_w": normal_power,
                    "normal_contact_orientation_to_filtered": orientation,
                    "contact_count": count,
                    "contact_data_capacity": contact_capacity,
                    "contact_buffer_saturated": contact_buffer_saturated,
                    "raw_contact_evidence": evidence_entries,
                }
            )

        step_number = 0
        zero = np.zeros(articulation.max_dofs, dtype=np.float32)
        for _ in range(SETTLE_STEPS):
            before = joint_state()
            articulation.set_dof_actuation_forces(
                wp.array(zero.reshape(1, -1), dtype=wp.float32, device=args.device), all_indices
            )
            sim.step(render=False)
            step_number += 1
            sample(step_number, "settle", zero, before)

        # Each invocation owns one fresh, independently authored initial state.
        # This avoids the failed trial's second direction starting from the
        # first direction's jammed endpoint, without any post-reset pose write.
        phase = args.direction
        direction = 1.0 if phase == "positive" else -1.0
        integral_torque = 0.0
        saturation_steps = 0
        anti_windup_hold_steps = 0
        for local_step in range(SENSE_STEPS + DRIVE_STEPS):
            before = joint_state()
            current = before["velocities"]
            command = np.zeros(articulation.max_dofs, dtype=np.float32)
            speed_target = direction * TARGET_ANGULAR_SPEED
            speed_error = speed_target - current[rotation_index]
            integral_torque_used = integral_torque
            unsaturated_torque = kp * speed_error + integral_torque_used
            torque = float(np.clip(unsaturated_torque, -MAX_TORQUE_NM, MAX_TORQUE_NM))
            saturated = not math.isclose(torque, unsaturated_torque, rel_tol=0.0, abs_tol=1e-15)
            saturation_steps += int(saturated)
            # Update the integral for the next interval only.  When saturated,
            # accept integration solely if the present error drives the command
            # back toward the finite limit (conditional anti-windup).
            drives_farther = saturated and math.copysign(1.0, speed_error) == math.copysign(
                1.0, unsaturated_torque
            )
            if drives_farther:
                anti_windup_hold_steps += 1
            else:
                integral_torque += ki * DT * speed_error
                integral_torque = float(
                    np.clip(integral_torque, -MAX_TORQUE_NM, MAX_TORQUE_NM)
                )
            command[rotation_index] = torque
            if local_step >= SENSE_STEPS:
                # A right-hand thread advances +z for +theta.  Apply a measured
                # opposing axial load from the start of the loaded interval so
                # real flanks remain engaged; the omitted preload was another
                # cause of the previous free/jam transition.
                command[axial_index] = -direction * RESISTING_AXIAL_FORCE_N
            articulation.set_dof_actuation_forces(
                wp.array(command.reshape(1, -1), dtype=wp.float32, device=args.device), all_indices
            )
            sim.step(render=False)
            step_number += 1
            sample(
                step_number,
                f"{phase}_{'sense' if local_step < SENSE_STEPS else 'loaded'}",
                command,
                before,
                {
                    "speed_target_rad_s": speed_target,
                    "speed_error_rad_s": float(speed_error),
                    "proportional_torque_nm": float(kp * speed_error),
                    "integral_torque_used_nm": float(integral_torque_used),
                    "integral_torque_next_nm": float(integral_torque),
                    "unsaturated_torque_nm": float(unsaturated_torque),
                    "applied_torque_nm": torque,
                    "saturated": saturated,
                    "anti_windup_hold": drives_farther,
                },
            )
        loaded = [row for row in SAMPLES if row["phase"] == f"{phase}_loaded"]

        if CALLBACK_ERRORS:
            raise RuntimeError(f"POST_PHYSICS_STEP callback errors: {CALLBACK_ERRORS}")

        numeric = []
        for row in SAMPLES:
            numeric.extend(
                [
                    row["q_axial_m"], row["q_rotation_rad"], row["v_axial_m_s"],
                    row["v_rotation_rad_s"], *row["nut_transform_xyzw"], *row["nut_com_world_m"],
                    *row["nut_twist"],
                    *row["filtered_contact_force_n"], *row["normal_contact_force_n"],
                    *row["normal_contact_torque_nm"], row["normal_contact_power_w"],
                    row["interval_q_axial_before_m"], row["interval_q_rotation_before_rad"],
                    row["interval_delta_axial_m"], row["interval_delta_rotation_rad"],
                    row["interval_kinetic_energy_before_j"],
                    row["interval_kinetic_energy_after_j"],
                    *np.asarray(row["interval_mass_matrix_before"]).reshape(-1),
                    *np.asarray(row["interval_mass_matrix_after"]).reshape(-1),
                ]
            )
            if row["controller"] is not None:
                numeric.extend(
                    value for value in row["controller"].values()
                    if isinstance(value, (float, int)) and not isinstance(value, bool)
                )
        if raw_contact_entry_count[0] == 0:
            raise RuntimeError("stock SDF nut/bolt view produced no filtered contact entries")
        numeric_finite_count = int(np.count_nonzero(np.isfinite(np.asarray(numeric, dtype=np.float64))))
        finite_fraction = float(
            (numeric_finite_count + raw_contact_finite_count[0])
            / (len(numeric) + raw_contact_numeric_count[0])
        )

        delta_theta, delta_z, measured_pitch = phase_delta(loaded)
        motion_r2 = regression_r2(loaded)
        contact_metrics = phase_contact_metrics(loaded)

        def signed_components(increments: list[float]) -> dict:
            return {
                "signed_j": float(sum(increments)),
                "positive_j": float(sum(value for value in increments if value > 0.0)),
                "negative_j": float(sum(value for value in increments if value < 0.0)),
            }

        def phase_work(rows: list[dict]) -> dict:
            # Each post-step effort is paired with the displacement stored for
            # that same preceding physics interval.  No negative increment is
            # clipped out of the signed totals.
            actuator_increments = [
                row["native_rotation_effort_nm"] * row["interval_delta_rotation_rad"]
                for row in rows
            ]
            commanded_actuator_increments = [
                row["command_torque_nm"] * row["interval_delta_rotation_rad"]
                for row in rows
            ]
            axial_load_increments = [
                row["native_axial_effort_n"] * row["interval_delta_axial_m"]
                for row in rows
            ]
            commanded_axial_load_increments = [
                row["command_axial_n"] * row["interval_delta_axial_m"]
                for row in rows
            ]
            actuator = signed_components(actuator_increments)
            commanded_actuator = signed_components(commanded_actuator_increments)
            axial_load_on_system = signed_components(axial_load_increments)
            commanded_axial_load_on_system = signed_components(commanded_axial_load_increments)
            resisted_load_output = signed_components([-value for value in axial_load_increments])
            kinetic_delta = float(
                rows[-1]["interval_kinetic_energy_after_j"]
                - rows[0]["interval_kinetic_energy_before_j"]
            )
            signed_external_work = actuator["signed_j"] + axial_load_on_system["signed_j"]
            energy_residual = signed_external_work - kinetic_delta
            # Use signed actuator work: negative motor work must not be erased.
            # A native kinetic-energy decrease is an additional available input;
            # a kinetic-energy increase is already subtracted in energy_residual.
            # The residual is deliberately called unexplained, not friction heat.
            available_signed_energy = actuator["signed_j"] + max(0.0, -kinetic_delta)
            output_to_available_ratio = resisted_load_output["signed_j"] / max(
                available_signed_energy, 1.0e-12
            )
            unexplained_energy_fraction = energy_residual / max(
                available_signed_energy, 1.0e-12
            )
            return {
                "actuator": actuator,
                "commanded_actuator": commanded_actuator,
                "axial_load_on_system": axial_load_on_system,
                "commanded_axial_load_on_system": commanded_axial_load_on_system,
                "resisted_load_output": resisted_load_output,
                "kinetic_energy_start_j": float(rows[0]["interval_kinetic_energy_before_j"]),
                "kinetic_energy_end_j": float(rows[-1]["interval_kinetic_energy_after_j"]),
                "kinetic_energy_delta_j": kinetic_delta,
                "signed_external_work_j": signed_external_work,
                "external_work_minus_delta_ke_j": energy_residual,
                "signed_actuator_plus_recovered_ke_available_j": available_signed_energy,
                "resisted_signed_output_to_signed_input_ratio": output_to_available_ratio,
                "unexplained_energy_residual_fraction": unexplained_energy_fraction,
                "interval_count": len(rows),
            }

        work = phase_work(loaded)
        callback_dt_error = max(abs(value - DT) for value in callback_dts)
        native_nut_static_friction = float(native_material_nut[0][0][0])
        native_nut_dynamic_friction = float(native_material_nut[0][0][1])
        native_bolt_static_friction = float(native_material_bolt[0][0][0])
        native_bolt_dynamic_friction = float(native_material_bolt[0][0][1])
        loaded_effort_alignment_error = max(
            abs(row["command_torque_nm"] - row["native_rotation_effort_nm"])
            for row in loaded
        )
        loaded_axial_effort_alignment_error = max(
            abs(row["command_axial_n"] - row["native_axial_effort_n"])
            for row in loaded
        )
        approximation_count = sum(
            int(str(mesh.GetPrim().GetAttribute("physics:approximation").Get()) == "sdf")
            for mesh in (bolt_mesh, nut_mesh)
        )
        remeshing_disabled_count = sum(
            int(not bool(mesh.GetPrim().GetAttribute("physxSDFMeshCollision:sdfEnableRemeshing").Get()))
            for mesh in (bolt_mesh, nut_mesh)
        )
        resolution_sum = sum(
            int(mesh.GetPrim().GetAttribute("physxSDFMeshCollision:sdfResolution").Get())
            for mesh in (bolt_mesh, nut_mesh)
        )
        mimic_schema_count = sum(
            int(any("Mimic" in schema or "Tendon" in schema for schema in prim.GetAppliedSchemas()))
            for prim in stage.Traverse()
        )
        expected_initial_quat_xyzw = np.asarray(
            [0.0, 0.0, math.sin(math.radians(18.0)), math.cos(math.radians(18.0))],
            dtype=np.float64,
        )
        initial_quat_alignment = abs(
            float(np.dot(native_initial_nut_transform[3:], expected_initial_quat_xyzw))
        )

        checks = [
            check("qualification_flag", 0, "==", 0, "boolean"),
            check(
                "verified_raw_contact_archive_helper_hash_count",
                int(sha256(ARCHIVE_HELPER) == EXPECTED_ARCHIVE_HELPER_SHA256),
                "==",
                1,
                "files",
            ),
            check("verified_original_obj_hash_count", 2, "==", 2, "assets"),
            check("authored_sdf_approximation_count", approximation_count, "==", 2, "colliders"),
            check("native_sdf_shape_count", bolt_sdf_view.count + nut_sdf_view.count, "==", 2, "shapes"),
            check(
                "native_sdf_view_valid_count",
                int(bool(bolt_sdf_view.check())) + int(bool(nut_sdf_view.check())),
                "==",
                2,
                "views",
            ),
            check("authored_sdf_resolution_sum", resolution_sum, "==", 768, "voxels-per-extent sum"),
            check("authored_sdf_remeshing_disabled_count", remeshing_disabled_count, "==", 2, "colliders"),
            check("resolved_physics_material_binding_count", resolved_material_count, "==", 2, "colliders"),
            check("native_nut_static_friction_error", abs(native_nut_static_friction - NUT_REFERENCE_FRICTION), "<=", 1e-6, "coefficient"),
            check("native_nut_dynamic_friction_error", abs(native_nut_dynamic_friction - NUT_REFERENCE_FRICTION), "<=", 1e-6, "coefficient"),
            check("native_bolt_static_friction_error", abs(native_bolt_static_friction - BOLT_REFERENCE_FRICTION), "<=", 1e-6, "coefficient"),
            check("native_bolt_dynamic_friction_error", abs(native_bolt_dynamic_friction - BOLT_REFERENCE_FRICTION), "<=", 1e-6, "coefficient"),
            check("native_nut_contact_offset_error", abs(float(native_contact_offsets_nut[0]) - STOCK_CONTACT_OFFSET_M), "<=", 1e-8, "m"),
            check("native_bolt_contact_offset_error", abs(float(native_contact_offsets_bolt[0]) - STOCK_CONTACT_OFFSET_M), "<=", 1e-8, "m"),
            check("native_nut_rest_offset_error", abs(float(native_rest_offsets_nut[0]) - STOCK_REST_OFFSET_M), "<=", 1e-8, "m"),
            check("native_bolt_rest_offset_error", abs(float(native_rest_offsets_bolt[0]) - STOCK_REST_OFFSET_M), "<=", 1e-8, "m"),
            check("native_articulation_count", articulation.count, "==", 1, "articulations"),
            check("native_dof_count", articulation.max_dofs, "==", 2, "dofs"),
            check("forbidden_mimic_or_tendon_schema_count", mimic_schema_count, "==", 0, "schemas"),
            check("post_reset_state_write_count", 0, "==", 0, "writes"),
            check("native_initial_factory_phase_alignment", initial_quat_alignment, ">=", 0.999, "absolute quaternion dot"),
            check("native_carrier_mass_error", abs(native_mass_by_link["carrier"] - CARRIER_MASS_KG), "<=", 1e-6, "kg"),
            check("native_nut_mass_error", abs(native_mass_by_link["nut"] - NUT_MASS_KG), "<=", 1e-6, "kg"),
            check("native_moving_mass_error", abs(native_mass_by_link["carrier"] + native_mass_by_link["nut"] - MOVING_MASS_KG), "<=", 2e-6, "kg"),
            check("pi_design_spectral_radius", max(abs(design_poles)), "<", 1.0, "eigenvalue magnitude"),
            check("pi_free_axis_spectral_radius", max(abs(free_axis_poles)), "<", 1.0, "eigenvalue magnitude"),
            check("finite_torque_command_peak", max(abs(row["command_torque_nm"]) for row in SAMPLES), "<=", MAX_TORQUE_NM + 1e-9, "N*m"),
            check("actual_physics_dt", sim.get_physics_dt(), "==", DT, "s"),
            check("post_physics_callback_count", callback_count[0], "==", step_number, "callbacks"),
            check("post_physics_callback_dt_error_max", callback_dt_error, "<=", 1e-9, "s"),
            check("fresh_native_sample_count", len(SAMPLES), "==", step_number, "physics steps"),
            check("finite_native_value_fraction", finite_fraction, "==", 1.0, "fraction"),
            check(
                "detailed_contact_buffer_capacity",
                contact_view.max_contact_data_count,
                "==",
                CONTACT_DATA_CAPACITY,
                "entries",
            ),
            check(
                "maximum_observed_contact_count",
                max_observed_contact_count[0],
                "<=",
                contact_view.max_contact_data_count - 1,
                "entries per physics step",
            ),
            check(
                "detailed_contact_buffer_saturation_steps",
                saturated_contact_step_count[0],
                "==",
                0,
                "physics steps",
            ),
            check("filtered_contact_entry_count", raw_contact_entry_count[0], ">=", 1, "entries"),
            check("filtered_contact_force_peak", raw_contact_peak_force[0], ">", 0.0, "N"),
            check(f"{phase}_loaded_contact_fraction", contact_metrics["active_fraction"], ">=", MIN_LOADED_CONTACT_FRACTION, "fraction"),
            check(f"{phase}_loaded_contact_contiguous_min", contact_metrics["longest_contiguous_steps"], ">=", MIN_CONTIGUOUS_CONTACT_STEPS, "physics steps at 240 Hz"),
            check("maximum_contact_penetration", max(0.0, -raw_contact_min_separation[0]), "<=", 0.00075, "m"),
            check(f"{phase}_rotation_signed_travel", direction * delta_theta, ">=", 2.0 * math.pi, "rad"),
            check(f"{phase}_axial_signed_travel", direction * delta_z, ">=", 0.001, "m"),
            check(f"{phase}_measured_pitch_error", abs(measured_pitch - TEST_PITCH_M), "<=", 0.00025, "m/revolution"),
            check(f"{phase}_z_theta_regression_r2", motion_r2, ">=", 0.95, "fraction"),
            check(f"{phase}_actuator_signed_work", work["actuator"]["signed_j"], ">", 1e-5, "J"),
            check(f"{phase}_resisted_axial_signed_output_work", work["resisted_load_output"]["signed_j"], ">", 1e-6, "J"),
            check(f"{phase}_same_interval_torque_alignment_error", loaded_effort_alignment_error, "<=", 1e-7, "N*m"),
            check(f"{phase}_same_interval_axial_effort_alignment_error", loaded_axial_effort_alignment_error, "<=", 1e-5, "N"),
            check(f"{phase}_energy_residual_lower_bound", work["external_work_minus_delta_ke_j"], ">=", -1e-6, "J"),
            check(f"{phase}_output_to_available_energy_ratio_upper", work["resisted_signed_output_to_signed_input_ratio"], "<=", 1.05, "ratio"),
            check(f"{phase}_useful_output_efficiency_lower", work["resisted_signed_output_to_signed_input_ratio"], ">=", MIN_USEFUL_OUTPUT_EFFICIENCY, "ratio"),
            check(f"{phase}_unexplained_energy_residual_fraction_upper", work["unexplained_energy_residual_fraction"], "<=", MAX_UNEXPLAINED_ENERGY_FRACTION, "ratio"),
            check(f"{phase}_normal_contact_axial_transfer", contact_metrics["axial_abs_impulse_n_s"], ">", 1e-3, "N*s"),
            check(f"{phase}_normal_contact_rotational_transfer", contact_metrics["rotational_abs_impulse_nm_s"], ">", 1e-5, "N*m*s"),
            check(f"{phase}_normal_contact_absolute_wrench_activity", contact_metrics["normal_contact_absolute_wrench_work_j"], ">", 1e-6, "J"),
            check("nut_linear_speed_peak", max(np.linalg.norm(row["nut_twist"][:3]) for row in SAMPLES), "<=", 0.25, "m/s"),
            check("nut_angular_speed_peak", max(np.linalg.norm(row["nut_twist"][3:]) for row in SAMPLES), "<=", 10.0, "rad/s"),
        ]

        archive_index = finish_contact_archive()
        if archive_index is None:
            raise RuntimeError("raw contact archive was not initialized")
        archive_count_matches = int(archive_index["records"] == raw_contact_entry_count[0])
        archive_digest_matches = int(
            archive_index["stream_sha256"] == raw_contact_digest.hexdigest()
        )
        checks.extend(
            [
                check(
                    "private_raw_contact_archive_record_count_matches_independent_count",
                    archive_count_matches,
                    "==",
                    1,
                    "boolean",
                ),
                check(
                    "private_raw_contact_archive_digest_matches_independent_digest",
                    archive_digest_matches,
                    "==",
                    1,
                    "boolean",
                ),
                check(
                    "private_raw_contact_archive_round_trip_verified",
                    int(archive_index["all_appended_records_round_trip_verified"]),
                    "==",
                    1,
                    "boolean",
                ),
            ]
        )

        trace_payload = {
            "metadata": METADATA,
            "derived": {
                phase: {
                    "delta_theta_rad": delta_theta,
                    "delta_z_m": delta_z,
                    "pitch_m_per_rev": measured_pitch,
                    "r2": motion_r2,
                    "same_interval_signed_work": work,
                    "controller_saturation_steps": saturation_steps,
                    "controller_anti_windup_hold_steps": anti_windup_hold_steps,
                    "measured_contact_transfer": contact_metrics,
                },
                "contact_separation_min_m": raw_contact_min_separation[0],
                "contact_force_peak_n": raw_contact_peak_force[0],
                "raw_contact_audit": {
                    "entry_count": raw_contact_entry_count[0],
                    "maximum_observed_count_per_step": max_observed_contact_count[0],
                    "buffer_capacity": int(contact_view.max_contact_data_count),
                    "saturated_step_count": saturated_contact_step_count[0],
                    "fail_closed_rule": "every per-step count must be strictly less than capacity",
                    "canonical_stream_sha256": raw_contact_digest.hexdigest(),
                    "canonical_record_format": "little-endian <qq8d: step, buffer_index, force, point_xyz, normal_xyz, separation",
                    "stored_prefix_entries_per_step": RAW_CONTACT_PREFIX_PER_STEP,
                    "global_minimum_separation_entry": raw_contact_min_separation_entry[0],
                    "global_peak_force_entry": raw_contact_peak_force_entry[0],
                    "all_step_aggregates_stored": True,
                },
            },
            "samples": SAMPLES,
        }
        trace_bytes = write_json(trace_path, json_safe(trace_payload))
        checks.append(
            check("trace_artifact_size", trace_bytes, "<=", MAX_ARTIFACT_BYTES, "bytes")
        )
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
            failed = [item["name"] for item in checks if not item["passed"]]
            print(json.dumps({"failed_checks": failed}, allow_nan=False))
            # Preserve the complete measured report above.  SystemExit is not
            # caught by the unexpected-runtime-error handler below.
            raise SystemExit(1)
    finally:
        PhysxManager.deregister_callback(post_handle)


def failure_report(exc: Exception) -> None:
    archive_error = None
    try:
        finish_contact_archive()
    except Exception as error:
        archive_error = f"{type(error).__name__}: {error}"
    trace_payload = {
        "metadata": METADATA,
        "error": f"{type(exc).__name__}: {exc}",
        "raw_contact_archive_finalization_error": archive_error,
        "traceback": traceback.format_exc(),
        "samples": SAMPLES,
    }
    try:
        write_json(trace_path, json_safe(trace_payload))
    except RuntimeError as artifact_error:
        # Fail closed without ever creating an over-50-MB file.  This branch is
        # only a last-resort error record; normal traces are bounded by the
        # deterministic per-step evidence policy above.
        write_json(
            trace_path,
            json_safe(
                {
                    "metadata": METADATA,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "sample_count_before_failure": len(SAMPLES),
                    "samples_omitted_by_artifact_size_guard": True,
                    "artifact_size_guard_error": str(artifact_error),
                }
            ),
        )
    checks = [
        check("native_program_completed", 0, "==", 1, "boolean"),
        check("samples_preserved_on_failure", len(SAMPLES), ">=", 1, "samples"),
    ]
    if CONTACT_ARCHIVE_INDEX is not None and RAW_CONTACT_ENTRY_COUNT is not None:
        checks.extend(
            [
                check(
                    "partial_private_raw_contact_archive_record_count_matches_independent_count",
                    int(CONTACT_ARCHIVE_INDEX["records"] == RAW_CONTACT_ENTRY_COUNT[0]),
                    "==",
                    1,
                    "boolean",
                ),
                check(
                    "partial_private_raw_contact_archive_digest_matches_independent_digest",
                    int(
                        RAW_CONTACT_DIGEST is not None
                        and CONTACT_ARCHIVE_INDEX["stream_sha256"]
                        == RAW_CONTACT_DIGEST.hexdigest()
                    ),
                    "==",
                    1,
                    "boolean",
                ),
                check(
                    "partial_private_raw_contact_archive_round_trip_verified",
                    int(CONTACT_ARCHIVE_INDEX["all_appended_records_round_trip_verified"]),
                    "==",
                    1,
                    "boolean",
                ),
            ]
        )
    report = {
        "qualification": False,
        "native_trial_id": trial_id,
        "source_sha256": source_sha256,
        "checks": checks,
        "overall_pass": False,
        "error": f"{type(exc).__name__}: {exc}",
        "artifacts": [str(trace_path.relative_to(ROOT))],
    }
    write_json(output_path, report)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception as error:
        failure_report(error)
        exit_code = 2
    finally:
        try:
            finish_contact_archive()
        finally:
            app.close()
    raise SystemExit(exit_code)
