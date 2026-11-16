"""Bounded native probe for the four-segment washer's breakable fixed joints.

Two otherwise identical, freely falling four-segment annuli strike separate rigid
plates.  One uses the production 200 N / 200 N-m joint limits and must remain
assembled; the other uses a deliberately weak 0.01 N / 0.01 N-m diagnostic
limit and must produce native JOINT_BREAK events and segment separation.  This
discovers the pinned runtime's standalone-joint mechanics/readback only.  It is
not the frozen washer-retention qualification or a robot demonstration.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-hz", type=int, choices=(240, 480), required=True)
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
import trimesh
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import create_prim, get_current_stage

RI, RO, THICKNESS = 0.004, 0.015, 0.004
DENSITY, WEDGES = 7850.0, 32
DROP_CENTER_Z, PLATE_TOP_Z = 0.100, 0.030
PRODUCTION_LIMIT, WEAK_LIMIT = 200.0, 0.01


def wedge_mesh(index: int) -> trimesh.Trimesh:
    da = 2.0 * math.pi / WEDGES
    a0, a1 = index * da, (index + 1) * da
    inner = RI / math.cos(da / 2.0)
    xy = [(inner * math.cos(a0), inner * math.sin(a0)),
          (RO * math.cos(a0), RO * math.sin(a0)),
          (RO * math.cos(a1), RO * math.sin(a1)),
          (inner * math.cos(a1), inner * math.sin(a1))]
    vertices = [(x, y, z) for z in (-THICKNESS / 2, THICKNESS / 2) for x, y in xy]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if not mesh.is_watertight or not mesh.is_convex:
        raise RuntimeError(f"invalid convex wedge {index}")
    return mesh


WEDGE_MESHES = [wedge_mesh(i) for i in range(WEDGES)]
SEGMENT_MESHES = [trimesh.util.concatenate(WEDGE_MESHES[8*i:8*(i+1)]) for i in range(4)]


def spawn_segment(prim_path, cfg, translation=None, orientation=None, **kwargs):
    stage = get_current_stage()
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    child_cfg = sim_utils.MeshCfg(rigid_props=None, mass_props=None,
        collision_props=cfg.collision_props, activate_contact_sensors=True,
        physics_material=cfg.physics_material, visual_material=cfg.visual_material)
    segment = int(prim_path.rsplit("S", 1)[1])
    for local_index, mesh in enumerate(WEDGE_MESHES[8*segment:8*(segment+1)]):
        _spawn_mesh_geom_from_mesh(f"{prim_path}/Wedge_{local_index:02d}", child_cfg, mesh, stage=stage)
    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    combined = SEGMENT_MESHES[segment]
    mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(prim_path))
    mass_api.CreateDensityAttr(DENSITY)
    mass_api.CreateMassAttr(float(combined.volume * DENSITY))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, combined.center_mass)))
    inertia = combined.moment_inertia * DENSITY
    eigvals, eigvecs = np.linalg.eigh(inertia)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1
    transform = np.eye(4); transform[:3, :3] = eigvecs
    q = trimesh.transformations.quaternion_from_matrix(transform)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, eigvals)))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
    return stage.GetPrimAtPath(prim_path)


def make_segment(path: str, center_x: float) -> RigidObject:
    cfg = sim_utils.MeshCfg(func=spawn_segment,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32,
            solver_velocity_iteration_count=8, max_depenetration_velocity=2.0),
        mass_props=None,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.45, dynamic_friction=0.35),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.40, 0.44)))
    return RigidObject(RigidObjectCfg(prim_path=path, spawn=cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(center_x, 0.0, DROP_CENTER_Z))))


def make_plate(name: str, x: float) -> None:
    cfg = sim_utils.CuboidCfg(size=(0.050, 0.050, 0.004),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.35),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.27, 0.32)))
    cfg.func(f"/World/{name}", cfg, translation=(x, 0.0, PLATE_TOP_Z - 0.002))
    # A small rigid boss under the first quadrant creates a reproducible,
    # asymmetric impact.  It is present in both cells, so the only experimental
    # variable is the fixed-joint break limit.
    boss = sim_utils.CuboidCfg(size=(0.006, 0.006, 0.010),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.35),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.30, 0.18)))
    boss.func(f"/World/{name}Boss", boss,
        translation=(x + 0.008, 0.008, PLATE_TOP_Z + 0.005))


def add_joints(cell: str, limit: float) -> list[str]:
    stage = get_current_stage(); paths = []
    for j in range(4):
        path = f"/World/{cell}/Joint_{j}"; paths.append(path)
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/{cell}/S{j}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/{cell}/S{(j+1)%4}")])
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateBreakForceAttr(limit)
        joint.CreateBreakTorqueAttr(limit)
    return paths


def pose(body: RigidObject) -> list[float]:
    return body.data.root_pose_w.torch[0].detach().cpu().tolist()


def pair_distance(poses: list[list[float]], j: int) -> float:
    return math.dist(poses[j][:3], poses[(j + 1) % 4][:3])


def main() -> None:
    dt = 1.0 / args.physics_hz
    steps = int(round(args.seconds * args.physics_hz))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device,
        render_interval=max(1, args.physics_hz // 20)))
    cells = (("Strong", -0.045, PRODUCTION_LIMIT), ("Weak", 0.045, WEAK_LIMIT))
    bodies: dict[str, list[RigidObject]] = {}
    joint_paths: dict[str, list[str]] = {}
    for name, x, limit in cells:
        make_plate(f"{name}Plate", x)
        bodies[name] = [make_segment(f"/World/{name}/S{j}", x) for j in range(4)]
        joint_paths[name] = add_joints(name, limit)

    break_events: list[dict] = []
    callback_errors: list[str] = []
    subscription = None
    try:
        import omni.physx
        from omni.physx.bindings._physx import SimulationEvent
        stream = omni.physx.get_physx_interface().get_simulation_event_stream_v2()
        def on_event(event):
            if int(event.type) != int(SimulationEvent.JOINT_BREAK):
                return
            try:
                payload = event.payload
                # carb.dictionary.Item.get requires an explicit default in the
                # pinned Kit runtime (unlike Python's dict.get).
                encoded = payload.get("jointPath", None)
                if encoded is None or len(encoded) != 2:
                    raise RuntimeError(f"missing/invalid jointPath payload: {encoded!r}")
                encoded_pair = [int(encoded[0]), int(encoded[1])]
                decoded = str(PhysicsSchemaTools.decodeSdfPath(*encoded_pair))
                # ``event.payload`` is a carb.dictionary.Item in Kit 110.3,
                # not a Python mapping: it supports get(key, default) but has
                # no keys() method.  Persist the actual native encoded pair
                # and decoded SdfPath without unsupported mapping introspection.
                break_events.append({"path": decoded, "encoded_joint_path": encoded_pair})
            except Exception as exc:
                callback_errors.append(f"{type(exc).__name__}: {exc}")
        subscription = stream.create_subscription_to_pop(on_event, name="segment-joint-break-probe")
    except Exception as exc:
        raise RuntimeError(f"native JOINT_BREAK subscription API unavailable: {exc}") from exc

    sim.reset()
    for group in bodies.values():
        for body in group: body.reset()
    initial = {name: [pose(body) for body in group] for name, group in bodies.items()}
    trace = []
    finite = True
    max_pair_change = {"Strong": 0.0, "Weak": 0.0}
    pair_change_history = {"Strong": [], "Weak": []}
    min_z = {"Strong": float("inf"), "Weak": float("inf")}
    for step in range(steps):
        sim.step(render=False)
        sample = {"step": step + 1, "time_s": (step + 1) * dt, "cells": {}}
        for name, group in bodies.items():
            for body in group: body.update(dt)
            poses = [pose(body) for body in group]
            finite = finite and bool(np.isfinite(np.asarray(poses)).all())
            changes = [abs(pair_distance(poses, j) - pair_distance(initial[name], j)) for j in range(4)]
            pair_change_history[name].append(max(changes))
            max_pair_change[name] = max(max_pair_change[name], *changes)
            min_z[name] = min(min_z[name], min(p[2] for p in poses))
            sample["cells"][name] = {"poses_xyzw": poses,
                "adjacent_root_distance_changes_m": changes}
        sample["break_event_count"] = len(break_events)
        trace.append(sample)

    stage = get_current_stage()
    strong_events = [e for e in break_events if e["path"].startswith("/World/Strong/")]
    weak_events = [e for e in break_events if e["path"].startswith("/World/Weak/")]
    authored_valid = {name: sum(stage.GetPrimAtPath(p).IsValid() for p in paths)
                      for name, paths in joint_paths.items()}
    final = {name: [pose(body) for body in group] for name, group in bodies.items()}
    native_masses = {name: [float(body.data.default_mass.torch[0, 0].cpu()) for body in group]
                     for name, group in bodies.items()}
    expected_segment_masses = [float(mesh.volume * DENSITY) for mesh in SEGMENT_MESHES]

    report_path = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = report_path.with_name(report_path.stem + "-trace.json.gz")
    workspace = Path(__file__).resolve().parent.parent
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream_file:
        json.dump({"scope": "standalone_four_segment_fixed_joint_break_discovery_only",
            "physics_sample_rate_hz": args.physics_hz, "cells": [list(c) for c in cells],
            "joint_paths": joint_paths, "break_events": break_events,
            "break_event_callback_errors": callback_errors,
            "initial_poses_xyzw": initial, "final_poses_xyzw": final,
            "native_segment_masses_kg": native_masses,
            "geometry_segment_masses_kg": expected_segment_masses,
            "physics_samples": trace}, stream_file, separators=(",", ":"))

    max_mass_error = max(abs(native_masses[name][j] - expected_segment_masses[j]) /
                         expected_segment_masses[j] for name in bodies for j in range(4))
    final_dwell_samples = min(args.physics_hz, steps)
    strong_final_dwell_max = max(pair_change_history["Strong"][-final_dwell_samples:])
    checks = [
        {"name":"physics_dt_absolute_error","measured":abs(float(sim.get_physics_dt())-dt),"operator":"<=","threshold":1e-9,"unit":"s"},
        {"name":"finite_native_pose_samples","measured":int(finite),"operator":"==","threshold":1,"unit":"bool"},
        {"name":"authored_convex_wedge_count","measured":sum(int(m.is_convex) for m in WEDGE_MESHES),"operator":"==","threshold":32,"unit":"colliders_per_annulus"},
        {"name":"native_segment_mass_relative_error","measured":max_mass_error,"operator":"<=","threshold":1e-5,"unit":"ratio"},
        {"name":"strong_cell_joint_break_event_count","measured":len(strong_events),"operator":"==","threshold":0,"unit":"events"},
        {"name":"weak_cell_joint_break_event_count","measured":len(weak_events),"operator":">","threshold":0,"unit":"events"},
        {"name":"joint_break_callback_error_count","measured":len(callback_errors),"operator":"==","threshold":0,"unit":"errors"},
        {"name":"strong_cell_final_one_second_maximum_adjacent_root_distance_change","measured":strong_final_dwell_max,"operator":"<=","threshold":0.0005,"unit":"m"},
        {"name":"weak_cell_maximum_adjacent_root_distance_change","measured":max_pair_change["Weak"],"operator":">","threshold":0.001,"unit":"m"},
        {"name":"strong_cell_authored_joint_prim_count","measured":authored_valid["Strong"],"operator":"==","threshold":4,"unit":"prims"},
        {"name":"weak_cell_authored_joint_prim_count","measured":authored_valid["Weak"],"operator":"==","threshold":4,"unit":"prims"},
        {"name":"fresh_physics_sample_count","measured":len(trace),"operator":"==","threshold":steps,"unit":"completed_physics_samples"},
    ]
    report_path.write_text(json.dumps({"checks":checks,
        "artifacts":[trace_path.resolve().relative_to(workspace).as_posix()]}, indent=2))
    subscription = None
    simulation_app.close()


if __name__ == "__main__":
    main()
