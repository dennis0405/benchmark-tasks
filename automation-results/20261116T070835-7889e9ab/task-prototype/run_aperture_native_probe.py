"""Native discovery probe for screen-aperture collision at 240 or 480 Hz.

This is not a qualification script.  It measures one bounded component question:
whether the proposed square aperture separates the frozen parallel-angle-cut
media mesh from a washer-OD conservative proxy across deterministic attitudes.  It deliberately
does not stand in for the segmented washer, joints, agitation, bore lodging,
full screen, robot, provenance scorer, or production task.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-hz", type=int, choices=(240, 480), required=True)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
import trimesh
import torch

APERTURE = 0.0127
PATCH = 0.050
FLOOR_THICKNESS = 0.003
MEDIA_D = 0.009525
MEDIA_L = 0.022225
WASHER_D = 0.030
WASHER_T = 0.004
MEDIA_MASS = 0.0035044969
MEDIA_DENSITY = 2500.0
WASHER_MASS = 0.0206170159
END_TRIALS = 20
TRANSVERSE_TRIALS = 20
WASHER_TRIALS = 40


def quat_y_then_z_xyzw(tilt: float, azimuth: float) -> tuple[float, float, float, float]:
    """Isaac Lab 3.x XYZW quaternion for Rz(azimuth)*Ry(tilt)."""
    ct, st = math.cos(tilt / 2), math.sin(tilt / 2)
    ca, sa = math.cos(azimuth / 2), math.sin(azimuth / 2)
    return (-sa * st, ca * st, sa * ct, ca * ct)


def spawn_aperture(cx: float, cy: float, name: str) -> None:
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.50, dynamic_friction=0.35, restitution=0.05)
    common = dict(
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0),
        physics_material=mat,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.35, 0.42)),
    )
    side = (PATCH - APERTURE) / 2
    specs = (
        ((side, PATCH, FLOOR_THICKNESS), (cx - (APERTURE + side) / 2, cy, -FLOOR_THICKNESS / 2)),
        ((side, PATCH, FLOOR_THICKNESS), (cx + (APERTURE + side) / 2, cy, -FLOOR_THICKNESS / 2)),
        ((APERTURE, side, FLOOR_THICKNESS), (cx, cy - (APERTURE + side) / 2, -FLOOR_THICKNESS / 2)),
        ((APERTURE, side, FLOOR_THICKNESS), (cx, cy + (APERTURE + side) / 2, -FLOOR_THICKNESS / 2)),
    )
    for j, (size, pos) in enumerate(specs):
        cfg = sim_utils.CuboidCfg(size=size, **common)
        cfg.func(f"/World/{name}/Rib_{j}", cfg, translation=pos)


def spawn_parallel_angle_cut_mesh(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the frozen 15-degree parallel-cut, 9.525 mm OD media body.

    The two end planes are parallel.  Their centerline separation is selected
    so the maximum axial envelope, including the plane slope across the OD, is
    exactly MEDIA_L.  A 32-sided convex mesh is used for native collision.
    """
    radius = MEDIA_D / 2
    slope = math.tan(math.radians(15.0))
    centerline_length = MEDIA_L - 2 * radius * slope
    vertices = []
    sections = 32
    for end_sign in (-1.0, 1.0):
        for i in range(sections):
            angle = 2 * math.pi * i / sections
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            vertices.append((x, y, slope * x + end_sign * centerline_length / 2))
    faces = []
    faces.append(tuple(reversed(range(sections))))
    faces.append(tuple(range(sections, 2 * sections)))
    for i in range(sections):
        j = (i + 1) % sections
        faces.append((i, j, sections + j, sections + i))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    from isaaclab.sim.utils import get_current_stage
    stage = get_current_stage()
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, stage=stage)
    return stage.GetPrimAtPath(prim_path)


def rigid_body(path: str, radius: float, height: float, mass: float, pos, quat, media=False) -> RigidObject:
    spawn_cfg = (
        sim_utils.MeshCfg(
            func=spawn_parallel_angle_cut_mesh,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=MEDIA_DENSITY),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.50, dynamic_friction=0.35, restitution=0.05
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.55, 0.20)),
        ) if media else sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.50, dynamic_friction=0.35, restitution=0.05
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.55, 0.20)),
        )
    )
    cfg = RigidObjectCfg(
        prim_path=path,
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
    )
    return RigidObject(cfg)


def axis_from_quat(q):
    """Return the local +Z axis in world coordinates from an XYZW quaternion."""
    x, y, z, w = q
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def half_extents(q, radius: float, height: float):
    """Exact directional support extents of a finite cylinder."""
    axis = axis_from_quat(q)
    return tuple(abs(a) * height / 2 + radius * math.sqrt(max(0.0, 1 - a * a)) for a in axis)


def quat_geodesic_angle(q0, q1):
    """Shortest rotation angle between XYZW unit quaternions."""
    dot = abs(sum(a * b for a, b in zip(q0, q1, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def swept_interval_certificate(a, b, cx, cy, radius, height):
    """Conservatively bound a cylinder's continuous sweep between two poses.

    Translation is linear over a PhysX interval.  Every material point is at
    most ``bounding_radius * rotation_angle`` from its endpoint solely due to
    rotation.  Expanding the endpoint support bounds by that amount therefore
    encloses the continuous swept cylinder without relying on intermediate
    discrete samples.
    """
    _, x0, y0, z0, *q0 = a
    _, x1, y1, z1, *q1 = b
    e0 = half_extents(q0, radius, height)
    e1 = half_extents(q1, radius, height)
    angle = quat_geodesic_angle(q0, q1)
    bounding_radius = math.hypot(radius, height / 2)
    rotation_pad = bounding_radius * angle
    x_bound = max(abs(x0 - cx) + e0[0], abs(x1 - cx) + e1[0]) + rotation_pad
    y_bound = max(abs(y0 - cy) + e0[1], abs(y1 - cy) + e1[1]) + rotation_pad
    z_min = min(z0 - e0[2], z1 - e1[2]) - rotation_pad
    z_max = max(z0 + e0[2], z1 + e1[2]) + rotation_pad
    return {
        "rib_clearance_m": min(APERTURE / 2 - x_bound, APERTURE / 2 - y_bound),
        "z_min_m": z_min,
        "z_max_m": z_max,
        "rotation_pad_m": rotation_pad,
    }


def read_native_pose_xyzw(body: RigidObject) -> list[float]:
    """Copy one source-backed ProxyArray pose to CPU in Isaac Lab XYZW order."""
    position = body.data.root_pos_w.torch[0].detach().cpu().tolist()
    quaternion = body.data.root_quat_w.torch[0].detach().cpu().tolist()
    return [*map(float, position), *map(float, quaternion)]


def main() -> None:
    dt = 1.0 / args.physics_hz
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, device=args.device, render_interval=max(1, args.physics_hz // 20))
    )
    trials = []
    spacing = 0.065
    definitions = []
    for kind, count in (("end_first", END_TRIALS), ("transverse", TRANSVERSE_TRIALS), ("washer_od_proxy", WASHER_TRIALS)):
        for i in range(count):
            index = len(definitions)
            cx = (index % 10 - 4.5) * spacing
            cy = (index // 10 - 3.5) * spacing
            az = 2 * math.pi * i / count
            if kind == "end_first":
                tilt, radius, height, mass = math.radians(5), MEDIA_D / 2, MEDIA_L, MEDIA_MASS
            elif kind == "transverse":
                tilt, radius, height, mass = math.radians(85), MEDIA_D / 2, MEDIA_L, MEDIA_MASS
            else:
                # Face-on through edge-on and back, deterministic in both azimuth and inclination.
                tilt = math.pi * i / (count - 1)
                radius, height, mass = WASHER_D / 2, WASHER_T, WASHER_MASS
            quat = quat_y_then_z_xyzw(tilt, az)
            hz = half_extents(quat, radius, height)[2]
            spawn_aperture(cx, cy, f"Patch_{index:03d}")
            body = rigid_body(
                f"/World/Body_{index:03d}", radius, height, mass,
                (cx, cy, 0.010 + hz), quat, media=kind != "washer_od_proxy",
            )
            definitions.append((kind, i, cx, cy, radius, height, body))
    sim.reset()
    for *_, body in definitions:
        body.reset()

    realized_media_mass = float(definitions[0][-1].data.body_mass.torch[0, 0].detach().cpu())
    native_inertia_tensor = definitions[0][-1].data.body_inertia.torch[0, 0].detach().reshape(3, 3)
    realized_media_principal_inertia = [
        float(v) for v in torch.linalg.eigvalsh(native_inertia_tensor).cpu().tolist()
    ]

    samples = 2 * args.physics_hz
    trace = [[] for _ in definitions]
    # Step zero is a native post-reset observation.  Keeping it in the trace
    # makes the initial-to-first-physics-step sweep part of the certificate.
    for row, (*_, body) in zip(trace, definitions, strict=True):
        row.append([0, *read_native_pose_xyzw(body)])
    for step in range(samples):
        sim.step(render=False)
        for row, (*_, body) in zip(trace, definitions, strict=True):
            body.update(dt)
            row.append([step + 1, *read_native_pose_xyzw(body)])

    for definition, row in zip(definitions, trace, strict=True):
        kind, trial_id, cx, cy, radius, height, _ = definition
        genuine_crossing = False
        minimum_rib_clearance = float("inf")
        outside_patch = False
        below_floor = False
        final_full_collider_above_floor = False
        _, _, _, initial_z, *initial_q = row[0]
        started_fully_above = initial_z - half_extents(initial_q, radius, height)[2] > 0.0
        crossing_active = False
        crossing_certified = True
        crossing_interval_count = 0
        max_rotation_pad = 0.0
        for sample in row:
            _, x, y, z, qx, qy, qz, qw = sample
            ex, ey, ez = half_extents((qx, qy, qz, qw), radius, height)
            clearance = min(APERTURE / 2 - (abs(x - cx) + ex), APERTURE / 2 - (abs(y - cy) + ey))
            minimum_rib_clearance = min(minimum_rib_clearance, clearance)
            outside_patch |= abs(x - cx) + ex > PATCH / 2 or abs(y - cy) + ey > PATCH / 2
            below_floor |= z + ez < -0.005
            final_full_collider_above_floor = z - ez > 0.0
        for previous, current in zip(row, row[1:]):
            swept = swept_interval_certificate(previous, current, cx, cy, radius, height)
            overlaps_floor_slab = (
                swept["z_min_m"] <= 0.0
                and swept["z_max_m"] >= -FLOOR_THICKNESS
            )
            if started_fully_above and overlaps_floor_slab:
                crossing_active = True
                crossing_interval_count += 1
                minimum_rib_clearance = min(minimum_rib_clearance, swept["rib_clearance_m"])
                max_rotation_pad = max(max_rotation_pad, swept["rotation_pad_m"])
                crossing_certified &= swept["rib_clearance_m"] >= 0.0
            _, _, _, z_now, *q_now = current
            ez_now = half_extents(q_now, radius, height)[2]
            if crossing_active and z_now + ez_now < -0.005:
                genuine_crossing = crossing_certified and crossing_interval_count > 0
                break
        trials.append({
            "kind": kind,
            "trial": trial_id,
            "genuine_full_body_aperture_passage": genuine_crossing,
            "genuine_non_passage": not genuine_crossing,
            "final_full_collider_above_floor": final_full_collider_above_floor,
            "ever_fully_below_floor": below_floor,
            "outside_finite_patch_escape": outside_patch,
            "minimum_full_footprint_rib_clearance_m": minimum_rib_clearance,
            "swept_floor_crossing_interval_count": crossing_interval_count,
            "maximum_swept_rotation_pad_m": max_rotation_pad,
            "final_pose_xyzw": row[-1][1:],
        })

    group = lambda name: [t for t in trials if t["kind"] == name]
    ends, transverse, washers = group("end_first"), group("transverse"), group("washer_od_proxy")
    fraction = lambda rows, key: sum(bool(r[key]) for r in rows) / len(rows)
    measured_dt = float(sim.get_physics_dt())
    report_path = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = report_path.with_name(report_path.stem + "-trace.json")
    executor_workspace = Path(__file__).resolve().parent.parent
    trace_artifact = trace_path.resolve().relative_to(executor_workspace)
    trace_path.write_text(json.dumps({
        "scope": "discovery_only_not_qualification",
        "physics_hz": args.physics_hz,
        "physics_sample_rate_hz": args.physics_hz,
        "trace_columns": ["step", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"],
        "trial_results": trials,
        "pose_traces": trace,
    }))
    report = {
        "checks": [
            {"name": "physics_dt_error_max", "measured": abs(measured_dt - dt), "operator": "<=", "threshold": 1e-9, "unit": "s"},
            {"name": "nominal_20hz_interval_physics_step_ratio", "measured": 0.05 / measured_dt, "operator": "==", "threshold": args.physics_hz / 20, "unit": "ratio_not_action_measurement"},
            {"name": "realized_media_mass", "measured": realized_media_mass, "operator": ">", "threshold": 0.0, "unit": "kg"},
            {"name": "realized_media_mass_relative_error", "measured": abs(realized_media_mass - MEDIA_MASS) / MEDIA_MASS, "operator": "<=", "threshold": 0.02, "unit": "ratio"},
            {"name": "realized_media_min_principal_inertia", "measured": min(realized_media_principal_inertia), "operator": ">", "threshold": 0.0, "unit": "kg_m2"},
            {"name": "end_first_genuine_passage_fraction", "measured": fraction(ends, "genuine_full_body_aperture_passage"), "operator": "==", "threshold": 1.0, "unit": "fraction_component_threshold"},
            {"name": "transverse_genuine_non_passage_fraction", "measured": fraction(transverse, "genuine_non_passage"), "operator": "==", "threshold": 1.0, "unit": "fraction_component_threshold"},
            {"name": "washer_od_proxy_genuine_non_passage_fraction", "measured": fraction(washers, "genuine_non_passage"), "operator": "==", "threshold": 1.0, "unit": "fraction_discovery_proxy"},
            {"name": "outside_patch_escape_fraction", "measured": fraction(trials, "outside_finite_patch_escape"), "operator": "==", "threshold": 0.0, "unit": "fraction_validity"},
            {"name": "deterministic_trial_count", "measured": len(trials), "operator": "==", "threshold": 80, "unit": "trials"},
        ],
        "artifacts": [trace_artifact.as_posix()],
        "limitations": [
            "The media collider is the frozen 15-degree parallel-angle-cut geometry; its native realized mass and inertia are reported. Solid washer-OD cylinders remain conservative discovery proxies, not the frozen four-segment washer assemblies.",
            "This probe does not test agitation release, washer-bore lodging, breakable joints, the full screen, spill scorer, YAM actions, or grading.",
            "The nominal 20 Hz ratio is timestep arithmetic; no action loop is measured here.",
            "Genuine passage uses a conservative continuous swept-cylinder bound for every physics interval overlapping the floor slab; uncertified/tunneling crossings are not counted.",
            "Non-passage is reported independently from final full-collider-above-floor state; a lodged or penetrating body is never labeled fully above the floor.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
