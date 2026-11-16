"""Native component probe for the frozen media-separation apparatus.

This preflight answers three still-open questions with fresh PhysX state:
(1) whether complete segmented annular washers remain above the full perforated
screen, (2) whether transverse ceramic media which bridge an aperture are
released by the contract's physical revolute-screen motion, and (3) whether the
full screen, two support beams, and receiver coexist without gross reset drift
or penetration.  It is not a robot demo or aggregate task qualification.
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import get_current_stage
import numpy as np
import torch
import trimesh
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

APERTURE = 0.0127
RIB = 0.003
PITCH = APERTURE + RIB
FLOOR_T = 0.003
PERFORATED_R = 0.150
SCREEN_INNER_R = 0.18115
SCREEN_OUTER_R = 0.18415
SCREEN_DEPTH = 0.0635
SCREEN_MASS = 0.40
MEDIA_D = 0.009525
MEDIA_L = 0.022225
MEDIA_DENSITY = 2500.0
WASHER_OD = 0.030
WASHER_ID = 0.008
WASHER_T = 0.004
WASHER_MASS = 0.0206170159
SUPPORT_TOP_Z = 0.130
SCREEN_FLOOR_Z = SUPPORT_TOP_Z + FLOOR_T


def material(sf=0.50, df=0.35):
    return sim_utils.RigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df, restitution=0.05)


def collision_props():
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0)


def static_box(path, size, pos, color=(0.25, 0.32, 0.38)):
    cfg = sim_utils.CuboidCfg(size=size, collision_props=collision_props(), physics_material=material(),
                              activate_contact_sensors=True,
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
    cfg.func(path, cfg, translation=pos)


def spawn_isolated_aperture(cx, cy, floor_z, name):
    """Finite collision fixture for one complete aperture retention trial."""
    side, patch = (0.050 - APERTURE) / 2, 0.050
    specs = (
        ((side, patch, FLOOR_T), (cx-(APERTURE+side)/2, cy, floor_z-FLOOR_T/2)),
        ((side, patch, FLOOR_T), (cx+(APERTURE+side)/2, cy, floor_z-FLOOR_T/2)),
        ((APERTURE, side, FLOOR_T), (cx, cy-(APERTURE+side)/2, floor_z-FLOOR_T/2)),
        ((APERTURE, side, FLOOR_T), (cx, cy+(APERTURE+side)/2, floor_z-FLOOR_T/2)),
    )
    for i, (size, pos) in enumerate(specs):
        static_box(f"/World/{name}/Rib_{i}", size, pos)
    # Deliberately no containment walls or auxiliary support: the frozen
    # washer trial is one central complete aperture and its surrounding floor.


def parallel_cut_mesh():
    radius = MEDIA_D / 2
    slope = math.tan(math.radians(15.0))
    length = MEDIA_L - 2 * radius * slope
    vertices, faces, n = [], [], 32
    for sign in (-1.0, 1.0):
        for i in range(n):
            a = 2 * math.pi * i / n
            x, y = radius * math.cos(a), radius * math.sin(a)
            vertices.append((x, y, slope * x + sign * length / 2))
    for i in range(1, n - 1):
        faces.extend(((0, i + 1, i), (n, n + i, n + i + 1)))
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((i, j, n + j), (i, n + j, n + i)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def quarter_annulus_mesh(segment):
    """Watertight non-convex 90-degree annular sector (the bore is real)."""
    ro, ri, z = WASHER_OD / 2, WASHER_ID / 2, WASHER_T / 2
    a0, a1 = segment * math.pi / 2, (segment + 1) * math.pi / 2
    n, vertices, faces = 16, [], []
    # Per angle: lower-inner, lower-outer, upper-inner, upper-outer.
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        ca, sa = math.cos(a), math.sin(a)
        vertices += [(ri*ca,ri*sa,-z),(ro*ca,ro*sa,-z),
                     (ri*ca,ri*sa,z),(ro*ca,ro*sa,z)]
    for i in range(n):
        a, b = 4*i, 4*(i+1)
        faces += [(a,b+1,a+1),(a,b,b+1),             # bottom
                  (a+2,a+3,b+3),(a+2,b+3,b+2),       # top
                  (a,b+2,b),(a,a+2,b+2),             # inner wall
                  (a+1,b+1,b+3),(a+1,b+3,a+3)]       # outer wall
    faces += [(0,1,3),(0,3,2),(4*n,4*n+2,4*n+3),(4*n,4*n+3,4*n+1)]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    assert mesh.is_watertight and not mesh.is_convex
    return mesh


def spawn_authored_mesh(mesh, mass, color):
    def _spawn(path, cfg, translation=None, orientation=None, **kwargs):
        stage = get_current_stage()
        _spawn_mesh_geom_from_mesh(path, cfg, mesh, translation, orientation, stage=stage)
        # Dynamic triangle meshes are invalid in PhysX.  Convex decomposition,
        # unlike convexHull, retains the sector's inner cylindrical wall.
        collision_prim = stage.GetPrimAtPath(path + "/geometry/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexDecomposition)
        inertia = mesh.moment_inertia * (mass / mesh.mass)
        vals, axes = np.linalg.eigh(inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        tf = np.eye(4); tf[:3, :3] = axes
        qw, qx, qy, qz = trimesh.transformations.quaternion_from_matrix(tf)
        api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path))
        api.CreateMassAttr(float(mass)); api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, mesh.center_mass)))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, vals)))
        api.CreatePrincipalAxesAttr(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
        return stage.GetPrimAtPath(path)
    return sim_utils.MeshCfg(func=_spawn,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32,
            solver_velocity_iteration_count=8, max_depenetration_velocity=2.0),
        mass_props=None, collision_props=collision_props(), physics_material=material(),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))


def rigid_mesh(path, mesh, mass, pos, quat=(0, 0, 0, 1), color=(0.7, 0.5, 0.2)):
    return RigidObject(RigidObjectCfg(prim_path=path, spawn=spawn_authored_mesh(mesh, mass, color),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat)))


def complete_aperture_centers():
    centers, lim = [], int(PERFORATED_R / PITCH) + 1
    for ix in range(-lim, lim + 1):
        for iy in range(-lim, lim + 1):
            x, y = ix * PITCH, iy * PITCH
            if all(math.hypot(x + sx * APERTURE / 2, y + sy * APERTURE / 2) <= PERFORATED_R
                   for sx in (-1, 1) for sy in (-1, 1)):
                centers.append((x, y))
    return centers


def spawn_full_screen_and_fixture(origin=(0.0, 0.0, SCREEN_FLOOR_Z), hinge_xy=(0.0, 0.0)):
    """Author full finite ribs, circular wall approximation, supports, receiver."""
    ox, oy, oz = origin
    root = UsdGeom.Xform.Define(get_current_stage(), "/World/Apparatus/Screen")
    root.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(SCREEN_MASS)
    # Complete square openings: floor is the complement represented by a grid
    # of finite-width ribs plus the solid annulus from perforated floor to wall.
    half = PERFORATED_R
    for i in range(-int(half / PITCH) - 1, int(half / PITCH) + 2):
        coord = (i + 0.5) * PITCH
        span = 2 * math.sqrt(max(0.0, half * half - coord * coord))
        if span > 0:
            static_box(f"/World/Apparatus/Screen/FloorX_{i:+03d}", (RIB, span, FLOOR_T),
                       (coord, 0, -FLOOR_T / 2))
            static_box(f"/World/Apparatus/Screen/FloorY_{i:+03d}", (span, RIB, FLOOR_T),
                       (0, coord, -FLOOR_T / 2))
    # 96-sided wall and the solid radial band are collision-enabled pieces.
    nwall = 96
    for i in range(nwall):
        a = 2 * math.pi * i / nwall
        chord = 2 * SCREEN_INNER_R * math.sin(math.pi / nwall)
        band_mid = (PERFORATED_R + SCREEN_INNER_R) / 2
        band_chord = 2 * band_mid * math.sin(math.pi / nwall) + 0.0005
        bpx, bpy = band_mid * math.cos(a), band_mid * math.sin(a)
        band_cfg = sim_utils.CuboidCfg(size=(SCREEN_INNER_R-PERFORATED_R, band_chord, FLOOR_T),
            collision_props=collision_props(), physics_material=material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.38, 0.45)))
        band_cfg.func(f"/World/Apparatus/Screen/Band_{i:03d}", band_cfg,
                      translation=(bpx, bpy, -FLOOR_T/2),
                      orientation=(0, 0, math.sin(a/2), math.cos(a/2)))
        px, py = (SCREEN_INNER_R + 0.0015) * math.cos(a), (SCREEN_INNER_R + 0.0015) * math.sin(a)
        cfg = sim_utils.CuboidCfg(size=(0.003, chord + 0.0005, SCREEN_DEPTH), collision_props=collision_props(),
            physics_material=material(), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.38, 0.45)))
        cfg.func(f"/World/Apparatus/Screen/Wall_{i:03d}", cfg, translation=(px, py, SCREEN_DEPTH/2),
                 orientation=(0, 0, math.sin(a/2), math.cos(a/2)))
    # Receiver and its two support beams use frozen dimensions.
    rz = 0.0
    static_box("/World/Apparatus/Receiver/Floor", (0.45, 0.40, 0.005), (ox, oy, rz + 0.0025), (0.18, 0.24, 0.30))
    for sx in (-1, 1):
        static_box(f"/World/Apparatus/Receiver/WallX{sx:+d}", (0.005, 0.40, 0.08),
                   (ox + sx*0.2225, oy, rz + 0.04))
    for sy in (-1, 1):
        static_box(f"/World/Apparatus/Receiver/WallY{sy:+d}", (0.44, 0.005, 0.08),
                   (ox, oy + sy*0.1975, rz + 0.04))
    for sy in (-0.10, 0.10):
        static_box(f"/World/Apparatus/Support_{sy:+.2f}", (0.52, 0.015, 0.010),
                   (ox, oy + sy, SUPPORT_TOP_Z - 0.005), (0.22, 0.28, 0.33))
    # World-to-screen revolute joint is the only actuation path used below.
    # Updating its drive target is a guarded joint command, not an object pose
    # or velocity write.
    joint = UsdPhysics.RevoluteJoint.Define(get_current_stage(), "/World/Apparatus/ScreenHinge")
    joint.CreateAxisAttr("X")
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Apparatus/Screen")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(ox + hinge_xy[0], oy + hinge_xy[1], oz))
    joint.CreateLocalPos1Attr(Gf.Vec3f(hinge_xy[0], hinge_xy[1], 0))
    joint.CreateLowerLimitAttr(-6.0); joint.CreateUpperLimitAttr(6.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force"); drive.CreateStiffnessAttr(500.0); drive.CreateDampingAttr(20.0)
    drive.CreateMaxForceAttr(50.0); drive.CreateTargetPositionAttr(0.0)
    return root, drive


def quat_ry(deg):
    a = math.radians(deg) / 2
    return (0.0, math.sin(a), 0.0, math.cos(a))


def quat_mul(a, b):
    """Hamilton product for the runtime's XYZW convention."""
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)


def quat_rz(deg):
    a = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(a), math.cos(a))


def read_pose(body):
    p = body.data.root_pos_w.torch[0].detach().cpu().tolist()
    q = body.data.root_quat_w.torch[0].detach().cpu().tolist()
    return [*map(float, p), *map(float, q)]


def support_z(q, mesh):
    x, y, z, w = q
    rot = trimesh.transformations.quaternion_matrix((w, x, y, z))[:3, :3]
    return max(abs((mesh.vertices @ rot.T)[:, 2]))


class PlaneCrossingTracker:
    """Accumulate a finite collider's downward z=0 crossing over many steps.

    A finite body does not put every vertex through a plane in one physics
    interval.  Each vertex/edge trajectory is therefore recorded on the
    interval where it crosses, and classification is deferred until the
    complete collider is below the plane.  For the convex media and convex
    washer segments, strict containment of all vertex intersections in one
    convex aperture also contains every swept edge intersection.
    """

    def __init__(self, initial_vertices, aperture_center, faces):
        self.previous = np.asarray(initial_vertices, dtype=float)
        self.center = np.asarray(aperture_center, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self.xy = []
        self.started_above = bool(np.all(self.previous[:, 2] > 0.0))

    def update(self, current_vertices):
        current = np.asarray(current_vertices, dtype=float)
        for a, b in zip(self.previous, current, strict=True):
            if a[2] >= 0.0 and b[2] < 0.0:
                denominator = a[2] - b[2]
                t = a[2] / denominator if denominator > 0.0 else 0.0
                self.xy.append(a[:2] + t * (b[:2] - a[:2]))
        # Slice every triangle edge at the plane at both ends of this
        # interval.  This catches rotating/straddling faces missed by vertex
        # trajectories and supplies signed rib clearance for the swept body.
        for vertices in (self.previous, current):
            for face in self.faces:
                triangle = vertices[face]
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    a, b = triangle[i], triangle[j]
                    if (a[2] >= 0.0) != (b[2] >= 0.0):
                        t = a[2] / (a[2] - b[2])
                        self.xy.append(a[:2] + t * (b[:2] - a[:2]))
        self.previous = current
        if not self.started_above or not np.all(current[:, 2] < 0.0):
            return None
        if not self.xy:
            return {"completed": True, "clear": False, "minimum_clearance_m": -1.0,
                    "classification": "unobserved_or_upward_crossing"}
        points = np.asarray(self.xy)
        clearance = APERTURE / 2 - np.max(np.abs(points - self.center), axis=1)
        clear = bool(np.all(clearance > 0.0))
        return {"completed": True, "clear": clear,
                "minimum_clearance_m": float(clearance.min()),
                "classification": "clear_aperture" if clear else "rib_or_outside"}


def main():
    dt = 1 / args.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device,
        render_interval=max(1, args.physics_hz // 20)))
    # Keep the complete apparatus spatially separate from the one central
    # washer aperture. Trials are sequential: there is never a bank of cases
    # interacting in one world.
    centers = complete_aperture_centers()
    screen_root, screen_drive = spawn_full_screen_and_fixture(
        origin=(0.0, 1.0, SCREEN_FLOOR_Z), hinge_xy=centers[0])
    floor_z, wx, wy = 0.25, 0.60, 0.0
    spawn_isolated_aperture(wx, wy, floor_z, "CentralWasherAperture")

    # One complete four-segment washer is reset for each frozen transform.
    wm = [quarter_annulus_mesh(j) for j in range(4)]
    washer_segments = [rigid_mesh(f"/World/TestWasher/S{j}", wm[j], WASHER_MASS/4,
        (wx, wy, floor_z + 0.05), color=(0.40, 0.42, 0.45)) for j in range(4)]
    stage = get_current_stage()
    def recreate_washer_joints():
        """Re-author joints so fracture/contact state cannot leak across trials."""
        for j in range(4):
            path = f"/World/TestWasher/Joint{j}"
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
            fixed = UsdPhysics.FixedJoint.Define(stage, path)
            fixed.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/TestWasher/S{j}")])
            fixed.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/TestWasher/S{(j+1)%4}")])
            fixed.CreateCollisionEnabledAttr(False)
            fixed.CreateBreakForceAttr(200.0); fixed.CreateBreakTorqueAttr(200.0)
    recreate_washer_joints()

    # One medium is reset for the 4 yaw x 5 offset bridge seeds.
    media_mesh = parallel_cut_mesh()
    bridge_body = rigid_mesh("/World/TestBridge", media_mesh, media_mesh.volume*MEDIA_DENSITY,
        (0.0, 1.0, SCREEN_FLOOR_Z + 0.05), quat_ry(90))

    bridge_contact = ContactSensor(ContactSensorCfg(
        prim_path="/World/TestBridge",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        filter_prim_paths_expr=["/World/Apparatus/Screen/.*"],
    ))
    support_contact = ContactSensor(ContactSensorCfg(
        prim_path="/World/Apparatus/Screen",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        filter_prim_paths_expr=[
            "/World/Apparatus/Support_-0.10",
            "/World/Apparatus/Support_+0.10",
        ],
    ))

    sim.reset()
    all_bodies = [*washer_segments, bridge_body]
    for body in all_bodies: body.reset()

    def reset_body(body, pos, quat):
        pose = torch.tensor([[*pos, *quat]], dtype=torch.float32, device=args.device)
        vel = torch.zeros((1, 6), dtype=torch.float32, device=args.device)
        body.write_root_pose_to_sim_index(pose); body.write_root_velocity_to_sim_index(vel)

    def world_vertices(pose, mesh):
        qx,qy,qz,qw = pose[3:]
        rot = trimesh.transformations.quaternion_matrix((qw,qx,qy,qz))[:3,:3]
        return mesh.vertices @ rot.T + np.asarray(pose[:3])

    def screen_local_vertices(world_points):
        """Transform sampled collider vertices into the actual moving screen frame."""
        usd = UsdGeom.XformCache().GetLocalToWorldTransform(screen_root.GetPrim())
        world_from_screen = np.asarray([[float(usd[r][c]) for c in range(4)] for r in range(4)])
        homogeneous = np.column_stack((world_points, np.ones(len(world_points))))
        return (homogeneous @ np.linalg.inv(world_from_screen))[:, :3]

    washer_results, physics_trace, trace_20hz = [], [], []
    trial = 0
    for alpha in (0,30,60,90,120,150,180,210,240,270):
        for beta in (0,90,180,270):
            ax,bz = math.radians(alpha)/2, math.radians(beta)/2
            quat=(math.cos(bz)*math.sin(ax),math.sin(bz)*math.sin(ax),
                  math.sin(bz)*math.cos(ax),math.cos(bz)*math.cos(ax))
            hz=max(support_z(quat,m) for m in wm); zc=floor_z+0.00001+hz
            # Reset the simulation and recreate every breakable joint before
            # each frozen transform; root writes are diagnostic initialisation,
            # never production scoring or a success action.
            recreate_washer_joints()
            # Reparse the freshly authored joints into PhysX before applying
            # the diagnostic initial state.
            sim.reset()
            for body in washer_segments: reset_body(body,(wx,wy,zc),quat)
            sim.forward()
            initial_segment_vertices = [world_vertices(read_pose(b), m)
                                        for b, m in zip(washer_segments, wm, strict=True)]
            trackers = [PlaneCrossingTracker(v - np.array((wx, wy, floor_z)), (0.0, 0.0), m.faces)
                        for v, m in zip(initial_segment_vertices, wm, strict=True)]
            escaped, tunneled, escaped_segments, crossing_records = False, False, [], []
            initial_relative = []
            poses0 = [read_pose(b) for b in washer_segments]
            for j in range(4):
                ta, tb = poses0[j], poses0[(j+1)%4]
                ra = trimesh.transformations.quaternion_matrix((ta[6],ta[3],ta[4],ta[5]))[:3,:3]
                rb = trimesh.transformations.quaternion_matrix((tb[6],tb[3],tb[4],tb[5]))[:3,:3]
                initial_relative.append((ra.T @ (np.asarray(tb[:3])-np.asarray(ta[:3])), ra.T @ rb))
            for step in range(3 * args.physics_hz):
                sim.step(render=False)
                for body in washer_segments: body.update(dt)
                segment_vertices=[world_vertices(read_pose(b),m) for b,m in zip(washer_segments,wm,strict=True)]
                for segment, segverts in enumerate(segment_vertices):
                    if segment in escaped_segments:
                        continue
                    record = trackers[segment].update(segverts - np.array((wx, wy, floor_z)))
                    if record is not None:
                        record.update({"segment": segment, "physics_step": step})
                        crossing_records.append(record)
                        escaped_segments.append(segment)
                        escaped = True
                        tunneled = tunneled or not record["clear"]
                if escaped_segments:
                    break
            poses=[read_pose(b) for b in washer_segments]
            current_relative=[]
            for j in range(4):
                ta, tb = poses[j], poses[(j+1)%4]
                ra = trimesh.transformations.quaternion_matrix((ta[6],ta[3],ta[4],ta[5]))[:3,:3]
                rb = trimesh.transformations.quaternion_matrix((tb[6],tb[3],tb[4],tb[5]))[:3,:3]
                current_relative.append((ra.T @ (np.asarray(tb[:3])-np.asarray(ta[:3])), ra.T @ rb))
            relative_errors=[float(np.linalg.norm(a[0]-b[0])) for a,b in zip(initial_relative,current_relative,strict=True)]
            relative_angle_errors=[]
            for initial_tf, current_tf in zip(initial_relative,current_relative,strict=True):
                delta = initial_tf[1].T @ current_tf[1]
                relative_angle_errors.append(math.degrees(math.acos(
                    float(np.clip((np.trace(delta)-1)/2,-1,1)))))
            joint_valid=[stage.GetPrimAtPath(f"/World/TestWasher/Joint{j}").IsValid() for j in range(4)]
            washer_results.append({"trial":trial,"alpha_deg":alpha,"beta_deg":beta,
                "rotation_order":"Rz(beta)Rx(alpha)","center_xy_m":[wx,wy],"center_z_m":zc,
                "escaped":escaped,"escaped_segment_indices":escaped_segments if escaped else [],
                "rib_or_outside_crossing":tunneled,"crossing_records":crossing_records,
                "joint_prim_valid_flags":joint_valid,
                "maximum_joint_endpoint_relative_translation_error_m":max(relative_errors),
                "maximum_joint_endpoint_relative_rotation_error_deg":max(relative_angle_errors),
                "segment_poses_xyzw":poses})
            trial += 1

    # Frozen bridge seeds: transverse body, four yaw values and five offsets.
    bridge_results=[]; qbase=quat_ry(90); bridge_center=(centers[0][0],centers[0][1]+1.0)
    offsets=((0,0),(0.001,0),(-0.001,0),(0,0.001),(0,-0.001))
    for yaw in (0,45,90,135):
        # Frozen transverse transform Rz(yaw) Ry(pi/2), XYZW.
        q=quat_mul(quat_rz(yaw), qbase)
        hz=support_z(q,media_mesh)
        for offset in offsets:
            pos=(bridge_center[0]+offset[0],bridge_center[1]+offset[1],SCREEN_FLOOR_Z+0.00001+hz)
            # Every gravity seed starts from a full simulator reset with a
            # neutral screen drive; no post-actuation screen/contact state is reused.
            sim.reset()
            screen_drive.GetTargetPositionAttr().Set(0.0)
            reset_body(bridge_body,pos,q); sim.forward()
            speeds=[]
            contact_tail=[]
            maximum_dynamic_sampled_penetration = 0.0
            gravity_tracker = PlaneCrossingTracker(
                screen_local_vertices(world_vertices(read_pose(bridge_body), media_mesh)),
                (bridge_center[0], bridge_center[1]-1.0), media_mesh.faces)
            gravity_crossing = None
            for step in range(3 * args.physics_hz):
                sim.step(render=False); bridge_body.update(dt)
                bridge_contact.update(dt, force_recompute=True)
                v=bridge_body.data.root_lin_vel_w.torch[0].detach().cpu().numpy()
                speeds.append(float(np.linalg.norm(v)))
                force = bridge_contact.data.normal_force_matrix_w.torch
                contact_tail.append(float(torch.linalg.vector_norm(force).detach().cpu()))
                pose_now = read_pose(bridge_body)
                local_now = screen_local_vertices(world_vertices(pose_now, media_mesh))
                # Sampled signed penetration into the finite floor: a point
                # within floor thickness is permitted only inside one clear
                # aperture.  This is reported as a physics-rate sample, not a
                # claim about unobserved PhysX substeps.
                for point in local_now:
                    if -FLOOR_T <= point[2] <= 0.0:
                        dx = abs((point[0] + PITCH/2) % PITCH - PITCH/2)
                        dy = abs((point[1] + PITCH/2) % PITCH - PITCH/2)
                        if dx > APERTURE/2 or dy > APERTURE/2:
                            maximum_dynamic_sampled_penetration = max(
                                maximum_dynamic_sampled_penetration,
                                min(point[2] + FLOOR_T, -point[2]))
                sample={"phase":"gravity","yaw_deg":yaw,"offset_xy_m":list(offset),
                    "physics_step":step,"pose_xyzw":pose_now,
                    "speed_mps":speeds[-1],"contact_force_N":contact_tail[-1],
                    "sampled_floor_penetration_m":maximum_dynamic_sampled_penetration}
                physics_trace.append(sample)
                if step % max(1,args.physics_hz//20)==0: trace_20hz.append(sample)
                if gravity_crossing is None:
                    gravity_crossing = gravity_tracker.update(
                        screen_local_vertices(world_vertices(read_pose(bridge_body), media_mesh)))
            settled=read_pose(bridge_body)
            sv=screen_local_vertices(world_vertices(settled,media_mesh))
            tail = 60
            bridged=bool(max(speeds[-tail:])<0.005 and min(contact_tail[-tail:])>1e-6 and
                         settled[2]>SCREEN_FLOOR_Z and sv[:,2].min()<=0.001 and
                         np.ptp(sv[:,0])>APERTURE)
            gravity_passed = bool(gravity_crossing and gravity_crossing["clear"])
            gravity_invalid_crossing = bool(gravity_crossing and not gravity_crossing["clear"])
            released=False; tunnel=False; min_clearance=-1.0; observed_roll=[]
            if bridged:
                # Recreate the original frozen seed in a fresh reset and let it
                # settle again.  The release trial never starts by writing the
                # previous trial's terminal pose.
                sim.reset()
                screen_drive.GetTargetPositionAttr().Set(0.0)
                reset_body(bridge_body,pos,q); sim.forward()
                for _ in range(3 * args.physics_hz):
                    sim.step(render=False); bridge_body.update(dt)
                release_tracker = PlaneCrossingTracker(
                    screen_local_vertices(world_vertices(read_pose(bridge_body), media_mesh)),
                    (bridge_center[0], bridge_center[1]-1.0), media_mesh.faces)
                for step in range(3*args.physics_hz):
                    t=step*dt
                    target=5.0*math.sin(4*math.pi*t) if t<2.0 else 0.0
                    screen_drive.GetTargetPositionAttr().Set(target)
                    sim.step(render=False); bridge_body.update(dt)
                    p=read_pose(bridge_body); vv=world_vertices(p,media_mesh)
                    # Query the actual driven rigid transform, not merely the command.
                    m=UsdGeom.XformCache().GetLocalToWorldTransform(screen_root.GetPrim())
                    observed_roll.append(math.degrees(math.atan2(float(m[2][1]),float(m[2][2]))))
                    current_local=screen_local_vertices(vv)
                    sample={"phase":"release","yaw_deg":yaw,"offset_xy_m":list(offset),
                        "physics_step":step,"pose_xyzw":p,"observed_screen_roll_deg":observed_roll[-1]}
                    physics_trace.append(sample)
                    if step % max(1,args.physics_hz//20)==0: trace_20hz.append(sample)
                    crossing = release_tracker.update(current_local)
                    if crossing is not None:
                        min_clearance=crossing["minimum_clearance_m"]
                        released=bool(crossing["clear"])
                        tunnel=not released
                        break
            bridge_results.append({"yaw_deg":yaw,"offset_xy_m":list(offset),"bridged":bridged,
                "gravity_passed_clear_aperture":gravity_passed,
                "gravity_rib_or_outside_crossing":gravity_invalid_crossing,
                "gravity_crossing":gravity_crossing,
                "released":released,"tunneled":tunnel,"minimum_vertex_rib_clearance_m":min_clearance,
                "minimum_final_60_interval_native_contact_force_N":min(contact_tail[-tail:]),
                "maximum_sampled_dynamic_floor_penetration_m":maximum_dynamic_sampled_penetration,
                "observed_screen_roll_range_deg":(max(observed_roll)-min(observed_roll)) if observed_roll else 0.0})

    bridged_count=sum(r["bridged"] for r in bridge_results)
    transverse_gravity_pass_count=sum(r["gravity_passed_clear_aperture"] for r in bridge_results)
    transverse_gravity_invalid_crossing_count=sum(r["gravity_rib_or_outside_crossing"] for r in bridge_results)
    released_count=sum(r["bridged"] and r["released"] and not r["tunneled"] for r in bridge_results)
    bridge_release_fraction=released_count/bridged_count if bridged_count else 0.0
    washer_escape_fraction = sum(r["escaped"] for r in washer_results) / len(washer_results)
    # Return the hinge to neutral and measure resolved support on both beams
    # over a fresh three-second post-actuation window.
    screen_drive.GetTargetPositionAttr().Set(0.0)
    support_force_min = [float("inf"), float("inf")]
    support_force_max = [0.0, 0.0]
    for _ in range(3 * args.physics_hz):
        sim.step(render=False)
        support_contact.update(dt, force_recompute=True)
        matrix = support_contact.data.normal_force_matrix_w.torch[0, 0]
        for beam in range(2):
            value = float(torch.linalg.vector_norm(matrix[beam]).detach().cpu())
            support_force_min[beam] = min(support_force_min[beam], value)
            support_force_max[beam] = max(support_force_max[beam], value)
    final = [read_pose(b) for b in all_bodies]
    finite_fraction = float(np.isfinite(np.asarray(final)).mean())
    screen_tf=UsdGeom.XformCache().GetLocalToWorldTransform(screen_root.GetPrim())
    screen_xyz=[float(screen_tf[3][i]) for i in range(3)]
    screen_roll=math.degrees(math.atan2(float(screen_tf[2][1]),float(screen_tf[2][2])))
    screen_pitch=math.degrees(math.atan2(-float(screen_tf[2][0]),
        math.hypot(float(screen_tf[2][1]),float(screen_tf[2][2]))))
    # Signed samples use the actual observed screen transform.  The screen
    # underside, rather than its root/pivot, is compared with support tops.
    tf_np=np.asarray([[float(screen_tf[r][c]) for c in range(4)] for r in range(4)])
    local_support_samples=[]
    for sy in (-0.10,0.10):
        for sx in np.linspace(-0.15,0.15,61):
            local_support_samples.append((sx,sy,-FLOOR_T,1.0))
    support_world=np.asarray(local_support_samples) @ tf_np
    beam0=support_world[:61,2]; beam1=support_world[61:,2]
    support_gap=max(abs(float(np.min(beam0))-SUPPORT_TOP_Z),
                    abs(float(np.min(beam1))-SUPPORT_TOP_Z))
    support_penetration=max(0.0, SUPPORT_TOP_Z-float(np.min(support_world[:,2])))
    cache=UsdGeom.XformCache()
    support_centers=[]
    for sy in (-0.10,0.10):
        prim=stage.GetPrimAtPath(f"/World/Apparatus/Support_{sy:+.2f}")
        tf=cache.GetLocalToWorldTransform(prim)
        support_centers.append(np.asarray([float(tf[3][i]) for i in range(3)]))
    receiver_tf=cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Apparatus/Receiver/Floor"))
    receiver_floor_center_z=float(receiver_tf[3][2])
    support_separation=float(np.linalg.norm(support_centers[1]-support_centers[0]))
    receiver_support_clearance=float(min(p[2] for p in support_centers)-receiver_floor_center_z)
    # Receiver walls end at z=0.08; the signed vertical clearance excludes
    # the two intended support contacts and covers the closest fixture below
    # every sampled screen floor/wall point at the neutral final pose.
    screen_receiver_signed_clearance=float(np.min(support_world[:,2])-0.08)

    report_path = args.report if args.report.is_absolute() else (Path.cwd()/args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = report_path.with_name(report_path.stem + "-trace.json")
    workspace = Path(__file__).resolve().parent.parent
    trace_path.write_text(json.dumps({"physics_sample_rate_hz":args.physics_hz,
        "washer_results":washer_results,"bridge_results":bridge_results,
        "physics_samples":physics_trace,"samples_20hz":trace_20hz}))
    checks=[
        {"name":"physics_dt_error","measured":abs(float(sim.get_physics_dt())-dt),"operator":"<=","threshold":1e-9,"unit":"s"},
        {"name":"complete_aperture_count","measured":len(centers),"operator":">=","threshold":1,"unit":"apertures"},
        {"name":"segmented_washer_trial_count","measured":len(washer_results),"operator":"==","threshold":40,"unit":"trials"},
        {"name":"washer_escape_rate","measured":washer_escape_fraction,"operator":"==","threshold":0.0,"unit":"escaped_per_40_trials"},
        {"name":"washer_trials_with_valid_joint_prims","measured":sum(all(r["joint_prim_valid_flags"]) for r in washer_results),"operator":"==","threshold":40,"unit":"trials"},
        {"name":"maximum_washer_joint_relative_translation_error","measured":max(r["maximum_joint_endpoint_relative_translation_error_m"] for r in washer_results),"operator":"<=","threshold":0.0005,"unit":"m"},
        {"name":"maximum_washer_joint_relative_rotation_error","measured":max(r["maximum_joint_endpoint_relative_rotation_error_deg"] for r in washer_results),"operator":"<=","threshold":0.5,"unit":"degrees"},
        {"name":"transverse_bridge_trial_count","measured":len(bridge_results),"operator":"==","threshold":20,"unit":"trials"},
        {"name":"transverse_gravity_clear_pass_rate","measured":transverse_gravity_pass_count/len(bridge_results),"operator":"<=","threshold":0.10,"unit":"fraction"},
        {"name":"transverse_gravity_rib_or_outside_crossing_count","measured":transverse_gravity_invalid_crossing_count,"operator":"==","threshold":0,"unit":"trials"},
        {"name":"gravity_bridged_seed_count","measured":bridged_count,"operator":">","threshold":0,"unit":"trials"},
        {"name":"physically_actuated_bridge_release_fraction","measured":bridge_release_fraction,"operator":">=","threshold":0.80,"unit":"released_per_bridged"},
        {"name":"washer_rib_or_outside_crossing_count","measured":sum(r["rib_or_outside_crossing"] for r in washer_results),"operator":"==","threshold":0,"unit":"trials"},
        {"name":"bridge_tunneling_count","measured":sum(r["tunneled"] for r in bridge_results),"operator":"==","threshold":0,"unit":"trials"},
        {"name":"observed_screen_roll_range_deg","measured":max((r["observed_screen_roll_range_deg"] for r in bridge_results),default=0.0),"operator":">=","threshold":8.0,"unit":"degrees"},
        {"name":"final_screen_lowest_geometry_support_top_gap","measured":support_gap,"operator":"<=","threshold":0.002,"unit":"m"},
        {"name":"final_screen_support_max_sampled_penetration","measured":support_penetration,"operator":"<=","threshold":0.00015,"unit":"m"},
        {"name":"maximum_dynamic_body_sampled_floor_penetration","measured":max(r["maximum_sampled_dynamic_floor_penetration_m"] for r in bridge_results),"operator":"<=","threshold":0.00015,"unit":"m"},
        {"name":"final_screen_receiver_signed_clearance","measured":screen_receiver_signed_clearance,"operator":">=","threshold":0.0,"unit":"m"},
        {"name":"final_screen_absolute_roll","measured":abs(screen_roll),"operator":"<=","threshold":3.0,"unit":"degrees"},
        {"name":"final_screen_absolute_pitch","measured":abs(screen_pitch),"operator":"<=","threshold":3.0,"unit":"degrees"},
        {"name":"left_support_resolved_contact_force_max","measured":support_force_max[0],"operator":">","threshold":0.0,"unit":"N"},
        {"name":"right_support_resolved_contact_force_max","measured":support_force_max[1],"operator":">","threshold":0.0,"unit":"N"},
        {"name":"native_support_beam_center_separation","measured":support_separation,"operator":"==","threshold":0.20,"unit":"m"},
        {"name":"native_receiver_floor_to_support_center_clearance","measured":receiver_support_clearance,"operator":">=","threshold":0.08,"unit":"m"},
        {"name":"finite_native_pose_fraction","measured":finite_fraction,"operator":"==","threshold":1.0,"unit":"fraction"},
        {"name":"physics_rate_trace_sample_count","measured":len(physics_trace),"operator":">","threshold":0,"unit":"samples"},
        {"name":"20hz_trace_sample_count","measured":len(trace_20hz),"operator":">","threshold":0,"unit":"samples"},
        {"name":"full_screen_complete_aperture_count","measured":len(centers),"operator":">=","threshold":1,"unit":"apertures"}]
    def passed(check):
        return {"==":lambda a,b:a==b,"<=":lambda a,b:a<=b,">=":lambda a,b:a>=b,
                ">":lambda a,b:a>b,"<":lambda a,b:a<b}[check["operator"]](
                    check["measured"], check["threshold"])
    report = {"checks":checks,
        "all_checks_passed":all(passed(check) for check in checks),
        "artifacts":[trace_path.resolve().relative_to(workspace).as_posix()],
        "limitations":["Component preflight only: it does not qualify breakable-joint impulses, bore lodging, robot grasp/lift, production scoring, camera visibility, or fresh-process demonstrations.",
        "The screen angular actuation is measured from native body outcomes; any unsupported aggregate-body behavior remains a failed check rather than being replaced by kinematic task-object writes.",
        "Washer joint integrity is sampled by native prim validity and relative endpoint transforms; native joint-impulse telemetry remains an explicit open question and is not inferred from prim activation."]}
    report_path.write_text(json.dumps(report, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
