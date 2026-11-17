"""Import-safe builders and geometry for the full-screen YAM preflight.

This module deliberately has no argument parsing or AppLauncher side effects.
The same polygons returned by :func:`floor_pieces` are used for both USD
collision authoring and the offline Shapely audit.
"""
from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.configclass import make_configclass
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from shapely.geometry import Point, Polygon, box
from shapely.ops import triangulate, unary_union
from shapely.geometry.polygon import orient
from yam_arena.embodiment import BimanualYamEmbodiment
from yam_arena.rig import TABLETOP as TABLETOP_RIG
from yam_arena.robot import asset_link_paths
from yam_arena.scene import env_cfg_callback, scene_assets

APERTURE, RIB, PITCH = 0.0127, 0.003, 0.0157
PERF_R, INNER_R, OUTER_R = 0.150, 0.18115, 0.18415
FLOOR_T, DEPTH, MASS_TARGET = 0.003, 0.0635, 0.40
TABLETOP_HEIGHT, SUPPORT_H = 0.75, 0.0595
SCREEN_X, SCREEN_Y = 0.37, 0.0
# The 50 mm receiver walls end at z=0.805 m.  The posts and screen underside
# are authored at z=0.8095 m, leaving 4.5 mm against the frozen 4 mm minimum while
# making the support contact a zero-gap reset rather than an overlap that the
# solver would have to conceal by depenetration.
SUPPORT_TOP = TABLETOP_HEIGHT + SUPPORT_H
SCREEN_ROOT_Z = SUPPORT_TOP + FLOOR_T
SUPPORT_POST_RADIUS = 0.004
SUPPORT_POST_X = 0.125
SUPPORT_POST_Y = 0.115
CONTROL_HZ = 20
DISK_RESOLUTION = 96
# GEOS ``buffer`` uses this many segments per quadrant for each aperture.
# The collision boundary is therefore a regular 20-gon whose across-flats
# diameter is APERTURE*cos(pi/20) = 12.5436 mm.  This is inside the frozen
# 12.5--12.9 mm clear-diameter interval while retaining a genuinely circular
# (polygonal CAD) opening rather than the superseded square prototype.
APERTURE_RESOLUTION = 5
# The first native grasp attempt produced 63 raw points for one filtered
# fingertip/screen pair.  Reserve four times that observed requirement so the
# complete pair slice is available during denser transients.  The runner still
# rejects any count that exceeds this allocation; truncation cannot qualify.
RAW_CONTACT_CAPACITY_PER_PRIM = 256
TAB_SIZE = (0.050, 0.045, 0.006)
TAB_Y = OUTER_R + 0.0175
TAB_Z = 0.020


def complete_holes():
    lim = int(PERF_R / PITCH) + 1
    return [(i * PITCH, j * PITCH) for i in range(-lim, lim + 1) for j in range(-lim, lim + 1)
            if all(math.hypot(i * PITCH + sx * APERTURE / 2,
                              j * PITCH + sy * APERTURE / 2) <= PERF_R
                   for sx in (-1, 1) for sy in (-1, 1))]


HOLES = complete_holes()
assert len(HOLES) == 261


def disk(radius):
    return Point(0, 0).buffer(radius, resolution=DISK_RESOLUTION)


def floor_pieces():
    """Non-overlapping convex triangles forming the circular-perforated disk.

    Collision meshes are authored as convex hulls, so passing a concave cell
    would silently fill part of an aperture.  GEOS constrained-Delaunay output
    is filtered by a strict representative point against the exact polygonal
    solid; the later symmetric-difference audit detects either missing solid
    material or filled aperture material.
    """
    boundary = disk(INNER_R)
    hole_union = unary_union([
        Point(x, y).buffer(APERTURE / 2, resolution=APERTURE_RESOLUTION)
        for x, y in HOLES
    ])
    solid = boundary.difference(hole_union)
    pieces = [p for p in triangulate(solid)
              if p.area > 1e-12 and solid.covers(p.representative_point())]
    return pieces, solid


def polygon_integrals(poly):
    """Return A, integral(x), integral(y), integral(x^2), integral(y^2), integral(xy)."""
    # Shapely boolean operations may reverse winding. Holes must subtract.
    poly = orient(poly, sign=1.0)
    vals = [0.0] * 6
    rings = [poly.exterior, *poly.interiors]
    for ring in rings:
        xy = list(ring.coords)
        for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
            c = x0*y1-x1*y0
            vals[0] += c/2
            vals[1] += (x0+x1)*c/6
            vals[2] += (y0+y1)*c/6
            vals[3] += (x0*x0+x0*x1+x1*x1)*c/12
            vals[4] += (y0*y0+y0*y1+y1*y1)*c/12
            vals[5] += (2*x0*y0+x0*y1+x1*y0+2*x1*y1)*c/24
    return vals


def mass_properties():
    """Integrate non-overlapping material layers and return native mass properties."""
    _, floor = floor_pieces()
    wall = disk(OUTER_R).difference(disk(INNER_R))
    tabs = unary_union([box(-TAB_SIZE[0]/2, s*TAB_Y-TAB_SIZE[1]/2,
                                TAB_SIZE[0]/2, s*TAB_Y+TAB_SIZE[1]/2) for s in (-1, 1)])
    # Union per disjoint z slab prevents rim/tab overlap from being counted twice.
    levels = sorted({-FLOOR_T, 0.0, DEPTH, TAB_Z-TAB_SIZE[2]/2, TAB_Z+TAB_SIZE[2]/2})
    layers = []
    for za, zb in zip(levels, levels[1:]):
        zm = (za+zb)/2
        shapes = []
        if -FLOOR_T <= zm <= 0: shapes.append(floor)
        if -FLOOR_T <= zm <= DEPTH: shapes.append(wall)
        if TAB_Z-TAB_SIZE[2]/2 <= zm <= TAB_Z+TAB_SIZE[2]/2: shapes.append(tabs)
        if shapes: layers.append((unary_union(shapes), za, zb))
    volume = mz = 0.0
    raw_ixx = raw_iyy = raw_izz = 0.0
    for shape, za, zb in layers:
        geoms = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
        for p in geoms:
            a, _, _, x2, y2, _ = polygon_integrals(p)
            dz = zb-za; z2 = (zb**3-za**3)/3
            volume += a*dz; mz += a*(zb**2-za**2)/2
            raw_ixx += y2*dz+a*z2
            raw_iyy += x2*dz+a*z2
            raw_izz += (x2+y2)*dz
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("modeled solid volume must be finite and positive")
    # The contract assumes a 0.40 kg screen. Effective uniform density is
    # calibrated to that assumption, not claimed as measured polymer density.
    density = MASS_TARGET / volume
    com_z = mz / volume
    inertia = (density * (raw_ixx - volume * com_z**2),
               density * (raw_iyy - volume * com_z**2), density * raw_izz)
    if not all(math.isfinite(v) and v > 0 for v in (*inertia, MASS_TARGET)):
        raise ValueError("modeled mass and principal inertias must be positive")
    return {"mass": MASS_TARGET, "com": (0.0, 0.0, com_z),
            "inertia": inertia, "effective_density_kg_m3": density,
            "volume": volume, "layers": layers, "floor": floor}


def processing_material():
    """Frozen uncalibrated ceramic/polymer contact assumption."""
    return sim_utils.RigidBodyMaterialCfg(static_friction=0.50, dynamic_friction=0.35,
        restitution=0.05, friction_combine_mode="average")


def grasp_material():
    """Disclosed high-friction material used only by the visible orange tabs."""
    return sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.02)


def prism(path, poly, za, zb, color, semantic_label=None):
    poly = orient(poly, sign=1.0)
    coords = list(poly.exterior.coords)[:-1]; n = len(coords)
    points = [(x, y, za) for x, y in coords] + [(x, y, zb) for x, y in coords]
    faces = [list(range(n-1, -1, -1)), list(range(n, 2*n))]
    faces += [[i, (i+1)%n, (i+1)%n+n, i+n] for i in range(n)]
    mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr(0.001)
    if semantic_label is not None:
        # Isaac Sim 6 / USD 0.25 removed the legacy ``pxr.Semantics`` schema.
        # Use Isaac Lab's public helper, which authors UsdSemantics.LabelsAPI
        # and is shared by the installed spawners and segmentation utilities.
        sim_utils.add_labels(mesh.GetPrim(), labels=[semantic_label], instance_name="class")
    return mesh.GetPrim()


def spawn_screen(prim_path, cfg, translation=None, orientation=None, **_):
    root = UsdGeom.Xform.Define(sim_utils.get_current_stage(), prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    props = mass_properties(); mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(props["mass"]); mass.CreateCenterOfMassAttr(Gf.Vec3f(*props["com"]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*props["inertia"]))
    material_path = f"{prim_path}/ProcessingPhysicsMaterial"
    processing_material().func(material_path, processing_material())
    grasp_material_path = f"{prim_path}/VisibleTabGraspPhysicsMaterial"
    grasp_material().func(grasp_material_path, grasp_material())
    pieces, _ = floor_pieces()
    for i, p in enumerate(pieces):
        prism(f"{prim_path}/Floor_{i:04d}", p, -FLOOR_T, 0, (0.28,0.39,0.48), "screen")
    # Wall and tabs are authored from the exact same layer geometries integrated above.
    wall = disk(OUTER_R).difference(disk(INNER_R))
    segments = 4 * DISK_RESOLUTION
    for i in range(segments):
        a0=2*math.pi*i/segments; a1=2*math.pi*(i+1)/segments
        p=Polygon([(INNER_R*math.cos(a0),INNER_R*math.sin(a0)),
          (OUTER_R*math.cos(a0),OUTER_R*math.sin(a0)),(OUTER_R*math.cos(a1),OUTER_R*math.sin(a1)),
          (INNER_R*math.cos(a1),INNER_R*math.sin(a1))])
        prism(f"{prim_path}/Wall_{i:03d}",p,-FLOOR_T,DEPTH,(0.32,0.44,0.54), "screen")
    for s in (-1,1):
        p=box(-TAB_SIZE[0]/2,s*TAB_Y-TAB_SIZE[1]/2,TAB_SIZE[0]/2,s*TAB_Y+TAB_SIZE[1]/2)
        prism(f"{prim_path}/Tab_{'pos' if s > 0 else 'neg'}",p,TAB_Z-TAB_SIZE[2]/2,TAB_Z+TAB_SIZE[2]/2,
              (0.78,0.55,0.18), f"screen_tab_{s:+d}")
    # ``Object._generate_rigid_cfg`` requests contact activation, but a custom
    # bare SpawnerCfg is returned unchanged and has no
    # ``activate_contact_sensors`` field for Isaac Lab's spawn wrapper to act
    # on.  Activate the public contact-report schema here, after the rigid body
    # exists and before PHYSICS_READY initializes the scene-config sensors.
    sim_utils.activate_contact_sensors(prim_path)
    # Apply the frozen processing material to the floor/wall by default, then
    # override only the two visibly orange grasp tabs.  A weaker root binding
    # is essential: a stronger root would silently defeat both tab overrides.
    sim_utils.bind_physics_material(prim_path, material_path, stronger_than_descendants=False)
    for sign in ("neg", "pos"):
        sim_utils.bind_physics_material(
            f"{prim_path}/Tab_{sign}", grasp_material_path, stronger_than_descendants=True
        )
    return root.GetPrim()


def geometry_audit():
    pieces, intended = floor_pieces(); actual = unary_union(pieces)
    return {"hole_count": len(actual.interiors) if actual.geom_type == "Polygon" else sum(len(g.interiors) for g in actual.geoms), "piece_count": len(pieces),
            "solid_connected_components": 1 if actual.geom_type == "Polygon" else len(actual.geoms),
            "symmetric_difference_area_m2": float(actual.symmetric_difference(intended).area),
            "intended_solid_area_m2": float(intended.area),
            "aperture_boundary_segments": 4 * APERTURE_RESOLUTION,
            "aperture_vertex_diameter_m": APERTURE,
            "aperture_clear_diameter_m": APERTURE * math.cos(math.pi / (4 * APERTURE_RESOLUTION))}


def native_screen_material_audit(stage):
    """Resolve composed physics bindings on representative real colliders."""
    root = "/World/envs/env_0/screen"
    rows = {}
    for label, path in {
        "floor": f"{root}/Floor_0000",
        "wall": f"{root}/Wall_000",
        "tab_neg": f"{root}/Tab_neg",
        "tab_pos": f"{root}/Tab_pos",
    }.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing collider for material audit: {path}")
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if not bound:
            raise RuntimeError(f"no resolved physics material on {path}")
        mat_prim = bound.GetPrim()
        # Isaac Sim 6 / OpenUSD authors the rigid contact coefficients through
        # UsdPhysics.MaterialAPI (``physics:*``).  PhysxMaterialAPI contains
        # only PhysX-specific extensions and therefore has no friction accessors.
        physics = UsdPhysics.MaterialAPI(mat_prim)
        rows[label] = {
            "collider_path": path,
            "material_path": mat_prim.GetPath().pathString,
            "static_friction": float(physics.GetStaticFrictionAttr().Get()),
            "dynamic_friction": float(physics.GetDynamicFrictionAttr().Get()),
        }
    return rows


def screen_object():
    return Object(name="screen", object_type=ObjectType.RIGID,
      spawner_cfg=sim_utils.SpawnerCfg(func=spawn_screen),
      initial_pose=Pose(position_xyz=(SCREEN_X, SCREEN_Y, SCREEN_ROOT_Z)))


MEDIA_DIAMETER, MEDIA_LENGTH, MEDIA_DENSITY = 0.009525, 0.022225, 2500.0
WASHER_OD, WASHER_BORE, WASHER_T, WASHER_DENSITY = 0.030, 0.008, 0.004, 7850.0
RECEIVER_INNER_XY, RECEIVER_WALL_T, RECEIVER_WALL_H = 0.280, 0.006, 0.050
RECEIVER_FLOOR_T = 0.004


def ceramic_material():
    # This material is also the media/media interface.  Keep its authored
    # coefficients distinct from the 0.50/0.35 ceramic/polymer screen
    # interface; silently reusing the screen material would erase the frozen
    # 0.40/0.30 media/media constitutive assumption.
    # PhysX selects the higher-priority combine mode.  ``max`` therefore gives
    # ceramic/ceramic 0.40/0.30 and ceramic/polymer max(0.40/0.30,
    # 0.50/0.35)=0.50/0.35.
    return sim_utils.RigidBodyMaterialCfg(static_friction=0.40, dynamic_friction=0.30,
        restitution=0.05, friction_combine_mode="max")


def steel_material():
    # ``min`` has priority over the processing material's ``average`` mode,
    # yielding the frozen steel/polymer 0.45/0.35 pair.
    return sim_utils.RigidBodyMaterialCfg(static_friction=0.45, dynamic_friction=0.35,
        restitution=0.05, friction_combine_mode="min")


def _configure_dynamic_root(root, solver_position_iterations=32, solver_velocity_iterations=8):
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    api.CreateSolverPositionIterationCountAttr(solver_position_iterations)
    api.CreateSolverVelocityIterationCountAttr(solver_velocity_iterations)
    api.CreateMaxDepenetrationVelocityAttr(2.0)


def _configure_collider(prim):
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(0.00015)
    api.CreateRestOffsetAttr(0.0)


def _rigid_root(prim_path, translation, mass_value, inertia):
    root = UsdGeom.Xform.Define(sim_utils.get_current_stage(), prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(mass_value)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    _configure_dynamic_root(root)
    return root


def spawn_angle_cut_media(prim_path, cfg, translation=None, orientation=None, **_):
    """Author one free 16-sided angle-cut ceramic cylinder, long axis local X."""
    r, length, n = MEDIA_DIAMETER / 2, MEDIA_LENGTH, 16
    volume = math.pi * r * r * length
    mass_value = MEDIA_DENSITY * volume
    i_axis = 0.5 * mass_value * r * r
    i_cross = mass_value * (3 * r * r + length * length) / 12
    root = _rigid_root(prim_path, translation, mass_value, (i_axis, i_cross, i_cross))
    slope = math.tan(math.radians(15.0))
    ring = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    points = []
    for side in (-1, 1):
        for y, z in ring:
            points.append((side * length / 2 + slope * z, y, z))
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), f"{prim_path}/Collider")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    mesh.CreateSubdivisionSchemeAttr("none"); mesh.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.78, 0.69)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    _configure_collider(mesh.GetPrim())
    material_path = f"{prim_path}/CeramicPhysicsMaterial"
    ceramic_material().func(material_path, ceramic_material())
    sim_utils.bind_physics_material(mesh.GetPath().pathString, material_path, stronger_than_descendants=True)
    sim_utils.activate_contact_sensors(prim_path)
    return root.GetPrim()


def spawn_washer(prim_path, cfg, translation=None, orientation=None, **_):
    """Author one free compound annulus as 32 convex wedge colliders."""
    ro, ri, n = WASHER_OD / 2, WASHER_BORE / 2, 32
    volume = math.pi * (ro * ro - ri * ri) * WASHER_T
    mass_value = WASHER_DENSITY * volume
    iz = 0.5 * mass_value * (ro * ro + ri * ri)
    ix = mass_value * (3 * (ro * ro + ri * ri) + WASHER_T * WASHER_T) / 12
    root = _rigid_root(prim_path, translation, mass_value, (ix, ix, iz))
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        poly = Polygon([(ri*math.cos(a0),ri*math.sin(a0)), (ro*math.cos(a0),ro*math.sin(a0)),
                        (ro*math.cos(a1),ro*math.sin(a1)), (ri*math.cos(a1),ri*math.sin(a1))])
        prism(f"{prim_path}/Wedge_{i:02d}", poly, -WASHER_T/2, WASHER_T/2, (0.35,0.37,0.40), "washer")
        _configure_collider(sim_utils.get_current_stage().GetPrimAtPath(f"{prim_path}/Wedge_{i:02d}"))
    material_path = f"{prim_path}/SteelPhysicsMaterial"
    steel_material().func(material_path, steel_material())
    sim_utils.bind_physics_material(prim_path, material_path, stronger_than_descendants=True)
    sim_utils.activate_contact_sensors(prim_path)
    return root.GetPrim()


def loaded_objects(payload_kind="full"):
    assets = []
    # Transverse, separated packing over central ribs; reset is above the physical floor.
    for index, (x, y) in enumerate((
        (-.0157,-.0471),(.0157,-.0471),(-.0157,-.0157),(.0157,-.0157),
        (-.0157,.0157),(.0157,.0157),(-.0157,.0471),(.0157,.0471))):
        if payload_kind in ("media", "full"):
            assets.append(Object(name=f"media_{index:02d}", object_type=ObjectType.RIGID,
                spawner_cfg=sim_utils.SpawnerCfg(func=spawn_angle_cut_media),
                initial_pose=Pose(position_xyz=(SCREEN_X+x, SCREEN_Y+y, SCREEN_ROOT_Z+MEDIA_DIAMETER/2+0.0005))))
    for index, (x, y) in enumerate(((.130,.070),(-.130,-.070))):
        if payload_kind in ("washers", "full"):
            assets.append(Object(name=f"washer_{index:02d}", object_type=ObjectType.RIGID,
                spawner_cfg=sim_utils.SpawnerCfg(func=spawn_washer),
                initial_pose=Pose(position_xyz=(SCREEN_X+x, SCREEN_Y+y, SCREEN_ROOT_Z+WASHER_T/2+0.0005))))
    return assets


def spawn_receiver(prim_path, cfg, translation=None, orientation=None, **_):
    """Author a fixed open receiver below the screen with a measurable free volume."""
    root = UsdGeom.Xform.Define(sim_utils.get_current_stage(), prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    inner_x, inner_y = ((float(RECEIVER_INNER_XY), float(RECEIVER_INNER_XY))
                        if isinstance(RECEIVER_INNER_XY, (int, float))
                        else tuple(map(float, RECEIVER_INNER_XY)))
    half_x, half_y = inner_x / 2, inner_y / 2
    outer_x, outer_y = half_x + RECEIVER_WALL_T, half_y + RECEIVER_WALL_T
    floor = box(-outer_x, -outer_y, outer_x, outer_y)
    prism(f"{prim_path}/Floor", floor, 0.0, RECEIVER_FLOOR_T, (0.20, 0.23, 0.26), "receiver")
    walls = {
        "XNeg": box(-outer_x, -outer_y, -half_x, outer_y),
        "XPos": box(half_x, -outer_y, outer_x, outer_y),
        "YNeg": box(-half_x, -outer_y, half_x, -half_y),
        "YPos": box(-half_x, half_y, half_x, outer_y),
    }
    for name, shape in walls.items():
        prism(f"{prim_path}/Wall_{name}", shape, RECEIVER_FLOOR_T,
              RECEIVER_FLOOR_T + RECEIVER_WALL_H, (0.24, 0.28, 0.31), "receiver")
    material_path = f"{prim_path}/PhysicsMaterial"
    processing_material().func(material_path, processing_material())
    sim_utils.bind_physics_material(prim_path, material_path, stronger_than_descendants=True)
    for child in ["Floor", *[f"Wall_{name}" for name in walls]]:
        _configure_collider(sim_utils.get_current_stage().GetPrimAtPath(f"{prim_path}/{child}"))
    return root.GetPrim()


def receiver_object():
    return Object(name="receiver", object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.SpawnerCfg(func=spawn_receiver),
        initial_pose=Pose(position_xyz=(SCREEN_X, SCREEN_Y, TABLETOP_HEIGHT + 0.001)))


def spawn_support_pair(prim_path, cfg, translation=None, orientation=None, **_):
    """Author one static support group as two annulus-only round posts."""
    root = UsdGeom.Xform.Define(sim_utils.get_current_stage(), prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    material_path = f"{prim_path}/PhysicsMaterial"
    processing_material().func(material_path, processing_material())
    for index, x in enumerate((-SUPPORT_POST_X, SUPPORT_POST_X)):
        cylinder = UsdGeom.Cylinder.Define(sim_utils.get_current_stage(), f"{prim_path}/Post_{index}")
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(SUPPORT_POST_RADIUS)
        cylinder.CreateHeightAttr(SUPPORT_H)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.25, 0.28)])
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim()).CreateContactOffsetAttr(0.001)
    sim_utils.bind_physics_material(prim_path, material_path, stronger_than_descendants=True)
    return root.GetPrim()


def fixture_assets():
    return [Object(name=f"support_{'pos' if s > 0 else 'neg'}", object_type=ObjectType.BASE,
      spawner_cfg=sim_utils.SpawnerCfg(func=spawn_support_pair),
      initial_pose=Pose(position_xyz=(SCREEN_X, s*SUPPORT_POST_Y,
                                      TABLETOP_HEIGHT + SUPPORT_H/2))) for s in (-1,1)]


def native_support_geometry_audit(stage):
    """Measure post clearance from the perforated disk from composed USD geometry."""
    clearances = []
    post_rows = []
    for sign in ("neg", "pos"):
        for index in range(2):
            path = f"/World/envs/env_0/support_{sign}/Post_{index}"
            cylinder = UsdGeom.Cylinder.Get(stage, path)
            if not cylinder:
                raise RuntimeError(f"missing authored support post: {path}")
            radius = float(cylinder.GetRadiusAttr().Get())
            matrix = UsdGeom.Xformable(cylinder.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            center = matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
            radial = math.hypot(float(center[0]) - SCREEN_X, float(center[1]) - SCREEN_Y)
            clearance = radial - radius - PERF_R
            clearances.append(clearance)
            post_rows.append({"path": path, "center_w_m": [float(v) for v in center],
                              "radius_m": radius, "perforated_region_clearance_m": clearance})
    return {"posts": post_rows, "minimum_perforated_region_clearance_m": min(clearances),
            "post_count": len(post_rows)}


def build_environment(include_loaded=False, payload_kind="full", enable_cameras=True):
    embodiment=BimanualYamEmbodiment(enable_cameras=enable_cameras, overview_cam=enable_cameras,
      # Keep the embodiment's verified robot-wide contact sensors enabled.
      # Their spawn configuration activates contact reporting on the YAM link
      # bodies; the four filtered sensors below then measure screen-only pairs.
      contact_sensors=True, robot_contact_offset=0.002, rig=TABLETOP_RIG)
    # Contact sensors must be part of the scene config before Arena constructs
    # the InteractiveScene.  Sensors created after make_registered() can miss
    # the one-shot PHYSICS_READY event and remain permanently uninitialized.
    scene_fields=[(f.name, f.type, getattr(embodiment.scene_config, f.name))
                  for f in dataclass_fields(embodiment.scene_config)]
    links=asset_link_paths()
    for arm_key, arm_prim in (("left", "LeftArm"), ("right", "RightArm")):
        for tip in ("tip_left", "tip_right"):
            name=f"{arm_key}_{tip}_screen_contact"
            cfg=ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/"+arm_prim+links[tip],
                update_period=0.0, history_length=0, track_air_time=False,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/screen"],
                max_contact_data_count_per_prim=RAW_CONTACT_CAPACITY_PER_PRIM)
            scene_fields.append((name, ContactSensorCfg, cfg))
    # Observe the two support groups independently.  This sensor is authored
    # before PHYSICS_READY for the same lifecycle reason as the fingertip
    # sensors above; post-hoc sensors can remain uninitialized in this runtime.
    scene_fields.append(("screen_support_contacts", ContactSensorCfg,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/screen",
            update_period=0.0, history_length=0, track_air_time=False,
            filter_prim_paths_expr=[
                # The support roots are grouping Xforms, not collision shapes.
                # Bind every actual Post collider explicitly.  The runner sums
                # each adjacent pair into one independently measured side.
                "{ENV_REGEX_NS}/support_neg/Post_0",
                "{ENV_REGEX_NS}/support_neg/Post_1",
                "{ENV_REGEX_NS}/support_pos/Post_0",
                "{ENV_REGEX_NS}/support_pos/Post_1"],
            max_contact_data_count_per_prim=RAW_CONTACT_CAPACITY_PER_PRIM)))
    if include_loaded:
        receiver_shapes = ["{ENV_REGEX_NS}/receiver/Floor", *[
            f"{{ENV_REGEX_NS}}/receiver/Wall_{side}"
            for side in ("XNeg", "XPos", "YNeg", "YPos")]]
        support_shapes = [
            f"{{ENV_REGEX_NS}}/support_{sign}/Post_{index}"
            for sign in ("neg", "pos") for index in range(2)]
        # These discovery sensors are declared before PHYSICS_READY.  The
        # screen sensor directly exposes unintended receiver support; the ten
        # loose-body sensors expose reset contacts against screen, receiver,
        # and supports without treating a 20 Hz endpoint as a penetration
        # qualification.
        scene_fields.append(("screen_fixture_contacts", ContactSensorCfg,
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/screen", update_period=0.0,
                history_length=0, track_air_time=False,
                filter_prim_paths_expr=[*receiver_shapes, *support_shapes],
                max_contact_data_count_per_prim=RAW_CONTACT_CAPACITY_PER_PRIM)))
        payload_names = ([f"media_{i:02d}" for i in range(8)] if payload_kind in ("media", "full") else [])
        payload_names += ([f"washer_{i:02d}" for i in range(2)] if payload_kind in ("washers", "full") else [])
        for name in payload_names:
            for category, filters in (("screen", ["{ENV_REGEX_NS}/screen"]),
                                      ("receiver", receiver_shapes),
                                      ("support", support_shapes)):
                scene_fields.append((f"{name}_{category}_contacts", ContactSensorCfg,
                    ContactSensorCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/{name}", update_period=0.0,
                        history_length=0, track_air_time=False,
                        filter_prim_paths_expr=filters,
                        max_contact_data_count_per_prim=RAW_CONTACT_CAPACITY_PER_PRIM)))
    embodiment.scene_config=make_configclass("FullScreenYamSceneCfg", scene_fields)()
    # This first manipulation component records RGB video for visual inspection.
    # Object-identity visibility remains unqualified until the full-scene stage.
    if enable_cameras:
        embodiment.camera_config.overview_cam.data_types = ["rgb"]
    return IsaacLabArenaEnvironment(name="full_screen_yam_preflight",
      scene=Scene([*scene_assets(objects=False,d435_collider=False),*fixture_assets(),screen_object(),
                   *([receiver_object(), *loaded_objects(payload_kind)] if include_loaded else [])]),
      embodiment=embodiment, env_cfg_callback=env_cfg_callback(CONTROL_HZ))


def build_loaded_environment():
    """Same public YAM scene with the frozen 8-media/2-washer free rigid batch."""
    environment = build_environment(include_loaded=True)
    environment.name = "loaded_screen_reset_preflight"
    return environment
