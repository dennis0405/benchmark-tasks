"""Import-safe builders and geometry for the full-screen YAM preflight.

This module deliberately has no argument parsing or AppLauncher side effects.
The same polygons returned by :func:`floor_pieces` are used for both USD
collision authoring and the offline Shapely audit.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from pxr import Gf, PhysxSchema, Semantics, UsdGeom, UsdPhysics, Vt
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from yam_arena.embodiment import BimanualYamEmbodiment
from yam_arena.rig import TABLETOP as TABLETOP_RIG
from yam_arena.scene import env_cfg_callback, scene_assets

APERTURE, RIB, PITCH = 0.0127, 0.003, 0.0157
PERF_R, INNER_R, OUTER_R = 0.150, 0.18115, 0.18415
FLOOR_T, DEPTH, DENSITY = 0.003, 0.0635, 1100.0
TABLETOP_HEIGHT, SUPPORT_H = 0.75, 0.010
SCREEN_X, SCREEN_Y = 0.37, 0.0
SUPPORT_TOP = TABLETOP_HEIGHT + 0.055
SCREEN_ROOT_Z = SUPPORT_TOP + FLOOR_T
CONTROL_HZ = 20
DISK_RESOLUTION = 96
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
    """Non-overlapping convex polygons whose union is the perforated disk."""
    boundary = disk(INNER_R)
    hole_union = unary_union([box(x-APERTURE/2, y-APERTURE/2,
                                  x+APERTURE/2, y+APERTURE/2) for x, y in HOLES])
    solid = boundary.difference(hole_union)
    # Grid lines at every hole edge partition the disk into rectangles.  Each
    # rectangle/disk intersection is convex because both operands are convex.
    cuts = sorted({-INNER_R, INNER_R, *[x+s*APERTURE/2 for x, _ in HOLES for s in (-1, 1)]})
    ycuts = sorted({-INNER_R, INNER_R, *[y+s*APERTURE/2 for _, y in HOLES for s in (-1, 1)]})
    pieces = []
    for xa, xb in zip(cuts, cuts[1:]):
        for ya, yb in zip(ycuts, ycuts[1:]):
            p = box(xa, ya, xb, yb).intersection(solid)
            if p.area > 1e-12:
                if p.geom_type == "Polygon": pieces.append(p)
                else: pieces.extend(g for g in p.geoms if g.area > 1e-12)
    return pieces, solid


def polygon_integrals(poly):
    """Return A, integral(x), integral(y), integral(x^2), integral(y^2), integral(xy)."""
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
        if 0 <= zm <= DEPTH: shapes.append(wall)
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
    com_z = mz/volume
    mass = DENSITY*volume
    return {"mass": mass, "com": (0.0, 0.0, com_z),
            "inertia": (DENSITY*(raw_ixx-volume*com_z**2),
                        DENSITY*(raw_iyy-volume*com_z**2), DENSITY*raw_izz),
            "volume": volume, "layers": layers, "floor": floor}


def material():
    return sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.02)


def prism(path, poly, za, zb, color, semantic_label=None):
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
        semantics = Semantics.SemanticsAPI.Apply(mesh.GetPrim(), "Semantics")
        semantics.CreateSemanticTypeAttr().Set("class")
        semantics.CreateSemanticDataAttr().Set(semantic_label)
    return mesh.GetPrim()


def spawn_screen(prim_path, cfg, translation=None, orientation=None, **_):
    root = UsdGeom.Xform.Define(sim_utils.get_current_stage(), prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    props = mass_properties(); mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(props["mass"]); mass.CreateCenterOfMassAttr(Gf.Vec3f(*props["com"]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*props["inertia"]))
    material_path = f"{prim_path}/PhysicsMaterial"
    material().func(material_path, material())
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
        prism(f"{prim_path}/Wall_{i:03d}",p,0,DEPTH,(0.32,0.44,0.54), "screen")
    for s in (-1,1):
        p=box(-TAB_SIZE[0]/2,s*TAB_Y-TAB_SIZE[1]/2,TAB_SIZE[0]/2,s*TAB_Y+TAB_SIZE[1]/2)
        prism(f"{prim_path}/Tab_{s:+d}",p,TAB_Z-TAB_SIZE[2]/2,TAB_Z+TAB_SIZE[2]/2,
              (0.78,0.55,0.18), f"screen_tab_{s:+d}")
    # Bind the authored high-friction material to every collision descendant.
    # Binding only the fixture material leaves the actual tab grasp dependent
    # on the simulator default, so the root binding is deliberately stronger
    # than any descendant binding.
    sim_utils.bind_physics_material(prim_path, material_path, stronger_than_descendants=True)
    return root.GetPrim()


def geometry_audit():
    pieces, intended = floor_pieces(); actual = unary_union(pieces)
    return {"hole_count": len(HOLES), "piece_count": len(pieces),
            "solid_connected_components": 1 if actual.geom_type == "Polygon" else len(actual.geoms),
            "symmetric_difference_area_m2": float(actual.symmetric_difference(intended).area),
            "intended_solid_area_m2": float(intended.area)}


def screen_object():
    return Object(name="screen", object_type=ObjectType.RIGID,
      spawner_cfg=sim_utils.SpawnerCfg(func=spawn_screen),
      initial_pose=Pose(position_xyz=(SCREEN_X, SCREEN_Y, SCREEN_ROOT_Z)))


def fixture_assets():
    return [Object(name=f"support_{s:+d}", object_type=ObjectType.BASE,
      spawner_cfg=sim_utils.CuboidCfg(size=(0.52,0.015,SUPPORT_H),
        collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=material()),
      initial_pose=Pose(position_xyz=(SCREEN_X,s*0.10,SUPPORT_TOP-SUPPORT_H/2))) for s in (-1,1)]


def build_environment():
    embodiment=BimanualYamEmbodiment(enable_cameras=True, overview_cam=True,
      robot_contact_offset=0.002, rig=TABLETOP_RIG)
    # Raw semantic IDs plus idToLabels make visibility object-identity-grounded;
    # RGB color alone is deliberately not used as evidence for screen visibility.
    embodiment.camera_config.overview_cam.data_types = ["rgb", "semantic_segmentation"]
    embodiment.camera_config.overview_cam.renderer_cfg.colorize_semantic_segmentation = False
    return IsaacLabArenaEnvironment(name="full_screen_yam_preflight",
      scene=Scene([*scene_assets(objects=False,d435_collider=False),*fixture_assets(),screen_object()]),
      embodiment=embodiment, env_cfg_callback=env_cfg_callback(CONTROL_HZ))
