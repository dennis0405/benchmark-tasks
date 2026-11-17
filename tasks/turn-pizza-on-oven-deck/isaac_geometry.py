"""Generate the full-scale oven, peel and compliant set-crust links for native PhysX."""

import math
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt
from scipy.spatial import ConvexHull

from layout import MOUTH, OVEN_Y, STONE_Z, PARK, GRIP_DISTANCE, HANDLE_Z


def v(x):
    return Gf.Vec3f(*[float(i) for i in x])


def xf(stage, path, pos, mass=None, kinematic=False):
    g = UsdGeom.Xform.Define(stage, path)
    g.AddTranslateOp().Set(Gf.Vec3d(*[float(i) for i in pos]))
    if mass is not None:
        rb = UsdPhysics.RigidBodyAPI.Apply(g.GetPrim())
        rb.CreateKinematicEnabledAttr(kinematic)
        UsdPhysics.MassAPI.Apply(g.GetPrim()).CreateMassAttr(float(mass))
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(g.GetPrim())
        px.CreateSolverPositionIterationCountAttr(32)
        px.CreateSolverVelocityIterationCountAttr(8)
        px.CreateMaxDepenetrationVelocityAttr(0.35)
        PhysxSchema.PhysxContactReportAPI.Apply(g.GetPrim()).CreateThresholdAttr(0.0)
    return g.GetPrim()


def mat(stage, name, friction):
    path = "/World/Looks/" + name
    material = UsdShade.Material.Define(stage, path)
    p = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    p.CreateStaticFrictionAttr(friction)
    p.CreateDynamicFrictionAttr(friction * 0.95)
    p.CreateRestitutionAttr(0.0)
    return material


def finish(g, color, material):
    g.CreateDisplayColorAttr([v(color)])
    if material:
        prim = g.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(0.00015)
        px.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    return g


def box(stage, path, centre, extent, color, material=None):
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set(Gf.Vec3d(*centre))
    g.AddScaleOp().Set(Gf.Vec3f(*extent))
    return finish(g, color, material)


def ball(stage, path, local, r, color, material=None):
    g = UsdGeom.Sphere.Define(stage, path)
    g.CreateRadiusAttr(r)
    g.AddTranslateOp().Set(Gf.Vec3d(*local))
    return finish(g, color, material)


def hull(stage, path, pts, color, material=None):
    pts = np.asarray(pts, dtype=np.float32)
    h = ConvexHull(pts)
    tri = h.simplices.copy()
    for i, idx in enumerate(tri):
        a, b, c = pts[idx]
        if np.dot(np.cross(b - a, c - a), h.equations[i, :3]) < 0:
            tri[i] = idx[[0, 2, 1]]
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.flatten().astype(np.int32)))
    g.CreateFaceVertexCountsAttr([3] * len(tri))
    g.CreateSubdivisionSchemeAttr("none")
    finish(g, color, material)
    if material:
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr(
            "convexHull"
        )
    return g


def rings(rs, n=48):
    return [
        (r * math.cos(a), r * math.sin(a), z)
        for r, z in rs
        for a in np.arange(n) * math.tau / n
    ]


def joint(stage, name, kind, child, anchor, axis, yaw_deg, limits, k, d, core):
    cls = UsdPhysics.PrismaticJoint if kind == "linear" else UsdPhysics.RevoluteJoint
    j = cls.Define(stage, "/World/Joints/" + name)
    j.CreateBody0Rel().SetTargets([core])
    j.CreateBody1Rel().SetTargets([child])
    j.CreateLocalPos0Attr(v(anchor))
    j.CreateLocalPos1Attr(v((0, 0, 0)))
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuat()
    j.CreateLocalRot0Attr(Gf.Quatf(rot))
    j.CreateLocalRot1Attr(Gf.Quatf(rot))
    j.CreateAxisAttr(axis)
    j.CreateLowerLimitAttr(limits[0])
    j.CreateUpperLimitAttr(limits[1])
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(k)
    drive.CreateDampingAttr(d)
    drive.CreateTargetPositionAttr(0.0)


def build_station(stage, base_path="/World/Oven/base"):
    stone = mat(stage, "Stone", 0.37)
    grey = (0.29, 0.26, 0.23)
    xf(stage, "/World/Stone", (MOUTH + 0.24, OVEN_Y, STONE_Z - 0.02), 1.0, True)
    box(stage, "/World/Stone/deck", (0, 0, 0), (0.48, 0.78, 0.04), grey, stone)
    for name, ctr, sz in [
        ("left", (MOUTH + 0.24, OVEN_Y + 0.40, 1.065), (0.48, 0.02, 0.13)),
        ("right", (MOUTH + 0.24, OVEN_Y - 0.40, 1.065), (0.48, 0.02, 0.13)),
        ("back", (MOUTH + 0.49, OVEN_Y, 1.065), (0.02, 0.78, 0.13)),
        ("roof", (MOUTH + 0.24, OVEN_Y, 1.135), (0.48, 0.82, 0.02)),
    ]:
        g = box(stage, "/World/Oven/" + name, ctr, sz, (0.18, 0.17, 0.16), stone)
        if name in ("roof", "right"):
            g.MakeInvisible()  # camera cutaway; collider retained
    box(
        stage,
        base_path,
        (MOUTH + 0.24, OVEN_Y, STONE_Z - 0.07),
        (0.50, 0.82, 0.10),
        (0.15, 0.16, 0.17),
    )
    for j, (ox, oy) in enumerate(
        [(0.04, -0.19), (0.44, -0.19), (0.04, 0.19), (0.44, 0.19)]
    ):
        box(
            stage,
            f"/World/Oven/leg{j}",
            (MOUTH + ox, OVEN_Y + oy, 0.82),
            (0.032, 0.032, 0.16),
            (0.17, 0.18, 0.18),
        )
    box(
        stage,
        "/World/Counter",
        (0.35, 0, 0.725),
        (0.8, 1.2, 0.05),
        (0.35, 0.28, 0.21),
        stone,
    )
    # An exterior landing shelf, flush with the stone, supports the long blade
    # while the robot repositions outside the mouth. The pizza must stay inside.
    box(
        stage,
        "/World/Stand/front",
        (MOUTH - 0.073, OVEN_Y, STONE_Z - 0.018),
        (0.146, 0.78, 0.036),
        grey,
        stone,
    )
    for n, x, z_top in [("rear", PARK[0] - 0.34, PARK[2] + HANDLE_Z - 0.009)]:
        box(
            stage,
            "/World/Stand/" + n,
            (x, PARK[1], (0.75 + z_top) / 2),
            (0.055, 0.048, z_top - 0.75),
            (0.25, 0.23, 0.21),
            stone,
        )
    # Shallow retaining lips receive the 16 mm shaft with 7 mm free play
    # on either side. The robot can lift the shaft straight out.
    for side in (-1, 1):
        box(
            stage,
            "/World/Stand/lip" + str(side + 1),
            (
                PARK[0] - 0.34,
                PARK[1] + side * 0.018,
                PARK[2] + HANDLE_Z - 0.009 + 0.0025,
            ),
            (0.050, 0.006, 0.005),
            (0.32, 0.28, 0.22),
            stone,
        )


def build_peel(stage, peel_path="/World/Peel"):
    metal = mat(stage, "Aluminum", 0.56)
    grip = mat(stage, "Grip", 1.15)
    steel = (0.6, 0.78, 0.68)
    tool = xf(stage, peel_path, PARK, 0.40)
    props = UsdPhysics.MassAPI(tool)
    props.CreateCenterOfMassAttr(v((-0.285, 0, 0.009)))
    props.CreateDiagonalInertiaAttr(v((0.00045, 0.017, 0.017)))
    props.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    hull(
        stage,
        peel_path + "/blade",
        rings([(0.076, -0.0008), (0.085, 0), (0.075, 0.0008)]),
        steel,
    )
    # Thin cooked convex meshes can lose bevel detail. Inscribed rectangular
    # strips give the native solver a finite top and bottom face at every slice.
    for j in range(24):
        a = -0.085 + 0.17 * j / 24
        b = a + 0.17 / 24
        radius = max(abs(a), abs(b))
        half = math.sqrt(max(0.000003, 0.085**2 - radius**2))
        thickness = 0.0016 * min(1.0, max(0.15, (0.085 - abs((a + b) / 2)) / 0.009))
        g = box(
            stage,
            peel_path + f"/strip{j}",
            ((a + b) / 2, 0.0, 0.0),
            (b - a, 2 * half, thickness),
            steel,
            metal,
        )
        g.MakeInvisible()
    box(
        stage,
        peel_path + "/shaft",
        (-0.365, 0, HANDLE_Z),
        (0.58, 0.016, 0.017),
        steel,
        metal,
    )
    box(
        stage,
        peel_path + "/handle",
        (-GRIP_DISTANCE, 0, HANDLE_Z),
        (0.16, 0.028, 0.024),
        (0.16, 0.17, 0.16),
        grip,
    )
    # The angled neck joins the offset shaft to the low blade.
    hull(
        stage,
        peel_path + "/neck",
        [
            (x, y, z + dz)
            for x, z in [(-0.115, HANDLE_Z), (-0.055, 0.0005)]
            for y in [-0.012, 0.012]
            for dz in [-0.001, 0.001]
        ],
        steel,
        metal,
    )


def build_food(stage, core_path="/World/Food/core"):
    bread = mat(stage, "Crust", 0.44)
    tan = (0.82, 0.45, 0.16)
    p0 = np.array([MOUTH + 0.23, OVEN_Y, STONE_Z + 0.002])
    bodies = [core_path]
    xf(stage, bodies[0], p0, 0.181)
    hull(
        stage,
        bodies[0] + "/base",
        rings([(0.101, 0.0005), (0.105, 0.002), (0.105, 0.015)]),
        tan,
        bread,
    )
    hull(
        stage,
        bodies[0] + "/sauce",
        rings([(0.098, 0.015), (0.097, 0.016)]),
        (0.78, 0.23, 0.065),
    )
    for j, (x, y) in enumerate(
        [(0, 0.063), (-0.055, -0.025), (0.044, -0.04), (0.035, 0.017), (-0.04, 0.035)]
    ):
        pts = np.array(rings([(0.016, 0.016), (0.016, 0.017)], 16)) + [x, y, 0]
        hull(
            stage,
            bodies[0] + f"/topping{j}",
            pts,
            (0.55, 0.085, 0.045) if j else (0.17, 0.40, 0.08),
        )
    for j in range(16):
        a = j * math.tau / 16
        radius = 0.105
        anchor = np.array([radius * math.cos(a), radius * math.sin(a), 0.008])
        path = f"/World/Food/rim{j:02d}"
        xf(stage, path, p0 + anchor, 0.09 / 16)
        bodies.append(path)
        pts = [
            (
                r * math.cos(a + b) - anchor[0],
                r * math.sin(a + b) - anchor[1],
                z - 0.008,
            )
            for r, z in [
                (radius, 0.001),
                (0.146, 0.0007),
                (0.15, 0.003),
                (0.15, 0.012),
                (0.144, 0.018),
                (radius, 0.015),
            ]
            for b in [-math.pi / 16, 0, math.pi / 16]
        ]
        hull(stage, path + "/crust", pts, tan if j % 5 else (0.63, 0.30, 0.10), bread)
        joint(
            stage,
            f"bend{j}",
            "angular",
            path,
            anchor,
            "Y",
            math.degrees(a),
            [-12, 12],
            0.22 * math.pi / 180,
            0.0008 * math.pi / 180,
            core_path,
        )
    for j in range(12):
        a = (j + 0.5) * math.tau / 12
        r = 0.062
        anchor = np.array([r * math.cos(a), r * math.sin(a), 0.005])
        path = f"/World/Food/pad{j:02d}"
        xf(stage, path, p0 + anchor, 0.001)
        bodies.append(path)
        ball(stage, path + "/patch", (0.0, 0.0, -0.0038), 0.005, tan, bread)
        joint(
            stage,
            f"compress{j}",
            "linear",
            path,
            anchor,
            "Z",
            0,
            [0, 0.002],
            800.0,
            1.3,
            core_path,
        )
    # Adjacent compliant links are parts of the same continuous food.
    for i, path in enumerate(bodies[:-1]):
        UsdPhysics.FilteredPairsAPI.Apply(
            stage.GetPrimAtPath(path)
        ).CreateFilteredPairsRel().SetTargets(bodies[i + 1 :])
    return bodies


def build(stage):
    build_station(stage)
    build_peel(stage)
    return build_food(stage)
