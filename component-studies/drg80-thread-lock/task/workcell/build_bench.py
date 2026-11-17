"""Author the instrumented workcell from its original compound collider assets."""

from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema


def continuous_ring(stage, parent, prefix):
    """Weld coincident vertices and omit the radial faces inside the solid."""
    from collections import Counter

    root = stage.GetPrimAtPath(parent)
    sectors = [p for p in root.GetChildren() if p.GetName().startswith(prefix)]
    if len(sectors) != 24:
        raise ValueError(f"expected 24 ring sectors under {parent}")
    points = []
    ids = {}
    exterior = {}
    for prim in sectors:
        mesh = UsdGeom.Mesh(prim)
        local = []
        for coord in mesh.GetPointsAttr().Get():
            key = tuple(round(v, 8) for v in coord)
            if key not in ids:
                ids[key] = len(points)
                points.append(coord)
            local.append(ids[key])
        indices = iter(mesh.GetFaceVertexIndicesAttr().Get())
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = tuple(local[next(indices)] for _ in range(count))
            key = tuple(sorted(face))
            if key in exterior:
                del exterior[key]
            else:
                exterior[key] = face
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    edges = Counter(
        tuple(sorted((a, b)))
        for face in exterior.values()
        for a, b in zip(face, face[1:] + face[:1])
    )
    if not edges or set(edges.values()) != {2}:
        raise ValueError("welded boundary is not closed")
    out = UsdGeom.Mesh.Define(stage, parent + "/BearingBoundary")
    out.CreatePointsAttr(points)
    out.CreateFaceVertexCountsAttr([len(f) for f in exterior.values()])
    out.CreateFaceVertexIndicesAttr([v for face in exterior.values() for v in face])
    out.CreateSubdivisionSchemeAttr("none")
    out.CreateVisibilityAttr("invisible")
    obj = out.GetPrim()
    UsdPhysics.CollisionAPI.Apply(obj)
    props = PhysxSchema.PhysxCollisionAPI.Apply(obj)
    props.CreateContactOffsetAttr(0.000025)
    props.CreateRestOffsetAttr(0.0)
    UsdPhysics.MeshCollisionAPI.Apply(obj).CreateApproximationAttr("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(obj)
    sdf.CreateSdfResolutionAttr(384)
    sdf.CreateSdfSubgridResolutionAttr(6)
    sdf.CreateSdfBitsPerSubgridPixelAttr("BitsPerPixel16")


def build(
    destination, assets, instrumented=False, velocity_iters=4, continuous_seat=False
):
    stage = Usd.Stage.CreateNew(str(destination))
    root = UsdGeom.Xform.Define(stage, "/Bench")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    api = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    api.CreateEnabledSelfCollisionsAttr(True)
    api.CreateSleepThresholdAttr(0.0)
    api.CreateSolverPositionIterationCountAttr(128)
    api.CreateSolverVelocityIterationCountAttr(velocity_iters)
    origin = (0.36, 0, 0.75)
    positions = {}

    def body(name, pos, source=None):
        positions[name] = tuple(a + b for a, b in zip(origin, pos))
        x = UsdGeom.Xform.Define(stage, "/Bench/" + name)
        if source:
            x.GetPrim().GetReferences().AddReference(str((assets / source).resolve()))
        else:
            UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            m = UsdPhysics.MassAPI.Apply(x.GetPrim())
            m.CreateMassAttr(1e-6)
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-10))
        x.AddTranslateOp().Set(Gf.Vec3d(*positions[name]))
        PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr(0)
        return x

    frame = body("frame", (0, 0, 0), "fixture/parts/frame.usda")
    for name, pos, src in (
        ("s_car", (0, -0.018, 0.1017), None),
        ("screw", (0, -0.018, 0.1017), "tools/socket_adjuster.usda"),
        ("n_car", (0, -0.018, 0.093), None),
        ("nut", (0, -0.018, 0.093), "tools/thin_nut.usda"),
        ("retainer", (0, -0.018, 0.0997), "tools/thin_nut.usda"),
        ("piston", (0, -0.018, 0.072), "fixture/parts/piston.usda"),
        ("left_jaw", (-0.030, 0.022, 0.095), "fixture/parts/left_jaw.usda"),
        ("right_jaw", (0.030, 0.022, 0.095), "fixture/parts/right_jaw.usda"),
        ("lever", (-0.07, -0.025, 0.07), "fixture/parts/lever.usda"),
    ):
        body(name, pos, src)

    def joint(name, cls, parent, child, axis=None, limits=None):
        j = cls.Define(stage, "/Bench/" + name)
        a = positions[child]
        if parent:
            j.CreateBody0Rel().SetTargets(["/Bench/" + parent])
            a = tuple(u - v for u, v in zip(a, positions[parent]))
        j.CreateBody1Rel().SetTargets(["/Bench/" + child])
        j.CreateLocalPos0Attr(Gf.Vec3f(*a))
        j.CreateLocalPos1Attr(Gf.Vec3f(0))
        if axis:
            j.CreateAxisAttr(axis)
        if limits:
            j.CreateLowerLimitAttr(limits[0])
            j.CreateUpperLimitAttr(limits[1])
        return j

    fixed = UsdPhysics.FixedJoint
    slide = UsdPhysics.PrismaticJoint
    hinge = UsdPhysics.RevoluteJoint
    mount = joint("mount", fixed, None, "frame")
    mount.CreateBreakForceAttr(150.0)
    mount.CreateBreakTorqueAttr(15.0)
    joint("sz", slide, "frame", "s_car", "Z", (-0.0048, 0.0048))
    joint("st", hinge, "s_car", "screw", "Z", (-1800.0, 1800.0))
    joint("nz", slide, "screw", "n_car", "Z", (-0.005, 0.005))
    joint("nt", hinge, "n_car", "nut", "Z", (-2160.0, 2160.0))
    if instrumented:
        for joint_name in ("st", "nt"):
            drive = UsdPhysics.DriveAPI.Apply(
                stage.GetPrimAtPath("/Bench/" + joint_name), "angular"
            )
            drive.CreateTypeAttr("force")
            drive.CreateMaxForceAttr(0.06)
            drive.CreateStiffnessAttr(0.01)
            drive.CreateDampingAttr(0.001)
            drive.CreateTargetPositionAttr(0.0)
    joint("retain", fixed, "screw", "retainer")
    piston = joint("p", slide, "frame", "piston", "Z", (0.0, 0.010297442586766544))
    left = joint("jl", hinge, "frame", "left_jaw", "Y", (-90.0, 0.0))
    right = joint("jr", hinge, "frame", "right_jaw", "Y", (0.0, 90.0))
    # Both jaw pivots and the piston participate in the same native constraint
    # island, including when the piston contacts the adjustable screw.
    import math

    for jaw, sign in ((left, 1), (right, -1)):
        mimic = PhysxSchema.PhysxMimicJointAPI.Apply(jaw.GetPrim(), "rotY")
        mimic.CreateReferenceJointRel().SetTargets([piston.GetPath()])
        mimic.CreateGearingAttr(sign * math.degrees(150))
        mimic.CreateOffsetAttr(0.0)
    flow = UsdPhysics.DriveAPI.Apply(piston.GetPrim(), "linear")
    flow.CreateTypeAttr("force")
    flow.CreateDampingAttr(350.0)
    flow.CreateStiffnessAttr(0.0)
    flow.CreateTargetVelocityAttr(0.0)
    joint("valve", hinge, "frame", "lever", "Y", (-35.0, 35.0))
    # Seat faces share the benchmark's lubricated metal bearing material.
    if continuous_seat:
        continuous_ring(stage, "/Bench/frame", "Cover")
        continuous_ring(stage, "/Bench/nut", "Sector")
    mat = UsdShade.Material.Define(stage, "/Bench/SeatMaterial")
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(0.08)
    phys.CreateDynamicFrictionAttr(0.05)
    phys.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr(
        "min"
    )
    for who in ("frame", "nut"):
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/Bench/" + who)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    mat, materialPurpose="physics"
                )
    stage.GetRootLayer().Save()
