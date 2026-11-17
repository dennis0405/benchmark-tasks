"""A steel depth stop on the hex driver leaves clearance above the socket floor."""

import math


def add_depth_stop(stage, root_path):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    root = stage.GetPrimAtPath(root_path)
    start, height, radius = 0.00475, 0.002, 0.00325
    center = start + height / 2
    # A convex prism avoids native thin-cylinder contact approximations.
    sides = 24
    collar = UsdGeom.Mesh.Define(stage, root_path + "/DepthStop")
    collar.CreateSubdivisionSchemeAttr("none")
    collar.CreatePointsAttr(
        [
            (
                radius * math.cos(math.tau * i / sides),
                radius * math.sin(math.tau * i / sides),
                z,
            )
            for z in (start, start + height)
            for i in range(sides)
        ]
    )
    faces = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j, sides + i])
    collar.CreateFaceVertexCountsAttr([len(f) for f in faces])
    collar.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    cp = collar.GetPrim()
    UsdPhysics.CollisionAPI.Apply(cp)
    UsdPhysics.MeshCollisionAPI.Apply(cp).CreateApproximationAttr("convexHull")
    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(cp).CreateMinThicknessAttr(0.00005)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(cp)
    contact.CreateContactOffsetAttr(0.000025)
    contact.CreateRestOffsetAttr(0)
    mat = UsdShade.Material.Get(stage, root_path + "/Looks/Steel")
    UsdShade.MaterialBindingAPI.Apply(cp).Bind(mat)

    # The original solid hex bit already occupies the bore of the collar.
    # Subtract its area and polar moment before shifting the combined inertia.
    inner_apothem = 0.00395 / 2
    side = inner_apothem / math.cos(math.pi / 6)
    hex_area = 3 * side * inner_apothem
    circle_area = sides * radius**2 * math.sin(math.tau / sides) / 2
    density = 7850
    extra = density * height * (circle_area - hex_area)
    polar = (
        density
        * height
        * (
            circle_area * radius**2 * (2 + math.cos(math.tau / sides)) / 6
            - hex_area * 5 * side**2 / 12
        )
    )
    mass_api = UsdPhysics.MassAPI(root)
    original = mass_api.GetMassAttr().Get()
    old_z = mass_api.GetCenterOfMassAttr().Get()[2]
    new_z = (original * old_z + extra * center) / (original + extra)
    transverse = (
        polar / 2
        + extra * height**2 / 12
        + original * (old_z - new_z) ** 2
        + extra * (center - new_z) ** 2
    )
    ix, iy, iz = mass_api.GetDiagonalInertiaAttr().Get()
    mass_api.GetMassAttr().Set(original + extra)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0, 0, new_z))
    mass_api.GetDiagonalInertiaAttr().Set(
        Gf.Vec3f(ix + transverse, iy + transverse, iz + polar)
    )
    return original + extra
