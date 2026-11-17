"""A planar lead-in on the two outer ends of the open wrench jaws."""

from math import cos, pi
import numpy as np
from scipy.spatial.transform import Rotation


def author_wrench_guide(stage):
    from pxr import UsdGeom, UsdPhysics, PhysxSchema

    for name, sign in (("UpperJaw", 1), ("LowerJaw", -1)):
        previous = stage.GetPrimAtPath("/Tool/" + name)
        UsdPhysics.CollisionAPI(previous).CreateCollisionEnabledAttr(False)
        UsdGeom.Imageable(previous).CreateVisibilityAttr("invisible")
        outline = [
            (-0.0085, 0.00665),
            (0.0085, 0.00665),
            (0.0135, 0.00785),
            (0.0135, 0.01115),
            (-0.0085, 0.01115),
        ]
        if sign < 0:
            outline = [(x, -y) for x, y in reversed(outline)]
        vertices = [(x, y, z) for z in (-0.002, 0.002) for x, y in outline]
        faces = [list(range(4, -1, -1)), list(range(5, 10))]
        faces.extend([i, (i + 1) % 5, (i + 1) % 5 + 5, i + 5] for i in range(5))
        mesh = UsdGeom.Mesh.Define(stage, "/Tool/Guided" + name)
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        mesh.CreateDisplayColorAttr(UsdGeom.Gprim(previous).GetDisplayColorAttr().Get())
        mesh.CreateSubdivisionSchemeAttr("none")
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr(0.000025)
        contact.CreateRestOffsetAttr(0.0)


def fit_wrench(pos, rot, nut_pose):
    axis = Rotation.from_quat(nut_pose[3:]).as_matrix()
    a = np.arange(6) * pi / 3
    corners = np.array(
        [
            (x, y, z)
            for x, y in zip(
                0.0065 / cos(pi / 6) * np.cos(a), 0.0065 / cos(pi / 6) * np.sin(a)
            )
            for z in (-0.002, 0.002)
        ]
    )
    local = (corners @ axis.T + nut_pose[:3] - pos) @ rot
    centre = (nut_pose[:3] - pos) @ rot
    return dict(
        along_m=float(centre[0]),
        across_m=float(centre[1]),
        height_m=float(centre[2]),
        face_clearance_m=0.00665 - float(np.abs(local[:, 1]).max()),
    )
