"""Clearance of the authored hex bit inside the six socket planes, in metres."""

from functools import lru_cache
import numpy as np

# Across flats from HexTip and SocketWall00..23 in the first-party USDs.
angles = np.arange(6) * np.pi / 3
radius = 0.001975 / np.cos(np.pi / 6)
rim = radius * np.column_stack((np.cos(angles), np.sin(angles)))
flat_normals = np.column_stack((np.cos(angles + np.pi / 6), np.sin(angles + np.pi / 6)))
CHAMFER = 0.0005


@lru_cache(maxsize=2)
def bit_mesh(chamfer):
    if chamfer not in (0.0, CHAMFER):
        raise ValueError("expected the original bit or the approved 0.50 mm bevel")
    rings = [(0.0, 0.001975 - chamfer)]
    if chamfer:
        rings.append((chamfer, 0.001975))
    rings.append((0.018, 0.001975))
    vertices = np.vstack(
        [np.column_stack((rim * (a / 0.001975), np.full(6, z))) for z, a in rings]
    )
    last = 6 * (len(rings) - 1)
    faces = [list(reversed(range(6))), list(range(last, last + 6))]
    for level in range(0, last, 6):
        for i in range(6):
            j = (i + 1) % 6
            faces.append([level + i, level + j, level + 6 + j, level + 6 + i])
    edges = sorted(
        {tuple(sorted((a, b))) for f in faces for a, b in zip(f, f[1:] + f[:1])}
    )
    return vertices, faces, edges


def author_bit(mesh, chamfer):
    from pxr import Gf

    v, faces, _ = bit_mesh(chamfer)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*(float(t) for t in pt)) for pt in v])
    mesh.GetFaceVertexCountsAttr().Set([len(f) for f in faces])
    mesh.GetFaceVertexIndicesAttr().Set([idx for f in faces for idx in f])


def hex_fit(bit_pos, bit_rot, screw_pos, screw_rot, chamfer=0.0):
    """Clip each actual edge to the socket depth before comparing wall and floor clearance."""
    bit_vertices, _, edges = bit_mesh(chamfer)
    local = (
        bit_vertices @ bit_rot.T + np.asarray(bit_pos) - np.asarray(screw_pos)
    ) @ screw_rot
    inside = (local[:, 2] >= -0.006) & (local[:, 2] <= 0)
    section = list(local[inside])
    for a, b in edges:
        v, w = local[a], local[b]
        for plane in (-0.006, 0.0):
            if (v[2] - plane) * (w[2] - plane) < 0.0:
                section.append(v + (w - v) * ((plane - v[2]) / (w[2] - v[2])))
    depth = -float(local[:6, 2].max())
    bottom = float(local[:, 2].min() + 0.006)
    side = (
        None
        if not section
        else 0.002075 - float(np.max(np.asarray(section)[:, :2] @ flat_normals.T))
    )
    return dict(
        depth_m=depth,
        straight_flank_depth_m=-float(local[6:12, 2].max()) if chamfer else depth,
        wall_clearance_m=side,
        floor_clearance_m=bottom,
    )


def six_socket_faces(stage, parent):
    """Group four adjacent native sectors into one convex wall with the same bore."""
    from pxr import UsdGeom, UsdPhysics, PhysxSchema

    for face in range(6):
        top = []
        for i in range(4):
            prim = stage.GetPrimAtPath(f"{parent}/SocketWall{4 * face + i:02d}")
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            # Every sector uses the same ordered upper ring: outer ends then inner ends.
            if not i:
                inside_start = points[7]
                top.append(points[4])
            top.append(points[5])
            inside_end = points[6]
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        top.extend([inside_end, inside_start])
        n = len(top)
        bottom = [(p[0], p[1], -0.006) for p in top]
        faces = [list(reversed(range(n))), list(range(n, 2 * n))]
        faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        mesh = UsdGeom.Mesh.Define(stage, f"{parent}/HexFlatCollider{face}")
        mesh.CreatePointsAttr(bottom + [tuple(p) for p in top])
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
        mesh.CreateVisibilityAttr("invisible")
        mesh.CreateSubdivisionSchemeAttr("none")
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        props = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        props.CreateContactOffsetAttr(0.000025)
        props.CreateRestOffsetAttr(0.0)
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateMinThicknessAttr(
            0.00005
        )
