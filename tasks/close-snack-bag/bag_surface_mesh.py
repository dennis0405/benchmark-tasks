"""Connected, open-ended thin-surface mesh for a 200 x 160 mm snack packet.

The material parameters used with this geometry are experimental and are not
calibrated to a measured physical packet.
"""
import numpy as np


def packet_mesh(nx=20, ny=16, flat_mouth=False):
    """Return vertices and triangles for two skins joined along three edges.

    X runs from the sealed end to the open mouth. Both skins have identical
    in-plane coordinates; the four mouth corners remain distinct, so its
    boundary is open. The sides and sealed end are stitched with triangles.
    """
    points = []
    for layer in (0, 1):
        for i in range(nx + 1):
            x = 0.20 * i / nx
            # A shallow filled-body silhouette tapers towards the open mouth.
            half_height = 0.003 + 0.014 * np.sin(np.pi * i / nx) ** 0.8
            if flat_mouth and x >= 0.14:
                half_height *= max(0.0, min(1.0, (0.16 - x) / 0.02))
                half_height += 0.0015 * max(0.0, min(1.0, (x - 0.14) / 0.02))
            for j in range(ny + 1):
                y = 0.16 * (j / ny - 0.5)
                points.append((x, y, 0.019 + (1 if layer else -1) * half_height))

    def index(layer, i, j):
        return layer * (nx + 1) * (ny + 1) + i * (ny + 1) + j

    faces = []
    for layer in (0, 1):
        for i in range(nx):
            for j in range(ny):
                a, b, c, d = (index(layer, i, j), index(layer, i + 1, j),
                              index(layer, i + 1, j + 1), index(layer, i, j + 1))
                faces.extend(((a, b, c), (a, c, d)) if layer else ((a, c, b), (a, d, c)))
    for i in range(nx):
        for j in (0, ny):
            a, b = index(0, i, j), index(0, i + 1, j)
            c, d = index(1, i + 1, j), index(1, i, j)
            faces.extend(((a, b, c), (a, c, d)) if j == 0 else ((a, c, b), (a, d, c)))
    for j in range(ny):
        a, b = index(0, 0, j), index(0, 0, j + 1)
        c, d = index(1, 0, j + 1), index(1, 0, j)
        faces.extend(((a, c, b), (a, d, c)))
    return np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def folded_positions(points, radius=0.003):
    """Fold the final 40 mm twice without changing the mesh rest shape.

    This produces reset positions only; the simulator's rest mesh remains flat.
    It is a geometric initial guess, not a demonstrated folding action.
    """
    out = points.copy()
    arc = np.pi * radius
    straight = 0.02 - arc
    mid = 0.019
    for i, (x, _y, z) in enumerate(points):
        u = float(x) - 0.16
        if u <= 0:
            continue
        offset = float(z) - mid
        if u < arc:
            t = u / radius
            cx, cz, nx, nz = 0.16 + radius*np.sin(t), radius*(1-np.cos(t)), -np.sin(t), np.cos(t)
        elif u < 0.02:
            cx, cz, nx, nz = 0.16 - (u - arc), 2*radius, 0.0, -1.0
        elif u < 0.02 + arc:
            t = (u - 0.02) / radius
            cx, cz, nx, nz = (0.16-straight-radius*np.sin(t),
                              2*radius+radius*(1-np.cos(t)), -np.sin(t), -np.cos(t))
        else:
            cx, cz, nx, nz = 0.16-straight+(u-0.02-arc), 4*radius, 0.0, 1.0
        out[i, 0] = cx + offset*nx
        out[i, 2] = mid + cz + offset*nz
    return out


def write_usd(path, flat_mouth=False, nx=20):
    from pxr import Gf, Usd, UsdGeom, Vt

    points, faces = packet_mesh(nx=nx, flat_mouth=flat_mouth)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Packet")
    mesh = UsdGeom.Mesh.Define(stage, "/Packet/film")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.ravel().tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([(0.94, 0.68, 0.04)])
    stage.SetDefaultPrim(root.GetPrim())
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()
