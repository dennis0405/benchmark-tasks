"""Connected, open-ended thin-surface mesh for a 200 x 160 mm snack packet.

The material parameters used with this geometry are experimental and are not
calibrated to a measured physical packet.
"""
import numpy as np


def packet_mesh(nx=20, ny=16, flat_mouth=False, mouth_half_gap=0.0015,
                taper_start=0.14, taper_end=0.16):
    """Return vertices and triangles for two skins joined along three edges.

    X runs from the sealed end to the open mouth. Both skins have identical
    in-plane coordinates; the four mouth corners remain distinct, so its
    boundary is open. The sides and sealed end are stitched with triangles.
    """
    if nx < 1 or ny < 1 or mouth_half_gap <= 0 or not 0 <= taper_start < taper_end <= 0.20:
        raise ValueError("invalid mesh dimensions or mouth taper")
    points = []
    for layer in (0, 1):
        for i in range(nx + 1):
            x = 0.20 * i / nx
            # A shallow filled-body silhouette tapers towards the open mouth.
            half_height = 0.003 + 0.014 * np.sin(np.pi * i / nx) ** 0.8
            if flat_mouth and x >= taper_start:
                t = max(0.0, min(1.0, (x - taper_start) / (taper_end - taper_start)))
                half_height = (1 - t) * half_height + t * mouth_half_gap
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
    if not 0 < radius < 0.02 / np.pi:
        raise ValueError("fold radius must fit within each 20 mm turn")
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


def write_usd(path, flat_mouth=False, nx=20, ny=16, mouth_half_gap=0.0015, crease_rest_fraction=0.0, fold_radius=0.003,
              taper_start=0.14, taper_end=0.16):
    from pxr import Gf, Usd, UsdGeom, Vt

    points, faces = packet_mesh(nx=nx, ny=ny, flat_mouth=flat_mouth, mouth_half_gap=mouth_half_gap,
                                taper_start=taper_start, taper_end=taper_end)
    if not 0 <= crease_rest_fraction <= 1:
        raise ValueError("crease_rest_fraction must be between 0 and 1")
    if crease_rest_fraction:
        points = ((1 - crease_rest_fraction) * points +
                  crease_rest_fraction * folded_positions(points, radius=fold_radius))
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
