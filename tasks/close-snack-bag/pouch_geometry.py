"""Experimental open pillow pouch, in metres; not a calibrated material."""
import numpy as np


def open_pouch(nx=80, ny=64, length=0.20, width=0.16,
               body_half_height=0.022, mouth_half_gap=0.002):
    """Two film faces sharing side/bottom seals, with an open top boundary."""
    if nx < 4 or ny < 4 or min(length, width, body_half_height, mouth_half_gap) <= 0:
        raise ValueError("positive dimensions and at least four cells per axis required")
    points, indices = [], np.empty((2, nx + 1, ny + 1), dtype=np.int32)
    for layer in range(2):
        for i in range(nx + 1):
            for j in range(ny + 1):
                if layer and (i == 0 or j == 0 or j == ny):
                    indices[layer, i, j] = indices[0, i, j]
                    continue
                u, v = i / nx, j / ny
                body = np.sin(np.pi * min(u / 0.88, 1.0)) ** 0.7
                across = np.sin(np.pi * v) ** 0.7
                mouth = max(0.0, (u - 0.88) / 0.12)
                h = across * (body_half_height * body + mouth_half_gap * mouth)
                indices[layer, i, j] = len(points)
                points.append((length * u, width * (v - 0.5), (1 if layer else -1) * h))
    faces = []
    for layer in range(2):
        for i in range(nx):
            for j in range(ny):
                a, b = indices[layer, i, j], indices[layer, i + 1, j]
                c, d = indices[layer, i + 1, j + 1], indices[layer, i, j + 1]
                # At the bottom-right corner a and c are both seam nodes;
                # use the other diagonal to avoid a four-face edge.
                if i == 0 and j == ny - 1:
                    faces.extend(((a, b, d), (b, c, d)) if layer else ((a, d, b), (b, d, c)))
                else:
                    faces.extend(((a, b, c), (a, c, d)) if layer else ((a, c, b), (a, d, c)))
    return np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32), indices


def write_usd(path, **kwargs):
    from pxr import Gf, Usd, UsdGeom, Vt
    points, faces, _ = open_pouch(**kwargs)
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
