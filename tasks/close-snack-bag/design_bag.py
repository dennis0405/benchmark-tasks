"""Add a procedural snack-packet skin to the converted USD, without changing collisions.

Run once after converting the bag MJCF asset, using Isaac Lab's Python environment.
All lettering is original block lettering and contains no identifying branding.
"""
from pathlib import Path
import math

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

PATH = Path(__file__).parent / "assets/bag_v4/payloads/base.usda"
stage = Usd.Stage.Open(str(PATH))
root = stage.GetDefaultPrim().GetPath()
body = root.AppendPath("Geometry/bag")
for name in ("foil_top", "bottom_crimp", "label", "chips_lettering"):
    stage.RemovePrim(body.AppendChild(name))
for name in ("bag_body", "bag_top", "bag_back", "bag_front", "bag_side_l", "bag_side_r"):
    UsdGeom.Imageable(stage.GetPrimAtPath(body.AppendChild(name))).MakeInvisible()


def mesh(name, points, faces, color):
    prim = UsdGeom.Mesh.Define(stage, body.AppendChild(name))
    prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    prim.CreateFaceVertexCountsAttr([len(f) for f in faces])
    prim.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    prim.CreateSubdivisionSchemeAttr("none")
    prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
    prim.CreateDoubleSidedAttr(True)
    return prim


def box(name, centre, size, color):
    prim = UsdGeom.Cube.Define(stage, body.AppendChild(name))
    prim.CreateSizeAttr(1)
    prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))


# A softly bulging rectangular packet, with rounded corners and slight foil ripples.
def surface_z(x, y):
    u, v = x / .060, y / .080
    bulge = max(0., 1-u*u) * max(0., 1-v*v)
    return .021 + .011 * bulge + .00035 * math.sin(11*u + 6*v)


n = 20
pts = []
for i in range(n + 1):
    x = -1 + 2 * i / n
    for j in range(n + 1):
        y = -1 + 2 * j / n
        xx = 0.060 * x * (1 - 0.10 * abs(y) ** 8)
        yy = 0.080 * y * (1 - 0.10 * abs(x) ** 8)
        z = surface_z(xx, yy)
        pts.append((xx, yy, z))
faces = []
for i in range(n):
    for j in range(n):
        k = i * (n + 1) + j
        faces.append((k, k + 1, k + n + 2, k + n + 1))
mesh("foil_top", pts, faces, (0.92, 0.54, 0.035))
# Narrow crimp stripes and a simple original label; these are visual-only shapes.
box("bottom_crimp", (-0.056, 0, 0.0213), (0.007, 0.145, 0.001), (0.72, 0.34, 0.015))
label_points, label_faces = [], []
for i in range(21):
    x = -.039 + i * .078 / 20
    for j in range(9):
        y = -.015 + j * .030 / 8
        label_points.append((x, y, surface_z(x, y) + .0004))
for i in range(20):
    for j in range(8):
        k = i * 9 + j
        label_faces.append((k, k+1, k+10, k+9))
mesh("label", label_points, label_faces, (0.52, 0.08, 0.045))
letters = {
    "C": ("11111", "10000", "10000", "10000", "10000", "10000", "11111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "S": ("11111", "10000", "10000", "11111", "00001", "00001", "11111"),
}
pixel = 0.002
points, polys = [], []
for li, letter in enumerate("CHIPS"):
    for row, line in enumerate(letters[letter]):
        for col, bit in enumerate(line):
            if bit != "1":
                continue
            x = (li * 6 + col - 14) * pixel
            y = (3 - row) * pixel
            k = len(points)
            points.extend([(px, py, surface_z(px, py) + .0008) for px, py in
                           ((x, y), (x + pixel * .8, y),
                            (x + pixel * .8, y + pixel * .8), (x, y + pixel * .8))])
            polys.append((k, k + 1, k + 2, k + 3))
mesh("chips_lettering", points, polys, (1.0, 0.95, 0.75))
stage.GetRootLayer().Save()
# USD may append an extra blank line on save; keep the generated asset stable.
PATH.write_text(PATH.read_text().rstrip() + "\n")
print(f"Updated visual-only packet skin in {PATH}")
