"""Technical CPU geometry preview; no simulator, policy or physics execution."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pxr import Gf, Usd, UsdGeom


def polygons(prim):
    if prim.IsA(UsdGeom.Mesh):
        obj = UsdGeom.Mesh(prim)
        points = list(obj.GetPointsAttr().Get())
        flat, counts = (
            list(obj.GetFaceVertexIndicesAttr().Get()),
            obj.GetFaceVertexCountsAttr().Get(),
        )
        faces, offset = [], 0
        for n in counts:
            faces.append(flat[offset : offset + n])
            offset += n
    elif prim.IsA(UsdGeom.Cube):
        half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
        points = [(x * half, y * half, z * half) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
        faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    elif prim.IsA(UsdGeom.Cylinder):
        obj = UsdGeom.Cylinder(prim)
        radius, height, n = obj.GetRadiusAttr().Get(), obj.GetHeightAttr().Get(), 24
        points = [
            (radius * np.cos(i * 2 * np.pi / n), radius * np.sin(i * 2 * np.pi / n), z)
            for z in (-height / 2, height / 2)
            for i in range(n)
        ]
        faces = [list(reversed(range(n))), list(range(n, n * 2))]
        faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    else:
        return []
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    vertices = np.array([list(transform.Transform(Gf.Vec3d(*p))) for p in points]) * 1000
    return [vertices[face] for face in faces]


def render(source, output):
    stage = Usd.Stage.Open(str(Path(source).resolve()))
    assert stage
    colors = {
        "frame": (0.24, 0.40, 0.55),
        "left_jaw": (0.65, 0.68, 0.72),
        "right_jaw": (0.65, 0.68, 0.72),
        "piston": (0.45, 0.51, 0.56),
        "adjustment_screw": (0.31, 0.34, 0.38),
        "nut73": (0.85, 0.60, 0.19),
        "nut74": (0.62, 0.47, 0.24),
    }
    light = np.array([0.3, -0.4, 0.9])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(12, 6), facecolor="white")
    for panel in (1, 2):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        for prim in stage.Traverse():
            faces = polygons(prim)
            if not faces:
                continue
            name = str(prim.GetPath()).split("/")[2]
            if panel == 2 and name not in {"adjustment_screw", "nut73", "nut74"}:
                continue
            base = np.array(colors.get(name, (0.6, 0.6, 0.6)))
            shaded = []
            for face in faces:
                normal = np.cross(face[1] - face[0], face[2] - face[0])
                norm = np.linalg.norm(normal)
                shade = 0.72 + 0.28 * abs(float(normal @ light / norm)) if norm else 0.8
                shaded.append(tuple(np.clip(base * shade, 0, 1)))
            ax.add_collection3d(
                Poly3DCollection(
                    faces, facecolors=shaded, edgecolors=(0.15, 0.18, 0.21, 0.2), linewidths=0.25
                )
            )
        if panel == 1:
            ax.set(
                xlim=(-75, 75),
                ylim=(-70, 70),
                zlim=(0, 160),
                title="Representative angular-gripper fixture",
            )
            ax.set_box_aspect((150, 140, 160))
            ax.view_init(25, -60)
        else:
            ax.set(
                xlim=(-22, 22),
                ylim=(-40, 4),
                zlim=(72, 116),
                title="Adjuster + nuts (cover hidden in this view)",
            )
            ax.set_box_aspect((44, 44, 44))
            ax.view_init(50, -65)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
    fig.suptitle(
        "CPU geometry proposal — diagnostic body colors — native contacts NOT validated",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "fixture/bench.usda")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "fixture_cpu_preview.png"
    )
    args = parser.parse_args()
    render(args.source, args.output)
