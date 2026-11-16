"""First-party convex tool geometry; CPU authoring, not native physics validation.

Dimensions and masses below are benchmark assumptions, not SCHUNK accessory data.
Keep each ring sector as a separate collider: one hull around the whole ring
would fill the socket/bore and destroy the intended manipulation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

TAU = 2.0 * math.pi
CONTACT_OFFSET = 0.000025


def hex_radius(theta: float, across_flats: float) -> float:
    """Regular hexagon with vertices at 0,60,... degrees and flats parallel to X."""
    delta = (theta % (math.pi / 3.0)) - math.pi / 6.0
    return 0.5 * across_flats / math.cos(delta)


def extruded_polygon(name, polygon, lower_z, upper_z, material="Steel"):
    count = len(polygon)
    points = [(x, y, z) for z in (lower_z, upper_z) for x, y in polygon]
    faces = [list(reversed(range(count))), list(range(count, 2 * count))]
    faces += [[i, (i + 1) % count, (i + 1) % count + count, i + count] for i in range(count)]
    return {
        "name": name,
        "kind": "mesh",
        "points": points,
        "faces": faces,
        "footprint": polygon,
        "material": material,
    }


def ring_sectors(name, outer_radius, inner_radius, lower_z, upper_z, count=24):
    pieces = []
    for index in range(count):
        a, b = TAU * index / count, TAU * (index + 1) / count
        polygon = [
            (outer_radius(a) * math.cos(a), outer_radius(a) * math.sin(a)),
            (outer_radius(b) * math.cos(b), outer_radius(b) * math.sin(b)),
            (inner_radius(b) * math.cos(b), inner_radius(b) * math.sin(b)),
            (inner_radius(a) * math.cos(a), inner_radius(a) * math.sin(a)),
        ]
        pieces.append(extruded_polygon(f"{name}{index:02}", polygon, lower_z, upper_z))
    return pieces


def box(name, size, center, material="Steel"):
    return {"name": name, "kind": "box", "size": size, "center": center, "material": material}


def cylinder(name, radius, height, center, material="Steel"):
    return {
        "name": name,
        "kind": "cylinder",
        "radius": radius,
        "height": height,
        "center": center,
        "material": material,
    }


def make_assets():
    tip_af, socket_af = 0.00395, 0.00415
    nut_af, bore_radius = 0.013, 0.0042
    r = tip_af / math.sqrt(3.0)
    driver = [
        extruded_polygon(
            "HexTip",
            [(r * math.cos(i * TAU / 6), r * math.sin(i * TAU / 6)) for i in range(6)],
            0.0,
            0.018,
        )
    ]
    driver += [
        cylinder("Shaft", 0.003, 0.057, (0, 0, 0.0465)),
        box("Grip", (0.024, 0.024, 0.050), (0, 0, 0.100), "Grip"),
    ]
    wrench = [
        box("UpperJaw", (0.022, 0.0045, 0.004), (0.0025, 0.0089, 0)),
        box("LowerJaw", (0.022, 0.0045, 0.004), (0.0025, -0.0089, 0)),
        box("Heel", (0.0065, 0.0223, 0.004), (-0.01175, 0, 0)),
        box("Neck", (0.013, 0.016, 0.004), (-0.019, 0, 0)),
        box("Handle", (0.112, 0.012, 0.004), (-0.077, 0, 0)),
    ]
    nut = ring_sectors(
        "Sector", lambda a: hex_radius(a, nut_af), lambda a: bore_radius, -0.002, 0.002
    )
    screw = [cylinder("SmoothThreadProxy", 0.0039, 0.018, (0, 0, -0.015))]
    screw += ring_sectors(
        "SocketWall", lambda a: 0.0039, lambda a: hex_radius(a, socket_af), -0.006, 0.0
    )
    return {
        "hex_driver": {
            "pieces": driver,
            "mass_kg": 0.09,
            "com": (0, 0, 0.080),
            "inertia": (0.00011, 0.00011, 0.000009),
            "datums": {"tip": (0, 0, 0), "grasp": (0, 0, 0.100)},
            "across_flats_m": tip_af,
        },
        "open_wrench": {
            "pieces": wrench,
            "mass_kg": 0.04,
            "com": (-0.060, 0, 0),
            "inertia": (0.000002, 0.000065, 0.000066),
            "datums": {"nut_axis": (0, 0, 0), "grasp": (-0.085, 0, 0)},
            "jaw_gap_m": 0.0133,
        },
        "thin_nut": {
            "pieces": nut,
            "mass_kg": 0.009,
            "com": (0, 0, 0),
            "inertia": (0.00000014, 0.00000014, 0.00000026),
            "datums": {"axis": (0, 0, 0)},
            "across_flats_m": nut_af,
            "bore_radius_m": bore_radius,
        },
        "socket_adjuster": {
            "pieces": screw,
            "mass_kg": 0.009,
            "com": (0, 0, -0.014),
            "inertia": (0.00000050, 0.00000050, 0.00000007),
            "datums": {"socket_mouth": (0, 0, 0)},
            "socket_across_flats_m": socket_af,
        },
    }


def vector(values):
    return "(" + ", ".join(f"{value:.12g}" for value in values) + ")"


def collision_schemas(mesh=False):
    names = ['"PhysicsCollisionAPI"', '"PhysxCollisionAPI"', '"MaterialBindingAPI"']
    if mesh:
        names.append('"PhysicsMeshCollisionAPI"')
    return "[" + ", ".join(names) + "]"


def piece_usd(piece):
    kind = piece["kind"]
    typ = {"mesh": "Mesh", "box": "Cube", "cylinder": "Cylinder"}[kind]
    lines = [
        f'    def {typ} "{piece["name"]}" (',
        f"        prepend apiSchemas = {collision_schemas(kind == 'mesh')}",
        "    ) {",
        "        bool physics:collisionEnabled = true",
        f"        float physxCollision:contactOffset = {CONTACT_OFFSET}",
        "        float physxCollision:restOffset = 0",
        f"        rel material:binding = </Asset/Looks/{piece['material']}>",
    ]
    if kind == "mesh":
        lines += [
            '        uniform token subdivisionScheme = "none"',
            '        uniform token physics:approximation = "convexHull"',
            "        point3f[] points = [" + ", ".join(map(vector, piece["points"])) + "]",
            "        int[] faceVertexCounts = ["
            + ", ".join(str(len(f)) for f in piece["faces"])
            + "]",
            "        int[] faceVertexIndices = ["
            + ", ".join(str(v) for f in piece["faces"] for v in f)
            + "]",
        ]
    else:
        lines += ["        double3 xformOp:translate = " + vector(piece["center"])]
        if kind == "box":
            lines += [
                "        double size = 1",
                "        double3 xformOp:scale = " + vector(piece["size"]),
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
            ]
        else:
            lines += [
                f"        double radius = {piece['radius']}",
                f"        double height = {piece['height']}",
                '        uniform token axis = "Z"',
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            ]
    return "\n".join([*lines, "    }"])


def asset_usd(asset):
    materials = []
    for name, color, metal in (
        ("Steel", (0.44, 0.47, 0.50), 0.85),
        ("Grip", (0.9, 0.25, 0.04), 0.0),
    ):
        materials.append(f'''        def Material "{name}" {{
            token outputs:surface.connect = </Asset/Looks/{name}/PBR.outputs:surface>
            def Shader "PBR" {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {vector(color)}
                float inputs:metallic = {metal}
                float inputs:roughness = 0.35
                token outputs:surface
            }}
        }}''')
    return (
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    bool physics:rigidBodyEnabled = true
"""
        + f"""    float physics:mass = {asset["mass_kg"]}
    point3f physics:centerOfMass = {vector(asset["com"])}
    float3 physics:diagonalInertia = {vector(asset["inertia"])}
    def Scope "Looks" {{
"""
        + "\n".join(materials)
        + "\n    }\n"
        + "\n".join(map(piece_usd, asset["pieces"]))
        + "\n}\n"
    )


def export(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assets = make_assets()
    files = {}
    for name, asset in assets.items():
        content = asset_usd(asset).encode()
        filename = name + ".usda"
        (destination / filename).write_bytes(content)
        files[filename] = hashlib.sha256(content).hexdigest()
    manifest = {
        "status": "CPU-authored; native contact/dynamics NOT validated",
        "generator": {
            "file": Path(__file__).name,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "units": "m, kg, kg*m^2",
        "license": "MIT (first-party geometry/code)",
        "assumptions": "All tool dimensions, clearances, masses and inertia are benchmark choices; no claim of SCHUNK accessory equivalence. Smooth thread proxy requires separate native mechanism model.",
        "contact_offset_m": CONTACT_OFFSET,
        "files": files,
        "assets": assets,
    }
    (destination / "geometry.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets")
    print(json.dumps(export(parser.parse_args().output)["files"], indent=2))
