"""Check a directly positioned, fully open clip against the initial folded surface.

This is a static geometry check only; it does not establish safe dynamic contact.
"""
import argparse
import json

import numpy as np
import open3d as o3d
from bag_surface_mesh import folded_positions, packet_mesh


def inspect(nx=40, ny=32, radius=0.0018, clip_x=0.18, clip_z=0.043):
    vertices, faces = packet_mesh(nx=nx, ny=ny, flat_mouth=True, mouth_half_gap=0.0003,
                                  taper_start=0.12, taper_end=0.14)
    vertices = folded_positions(vertices, radius=radius) + [0, 0, 0.028]
    packet = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.astype(float)),
                                        o3d.utility.Vector3iVector(faces))
    yaw = np.diag([-1.0, -1.0, 1.0])
    c, s = np.cos(-0.55), np.sin(-0.55)
    hinge = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    result = {}
    for name, center, half, upper in (
        ("lower_jaw", (0.022, 0, -0.004), (0.022, 0.05, 0.003), False),
        ("upper_jaw", (0.022, 0, 0.004), (0.022, 0.05, 0.003), True),
        ("lower_handle", (-0.018, 0, -0.008), (0.018, 0.012, 0.003), False),
        ("upper_handle", (-0.018, 0, 0.008), (0.018, 0.012, 0.003), True),
    ):
        half, center = np.asarray(half), np.asarray(center)
        transform = yaw @ hinge if upper else yaw
        local = (vertices - [clip_x, 0, clip_z]) @ transform
        d = np.abs(local - center) - half
        signed = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
        box = o3d.geometry.TriangleMesh.create_box(*(2 * half))
        box.translate(center - half)
        box.rotate(transform, center=(0, 0, 0))
        box.translate((clip_x, 0, clip_z))
        result[name] = {"mesh_intersects_box_surface": bool(packet.is_intersecting(box)),
                        "vertices_inside_box": int((signed < 0).sum()),
                        "min_vertex_distance_m": float(signed.min())}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-x", type=float, default=0.18)
    parser.add_argument("--clip-z", type=float, default=0.043)
    args = parser.parse_args()
    print(json.dumps(inspect(clip_x=args.clip_x, clip_z=args.clip_z), indent=2))
