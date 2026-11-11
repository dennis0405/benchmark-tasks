"""Conservative reset-time pouch/support collision check for Newton probes."""
import numpy as np
import open3d as o3d

from bag_surface_mesh import folded_positions
from pouch_geometry import open_pouch


def initial_overlap(nx, ny, radius, center, size):
    points, faces, _ = open_pouch(nx=nx, ny=ny, mouth_half_gap=.0003, foldable_mouth=True)
    shift = np.array([0., 0., .019])
    points = folded_positions(points + shift, radius=radius) - shift + [0, 0, .025]
    center, size = np.asarray(center), np.asarray(size)
    if center.shape != (3,) or size.shape != (3,) or np.any(size <= 0):
        raise ValueError("invalid support box")
    packet = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points.astype(float)),
                                      o3d.utility.Vector3iVector(faces))
    box = o3d.geometry.TriangleMesh.create_box(*size)
    box.translate(center - size / 2)
    inside = np.all(np.abs(points - center) < size / 2, axis=1)
    return bool(np.any(inside) or packet.is_intersecting(box))
