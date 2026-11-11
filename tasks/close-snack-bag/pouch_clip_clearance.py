"""Static surface-vs-box intersection check for a manually initialized open clip."""
import argparse
import json
import numpy as np
import open3d as o3d
from bag_surface_mesh import folded_positions
from pouch_geometry import open_pouch


def inspect(clip_x=.18, clip_z=.021, nx=80, ny=64):
    p, faces, _ = open_pouch(nx=nx, ny=ny, mouth_half_gap=.0003, foldable_mouth=True)
    shift = np.array([0., 0., .019])
    p = folded_positions(p + shift, radius=.0018) - shift + [0, 0, .025]
    packet = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p), o3d.utility.Vector3iVector(faces))
    yaw = np.diag([-1.,-1.,1.])
    c, s = np.cos(-.55), np.sin(-.55)
    hinge = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    result = {}
    for name, center, half, upper in (
        ('lower_jaw',(.022,0,-.004),(.022,.05,.003),False),
        ('upper_jaw',(.022,0,.004),(.022,.05,.003),True),
        ('lower_handle',(-.018,0,-.008),(.018,.012,.003),False),
        ('upper_handle',(-.018,0,.008),(.018,.012,.003),True)):
        center, half = np.asarray(center), np.asarray(half)
        transform = yaw @ hinge if upper else yaw
        local = (p - [clip_x,0,clip_z]) @ transform
        distance = np.abs(local - center) - half
        box = o3d.geometry.TriangleMesh.create_box(*(2*half))
        box.translate(center-half); box.rotate(transform, center=(0,0,0)); box.translate((clip_x,0,clip_z))
        result[name] = {'intersects': bool(packet.is_intersecting(box)),
                        'vertices_inside': int((distance.max(axis=1) < 0).sum())}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-x', type=float, default=.18)
    parser.add_argument('--clip-z', type=float, default=.021)
    a = parser.parse_args()
    print(json.dumps(inspect(a.clip_x, a.clip_z), indent=2))
