"""Check sampled packet triangles against the actual saved clip box poses.

Conservative Open3D intersection can count boundary touches; this is not a
continuous-collision test. It assumes the four box dimensions in clip_v5 USD.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from check_surface_intersections import strict_crossing


BOXES = {'lower_jaw': (0, (.022,0,-.004), (.022,.05,.003)),
         'upper_jaw': (1, (.022,0,.004), (.022,.05,.003)),
         'lower_handle': (0, (-.018,0,-.008), (.018,.012,.003)),
         'upper_handle': (1, (-.018,0,.008), (.018,.012,.003))}


def inspect(points, faces, poses):
    return {name: inspect_box(points, faces, poses[body], center, half)
            for name, (body, center, half) in BOXES.items()}


def inspect_box(points, faces, pose, center, half):
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points.astype(float)),
                                   o3d.utility.Vector3iVector(faces))
    pos=np.asarray(pose[:3]); rot=Rotation.from_quat(pose[3:]).as_matrix()
    center,half=np.asarray(center),np.asarray(half)
    local=(points-pos) @ rot
    d=np.abs(local-center)-half
    box=o3d.geometry.TriangleMesh.create_box(*(2*half))
    box.translate(center-half);box.rotate(rot,center=(0,0,0));box.translate(pos)
    hit = bool(mesh.is_intersecting(box))
    strict = 0
    if hit:
        box_points = np.asarray(box.vertices)
        box_faces = np.asarray(box.triangles)
        combined_points = np.concatenate((points, box_points))
        combined_faces = np.concatenate((faces, box_faces + len(points)))
        combined = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(combined_points.astype(float)),
                                            o3d.utility.Vector3iVector(combined_faces))
        for i,j in np.asarray(combined.get_self_intersecting_triangles()):
            if (i < len(faces)) != (j < len(faces)):
                strict += int(strict_crossing(combined_points[combined_faces[i]], combined_points[combined_faces[j]]))
    return {'surface_intersection_or_touch':hit, 'strict_crossings':strict,
            'vertices_inside':int(np.sum(np.all(d < -1e-7,axis=1)))}


def main(directory):
    data=np.load(directory/'surface-snapshots.npz')
    results=json.loads((directory/'newton-result.json').read_text())
    report={}
    for row in results['samples']:
        if not row.get('clip_finite') or 'clip_body_poses_xyzw' not in row:
            continue
        key=f"nodes_{row['step']}"
        report[key]=inspect(data[key],data['faces'],row['clip_body_poses_xyzw'])
        print(key,json.dumps(report[key]),flush=True)
    (directory/'clip-intersections.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)
