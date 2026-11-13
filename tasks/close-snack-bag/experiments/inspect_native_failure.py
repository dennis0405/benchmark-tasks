"""Locate a saved first intrusion using an independent CPU triangle/box test."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from check_native_clip import penetrating_triangles, ground_clearance
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pouch_geometry import open_pouch

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
a = p.parse_args()
data = np.load(a.run/'failure-state.npz')
shapes = json.loads((a.run/'clip-shapes.json').read_text())
args = json.loads((a.run/'arguments.json').read_text())
rest, _, _ = open_pouch(nx=args['nx'], ny=args['ny'], mouth_half_gap=.0003, foldable_mouth=True)
reports = []
for sid, (owner, local, half) in enumerate(zip(shapes['shape_body'], shapes['shape_local'], shapes['shape_scale'])):
    if owner < 0:
        continue
    body = data['bodies'][owner]
    br = Rotation.from_quat(body[3:]).as_matrix()
    sr = Rotation.from_quat(local[3:]).as_matrix()
    pos, rot = body[:3]+br@local[:3], br@sr
    relative = (data['q']-pos)@rot
    hits = penetrating_triangles(relative, data['faces'], half)
    if not len(hits):
        continue
    vertices = data['faces'][hits].ravel()
    local_points = relative[vertices]
    inside = np.all(np.abs(local_points) < np.asarray(half), axis=1)
    depth = np.min(np.asarray(half)-np.abs(local_points[inside]), axis=1) if inside.any() else []
    reports.append({'shape': sid, 'body': owner, 'triangle_ids': hits.tolist(),
        'material_min_m': rest[vertices].min(axis=0).tolist(),
        'material_max_m': rest[vertices].max(axis=0).tolist(),
        'world_min_m': data['q'][vertices].min(axis=0).tolist(),
        'world_max_m': data['q'][vertices].max(axis=0).tolist(),
        'box_local_min_m': local_points.min(axis=0).tolist(),
        'box_local_max_m': local_points.max(axis=0).tolist(),
        'box_half_extents_m': half,
        'max_inside_vertex_depth_m': float(np.max(depth)) if len(depth) else None})
report = {'scope': 'CPU SAT; vertex depth is not a triangle minimum-translation depth', 'rows': reports,
          'ground':ground_clearance(data['q'],data['bodies'],shapes)}
(a.run/'failure-cpu-check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
