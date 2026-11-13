"""A/B test: authored joint rest pose versus episode's open starting pose.

VBD caches drive/limit rest-angle offsets at construction. Changing joint_q
without updating the authored body poses may make those defaults inconsistent.
Source: newton 1.7 SolverVBD joint-limit/rest-angle documentation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import newton
import warp as wp
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
p.add_argument('--inconsistent-rest', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
b = newton.ModelBuilder()
b.default_shape_cfg.gap = .0005
b.add_ground_plane()
b.add_usd(str(Path(__file__).resolve().parents[1] / 'assets/clip_v5/clip_v5.usda'),
          floating=True, xform=wp.transform(wp.vec3(.184, 0, .0252), wp.quat(0, 0, 1, 0)),
          override_root_xform=True)
h = next(i for i, n in enumerate(b.joint_label) if n.endswith('clip_hinge'))
qi, di = b.joint_q_start[h], b.joint_qd_start[h]
if a.inconsistent_rest:
    b.joint_q[qi] = -.55
b.joint_target_q[qi] = 0.
b.joint_target_ke[di], b.joint_target_kd[di] = .13, .003
b.joint_target_mode[di] = newton.JointTargetMode.POSITION
root, upper = b.joint_parent[h], b.joint_child[h]
b.color()
m = b.finalize(device='cpu')
s0, s1 = m.state(), m.state()
initial = m.joint_q.numpy().copy()
initial[qi] = -.55
q_initial = wp.array(initial, dtype=wp.float32, device='cpu')
for state in (s0, s1):
    if state.joint_q is not None:
        state.joint_q.assign(initial)
    newton.eval_fk(m, q_initial, m.joint_qd, state)
solver = newton.solvers.SolverVBD(m, iterations=50, rigid_compliant_alm=True)
pipeline = newton.CollisionPipeline(m)
contacts, ctrl = pipeline.contacts(), m.control()
samples = []
for step in range(601):
    if step % 60 == 0:
        body = s0.body_q.numpy()
        angle = (Rotation.from_quat(body[root, 3:]).inv() *
                 Rotation.from_quat(body[upper, 3:])).as_rotvec()[1]
        samples.append({'step': step, 'angle_rad': float(angle)})
    if step == 600:
        break
    s0.clear_forces()
    pipeline.collide(s0, contacts)
    solver.step(s0, s1, ctrl, contacts, 1/600)
    s0, s1 = s1, s0
lo, hi = b.joint_limit_lower[di], b.joint_limit_upper[di]
result = {'inconsistent_rest': a.inconsistent_rest, 'authored_joint_q': float(m.joint_q.numpy()[qi]),
          'samples': samples, 'limits': [lo, hi],
          'inside_limits': all(lo-.01 <= x['angle_rad'] <= hi+.01 for x in samples),
          'closed': abs(samples[-1]['angle_rad']) < .03}
(a.output / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
