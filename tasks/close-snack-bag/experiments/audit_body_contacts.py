"""Offline collision query at recorded body poses, including VBD poses.

No forward-kinematic projection and no recorded rollout is edited. Reported
separations are collision geometry diagnostics, not reconstructed forces.
"""
import argparse,hashlib,io,json,sys
from pathlib import Path
import numpy as np
import newton
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--step',type=int);a=p.parse_args()
args=json.loads((a.run/'arguments.json').read_text());raw=(a.run/'states.npz').read_bytes();states=np.load(io.BytesIO(raw))
step=a.step if a.step is not None else max(int(k[2:]) for k in states if k.startswith('b_'))
sim=make_environment(ready_start=True,with_clip=True,rigid_only=True,coupling='one_way',
    workspace=args['workspace'],clip_style=args['clip_style'],control_hz=20,physics_hz=1200,
    support_extension=args.get('support_extension_mm',0.)/1000.,tool_rack_style=args.get('tool_rack_style','wide'),
    tool_parking_shift_x=args.get('tool_parking_shift_x',0.),rebase_z=args.get('rebase_z',False),board_skin_mm=args.get('board_skin_mm',0.))
b=states[f'b_{step:05d}'];assert b.shape==sim.state.body_q.numpy().shape
sim.state.body_q.assign(b);sim.state.body_qd.zero_()
pipe=newton.CollisionPipeline(sim.model);contacts=pipe.contacts();pipe.collide(sim.state,contacts)
owners=sim.model.shape_body.numpy();labels=list(map(str,sim.model.shape_label));bodylabels=list(map(str,sim.model.body_label))
count=int(contacts.rigid_contact_count.numpy()[0]);s0=contacts.rigid_contact_shape0.numpy();s1=contacts.rigid_contact_shape1.numpy()
p0=contacts.rigid_contact_point0.numpy();p1=contacts.rigid_contact_point1.numpy();norm=contacts.rigid_contact_normal.numpy()
m0=contacts.rigid_contact_margin0.numpy();m1=contacts.rigid_contact_margin1.numpy()
rows=[]
def world(point,owner):return point if owner<0 else b[owner,:3]+Rotation.from_quat(b[owner,3:]).apply(point)
for i in range(min(count,contacts.rigid_contact_max)):
 pair=[int(s0[i]),int(s1[i])];names=[labels[s] for s in pair];body=[int(owners[s]) for s in pair]
 if not any('finger' in x or 'clip' in x or 'tool_' in x for x in names):continue
 pos0,pos1=world(p0[i],body[0]),world(p1[i],body[1]);gap=float(norm[i]@(pos1-pos0)-m0[i]-m1[i])
 if gap>.001:continue
 rows.append(dict(shapes=pair,labels=names,owners=body,separation_m=gap,point0=pos0.tolist(),point1=pos1.tolist()))
rows.sort(key=lambda r:r['separation_m'])
report=dict(scope=__doc__,step=step,states_sha256=hashlib.sha256(raw).hexdigest(),contacts=rows)
(a.run/f'body-contacts-{step:05d}.json').write_text(json.dumps(report,indent=2))
for row in rows[:40]:print(round(row['separation_m']*1e6,2),'um',[n.rsplit('/',1)[-1] for n in row['labels']],row['point0'])
sim.close()
