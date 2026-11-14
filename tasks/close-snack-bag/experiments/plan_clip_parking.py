"""Host IK checks for end-pad grasp, lift, and the final flat tool route.

No dynamics, collision certification, or changes to any recorded rollout.
"""
import json,sys
from pathlib import Path
import numpy as np
import newton
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment
from fold_controller import FoldController
sim=make_environment(ready_start=True,with_clip=True,rigid_only=True,rigid_cpu=True,
    workspace='raised_flat',clip_style='thin_compact',control_hz=20,physics_hz=1200)
from tests.fixtures.grasp import open_session
session=open_session(sim,9781);c=FoldController(sim,session);ik=sim.arm('right')
seed=np.load('/data/robocurve/runs/sequential-rim-press-v16/seeds/clip_approach/clip-entry.npz')
for state in (sim.pipeline.state_0,sim.pipeline.state_1):
    state.joint_q.assign(seed['j']);newton.eval_fk(sim.model,state.joint_q,state.joint_qd,state)
current=seed['j'].copy();tcp,turn=ik.ee_pose(sim.state)
local=Rotation.from_quat(turn).inv().apply(c.takes._fold_jaws(c.film,'right')-tcp)
angle=-.32;upper=Rotation.from_euler('y',angle).apply([-.083,0,.04]);closed=(upper+[-.083,0,-.04])/2
rows=[]
def solve(q,root,rot,offset,relative):
    quat=(rot*relative).as_quat();target=np.asarray(root)+rot.apply(offset)-Rotation.from_quat(quat).apply(local)
    with ik.host_fk():solved=ik.solve_q_full(q,target,quat)
    pos,actual=ik.fk_ee(solved)
    error=float(np.linalg.norm(pos-target));angular=float((Rotation.from_quat(actual).inv()*Rotation.from_quat(quat)).magnitude())
    return solved,dict(position_m=error,angle_rad=angular,feasible=error<.002 and angular<.03)
for sign in (1,-1):
    nominal=Rotation.from_quat(c.takes._frame_quat((sign,0,0),(0,1,0)))
    relative=Rotation.from_euler('x',-np.pi/2)*nominal
    q=current.copy();route=[]
    for root in ((.66,0,1.04),(.576,0,.9694),(.66,0,.9694)):
        q,r=solve(q,root,Rotation.identity(),closed,relative);route.append(r)
    print('FINAL',sign,route,flush=True)
    for x,y in ((.62,-.30),(.62,-.24),(.66,-.30),(.58,-.24)):
        for z in (.83,.88,.93,.98):
            for yaw in (0.,-45.,45.):
                rot=Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_euler('x',np.pi/2)
                q=current.copy();checks=[]
                for dz,offset in ((.15,[-.083,0,0]),(0.,[-.083,0,0]),(.11,closed)):
                    q,r=solve(q,[x,y,z+dz],rot,offset,relative);checks.append(r)
                rows.append(dict(root_m=[x,y,z],yaw_deg=yaw,approach_sign=sign,
                    parking_checks=checks,final_checks=route,feasible=all(r['feasible'] for r in checks+route)))
out=Path('/data/robocurve/runs/clip-parking-plan.json')
out.write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2))
good=[r for r in rows if r['feasible']]
print('FEASIBLE',len(good),json.dumps(good[:8],indent=2),flush=True)
session.teardown();sim.close()
