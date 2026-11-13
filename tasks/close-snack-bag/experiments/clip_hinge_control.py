"""Isolate imported spring-clip hinge before trusting a rigid/film coupling."""
import json,sys,time
from pathlib import Path
import numpy as np
import warp as wp
import newton
from scipy.spatial.transform import Rotation
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
b=newton.ModelBuilder();b.default_shape_cfg.gap=.0005;b.add_ground_plane()
b.add_usd(str(Path(__file__).resolve().parents[1]/"assets/clip_v5/clip_v5.usda"),floating=True,
    xform=wp.transform(wp.vec3(.184,0,.0252),wp.quat(0,0,1,0)),override_root_xform=True)
h=next(i for i,n in enumerate(b.joint_label) if n.endswith("clip_hinge"))
qi,di=b.joint_q_start[h],b.joint_qd_start[h]
b.joint_q[qi]=-.55;b.joint_target_q[qi]=0.
b.joint_target_ke[di]=.13;b.joint_target_kd[di]=.003;b.joint_target_mode[di]=newton.JointTargetMode.POSITION
root,upper=b.joint_parent[h],b.joint_child[h]
b.color()
m=b.finalize(device="cpu");s0,s1=m.state(),m.state()
newton.eval_fk(m,m.joint_q,m.joint_qd,s0);newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
solver_name=sys.argv[2] if len(sys.argv)>2 else "vbd"
solver=(newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True) if solver_name=="mujoco" else newton.solvers.SolverVBD(m,iterations=50,rigid_compliant_alm=True))
pipeline=newton.CollisionPipeline(m);contacts=pipeline.contacts();ctrl=m.control()
def sample(step):
    body=s0.body_q.numpy()
    angle=(Rotation.from_quat(body[root,3:]).inv()*Rotation.from_quat(body[upper,3:])).as_rotvec()[1]
    return {"step":step,"jaw_angle":float(angle),"body_poses":body.tolist()}
rows=[sample(0)];start=time.monotonic()
for step in range(600):
    s0.clear_forces();pipeline.collide(s0,contacts);solver.step(s0,s1,ctrl,contacts,1/600)
    s0,s1=s1,s0
    if (step+1)%60==0:rows.append(sample(step+1))
result={"solver":solver_name,"device":"cpu","limits":[b.joint_limit_lower[di],b.joint_limit_upper[di]],
    "samples":rows,"wall_seconds":time.monotonic()-start}
(out/"result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
