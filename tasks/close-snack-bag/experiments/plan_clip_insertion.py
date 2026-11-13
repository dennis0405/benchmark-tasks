"""Offline clip-route IK from a recorded successful rigid grasp, no dynamics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import newton
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment
from fold_controller import FoldController


def main():
    p=argparse.ArgumentParser();p.add_argument("grasp_run",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    sim=make_environment(ready_start=True,with_clip=True,rigid_only=True,rigid_cpu=True,
                         workspace="raised_flat",clip_style="thin_lower",control_hz=20,physics_hz=2400)
    from tests.fixtures.grasp import open_session
    session=open_session(sim,9771);c=FoldController(sim,session)
    data=np.load(a.grasp_run/"states.npz");initial=data[sorted(k for k in data.files if k.startswith("j_"))[-1]]
    for state in (sim.pipeline.state_0,sim.pipeline.state_1):
        state.joint_q.assign(initial);newton.eval_fk(sim.model,state.joint_q,state.joint_qd,state)
    measured=json.loads((a.grasp_run/"controller-marks.json").read_text())["clip_grasp_transform"]
    rotation=Rotation.from_quat(measured["relative_quat"])
    offset=np.array(measured["offset"]);ik=sim.arm("right");rows=[]
    for x in (.58,.60,.62,.65,.66):
        for z in (.9692,1.008,1.04):
            root=np.array([x,0,z]);quat=rotation.as_quat()
            jaw=root+offset;target=c.takes._fold_targets(c.film,{"right":jaw},{"right":quat})["right"]
            with ik.host_fk():solved=ik.solve_q_full(initial,target,quat)
            pos,turn=ik.fk_ee(solved)
            miss=float(np.linalg.norm(pos-target));angle=float((Rotation.from_quat(turn).inv()*rotation).magnitude())
            rows.append(dict(root_m=root.tolist(),target_tcp_m=target.tolist(),position_error_m=miss,
                angle_error_rad=angle,feasible=miss<.004 and angle<.05,joints=solved[ik.arm_coords].tolist()))
    report=dict(scope=__doc__,grasp_source=str(a.grasp_run),grasp=measured,rows=rows)
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))
    session.teardown();sim.close()


if __name__=="__main__":main()
