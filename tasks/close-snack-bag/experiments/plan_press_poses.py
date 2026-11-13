"""Compare press wrist reachability before running costly cloth physics.

Offline FK/IK only. No dynamic or collision-success claim.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import newton

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment
from fold_controller import FoldController


from finger_geometry import low_offset


def main():
    p=argparse.ArgumentParser();p.add_argument("seed",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--grid",action="store_true");a=p.parse_args()
    sim=make_environment(ready_start=True,rigid_only=True,with_clip=True,rigid_cpu=True,
                         workspace="raised_flat",clip_style="thin_lower",side_access=False,
                         control_hz=20,physics_hz=2400,support_extension=.014)
    from tests.fixtures.grasp import open_session
    session=open_session(sim,9770);c=FoldController(sim,session)
    initial=np.load(a.seed/"clip-entry.npz")["j"].copy()
    for state in (sim.pipeline.state_0,sim.pipeline.state_1):
        state.joint_q.assign(initial);newton.eval_fk(sim.model,state.joint_q,state.joint_qd,state)
    output=[]
    for arm,sign in (("left",1.),("right",-1.)):
        if a.grid and arm=="right":continue
        ik=sim.arm(arm)
        candidates=[("reference",c.takes._fold_presser(arm))]
        for pitch in (15,25,35,45):
            r=np.deg2rad(pitch)
            candidates.append((f"side_{pitch}",c.takes._frame_quat((0,-sign*np.cos(r),-np.sin(r)),(0,0,sign))))
            for closing in (-1,1):
                candidates.append((f"forward_{pitch}_{closing}",c.takes._frame_quat((np.cos(r),0,-np.sin(r)),(0,0,sign*closing))))
        for name,quat in candidates:
            offset=low_offset(sim,arm,quat,ik)
            points=([(.698,y,.970) for y in (sign*.055,sign*.070)] if not a.grid else
                    [(x,sign*.070,z) for x in (.60,.65,.70) for z in (.970,1.02,1.07,1.12)])
            for x,y,z in points:
                target=np.array([x,y,z])-offset
                with ik.host_fk():solved=ik.solve_q_full(initial,target,quat)
                pos,turn=ik.fk_ee(solved)
                miss=float(np.linalg.norm(pos-target));angle=float((Rotation.from_quat(turn).inv()*Rotation.from_quat(quat)).magnitude())
                output.append(dict(arm=arm,pose=name,tip_y=y,tip_x=x,tip_z=z,quat=quat.tolist(),lowest_offset=offset.tolist(),
                    target_tcp=target.tolist(),position_error_m=miss,angle_error_rad=angle,
                    feasible=miss<.004 and angle<.05,joints=solved[ik.arm_coords].tolist()))
    a.output.write_text(json.dumps(output,indent=2))
    print(json.dumps([{k:r[k] for k in ("arm","pose","tip_x","tip_y","tip_z","position_error_m","angle_error_rad","feasible")} for r in output if r["feasible"]],indent=2))
    session.teardown();sim.close()


if __name__=="__main__":main()
