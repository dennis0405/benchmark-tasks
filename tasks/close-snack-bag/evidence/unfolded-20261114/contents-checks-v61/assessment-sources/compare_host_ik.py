"""Compare upstream host/device IK on measured poses, without advancing physics."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment


def main():
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path)
    args=parser.parse_args()
    sim=make_environment(coupling="unified",ready_start=True,with_clip=True,
        with_contents=True,contents_mode="brittle",workspace="raised_flat",
        clip_style="thin_buttons",tool_rack_style="end_access",tool_parking_shift_x=.04,
        control_hz=20,physics_hz=2400,rebase_z=True,board_skin_mm=1.,
        rigid_contact_threads=32,body_particle_buffer=16384)
    from tests.fixtures.grasp import open_session
    session=open_session(sim,9765);seed=sim.state.joint_q.numpy().copy()
    report={}
    for arm,panel in session.arms.items():
        ik=panel.ik;position,quat=ik.ee_pose(sim.state)
        rows=[]
        for delta in ([.002,0,0],[0,.002,.002],[.004,0,.003]):
            target=position+delta;answers={}
            for mode in ("device","host"):
                start=time.perf_counter()
                with ik.host_fk() if mode=="host" else nullcontext():
                    q=ik.solve_q_full(seed,target,quat,iterations=60)
                elapsed=time.perf_counter()-start
                actual,rotation=ik.fk_ee(q)
                answers[mode]=dict(seconds=elapsed,
                    position_error_m=float(np.linalg.norm(actual-target)),
                    angle_error_rad=float((Rotation.from_quat(rotation).inv()*Rotation.from_quat(quat)).magnitude()),
                    arm_q=q[ik.arm_coords].tolist())
            rows.append(dict(delta_m=delta,answers=answers))
        report[arm]=dict(host_reason=getattr(ik,"fk_host_reason",None),probes=rows)
    args.output.write_text(json.dumps(dict(scope=__doc__,results=report,
        limitation="First device solve includes compilation. The upstream solver may use Pyroki in both scopes; this comparison is not evidence of a live host-FK speedup."),indent=2))
    session.close()


if __name__=="__main__":main()
