"""Offline IK and native rigid collision query for continuing a rim fold.

Candidate board widths modify only this disposable query model. No recorded
state, live rollout, or deformable film is advanced by this planner.
"""
import argparse,json,sys
from pathlib import Path
import mujoco,newton,numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment
from fold_controller import FoldController
from tests.fixtures.grasp import open_session


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('seed',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--jaw-z',type=float,default=.006)
    args=p.parse_args()
    sim=make_environment(device='cpu',rigid_cpu=True,rigid_only=True,ready_start=True,
        with_clip=True,workspace='raised_flat',rebase_z=True,tool_rack_style='end_access',
        tool_parking_shift_x=.04,clip_style='thin_buttons',support_extension=.014)
    session=open_session(sim,9793);c=FoldController(sim,session)
    seed=np.load(args.seed/'clip-entry.npz')
    sim.state.body_q.assign(seed['b']);sim.state.joint_q.assign(seed['j'])
    locals={}
    for arm in ('left','right'):
        tcp,quat=sim.arm(arm).ee_pose(sim.state)
        locals[arm]=Rotation.from_quat(quat).inv().apply(c.takes._fold_jaws(c.film,arm)-tcp)
    solver=sim.pipeline.program.solver;solver=getattr(solver,'rigid',solver)
    m,d=solver.mj_model,solver.mj_data;mapping=solver.mjc_geom_to_newton_shape.numpy()[0]
    labels=list(map(str,sim.model.shape_label))
    boards=[i for i,s in enumerate(mapping) if s>=0 and labels[s]=='board']
    if len(boards)!=1:raise ValueError('expected one native board geometry')
    rows=[]
    for width in (.12,.06,.04):
        for geom in boards:m.geom_size[geom,1]=width
        q=seed['j'].copy()
        for angle in (88,105,120,140,155,165,175):
            residuals={}
            for arm,sign in (('left',1),('right',-1)):
                ik=sim.arm(arm);quat=c.takes._fold_hand(arm,angle)
                target=np.array([.712,sign*.055,args.jaw_z])-Rotation.from_quat(quat).apply(locals[arm])
                with ik.host_fk():q=ik.solve_q_full(q,target,quat)
                actual,turn=ik.fk_ee(q)
                residuals[arm]=dict(position_m=float(np.linalg.norm(actual-target)),
                    angle_rad=float((Rotation.from_quat(turn).inv()*Rotation.from_quat(quat)).magnitude()))
            sim.state.joint_q.assign(q);newton.eval_fk(sim.model,sim.state.joint_q,sim.state.joint_qd,sim.state)
            solver._update_mjc_data(d,sim.model,sim.state);mujoco.mj_forward(m,d)
            hits=[]
            for contact in d.contact:
                if contact.dist>=-1e-5 or not any(int(g) in boards for g in contact.geom):continue
                other=next(int(g) for g in contact.geom if int(g) not in boards)
                shape=int(mapping[other]);label=labels[shape] if shape>=0 else str(other)
                if 'yam' in label:hits.append(dict(label=label,distance_m=float(contact.dist)))
            row=dict(board_half_width_m=width,turn_deg=angle,jaw_z_m=args.jaw_z,
                     residuals=residuals,robot_board_intersections=hits)
            rows.append(row);print('WRAP_QUERY',width,angle,residuals,
                [(h['label'].rsplit('/',1)[-1],round(h['distance_m']*1000,2)) for h in hits[:8]],flush=True)
    args.output.write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2))
    session.teardown();sim.close()


if __name__=='__main__':main()
