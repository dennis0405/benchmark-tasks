"""Newton VBD YAM TCP path check, without the snack scene."""
import argparse
import json
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.physics import PhysicsCfg

parser=argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('output',type=Path)
parser.add_argument('--physics',default='newton_vbd',choices=['newton_vbd'])
add_launcher_args(parser)
parser.set_defaults(visualizer=['kit'])
a=parser.parse_args()


def main():
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    a.output.mkdir(parents=True,exist_ok=True)
    with launch_simulation(cfg=PhysicsCfg(),launcher_args=a) as cfg:
        import isaaclab.sim as sim_utils
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from yam_arena.robot import yam_arm_cfg
        from yam_arena.station import BASES
        from yam_arena.constants import READY, TCP_OFFSET
        from yam_arena import ik
        cfg.num_substeps=8; cfg.solver_cfg.iterations=30
        dt=1/240
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=a.device,physics=cfg))
        ground=sim_utils.GroundPlaneCfg();ground.func('/World/ground',ground)
        arms=[]
        for side in ('left','right'):
            conf=yam_arm_cfg(f'/World/{side}_robot',BASES[side]);arms.append(conf.class_type(conf))
        sim.reset()
        for arm in arms:
            arm.update(dt)
            defaults = arm.data.default_joint_pos.torch.clone()
            arm.write_joint_state_to_sim_index(position=defaults, velocity=torch.zeros_like(defaults))
        sim.forward()
        for arm in arms: arm.update(dt)
        labels=list(NewtonManager.get_model().body_label)
        indices=[labels.index(f'/World/{side}_robot/Geometry/base/link1/link2/link3/link4/link5/gripper') for side in ('left','right')]
        def tcp_positions():
            poses=NewtonManager.get_state_0().body_q.numpy()[indices]
            return [pose[:3]+Rotation.from_quat(pose[3:]).apply(TCP_OFFSET) for pose in poses]
        q0=np.array(READY,dtype=float)
        before=tcp_positions()
        targets=[]; qgoals=[]; ik_errors=[]; fk_offsets=[]
        for side, observed in zip(('left','right'),before):
            position,rotation=ik.tcp_pose(q0,BASES[side])
            fk_offsets.append(float(np.linalg.norm(observed-position)))
            goal=position+np.array([.02,0,0]); q,err,_=ik.solve(goal,ik.matrix_to_quat_xyzw(rotation),q0,base_pos=BASES[side])
            targets.append(goal);qgoals.append(q);ik_errors.append(err)
        ids=[arm.find_joints([f'joint{i}' for i in range(1,7)],preserve_order=True)[0] for arm in arms]
        for step in range(480):
            t=min(1.,(step+1)/240)
            t=t*t*(3-2*t)
            for arm,arm_ids,qgoal in zip(arms,ids,qgoals):
                command=torch.tensor((q0+t*(qgoal-q0))[None],dtype=torch.float32,device=a.device)
                arm.actuators.target_command.set_position_index(value=command,joint_ids=arm_ids)
                gravity=arm.data.gravity_compensation_forces.torch
                if gravity.shape[1] == arm.num_joints + 6:
                    gravity=gravity[:,6:]
                arm.actuators.target_command.set_effort_index(value=gravity[:,arm_ids].contiguous(),joint_ids=arm_ids)
                arm.write_data_to_sim()
            sim.step(render=False)
            for arm in arms:arm.update(dt)
        after=tcp_positions()
        result={'fk_initial_error_m':fk_offsets,'ik_plan_error_m':ik_errors,
                'targets':[x.tolist() for x in targets], 'actual':[x.tolist() for x in after],
                'actual_target_error_m':[float(np.linalg.norm(x-y)) for x,y in zip(after,targets)],
                'finite':bool(all(np.isfinite(x).all() for x in after))}
        print('YAM_IK',json.dumps(result),flush=True)
        (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
