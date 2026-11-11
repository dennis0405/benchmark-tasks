"""Newton VBD + YAM joint-control smoke test; no snack task or demo."""
import argparse
import json
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('output',type=Path)
parser.add_argument('--physics',default='newton_vbd',choices=['newton_vbd'])
parser.add_argument('--duration',type=float,default=.4)
parser.add_argument('--delta',type=float,default=.15)
add_launcher_args(parser)
parser.set_defaults(visualizer=['kit'])
a = parser.parse_args()

import numpy as np
import torch
from isaaclab.physics import PhysicsCfg


def main():
    a.output.mkdir(parents=True,exist_ok=True)
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=a) as cfg:
        import isaaclab.sim as sim_utils
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from yam_arena.robot import yam_arm_cfg
        from yam_arena.station import BASES
        cfg.num_substeps = 8
        cfg.solver_cfg.iterations = 30
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/240, device=a.device, physics=cfg))
        ground = sim_utils.GroundPlaneCfg()
        ground.func('/World/ground',ground)
        robots = []
        for side in ('left','right'):
            arm_cfg = yam_arm_cfg(f'/World/{side}_robot', BASES[side])
            arm = arm_cfg.class_type(arm_cfg)
            robots.append(arm)
        sim.reset()
        for arm in robots: arm.update(1/240)
        model = NewtonManager.get_model()
        labels = list(model.body_label)
        print('YAM_BODIES',labels,flush=True)
        before = NewtonManager.get_state_0().body_q.numpy().copy()
        joints = []
        for arm in robots:
            print('YAM_JOINTS',arm.joint_names,flush=True)
            ids,_ = arm.find_joints(['joint1'], preserve_order=True)
            q = arm.data.default_joint_pos.torch[:,ids].clone()
            arm.set_joint_position_target_index(target=q + a.delta, joint_ids=ids)
            joints.append(ids)
        samples=[]
        for step in range(round(a.duration*240)):
            for arm in robots: arm.write_data_to_sim()
            sim.step(render=False)
            for arm in robots: arm.update(1/240)
            if step in (0,round(a.duration*240)-1):
                q = NewtonManager.get_state_0().body_q.numpy()
                displacement = np.linalg.norm(q[:,:3]-before[:,:3],axis=1)
                row={'step':step+1,'finite':bool(np.isfinite(q).all()),
                     'gripper_positions': {side: q[labels.index(f'/World/{side}_robot/Geometry/base/link1/link2/link3/link4/link5/gripper'), :3].tolist() for side in ('left','right')},
                     'max_body_displacement_m':float(displacement.max()),
                     'max_body_displacement_label':labels[int(displacement.argmax())]}
                samples.append(row); print('YAM_NEWTON',json.dumps(row),flush=True)
        (a.output/'result.json').write_text(json.dumps({'delta':a.delta,'samples':samples,'labels':labels},indent=2,allow_nan=False))


if __name__=='__main__': main()
