"""Native rack acquisition feasibility with the actual free USD tools and both YAM arms."""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", type=Path)
parser.add_argument("--leave-open",action="store_true",help="negative pickup control with identical arm targets")
args = parser.parse_args()
app = AppLauncher(headless=True,enable_cameras=bool(args.video)).app

import numpy as np
import torch
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from yam_arena import ik, scene, station, task_file
from yam_arena.constants import PAD_TIP_PAST_TCP, READY
from yam_arena.demo import grip_rotation
from yam_arena.embodiment import BimanualYamEmbodiment


def main():
    rack = json.loads((args.workspace/"input/fixture-geometry-contract.json").read_text())["rack"]
    rig = station.rig()
    embodiment = BimanualYamEmbodiment(rig=rig, enable_cameras=bool(args.video), cam_width=480, cam_height=272, contact_sensors=True)
    objects = list(scene.task_room("station", rig.top_cam))
    sides = ("left", "right")
    names = ("driver", "wrench")
    origins = (rack["D_reset_tip_world_m"], rack["W_reset_nut_axis_world_m"])
    grasps = (np.array(origins[0]) + (0,.100,0), np.array(origins[1]) + (-.085,0,0))
    for i, (label, filename, pose, rotation) in enumerate(zip(
        names, ("hex_driver","open_wrench"), origins,
        ((-math.sqrt(.5),0,0,math.sqrt(.5)), (0,0,0,1))
    )):
        objects.append(scene.Object(name=label, object_type=scene.ObjectType.RIGID,
            usd_path=str(args.workspace/"task/assets/tools"/(filename+".usda")),
            initial_pose=scene.Pose(position_xyz=tuple(pose), rotation_xyzw=rotation)))
        key = ("D","W")[i]
        low, high = np.array(rack[key+"_cradle_bounds_world_m"])
        base_size = rack["cradle_base_boxes_m"][key]
        base_z = low[2]+base_size[2]/2
        objects.append(scene.task_object(task_file.Box(label+"_rack",
            pos=(*((low+high)/2)[:2],base_z), size=tuple(base_size))))
        for j, (x,y) in enumerate(rack[key+"_support_centers_xy_m"]):
            top = rack["D_support_top_z_m"][("shaft","grip_end")[j]] if i==0 else rack["W_support_top_z_m"]
            width = rack["D_support_cross_section_m"] if i==0 else rack["support_cross_section_m"]
            bottom = base_z+base_size[2]/2
            objects.append(scene.task_object(task_file.Box(f"{label}_support_{j}",
                pos=(x,y,(top+bottom)/2),size=(*width,top-bottom))))
    arena = scene.IsaacLabArenaEnvironment(name="drg80_rack_acquisition",scene=scene.Scene(objects),
        embodiment=embodiment,env_cfg_callback=scene.env_cfg_callback(20))
    env = ArenaEnvBuilder(arena,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device="cuda:0")).make_registered()
    writer=None
    try:
        if args.video:
            import imageio.v2 as imageio
            writer=imageio.get_writer(str(args.video),fps=10,codec="libx264",macro_block_size=2)
        env.reset()
        base=env.unwrapped
        trace=[]
        previous = np.tile(READY,(2,1))
        angles={}
        for side, grasp, yaw in zip(sides,grasps,(math.pi/2,0)):
            alternatives=[]
            for az in (yaw,yaw+math.pi):
                q=np.array(READY)
                points=[]
                err=0.
                for height in (.05,0.,.05):
                    q,pe,ae=ik.solve(grasp+(0,0,PAD_TIP_PAST_TCP+height),
                        ik.matrix_to_quat_xyzw(grip_rotation(az)),q_init=q,base_pos=rig.bases[side],iters=300)
                    points.append(q)
                    err=max(err,pe+ae*.05)
                alternatives.append((err,points))
            err,angles[side]=min(alternatives,key=lambda x:x[0])
            if err>.002:
                raise ValueError(f"{side} rack target IK error {err}")
        def record(phase):
            trace.append(dict(phase=phase, tools={name:base.scene[name].data.root_pos_w.torch[0].cpu().tolist() for name in names},
                fingers={side:base.scene[side+"_robot"].data.joint_pos.torch[0,6:8].cpu().tolist() for side in sides}))
        for phase,target,grip,steps in (("settle",None,1,30),("approach",0,1,70),
                ("descend",1,1,50),("close",1,0,30),("lift",2,0,50),("hold",2,0,30)):
            if args.leave_open:
                grip=1
            goal=previous if target is None else np.stack([angles[side][target] for side in sides])
            for step in range(1,steps+1):
                t=step/steps
                joint=previous+(goal-previous)*t*t*(3-2*t)
                act=np.r_[joint[0],grip,joint[1],grip]
                env.step(torch.as_tensor(act[None],device=base.device,dtype=torch.float32))
                record(phase)
                if writer and step%2==0:
                    def frame(camera):
                        rgb=base.scene[camera].data.output["rgb"]
                        rgb=rgb.torch if hasattr(rgb,"torch") else rgb
                        return rgb[0,...,:3].cpu().numpy().astype(np.uint8)
                    overview=frame("top_cam")
                    wrists=np.concatenate([frame(k)[::2,::2] for k in ("left_cam","right_cam")],axis=1)
                    writer.append_data(np.concatenate([overview,wrists],axis=0))
            previous=goal.copy()
        settled=next(r for r in reversed(trace) if r["phase"]=="settle")
        lift={name:min(r["tools"][name][2]-settled["tools"][name][2] for r in trace if r["phase"]=="hold") for name in names}
        args.output.with_suffix(".trace.json").write_text(json.dumps(trace,allow_nan=False))
        args.output.write_text(json.dumps(dict(min_held_lift_m=lift, held_30mm=all(h>.03 for h in lift.values()),
            leave_open=args.leave_open,scope="rack pickup only; no task points",physics_dt=float(base.physics_dt),control_dt=float(base.step_dt)),indent=2))
    finally:
        if writer:
            writer.close()
        env.close()


try:
    main()
finally:
    app.close()
