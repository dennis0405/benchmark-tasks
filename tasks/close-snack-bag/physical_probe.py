"""Diagnose a pre-positioned clip. This is not a robot demonstration.

This intentionally writes the clip's initial pose once after reset. Afterwards
only the physics engine acts; it must never be used as task-submission media.
"""
import argparse
from pathlib import Path
import numpy as np


def main(out, tool):
    import torch
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.constants import READY
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    from yam_arena import station

    emb = yam_arena_embodiment(task=str(Path(__file__).with_name("arena_scene.py")), object_jitter=0,
                               control_hz=20, cam_width=320, cam_height=176)
    emb.reset(Scene(id="contact-probe", instruction="diagnostic"))
    from yam_arena.scene import object_states
    task = emb.task.module
    states = object_states(emb._env, ["bag", tool])
    bag = states["bag"]
    target = bag.pos + task._rotation(bag.quat_xyzw) @ np.array([0.125, -0.025, 0.0213])
    clip = emb._env.unwrapped.scene[tool]
    device = emb._env.unwrapped.device
    pose = torch.tensor([[*target, 0, 0, 1, 0]], device=device, dtype=torch.float32)
    clip.write_root_pose_to_sim_index(root_pose=pose)
    clip.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device))
    ids, _ = clip.find_joints(["clip_hinge"], preserve_order=True)
    angle = torch.tensor([[-0.10]], device=device)
    clip.write_joint_state_to_sim_index(position=angle, velocity=torch.zeros_like(angle), joint_ids=ids)
    clip.set_joint_position_target_index(target=torch.zeros_like(angle), joint_ids=ids)
    emb._env.unwrapped.sim.forward()
    state = object_states(emb._env, ["bag", tool])
    print("initial target", target, "actual", state[tool].pos, "quat", state[tool].quat_xyzw,
          "joints", state[tool].joints, "geometry", task._measure(state, tool), flush=True)
    ready = np.r_[READY, 1.0]
    action = Action(data=pack(ready, ready))
    frames = []
    for step in range(120):
        result = emb.step(action)
        frames.append(np.asarray(result.observation.images["top_cam"]))
        if step < 3 or (step + 1) % 20 == 0:
            state = object_states(emb._env, ["bag", tool])
            print("probe", step + 1, "fold", state["bag"].joints, tool, state[tool].pos,
                  "angle", state[tool].joints, "score", result.info.get("score"),
                  "vel", state["bag"].lin_vel, state["bag"].ang_vel, state[tool].lin_vel,
                  state[tool].ang_vel, "monitor", task._STATE, "geometry", task._measure(state, tool), flush=True)
    import imageio
    imageio.mimsave(out, frames, fps=20, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    print("Diagnostic only. The object was placed directly at reset; no robot task was performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--tool", choices=("clip", "peg"), default="clip")
    args = parser.parse_args()
    main(args.output, args.tool)
