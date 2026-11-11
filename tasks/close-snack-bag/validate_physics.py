"""Run physical scoring controls. Object placement here is NOT a robot demo.

Each case is reset separately, then stepped with both arms at READY. These
controls test the simulator/scorer interface, not real-world fidelity.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main(output, selected=None):
    import torch
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.constants import READY
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    emb = yam_arena_embodiment(task=str(Path(__file__).with_name("arena_scene.py")), object_jitter=0,
                               control_hz=20, cam_width=320, cam_height=176)
    ready = np.r_[READY, 1.0]
    action = Action(data=pack(ready, ready))
    cases = [("unclipped", None, 0), ("clip", "clip", 0), ("peg", "peg", 0),
             ("partial_overlap", "clip", 0.085), ("spilled", "clip", 0), ("fallen", None, 0)]
    report = {}
    for name, tool, offset in cases:
        if selected and name != selected:
            continue
        emb.reset(Scene(id=name, instruction="diagnostic"))
        from yam_arena.scene import object_states
        task = emb.task.module
        env = emb._env.unwrapped
        bag = object_states(emb._env, ["bag"])["bag"]
        if tool:
            target = bag.pos + task._rotation(bag.quat_xyzw) @ np.array([0.125, -0.025 + offset, 0.0213])
            asset = env.scene[tool]
            asset.write_root_pose_to_sim_index(root_pose=torch.tensor([[*target, 0, 0, 1, 0]],
                                               device=env.device, dtype=torch.float32))
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.device))
            ids, _ = asset.find_joints(["clip_hinge"], preserve_order=True)
            angle = torch.tensor([[-0.10]], device=env.device)
            asset.write_joint_state_to_sim_index(position=angle, velocity=torch.zeros_like(angle), joint_ids=ids)
            asset.set_joint_position_target_index(target=torch.zeros_like(angle), joint_ids=ids)
        if name == "spilled":
            chip = env.scene["chip0"]
            pos = bag.pos + np.array([0.30, 0, 0.02])
            chip.write_root_pose_to_sim(torch.tensor([[*pos, 1, 0, 0, 0]], device=env.device, dtype=torch.float32))
            chip.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        if name == "fallen":
            states = object_states(emb._env, ["bag", "chip0", "chip1", "chip2", "chip3"])
            delta = np.array([0, 0, -0.15])
            for object_name, state in states.items():
                x, y, z, w = state.quat_xyzw
                pose = torch.tensor([[*(state.pos + delta), w, x, y, z]], device=env.device,
                                    dtype=torch.float32)
                asset = env.scene[object_name]
                if object_name == "bag":
                    asset.write_root_pose_to_sim_index(root_pose=pose)
                    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.device))
                else:
                    asset.write_root_pose_to_sim(pose)
                    asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        env.sim.forward()
        samples = []
        for step in range(100):
            result = emb.step(action)
            if step in (0, 58, 59, 79, 99):
                state = object_states(emb._env, ["bag", tool] if tool else ["bag"])
                samples.append({"step": step + 1, "score": float(result.info.get("score", 0)),
                                "hold_steps": task._STATE["hold_steps"],
                                "violation": task._STATE["violation"],
                                "contacts": task._STATE["contacts"].copy(),
                                "geometry": task._measure(state, tool) if tool else None,
                                "fold": state["bag"].joints})
        report[name] = {"samples": samples, "reason": task._STATE.get("reason")}
        print(name, json.dumps(report[name], default=lambda value: value.item()), flush=True)
    Path(output).write_text(json.dumps(report, indent=2, default=lambda value: value.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--case", choices=("unclipped", "clip", "peg", "partial_overlap", "spilled", "fallen"))
    args = parser.parse_args()
    main(args.output, args.case)
