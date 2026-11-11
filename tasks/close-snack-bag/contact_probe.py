"""Inspect PhysX clip/bag contacts after direct placement; not a robot demo."""
from pathlib import Path

import numpy as np


def main():
    import torch
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.constants import READY
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    emb = yam_arena_embodiment(task=str(Path(__file__).with_name("arena_scene.py")), object_jitter=0,
                               control_hz=20, cam_width=320, cam_height=176)
    emb.reset(Scene(id="contact-diagnostic", instruction="diagnostic"))
    import carb
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_physx.physics.physx_manager import PhysxManager
    from pxr import UsdPhysics
    from yam_arena.scene import object_states

    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    sensors, filters = [], []
    for prim in get_current_stage().Traverse():
        p = prim.GetPath().pathString
        if "/env_0/" not in p:
            continue
        if "/clip/" in p and "PhysxContactReportAPI" in prim.GetAppliedSchemas():
            sensors.append(p)
        if "/bag/" in p and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            filters.append(p)
    print("CONTACT_PRIMS", sensors, filters, flush=True)
    if not sensors or not filters:
        raise RuntimeError("contact prims missing; diagnostic is inconclusive")
    view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
        sensors, filter_patterns=[filters] * len(sensors), max_contact_data_count=512)
    env = emb._env.unwrapped
    bag = object_states(emb._env, ["bag"])["bag"]
    task = emb.task.module
    target = bag.pos + task._rotation(bag.quat_xyzw) @ np.array([0.125, -0.025, 0.0213])
    clip = env.scene["clip"]
    clip.write_root_pose_to_sim_index(root_pose=torch.tensor([[*target, 0, 0, 1, 0]], device=env.device,
                                                            dtype=torch.float32))
    clip.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.device))
    ids, _ = clip.find_joints(["clip_hinge"], preserve_order=True)
    angle = torch.tensor([[-0.10]], device=env.device)
    clip.write_joint_state_to_sim_index(position=angle, velocity=torch.zeros_like(angle), joint_ids=ids)
    clip.set_joint_position_target_index(target=torch.zeros_like(angle), joint_ids=ids)
    env.sim.forward()
    ready = np.r_[READY, 1.0]
    action = Action(data=pack(ready, ready))
    for step in range(100):
        result = emb.step(action)
        if step in (0, 19, 39, 59, 79, 99):
            forces, _, _, sep, counts, starts = view.get_contact_data(dt=PhysxManager.get_physics_dt())
            def np_array(x):
                return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x.numpy())
            f, s, c, start = map(np_array, (forces, sep, counts, starts))
            print("CONTACT", step + 1, "score", result.info.get("score"), "counts", c.tolist(),
                  "starts", start.tolist(), "force", f[:int(c.sum())].tolist(),
                  "separation", s[:int(c.sum())].tolist(), flush=True)


if __name__ == "__main__":
    main()
