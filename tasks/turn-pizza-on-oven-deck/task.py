"""RoboCurve Inspect Robots task file. Use inspect_adapter.py for the native scene."""

import gymnasium as gym
from yam_arena.task_file import Asset

NAME = "turn-pizza-on-oven-deck"
INSTRUCTION = "Turn the pizza half a turn with the turning peel, keeping it centered on the oven deck. Leave it topping-side up and put the peel back on both supports."
RUN_LIMIT = 5000
SCENE = "tabletop"
# The reference controller starts with the open right gripper above the handle.
RIGHT_START = (
    -0.8685441,
    0.97630686,
    1.68516028,
    -1.69297004,
    -0.56202787,
    -0.68905151,
)
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0
OBJECTS = (
    Asset("peel", "turning_peel", (0.62, -0.35), 0.74, mass=0.4),
    Asset("pizza", "partially_baked_pizza", (0.95, -0.26), 0.30, mass=0.283),
)
_active_env = None


def score(states):
    if _active_env is None or _active_env.grade is None:
        raise RuntimeError("Load the task-local native-runtime.json adapter.")
    return _active_env.grade.score


def success(states):
    return score(states) == 1.0


class ScoredEnv(gym.Wrapper):
    def __init__(self, env, warmup_steps):
        super().__init__(env)
        import numpy as np
        import warp as wp
        from grading import Grade
        from yam_arena.robot import asset_link_paths

        root = "/World/envs/env_0"
        sim = env.unwrapped.sim
        view = sim.physics_manager.get_physics_sim_view()
        arms = [
            root + "/" + side + link
            for side in ("LeftArm", "RightArm")
            for link in asset_link_paths().values()
        ]
        self.grade = Grade(view, [root + "/pizza"], arms, tool_path=root + "/peel")
        self.parts = []
        for name in ("rim", "pad"):
            bodies = view.create_rigid_body_view("/World/Food/" + name + "*")
            poses = bodies.get_transforms()
            self.parts.append(
                (
                    bodies,
                    wp.clone(poses),
                    wp.zeros_like(bodies.get_velocities()),
                    wp.array(
                        np.arange(bodies.count, dtype=np.int32), device=poses.device
                    ),
                )
            )
        self.warmup = warmup_steps
        self.controls = 0
        self.live = False
        self.original_step = sim.step
        sim.step = self._physics_step

    def _physics_step(self, *args, **kwargs):
        out = self.original_step(*args, **kwargs)
        if self.live and self.controls >= 0:
            self.grade.observe()
        return out

    def reset(self, **kwargs):
        self.live = False
        result = self.env.reset(**kwargs)
        for bodies, poses, velocities, indices in self.parts:
            bodies.set_transforms(poses, indices)
            bodies.set_velocities(velocities, indices)
        self.grade.reset()
        self.controls = -self.warmup
        self.live = True
        return result

    def step(self, action):
        if not self.live or self.controls >= RUN_LIMIT:
            raise RuntimeError("Reset the episode before sending further actions.")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.controls += 1
        info["pizza_score"] = self.grade.score
        return obs, reward, terminated, truncated | (self.controls >= RUN_LIMIT), info

    def close(self):
        self.env.unwrapped.sim.step = self.original_step
        self.live = False
        return self.env.close()


def make_env(device="cuda:0", enable_cameras=True, warmup_steps=0):
    from arena_scene import build
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    cfg = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)
    return ScoredEnv(
        ArenaEnvBuilder(build(enable_cameras), cfg).make_registered(), warmup_steps
    )
