"""Shared RoboCurve runner bridge for the existing rigid snack-bag task.

The scene, contact monitor and bimanual controller remain the legacy ones;
this bridge has not been established as equivalent until a real run succeeds.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

import arena_scene as legacy
from bridge_scoring import BridgeScorer
from demo_robot import Arm, plan

NAME = "close-snack-bag"
INSTRUCTION = json.loads(Path(__file__).with_name("task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 2400
OBJECT_NAMES = ("bag", "clip", "peg", "chip0", "chip1", "chip2", "chip3")


def make_environment(embodiment, device):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from yam_arena.scene import build_task_environment
    description = SimpleNamespace(name=NAME, scene=legacy.SCENE, objects=legacy.OBJECTS)
    cfg = build_task_environment(embodiment, description, control_hz=HZ)
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def read_object_states(env):
    from yam_arena.scene import object_states
    states = object_states(env, OBJECT_NAMES)
    # Articulations use the body-origin pose supplied by scenekit. The stock
    # ObjectState for a box can instead hold an AABB centre, so read its exact
    # root separately for the trajectory contract.
    poses = {name: (np.r_[state.pos, state.quat_xyzw] if name in ("bag", "clip", "peg")
                    else env.unwrapped.arena_world.get_pose_w(name)[0].cpu().numpy())
             for name, state in states.items()}
    return states, poses


class Dynamics:
    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def before_step(self, step, action):
        # The old embodiment restores the pre-folded initial state after it
        # settles. This write occurs only during reset, never during rollout.
        if step == -1:
            legacy._reset(self.env)

    def after_step(self, step, objects, root_poses, robot):
        legacy._check_clear(self.env)
        tools = {}
        for name in ("clip", "peg"):
            measured = legacy._measure(objects, name)
            tools[name] = {k: bool(measured[k]) for k in ("folded", "aligned", "jaw_on_layers", "still")}
            tools[name]["contact"] = bool(legacy._STATE["contacts"][name])
            tools[name]["overlap"] = float(measured["overlap"])
        return {"violation": bool(legacy._STATE["violation"]),
                "clear": bool(legacy._STATE["arms_clear"]), "tools": tools}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return BridgeScorer()


def demo():
    # The legacy demo is an adaptive joint-target generator, not waypoints.
    return {}


class DemoController:
    def __init__(self, env):
        self.left, self.right = Arm("left"), Arm("right")
        self.steps = plan(self.left, self.right, SimpleNamespace(_env=env), legacy)
        self.done = False

    def action(self, clock):
        if not self.done:
            try:
                next(self.steps)
            except StopIteration:
                self.done = True
        return np.r_[self.left.q, self.left.grip, self.right.q, self.right.grip]


def make_demo_controller(env, plans):
    return DemoController(env)
