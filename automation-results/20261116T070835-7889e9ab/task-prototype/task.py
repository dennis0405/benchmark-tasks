"""Single-environment RoboCurve YAM adapter for any 14D action policy."""

import gymnasium as gym
from yam_arena.task_file import Asset

SCENE = "tabletop"
SUCCESS_HOLD_S = 0.0  # The native scorer already counts the complete 3-second dwell.
SCORE_OVER = "end"
# Symbolic names for Inspect's object-state lookup. The native scene constructor
# builds the actual compound geometry for these identifiers.
OBJECTS = (
    Asset("screen", "perforated_finishing_screen", (0.37, 0), 0.3683, mass=0.40),
    *(
        Asset(
            f"media_{i:02d}",
            "angle_cut_ceramic_rod",
            (0.37 + (0.03 if i % 2 else -0.03), (i // 2 - 1.5) * 0.03),
            0.0248,
            mass=0.00396,
        )
        for i in range(8)
    ),
    Asset("washer_00", "annular_steel_washer", (0.50, 0.07), 0.03, mass=0.0206),
    Asset("washer_01", "annular_steel_washer", (0.24, -0.07), 0.03, mass=0.0206),
)

_active_env = None


def score(states):
    if _active_env is None or _active_env.grade is None:
        raise RuntimeError("Use the task's native-runtime.json adapter for this scene.")
    return _active_env.grade.score


def success(states):
    return score(states) >= 1.0


NAME = "separate-ceramic-finishing-media"
INSTRUCTION = "Separate the ceramic finishing media into the receiver while retaining both steel washers in the screen. Return the screen to both supports and release it."
RUN_LIMIT = 4800


def build(enable_cameras=True):
    import full_screen_scene as scene

    return scene.build_environment(True, "full", enable_cameras)


class ScoredEnv(gym.Wrapper):
    def __init__(self, env, warmup_steps=0):
        super().__init__(env)
        self.warmup_steps = warmup_steps
        self.grade = None
        self.controls = 0
        self.native_step = env.unwrapped.sim.step
        env.unwrapped.sim.step = self._native_step

    def _native_step(self, *args, **kwargs):
        result = self.native_step(*args, **kwargs)
        if self.grade is not None:
            self.grade.observe()
        return result

    def reset(self, **kwargs):
        self.grade = None
        result = self.env.reset(**kwargs)
        from grade import NativeGrade
        import full_screen_scene as scene

        self.grade = NativeGrade(self.env.unwrapped, scene)
        self.controls = -self.warmup_steps
        return result

    def step(self, action):
        if self.grade is None or self.controls >= RUN_LIMIT:
            raise RuntimeError("Reset the task before starting a new trial.")
        observations, reward, terminated, truncated, info = self.env.step(action)
        self.controls += 1
        info["sieve_score"] = self.grade.score
        if self.controls == RUN_LIMIT:
            truncated = truncated | True
        return observations, reward, terminated, truncated, info

    def close(self):
        self.env.unwrapped.sim.step = self.native_step
        self.grade = None
        return self.env.close()


def make_env(device="cuda:0", enable_cameras=True, warmup_steps=0):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    config = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)
    return ScoredEnv(
        ArenaEnvBuilder(build(enable_cameras), config).make_registered(), warmup_steps
    )
