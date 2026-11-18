"""Public YAM entry point; prototype, not a qualified task yet."""
from pathlib import Path
import json
from scoring import HZ, Scorer

NAME='pipe-pressure-leak-test'
INSTRUCTION=json.loads(Path(__file__).with_name('task.json').read_text())['instruction']
MAX_STEPS=4000
OBJECT_NAMES=('base','lever','isolation','drain','selector')
BASE=(0.,0.,.7475)
SETUP_CAMERA_EYE=(1.02,-.85,1.48)
SETUP_CAMERA_TARGET=(.37,.04,.93)

def make_environment(embodiment,device,control_hz=HZ):
    import numpy as np
    from yam_arena import ik
    eye=np.array([-.15,-.9,1.60]);target=np.array([.41,.08,.95])
    forward=(target-eye)/np.linalg.norm(target-eye)
    left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left)
    up=np.cross(forward,left)
    embodiment.camera_config.overview_cam.offset.pos=tuple(eye)
    embodiment.camera_config.overview_cam.offset.rot=tuple(ik.matrix_to_quat_xyzw(np.column_stack([forward,left,up])))
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.pose import Pose
    from yam_arena.scene import task_room,env_cfg_callback
    objects=task_room('station',embodiment.rig.top_cam)
    for i,o in enumerate(objects):
        if isinstance(o,DomeLight): objects[i]=DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=600.,color=(.86,.90,.92)))
    objects.append(Object(name='hydro',usd_path=str(Path(__file__).parent/'assets/hydrostatic-test-pump/pump.usda'),object_type=ObjectType.BASE,initial_pose=Pose(position_xyz=BASE)))
    standard=env_cfg_callback(control_hz)
    def configure(cfg):
        standard(cfg);cfg.sim.dt=1/240;cfg.decimation=12;cfg.sim.render_interval=12
        cfg.episode_length_s=MAX_STEPS/control_hz+3;cfg.sim.device='cpu'
        return cfg
    c=IsaacLabArenaEnvironment(name=NAME.replace('-','_'),scene=Scene(objects),embodiment=embodiment,env_cfg_callback=configure)
    import gymnasium as gym
    class SpecimenFixture(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            self.unwrapped.pipe_sound_specimen = bool((seed or 0) % 2)
            return super().reset(seed=seed, options=options)
    return SpecimenFixture(ArenaEnvBuilder(c,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered())

def make_dynamics(env):
    from observer import Observer
    if not hasattr(env.unwrapped,'_hydro_observer'): env.unwrapped._hydro_observer=Observer(env)
    return env.unwrapped._hydro_observer

def read_object_states(env):return make_dynamics(env).states()
def make_scorer():return Scorer()
def demo():return {}

def make_demo_controller(env,plans):
    from controller import Controller
    return Controller(make_dynamics(env))
