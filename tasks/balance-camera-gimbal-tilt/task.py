"""Blackwell's public YAM task interface. The rig settles under gravity after reset."""
from pathlib import Path
from scoring import HZ, Scorer
import json
NAME = 'balance-camera-gimbal-tilt'
INSTRUCTION = json.loads(Path(__file__).with_name('task.json').read_text())['instruction']
MAX_STEPS = 2400
OBJECT_NAMES = ('stand','cradle','camera','brake')
BASE = (.35, 0., .7475)
SETUP_CAMERA_EYE=(.77,-.76,1.24)
SETUP_CAMERA_TARGET=(.35,0.,.88)

def make_environment(embodiment, device, control_hz=HZ):
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
    # Increase diffuse room light so the dark camera body and fingers are visible in shadow.
    dome_index=next(i for i,obj in enumerate(objects) if isinstance(obj,DomeLight))
    objects[dome_index]=DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=600.,color=(.86,.90,.92)))
    objects.append(Object(name='gimbal',usd_path=str(Path(__file__).parent/'assets/tilt-gimbal/gimbal.usda'),
                          object_type=ObjectType.BASE,initial_pose=Pose(position_xyz=BASE)))
    standard=env_cfg_callback(control_hz)
    def configure(cfg):
        standard(cfg)
        cfg.sim.dt=1/240
        cfg.decimation=12
        cfg.sim.render_interval=12
        cfg.episode_length_s=MAX_STEPS/control_hz+3.
        # The native PhysX CPU solver exposes individual pad impulses; RTX renders the cameras.
        cfg.sim.device='cpu'
        return cfg
    config=IsaacLabArenaEnvironment(name='balance_camera_gimbal_tilt',scene=Scene(objects),embodiment=embodiment,env_cfg_callback=configure)
    return ArenaEnvBuilder(config,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered()

def make_scorer():
    return Scorer()

def make_dynamics(env):
    from observer import Observer
    if not hasattr(env.unwrapped,'_gimbal_observer'):
        env.unwrapped._gimbal_observer=Observer(env)
    return env.unwrapped._gimbal_observer

def read_object_states(env):
    return make_dynamics(env).states()

def demo():
    return {}

def make_demo_controller(env,plans):
    from controller import Controller
    return Controller(make_dynamics(env))
