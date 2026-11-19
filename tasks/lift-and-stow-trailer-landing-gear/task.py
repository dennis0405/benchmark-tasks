"""Blackwell's public YAM task interface. The rig settles under gravity after reset."""
from pathlib import Path
from scoring import Scorer

HZ=20
import json
NAME = 'load-coupled-trailer-landing-gear'
INSTRUCTION = json.loads(Path(__file__).with_name('task.json').read_text())['instruction']
MAX_STEPS = 1400
OBJECT_NAMES = ('gearbox','shoe','slider','spindle','crank')

SETUP_CAMERA_EYE=(2.06,-1.97,1.60)
SETUP_CAMERA_TARGET=(.25,-.52,.60)

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
    import numpy as np
    from yam_arena import ik
    overview=getattr(embodiment.camera_config,'overview_cam',None)
    if overview is not None:
        eye=np.array([2.06,-1.97,1.60]);target=np.array([.25,-.52,.60]);forward=target-eye;forward/=np.linalg.norm(forward)
        left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left)
        overview.offset.pos=tuple(eye);overview.offset.rot=tuple(ik.matrix_to_quat_xyzw(np.column_stack((forward,left,np.cross(forward,left)))))
    objects=task_room('station',embodiment.rig.top_cam)
    # Increase diffuse room light so the dark camera body and fingers are visible in shadow.
    dome_index=next(i for i,obj in enumerate(objects) if isinstance(obj,DomeLight))
    objects[dome_index]=DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=600.,color=(.86,.90,.92)))
    objects.append(Object(name='landing',usd_path=str(Path(__file__).parent/'assets/grounded-landing-gear/rig.usda'),
                          object_type=ObjectType.BASE,initial_pose=Pose(position_xyz=(0.,0.,0.))))
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
    config=IsaacLabArenaEnvironment(name='lift_and_stow_trailer_landing_gear',scene=Scene(objects),embodiment=embodiment,env_cfg_callback=configure)
    return ArenaEnvBuilder(config,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered()

def make_scorer():
    return Scorer()

def make_dynamics(env):
    from observer import Observer
    if not hasattr(env.unwrapped,'_landing_observer'):
        env.unwrapped._landing_observer=Observer(env)
    return env.unwrapped._landing_observer

def read_object_states(env):
    return make_dynamics(env).states()

def demo():
    return {}

def make_demo_controller(env,plans):
    from controller import Controller
    return Controller(make_dynamics(env))
