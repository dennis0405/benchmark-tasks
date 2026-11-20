"""Blackwell's public YAM task interface. The rig settles under gravity after reset."""
from pathlib import Path
from scoring import Scorer

HZ=20
import json
NAME = 'a400-full-travel'
INSTRUCTION = json.loads(Path(__file__).with_name('task.json').read_text())['instruction']
MAX_STEPS = 24000
OBJECT_NAMES = ('gearbox','shoe','slider','spindle','crank')

SETUP_CAMERA_EYE=(-1.20,-1.85,1.58)
SETUP_CAMERA_TARGET=(.37,-.29,.80)

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
        eye=np.array([-.97,-1.65,1.48]);target=np.array([.50,-.25,.83]);forward=target-eye;forward/=np.linalg.norm(forward)
        left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left)
        overview.offset.pos=tuple(eye);overview.offset.rot=tuple(ik.matrix_to_quat_xyzw(np.column_stack((forward,left,np.cross(forward,left)))))
    objects=task_room('station',embodiment.rig.top_cam)
    # Put the arms on a side bench and a separate fixed plinth; the entire crank sweep is in the aisle.
    from yam_arena.scene import _box, _Table, STATION_TABLE_COLOR
    objects=[obj for obj in objects if obj.name!='table']
    objects.extend([
        _box("side_bench",(.83,.415,.06),(-.335,.4425,.7175),STATION_TABLE_COLOR,cls=_Table),
        _box("bench_leg_one",(.075,.075,.70),(-.62,.565,.3375),(.26,.29,.30)),
        _box("bench_leg_two",(.075,.075,.70),(.0,.30,.3375),(.26,.29,.30)),
        _box("right_arm_plinth",(.17,.15,.76),(0.,-.305,.3675),(.34,.37,.38)),
    ])
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
