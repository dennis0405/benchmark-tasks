from pathlib import Path
import json
import numpy as np
from layout import BASE
HERE=Path(__file__).resolve().parent
NAME='square-wood-frame'
INSTRUCTION='Square the wooden frame with the diagonal clamp. Leave the clamp in place and release both hands.'
HZ=20
MAX_STEPS=800
OBJECT_NAMES=('Rail0','Rail1','Rail2','Rail3','Fixed','Slide')

SETUP_CAMERA_EYE=(.76,-.24,1.49)
SETUP_CAMERA_TARGET=(.32,0,.80)

def make_environment(embodiment,device,control_hz=HZ):
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
    from yam_arena import ik
    cam=getattr(embodiment.camera_config,'overview_cam',None)
    if cam is not None:
        cam.width, cam.height = 960, 640
        eye=np.array(SETUP_CAMERA_EYE); f=np.array(SETUP_CAMERA_TARGET)-eye;f/=np.linalg.norm(f)
        l=np.cross([0.,0.,1.],f);l/=np.linalg.norm(l)
        cam.offset.pos=tuple(eye);cam.offset.rot=tuple(ik.matrix_to_quat_xyzw(np.column_stack((f,l,np.cross(f,l)))))
    items=task_room('station',embodiment.rig.top_cam)
    light=next(i for i,obj in enumerate(items) if isinstance(obj,DomeLight))
    items[light]=DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=820.,color=(.98,.96,.93)))
    items.append(Object(name='frame',usd_path=str(HERE/'rig.usda'),object_type=ObjectType.BASE,initial_pose=Pose(position_xyz=tuple(BASE))))
    def configure(cfg):
        env_cfg_callback(control_hz)(cfg)
        cfg.sim.dt=1/300;cfg.decimation=15;cfg.sim.render_interval=15;cfg.sim.device='cpu'
        cfg.episode_length_s=MAX_STEPS/control_hz+4.
        return cfg
    arena=IsaacLabArenaEnvironment(name='square_wood_frame',scene=Scene(items),embodiment=embodiment,env_cfg_callback=configure)
    return ArenaEnvBuilder(arena,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered()

def make_dynamics(env):
    from observer import Observer
    if not hasattr(env.unwrapped, '_frame_observer'):
        env.unwrapped._frame_observer = Observer(env)
    return env.unwrapped._frame_observer

def read_object_states(env):
    return make_dynamics(env).states()

def make_scorer():
    from scoring import Scorer
    return Scorer()

def demo():
    return {}

def make_demo_controller(env, plans):
    from controller import Controller
    observer = make_dynamics(env)
    controller = Controller()
    class Scripted:
        def action(self, t):
            left_right = []
            for side in ('left', 'right'):
                arm = env.unwrapped.scene[side + '_robot']
                q = arm.data.joint_pos.torch[0].cpu().numpy()
                left_right.append([q[arm.joint_names.index(f'joint{i}')] for i in range(1,7)])
            poses = observer.poses()
            delta = observer.geometry(poses)['difference_mm'] / 1000.
            return controller.action(t, delta, np.asarray(left_right))
    return Scripted()
