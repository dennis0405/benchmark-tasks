from pathlib import Path
import json
import numpy as np
from scoring import Scorer
HERE=Path(__file__).resolve().parent
NAME='inner-disc-knockout'
INSTRUCTION=json.loads((HERE/'task.json').read_text())['instruction']
HZ=20
MAX_STEPS=240
OBJECT_NAMES=('Disc','Ring')
BASE=(.30,0.,.84)
SETUP_CAMERA_EYE=(.34,.24,1.17)
SETUP_CAMERA_TARGET=(.282,.047,.84)

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
        eye=np.array(SETUP_CAMERA_EYE); f=np.array(SETUP_CAMERA_TARGET)-eye;f/=np.linalg.norm(f)
        l=np.cross([0.,0.,1.],f);l/=np.linalg.norm(l)
        cam.offset.pos=tuple(eye);cam.offset.rot=tuple(ik.matrix_to_quat_xyzw(np.column_stack((f,l,np.cross(f,l)))))
    from tool import spawn_arm_with_blade
    embodiment.scene_config.right_robot.spawn.func=spawn_arm_with_blade
    items=task_room('station',embodiment.rig.top_cam)
    light=next(i for i,obj in enumerate(items) if isinstance(obj,DomeLight))
    items[light]=DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=820.,color=(.98,.96,.93)))
    items.append(Object(name='knockout',usd_path=str(HERE/'assets/electrical-knockout/box.usda'),object_type=ObjectType.BASE,initial_pose=Pose(position_xyz=BASE)))
    def configure(cfg):
        env_cfg_callback(control_hz)(cfg)
        cfg.sim.dt=.002;cfg.decimation=25;cfg.sim.render_interval=25;cfg.sim.device='cpu'
        cfg.episode_length_s=MAX_STEPS/control_hz+4.
        return cfg
    arena=IsaacLabArenaEnvironment(name='concentric_knockout',scene=Scene(items),embodiment=embodiment,env_cfg_callback=configure)
    return ArenaEnvBuilder(arena,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered()
def make_scorer():return Scorer()
def make_dynamics(env):
    from observer import Observer
    if not hasattr(env.unwrapped,'_knockout_measure'):env.unwrapped._knockout_measure=Observer(env)
    return env.unwrapped._knockout_measure
def read_object_states(env):return make_dynamics(env).states()
def demo():return {}
def make_demo_controller(env,plans):
    from controller import Controller
    return Controller(make_dynamics(env))
