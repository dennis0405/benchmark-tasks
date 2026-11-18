"""Arena scene and first physical waypoint plan for the calibrated YAM station."""
from pathlib import Path
import json
from scoring import Scorer, FLOOR, PAN_HANDLE_Y

HERE = Path(__file__).resolve().parent
NAME = "sweep-crumbs-into-dustpan"
INSTRUCTION = json.loads((HERE / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 720
OBJECT_NAMES = ("pan", "broom", *(f"crumb{i}" for i in range(8)))
CENTER_X = .30
Y_OFFSET = -.065

def station_pose(y, height):
    return (CENTER_X, y + Y_OFFSET, FLOOR + height)

# Irregular offsets cover 150 by 133 mm; the brush is only 93 mm wide.
CRUMBS_XY = ((-.076,-.139),(-.045,-.062),(-.028,-.104),(-.012,-.026),
             (.018,-.159),(.042,-.083),(.074,-.046),(.060,-.122))
START = {"pan": station_pose(.015,.0005), "broom": (CENTER_X-.038, Y_OFFSET-.190,FLOOR+.0005)}
for i,(x,y) in enumerate(CRUMBS_XY):
    START[f"crumb{i}"] = (CENTER_X+x, Y_OFFSET+y, FLOOR+.004)

def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from yam_arena.scene import env_cfg_callback, task_room

    overview = getattr(embodiment.camera_config, "overview_cam", None)
    if overview is not None:
        import numpy as np
        from yam_arena import ik
        eye = np.array([.74,-.32,1.46])
        target = np.array([.30,-.045,.84])
        forward = target-eye
        forward /= np.linalg.norm(forward)
        left = np.cross([0.,0.,1.],forward)
        left /= np.linalg.norm(left)
        overview.offset.pos = tuple(eye)
        overview.offset.rot = tuple(ik.matrix_to_quat_xyzw(np.column_stack((forward,left,np.cross(forward,left)))))
    objects = task_room("station", embodiment.rig.top_cam)
    for name, pos in START.items():
        asset = name if name in ("pan", "broom") else f"crumb{int(name[5:])%2}"
        path = HERE / "assets" / "hand-brush-and-dustpan" / (asset+".usda")
        objects.append(Object(name=name, usd_path=str(path), object_type=ObjectType.RIGID,
                              initial_pose=Pose(position_xyz=pos)))
    standard = env_cfg_callback(control_hz)
    def configure(cfg):
        standard(cfg)
        cfg.sim.device = "cpu"
        cfg.sim.dt = 1/240
        cfg.decimation = 12
        cfg.sim.render_interval = 12
        cfg.episode_length_s = MAX_STEPS/HZ+3
        return cfg
    cfg = IsaacLabArenaEnvironment(name=NAME.replace("-","_"), scene=Scene(objects),
                                   embodiment=embodiment, env_cfg_callback=configure)
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device="cpu")).make_registered()

def demo():
    from yam_arena.demo import Waypoint
    from yam_arena.constants import FINGER_OPEN, FINGER_CLOSED

    def point(duration, x, y, z, grip, label):
        return Waypoint(duration,(CENTER_X+x,Y_OFFSET+y,FLOOR+z),grip,label)

    def approach(x,y):
        return [point(1.8,x,y,.159,FINGER_OPEN,"approach"),
                point(1.2,x,y,.107,FINGER_OPEN,"descend"),
                point(1.,x,y,.107,FINGER_CLOSED,"grip")]

    right = approach(-.038,-.190)
    def move(t,x,y,z,label):
        right.append(point(t,x,y,z,FINGER_CLOSED,label))

    for side in (-1,1):
        x=side*.038
        if side==1:
            move(2.,x,-.190,.146,"return above second patch")
            move(1.,x,-.190,.109,"lower behind scattered pieces")
        move(3.8,x,-.005,.109,"sweep towards lip")
        move(.7,x,.018,.110,"meet lip")
        move(1.2,x,.052,.115,"climb ramp")
        move(.8,x,.077,.115,"finish on deck")
        move(.65,x,.077,.136,"raise brush clear")
    move(1.4,.03,-.11,.155,"withdraw brush")
    sweep_time = sum(w.duration for w in right)-4.
    left=approach(0.,.015+PAN_HANDLE_Y)
    left.extend([point(sweep_time,0.,.015+PAN_HANDLE_Y,.107,FINGER_CLOSED,"hold pan flush"),
                 point(1.6,0.,.015+PAN_HANDLE_Y,.179,FINGER_CLOSED,"lift pan"),
                 point(3.8,0.,.015+PAN_HANDLE_Y,.179,FINGER_CLOSED,"hold loaded pan")])
    right.append(point(5.4,.03,-.11,.155,FINGER_CLOSED,"hold clear"))
    return {"left":left, "right":right}

def make_demo_controller(env,plans):
    from controller import ToolController
    return ToolController(env,plans)

def make_scorer():
    return Scorer()
