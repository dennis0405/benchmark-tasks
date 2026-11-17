"""Fixed single-environment Arena scene. The demo uses these same USD builders."""

from dataclasses import replace
import isaaclab.sim as sim
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.isaaclab_arena_environment import (
    IsaacLabArenaEnvironment,
)
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from yam_arena.embodiment import BimanualYamEmbodiment
from yam_arena.rig import TABLETOP, Camera, Lens
from task import RIGHT_START
from yam_arena.scene import env_cfg_callback
from isaac_geometry import (
    MOUTH,
    OVEN_Y,
    PARK,
    STONE_Z,
    build_station,
    build_peel,
    build_food,
)


def spawn_station(prim_path, cfg, **kwargs):
    stage = sim.get_current_stage()
    build_station(stage, prim_path)
    light = sim.DomeLightCfg(intensity=650.0)
    light.func("/World/PizzaLight", light)
    return stage.GetPrimAtPath(prim_path)


def spawn_peel(prim_path, cfg, **kwargs):
    stage = sim.get_current_stage()
    build_peel(stage, prim_path)
    return stage.GetPrimAtPath(prim_path)


def spawn_pizza(prim_path, cfg, **kwargs):
    stage = sim.get_current_stage()
    build_food(stage, prim_path)
    return stage.GetPrimAtPath(prim_path)


def configure(cfg):
    cfg = env_cfg_callback(20)(cfg)
    cfg.episode_length_s = 251.0
    return cfg


def build(enable_cameras):
    def obj(name, kind, spawn, pos):
        return Object(
            name=name,
            object_type=kind,
            spawner_cfg=sim.SpawnerCfg(func=spawn),
            initial_pose=Pose(position_xyz=tuple(map(float, pos))),
        )

    assets = [
        obj(
            "oven_fixture",
            ObjectType.BASE,
            spawn_station,
            (MOUTH + 0.24, OVEN_Y, STONE_Z - 0.07),
        ),
        obj("peel", ObjectType.RIGID, spawn_peel, PARK),
        obj(
            "pizza",
            ObjectType.RIGID,
            spawn_pizza,
            (MOUTH + 0.23, OVEN_Y, STONE_Z + 0.002),
        ),
    ]
    # Vertical optical axis, world X points right; the full peel and oven fit.
    rig = replace(
        TABLETOP,
        top_cam=Camera(
            (0.62, -0.24, 2.45), (0.0, 0.0, 0.0, 1.0), "opengl", Lens.from_focal(18.0)
        ),
    )
    arms = BimanualYamEmbodiment(
        enable_cameras=enable_cameras,
        overview_cam=enable_cameras,
        contact_sensors=True,
        robot_contact_offset=0.0003,
        rig=rig,
    )
    arms.scene_config.right_robot.init_state.joint_pos.update(
        {f"joint{i}": angle for i, angle in enumerate(RIGHT_START, 1)}
    )
    return IsaacLabArenaEnvironment(
        name="turn_pizza_on_oven_deck",
        embodiment=arms,
        scene=Scene(assets),
        env_cfg_callback=configure,
    )
