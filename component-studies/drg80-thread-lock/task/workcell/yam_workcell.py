"""Physical bimanual adjustment trial with an instrumented, mounted workcell."""

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", type=Path)
parser.add_argument("--continuous-seat", action="store_true")
parser.add_argument("--transmitted-reaction", action="store_true")
parser.add_argument("--hz", type=int, choices=(240, 480), default=240)
tips = parser.add_mutually_exclusive_group()
tips.add_argument(
    "--square-bit", action="store_true", help="reproduce the original unbevelled bit"
)
parser.add_argument("--contact-guided", action="store_true")
tips.add_argument("--long-lead", action="store_true")
parser.add_argument(
    "--adaptive-stroke",
    action="store_true",
    help="End torque early at the physical wrist limit and reindex in free space.",
)
parser.add_argument("--seated-regrasp", action="store_true")
parser.add_argument(
    "--depth-stop",
    action="store_true",
    help="Use the removable 4.75 mm bit depth stop.",
)
parser.add_argument("--supported-turn", action="store_true")
parser.add_argument("--align-before-feed", action="store_true")
parser.add_argument("--wrench-preload-n", type=float, default=5.0)
parser.add_argument(
    "--check-valve-close",
    action="store_true",
    help="Test the real YAM lever closure separately on the unadjusted workcell.",
)
parser.add_argument(
    "--push-valve",
    action="store_true",
    help="Press either side of the valve handle with closed fingertips.",
)
parser.add_argument(
    "--check-wrench-stroke",
    action="store_true",
    help="Test a single free wrench stroke on the unadjusted mechanism.",
)
parser.add_argument(
    "--check-valve-reversal",
    action="store_true",
    help="Check both physical detents without claiming a calibrated endpoint.",
)
parser.add_argument(
    "--guided-wrench",
    action="store_true",
    help="Use tool-jaw lead-ins and verify the physical nut is between the flats before torque.",
)
parser.add_argument(
    "--replay-actions",
    type=Path,
    help="Replay logged 14-D joint commands from reset through normal YAM actions.",
)
parser.add_argument("--quick-valve", action="store_true")
parser.add_argument(
    "--check-three-valve-cycles",
    action="store_true",
    help="Measure three real reversals at the original unadjusted stop.",
)
parser.add_argument(
    "--reserve-retreat",
    action="store_true",
    help="Reserve wrist travel for pulling out of the socket while turning.",
)
parser.add_argument("--observe-reset", action="store_true")
parser.add_argument("--room-damage", action="store_true")
parser.add_argument("--solver", choices=("tgs", "pgs"), default="tgs")
parser.add_argument("--contact-last", action="store_true")
parser.add_argument("--driver-correction-rad", type=float, default=0.0)
parser.add_argument(
    "--external-mount",
    action="store_true",
    help="Attach the workcell with a finite-strength external fixed joint.",
)
parser.add_argument("--implicit-axial", action="store_true")
parser.add_argument("--native-friction", action="store_true")
parser.add_argument("--rise-seconds", type=float, default=0.0)

args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.video:
    args.video.parent.mkdir(parents=True, exist_ok=True)
app = AppLauncher(headless=True, enable_cameras=bool(args.video)).app

import numpy as np
import torch
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from yam_arena import ik, scene, station, task_file
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import grip_rotation
from yam_arena.embodiment import BimanualYamEmbodiment


def task_assets():
    assets = args.workspace / "task/assets"
    return assets if assets.is_dir() else args.workspace / "task-prototype/assets"


def tool_asset(filename):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    path = args.output.parent / (args.output.stem + "_" + filename + ".usda")
    stage = Usd.Stage.CreateNew(str(path.resolve()))
    root = UsdGeom.Xform.Define(stage, "/Tool")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(
        str((task_assets() / "tools" / (filename + ".usda")).resolve())
    )
    if filename == "hex_driver":
        author_bit(
            UsdGeom.Mesh(stage.GetPrimAtPath("/Tool/HexTip")),
            bit_chamfer,
        )
        if args.depth_stop:
            from depth_stop import add_depth_stop

            add_depth_stop(stage, "/Tool")
    if filename == "open_wrench" and args.guided_wrench:
        from wrench_guide import author_wrench_guide

        author_wrench_guide(stage)
    materials = {}
    for name, mus in (("Polymer", (1.2, 1.0)), ("Steel", (0.45, 0.3))):
        mat = UsdShade.Material.Define(stage, "/Tool/Physics" + name)
        props = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        props.CreateStaticFrictionAttr(mus[0])
        props.CreateDynamicFrictionAttr(mus[1])
        props.CreateRestitutionAttr(0.0)
        materials[name] = mat
    for prim in Usd.PrimRange(root.GetPrim()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            key = (
                "Polymer"
                if prim.GetName() == "Grip" and filename == "hex_driver"
                else "Steel"
            )
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                materials[key], materialPurpose="physics"
            )
    stage.GetRootLayer().Save()
    return str(path.resolve())


def load_station():
    manifest = args.workspace / "input/fixture-geometry-contract.json"
    if not manifest.is_file():
        manifest = (
            args.workspace / "task-prototype/assets/fixture-geometry-contract.json"
        )
    rack = json.loads(manifest.read_text())["rack"]
    rig = station.rig()
    embodiment = BimanualYamEmbodiment(
        rig=rig,
        enable_cameras=bool(args.video),
        cam_width=480,
        cam_height=272,
        contact_sensors=True,
    )
    objects = list(scene.task_room("station", rig.top_cam))
    sides = ("left", "right")
    names = ("driver", "wrench")
    origins = (rack["D_reset_tip_world_m"], rack["W_reset_nut_axis_world_m"])
    grasps = (
        np.array(origins[0]) + (0, 0.100, 0),
        np.array(origins[1]) + (-0.085, 0, 0),
    )
    for i, (label, filename, pose, rotation) in enumerate(
        zip(
            names,
            ("hex_driver", "open_wrench"),
            origins,
            ((-math.sqrt(0.5), 0, 0, math.sqrt(0.5)), (0, 0, 0, 1)),
        )
    ):
        objects.append(
            scene.Object(
                name=label,
                object_type=scene.ObjectType.RIGID,
                usd_path=tool_asset(filename),
                initial_pose=scene.Pose(
                    position_xyz=tuple(pose), rotation_xyzw=rotation
                ),
            )
        )
        key = ("D", "W")[i]
        low, high = np.array(rack[key + "_cradle_bounds_world_m"])
        base_size = rack["cradle_base_boxes_m"][key]
        base_z = low[2] + base_size[2] / 2
        objects.append(
            scene.task_object(
                task_file.Box(
                    label + "_rack",
                    pos=(*((low + high) / 2)[:2], base_z),
                    size=tuple(base_size),
                )
            )
        )
        for j, (x, y) in enumerate(rack[key + "_support_centers_xy_m"]):
            top = (
                rack["D_support_top_z_m"][("shaft", "grip_end")[j]]
                if i == 0
                else rack["W_support_top_z_m"]
            )
            width = (
                rack["D_support_cross_section_m"]
                if i == 0
                else rack["support_cross_section_m"]
            )
            bottom = base_z + base_size[2] / 2
            objects.append(
                scene.task_object(
                    task_file.Box(
                        f"{label}_support_{j}",
                        pos=(x, y, (top + bottom) / 2),
                        size=(*width, top - bottom),
                    )
                )
            )
    return objects, rig, embodiment, sides, names, grasps


from scipy.spatial.transform import Rotation
from isaaclab_physx.physics.physx_manager import IsaacEvents, PhysxManager
from build_bench import build
from native_bench import Bench
from hex_fit import CHAMFER, LONG_CHAMFER, author_bit, hex_fit

bit_chamfer = 0.0 if args.square_bit else LONG_CHAMFER if args.long_lead else CHAMFER
from contact_limits import contact_bounds
from tool_bounds import ToolBounds
from adjustment_volume import AdjustmentVolume
from nut_retention import nut_retention
from room_contacts import RoomContacts
from workcell_clearance import WorkcellClearance
from service_paths import wrench_flat, lever_grasp, lever_pose


def checkpoint_actions(output, trace):
    # Persist only completed controls. A provider stop may bypass finally;
    # the last prefix can still be replayed from the original reset.
    prefix = output.with_suffix(".actions.partial.json")
    writing = prefix.with_suffix(".tmp")
    with writing.open("w") as handle:
        json.dump(
            [
                {"phase": r["phase"], "action": r["action"]}
                for r in trace
            ],
            handle,
            allow_nan=False,
            separators=(",", ":"),
        )
        handle.flush()
        os.fsync(handle.fileno())
    writing.replace(prefix)


def main():
    objects, rig, embodiment, sides, names, grasps = load_station()
    fixture = args.output.with_suffix(".usda").resolve()
    build(
        fixture,
        task_assets(),
        continuous_seat=args.continuous_seat,
        implicit_axial=args.implicit_axial,
        external_mount=args.external_mount,
    )
    objects.append(
        scene.Object(
            name="gripper_bench",
            object_type=scene.ObjectType.BASE,
            usd_path=str(fixture),
            initial_pose=scene.Pose(position_xyz=(0, 0, 0), rotation_xyzw=(0, 0, 0, 1)),
        )
    )

    def physics_config(cfg):
        cfg = scene.env_cfg_callback(20)(cfg)
        cfg.sim.dt = 1.0 / args.hz
        cfg.decimation = args.hz // 20
        cfg.sim.render_interval = cfg.decimation
        if cfg.sim.physics is None:
            from isaaclab_physx.physics import PhysxCfg

            cfg.sim.physics = PhysxCfg()
        cfg.sim.physics.solver_type = int(args.solver == "tgs")
        cfg.sim.physics.solve_articulation_contact_last = args.contact_last
        cfg.sim.physics.enable_external_forces_every_iteration = args.solver == "tgs"
        return cfg

    arena = scene.IsaacLabArenaEnvironment(
        name="drg80_instrumented_yam",
        scene=scene.Scene(objects),
        embodiment=embodiment,
        env_cfg_callback=physics_config,
    )
    env = ArenaEnvBuilder(
        arena, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cuda:0")
    ).make_registered()
    results = {
        "completed": [],
        "physics_ticks": 0,
        "scope": "physical YAM workcell commissioning, no competition grade",
        "solver": args.solver,
        "rate_hz": args.hz,
        "contact_guided": args.contact_guided,
        "supported_turn": args.supported_turn,
        "align_before_feed": args.align_before_feed,
        "wrench_preload_target_n": args.wrench_preload_n,
        "seated_regrasp": args.seated_regrasp,
        "adaptive_stroke": args.adaptive_stroke,
        "reserve_retreat": args.reserve_retreat,
        "depth_stop": args.depth_stop,
        "guided_wrench": args.guided_wrench,
        "external_mount": args.external_mount,
        "bit_chamfer_m": bit_chamfer,
        "implicit_axial": args.implicit_axial,
        "transmitted_reaction": args.transmitted_reaction,
        "driver_correction_rad": args.driver_correction_rad,
    }
    trace = []
    callbacks = []
    writer = None
    room_contacts = None
    grade_bound = None
    break_subscription = None
    try:
        env.reset()
        base = env.unwrapped
        if args.video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(args.video), fps=10, codec="libx264", macro_block_size=2
            )
        root = next(
            str(x.GetPath().GetParentPath())
            for x in base.sim.stage.Traverse()
            if x.GetName() == "sz"
        )
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools

        results["joint_breaks"] = []

        def joint_event(event):
            if event.type == int(SimulationEvent.JOINT_BREAK):
                path = str(
                    PhysicsSchemaTools.decodeSdfPath(*event.payload["jointPath"])
                )
                results["joint_breaks"].append(
                    dict(path=path, observed_tick=results["physics_ticks"])
                )
                results.setdefault(
                    "callback_error", f"required workcell joint broke: {path}"
                )

        break_subscription = (
            get_physx_interface()
            .get_simulation_event_stream_v2()
            .create_subscription_to_pop(joint_event)
        )
        targets = [base.scene[name].root_physx_view.prim_paths[0] for name in names]
        bench = Bench(
            root,
            round(1 / base.physics_dt),
            tools=targets,
            native_friction=args.native_friction,
            transmitted_reaction=args.transmitted_reaction,
            implicit_axial=args.implicit_axial,
            rise_seconds=args.rise_seconds,
        )

        from runpy import run_path

        scoring = run_path(
            str(Path(__file__).resolve().parent.parent / "stop_lock_score.py")
        )
        from native_grade import NativeGradeBound

        grade_bound = NativeGradeBound(scoring, round(1 / base.physics_dt))
        powered = scoring["ElasticLockMotion"](round(1 / base.physics_dt))
        protected = [
            root + "/" + name
            for name in ("screw", "nut", "retainer", "left_jaw", "right_jaw", "piston")
        ]
        external = list(targets)
        for side in sides:
            links = base.scene[side + "_robot"].root_physx_view.link_paths[0]
            external.extend(links)
        contacts = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            protected,
            filter_patterns=[external for _ in protected],
            max_contact_data_count=2048,
        )

        driver_impulse = np.zeros(3)

        rack = json.loads(
            (task_assets() / "fixture-geometry-contract.json").read_text()
        )["rack"]
        cradle = [np.array(rack[k + "_cradle_bounds_world_m"]) for k in ("D", "W")]
        geometry = [
            ToolBounds(args.output.with_name(args.output.stem + "_" + name + ".usda"))
            for name in ("hex_driver", "open_wrench")
        ]
        tool_view = PhysxManager.get_physics_sim_view().create_rigid_body_view(targets)
        tool_paths = list(tool_view.prim_paths)
        if set(tool_paths) != set(targets):
            raise ValueError("tool containment body view did not resolve both tools")
        tool_order = [tool_paths.index(path) for path in targets]
        volumes = [AdjustmentVolume(body) for body in geometry]
        finger_views = []
        for side in sides:
            view = base.scene[side + "_robot"].root_physx_view
            dofs = list(view.get_metatype(0).dof_names)
            finger_views.append((view, [dofs.index("joint7"), dofs.index("joint8")]))
        guides = [root + "/" + k for k in ("screw", "nut", "retainer")]
        guide_view = PhysxManager.get_physics_sim_view().create_rigid_body_view(guides)
        guide_paths = list(guide_view.prim_paths)
        if set(guide_paths) != set(guides):
            raise ValueError("nut retention view did not resolve all three bodies")
        guide_order = [guide_paths.index(path) for path in guides]
        if args.room_damage:
            room_contacts = RoomContacts(
                PhysxManager.get_physics_sim_view(),
                base.sim.stage,
                round(1 / base.physics_dt),
            )
        clear_view = WorkcellClearance(
            PhysxManager.get_physics_sim_view(),
            base.sim.stage,
            root,
            targets,
            [
                base.scene[side + "_robot"].root_physx_view.link_paths[0]
                for side in sides
            ],
        )
        from tool_extraction import certify_extraction

        prior_rack = None
        if any(g.handle is None for g in geometry):
            raise ValueError("tool has no designated physical grasp handle")

        def pre(*args, **kwargs):
            if "callback_error" in results:
                return
            try:
                bench.step()
                driver_impulse[:] += bench.driver_contact_force * base.physics_dt
            except Exception as e:
                results["callback_error"] = repr(e)

        latest = None
        reference_frame = bench.bodies["frame"].get_transforms().numpy()[0].copy()

        def post(*args, **kwargs):
            nonlocal latest, prior_rack
            results["physics_ticks"] += 1
            r = bench.read()
            latest = r
            if room_contacts is not None:
                q, v = bench.state()
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    results.setdefault(
                        "callback_error", "nonfinite workcell joint state"
                    )
                for view, _ in finger_views:
                    if (
                        not np.isfinite(view.get_dof_positions().numpy()).all()
                        or not np.isfinite(view.get_dof_velocities().numpy()).all()
                    ):
                        results.setdefault(
                            "callback_error", "nonfinite YAM joint state"
                        )
                maximum = 3 + math.degrees(300 * (bench.gap0[1] + r["sz"]))
                r["configured_max_opening_deg"] = maximum
                if not 20 <= maximum <= 180:
                    results.setdefault(
                        "callback_error",
                        f"configured jaw opening outside physical range: {maximum}",
                    )
            poses = tool_view.get_transforms().numpy()[tool_order]
            contained = [
                body.in_cradle(pose[:3], Rotation.from_quat(pose[3:]).as_matrix(), box)
                for body, pose, box in zip(geometry, poses, cradle)
            ]
            if room_contacts is not None:
                violation = room_contacts.observe(results["physics_ticks"], contained)
                if violation:
                    results.setdefault(
                        "callback_error",
                        f"room contact or tool-loss failure: {violation}",
                    )
            if contained != prior_rack:
                results.setdefault("rack_transitions", []).append(
                    dict(
                        tick=results["physics_ticks"],
                        driver=contained[0],
                        wrench=contained[1],
                    )
                )
                prior_rack = contained
            r["tools_racked"] = all(contained)
            bounds = (
                clear_view.world_bounds()
                if all(contained) or room_contacts is not None
                else None
            )
            if room_contacts is not None:
                lo, hi = [a[clear_view.fixture_shapes] for a in bounds]
                certificates = []
                for i, (pose, geom, record) in enumerate(
                    zip(poses, geometry, room_contacts.last_tool_observations)
                ):
                    rot = Rotation.from_quat(pose[3:]).as_matrix()
                    handle = geom.handle @ rot.T + pose[:3]
                    certificates.append(
                        certify_extraction(
                            np.array(record["centre"]),
                            handle,
                            rot[:, 2 if i == 0 else 0],
                            lo,
                            hi,
                        )
                    )
                r["tool_extraction_certificates"] = certificates
                if any(c is None for c in certificates):
                    results.setdefault(
                        "first_unresolved_extraction_tick", results["physics_ticks"]
                    )
            if all(contained):
                r["robot_tool_clearance_lower_bound_m"] = clear_view.lower_bound(bounds)
                r["clearance_certified_30mm"] = (
                    r["robot_tool_clearance_lower_bound_m"] >= 0.030
                )

            frame = bench.bodies["frame"].get_transforms().numpy()[0]
            retention = nut_retention(*guide_view.get_transforms().numpy()[guide_order])
            r.update(retention)
            if not r["nuts_retained"]:
                results.setdefault(
                    "callback_error",
                    f"nut retention failed at tick {results['physics_ticks']}: {retention}",
                )
            frame_rotation = Rotation.from_quat(frame[3:]).as_matrix()
            r["adjustment_clearances_m"] = [
                volume.distance(
                    pose[:3],
                    Rotation.from_quat(pose[3:]).as_matrix(),
                    frame[:3],
                    frame_rotation,
                )
                for pose, volume in zip(poses, volumes)
            ]
            r["tools_outside_adjustment"] = min(r["adjustment_clearances_m"]) > 0.0
            fingers = np.concatenate(
                [view.get_dof_positions().numpy()[0, ids] for view, ids in finger_views]
            )
            r["measured_finger_joints_m"] = fingers.tolist()
            r["fingers_open"] = bool(
                np.all(fingers >= -0.04705) and np.all(fingers <= -0.03756)
            )
            moved = float(np.linalg.norm(frame[:3] - reference_frame[:3]))
            tilted = math.degrees(
                2
                * math.acos(float(np.clip(abs(frame[3:] @ reference_frame[3:]), 0, 1)))
            )
            if not np.isfinite(frame).all() or moved > 0.010 or tilted > 5.0:
                results.setdefault(
                    "callback_error",
                    f"fixture mount displaced: {moved} m, {tilted} degrees",
                )
            if (
                r["jl"] < math.radians(-91)
                or r["jl"] > math.radians(1)
                or r["jr"] < math.radians(-1)
                or r["jr"] > math.radians(91)
            ):
                results.setdefault(
                    "callback_error", "jaw exceeded physical joint range"
                )
            contact_data = contacts.get_contact_data(dt=base.physics_dt)
            count = contact_data[4].numpy()
            released = not np.any(count)
            r["released"] = bool(released)
            if not released:
                force, pen = contact_bounds(
                    contact_data[0].numpy(),
                    contact_data[3].numpy(),
                    count,
                    contact_data[5].numpy(),
                )
                peaks = results.setdefault(
                    "protected_contact_peaks", dict(force_n=0.0, penetration_m=0.0)
                )
                peaks["force_n"] = max(peaks["force_n"], force)
                peaks["penetration_m"] = max(peaks["penetration_m"], pen)
                # Every intended and unintended pair is subject to both of these
                # outer limits. Absence of a violation is not a room damage grade.
                if force > 250 or pen > 0.002:
                    results.setdefault(
                        "callback_error",
                        f"protected contact exceeded universal limit at tick {results['physics_ticks']}: {force:.3f} N, {pen:.6f} m",
                    )
            violated = powered.observe(
                results["physics_ticks"],
                r["actuator_n"],
                r["sz"],
                r["st"],
                r["st"] + r["nt"],
                r["stop_n"],
                released,
                abs(r["angular_speed"]) <= 2 and abs(r["piston_speed"]) <= 0.005,
            )
            if violated:
                results.setdefault("powered_fault", powered.first_failure)
                results.setdefault(
                    "callback_error",
                    f"powered adjustment fault {powered.first_failure}",
                )
            if (
                abs(r["sz"]) > 0.0049
                or abs(r["nz"]) > 0.0051
                or r["seat_gap"] < -0.00010
            ):
                results.setdefault(
                    "callback_error",
                    f"native guide/seat failed at tick {results['physics_ticks']}: sz={r['sz']}, nz={r['nz']}, gap={r['seat_gap']}",
                )

            r["score_upper_bound"] = grade_bound.observe(
                r,
                results["physics_ticks"],
                bool(
                    room_contacts
                    and (
                        room_contacts.known_damage.failure
                        or any(m.failure for m in room_contacts.drop)
                    )
                ),
            )

        def observe_after_step(*args, **kwargs):
            if "callback_error" in results:
                return
            try:
                post()
            except Exception as exc:
                results["callback_error"] = (
                    f"native observer failed at tick {results['physics_ticks']}: {exc!r}"
                )

        for event, callback in (
            (IsaacEvents.PRE_PHYSICS_STEP, pre),
            (IsaacEvents.POST_PHYSICS_STEP, observe_after_step),
        ):
            callbacks.append(
                PhysxManager.register_callback(
                    callback,
                    event,
                    order=0,
                    name=event.name + "_drg80",
                    wrap_weak_ref=False,
                )
            )

        def tool(i):
            t = base.scene[names[i]].root_physx_view.get_transforms().numpy()[0]
            return t[:3].copy(), Rotation.from_quat(t[3:]).as_matrix()

        def joints(i):
            return (
                base.scene[sides[i] + "_robot"]
                .data.joint_pos.torch[0, :6]
                .cpu()
                .numpy()
                .copy()
            )

        commands = np.stack([joints(i) for i in range(2)])

        def move(label, goals, grips, steps):
            initial = commands.copy()
            target = np.asarray(goals)
            for j in range(1, steps + 1):
                t = j / steps
                q = initial + (target - initial) * t * t * (3 - 2 * t)
                action = np.r_[q[0], grips[0], q[1], grips[1]]
                if len(trace) >= 4800:
                    raise ValueError("native task control horizon reached")
                driver_impulse.fill(0.0)
                before = results["physics_ticks"]
                env.step(
                    torch.as_tensor(
                        action[None], device=base.device, dtype=torch.float32
                    )
                )
                if "callback_error" in results:
                    raise RuntimeError(results["callback_error"])
                if results["physics_ticks"] - before != round(
                    base.step_dt / base.physics_dt
                ):
                    raise RuntimeError("missing native physics callbacks")
                poses = [tool(i) for i in range(2)]
                rec = dict(
                    phase=label,
                    action=action.tolist(),
                    measured_arm_joints_rad=[joints(i).tolist() for i in range(2)],
                    driver_contact_n=(driver_impulse / base.step_dt).tolist(),
                    fixture=latest,
                    tools=[p.tolist() for p, _ in poses],
                    tool_quats_xyzw=[
                        Rotation.from_matrix(r).as_quat().tolist() for _, r in poses
                    ],
                )
                trace.append(rec)
                if len(trace) % 80 == 0:
                    checkpoint_actions(args.output, trace)
                if writer and len(trace) % 2 == 0:
                    rgb = base.scene["top_cam"].data.output["rgb"]
                    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
                    writer.append_data(rgb[0, ..., :3].cpu().numpy().astype(np.uint8))
            commands[:] = target
            if not results["completed"] or results["completed"][-1] != label:
                results["completed"].append(label)
            if steps == 1:
                if len(trace) % 80 == 0:
                    tip = rec["tools"][0]
                    m = rec["fixture"]
                    print(
                        "TRACK",
                        len(trace),
                        label,
                        "driver_tip_m",
                        tip,
                        "screw_rad",
                        m["st"],
                        "nut_relative_rad",
                        m["nt"],
                        "seat_n",
                        m["seat_n"],
                        flush=True,
                    )
                return
            print(
                label,
                trace[-1]["tools"],
                "screw_rad",
                round(trace[-1]["fixture"]["st"], 4),
                flush=True,
            )

        if args.replay_actions:
            import hashlib

            source = args.replay_actions.resolve()
            if source == args.output.with_suffix(".trace.json").resolve():
                raise ValueError("input trajectory and output trace must differ")
            data = source.read_bytes()
            rows = json.loads(data)
            if not 1 <= len(rows) <= 4800:
                raise ValueError("replay length outside task horizon")
            results["replay"] = dict(
                source=source.name,
                sha256=hashlib.sha256(data).hexdigest(),
                actions=len(rows),
            )
            detents = []
            for row in rows:
                action = np.asarray(row["action"], dtype=float)
                if (
                    action.shape != (14,)
                    or not np.isfinite(action).all()
                    or not np.all((action[[6, 13]] >= 0) & (action[[6, 13]] <= 1))
                ):
                    raise ValueError("invalid normal YAM command in saved trajectory")
                move(
                    str(row["phase"]),
                    np.stack([action[:6], action[7:13]]),
                    action[[6, 13]],
                    1,
                )
                angle = math.degrees(latest["lever"])
                detent = 1 if angle >= 25 else -1 if angle <= -25 else 0
                if not detents or detents[-1]["detent"] != detent:
                    detents.append(dict(action=len(trace), detent=detent))
            results["replay"].update(
                detent_transitions=detents,
                terminal_jaw_deg=latest["angle"],
                terminal_valve_deg=math.degrees(latest["lever"]),
                terminal_piston_n=latest["actuator_n"],
            )
            return

        def run_valve(grip_state, cold_test=False):
            # A real parallel-jaw grasp on the instrument handle drives the pivot.
            pivot = np.array(bench.poses()["lever"][:3])

            if args.push_valve:
                from valve_push import push_valve

                directions = (
                    ([-1, 1] if args.check_valve_reversal else [-1])
                    if cold_test
                    else [1, -1, 1, -1, 1, -1, 1]
                )
                if cold_test and args.check_three_valve_cycles:
                    directions = [-1, 1] * 3
                for direction in directions:
                    sample = push_valve(
                        direction,
                        pivot,
                        bench,
                        rig.bases,
                        joints,
                        commands,
                        grip_state,
                        move,
                        quick=args.quick_valve,
                    )
                    results.setdefault("valve_pushes", []).append(sample)
                    if cold_test and direction == -1:
                        results["cold_close"] = sample
                    valid = (
                        direction * sample["lever_deg"] >= 25
                        and direction * sample["actuator_n"] > 1
                    )
                    endpoint = (
                        abs(sample["jaw_deg"] - 90) <= 3
                        if direction == 1
                        else sample["jaw_deg"] <= 10
                    )
                    if cold_test and direction == 1:
                        endpoint = sample["jaw_deg"] > 10
                    if not valid or not endpoint:
                        raise ValueError(
                            f"side press missed physical detent or jaw endpoint: {sample}"
                        )
                if not cold_test:
                    results["actual_open_endpoints_deg"] = [
                        r["jaw_deg"] for r in results["valve_pushes"][::2]
                    ]
                    move("equipment_clear", [READY, READY], [1, 1], 60)
                    move("clear_dwell", [READY, READY], [1, 1], 30)
                return
            residual, hand, neutral, tilt, yaw = lever_grasp(
                pivot, [joints(0), joints(1)], rig.bases
            )
            results["lever_plan"] = dict(
                arm=sides[hand],
                neutral_tilt_deg=tilt,
                yaw_deg=yaw,
                weighted_ik_error_m=residual,
            )
            arm = rig.bases[sides[hand]]

            def handle_motion(label, deg, steps, extra_z=0.0):
                goals = commands.copy()
                pos, rot = lever_pose(pivot, deg, neutral, extra_z)
                q, ep, ea = ik.solve(
                    pos,
                    ik.matrix_to_quat_xyzw(rot),
                    q_init=joints(hand),
                    base_pos=arm,
                    iters=180,
                )
                if ep > 0.003 or ea > 0.04:
                    raise ValueError("lever IK infeasible")
                goals[hand] = q
                move(label, goals, grip_state, steps)
                # Factory joint gains deflect under gravity and contact. Correct
                # actual TCP tracking with bounded joint-position actions.
                for _ in range(4):
                    actual, orientation = ik.tcp_pose(joints(hand), arm)
                    error = pos - actual
                    rotation_error = Rotation.from_matrix(
                        rot @ orientation.T
                    ).as_rotvec()
                    if (
                        np.linalg.norm(error) < 0.001
                        and np.linalg.norm(rotation_error) < 0.025
                    ):
                        break
                    cp, cr = ik.tcp_pose(commands[hand], arm)
                    corr = Rotation.from_rotvec(
                        np.clip(0.5 * rotation_error, -0.025, 0.025)
                    ).as_matrix()
                    goals[hand], _, _ = ik.solve(
                        cp + np.clip(0.6 * error, -0.002, 0.002),
                        ik.matrix_to_quat_xyzw(corr @ cr),
                        q_init=commands[hand],
                        base_pos=arm,
                        iters=40,
                    )
                    move(label + "_trim", goals, grip_state, 3)

            # A 16 mm handle needs a 32 mm approach opening. Wide-open pads
            # otherwise sweep the neighbouring jaw when approached from the side.
            grip_state[hand] = 0.016 / abs(FINGER_OPEN)
            move("valve_preshape", commands.copy(), grip_state, 15)
            handle_motion("valve_above", 0, 40, 0.035)
            handle_motion("valve_grasp", 0, 30)
            grip_state[hand] = (0.008 - 0.0016) / abs(FINGER_OPEN)
            handle_motion("valve_grip_close", 0, 20)
            if cold_test:
                for degree in range(0, -31, -5):
                    handle_motion("cold_valve_close", degree, 5)
                handle_motion("cold_close_dwell", -30, 24)
                final = bench.read()
                result = dict(
                    lever_deg=math.degrees(final["lever"]),
                    jaw_deg=final["angle"],
                    actuator_n=final["actuator_n"],
                )
                results["cold_close"] = result
                if not (
                    -35 <= result["lever_deg"] <= -25
                    and result["jaw_deg"] <= 10
                    and result["actuator_n"] < -1
                ):
                    raise ValueError(f"physical close detent not attained: {result}")
                return
            endpoints = []
            for cycle in range(4):
                for degree in range(0 if cycle == 0 else -30, 31, 5):
                    handle_motion("valve_open", degree, 5)
                handle_motion("open_dwell", 30, 24)
                sample = bench.read()
                if abs(sample["angle"] - 90) > 3 or sample["stop_n"] < 0.1:
                    raise ValueError(
                        "open endpoint missing native contact or outside tolerance"
                    )
                endpoints.append(sample["angle"])
                if cycle < 3:
                    for degree in range(30, -31, -5):
                        handle_motion("valve_close", degree, 5)
                    handle_motion("close_dwell", -30, 24)
                    if bench.read()["angle"] > 10:
                        raise ValueError("jaw did not physically close")
            results["actual_open_endpoints_deg"] = endpoints
            grip_state[hand] = 1
            handle_motion("valve_release", 30, 20)
            move("equipment_clear", [READY, READY], grip_state, 60)
            move("clear_dwell", [READY, READY], grip_state, 30)

        if (
            args.check_valve_close
            or args.check_valve_reversal
            or args.check_three_valve_cycles
        ):
            move("unpowered_reset_settle", commands.copy(), [1, 1], 12)
            run_valve([1.0, 1.0], cold_test=True)
            return
        if args.observe_reset:
            move("observe_reset", commands.copy(), [1.0, 1.0], 12)
            results["reset_observation"] = {
                key: latest[key]
                for key in (
                    "robot_tool_clearance_lower_bound_m",
                    "clearance_certified_30mm",
                    "nut_guide_overlap_m",
                    "retainer_installed_error_m",
                    "nuts_retained",
                    "adjustment_clearances_m",
                    "tools_outside_adjustment",
                    "tools_racked",
                    "measured_finger_joints_m",
                    "fingers_open",
                )
            }
            return
        plans = []
        for side, grasp, yaw in zip(sides, grasps, (math.pi / 2, 0)):
            answers = []
            for az in (yaw, yaw + math.pi):
                q = np.array(READY)
                worst = 0
                path = []
                for h in (0.05, 0.0, 0.05):
                    q, ep, ea = ik.solve(
                        grasp + (0, 0, PAD_TIP_PAST_TCP + h),
                        ik.matrix_to_quat_xyzw(grip_rotation(az)),
                        q_init=q,
                        base_pos=rig.bases[side],
                        iters=300,
                    )
                    path.append(q)
                    worst = max(worst, ep + ea * 0.05)
                answers.append((worst, path))
            error, path = min(answers, key=lambda x: x[0])
            if error > 0.002:
                raise ValueError(f"rack IK {side} {error}")
            plans.append(path)
        # A 24 mm driver and 12 mm wrench do not need a commanded zero
        # aperture. Leave 1.6 mm of pad compression per finger (~8 N at
        # the stock 5000 N/m gain) to avoid exciting the free handle.
        held_openings = [
            (0.012 - 0.0016) / abs(FINGER_OPEN),
            (0.006 - 0.0016) / abs(FINGER_OPEN),
        ]
        for label, idx, g, n in (
            ("settle", None, 1, 24),
            ("approach", 0, 1, 60),
            ("descend", 1, 1, 45),
            ("close", 1, 0, 25),
            ("lift", 2, 0, 50),
            ("hold", 2, 0, 25),
        ):
            move(
                label,
                [p[idx] for p in plans] if idx is not None else [READY, READY],
                [1.0, 1.0] if g else held_openings,
                n,
            )
        lift = np.array(trace[-1]["tools"])[:, 2] - np.array(trace[23]["tools"])[:, 2]
        results["held_lift_m"] = lift.tolist()
        if min(lift) < 0.030:
            raise ValueError("tools not both held after lift")
        grip_state = held_openings.copy()
        rack_poses = [tool(i) for i in range(2)]
        offsets = []
        for i in range(2):
            cp, cr = ik.tcp_pose(joints(i), rig.bases[sides[i]])
            tp, tr = tool(i)
            offsets.append((tr.T @ (cp - tp), tr.T @ cr))

        def solve_tool(i, pos, rot):
            offset, orient = offsets[i]
            return ik.solve(
                np.asarray(pos) + rot @ offset,
                ik.matrix_to_quat_xyzw(rot @ orient),
                q_init=joints(i),
                base_pos=rig.bases[sides[i]],
                iters=175,
            )

        def go_tool(label, i, pos, rot, steps):
            q, ep, ea = solve_tool(i, pos, rot)
            if ep > 0.003 or ea > 0.045:
                raise ValueError(f"{label} unreachable: {ep:.4f} m {ea:.4f} rad")
            goals = commands.copy()
            goals[i] = q
            move(label, goals, grip_state, steps)
            tp, tr = tool(i)
            # Joint tracking under gravity can have a centimetre of static sag.
            # Integrate observed tool-space error into the commanded wrist pose
            # and keep every correction on the normal bimanual action interface.
            tolerance = (
                0.00018 if label in ("driver_at_socket", "driver_engage") else 0.002
            )
            for correction in range(10):
                delta = np.asarray(pos) - tp
                rotation = Rotation.from_matrix(rot @ tr.T).as_rotvec()
                if (
                    np.linalg.norm(delta) <= tolerance
                    and np.linalg.norm(rotation) < 0.012
                ):
                    break
                if np.linalg.norm(delta) > 0.09:
                    raise ValueError(f"{label} outside servo capture range")
                er = Rotation.from_rotvec(rotation * 0.50).as_matrix()
                cp, cr = ik.tcp_pose(goals[i], rig.bases[sides[i]])
                grasp_to_tip = tp - ik.tcp_pose(joints(i), rig.bases[sides[i]])[0]
                newp = cp + 0.50 * delta + grasp_to_tip - er @ grasp_to_tip
                newq, cp_err, cr_err = ik.solve(
                    newp,
                    ik.matrix_to_quat_xyzw(er @ cr),
                    q_init=goals[i],
                    base_pos=rig.bases[sides[i]],
                    iters=120,
                )
                if cp_err > 0.003 or cr_err > 0.045:
                    break
                goals[i] = newq
                move(label + "_trim", goals, grip_state, 10)
                move(label + "_hold", goals, grip_state, 12)
                tp, tr = tool(i)
            error = float(np.linalg.norm(tp - pos))
            tcp_measured = ik.tcp_pose(joints(i), rig.bases[sides[i]])[0]
            tcp_target = np.asarray(pos) + rot @ offsets[i][0]
            rot_error = Rotation.from_matrix(rot.T @ tr).magnitude()
            results.setdefault("waypoints", []).append(
                dict(
                    name=label,
                    tip_error_m=error,
                    ik_m=ep,
                    tcp_error_m=float(np.linalg.norm(tcp_measured - tcp_target)),
                    orientation_error_rad=float(rot_error),
                    actual_tip=tp.tolist(),
                    target_tip=np.asarray(pos).tolist(),
                )
            )
            if error < 0.012:
                # Re-identify the rigid transform at each measured grasp to
                # compensate for small contact shifts and finger compliance.
                cp, cr = ik.tcp_pose(joints(i), rig.bases[sides[i]])
                offsets[i] = (tr.T @ (cp - tp), tr.T @ cr)
            if error > max(0.004, 4 * tolerance):
                raise ValueError(f"{label} tool slip or tracking error {error:.4f} m")

        def body_pose(name):
            return bench.bodies[name].get_transforms().numpy()[0].copy()

        # Stay in the grasp's measured wrist frame during contact. Recomputing
        # a full free-space IK command here reintroduces static arm sag.
        def servo_tool(i, angle, tip_target, n, label):
            goal_rot = Rotation.from_euler("z", angle).as_matrix()
            for _ in range(n):
                tp, tr = tool(i)
                actual_tcp = ik.tcp_pose(joints(i), rig.bases[sides[i]])[0]
                cp, cr = ik.tcp_pose(commands[i], rig.bases[sides[i]])
                dr = Rotation.from_matrix(goal_rot @ tr.T).as_rotvec()
                step_r = Rotation.from_rotvec(
                    np.clip(0.12 * dr, -0.022, 0.022)
                ).as_matrix()
                delta = np.asarray(tip_target) - tp
                arm = tp - actual_tcp
                step_xyz = (
                    cp + np.clip(0.10 * delta, -0.00045, 0.00045) + arm - step_r @ arm
                )
                goals = commands.copy()
                goals[i], err, ea = ik.solve(
                    step_xyz,
                    ik.matrix_to_quat_xyzw(step_r @ cr),
                    q_init=commands[i],
                    base_pos=rig.bases[sides[i]],
                    iters=30,
                    tol=0.000003 if args.reserve_retreat else 0.000015,
                )
                if err > 0.0012 or ea > 0.025:
                    if args.adaptive_stroke and label == "driver_in_socket":
                        return False
                    raise ValueError(f"{label} wrist workspace exhausted")
                move(label, goals, grip_state, 1)
            return True

        def tighten_nut(one_stroke=False):
            # The second hand engages the nut laterally. Keep the left wrist's
            # commanded position while turning, then withdraw the tools sequentially.
            nut = lambda: body_pose("nut")[:3]
            nut_phase = float(bench.state()[0][1] + bench.state()[0][3])
            tp, tr = tool(1)
            cp, cr = ik.tcp_pose(joints(1), rig.bases["right"])
            offsets[1] = (tr.T @ (cp - tp), tr.T @ cr)
            _, yaw, wr = wrench_flat(
                nut(), nut_phase, *offsets[1], joints(1), rig.bases["right"]
            )
            results["selected_wrench_yaw_deg"] = math.degrees(yaw)
            go_tool(
                "wrench_above_approach",
                1,
                nut() - 0.036 * wr[:, 0] + [0, 0, 0.04],
                wr,
                52,
            )
            go_tool("wrench_level", 1, nut() - 0.036 * wr[:, 0], wr, 38)
            home = yaw
            for stroke in range(6):
                # Approach between the flats and turn in short increments. The
                # free wrench must turn the native relative nut angle under contact.
                if args.guided_wrench:
                    from wrench_guide import fit_wrench

                    feed = 0.030
                    ready = 0
                    for _ in range(320):
                        pos, orient = tool(1)
                        nut_pose = body_pose("nut")
                        contact_fit = fit_wrench(pos, orient, nut_pose)
                        rot_nut = Rotation.from_quat(nut_pose[3:]).as_matrix()
                        phi = math.atan2(rot_nut[1, 0], rot_nut[0, 0])
                        desired = yaw + math.remainder(phi - yaw, math.pi / 3)
                        yaw += np.clip(desired - yaw, -0.012, 0.012)
                        wr = Rotation.from_euler("z", yaw).as_matrix()
                        centered = (
                            abs(contact_fit["across_m"]) < 0.00045
                            and abs(contact_fit["height_m"]) < 0.00055
                        )
                        if centered or feed > 0.022:
                            feed = max(0.0, feed - 0.00032)
                        servo_tool(
                            1,
                            yaw,
                            nut_pose[:3] - feed * wr[:, 0],
                            1,
                            "wrench_guided_seat",
                        )
                        seated = (
                            abs(contact_fit["along_m"]) < 0.0016
                            and centered
                            and contact_fit["face_clearance_m"] >= -0.00008
                        )
                        ready = ready + 1 if seated else 0
                        if ready >= 3:
                            break
                    results.setdefault("wrench_seats", []).append(contact_fit)
                    if ready < 3:
                        raise ValueError(
                            f"nut did not seat between the wrench flats: {contact_fit}"
                        )
                else:
                    for dist in np.linspace(0.036, 0.0, 36):
                        servo_tool(1, yaw, nut() - dist * wr[:, 0], 3, "wrench_seat")
                old_nt = latest["nt"]
                seat = []
                for j in range(14):
                    yaw -= math.radians(4.5)
                    wr = Rotation.from_euler("z", yaw).as_matrix()
                    servo_tool(1, yaw, nut(), 6, "wrench_tighten")
                    seat.append(latest["seat_n"])
                    if len(seat) >= 3 and min(seat[-3:]) >= args.wrench_preload_n:
                        break
                results.setdefault("wrench_strokes", []).append(
                    dict(
                        relative_turn_rad=latest["nt"] - old_nt, seat_n=latest["seat_n"]
                    )
                )
                if one_stroke:
                    if latest["nt"] - old_nt > -math.radians(20):
                        raise ValueError(
                            "free wrench did not transmit 20 degrees of physical rotation"
                        )
                    return
                for dist in np.linspace(0.0, 0.035, 22):
                    servo_tool(1, yaw, nut() - dist * wr[:, 0], 3, "wrench_back_out")
                if min(r["fixture"]["seat_n"] for r in trace[-5:]) >= max(
                    3.0, 0.90 * args.wrench_preload_n
                ):
                    break
                if latest["nt"] - old_nt > -0.08:
                    raise ValueError("wrench failed to rotate the physical nut")
                phase = latest["st"] + latest["nt"]
                yaw = phase + round((home - phase) / (math.pi / 3)) * math.pi / 3
                wr = Rotation.from_euler("z", yaw).as_matrix()
                go_tool("wrench_reindex", 1, nut() - 0.036 * wr[:, 0], wr, 35)
            if latest["seat_n"] < 2.0:
                raise ValueError("no stable nut preload after wrench removal")

        if args.check_wrench_stroke:
            tighten_nut(one_stroke=True)
            return
        screw = lambda: body_pose("screw")[:3]
        above = screw() + [0, 0, 0.050]
        # Hex socket indexed every 60 degrees. Choose a reachable wrist at the
        # top of the fixture, then advance only via commanded robot joint actions.
        options = []
        for k in range(6):
            rot = Rotation.from_euler("z", k * math.pi / 3).as_matrix()
            q, ep, ea = solve_tool(0, above, rot)
            # Prefer an index with a continuous clockwise working arc at
            # the real socket, using the measured rack grasp transform.
            offset, orient = offsets[0]
            reachable = 0
            warm = q.copy()
            if ep < 0.002 and ea < 0.04:
                for deg in range(0, 281, 10):
                    trial = Rotation.from_euler(
                        "z", k * math.pi / 3 - math.radians(deg)
                    ).as_matrix()
                    wrist = screw() - [0, 0, 0.0026] + trial @ offset
                    candidate, pe, pr = ik.solve(
                        wrist,
                        ik.matrix_to_quat_xyzw(trial @ orient),
                        q_init=warm,
                        base_pos=rig.bases["left"],
                        iters=80,
                    )
                    if pe > 0.0015 or pr > 0.030 or max(abs(candidate - warm)) > 1.25:
                        break
                    reachable = deg
                    warm = candidate
            options.append(
                (
                    -reachable,
                    ep + ea * 0.05 + 0.00001 * np.linalg.norm(q - joints(0)),
                    k,
                    rot,
                )
            )
        neg_span, _, idx, rot = min(options, key=lambda item: item[:3])
        results["planned_clockwise_arc_deg"] = -neg_span
        print("INDEX ARCS", [(k, -n) for n, _, k, _ in options], flush=True)
        results["hex_index"] = idx
        # Transport above the rack before rotating to keep the long free driver clear.
        tp, tr = tool(0)
        high = np.array([tp[0], tp[1], tp[2] + 0.075])
        go_tool("driver_clear_rack", 0, high, tr, 48)
        from scipy.spatial.transform import Slerp

        interp = Slerp([0.0, 1.0], Rotation.from_matrix([tr, rot]))
        # Keep the mid-handle clear of the workcell throughout the vertical swing.
        for j in range(1, 5):
            u = j / 4
            go_tool(
                f"driver_orient_{j}",
                0,
                high + u * (above - high),
                interp(u).as_matrix(),
                32,
            )
        go_tool("driver_above_socket", 0, above, rot, 20)
        # Stop the free-space transfer above the hardware. The 0.20 mm
        # difference across flats leaves only 0.10 mm clearance per side.
        go_tool("driver_preinsert", 0, screw() + [0, 0, 0.016], rot, 36)

        def measured_fit():
            pos, rot = tool(0)
            part = body_pose("screw")
            return hex_fit(
                pos,
                rot,
                part[:3],
                Rotation.from_quat(part[3:]).as_matrix(),
                bit_chamfer,
            )

        def insert_driver(indexed, initial_height):
            height = initial_height
            wrist_index = indexed
            free_align = args.align_before_feed
            align_count = 0
            last_alignment = -120
            seated_steps = 0
            centre_error = np.zeros(2)
            prev_p, prev_r = tool(0)
            tip_step = np.zeros(3)
            angular_step = np.zeros(3)
            for step in range(640 if initial_height > 0.009 else 430):
                tip_pos, tool_rot = tool(0)
                tip_step = 0.65 * tip_step + 0.35 * (tip_pos - prev_p)
                angular_step = (
                    0.65 * angular_step
                    + 0.35 * Rotation.from_matrix(tool_rot @ prev_r.T).as_rotvec()
                )
                prev_p, prev_r = tip_pos, tool_rot
                part = body_pose("screw")
                part_rot = Rotation.from_quat(part[3:]).as_matrix()
                socket_yaw = math.atan2(part_rot[1, 0], part_rot[0, 0])
                indexed = (
                    socket_yaw
                    + round((wrist_index - socket_yaw) / (math.pi / 3)) * math.pi / 3
                )
                if free_align:
                    # Index only in free space. Once the bevel bears load the
                    # mobile screw can rotate without the hand chasing its phase.
                    if tip_pos[2] - part[2] > 0.00065:
                        wrist_index = indexed
                        actual_yaw = math.atan2(tool_rot[1, 0], tool_rot[0, 0])
                        phase_error = math.remainder(indexed - actual_yaw, math.tau)
                        if (
                            abs(phase_error) < 0.055
                            and np.linalg.norm(tip_pos[:2] - part[:2]) < 0.00055
                        ):
                            align_count += 1
                        else:
                            align_count = 0
                        if align_count >= 5:
                            free_align = False
                            last_alignment = step
                            align_count = 0
                if args.contact_guided:
                    # Let a freely turning screw index against the stationary wrist.
                    indexed = wrist_index
                goal_rot = Rotation.from_euler("z", indexed).as_matrix()
                err_rot = Rotation.from_matrix(goal_rot @ tool_rot.T).as_rotvec()
                delta = np.asarray(part[:3]) - tip_pos
                radial = np.linalg.norm(delta[:2])
                centre_error = 0.85 * centre_error + 0.15 * delta[:2]
                axial_force = bench.tool_efforts()[0]
                aligned = (
                    np.linalg.norm(centre_error) < 0.00032
                    and radial < 0.00055
                    and np.linalg.norm(err_rot[:2]) < 0.020
                    and abs(err_rot[2]) < 0.055
                )
                if args.square_bit:
                    centered = aligned
                else:
                    centered = (
                        radial < 0.00040
                        and np.linalg.norm(err_rot[:2]) < 0.014
                        and abs(err_rot[2]) < 0.09
                    )
                if height > 0.005 or centered:
                    if abs(axial_force) < 2.0:
                        height = max(
                            -0.0035, height - (0.00055 if height > 0.005 else 0.00013)
                        )
                    elif height < 0.002 and radial > 0.0001:
                        height = min(0.003, height + 0.00012)
                # Search nearby faces only when the tip is at the mouth under load.
                if (
                    args.square_bit
                    and abs(axial_force) > 1.0
                    and -0.0004 < tip_pos[2] - part[2] < 0.001
                ):
                    delta[:2] += 0.00015 * np.array(
                        [math.cos(step * 0.30), math.sin(step * 0.30)]
                    )
                fit = hex_fit(
                    tip_pos,
                    tool_rot,
                    part[:3],
                    part_rot,
                    bit_chamfer,
                )
                wall = fit["wall_clearance_m"]
                if (wall is not None and wall < -0.00008) or fit[
                    "floor_clearance_m"
                ] < 0.0:
                    # Back the bit out of face interference before recentering.
                    height = max(height, 0.0009)
                engaged = (
                    fit["straight_flank_depth_m"]
                    >= (0.0029 if args.supported_turn else 0.0016)
                    and wall is not None
                    and wall >= -0.00005
                    and fit["floor_clearance_m"] >= -0.00005
                    and radial < 0.00022
                )
                seated_steps = seated_steps + 1 if engaged else 0
                if seated_steps >= 3:
                    break
                if (
                    not args.square_bit
                    and wall is None
                    and tip_pos[2] > part[2] + 0.0005
                    and not centered
                ):
                    # Keep a clearance plane until the measured free tip is over the opening.
                    height = max(height, 0.0010)
                if args.contact_guided:
                    # A remote-centre approach commands the wrist only. Allow the
                    # physical bevel reaction to displace the held tip laterally.
                    force = driver_impulse / base.step_dt
                    height_error = tip_pos[2] - part[2]
                    if step == 0:
                        feed_height = initial_height
                    if (
                        radial < (0.00075 if args.long_lead else 0.00048)
                        and np.linalg.norm(err_rot[:2]) < 0.025
                    ):
                        feed_height = max(
                            -0.0042 if args.supported_turn else -0.0026,
                            feed_height
                            - (0.00038 if height_error > 0.002 else 0.000055),
                        )
                    if force[2] > 0.45 or (wall is not None and wall < -0.00009):
                        feed_height = max(feed_height, height_error + 0.00008)
                    if (
                        args.align_before_feed
                        and step - last_alignment > 90
                        and 0.0 < fit["depth_m"] < 0.0011
                    ):
                        actual_yaw = math.atan2(tool_rot[1, 0], tool_rot[0, 0])
                        off_flat = abs(
                            math.remainder(socket_yaw - actual_yaw, math.pi / 3)
                        )
                        if off_flat > 0.12:
                            free_align = True
                            last_alignment = step
                    if free_align:
                        feed_height = max(feed_height, 0.0015)
                    height = feed_height
                    if height_error < 0.0015:
                        delta[:2] += np.clip(0.0012 * force[:2], -0.00035, 0.00035)
                delta[2] += height
                if (
                    not args.contact_guided
                    and height <= -0.0035
                    and abs(delta[2]) < 0.00025
                    and radial < 0.00015
                ):
                    break
                goals = commands.copy()
                cp, cr = ik.tcp_pose(goals[0], rig.bases["left"])
                actual_tcp = ik.tcp_pose(joints(0), rig.bases["left"])[0]
                arm_to_tip = tip_pos - actual_tcp
                dq = Rotation.from_rotvec(
                    np.clip(0.033 * err_rot - 0.80 * angular_step, -0.003, 0.003)
                ).as_matrix()
                next_xyz = (
                    cp
                    + np.clip(0.045 * delta - 0.85 * tip_step, -0.0008, 0.0008)
                    + arm_to_tip
                    - dq @ arm_to_tip
                )
                goals[0], pe, ae = ik.solve(
                    next_xyz,
                    ik.matrix_to_quat_xyzw(dq @ cr),
                    q_init=goals[0],
                    base_pos=rig.bases["left"],
                    iters=18,
                    tol=0.000003 if args.reserve_retreat else 0.000015,
                )
                if pe > 0.002 or ae > 0.035:
                    raise ValueError("guarded insertion wrist travel exhausted")
                move("guarded_insert", goals, grip_state, 1)
                if step % 20 == 0:
                    trace[-1]["guard"] = dict(
                        height=height,
                        radial=float(radial),
                        rotation=err_rot.tolist(),
                        tip_step=tip_step.tolist(),
                        axial_force=axial_force,
                    )
            tip = tool(0)[0] - screw()
            reading = dict(
                radial_m=float(np.linalg.norm(tip[:2])),
                depth_m=float(-tip[2]),
                target_height_m=height,
            )
            reading.update(measured_fit())
            results.setdefault("insertions", []).append(reading)
            if len(results["insertions"]) == 1:
                results["engagement_radial_m"] = reading["radial_m"]
                results["engagement_depth_m"] = reading["depth_m"]
            if (
                reading["radial_m"] > 0.00045
                or reading["straight_flank_depth_m"] < 0.001
                or reading["wall_clearance_m"] is None
                or reading["wall_clearance_m"] < -0.00005
                or reading["floor_clearance_m"] < -0.00005
            ):
                raise ValueError(f"insufficient physical hex engagement: {reading}")
            return indexed

        indexed = insert_driver(idx * math.pi / 3, 0.014)

        target_angle = (
            math.radians(87) / 300 - 0.0057
        ) * math.tau / 0.001 + args.driver_correction_rad
        start_angle = latest["st"]
        lead = 0.001 / math.tau

        def drive_tip(depth):
            return screw() - [0.0, 0.0, depth]

        def regrasp_in_socket(sweep):
            # The socket supports the tool. The only outputs are normal YAM arm
            # and finger commands. Monitor the free bit throughout release.
            tip, tool_r = tool(0)
            grasp = tip + tool_r[:, 2] * 0.100
            initial_pos, initial_rot = ik.tcp_pose(joints(0), rig.bases["left"])
            arm = initial_pos - grasp
            grip_state[0] = 1.0
            move("driver_unpinch", commands.copy(), grip_state, 10)
            for fraction in np.linspace(0.0, 1.0, 31)[1:]:
                free = measured_fit()
                if (
                    free["straight_flank_depth_m"] < 0.001
                    or free["wall_clearance_m"] is None
                    or free["wall_clearance_m"] < -0.00008
                    or free["floor_clearance_m"] < -0.00005
                ):
                    raise ValueError(f"unheld driver lost its socket support: {free}")
                tip, tool_r = tool(0)
                pivot = tip + 0.100 * tool_r[:, 2]
                r = Rotation.from_euler("z", fraction * sweep).as_matrix()
                goal = pivot + r @ arm
                targets = commands.copy()
                targets[0], pe, re = ik.solve(
                    goal,
                    ik.matrix_to_quat_xyzw(r @ initial_rot),
                    q_init=commands[0],
                    base_pos=rig.bases["left"],
                    iters=45,
                    tol=0.000035,
                )
                if pe > 0.0012 or re > 0.025:
                    raise ValueError("empty wrist cannot reach next handle grasp")
                move("driver_empty_wrist_return", targets, grip_state, 2)
            # The 24 mm square grip presents a larger width on its diagonals.
            # Infer the actual fingertip closing axis from the two measured pad poses.
            links = base.scene["left_robot"].root_physx_view
            paths = list(links.link_paths[0])
            pad_ids = [
                i
                for i, path in enumerate(paths)
                if str(path).rsplit("/", 1)[-1].lower() in ("tip_left", "tip_right")
            ]
            if len(pad_ids) != 2:
                raise ValueError(f"could not identify two finger links in {paths}")
            centres = links.get_link_transforms().numpy()[0, pad_ids, :3]
            axis = tool(0)[1].T @ (centres[1] - centres[0])
            axis /= np.linalg.norm(axis)
            half_width = (
                abs(axis[0]) * 0.012 + abs(axis[1]) * 0.012 + abs(axis[2]) * 0.025
            )
            grip_state[0] = np.clip((half_width - 0.0016) / abs(FINGER_OPEN), 0.0, 1.0)
            move("driver_repinch", commands.copy(), grip_state, 15)
            free = measured_fit()
            if (
                free["straight_flank_depth_m"] < 0.001
                or free["wall_clearance_m"] is None
                or free["wall_clearance_m"] < -0.00008
                or free["floor_clearance_m"] < -0.00005
            ):
                raise ValueError(f"driver displaced during physical regrasp: {free}")
            _, r = tool(0)
            return math.atan2(r[1, 0], r[0, 0])

        home_yaw = indexed
        for stroke in range(7):
            initial_angle = latest["st"]
            needed = initial_angle - target_angle
            if abs(needed) < 0.14:
                break
            if needed < 0:
                raise ValueError("driver passed target setting")
            yaw_start = indexed
            # A hex face can be re-indexed every 60 degrees. Include a small
            # allowance for flank backlash and steer to the measured screw angle.
            arc = max(50, min(270, -neg_span - 12))
            increment = 2.0 if args.supported_turn else 4.5
            depth = 0.0040 if args.supported_turn else 0.0026
            within_reach = True
            for j in range(math.ceil(arc / increment)):
                remaining = latest["st"] - target_angle
                if remaining < 0.10:
                    break
                if args.reserve_retreat and initial_angle - latest["st"] > 0.12:
                    # On the measured failed third stroke, joint4 saturated
                    # before the driver could retract. Check the current arm and
                    # an actual 12 mm straight withdrawal before continuing.
                    q = commands[0]
                    cp, cr = ik.tcp_pose(q, rig.bases["left"])
                    retreat_q, retreat_m, retreat_rad = ik.solve(
                        cp + [0, 0, 0.012],
                        ik.matrix_to_quat_xyzw(cr),
                        q_init=q,
                        base_pos=rig.bases["left"],
                        iters=75,
                        tol=0.00004,
                    )
                    margin = min(
                        float(np.minimum(q - ik.LOWER, ik.UPPER - q).min()),
                        float(
                            np.minimum(retreat_q - ik.LOWER, ik.UPPER - retreat_q).min()
                        ),
                    )
                    if (
                        margin < math.radians(12)
                        or retreat_m > 0.0005
                        or retreat_rad > 0.01
                    ):
                        _, actual_rot = tool(0)
                        indexed = math.atan2(actual_rot[1, 0], actual_rot[0, 0])
                        results.setdefault("retreat_reserves", []).append(
                            dict(
                                tick=results["physics_ticks"],
                                screw_rad=latest["st"],
                                joint_margin_deg=math.degrees(margin),
                                fit=measured_fit(),
                            )
                        )
                        within_reach = False
                        break
                indexed -= min(math.radians(increment), remaining)
                within_reach = servo_tool(
                    0, indexed, drive_tip(depth), 7, "driver_in_socket"
                )
                if not within_reach:
                    if initial_angle - latest["st"] < 0.1:
                        raise ValueError("available arc too short to turn the screw")
                    _, real_rot = tool(0)
                    indexed = math.atan2(real_rot[1, 0], real_rot[0, 0])
                    results.setdefault("wrist_reindexes", []).append(
                        dict(
                            tick=results["physics_ticks"],
                            angle=latest["st"],
                            fit=measured_fit(),
                        )
                    )
                    break
                surface = measured_fit()
                if (
                    args.supported_turn
                    and 0.0 < surface["straight_flank_depth_m"] < 0.0018
                    and surface["wall_clearance_m"] is not None
                    and surface["wall_clearance_m"] >= -0.00008
                ):
                    # Pause twisting. Hold the measured tool yaw and feed axially until
                    # the full-width land has regained contact length.
                    _, actual_rot = tool(0)
                    held = math.atan2(actual_rot[1, 0], actual_rot[0, 0])
                    for _ in range(36):
                        servo_tool(0, held, drive_tip(depth), 1, "driver_reseat")
                        surface = measured_fit()
                        if (
                            surface["straight_flank_depth_m"] >= 0.0022
                            or surface["straight_flank_depth_m"] <= 0
                            or surface["wall_clearance_m"] is None
                            or surface["wall_clearance_m"] < -0.00008
                        ):
                            break
                if (
                    surface["straight_flank_depth_m"] < 0.0008
                    or surface["wall_clearance_m"] is None
                    or surface["wall_clearance_m"] < -0.00008
                    or surface["floor_clearance_m"] < -0.00005
                ):
                    results.setdefault("lost_drive_fit", surface)
                    raise ValueError(
                        f"driver lost flank contact during torque: {surface}"
                    )
                if yaw_start - indexed >= math.radians(arc):
                    break
            # Only hold a reachable target. A shortened stroke already holds
            # the last in-range physical command and must retreat axially.
            if within_reach:
                servo_tool(0, indexed, drive_tip(depth), 10, "driver_stroke_hold")
            state = bench.read()
            turn = state["st"] - initial_angle
            tip = tool(0)[0] - screw()
            results.setdefault("driver_strokes", []).append(
                dict(
                    turn_rad=turn,
                    screw_rad=state["st"],
                    depth_m=float(-tip[2]),
                    radial_m=float(np.linalg.norm(tip[:2])),
                    thread_extension_m=state["sz"] - lead * state["st"],
                )
            )
            results["measured_driver_turn_rad"] = state["st"] - start_angle
            print("DRIVER STROKE", results["driver_strokes"][-1], flush=True)
            if abs(state["st"] - target_angle) < 0.16:
                break
            if turn > -0.10 or tip[2] > -0.001:
                raise ValueError("driver stroke lost measured engagement")
            if args.seated_regrasp:
                indexed = regrasp_in_socket(
                    math.remainder(yaw_start - indexed, math.tau)
                )
                continue
            # Withdraw straight and verify clearance before moving to another flat.
            if args.reserve_retreat:
                for _ in range(13):
                    servo_tool(0, indexed, drive_tip(-0.007), 5, "driver_withdraw")
                    if (tool(0)[0] - screw())[2] >= 0.005:
                        break
            else:
                servo_tool(0, indexed, drive_tip(-0.007), 65, "driver_withdraw")
            if (tool(0)[0] - screw())[2] < 0.003:
                raise ValueError("driver failed to clear socket for re-index")
            part = body_pose("screw")
            sr = Rotation.from_quat(part[3:]).as_matrix()
            sy = math.atan2(sr[1, 0], sr[0, 0])
            indexed = sy + round((home_yaw - sy) / (math.pi / 3)) * (math.pi / 3)
            servo_tool(
                0,
                indexed,
                drive_tip(-0.007),
                45 if args.reserve_retreat else 65,
                "driver_reindex",
            )
            indexed = insert_driver(indexed, 0.007)
        if abs(latest["st"] - target_angle) > 0.20:
            raise ValueError("measured screw angle outside 90-degree setting approach")
        tighten_nut()
        go_tool("driver_back_out", 0, screw() + [0, 0, 0.035], tool(0)[1], 30)
        for i in (1, 0):
            rp, rr = rack_poses[i]
            go_tool(names[i] + "_return", i, rp, rr, 55)
            go_tool(names[i] + "_place", i, rp - [0, 0, lift[i]], rr, 36)
            grip_state[i] = 1
            move(names[i] + "_release", commands.copy(), grip_state, 18)
            goals = commands.copy()
            goals[i] = plans[i][0]
            move(names[i] + "_clear", goals, grip_state, 36)
        preload = [r["fixture"]["seat_n"] for r in trace[-10:]]
        results["released_preload_min_n"] = min(preload)
        if min(preload) < 2:
            raise ValueError("insufficient measured preload after releasing both tools")
        run_valve(grip_state)

    except Exception as exc:
        results["failure"] = str(exc)
        print("WORKCELL TRIAL", str(exc), flush=True)
    finally:
        if grade_bound is not None:
            results["local_score_bound"] = grade_bound.report()
            if grade_bound.grader.tick != results["physics_ticks"]:
                results["local_score_bound"]["certain_current_zero"] = False
                results["local_score_bound"]["incomplete_native_stream"] = True
        if room_contacts is not None:
            results["room_contacts"] = room_contacts.report()

        def number(value):
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        def save(path, data, indent=None):
            temp = path.with_name(path.name + ".partial")
            temp.write_text(
                json.dumps(data, default=number, allow_nan=False, indent=indent)
            )
            temp.replace(path)

        path = args.output.with_suffix(".trace.json")
        try:
            save(path, trace)
            results["trace_file"] = path.name
        except (OSError, TypeError, ValueError) as error:
            results["trace_error"] = str(error)
            results.setdefault("failure", "raw trace could not be preserved")
        save(args.output, results, indent=2)
        for callback in callbacks:
            callback.unregister()
        break_subscription = None
        if writer:
            writer.close()
        env.close()


try:
    main()
finally:
    app.close()
