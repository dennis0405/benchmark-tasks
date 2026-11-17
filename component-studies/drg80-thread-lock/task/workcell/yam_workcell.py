"""Physical bimanual adjustment trial with an instrumented, mounted workcell."""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", type=Path)
parser.add_argument("--continuous-seat", action="store_true")
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
from yam_arena.constants import PAD_TIP_PAST_TCP, READY
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


def main():
    objects, rig, embodiment, sides, names, grasps = load_station()
    fixture = args.output.with_suffix(".usda").resolve()
    build(fixture, task_assets(), continuous_seat=args.continuous_seat)
    objects.append(
        scene.Object(
            name="gripper_bench",
            object_type=scene.ObjectType.BASE,
            usd_path=str(fixture),
            initial_pose=scene.Pose(position_xyz=(0, 0, 0), rotation_xyzw=(0, 0, 0, 1)),
        )
    )
    arena = scene.IsaacLabArenaEnvironment(
        name="drg80_instrumented_yam",
        scene=scene.Scene(objects),
        embodiment=embodiment,
        env_cfg_callback=scene.env_cfg_callback(20),
    )
    env = ArenaEnvBuilder(
        arena, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cuda:0")
    ).make_registered()
    results = {
        "completed": [],
        "physics_ticks": 0,
        "scope": "physical YAM workcell commissioning, no competition grade",
    }
    trace = []
    callbacks = []
    writer = None
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
        targets = [base.scene[name].root_physx_view.prim_paths[0] for name in names]
        bench = Bench(
            root,
            round(1 / base.physics_dt),
            tools=targets,
            native_friction=args.native_friction,
            rise_seconds=args.rise_seconds,
        )

        from runpy import run_path

        scoring = run_path(
            str(Path(__file__).resolve().parent.parent / "stop_lock_score.py")
        )
        powered = scoring["ElasticLockMotion"](round(1/base.physics_dt))
        protected=[root+"/"+name for name in ("screw","nut","retainer","left_jaw","right_jaw","piston")]
        external=list(targets)
        for side in sides:
            links=base.scene[side+"_robot"].root_physx_view.link_paths[0]
            external.extend(links)
        contacts=PhysxManager.get_physics_sim_view().create_rigid_contact_view(protected,
            filter_patterns=[external for _ in protected],max_contact_data_count=2048)

        def pre(*args, **kwargs):
            if "callback_error" in results:
                return
            try:
                bench.step()
            except Exception as e:
                results["callback_error"] = repr(e)

        def post(*args, **kwargs):
            results["physics_ticks"] += 1
            r = bench.read()
            violated = powered.observe(
                results["physics_ticks"],
                r["actuator_n"],
                r["sz"],
                r["st"],
                r["st"] + r["nt"],
                r["stop_n"],
                not np.any(contacts.get_contact_data(dt=base.physics_dt)[4].numpy()),
                abs(r["angular_speed"])<=2 and abs(r["piston_speed"])<=.005,
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

        for event, callback in (
            (IsaacEvents.PRE_PHYSICS_STEP, pre),
            (IsaacEvents.POST_PHYSICS_STEP, post),
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
                rec = dict(
                    phase=label,
                    fixture=bench.read(),
                    tools=[tool(i)[0].tolist() for i in range(2)],
                )
                trace.append(rec)
                if writer and j % 2 == 0:
                    rgb = base.scene["top_cam"].data.output["rgb"]
                    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
                    writer.append_data(rgb[0, ..., :3].cpu().numpy().astype(np.uint8))
            commands[:] = target
            if not results["completed"] or results["completed"][-1] != label:
                results["completed"].append(label)
            if label == "guarded_insert":
                return
            print(
                label,
                trace[-1]["tools"],
                "screw_rad",
                round(trace[-1]["fixture"]["st"], 4),
                flush=True,
            )

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
                [g, g],
                n,
            )
        lift = np.array(trace[-1]["tools"])[:, 2] - np.array(trace[23]["tools"])[:, 2]
        results["held_lift_m"] = lift.tolist()
        if min(lift) < 0.030:
            raise ValueError("tools not both held after lift")
        grip_state = [0, 0]
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

        screw = lambda: np.array(bench.poses()["screw"][:3])
        above = screw() + [0, 0, 0.050]
        # Hex socket indexed every 60 degrees. Choose a reachable wrist at the
        # top of the fixture, then advance only via commanded robot joint actions.
        options = []
        for k in range(6):
            rot = Rotation.from_euler("z", k * math.pi / 3).as_matrix()
            q, ep, ea = solve_tool(0, above, rot)
            options.append(
                (ep + ea * 0.05 + 0.00001 * np.linalg.norm(q - joints(0)), k, rot)
            )
        _, idx, rot = min(options, key=lambda item: item[0])
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
        height = 0.014
        indexed = idx * math.pi / 3
        aligned = 0
        prev_p,prev_r=tool(0)
        tip_step=np.zeros(3)
        angular_step=np.zeros(3)
        for step in range(640):
            tip_pos, tool_rot = tool(0)
            tip_step=.65*tip_step+.35*(tip_pos-prev_p)
            angular_step=.65*angular_step+.35*Rotation.from_matrix(tool_rot@prev_r.T).as_rotvec()
            prev_p,prev_r=tip_pos,tool_rot
            part = bench.poses()["screw"]
            part_rot = Rotation.from_quat(part[3:]).as_matrix()
            socket_yaw = math.atan2(part_rot[1, 0], part_rot[0, 0])
            indexed = (
                socket_yaw + round((indexed - socket_yaw) / (math.pi / 3)) * math.pi / 3
            )
            goal_rot = Rotation.from_euler("z", indexed).as_matrix()
            err_rot = Rotation.from_matrix(goal_rot @ tool_rot.T).as_rotvec()
            delta = np.asarray(part[:3]) - tip_pos
            radial = np.linalg.norm(delta[:2])
            if radial < 0.00011 and np.linalg.norm(err_rot[:2]) < 0.015 and abs(err_rot[2]) < .050:
                aligned += 1
            else:
                aligned = 0
            if height > 0.005 or aligned >= 2:
                height = max(-0.0035, height - (0.00055 if height > 0.005 else 0.00018))
            axial_force = bench.tool_efforts()[0]
            if height < 0.001 and abs(axial_force) > 3 and radial > 0.0001:
                height = min(0.004, height + 0.0002)
            delta[2] += height
            if height <= -0.0035 and abs(delta[2]) < 0.00025 and radial < 0.00015:
                break
            goals = commands.copy()
            cp, cr = ik.tcp_pose(goals[0], rig.bases["left"])
            actual_tcp = ik.tcp_pose(joints(0), rig.bases["left"])[0]
            arm_to_tip = tip_pos - actual_tcp
            dq = Rotation.from_rotvec(
                np.clip(0.033 * err_rot - .80*angular_step, -0.003, 0.003)
            ).as_matrix()
            next_xyz = (
                cp
                + np.clip(0.045 * delta - .85*tip_step, -0.0008, 0.0008)
                + arm_to_tip
                - dq @ arm_to_tip
            )
            goals[0], pe, ae = ik.solve(
                next_xyz,
                ik.matrix_to_quat_xyzw(dq @ cr),
                q_init=goals[0],
                base_pos=rig.bases["left"],
                iters=18,
                tol=0.000015,
            )
            if pe > 0.002 or ae > 0.035:
                raise ValueError("guarded insertion wrist travel exhausted")
            move("guarded_insert", goals, grip_state, 1)
            if step%20==0:
                trace[-1]["guard"]=dict(height=height,radial=radial,rotation=err_rot.tolist(),
                    tip_step=tip_step.tolist(),axial_force=axial_force)
        results["commanded_insertion_height_m"] = height
        tip = tool(0)[0] - screw()
        results["engagement_radial_m"] = float(np.linalg.norm(tip[:2]))
        results["engagement_depth_m"] = float(-tip[2])
        if np.linalg.norm(tip[:2]) > 0.0015 or tip[2] > -0.001:
            raise ValueError(
                "insufficient measured hex engagement before applying torque"
            )
        before = bench.state()[0][1]
        for k in range(1, 13):
            angle = indexed - k * math.radians(20)
            rot = Rotation.from_euler("z", angle).as_matrix()
            go_tool(f"driver_turn_{20 * k}", 0, screw() - [0, 0, 0.004], rot, 18)
            turned = float(bench.state()[0][1] - before)
            results["measured_driver_turn_rad"] = turned
            if k >= 2 and abs(turned) < 0.15:
                raise ValueError("no corresponding physical screw rotation")
            if abs(bench.state()[0][1] + 4.01) < 0.19:
                break
        if abs(bench.state()[0][1] + 4.01) > 0.22:
            raise ValueError("measured screw angle outside 90-degree setting approach")
        # The second hand engages the nut laterally. Keep the left wrist's
        # commanded position while turning, then withdraw the tools sequentially.
        nut = lambda: np.array(bench.poses()["nut"][:3])
        nut_phase = float(bench.state()[0][1] + bench.state()[0][3])
        options = []
        for k in range(6):
            yaw = nut_phase + k * math.pi / 3
            wr = Rotation.from_euler("z", yaw).as_matrix()
            pre_nut = nut() - 0.036 * wr[:, 0]
            q, ep, ea = solve_tool(1, pre_nut, wr)
            options.append(
                (ep + 0.05 * ea + 0.00002 * np.linalg.norm(q - joints(1)), yaw, wr)
            )
        _, yaw, wr = min(options, key=lambda x: x[0])
        go_tool(
            "wrench_above_approach", 1, nut() - 0.036 * wr[:, 0] + [0, 0, 0.04], wr, 52
        )
        go_tool("wrench_level", 1, nut() - 0.036 * wr[:, 0], wr, 38)
        go_tool("wrench_seat", 1, nut(), wr, 38)
        for stroke in range(1, 13):
            wr = Rotation.from_euler("z", yaw - math.radians(5 * stroke)).as_matrix()
            go_tool("wrench_tighten", 1, nut(), wr, 15)
            preload = bench.read()["seat_n"]
            if preload >= 4.0:
                break
        go_tool("wrench_back_out", 1, nut() - 0.038 * wr[:, 0], wr, 30)
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
        # A real parallel-jaw grasp on the instrument handle drives the pivot.
        pivot = np.array(bench.poses()["lever"][:3])

        def lever_tcp(deg):
            axis = Rotation.from_euler("y", deg, degrees=True).apply([0, 0, 1])
            return pivot + 0.070 * axis + np.array([0, 0, PAD_TIP_PAST_TCP])

        hand = 1
        arm = rig.bases[sides[hand]]

        def handle_motion(label, deg, steps, extra_z=0.0):
            goals = commands.copy()
            q, ep, ea = ik.solve(
                lever_tcp(deg) + [0, 0, extra_z],
                ik.matrix_to_quat_xyzw(grip_rotation(0)),
                q_init=joints(hand),
                base_pos=arm,
                iters=180,
            )
            if ep > 0.003 or ea > 0.04:
                raise ValueError("lever IK infeasible")
            goals[hand] = q
            move(label, goals, grip_state, steps)

        handle_motion("valve_above", 0, 40, 0.035)
        handle_motion("valve_grasp", 0, 30)
        grip_state[hand] = 0
        handle_motion("valve_grip_close", 0, 20)
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

    except Exception as exc:
        results["failure"] = str(exc)
        print("WORKCELL TRIAL", str(exc), flush=True)
    finally:
        args.output.write_text(json.dumps(results, indent=2))
        args.output.with_suffix(".trace.json").write_text(json.dumps(trace))
        for callback in callbacks:
            callback.unregister()
        if writer:
            writer.close()
        env.close()


try:
    main()
finally:
    app.close()
