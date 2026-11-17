"""Run the actual YAM joint drives and free peel/food in Isaac Sim PhysX."""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cycles", type=int, default=32)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--video", action="store_true")
parser.add_argument("--camera-stride", type=int, choices=(1, 2, 4), default=2)
parser.add_argument(
    "--arena",
    action="store_true",
    help="Send 14-D targets to the public policy environment",
)
parser.add_argument("--park-control", action="store_true")
parser.add_argument("--pace", type=float, default=0.9)
parser.add_argument("--side", type=float, default=0.158)
parser.add_argument("--center-gain", type=float, default=1.1)
parser.add_argument("--aim", type=float, default=16.0)
parser.add_argument("--sweep", type=float, default=15.0)
parser.add_argument("--support", type=float, default=1.15)
parser.add_argument("--loaded-z", type=float, default=0.004)
parser.add_argument("--rate", type=int, default=240)
args = parser.parse_args()
app = AppLauncher(headless=True, enable_cameras=args.video).app
import omni.usd
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from yam_arena.constants import RIGHT_BASE_POS, LEFT_BASE_POS, READY
from yam_arena.ik import LOWER, UPPER, tcp_pose
from yam_arena.robot import yam_arm_cfg, asset_link_paths
from isaac_geometry import MOUTH, OVEN_Y, STONE_Z, PARK, GRIP_DISTANCE, HANDLE_Z, build


class NativeRun:
    def __init__(self):
        self.dt = 1 / args.rate
        self.env = None
        self.pose_rot = np.column_stack(
            (
                [0.0, -1.0, 0.0],
                [-0.70710678, 0.0, -0.70710678],
                [0.70710678, 0.0, -0.70710678],
            )
        )
        self.q = np.array(READY)
        if args.arena:
            self._open_policy_environment()
        else:
            self._open_native()
        self._prepare_episode()

    def _open_native(self):
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(device="cuda:0", dt=self.dt, render_interval=8)
        )
        self.stage = omni.usd.get_context().get_stage()
        self.bodies = build(self.stage)
        light = sim_utils.DomeLightCfg(intensity=650.0)
        light.func("/World/Light", light)
        first = self.ik(PARK + [-GRIP_DISTANCE, 0, HANDLE_Z + 0.065], self.pose_rot)
        if args.park_control:
            self.q = first = np.array(READY)
        cfg = yam_arm_cfg("/World/Yam", RIGHT_BASE_POS, contact_offset=0.0003)
        cfg.init_state.joint_pos.update(
            {f"joint{i + 1}": float(x) for i, x in enumerate(first)}
        )
        self.robot = Articulation(cfg)
        self.left = Articulation(
            yam_arm_cfg("/World/LeftYam", LEFT_BASE_POS, contact_offset=0.0003)
        )
        self.peel_path = "/World/Peel"
        self.food_sources = ["/World/Food/*"]
        self.arm_roots = ("/World/Yam", "/World/LeftYam")

    def _open_policy_environment(self):
        import task
        from yam_arena.constants import FINGER_OPEN, SHOULDER, WRIST

        if args.rate != 240:
            raise ValueError("The policy environment integrates at 240 Hz.")
        self.env = task.make_env(device="cuda:0", enable_cameras=False, warmup_steps=10)
        self.env.reset(seed=17)
        self.ready_action = torch.tensor(
            [[*READY, 1.0, *task.RIGHT_START, 1.0]], device="cuda:0"
        )
        for _ in range(10):
            self.env.step(self.ready_action)
        base = self.env.unwrapped
        self.sim = base.sim
        self.robot, self.left = base.scene["right_robot"], base.scene["left_robot"]
        self.peel_path = "/World/envs/env_0/peel"
        self.bodies = [
            "/World/envs/env_0/pizza",
            *[f"/World/Food/rim{i:02}" for i in range(16)],
            *[f"/World/Food/pad{i:02}" for i in range(12)],
        ]
        self.food_sources = [self.bodies[0], "/World/Food/*"]
        self.arm_roots = ("/World/envs/env_0/RightArm", "/World/envs/env_0/LeftArm")
        self.gains = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
        self.opening = FINGER_OPEN

    def _prepare_episode(self):
        self.writer = None
        self.continuous_angle = 0.0
        self.previous_angle = None
        if args.video:
            import omni.replicator.core as rep
            import imageio.v2 as imageio

            self.cam = rep.create.camera(
                position=(-0.20, -1.68, 2.10),
                look_at=(0.53, OVEN_Y, 0.96),
                focal_length=19.0,
            )
            self.product = rep.create.render_product(self.cam, (800, 520))
            self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb.attach([self.product])
            self.top_cam = rep.create.camera(
                position=(MOUTH + 0.23, OVEN_Y, 1.92),
                rotation=(0.0, 0.0, 0.0),
                focal_length=34.0,
            )
            self.top_product = rep.create.render_product(self.top_cam, (480, 384))
            self.top_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            self.top_rgb.attach([self.top_product])
            self.writer = imageio.get_writer(
                str(args.out.with_suffix(".mp4")),
                fps=20 / args.camera_stride,
                codec="libx264",
                quality=7,
            )
        if self.env is None:
            self.sim.reset()
            self.robot.reset()
            self.left.reset()
            self.robot.write_joint_position_to_sim_index(
                position=self.robot.data.default_joint_pos.torch.clone()
            )
            self.robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(self.robot.data.default_joint_pos.torch)
            )
        self.pv = self.sim.physics_manager.get_physics_sim_view()
        self.peel = self.pv.create_rigid_body_view(self.peel_path)
        self.food = self.pv.create_rigid_body_view(self.bodies[0])
        self.sensor = self.pv.create_rigid_contact_view(
            self.food_sources,
            filter_patterns=[["/World/Stone"]] * len(self.food_sources),
            max_contact_data_count=256,
        )
        print(
            "PEEL MASS, COM, INERTIA",
            self.peel.get_masses().numpy(),
            self.peel.get_coms().numpy(),
            self.peel.get_inertias().numpy(),
            flush=True,
        )
        filters = [
            "/World/Stand/rear",
            "/World/Stand/front",
            "/World/Stone",
            *[self.arm_roots[0] + v for v in asset_link_paths().values()],
        ]
        self.tool_forces = self.pv.create_rigid_contact_view(
            [self.peel_path], filter_patterns=[filters], max_contact_data_count=256
        )
        self.filters = filters
        self.peel_food = self.pv.create_rigid_contact_view(
            [self.peel_path], filter_patterns=[self.bodies], max_contact_data_count=256
        )
        from grading import Grade

        arm_paths = [
            root + link
            for root in self.arm_roots
            for link in asset_link_paths().values()
        ]
        self.grade = (
            self.env.grade
            if self.env is not None
            else Grade(self.pv, self.bodies, arm_paths, args.rate)
        )
        self.deck_load = 2.75
        self.arm_ids = self.robot.find_joints(
            [f"joint{i}" for i in range(1, 7)], preserve_order=True
        )[0]
        self.finger_ids = self.robot.find_joints(
            ["joint7", "joint8"], preserve_order=True
        )[0]
        self.tick = 0
        self.log = []
        self.samples = []
        self.fail = None
        self.center = np.array([MOUTH + 0.23, OVEN_Y])
        self.active = False
        self.slip = 0.0
        self.gap = 0
        self.maxgap = 0
        self.shift = 0.0
        self.minforce = 1e6
        self.tip = PARK.copy()
        self.rot = self.pose_rot.copy()
        self.peel_yaw = 0.0
        self.load_trim = 0.0

    def ik(self, pos, rot):
        def residual(q):
            xyz, R = tcp_pose(q, RIGHT_BASE_POS)
            return np.r_[xyz - pos, 0.23 * Rotation.from_matrix(rot.T @ R).as_rotvec()]

        best = least_squares(residual, self.q, bounds=(LOWER, UPPER), max_nfev=25)
        if np.linalg.norm(best.fun[:3]) > 0.015:
            best = min(
                [
                    best,
                    *[
                        least_squares(residual, q, bounds=(LOWER, UPPER), max_nfev=75)
                        for q in [READY, (0, 1.7, 1.8, 0.5, 0.0, -0.3)]
                    ],
                ],
                key=lambda v: v.cost,
            )
        self.q = best.x
        return self.q

    def state(self, view):
        return view.get_transforms().numpy()[0].astype(float)

    def motion(self, label, to, t, grip=0, carry=True, yaw=0.0):
        begin = self.tip.copy()
        to = np.asarray(to)
        start_yaw = self.peel_yaw
        steps = round(t * 20 * (args.pace if carry else 1.0))
        for i in range(steps + (12 if carry else 0)):
            u = min(1.0, (i + 1) / steps)
            s = u * u * (3 - 2 * u)
            p = begin + (to - begin) * s
            measured = self.state(self.peel)
            desired = p + [-GRIP_DISTANCE, 0, HANDLE_Z]
            rot = self.pose_rot
            # Feedback around the measured, freely grasped tool. Only motor targets change.
            tool_torque = np.zeros(6)
            if carry:
                measured_rot = Rotation.from_quat(measured[3:]).as_matrix()
                qnow = (
                    self.robot.data.joint_pos.torch[0, self.arm_ids]
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                tcp, actual_rot = tcp_pose(qnow, RIGHT_BASE_POS)
                com = measured[:3] + measured_rot @ np.array([-0.285, 0, 0.009])
                local_com = actual_rot.T @ (com - tcp)
                for k in range(6):
                    pert = qnow.copy()
                    pert[k] += 0.0001
                    pt, r = tcp_pose(pert, RIGHT_BASE_POS)
                    tool_torque[k] = (
                        0.40 * 9.81 * ((pt + r @ local_com)[2] - com[2]) / 0.0001
                    )
                grip_local = measured_rot.T @ (tcp - measured[:3])
                heading = np.deg2rad(start_yaw + (yaw - start_yaw) * s)
                frame = Rotation.from_euler("z", heading).as_matrix()
                rotation_err = Rotation.from_matrix(frame @ measured_rot.T).as_rotvec()
                rel = measured_rot.T @ actual_rot
                grip_local = np.clip(
                    grip_local,
                    [-GRIP_DISTANCE - 0.023, -0.015, HANDLE_Z - 0.020],
                    [-GRIP_DISTANCE + 0.023, 0.015, HANDLE_Z + 0.020],
                )
                if label == "lift from rest" and i == steps + 11:
                    self.grasp_frame = rel
                    self.grasp_local = grip_local.copy()
                if hasattr(self, "grasp_frame") and not label.startswith("rack "):
                    rel = self.grasp_frame
                    grip_local = self.grasp_local
                if label in ("load edge", "turn"):
                    self.load_trim = float(
                        np.clip(
                            self.load_trim
                            + np.clip(
                                0.00023 * (self.deck_load - args.support),
                                -0.00045,
                                0.00045,
                            ),
                            -0.003,
                            0.009,
                        )
                    )
                else:
                    self.load_trim *= 0.80
                goal_tip = p.copy()
                goal_tip[2] += self.load_trim
                err = np.clip(
                    goal_tip - measured[:3],
                    [-0.022, -0.022, -0.005],
                    [0.022, 0.022, 0.005],
                )
                desired = goal_tip + frame @ grip_local + 0.60 * err
                rot = (
                    Rotation.from_rotvec(
                        0.65 * np.clip(rotation_err, -0.10, 0.10)
                    ).as_matrix()
                    @ frame
                    @ rel
                )
            if label == "open on rack":
                q = self.release_q
            elif label in ("clear rack", "observe parked peel"):
                desired = self.release_tcp + [
                    0.0,
                    0.0,
                    0.08 * (s if label == "clear rack" else 1.0),
                ]
                q = self.ik(desired, self.release_rotation)
            else:
                q = np.array(READY) if args.park_control else self.ik(desired, rot)
            target = torch.tensor([q.tolist()], dtype=torch.float32, device="cuda:0")
            finger_target = (
                self.open_from + (grip - self.open_from) * s
                if label == "open on rack"
                else grip
            )
            fingers = torch.full(
                (1, 2), float(finger_target), dtype=torch.float32, device="cuda:0"
            )
            extra = torch.tensor(
                [tool_torque.tolist()], dtype=torch.float32, device="cuda:0"
            )
            if self.env is not None:
                # Fold the known peel payload into joint position targets. The
                # policy-facing action only contains angles and gripper openings.
                action = self.ready_action.clone()
                action[0, 7:13] = torch.as_tensor(
                    np.clip(q + tool_torque / self.gains, LOWER, UPPER), device="cuda:0"
                )
                action[0, 13] = float(np.clip(finger_target / self.opening, 0.0, 1.0))
                self.env.step(action)
                self.tick += args.rate // 20
                f = self.sensor.get_contact_force_matrix(dt=self.dt).numpy()
                self.deck_load = float(np.abs(f[..., 2]).sum())
                self.minforce = self.grade.min_deck
                self.shift = self.grade.shift
                self.maxgap = int(self.grade.fault == "lost deck support")
            else:
                for j in range(args.rate // 20):
                    cmd = self.robot.actuators.target_command
                    cmd.set_position_index(value=target, joint_ids=self.arm_ids)
                    cmd.set_position_index(value=fingers, joint_ids=self.finger_ids)
                    grav = self.robot.data.gravity_compensation_forces.torch
                    cmd.set_effort_index(
                        value=(grav[:, self.arm_ids] + extra).contiguous(),
                        joint_ids=self.arm_ids,
                    )
                    self.left.actuators.target_command.set_position_index(
                        value=self.left.data.default_joint_pos.torch
                    )
                    self.left.actuators.target_command.set_effort_index(
                        value=self.left.data.gravity_compensation_forces.torch[
                            :, self.arm_ids
                        ].contiguous(),
                        joint_ids=self.arm_ids,
                    )
                    self.left.write_data_to_sim()
                    self.robot.write_data_to_sim()
                    self.sim.step(render=False)
                    self.robot.update(self.dt)
                    self.left.update(self.dt)
                    self.tick += 1
                    if self.tick >= round(0.5 / self.dt):
                        self.grade.observe()
                    if self.active:
                        f = self.sensor.get_contact_force_matrix(dt=self.dt).numpy()
                        normal = float(np.abs(f[..., 2]).sum())
                        self.minforce = min(self.minforce, normal)
                        self.deck_load = normal
                        self.gap = self.gap + 1 if normal < 0.001 else 0
                        self.maxgap = max(self.maxgap, self.gap)
                        food = self.state(self.food)
                        self.shift = max(
                            self.shift, float(np.linalg.norm(food[:2] - self.center))
                        )
            food = self.state(self.food)
            peel = self.state(self.peel)
            pizza_yaw = Rotation.from_quat(food[3:]).as_euler("zyx", degrees=True)[0]
            if self.previous_angle is not None:
                self.continuous_angle += (
                    pizza_yaw - self.previous_angle + 180.0
                ) % 360.0 - 180.0
            self.previous_angle = pizza_yaw
            record = dict(
                continuous_yaw=self.continuous_angle,
                t=round(self.tick * self.dt, 3),
                phase=label,
                peel=peel.tolist(),
                food=food.tolist(),
                yaw=float(pizza_yaw),
                right_target=q.tolist(),
                grip_target=float(finger_target),
                tool_gravity=tool_torque.tolist(),
                target_head=p.tolist(),
                right_q=self.robot.data.joint_pos.torch[0].cpu().tolist(),
            )
            record["deck_N"] = round(self.deck_load, 4)
            record["blade_food_N"] = round(
                float(
                    np.linalg.norm(
                        self.peel_food.get_contact_force_matrix(dt=self.dt)
                        .numpy()
                        .reshape(-1, 3),
                        axis=1,
                    ).sum()
                ),
                4,
            )
            record["forces"] = {
                n: round(float(np.linalg.norm(f)), 3)
                for n, f in zip(
                    self.filters,
                    self.tool_forces.get_contact_force_matrix(dt=self.dt)
                    .numpy()
                    .reshape(-1, 3),
                )
                if np.linalg.norm(f) > 0.005
            }
            if self.env is not None:
                record["action"] = action[0].tolist()
            self.samples.append(record)
            if self.writer and len(self.samples) % args.camera_stride == 0:
                # The raw Replicator annotators need an explicit Kit pump in headless Lab.
                # Without it get_data() can repeat the very first image for the entire run.
                from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
                    ensure_isaac_rtx_render_update,
                )

                self.sim.forward()
                ensure_isaac_rtx_render_update(force=True)
                image = np.asarray(self.rgb.get_data())
                top = np.asarray(self.top_rgb.get_data())
                if image.ndim == 3 and top.ndim == 3 and image.size and top.size:
                    from PIL import Image, ImageDraw, ImageFont

                    canvas = Image.new("RGB", (1280, 528), (24, 29, 34))
                    canvas.paste(Image.fromarray(image[:, :, :3]), (0, 4))
                    canvas.paste(
                        Image.fromarray(image[:, :, :3])
                        .crop((385, 71, 685, 311))
                        .resize((480, 384)),
                        (800, 12),
                    )
                    draw = ImageDraw.Draw(canvas)
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23
                    )
                    draw.text(
                        (821, 401),
                        f"{label}   |   {self.tick * self.dt:.1f} s",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (821, 439),
                        f"Rotation  {self.continuous_angle:+.1f} deg",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (821, 477),
                        f"Offset  {1000 * np.linalg.norm(food[:2] - self.center):.1f} mm    Deck  {self.deck_load:.2f} N",
                        font=font,
                        fill="white",
                    )
                    self.writer.append_data(np.asarray(canvas))
        self.tip = to.copy()
        self.peel_yaw = yaw
        print(
            "PHASE",
            label,
            "head",
            np.round(peel[:3], 3),
            "wanted",
            np.round(to, 3),
            "shift",
            round(self.shift * 1000, 1),
            "free_ms",
            round(self.maxgap * self.dt * 1000, 1),
            "yaw",
            round(pizza_yaw, 2),
            "food N",
            record["blade_food_N"],
            "stone N",
            record["deck_N"],
            flush=True,
        )
        self.log.append(self.samples[-1])
        return peel, food

    def run(self):
        self.tip = PARK + [0, 0, 0.065]
        self.motion("settle", self.tip, 0.6, -0.047, False)
        if args.park_control:
            return
        self.motion("approach handle", PARK, 1.6, -0.047, False)
        self.motion("pinch handle", PARK, 0.9, 0, False)
        pose, _ = self.motion("lift from rest", PARK + [0, 0, 0.035], 1.4)
        if pose[2] < PARK[2] + 0.009 or np.linalg.norm(pose[:2] - PARK[:2]) > 0.05:
            self.fail = "physical grasp did not retain the peel during lift"
            return
        self.active = True
        self.sides()
        self.park_tool()

    def park_tool(self):
        self.motion("rack approach", PARK + [0, 0, 0.045], 2.4)
        self.motion("rack alignment", PARK + [0, 0, 0.012], 1.3)
        self.motion("rack seating", PARK + [0, 0, -0.001], 1.2)
        self.release_q = (
            self.robot.data.joint_pos.torch[0, self.arm_ids]
            .cpu()
            .numpy()
            .astype(float)
            .copy()
        )
        self.release_tcp, self.release_rotation = tcp_pose(
            self.release_q, RIGHT_BASE_POS
        )
        self.open_from = float(
            self.robot.data.joint_pos.torch[0, self.finger_ids].mean().item()
        )
        self.motion("open on rack", PARK, 1.2, -0.047, False)
        self.motion("clear rack", PARK, 0.9, -0.047, False)
        self.motion("observe parked peel", PARK, 3.2, -0.047, False)

    def sides(self):
        low = STONE_Z + 0.001
        gains = {1: 0.80, -1: 0.65}
        for i in range(args.cycles):
            c = self.state(self.food)[:2]
            side = 1 if i % 2 == 0 else -1
            # Enter the oven in a clear lane. Both contacts produce clockwise torque.
            start_angle = self.continuous_angle
            lane = side * 0.274
            outside = np.r_[MOUTH - 0.038, c[1] + lane, low]
            half = 0.030
            x = c[0] - side * half
            aim = side * args.aim
            self.motion("align", outside, 1.1, yaw=aim)
            tip, _ = self.motion(
                "enter clear lane", [x, c[1] + lane, low], 1.05, yaw=aim
            )
            if np.linalg.norm(tip[:3] - [x, c[1] + lane, low]) > 0.04:
                self.fail = "blade missed the approach lane"
                break
            correction = np.clip(
                args.center_gain * (self.center[1] - c[1]), -0.026, 0.026
            )
            contact_y = c[1] + side * args.side + correction
            self.motion("insert", [x, contact_y, low], 0.75, yaw=aim)
            self.motion(
                "load edge", [x, contact_y, STONE_Z + args.loaded_z], 0.48, yaw=aim
            )
            head = self.state(self.peel)[:2]
            c = self.state(self.food)[:2]
            signed = np.clip(
                (-180.0 - self.continuous_angle) / gains[side], -args.sweep, args.sweep
            )
            rotate = Rotation.from_euler("z", signed, degrees=True).as_matrix()[:2, :2]
            end = (
                c
                + rotate @ (head - c)
                + np.clip(1.15 * (self.center - c), -0.021, 0.021)
            )
            self.motion("turn", np.r_[end, STONE_Z + args.loaded_z], 0.66, yaw=aim)
            self.motion("unload", np.r_[end, low], 0.32, yaw=aim)
            free_y = self.state(self.food)[1] + lane
            self.motion("clear rim", [end[0], free_y, low], 0.85, yaw=aim)
            self.motion("lift clear", [end[0], free_y, low + 0.022], 0.42, yaw=aim)
            self.motion("withdraw", outside + [0, 0, 0.022], 0.82, yaw=aim)
            if self.shift > 0.05 or self.maxgap:
                self.fail = "centre or deck support exceeded limit"
                break
            if abs(signed) > 6:
                observed = (self.continuous_angle - start_angle) / signed
                gains[side] = float(
                    np.clip(0.65 * gains[side] + 0.35 * observed, 0.35, 1.6)
                )
            tolerance = 4.8 if self.env is not None else 2.0
            if abs(self.continuous_angle + 180.0) <= tolerance:
                break
            # Keep enough controls for an unhurried return and hands-off dwell.
            if self.env is not None and len(self.samples) + 500 >= 5000:
                break

    def save(self):
        if self.writer:
            self.writer.close()
        args.out.write_text(
            json.dumps(
                dict(
                    engine="Isaac Sim PhysX / YAM",
                    fail=self.fail,
                    grade=self.grade.result(),
                    completed_cycles=sum(x["phase"] == "withdraw" for x in self.log),
                    max_shift_mm=self.shift * 1000,
                    longest_unloaded_ms=self.maxgap * self.dt * 1000,
                    min_support_N=self.minforce if self.active else None,
                    phases=self.log,
                    samples=self.samples,
                )
            )
        )


if __name__ == "__main__":
    try:
        episode = NativeRun()
        try:
            episode.run()
        finally:
            episode.save()
    finally:
        app.close()
