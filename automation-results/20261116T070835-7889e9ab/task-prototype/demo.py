"""Two-arm native manipulation trial. Record actions, free-body trajectories and visible results."""

import argparse, json, math, os
from pathlib import Path

os.environ["HEADLESS"] = "1"
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--idle-steps", type=int, default=0)
p.add_argument("--replay", type=Path)
p.add_argument("--report", type=Path, required=True)
p.add_argument("--render-video", action="store_true")
p.add_argument("--seed", type=int, default=8261)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = args.render_video
app = AppLauncher(args)
import torch, numpy as np, imageio.v2 as imageio
from yam_arena import demo
from yam_arena.constants import (
    FINGER_CLOSED as closed,
    FINGER_OPEN as opened,
    LEFT_BASE_POS,
    RIGHT_BASE_POS,
    READY,
)
import task
import full_screen_scene as sc

args.report.parent.mkdir(parents=True, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env = task.make_env(args.device, args.render_video)
base = env.unwrapped
env.reset(seed=args.seed)
grip_z = sc.SCREEN_ROOT_Z + sc.TAB_Z + 0.013
height = 0.067
freq = 1.8
cycles = 15


class Plan(demo.ArmPlan):
    def target(self, t):
        pos, rot, g, label = super().target(t)
        if label == "agitate":
            u = max(0.0, t - 9.0)
            envelope = min(1.0, u / 2.0, max(0.0, (cycles / freq - u) / 2.0))
            roll = np.deg2rad(4.0) * envelope * np.sin(2 * np.pi * 0.22 * u)
            cr, sr = np.cos(roll), np.sin(roll)
            Rroll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            arm_y = self.waypoints[0].pos[1]
            pos = pos + np.array([0, arm_y * (cr - 1), arm_y * sr])
            rot = Rroll @ rot
            pos = pos + np.array(
                [
                    0.003 * envelope * np.sin(2 * np.pi * freq * u),
                    0.022 * envelope * np.sin(2 * np.pi * freq * u),
                    0.0,
                ]
            )
        return pos, rot, g, label


def plan(side, basepos):
    x, y = sc.SCREEN_X, side * sc.TAB_Y
    times = [
        (2, 0.1, opened, "approach"),
        (2, 0, opened, "descend"),
        (1, 0, closed, "pinch"),
        (2, height, closed, "lift"),
        (2, height, closed, "hold"),
        (cycles / freq, height, closed, "agitate"),
        (1, height, closed, "stabilize"),
        (2, 0, closed, "reseat"),
        (1, 0, opened, "release"),
        (1.5, 0.16, opened, "retreat"),
        (4.0, 0.16, opened, "dwell"),
    ]
    return Plan(
        [demo.Waypoint(t, (x, y, grip_z + z), g, l) for t, z, g, l in times], basepos
    )


plans = [plan(1, LEFT_BASE_POS), plan(-1, RIGHT_BASE_POS)]
cam = base.scene.sensors.get("overview_cam")
trace = []
from contextlib import nullcontext

grade = env.grade
video = args.report.with_suffix(".mp4")
recorded_actions = None
if args.replay:
    import json

    recorded_actions = [
        np.asarray(row["action"], dtype=float)
        for row in json.loads(args.replay.read_text())
    ]
    if any(a.shape != (14,) or not np.isfinite(a).all() for a in recorded_actions):
        raise ValueError("Replay requires finite 14D YAM actions")
episode = (
    range(args.idle_steps)
    if args.idle_steps
    else range(len(recorded_actions))
    if recorded_actions is not None
    else range(-round(3.0 * 20), math.ceil(plans[0].duration * 20))
)
with (
    imageio.get_writer(video, fps=20, codec="libx264", quality=8)
    if args.render_video
    else nullcontext()
) as writer:
    for k in episode:
        if args.idle_steps:
            actions = [(np.r_[READY, 1.0], "idle")] * 2
        elif recorded_actions is not None:
            actions = [
                (recorded_actions[k][:7], "replay"),
                (recorded_actions[k][7:], "replay"),
            ]
        elif k < 0:
            actions = [(np.r_[READY, 1.0], "settle")] * 2
        else:
            actions = [v.action7(k / 20) for v in plans]
        action = np.concatenate([x[0] for x in actions])
        env.step(torch.tensor(action, device=args.device, dtype=torch.float32)[None])
        bodies = {
            n: b.root_view.get_transforms().numpy()[0].tolist()
            for n, b in base.scene.rigid_objects.items()
        }
        trace.append(
            dict(
                time=(len(trace) + 1) / 20,
                phase=actions[0][1],
                score=grade.score,
                received=len(grade.passage.received),
                bodies=bodies,
                action=action.tolist(),
            )
        )
        if k % 40 == 0:
            screen = bodies["screen"]
            print(f"GRADE {grade.last}", flush=True)
            print(
                f"STEP {k} {actions[0][1]} screen={screen[:3]} collected={sum(v[2] < 0.79 for n, v in bodies.items() if n.startswith('media_'))}",
                flush=True,
            )
        if writer:
            writer.append_data(
                cam.data.output["rgb"].torch[0].cpu().numpy()[:, :, :3].astype("uint8")
            )
        if any(
            abs(v[0]) > 1.5 or abs(v[1]) > 1.5 or v[2] < 0.50 for v in bodies.values()
        ):
            break
tp = args.report.with_suffix(".trace.json")
tp.write_text(json.dumps(trace, separators=(",", ":")))
row = trace[-1]
report = {
    "last_phase": row["phase"],
    "final_poses": row["bodies"],
    "max_lift": max(s["bodies"]["screen"][2] for s in trace) - sc.SCREEN_ROOT_Z,
    "checks": [
        {
            "name": "executed_control_steps",
            "measured": len(trace),
            "operator": ">",
            "threshold": 0,
            "unit": "frames",
        }
    ],
    "artifacts": [x.name for x in ([tp, video] if args.render_video else [tp])],
}
native = args.report.with_suffix(".physics.npz")
np.savez_compressed(native, poses=grade.history, names=grade.body_names, dt=grade.dt)
report["artifacts"].append(native.name)
report.update(
    grading=grade.last,
    maximum_score=grade.max_score,
    scope="Native single-episode result",
    seed=args.seed,
    control_steps=len(trace),
    physics_hz=round(1 / grade.dt),
    control_hz=20,
)
args.report.write_text(json.dumps(report))
env.close()
app.app.close()
