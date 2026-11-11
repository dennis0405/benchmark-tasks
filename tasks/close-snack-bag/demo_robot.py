"""Experimental contact-only YAM controller for the folded-bag task.

All commands are 14-D YAM joint targets; there are no object pose/joint writes
after reset. A run can fail, and its recorded result is not a claimed demo.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HZ = 20


class Arm:
    def __init__(self, side):
        from yam_arena import ik, station
        from yam_arena.constants import READY
        self.base = np.asarray(station.BASES[side])
        self.q = np.asarray(READY, float).copy()
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = self.pos.copy(), self.rot.copy()
        self.grip = 1.0
        self.worst_ik = 0.0

    def move(self, pos, rot, seconds, grip=None):
        from yam_arena import ik
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        count = max(1, round(seconds * HZ))
        for k in range(1, count + 1):
            t = ik.smoothstep(k / count)
            self.pos = p0 + t * (np.asarray(pos) - p0)
            self.rot = ik.slerp_matrix(r0, rot, t)
            self.grip = g0 + t * (g1 - g0)
            self.q, error, _ = ik.solve(self.pos, ik.matrix_to_quat_xyzw(self.rot), self.q,
                                        base_pos=self.base, iters=80)
            self.worst_ik = max(self.worst_ik, error)
            yield


def together(*generators):
    pending = list(map(iter, generators))
    while pending:
        for generator in pending[:]:
            try:
                next(generator)
            except StopIteration:
                pending.remove(generator)
        if pending:
            yield


def pause(seconds):
    for _ in range(round(seconds * HZ)):
        yield


def plan(left, right, emb, task):
    from yam_arena import ik
    from yam_arena.demo import grip_rotation, axis_angle_matrix
    from yam_arena.scene import object_states
    down = grip_rotation(0)
    clip = object_states(emb._env, ["clip"])["clip"]
    # Left presses one side of the fold; right squeezes the upright clip's handles.
    handle = clip.pos + task._rotation(clip.quat_xyzw) @ np.array([-0.018, 0, 0])
    left_pos = np.array([0.32, 0.175, task.TOP + task.SUPPORT + 0.056])
    yield from together(left.move(left_pos + [0, 0, 0.08], down, 0.8, 0),
                        right.move(handle + [0, 0, 0.10], down, 0.8, 1))
    yield from together(left.move(left_pos, down, 0.8, 0),
                        right.move(handle + [0, 0, 0.013], down, 0.8, 1))
    yield from right.move(right.pos, down, 1.2, 0)
    yield from pause(.5)
    # Capture measured grasp transform after gripping, solely for motion planning.
    clip = object_states(emb._env, ["clip"])["clip"]
    cr = task._rotation(clip.quat_xyzw)
    grasp_offset = cr.T @ (right.pos - clip.pos)
    relative_rot = cr.T @ right.rot
    yield from right.move(right.pos + [0, 0, 0.09], right.rot, 1.8, 0)
    desired_clip_rot = np.diag([-1.0, -1.0, 1.0])
    desired_grip_rot = desired_clip_rot @ relative_rot
    yield from right.move(right.pos, desired_grip_rot, 2.0, 0)
    # Approach from beyond the lip then advance the open jaws over the fold.
    bag = object_states(emb._env, ["bag"])["bag"]
    br = task._rotation(bag.quat_xyzw)
    target = bag.pos + br @ np.array([0.125, -0.025, 0.0213])
    staging = target + [0.045, 0, 0]
    yield from right.move(staging + desired_clip_rot @ grasp_offset,
                          desired_grip_rot, 2.5, 0)
    # Compensate for clip slip in the gripper before bringing it into contact.
    for _ in range(6):
        clip = object_states(emb._env, ["clip"])["clip"]
        correction = np.clip(staging - clip.pos, -0.01, 0.01)
        print("STAGE_CORRECTION", np.round(correction, 4), "pose", np.round(clip.pos, 4), flush=True)
        if np.linalg.norm(correction) < 0.002:
            break
        yield from right.move(right.pos + correction, desired_grip_rot, 0.5, 0)
    yield from right.move(right.pos + target - staging, desired_grip_rot, 2.5, 0)
    for _ in range(4):
        clip = object_states(emb._env, ["clip"])["clip"]
        correction = np.clip(target - clip.pos, -0.012, 0.012)
        states = object_states(emb._env, ["bag", "clip"])
        print("CLIP_CORRECTION", np.round(correction, 4), "pose", np.round(clip.pos, 4),
              "quat", np.round(clip.quat_xyzw, 3), "geometry", task._measure(states), flush=True)
        if np.linalg.norm(correction) < 0.002:
            break
        yield from right.move(right.pos + correction, desired_grip_rot, 0.6, 0)
    yield from right.move(right.pos, desired_grip_rot, 1.0, 1)
    # Slide clear of the handles before lifting, so the opening fingers do not pull the clip off.
    yield from right.move(right.pos + [0.06, 0, 0], desired_grip_rot, 1.8, 1)
    yield from right.move(right.pos + [0, 0, 0.08], desired_grip_rot, 1.5, 1)
    yield from left.move(left.pos + [0, 0, 0.09], down, 1.2, 1)
    yield from together(left.move(*left.home, 2.0, 1), right.move(*right.home, 2.0, 1))
    yield from pause(3.5)


def run(output, record):
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    folder = Path(__file__).resolve().parent
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=str(folder / "arena_scene.py"), object_jitter=0, control_hz=HZ,
                               cam_width=320, cam_height=176)
    obs = emb.reset(Scene(id="snack", instruction="Keep the chips from going stale."))
    from yam_arena.scene import object_states
    task = emb.task.module
    left, right = Arm("left"), Arm("right")
    controller = plan(left, right, emb, task)
    frames = {name: [] for name in obs.images}
    if record:
        for name, image in obs.images.items():
            frames[name].append(np.asarray(image))
    success, last_info, steps = False, {}, 0
    for step in range(2400):
        try:
            next(controller)
        except StopIteration:
            break
        result = emb.step(Action(data=pack(np.r_[left.q, left.grip], np.r_[right.q, right.grip])))
        steps, last_info = step + 1, result.info
        if record:
            for name, image in result.observation.images.items():
                frames[name].append(np.asarray(image))
        if steps % 40 == 0:
            objects = object_states(emb._env, ["bag", "clip"])
            print("step", steps, "score", last_info.get("score"), "fold", objects["bag"].joints,
                  "clip_pos", np.round(objects["clip"].pos, 4), "clip_angle", objects["clip"].joints,
                  "violation", task._STATE["violation"], task._STATE["reason"], flush=True)
        if result.terminated:
            success = result.termination_reason == "success"
            break
    result = {"success": success, "score": float(last_info.get("score", 0)), "steps": steps,
              "violation": task._STATE["violation"], "reason": task._STATE["reason"],
              "worst_ik_mm": {"left": left.worst_ik * 1000, "right": right.worst_ik * 1000}}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    if record:
        import imageio
        for name, images in frames.items():
            imageio.mimsave(output / f"{name}.mp4", images, fps=HZ, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    run(args.output, args.record)
