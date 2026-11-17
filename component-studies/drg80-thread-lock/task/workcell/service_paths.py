"""IK screening for physically approachable wrench flats and lever grasps."""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik
from yam_arena.demo import grip_rotation


def screen(waypoints, start, base):
    q = start.copy()
    error = 0.0
    for pos, rot in waypoints:
        nxt, pe, re = ik.solve(
            pos, ik.matrix_to_quat_xyzw(rot), q_init=q, base_pos=base, iters=95
        )
        error = max(error, pe + 0.05 * re)
        if pe > 0.002 or re > 0.035:
            return False, error
        q = nxt
    return True, error


def lever_pose(pivot, degree, neutral, approach=0.0):
    link = Rotation.from_euler("y", degree, degrees=True).as_matrix()
    rot = link @ neutral
    grip = pivot + 0.070 * link[:, 2]
    return grip - approach * rot[:, 2], rot


def lever_grasp(pivot, starts, bases):
    # Approach from the open front side of the table. Coming across the
    # gripper body from the rear let a finger push a jaw during a valve test.
    for bearing in (90, 65, 115, 40, 140, 0):
        front = Rotation.from_euler("z", bearing, degrees=True).as_matrix()
        solutions = []
        for i, side in enumerate(("left", "right")):
            for tilt in (45, 65, 85):
                for yaw in (0, 90, 180, 270):
                    lean = Rotation.from_euler(
                        "x", tilt if side == "left" else -tilt, degrees=True
                    ).as_matrix()
                    neutral = front @ lean @ grip_rotation(math.radians(yaw))
                    if abs(neutral[2, 0]) > 0.40 or neutral[0, 2] < 0.30:
                        continue
                    track = [
                        lever_pose(pivot, angle, neutral, gap)
                        for angle, gap in [
                            (0, 0.035),
                            (0, 0),
                            (15, 0),
                            (30, 0),
                            (0, 0),
                            (-15, 0),
                            (-30, 0),
                            (0, 0),
                            (30, 0),
                        ]
                    ]
                    possible, residual = screen(track, starts[i], bases[side])
                    if possible:
                        solutions.append((residual, i, neutral, tilt, yaw))
        if solutions:
            return min(solutions, key=lambda item: item[0])
    raise ValueError("no front approach spans both valve detents")


def wrench_flat(center, phase, offset, orient, start, base):
    choices = []
    for k in range(6):
        yaw = phase + k * math.pi / 3
        rot = Rotation.from_euler("z", yaw).as_matrix()
        poses = [
            (center + rot @ (offset - [d, 0, 0]) + [0, 0, h], rot @ orient)
            for d, h in ((0.036, 0.04), (0.036, 0), (0.012, 0), (0, 0))
        ]
        for step in range(7, 65, 7):
            r = Rotation.from_euler("z", yaw - math.radians(step)).as_matrix()
            poses.append((center + r @ offset, r @ orient))
        possible, residual = screen(poses, start, base)
        if possible:
            choices.append((residual, yaw, rot))
    if not choices:
        raise ValueError("no wrench flat provides a reachable tightening arc")
    return min(choices, key=lambda item: item[0])
