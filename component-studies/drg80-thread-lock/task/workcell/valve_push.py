"""Actuate the handle from its exposed side using the closed stock YAM fingertips."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from yam_arena import ik
from yam_arena.constants import PAD_TIP_PAST_TCP
from yam_arena.demo import grip_rotation

from service_paths import screen


def push_valve(direction, pivot, bench, bases, joints, commands, grips, move):
    side_names = ("left", "right")
    height = 0.066
    initial = math.degrees(bench.read()["lever"])
    initial_x = height * math.tan(math.radians(initial)) - direction * 0.036
    final_x = direction * 0.051

    def waypoint(x, z, rot):
        # Put the distal pad surfaces at the commanded point. The fingers stay
        # closed and press the cylindrical handle by ordinary rigid contact.
        return pivot + [x, 0, z] - PAD_TIP_PAST_TCP * rot[:, 2]

    candidates = []
    for i, hand in enumerate(side_names):
        for tilt in (0, 25, -25):
            for yaw in (0, 90, 180, 270):
                rot = Rotation.from_euler(
                    "x", tilt, degrees=True
                ).as_matrix() @ grip_rotation(math.radians(yaw))
                coords = [
                    (initial_x, 0.13),
                    (initial_x, height),
                    (0, height),
                    (final_x, height),
                    (final_x, 0.13),
                ]
                possible, residual = screen(
                    [(waypoint(x, z, rot), rot) for x, z in coords],
                    joints(i),
                    bases[hand],
                )
                if possible:
                    candidates.append((residual, i, tilt, yaw, rot))
    if not candidates:
        raise ValueError("closed-pad valve contact path exceeds wrist travel")
    _, i, tilt, yaw, rot = min(candidates, key=lambda c: c[0])
    base = bases[side_names[i]]
    grips[i] = 0.0
    move("valve_prepare_pusher", commands.copy(), grips, 15)

    def advance(label, x, z, steps):
        goals = commands.copy()
        goals[i], ep, ea = ik.solve(
            waypoint(x, z, rot),
            ik.matrix_to_quat_xyzw(rot),
            q_init=joints(i),
            base_pos=base,
            iters=150,
        )
        if ep > 0.002 or ea > 0.035:
            raise ValueError(f"{label} exceeds elbow or wrist travel")
        move(label, goals, grips, steps)

    advance("valve_overhead", initial_x, 0.13, 45)
    advance("valve_near_handle", initial_x, height, 32)
    history = []
    for x in np.linspace(initial_x, final_x, 45):
        advance("valve_side_press", x, height, 3)
        angle = math.degrees(bench.read()["lever"])
        history.append((float(x), angle))
        if direction * angle >= 28:
            break
    advance("valve_vertical_release", x, 0.13, 28)
    move("valve_detent_settle", commands.copy(), grips, 24)
    final = bench.read()
    return {
        "arm": side_names[i],
        "tilt_deg": tilt,
        "yaw_deg": yaw,
        "last_press_deg": history[-1][1],
        "range_deg": [min(a for _, a in history), max(a for _, a in history)],
        "lever_deg": math.degrees(final["lever"]),
        "jaw_deg": final["angle"],
        "actuator_n": final["actuator_n"],
    }
