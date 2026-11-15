"""Industrial LM-rail datum seating prototype on the calibrated YAM station.

Two arms share a 320 mm steel rail, lift it from the preparation area, place it on the
mounting base, and register the rail's marked side against the blue side reference.
The fixed orange end stops make longitudinal placement visible. This rapid prototype
ends before bolt insertion or torqueing and makes no claim about micron straightness.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from yam_arena.demo import Waypoint
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

RAIL_LENGTH = 0.320
RAIL_WIDTH = 0.028
RAIL_HEIGHT = 0.016
BASE_TOP = TOP + 0.012
TARGET_X = 0.280
DATUM_INNER_Y = -0.006
TARGET_Y = DATUM_INNER_Y - RAIL_WIDTH / 2
TARGET_Z = BASE_TOP + RAIL_HEIGHT / 2

RAIL = Box(
    "lm_rail",
    pos=(TARGET_X, -0.100, TOP + RAIL_HEIGHT / 2),
    size=(RAIL_LENGTH, RAIL_WIDTH, RAIL_HEIGHT),
    mass=0.80,
    color=(0.42, 0.46, 0.50),
)
BASE = Box(
    "mounting_base",
    pos=(TARGET_X, 0.030, TOP + 0.006),
    size=(0.380, 0.150, 0.012),
    color=(0.16, 0.18, 0.20),
)
DATUM = Box(
    "side_datum",
    pos=(TARGET_X, DATUM_INNER_Y + 0.006, BASE_TOP + 0.014),
    size=(0.360, 0.012, 0.028),
    color=(0.10, 0.34, 0.82),
)
STOP_NEAR = Box(
    "end_stop_near",
    pos=(TARGET_X - RAIL_LENGTH / 2 - 0.016, TARGET_Y, BASE_TOP + 0.012),
    size=(0.012, 0.050, 0.024),
    color=(0.92, 0.42, 0.06),
)
STOP_FAR = Box(
    "end_stop_far",
    pos=(TARGET_X + RAIL_LENGTH / 2 + 0.016, TARGET_Y, BASE_TOP + 0.012),
    size=(0.012, 0.050, 0.024),
    color=(0.92, 0.42, 0.06),
)
OBJECTS = [RAIL, BASE, DATUM, STOP_NEAR, STOP_FAR]
SUCCESS_HOLD_S = 2.0
SCORE_OVER = "end"

_trace_path = os.environ.get("RAPID_BODY_TRACE")
_sample_index = 0


def _yaw(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _errors(objects):
    rail = objects["lm_rail"]
    yaw = _yaw(rail.quat_xyzw)
    yaw_line_error = abs(math.atan2(math.sin(yaw), math.cos(yaw)))
    x, y, z, q = (float(v) for v in rail.quat_xyzw)
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*q),2*(x*z+y*q)],
                         [2*(x*y+z*q),1-2*(x*x+z*z),2*(y*z-x*q)],
                         [2*(x*z-y*q),2*(y*z+x*q),1-2*(x*x+y*y)]])
    endpoints = [rail.pos + rotation @ np.array([side*RAIL_LENGTH/2,RAIL_WIDTH/2,0.0]) for side in (-1,1)]
    gaps = [DATUM_INNER_Y-float(point[1]) for point in endpoints]
    datum_gap = max(abs(gap) for gap in gaps)
    return {
        "centre_xy_m": float(np.linalg.norm(rail.pos[:2] - np.array([TARGET_X, TARGET_Y]))),
        "height_m": abs(float(rail.pos[2]) - TARGET_Z),
        "yaw_rad": yaw_line_error,
        "datum_gap_m": datum_gap,
        "datum_penetration_m": max(0.0,-min(gaps)),
        "speed_m_s": float(np.linalg.norm(rail.lin_vel)),
        "spin_rad_s": float(np.linalg.norm(rail.ang_vel)),
    }


def _trace(objects, errors) -> None:
    global _sample_index
    if not _trace_path:
        return
    record = {
        "sample_index": _sample_index,
        "errors": errors,
        "bodies": {
            name: {
                "pos_m": state.pos.tolist(),
                "quat_xyzw": state.quat_xyzw.tolist(),
                "lin_vel_m_s": state.lin_vel.tolist(),
                "ang_vel_rad_s": state.ang_vel.tolist(),
                "aabb_low_m": state.low.tolist(),
                "aabb_high_m": state.high.tolist(),
            }
            for name, state in objects.items()
        },
    }
    path = Path(_trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    _sample_index += 1


def success(objects):
    e = _errors(objects)
    return bool(
        e["centre_xy_m"] <= 0.012
        and e["height_m"] <= 0.006
        and e["yaw_rad"] <= math.radians(4.0)
        and e["datum_gap_m"] <= 0.006
        and e["datum_penetration_m"] <= 0.001
        and e["speed_m_s"] < 0.01
        and e["spin_rad_s"] < 0.5
    )


def distance(objects):
    e = _errors(objects)
    return float(math.hypot(e["centre_xy_m"], e["height_m"]) + e["datum_gap_m"])


def score(objects):
    e = _errors(objects)
    _trace(objects, e)
    if success(objects):
        return 1.0
    if e["height_m"] <= 0.012 and e["centre_xy_m"] <= 0.030 and e["yaw_rad"] <= math.radians(10.0):
        return 0.75 if e["datum_gap_m"] <= 0.015 else 0.50
    if float(objects["lm_rail"].low[2]) >= TOP + 0.020:
        return 0.25
    return 0.0


def _arm_waypoints(x: float, side: str):
    sy = RAIL.pos[1]
    sz = RAIL.pos[2]
    hover = TOP + 0.135
    place_z = TARGET_Z + 0.004
    pre_y = TARGET_Y - 0.018
    push_y = TARGET_Y - 0.058
    home_y = 0.23 if side == "left" else -0.23
    return [
        Waypoint(1.8, (x, sy, hover), FINGER_OPEN, "approach shared rail"),
        Waypoint(1.5, (x, sy, sz), FINGER_OPEN, "descend to rail"),
        Waypoint(1.0, (x, sy, sz), FINGER_CLOSED, "close on rail"),
        Waypoint(1.5, (x, sy, hover), FINGER_CLOSED, "bimanual lift"),
        Waypoint(2.0, (x, pre_y, hover), FINGER_CLOSED, "carry clear of side datum"),
        Waypoint(1.5, (x, pre_y, place_z), FINGER_CLOSED, "lower on base"),
        Waypoint(0.8, (x, pre_y, place_z), FINGER_OPEN, "release onto base"),
        Waypoint(1.0, (x, pre_y, hover), FINGER_OPEN, "clear rail"),
        Waypoint(0.8, (x, push_y, hover), FINGER_CLOSED, "prepare contact seating"),
        Waypoint(1.0, (x, push_y, TARGET_Z+0.003), FINGER_CLOSED, "approach trailing edge"),
        Waypoint(2.0, (x, TARGET_Y-0.023, TARGET_Z+0.003), FINGER_CLOSED, "gently seat both ends"),
        Waypoint(0.8, (x, push_y, TARGET_Z+0.003), FINGER_CLOSED, "remove seating contact"),
        Waypoint(0.9, (x, push_y, hover), FINGER_OPEN, "lift clear"),
        Waypoint(1.3, (0.18, home_y, hover), FINGER_OPEN, "park away from fixture"),
    ]


def demo():
    return {
        "left": _arm_waypoints(TARGET_X - 0.070, "left"),
        "right": _arm_waypoints(TARGET_X + 0.070, "right"),
    }
