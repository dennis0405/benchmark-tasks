"""Experimental folded-packet and spring-clip task for the YAM station.

The scorer uses PhysX jaw contacts and approximate robot clearance. The wrapper
is a rigid approximation, not deformable film; validation is still in progress.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import math
import os

import numpy as np
from scenekit import UsdBody, on_reset, on_step, robot_links
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUPPORT = 0.025
SCORE_OVER = "end"
# The monitor below counts 60 control steps of hold; avoid awarding 1.0 early.
SUCCESS_HOLD_S = 0.05
INSTRUCTION = "Keep the chips from going stale."
OBJECTS = [
    Box("support", pos=(0.24, 0.11, TOP + SUPPORT / 2), size=(0.12, 0.16, SUPPORT),
        color=(0.62, 0.52, 0.40)),
    UsdBody("bag", usd="assets/bag_v4/bag_v4.usda", pos=(0.24, 0.11, TOP + SUPPORT + 0.022),
            size=(0.20, 0.16, 0.05), articulation=True, joint_pos=(("fold", -3.10), ("fold2", 3.10)),
            joint_stiffness=0.0005, joint_damping=0.01),
    UsdBody("clip", usd="assets/clip_v5/clip_v5.usda", pos=(0.26, -0.12, TOP + 0.052),
            rot_xyzw=(0.70710678, 0.0, 0.0, 0.70710678),
            size=(0.08, 0.10, 0.014), articulation=True, joint_pos=(("clip_hinge", 0.0),),
            joint_stiffness=0.13, joint_damping=0.003),
    UsdBody("peg", usd="assets/wide_peg_v5/wide_peg_v5.usda", pos=(0.15, -0.22, TOP + 0.052),
            rot_xyzw=(0.70710678, 0.0, 0.0, 0.70710678),
            size=(0.08, 0.10, 0.014), articulation=True, joint_pos=(("clip_hinge", 0.0),),
            joint_stiffness=0.13, joint_damping=0.003),
    UsdBody("band", usd="assets/band_v3/band_v3.usda", pos=(0.38, -0.21, TOP + 0.003),
            size=(0.054, 0.04, 0.003), mass=0.0016),
    UsdBody("paperclip", usd="assets/paperclip_v3/paperclip_v3.usda", pos=(0.46, -0.21, TOP + 0.003),
            size=(0.025, 0.011, 0.002), mass=0.001),
    *(Box(f"chip{i}", pos=(0.24 + dx, 0.11 + dy, TOP + SUPPORT + 0.012), size=(0.025, 0.020, 0.003),
          mass=0.001, yaw=(i % 2) * 0.3, color=(0.93, 0.72, 0.28))
      for i, (dx, dy) in enumerate(((-0.03, -0.04), (-0.03, 0.04), (0.02, -0.04), (0.02, 0.04)))),
]

_STATE = {"arms_clear": False, "violation": False, "hold_steps": 0, "holding_tool": None,
          "contacts": {"clip": False, "peg": False}, "contact_monitor": None}


def _rotation(q):
    x, y, z, w = np.asarray(q, dtype=float)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def _measure(objects, tool="clip"):
    """Geometric candidate checks, combined with contact monitoring in score()."""
    bag, clip = objects["bag"], objects[tool]
    rb, rc = _rotation(bag.quat_xyzw), _rotation(clip.quat_xyzw)
    p = rb.T @ (clip.pos - bag.pos)
    relative = rb.T @ rc
    fold = bag.joints["fold"]
    fold2 = bag.joints["fold2"]
    jaw = clip.joints["clip_hinge"]
    folded = -3.20 <= fold <= -2.85 and 2.85 <= fold2 <= 3.20
    # Clip jaws point towards -bag X. Compute overlap across the 160 mm mouth.
    overlap = max(0.0, min(0.08, p[1] + 0.05) - max(-0.08, p[1] - 0.05))
    aligned = (relative[0, 0] < -math.cos(math.radians(10)) and
               relative[2, 2] > math.cos(math.radians(10)) and overlap >= 0.01 and
               0.112 <= p[0] <= 0.130)
    # Heights in bag frame of the lower top surface and upper bottom surface at jaw midpoint.
    c, s = math.cos(jaw), math.sin(jaw)
    lower = p + relative @ np.array([0.025, 0.0, -0.001])
    upper = p + relative @ np.array([0.025*c + 0.001*s, 0.0, -0.025*s + 0.001*c])
    # Compare against the two resting film surfaces, not against a proximity-only proxy.
    lower_gap = abs(lower[2] - (0.021 - 0.0007))
    upper_gap = abs(upper[2] - (0.0240 + 0.0007))
    jaw_on_layers = -0.45 < jaw < -0.03 and lower_gap <= 0.0005 and upper_gap <= 0.0005
    still = bag.resting(max_speed=0.02, max_spin=0.3) and clip.resting(max_speed=0.02, max_spin=0.3)
    return {"folded": folded, "aligned": aligned, "jaw_on_layers": jaw_on_layers, "still": still, "overlap": overlap,
            "lower_gap": float(lower_gap), "upper_gap": float(upper_gap), "jaw_angle": float(jaw),
            "relative_pos": p.tolist()}


@on_reset
def _reset(environment):
    _STATE["arms_clear"] = False
    _STATE["violation"] = False
    _STATE["hold_steps"] = 0
    _STATE["holding_tool"] = None
    _STATE["reason"] = None
    _STATE["steps"] = 0
    _STATE["contacts"] = {"clip": False, "peg": False}
    _STATE["contact_monitor"] = None
    try:
        from contact_monitor import ContactMonitor
        _STATE["contact_monitor"] = ContactMonitor()
    except Exception as exc:
        print(f"SNACK contact monitor unavailable; scoring fails closed: {exc!r}", flush=True)
    # The embodiment settles the scene before returning reset; restore the two
    # pre-folded joints once at reset, never during the robot rollout.
    import torch
    bag = environment.unwrapped.scene["bag"]
    ids, _ = bag.find_joints(["fold", "fold2"], preserve_order=True)
    pos = torch.tensor([[-3.10, 3.10]], device=environment.unwrapped.device)
    bag.write_joint_state_to_sim_index(position=pos, velocity=torch.zeros_like(pos), joint_ids=ids)
    bag.set_joint_position_target_index(target=torch.zeros_like(pos), joint_ids=ids)


@on_step
def _check_clear(environment):
    from yam_arena.scene import object_states
    states = object_states(environment, ["bag", "clip", "peg", "chip0", "chip1", "chip2", "chip3"])
    monitor = _STATE.get("contact_monitor")
    try:
        _STATE["contacts"] = monitor.read() if monitor else {"clip": False, "peg": False}
    except Exception as exc:
        print(f"SNACK contact read failed; scoring fails closed: {exc!r}", flush=True)
        _STATE["contact_monitor"] = None
        _STATE["contacts"] = {"clip": False, "peg": False}
    bag = states["bag"]
    rot = _rotation(bag.quat_xyzw)
    # These are rigid food proxies, not a granular or deformable chip model.
    for i in range(4):
        xyz = rot.T @ (states[f"chip{i}"].pos - bag.pos)
        # Bounding region includes tolerance around the rigid shell. This is
        # a center-based spill proxy, not exact chip/packet containment.
        if not (-0.075 <= xyz[0] <= 0.075 and abs(xyz[1]) <= 0.095 and -0.035 <= xyz[2] <= 0.035):
            if not _STATE["violation"]:
                _STATE["reason"] = f"chip{i} outside packet: {np.round(xyz, 4).tolist()}"
            _STATE["violation"] = True
    if bag.pos[2] < TOP - 0.02:
        if not _STATE["violation"]:
            _STATE["reason"] = "packet fell below tabletop"
        _STATE["violation"] = True
    points = [states["bag"].pos + _rotation(states["bag"].quat_xyzw) @ np.array([0.08, 0, 0.023]),
              states["clip"].pos, states["peg"].pos]
    links = robot_links(environment)
    # Link origins are an approximation; clearance is intentionally conservative.
    _STATE["arms_clear"] = all(np.linalg.norm(p - target) > 0.08 for arm in links.values()
                               for p in arm.values() for target in points)
    holding = None
    if not _STATE["violation"] and _STATE["arms_clear"]:
        for tool in ("clip", "peg"):
            m = _measure(states, tool)
            if (all(m[k] for k in ("folded", "aligned", "jaw_on_layers", "still")) and
                    _STATE["contacts"][tool] and m["overlap"] >= 0.096):
                holding = tool
                break
    _STATE["hold_steps"] = (_STATE["hold_steps"] + 1 if holding and holding == _STATE["holding_tool"] else
                            1 if holding else 0)
    _STATE["holding_tool"] = holding
    _STATE["steps"] += 1
    if os.environ.get("SNACK_DEBUG") and _STATE["steps"] % 20 == 0:
        print("SNACK", _STATE["steps"], "bag", bag.pos, bag.quat_xyzw, bag.joints,
              "clip", states["clip"].pos, states["clip"].quat_xyzw, states["clip"].joints,
              "violation", _STATE["violation"], "metric", _measure(states), flush=True)


def score(objects):
    if _STATE["violation"]:
        return 0.0
    result = 0.0
    for tool in ("clip", "peg"):
        if tool not in objects:
            continue
        m = _measure(objects, tool)
        if not (m["folded"] and m["aligned"] and m["jaw_on_layers"] and _STATE["contacts"].get(tool, False)):
            continue
        if (m["overlap"] >= 0.096 and m["still"] and _STATE["arms_clear"] and
                _STATE["hold_steps"] >= 60 and _STATE["holding_tool"] == tool):
            return 1.0
        result = 0.5
    return result


def success(objects):
    return score(objects) == 1.0


def demo():
    """Smoke test by default; the optional hold probe is not a full task demo."""
    if os.environ.get("SNACK_DEMO") == "hold_probe":
        from yam_arena.demo import Waypoint, FINGER_CLOSED, FINGER_OPEN
        return {
            "left": [Waypoint(1.3, (0.32, 0.11, TOP + 0.14), FINGER_CLOSED, "above fold"),
                     Waypoint(1.1, (0.32, 0.11, TOP + 0.060), FINGER_CLOSED, "press fold"),
                     Waypoint(5.0, (0.32, 0.11, TOP + 0.060), FINGER_CLOSED, "hold fold")],
            "right": [Waypoint(1.3, (0.242, -0.12, TOP + 0.14), FINGER_OPEN, "above handle"),
                      Waypoint(1.1, (0.242, -0.12, TOP + 0.027), FINGER_OPEN, "handle"),
                      Waypoint(0.8, (0.242, -0.12, TOP + 0.027), FINGER_CLOSED, "grasp"),
                      Waypoint(1.5, (0.242, -0.12, TOP + 0.14), FINGER_CLOSED, "lift"),
                      Waypoint(1.5, (0.30, -0.04, TOP + 0.14), FINGER_CLOSED, "move")],
        }
    return {}
