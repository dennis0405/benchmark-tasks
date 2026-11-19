"""Painting: "Close the paint can."

A half-full 1-gallon paint can stands on a canvas drop cloth, its lid set on the ring but not tapped
down (about 4 mm proud, wet paint in the groove), and an 8 oz rubber mallet lies beside it. The lid is
closed by tapping its rim all the way round with the mallet while the other hand braces the can.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import simkit  # noqa: E402
import rim
import paint as P  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "paint-can-lid-close"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
SCENE_ONLY = False
MAX_STEPS = 80 if SCENE_ONLY else 1800
OBJECT_NAMES = ("can", "mallet", *rim.SECTORS)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (P.cloth_object(), rim.can(), P.mallet_object(), *(rim.sector(i) for i in range(8)))
    objects += [yscene.task_object(s) for s in specs]
    def configure(cfg):
        yscene.env_cfg_callback(control_hz)(cfg)
        cfg.sim.dt=1/480; cfg.decimation=24; cfg.sim.render_interval=24
        cfg.episode_length_s=MAX_STEPS/control_hz+4
        return cfg
    cfg = IsaacLabArenaEnvironment(name="yam_paint_can_lid_close", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=configure)
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


# -- measured hands --------------------------------------------------------------------------------
PAD_T = 0.008                        # finger pad box: thickness out from its face ...
PAD_W = 0.024                        # ... width across the jaw ...
PAD_L = 0.045                        # ... length back from the tip


def _bases():
    from yam_arena import station
    return {side: np.asarray(station.BASES[side], float) for side in ("left", "right")}


def hand_points(robot):
    """Per hand: the finger pads as oriented boxes (centre, axes, half-sizes), placed from the measured
    joints and finger travel by the URDF's forward kinematics at the runner's arm bases."""
    from yam_arena import ik
    bases = _bases()
    out = {}
    for side in ("left", "right"):
        q = np.asarray(robot[side]["joints"], float)
        pos, R = ik.tcp_pose(q, bases[side])
        half = abs(float(robot[side]["fingers"][0]))
        tip = pos + R[:, 2] * PAD_TIP
        boxes = []
        for sgn in (-1.0, 1.0):
            c = tip + R[:, 0] * sgn * (half + PAD_T / 2) - R[:, 2] * (PAD_L / 2)
            boxes.append((c, R.copy(), np.array([PAD_T / 2, PAD_W / 2, PAD_L / 2])))
        out[side] = (tip, half, boxes)
    return out


def head_faces(pose):
    """The mallet head's two striking faces (centres, world) and its axis."""
    p = np.asarray(pose, float)
    ax = P.mat(p) @ np.array([0.0, 1.0, 0.0])
    return p[:3] + ax * P.HEAD_L / 2, p[:3] - ax * P.HEAD_L / 2, ax


from measurement import Dynamics


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- scripted demo -----------------------------------------------------------------------------------
# Right hand: takes the mallet's handle from above (fingers across it), lifts it and turns it a quarter turn
# about the handle so the head's faces point up and down (the grip now side-on, fingers above and below the
# handle), then taps the lid's rim sector by sector round the can and lays the mallet back on the cloth.
# Left hand: braces the can's wall from the +y side, fingers closed, low down, clear of the strikes.
Z = np.array([0.0, 0.0, 1.0])
ALONG = 0.045                          # grip point: this far up the handle from the head's centre (by its
                                       # centre of mass, so the head-heavy mallet does not swing in the pinch)
DYAW = math.radians(-40.0)             # the handle's heading at a strike: from the head toward the right base, turned
HOVER = 0.052                          # the face this high over the lid before a tap
PRESS = 0.022                          # ... and driven this far below its top on the tap
ORDER = (4, 5, 6, 7, 0, 1, 2, 3)       # sectors 180, 225, ... 135 degrees: start at the front
REST = {"left": (0.14, 0.22, P.TOP + 0.26), "right": (0.14, -0.22, P.TOP + 0.26)}


def frame(a, close):
    """TCP rotation: approach a (local z), fingers closing along ``close`` (local x)."""
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - np.dot(c, a) * a
    c = c / np.linalg.norm(c)
    return np.stack([c, np.cross(a, c), a], axis=1)


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def _h(d):
    h = np.cross(Z, d)
    return h / np.linalg.norm(h)


def strike_pose(k, lift=0.0, press=0.0):
    """TCP and rotation that put the mallet's lower face on sector k's rim point (+lift above the lid's
    top, -press into it), head axis vertical, handle level heading toward the right arm (turned DYAW)."""
    a = math.radians(45.0 * k)
    s = np.array([P.CAN_C[0] + 0.068 * math.cos(a), P.CAN_C[1] + 0.068 * math.sin(a), P.FLOOR + rim.FREE_TOP])
    head = s + Z * (P.HEAD_L / 2 + lift - press)
    v = _bases()["right"][:2] - head[:2]
    th = math.atan2(v[1], v[0]) + DYAW
    d = np.array([math.cos(th), math.sin(th), 0.0])                     # head -> butt
    g = head + d * ALONG
    R = frame(-_h(d), Z)                                                 # approach across the handle, jaws above/below
    return g + _h(d) * PAD_TIP, R


def pick_pose(lift=0.0):
    """TCP over the lying mallet's handle grip point, approach down, fingers across the handle (the roll
    chosen so the hand's local y runs from the butt toward the head, as in the strike grip)."""
    d = P.mallet_dir()
    head = np.array([P.HEAD_C[0], P.HEAD_C[1], P.FLOOR + P.HEAD_R])
    droop = (P.HEAD_R - P.HANDLE_R) / P.HANDLE_L                          # the handle rests a little sloped
    g = head + d * ALONG - Z * (ALONG * droop)
    R = frame(-Z, -_h(d))
    return g + Z * (PAD_TIP + lift), R


def turned_pose(lift):
    """The mallet lifted over its pick spot and turned a quarter turn about the handle: faces up/down."""
    d = P.mallet_dir()
    g, _ = pick_pose(lift)
    g = g - Z * PAD_TIP
    return g + _h(d) * PAD_TIP, frame(-_h(d), Z)


def brace_pose(press=0.0):
    """Left hand: fingertips against the can's wall on its +y side, 90 mm up, approach in and down."""
    t = math.radians(30.0)
    a = np.array([0.0, -math.cos(t), -math.sin(t)])
    w = np.array([P.CAN_C[0], P.CAN_C[1] + P.R_CAN + 0.004 - press, P.FLOOR + 0.090])
    return w - a * PAD_TIP, frame(a, (1.0, 0.0, 0.0))


def plan_poses():
    """Every pose the demo commands (for reach checks)."""
    out = [("right", *pick_pose(0.08)), ("right", *pick_pose(0.0)), ("right", *turned_pose(0.14)),
           ("right", *strike_pose(ORDER[0], lift=0.12)),
           ("left", *brace_pose(0.0)), ("left", *brace_pose(0.003))]
    for k in ORDER:
        out += [("right", *strike_pose(k, lift=HOVER)), ("right", *strike_pose(k, press=PRESS))]
    return out


def demo():
    if SCENE_ONLY:
        return {side: [_w(1.5, REST[side], FINGER_OPEN, "look", DOWN),
                       _w(MAX_STEPS / HZ, REST[side], FINGER_OPEN, "hold", DOWN)] for side in ("left", "right")}
    right = [_w(1.5, REST["right"], FINGER_OPEN, "look", DOWN)]
    p, R = pick_pose(0.08)
    right += [_w(2.5, p, FINGER_OPEN, "over the mallet's handle", R)]
    p, R = pick_pose(0.0)
    right += [_w(1.5, p, FINGER_OPEN, "round the handle", R), _w(0.8, p, FINGER_CLOSED, "grip the handle", R)]
    p, R = pick_pose(0.10)
    right += [_w(1.5, p, FINGER_CLOSED, "lift the mallet", R)]
    p, R = turned_pose(0.14)
    right += [_w(2.0, p, FINGER_CLOSED, "turn it so the head's faces point down", R)]
    t_ready = 1.5 + 2.5 + 1.5 + 0.8 + 1.5 + 2.0 + 2.0
    p, R = strike_pose(ORDER[0], lift=0.12)
    right += [_w(2.0, p, FINGER_CLOSED, "up over the can", R)]
    for i, k in enumerate(ORDER):
        p, R = strike_pose(k, lift=HOVER)
        right += [_w(1.2 if i == 0 else 1.0, p, FINGER_CLOSED, f"over sector {k}", R)]
        p2, _ = strike_pose(k, press=PRESS)
        right += [_w(0.16, p2, FINGER_CLOSED, f"tap sector {k}", R), _w(0.24, p2, FINGER_CLOSED, "on the rim", R),
                  _w(0.35, p, FINGER_CLOSED, "lift off", R)]
    t_strikes = 1.2 + 7 * 1.0 + 8 * (0.25 + 0.15 + 0.35)
    p, R = strike_pose(ORDER[-1], lift=0.12)
    right += [_w(1.2, p, FINGER_CLOSED, "up off the lid", R)]
    p, R = turned_pose(0.14)
    right += [_w(2.5, p, FINGER_CLOSED, "back over the cloth", R)]
    p, R = pick_pose(0.10)
    right += [_w(2.0, p, FINGER_CLOSED, "turn it flat again", R)]
    p, R = pick_pose(0.004)
    right += [_w(1.5, p, FINGER_CLOSED, "lay it on the cloth", R), _w(0.8, p, FINGER_OPEN, "let go", R)]
    p, R = pick_pose(0.10)
    right += [_w(1.5, p, FINGER_OPEN, "hand up", R), _w(2.0, REST["right"], FINGER_OPEN, "rest", DOWN),
              _w(3.0, REST["right"], FINGER_OPEN, "hold", DOWN)]
    RL = frame(-Z, (1.,0.,0.))
    grip = rim.LUG + Z*PAD_TIP
    above = grip + Z*.065
    left = [_w(1.5, REST["left"], FINGER_OPEN, "ready", DOWN),
            _w(2.8, above, FINGER_OPEN, "above can tab", RL),
            _w(1.5, grip, FINGER_OPEN, "straddle tab", RL),
            _w(.8, grip, FINGER_CLOSED, "pinch can tab", RL),
            _w(t_ready+t_strikes+.9-6.6, grip, FINGER_CLOSED, "hold through taps", RL),
            _w(.8, grip, FINGER_OPEN, "release tab", RL),
            _w(1.2, above, FINGER_OPEN, "withdraw", RL),
            _w(2., REST["left"], FINGER_OPEN, "clear", DOWN), _w(12., REST["left"], FINGER_OPEN, "rest", DOWN)]
    return {"left": left, "right": right}
