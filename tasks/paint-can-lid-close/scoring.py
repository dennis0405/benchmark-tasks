"""Stateful scorer for "Close the paint can." From the can's, lid's and mallet's measured poses and
velocities, the robot's finger pads (oriented boxes from the measured joints and jaw opening by the URDF's
forward kinematics, task.hand_points, reported as mechanics) and the seated sectors task.Dynamics reports
from measured mallet strikes."""
from __future__ import annotations

import math

import numpy as np

import paint as P

HZ = 20
HOLD_S = 3.0
SEATED_MM = 1.0                       # the lid within this of seated (flush in the ring)
SHOVE_MM = 15.0                       # the can pushed further than this over the cloth ...
TILT_DEG = 8.0                        # ... or tipped further than this: struck unsteadied (caps 0.5)
CLEAR = 0.05                          # finger pads at least this far from the mallet, the can and the lid
REST_V = 0.02                         # m/s: the mallet at rest ...
REST_W = 0.20                         # rad/s: ... not rolling
DOWN_Z = 0.06                         # the mallet's head centre this low over the cloth: laid down


def _v(o, key):
    v = o.get(key, (0, 0, 0)) if isinstance(o, dict) else getattr(o, key, (0, 0, 0))
    return float(np.linalg.norm(np.asarray(v, float)))


def _box_pts(box):
    c, R, h = (np.asarray(x, float) for x in box)
    pts = [c]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(c + R @ (h * np.array([sx, sy, sz])))
    return np.asarray(pts)


def _seg_dist(pts, a, b):
    ab = b - a
    t = np.clip(((pts - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.unsteady = False
        self.fell = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("lid seating measurements missing")
        self.step = step
        can = np.asarray(root_poses["can"], float)
        lid = np.asarray(root_poses["rim0"], float)
        mal = np.asarray(root_poses["mallet"], float)
        moved = math.hypot(can[0] - P.CAN_C[0], can[1] - P.CAN_C[1])
        up = P.mat(can) @ np.array([0.0, 0.0, 1.0])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        if moved > SHOVE_MM / 1000 or tilt > TILT_DEG:
            self.unsteady = True
        if mal[2] < P.TOP - 0.05 or tilt > 45.0:
            self.fell = True
        n = int(mechanics["seated_n"])
        proud = float(mechanics["lid_proud_mm"])
        seated = n == 8 and proud <= SEATED_MM
        # finger pads against the mallet (handle segment and head), the can and the lid
        pts = np.vstack([_box_pts(b) for b in mechanics["pad_boxes"]])
        R = P.mat(mal)
        butt = mal[:3] + R @ np.array([P.HANDLE_L, 0.0, 0.0])
        ax = R @ np.array([0.0, 1.0, 0.0])
        d_handle = _seg_dist(pts, mal[:3], butt) - P.HANDLE_R
        d_head = _seg_dist(pts, mal[:3] - ax * P.HEAD_L / 2, mal[:3] + ax * P.HEAD_L / 2) - P.HEAD_R
        rel = pts[:, :2] - can[:2]
        d_side = np.linalg.norm(rel, axis=1) - P.R_CAN
        d_top = pts[:, 2] - (lid[2] + P.LID_T / 2)
        d_can = np.maximum(d_side, d_top)
        gap = float(min(d_handle.min(), d_head.min(), d_can.min()))
        open_hands = all(min(abs(x) for x in robot[s]["fingers"]) > .015 for s in ("left","right"))
        clear = gap >= CLEAR and open_hands and not mechanics["left_contacts"] and not mechanics["right_contacts"]
        o = objects.get("mallet", {}) if isinstance(objects, dict) else {}
        rest = _v(o, "lin_vel") < REST_V and _v(o, "ang_vel") < REST_W
        laid = mal[2] - P.FLOOR < DOWN_Z and mechanics["cloth_support"]
        score = 0.0
        if n >= 3:
            score = 0.25
        if n >= 6:
            score = 0.5
        if seated:
            score = 0.75
        done = seated and laid and rest and clear
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.unsteady:
            score = min(score, 0.5)
        if self.fell:
            score = 0.0
        return {"score": score, "success": score == 1.0, "sectors_seated": n, "lid_proud_mm": round(proud, 2),
                "lid_seated": bool(seated), "can_moved_mm": round(1000 * moved, 1), "can_tilt_deg": round(tilt, 2),
                "unsteady": self.unsteady, "mallet_laid": bool(laid), "mallet_at_rest": bool(rest),
                "pad_gap_mm": round(1000 * gap, 1), "hands_clear": bool(clear), "held_frames": self.held,
                "clear": bool(clear), "dropped": bool(self.fell)}
