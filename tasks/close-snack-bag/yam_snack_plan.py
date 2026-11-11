"""Joint-target-only experimental bimanual plan for the snack task.

The plan can fail. It never writes a packet or clip pose. It assumes an upright
clip whose handles are near (0.207, -0.14, 0.05), in the probe world frame.
"""
import numpy as np
from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.demo import grip_rotation

HZ = 20


class ArmPlan:
    def __init__(self, side, base, start):
        self.side, self.base = side, np.asarray(base)
        self.rot = grip_rotation(0)
        self.q, error, _ = ik.solve(start, ik.matrix_to_quat_xyzw(self.rot), READY, base_pos=self.base, iters=300)
        if error > .003:
            raise RuntimeError(f'{side} starting hover is unreachable: {error:.4f} m')
        self.pos = np.asarray(start, float)
        self.grip = 1.0
        self.worst_ik_error = error

    def move(self, target, rotation, seconds, grip=None):
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        for i in range(1, max(1, round(seconds * HZ)) + 1):
            f = ik.smoothstep(i / max(1, round(seconds * HZ)))
            self.pos = p0 + f * (np.asarray(target) - p0)
            self.rot = ik.slerp_matrix(r0, rotation, f)
            self.grip = g0 + f * (g1-g0)
            self.q, error, _ = ik.solve(self.pos, ik.matrix_to_quat_xyzw(self.rot), self.q,
                                         base_pos=self.base, iters=80)
            self.worst_ik_error = max(self.worst_ik_error, error)
            yield


def together(*generators):
    active=list(map(iter, generators))
    while active:
        for g in active[:]:
            try: next(g)
            except StopIteration: active.remove(g)
        if active: yield


def wait(seconds):
    for _ in range(round(HZ * seconds)): yield


def plan(left, right, clip_pose):
    down = grip_rotation(0)
    # Pre-positioned at reset; left presses while right approaches the clip.
    yield from together(left.move([.145,.067,.025],down,.5,0),
                        right.move([.207,-.14,.065],down,.8,1))
    yield from right.move(right.pos,down,1.0,0)
    yield from wait(.3)
    # Store grasp transform only for planning; no simulation state is written.
    position, rotation = clip_pose()
    grasp_offset = rotation.T @ (right.pos-position)
    relative_rot = rotation.T @ right.rot
    yield from right.move(right.pos+[0,0,.07],down,1.2,0)
    desired_clip_rot = np.diag([-1.,-1.,1.])
    desired_grip_rot = desired_clip_rot @ relative_rot
    yield from right.move(right.pos,desired_grip_rot,1.5,0)
    target = np.array([.17,0,.012])
    staging = target + [.05,0,.02]
    yield from right.move(staging+desired_clip_rot@grasp_offset,desired_grip_rot,1.6,0)
    yield from right.move(target+desired_clip_rot@grasp_offset,desired_grip_rot,1.4,0)
    yield from right.move(right.pos,desired_grip_rot,.8,1)
    yield from right.move(right.pos+[.06,0,.06],desired_grip_rot,1.2,1)
    yield from left.move(left.pos+[0,0,.1],down,1.0,1)
    yield from wait(3.2)
