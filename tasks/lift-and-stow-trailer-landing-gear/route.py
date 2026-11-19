"""Extend the low sweep to 55 degrees, then shift the sleeve at fixed crank angle."""
import numpy as np
from yam_arena import ik
from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS


def translate_sleeve(row, slide, phase, axis):
    delta = (row['s'] - slide) * axis
    joints = []
    for q, base in zip(row['q'], (LEFT_BASE_POS, RIGHT_BASE_POS)):
        p, rot = ik.tcp_pose(q, base)
        solved, err, _ = ik.solve(p+delta, ik.matrix_to_quat_xyzw(rot), q, base_pos=base,
                                  iters=70, damping=.015)
        if err > .001:
            raise ValueError(f'Sleeve waypoint unavailable, {err*1000:.2f} mm residual')
        joints.append(solved.tolist())
    return dict(row, q=joints, s=slide, phase=phase)


def grounded_route(rows, axis, early_shift=False):
    lead = [r for r in rows if r['phase'] in ('approach','grasp')]
    sweep = [r for r in rows if r['phase'] in ('low','high')]
    shift = [r for r in rows if r['phase'] == 'shift']
    if early_shift:
        # Identical device and score, but push the same sleeve directly after grasp.
        return lead + [dict(translate_sleeve(lead[-1], r['s'], 'shift', axis), duration=r['duration']) for r in shift]
    low = [translate_sleeve(r, 0., 'low', axis) for r in sweep if r['a'] <= 55.]
    pivot = next(r for r in sweep if r['a'] == 55.)
    transition = [dict(translate_sleeve(pivot, r['s'], 'shift', axis), duration=r['duration']) for r in shift]
    return lead + low + transition + [r for r in rows if r['phase']=='high' and r['a']>55.] + [
        r for r in rows if r['phase'] not in ('approach','grasp','low','shift','high')]
