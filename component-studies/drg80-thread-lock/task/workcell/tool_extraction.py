"""Sufficient geometric tests for the 35 mm straight-handle extraction corridor."""

import numpy as np


def ray_hits_boxes(start, direction, lo, hi):
    # An expanded AABB contains every point within the specified radius of
    # its collider. Missing every box therefore certifies a clear corridor.
    t0 = np.zeros(len(lo))
    t1 = np.full(len(lo), np.inf)
    possible = np.ones(len(lo), dtype=bool)
    for i, d in enumerate(direction):
        if abs(d) < 1e-12:
            possible &= (start[i] >= lo[:, i]) & (start[i] <= hi[:, i])
        else:
            a = (lo[:, i] - start[i]) / d
            b = (hi[:, i] - start[i]) / d
            t0 = np.maximum(t0, np.minimum(a, b))
            t1 = np.minimum(t1, np.maximum(a, b))
    return bool(np.any(possible & (t1 >= t0)))


def certify_extraction(centroid, handle, axis, lo, hi):
    if np.any(centroid < lo.min(axis=0)) or np.any(centroid > hi.max(axis=0)):
        return "centroid_outside_fixture_envelope"
    radius = 0.0175
    midpoint = (handle.min(axis=0) + handle.max(axis=0)) / 2
    proj = (handle - midpoint) @ axis
    perp = handle - midpoint - proj[:, None] * axis
    if np.max(np.linalg.norm(perp, axis=1)) > radius:
        return None
    # Include the whole handle, including the trailing cap at the start.
    for direction, start in (
        (axis, midpoint + proj.min() * axis),
        (-axis, midpoint + proj.max() * axis),
    ):
        if not ray_hits_boxes(start, direction, lo - radius, hi + radius):
            return "straight_handle_corridor"
    return None
