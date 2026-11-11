"""Experimental rate-independent hinge plasticity; parameters are uncalibrated."""
import numpy as np


def hinge_angles(points, edges):
    """Return Newton VBD signed dihedral angles and nondegenerate-edge mask."""
    edges = np.asarray(edges)
    valid = np.all(edges >= 0, axis=1)
    safe = np.maximum(edges, 0)
    x0, x1, x2, x3 = (points[safe[:, i]] for i in range(4))
    n1 = np.cross(x2 - x0, x3 - x0)
    n2 = np.cross(x3 - x1, x2 - x1)
    direction = x3 - x2
    norm1, norm2, norm_e = (np.linalg.norm(v, axis=1) for v in (n1, n2, direction))
    valid &= (norm1 >= 1e-6) & (norm2 >= 1e-6) & (norm_e >= 1e-6)
    denom = np.maximum(norm1 * norm2, 1e-30)
    cosine = np.sum(n1 * n2, axis=1) / denom
    sine = np.sum(np.cross(n1, n2) * direction, axis=1) / np.maximum(denom * norm_e, 1e-30)
    return np.arctan2(sine, cosine), valid


def yielded_rest_angles(rest_angles, angles, eligible, yield_angle):
    """Update permanent rest angle only beyond yield; no healing below yield.

    This simple perfect-plasticity rule has no time dependence, hardening, or
    calibration. Align the rest angle to the solver angle's current 2*pi branch
    so Newton's unwrapped (theta - rest) remains a short angular difference.
    """
    if not 0 <= yield_angle < np.pi:
        raise ValueError("yield angle must be in [0, pi)")
    delta = (angles - rest_angles + np.pi) % (2 * np.pi) - np.pi
    aligned_rest = angles - delta
    amount = np.maximum(np.abs(delta) - yield_angle, 0.0)
    candidate = aligned_rest + np.sign(delta) * amount
    return np.where(eligible, candidate, rest_angles)
