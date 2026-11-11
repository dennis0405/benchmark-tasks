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
    calibration. Keep Newton's unwrapped angle convention; avoid branch-cut cases.
    """
    if not 0 <= yield_angle < np.pi:
        raise ValueError("yield angle must be in [0, pi)")
    delta = angles - rest_angles
    if np.any(np.abs(delta[eligible]) > np.pi):
        raise ValueError("angle branch crossing requires a consistent unwrapping model")
    amount = np.maximum(np.abs(delta) - yield_angle, 0.0)
    return rest_angles + np.where(eligible, np.sign(delta) * amount, 0.0)
