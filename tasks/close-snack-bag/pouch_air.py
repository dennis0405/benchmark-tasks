"""Uncalibrated lumped isothermal gas model for an open-mouth pouch.

The NumPy functions are a reference for the experimental Newton GPU callback.
The virtual cap closes the mouth for volume calculation. The GPU callback adds
virtual-cap traction only in an imposed sealed control. This model does not
resolve local airflow or film permeability.
"""
import math
import numpy as np

R = 8.314462618
M_AIR = 0.02897


def mouth_cap(grid):
    """Triangulate the open mouth; orientation points outward along +x."""
    faces = []
    for j in range(grid.shape[2] - 1):
        a, b = grid[0, -1, j], grid[0, -1, j + 1]
        c, d = grid[1, -1, j + 1], grid[1, -1, j]
        for face in ((a, b, c), (a, c, d)):
            if len(set(face)) == 3:
                faces.append(face)
    return np.array(faces, dtype=np.int32)


def enclosed_volume(points, faces, cap):
    triangles = points[np.vstack((faces, cap))].astype(np.float64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    volume = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError('cavity volume must be positive and finite')
    return volume


def mouth_area(points, cap):
    triangles = points[cap].astype(np.float64)
    return float((np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1) / 2).sum())


def mouth_channel_area(points, grid, first_column, contact_gap):
    """Approximate the narrowest paired-skin opening through the mouth.

    This is a geometric leak proxy: it does not resolve folds or microchannels.
    """
    if not 0 <= first_column < grid.shape[1] or contact_gap < 0:
        raise ValueError("invalid mouth range or contact gap")
    widths = []
    for i in range(first_column, grid.shape[1]):
        lower, upper = points[grid[0, i]], points[grid[1, i]]
        center = (lower + upper) / 2
        span = np.linalg.norm(np.diff(center, axis=0), axis=1)
        gap = np.maximum(np.linalg.norm(upper-lower, axis=1) - contact_gap, 0)
        widths.append(float(np.sum(span * (gap[1:] + gap[:-1]) / 2)))
    return min(widths)


def film_pressure_forces(points, faces, gauge_pressure):
    triangles = points[faces].astype(np.float64)
    f = gauge_pressure * np.cross(triangles[:, 1] - triangles[:, 0],
                                  triangles[:, 2] - triangles[:, 0]) / 6
    out = np.zeros_like(points, dtype=np.float64)
    for i in range(3):
        np.add.at(out, faces[:, i], f)
    return out


def flow_step(moles, volume, area, dt, temperature=293.15, ambient=101325., discharge=0.6):
    """Vent toward ambient through an orifice, bounded to avoid overshoot.

    Area is effective opening area, not a fitted permeability; incompressible
    orifice flow is a low-pressure approximation, not CFD or acoustic air flow.
    """
    if min(moles, area, dt) < 0 or min(volume, temperature, ambient) <= 0 or not 0 <= discharge <= 1:
        raise ValueError('invalid gas state or parameters')
    pressure = moles * R * temperature / volume
    difference = pressure - ambient
    rho = max(pressure if difference >= 0 else ambient, 1e-9) * M_AIR / (R * temperature)
    mole_flux = discharge * area * math.sqrt(2 * rho * abs(difference)) / M_AIR
    equilibrium = ambient * volume / (R * temperature)
    return moles - math.copysign(min(mole_flux * dt, abs(moles - equilibrium)), difference) if difference else moles
