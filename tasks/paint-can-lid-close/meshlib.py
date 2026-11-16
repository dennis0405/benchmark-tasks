"""Small NumPy mesh builders for the task's visual meshes: surfaces of revolution, lofted rounded
sections, swept tubes and rounded boxes. Everything returns (points (N, 3), triangles (M, 3))."""
from __future__ import annotations

import math

import numpy as np


def merge(*meshes):
    pts, tris, n = [], [], 0
    for p, t in meshes:
        p = np.asarray(p, float)
        t = np.asarray(t, int)
        pts.append(p)
        tris.append(t + n)
        n += len(p)
    return np.vstack(pts), np.vstack(tris)


def transform(mesh, R=None, offset=(0.0, 0.0, 0.0)):
    p, t = mesh
    p = np.asarray(p, float)
    if R is not None:
        p = p @ np.asarray(R, float).T
    return p + np.asarray(offset, float), t


def _grid(rings, closed=True, cap0=False, cap1=False):
    """Triangles joining consecutive rings of equal length (each ring a closed loop if ``closed``)."""
    rings = [np.asarray(r, float) for r in rings]
    m = len(rings[0])
    pts = np.vstack(rings)
    tris = []
    segs = m if closed else m - 1
    for i in range(len(rings) - 1):
        a, b = i * m, (i + 1) * m
        for j in range(segs):
            j2 = (j + 1) % m
            tris += [(a + j, b + j, b + j2), (a + j, b + j2, a + j2)]
    extra = []
    if cap0:
        c = len(pts) + len(extra)
        extra.append(rings[0].mean(axis=0))
        tris += [(c, (j + 1) % m, j) for j in range(m)]
    if cap1:
        c = len(pts) + len(extra)
        extra.append(rings[-1].mean(axis=0))
        base = (len(rings) - 1) * m
        tris += [(c, base + j, base + (j + 1) % m) for j in range(m)]
    if extra:
        pts = np.vstack([pts, np.asarray(extra)])
    return pts, np.asarray(tris, int)


def revolve(profile, n=48, axis="Z", cap0=False, cap1=False, a0=0.0, a1=2 * math.pi):
    """Surface of revolution of ``profile`` [(radius, height), ...] about ``axis`` through the origin."""
    full = abs((a1 - a0) - 2 * math.pi) < 1e-9
    angles = np.linspace(a0, a1, n, endpoint=not full)
    rings = []
    for r, h in profile:
        ring = np.stack([r * np.cos(angles), r * np.sin(angles), np.full_like(angles, h)], axis=1)
        rings.append(ring)
    p, t = _grid(rings, closed=full, cap0=cap0, cap1=cap1)
    return _axis(p, axis), t


def _axis(p, axis):
    p = np.asarray(p, float)
    if axis == "Z":
        return p
    if axis == "X":
        return p[:, [2, 0, 1]]
    if axis == "Y":
        return p[:, [1, 2, 0]]
    raise ValueError(axis)


def superellipse(w, h, e=4.0, n=48):
    """A rounded rectangle (superellipse of exponent ``e``) of width w and height h, centred."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = (w / 2) * np.sign(c) * np.abs(c) ** (2.0 / e)
    y = (h / 2) * np.sign(s) * np.abs(s) ** (2.0 / e)
    return np.stack([x, y], axis=1)


def loft(sections, axis="Y", n=48, cap0=True, cap1=True):
    """Loft through rounded sections [(station, width, height, exponent, du, dv), ...] along ``axis``;
    each section is a superellipse centred at (du, dv) in the plane across the axis."""
    rings = []
    for sec in sections:
        st, w, h, e = sec[:4]
        du, dv = (sec[4], sec[5]) if len(sec) > 5 else (0.0, 0.0)
        xy = superellipse(w, h, e, n)
        ring = np.stack([xy[:, 0] + du, xy[:, 1] + dv, np.full(n, st)], axis=1)
        rings.append(ring)
    p, t = _grid(rings, closed=True, cap0=cap0, cap1=cap1)
    if axis == "Z":
        return p, t
    if axis == "Y":                      # section (u, v) -> (x, z), station -> y
        return p[:, [0, 2, 1]], t
    if axis == "X":                      # section (u, v) -> (y, z), station -> x
        return p[:, [2, 0, 1]], t
    raise ValueError(axis)


def rbox(size, e=6.0, n=32, axis="Z"):
    """A rounded box: ``size`` (x, y, z), its edges round about ``axis``."""
    sx, sy, sz = size
    if axis == "Z":
        return loft([(-sz / 2, sx, sy, e), (sz / 2, sx, sy, e)], axis="Z", n=n)
    if axis == "Y":
        return loft([(-sy / 2, sx, sz, e), (sy / 2, sx, sz, e)], axis="Y", n=n)
    return loft([(-sx / 2, sy, sz, e), (sx / 2, sy, sz, e)], axis="X", n=n)


def tube(path, r, n=10):
    """A tube of radius r swept along the polyline ``path`` (smoothed corners by the caller)."""
    P = np.asarray(path, float)
    rings = []
    prev_u = None
    for i in range(len(P)):
        d = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        d /= np.linalg.norm(d)
        u = np.cross(d, [0.0, 0.0, 1.0]) if prev_u is None else prev_u - np.dot(prev_u, d) * d
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(d, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        prev_u = u
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rings.append(P[i] + r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)))
    return _grid(rings, closed=True, cap0=True, cap1=True)


def bend(points, radius=0.01, steps=6):
    """Round a polyline's corners with arcs of ``radius`` (for wire leads)."""
    P = [np.asarray(p, float) for p in points]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        u = (a - b) / np.linalg.norm(a - b)
        w = (c - b) / np.linalg.norm(c - b)
        r = min(radius, 0.45 * np.linalg.norm(a - b), 0.45 * np.linalg.norm(c - b))
        p0, p1 = b + u * r, b + w * r
        for k in range(steps + 1):
            s = k / steps
            out.append((1 - s) ** 2 * p0 + 2 * (1 - s) * s * b + s ** 2 * p1)
    out.append(P[-1])
    return np.asarray(out)


def fillet_profile(r0, r1, h0, h1, f=0.002, steps=4):
    """A disc profile (r, h): from the axis at h0 out to r1 with filleted rim, back to r0 at h1."""
    prof = [(r0, h0)]
    for k in range(steps + 1):
        a = -math.pi / 2 + (math.pi / 2) * k / steps
        prof.append((r1 - f + f * math.cos(a), h0 + f + f * math.sin(a)))
    for k in range(steps + 1):
        a = (math.pi / 2) * k / steps
        prof.append((r1 - f + f * math.cos(a), h1 - f + f * math.sin(a)))
    prof.append((r0, h1))
    return prof


def heightfield(sx, sy, zfun, z0, nx=48, ny=36):
    """A slab whose top is z = zfun(x, y) over [-sx/2, sx/2] x [-sy/2, sy/2] and whose bottom is flat at z0."""
    xs = np.linspace(-sx / 2, sx / 2, nx)
    ys = np.linspace(-sy / 2, sy / 2, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = zfun(X, Y)
    top = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    bot = np.stack([X, Y, np.full_like(X, z0)], axis=-1).reshape(-1, 3)
    pts = np.vstack([top, bot])
    nb = nx * ny
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            tris += [(a, b, c), (a, c, d), (nb + a, nb + c, nb + b), (nb + a, nb + d, nb + c)]
    edge = [(i, 0) for i in range(nx)] + [(nx - 1, j) for j in range(1, ny)] + \
           [(i, ny - 1) for i in range(nx - 2, -1, -1)] + [(0, j) for j in range(ny - 2, 0, -1)]
    idx = [i * ny + j for i, j in edge]
    for k in range(len(idx)):
        a, b = idx[k], idx[(k + 1) % len(idx)]
        tris += [(a, nb + a, nb + b), (a, nb + b, b)]
    return pts, np.asarray(tris, int)
