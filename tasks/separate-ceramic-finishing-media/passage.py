"""Measure ceramic crossings in the moving screen's frame from rigid-collider vertices."""

import math
import numpy as np


def rotation(pose):
    x, y, z, w = pose[3:]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def end_rings(radius, half_length, slope, n=16):
    a = np.arange(n) * 2 * math.pi / n
    y, z = radius * np.cos(a), radius * np.sin(a)
    return np.vstack(
        [np.column_stack((sign * half_length + slope * z, y, z)) for sign in (-1, 1)]
    )


def slice_slab(points, edges, thickness):
    z = points[:, 2]
    slices = [points[(z <= 0) & (z >= -thickness), :2]]
    for plane in [0.0, -thickness]:
        start, end = points[edges[:, 0]], points[edges[:, 1]]
        crosses = (start[:, 2] < plane) != (end[:, 2] < plane)
        start, end = start[crosses], end[crosses]
        if len(start):
            fraction = ((plane - start[:, 2]) / (end[:, 2] - start[:, 2]))[:, None]
            slices.append(start[:, :2] + fraction * (end[:, :2] - start[:, :2]))
    return np.vstack(slices)


class Passage:
    def __init__(self, scene):
        self.scene = scene
        self.holes = np.asarray(scene.HOLES)
        self.media = end_rings(
            scene.MEDIA_DIAMETER / 2, scene.MEDIA_LENGTH / 2, math.tan(math.radians(15))
        )
        self.edges = np.array(
            [(j * 16 + i, j * 16 + (i + 1) % 16) for i in range(16) for j in range(2)]
            + [(i, i + 16) for i in range(16)]
        )
        a = (np.arange(20) + 0.5) * 2 * math.pi / 20
        self.normals = np.column_stack((np.cos(a), np.sin(a)))
        self.clearance = scene.APERTURE / 2 * math.cos(math.pi / 20)
        self.candidates = {}
        self.passed = {}
        self.invalid = {}
        self.previous = {}
        self.received = set()
        self.max_receiver_count = 0

    def advance(self, poses, step):
        s = self.scene
        screen = poses["screen"]
        basis = rotation(screen)
        rx = s.RECEIVER_INNER_XY / 2
        for name, body in poses.items():
            if not name.startswith("media_"):
                continue
            world = self.media @ rotation(body).T + body[:3]
            local = (world - screen[:3]) @ basis
            prev = self.previous.get(name)
            self.previous[name] = local
            key = self.candidates.get(name)
            if name not in self.passed and name not in self.invalid:
                # Midpoint slices catch a body traversing the whole 3 mm plate
                # between solver samples. At these speeds interpolation is local.
                path = [local] if prev is None else []
                if (
                    prev is not None
                    and prev[:, 2].min() >= -0.0005
                    and local[:, 2].min() >= -0.0005
                ):
                    key = None
                elif prev is not None:
                    distance = np.linalg.norm(local - prev, axis=1).max()
                    n = max(1, math.ceil(distance / 0.00025))
                    if n > 128:
                        self.invalid[name] = "unresolved displacement"
                        continue
                    path.extend(
                        prev + (local - prev) * (i / n) for i in range(1, n + 1)
                    )
                for points in path:
                    if points[:, 2].min() >= -0.0005:
                        key = None
                        continue
                    if points[:, 2].max() < -s.FLOOR_T:
                        if key is not None and key >= 0:
                            self.passed[name] = (step, key)
                        else:
                            self.invalid[name] = "crossing outside a single open hole"
                        break
                    # Test the solid deeper than the existing 0.5 mm contact tolerance.
                    # At a rounded rod on a rib, points touching the top face may
                    # project mm beyond the opening while remaining only microns deep.
                    tolerance = 0.0005
                    probe = points.copy()
                    probe[:, 2] += tolerance
                    cut = slice_slab(probe, self.edges, s.FLOOR_T - 2 * tolerance)
                    if not len(cut):
                        continue
                    i = int(
                        np.argmin(np.linalg.norm(self.holes - cut.mean(axis=0), axis=1))
                    )
                    inside = (
                        (cut - self.holes[i]) @ self.normals.T
                    ).max() <= self.clearance + 0.0005
                    if not inside or (key is not None and key != i):
                        key = -1
                    elif key is None:
                        key = i
                self.candidates[name] = key
            center = np.array([s.SCREEN_X, s.SCREEN_Y])
            z = world[:, 2]
            caught = (
                np.abs(world[:, :2] - center).max() <= rx + 0.0005
                and z.min() >= s.TABLETOP_HEIGHT + 0.001 + s.RECEIVER_FLOOR_T - 0.0005
                and z.max()
                <= s.TABLETOP_HEIGHT
                + 0.001
                + s.RECEIVER_FLOOR_T
                + s.RECEIVER_WALL_H
                + 0.0005
            )
            if caught and name in self.passed:
                self.received.add(name)
            elif name in self.received:
                self.invalid[name] = "exited receiver"
        self.max_receiver_count = max(self.max_receiver_count, len(self.received))
