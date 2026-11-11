"""Offline triangle-level self-intersection check of saved surface snapshots.

This geometric check does not prove the absence of near intersections,
rigid-object penetration, or bad contact forces.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def _edge_hits_triangle(start, end, triangle, epsilon=1e-9):
    """Strict segment/triangle intersection; excludes boundary-only touches."""
    a, b, c = triangle
    direction, e1, e2 = end - start, b - a, c - a
    h = np.cross(direction, e2)
    det = float(np.dot(e1, h))
    if abs(det) < 1e-18:
        return False
    inv = 1.0 / det
    s = start - a
    u = inv * np.dot(s, h)
    q = np.cross(s, e1)
    v = inv * np.dot(direction, q)
    t = inv * np.dot(e2, q)
    return epsilon < u and epsilon < v and u + v < 1 - epsilon and epsilon < t < 1 - epsilon


def _coplanar_overlap(a, b, normal):
    """True for positive-area intersection of coplanar triangles, not edge contact."""
    axis = int(np.argmax(np.abs(normal)))
    left, right = np.delete(a, axis, axis=1), np.delete(b, axis, axis=1)
    def cross2(u, v):
        return u[0] * v[1] - u[1] * v[0]
    orientation = np.sign(cross2(right[1] - right[0], right[2] - right[0]))
    polygon = [point.copy() for point in left]
    for i in range(3):
        start, end = right[i], right[(i + 1) % 3]
        if not polygon:
            return False
        clipped = []
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            cp, cq = orientation * cross2(end - start, p - start), orientation * cross2(end - start, q - start)
            if cp >= 0:
                clipped.append(p)
            if (cp < 0 < cq) or (cq < 0 < cp):
                clipped.append(p + (q - p) * (cp / (cp - cq)))
        polygon = clipped
    if len(polygon) < 3:
        return False
    area = abs(sum(cross2(p, q) for p, q in zip(polygon, polygon[1:] + polygon[:1]))) * 0.5
    return area > 1e-12


def strict_crossing(a, b):
    normal_a = np.cross(a[1] - a[0], a[2] - a[0])
    normal_b = np.cross(b[1] - b[0], b[2] - b[0])
    n_a, n_b = np.linalg.norm(normal_a), np.linalg.norm(normal_b)
    if min(n_a, n_b) < 1e-16:
        return False
    parallel = np.linalg.norm(np.cross(normal_a, normal_b)) <= 1e-7 * n_a * n_b
    if parallel:
        if np.max(np.abs((b - a[0]) @ normal_a)) / n_a > 1e-9:
            return False
        return _coplanar_overlap(a, b, normal_a)
    return any(_edge_hits_triangle(a[i], a[(i + 1) % 3], b) or
               _edge_hits_triangle(b[i], b[(i + 1) % 3], a) for i in range(3))


def main(path):
    data = np.load(path)
    faces = data["faces"]
    report = {}
    for key in sorted((key for key in data.files if key.startswith("nodes_")),
                      key=lambda key: int(key.split("_")[1])):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(data[key].astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        candidates = np.asarray(mesh.get_self_intersecting_triangles())
        crossings = int(sum(strict_crossing(data[key][faces[a]], data[key][faces[b]]) for a, b in candidates))
        report[key] = {"open3d_candidate_pairs": len(candidates), "strict_crossing_pairs": crossings}
        print(key, "candidates", len(candidates), "strict_crossings", crossings, flush=True)
    path.with_name("surface-intersections.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    main(parser.parse_args().snapshot)
