"""Diagnostic vertical rays from content nodes through the actual pouch triangles.

Only valid for sample points away from the open mouth. This checks nodes, not
whole tetrahedra or continuous time, and ray parity can be ambiguous at edges.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def ray_diagnostic(points, faces, samples):
    triangles = points[faces].astype(np.float64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    usable = np.abs(det) > 1e-14
    heights = []
    for x, y, z in samples:
        dx, dy = x - a[:, 0], y - a[:, 1]
        u = np.divide(dx * ac[:, 1] - dy * ac[:, 0], det, out=np.zeros_like(det), where=usable)
        v = np.divide(ab[:, 0] * dy - ab[:, 1] * dx, det, out=np.zeros_like(det), where=usable)
        mask = usable & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        raw = np.sort((a[:, 2] + u * ab[:, 2] + v * ac[:, 2])[mask])
        # Triangles sharing an edge may report the same height twice.
        crossing = raw[np.r_[True, np.diff(raw) > 1e-7]] if len(raw) else raw
        if len(crossing) != 2:
            heights.append({"status": "ambiguous", "crossings": len(crossing)})
        else:
            gap = min(float(z - crossing[0]), float(crossing[1] - z))
            heights.append({"status": "inside" if gap >= 0 else "outside", "clearance_m": gap})
    return heights


def main(path):
    data = np.load(path)
    report = {}
    for key in sorted((k for k in data.files if k.startswith("nodes_")), key=lambda k: int(k.split("_")[1])):
        entries = ray_diagnostic(data[key], data["faces"], data[key.replace("nodes_", "contents_")])
        clearances = [e["clearance_m"] for e in entries if "clearance_m" in e]
        row = {"inside_nodes": sum(e["status"] == "inside" for e in entries),
               "outside_nodes": sum(e["status"] == "outside" for e in entries),
               "ambiguous_nodes": sum(e["status"] == "ambiguous" for e in entries),
               "min_vertical_clearance_m": min(clearances) if clearances else None}
        report[key] = row
        print(key, json.dumps(row), flush=True)
    path.with_name("content-node-clearance.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    main(parser.parse_args().snapshot)
