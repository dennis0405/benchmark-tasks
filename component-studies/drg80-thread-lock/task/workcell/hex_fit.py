"""Clearance of the authored hex bit inside the six socket planes, in metres."""

import numpy as np

# Across flats from HexTip and SocketWall00..23 in the first-party USDs.
angles = np.arange(6) * np.pi / 3
radius = 0.001975 / np.cos(np.pi / 6)
rim = radius * np.column_stack((np.cos(angles), np.sin(angles)))
flat_normals = np.column_stack((np.cos(angles + np.pi / 6), np.sin(angles + np.pi / 6)))
bit_vertices = np.vstack([np.column_stack((rim, np.full(6, z))) for z in (0.0, 0.018)])
edges = [(l + i, l + (i + 1) % 6) for l in (0, 6) for i in range(6)] + [
    (i, i + 6) for i in range(6)
]


def hex_fit(bit_pos, bit_rot, screw_pos, screw_rot):
    """Clip the real prism to the 6 mm deep socket before comparing face planes."""
    local = (
        bit_vertices @ bit_rot.T + np.asarray(bit_pos) - np.asarray(screw_pos)
    ) @ screw_rot
    inside = (local[:, 2] >= -0.006) & (local[:, 2] <= 0)
    section = list(local[inside])
    for a, b in edges:
        v, w = local[a], local[b]
        for plane in (-0.006, 0.0):
            if (v[2] - plane) * (w[2] - plane) < 0.0:
                section.append(v + (w - v) * ((plane - v[2]) / (w[2] - v[2])))
    depth = -float(local[:6, 2].max())
    bottom = float(local[:, 2].min() + 0.006)
    side = (
        None
        if not section
        else 0.002075 - float(np.max(np.asarray(section)[:, :2] @ flat_normals.T))
    )
    return dict(depth_m=depth, wall_clearance_m=side, floor_clearance_m=bottom)
