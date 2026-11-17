"""Resolve top-surface load in each food body's local coordinates."""

import numpy as np
from scipy.spatial.transform import Rotation

RIM_COUNT = 16
CORE = RIM_COUNT
SUPPORT_N = 0.015


def top_surface_load(points, normals, magnitudes, body_ids, core, rims):
    """Normal force on the crust top, including its upper chamfer.

    Rim geometry is expressed relative to a hinge 8 mm above the pizza
    reference. Its upper chamfer starts at +4 mm. The core has a flat
    top at +15 mm. The 0.5 mm skin allowance covers both contact offsets.
    The underside pads are internal support surfaces, below the crust.
    """
    load = 0.0
    for selected, poses, height in (
        (body_ids < CORE, rims, 0.004),
        (body_ids == CORE, core[None, :], 0.015),
    ):
        if not selected.any():
            continue
        indices = (
            body_ids[selected]
            if len(poses) > 1
            else np.zeros(selected.sum(), dtype=int)
        )
        body = poses[indices]
        up = Rotation.from_quat(body[:, 3:]).apply([0.0, 0.0, 1.0])
        z = np.sum((points[selected] - body[:, :3]) * up, axis=1)
        vertical = np.abs(np.sum(normals[selected] * up, axis=1))
        # Upper bevel is sloped; a vertical side face has no upward component.
        top = (z >= height - 0.0005) & (vertical >= 0.35)
        load += float(magnitudes[selected][top].sum())
    return load


class FoodContact:
    def __init__(self, physics_view, tool_path, core_path, dt):
        self.dt = dt
        rim_paths = [f"/World/Food/rim{i:02}" for i in range(RIM_COUNT)]
        pad_paths = [f"/World/Food/pad{i:02}" for i in range(12)]
        self.rims = physics_view.create_rigid_body_view("/World/Food/rim*")
        paths = list(self.rims.prim_paths)
        self.rim_rows = [paths.index(p) for p in rim_paths]
        filters = rim_paths + [core_path] + pad_paths
        self.nfilters = len(filters)
        self.contacts = physics_view.create_rigid_contact_view(
            [tool_path], filter_patterns=[filters], max_contact_data_count=1024
        )

    def loads(self, core):
        force, point, normal, _, counts, starts = self.contacts.get_contact_data(
            dt=self.dt
        )
        counts = counts.numpy().reshape(-1)
        pairs = np.flatnonzero(counts)
        if not len(pairs):
            return 0.0, 0.0
        starts = starts.numpy().reshape(-1)
        rows = np.concatenate(
            [np.arange(starts[i], starts[i] + counts[i]) for i in pairs]
        )
        indices = np.repeat(pairs % self.nfilters, counts[pairs])
        poses = self.rims.get_transforms().numpy()[self.rim_rows]
        forces = np.abs(force.numpy().reshape(-1)[rows])
        top = top_surface_load(
            point.numpy()[rows], normal.numpy()[rows], forces, indices, core, poses
        )
        return float(forces.sum()) - top, top
