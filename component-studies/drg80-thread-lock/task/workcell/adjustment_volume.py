"""Signed CAD separation of each complete tool from the finite adjustment cylinder."""

import coal
import numpy as np


class AdjustmentVolume:
    # Both datums are above the fixture origin. Specified offsets are 10 mm
    # below the 90 mm cover seat and 35 mm above the 101.7 mm screw mouth.
    lower = 0.090 - 0.010
    upper = 0.1017 + 0.035
    centre = np.array([0.0, -0.018, (lower + upper) / 2])
    cylinder = coal.Cylinder(0.045, upper - lower)

    def __init__(self, geometry):
        self.parts = []
        for vertices in geometry.convex_parts:
            v = coal.StdVec_Vec3s()
            for pt in vertices:
                v.append(pt)
            self.parts.append(
                (coal.Convex.convexHull(v, True, "Qt"), np.eye(3), np.zeros(3))
            )
        for centre, axial, radial in geometry.cylinders:
            r = np.linalg.norm(radial, axis=1)
            basis = np.column_stack(
                [radial[0] / r[0], radial[1] / r[1], axial / np.linalg.norm(axial)]
            )
            if not np.isclose(r[0], r[1]) or not np.allclose(
                basis.T @ basis, np.eye(3)
            ):
                raise ValueError("expected the authored circular tool shaft")
            if np.linalg.det(basis) < 0:
                basis[:, 0] *= -1
            self.parts.append(
                (coal.Cylinder(float(r[0]), 2 * np.linalg.norm(axial)), basis, centre)
            )
        self.request = coal.DistanceRequest()
        self.request.enable_signed_distance = True

    def distance(self, xyz, rot, frame_xyz, frame_rot):
        volume_pose = coal.Transform3s(frame_rot, frame_xyz + frame_rot @ self.centre)
        d = float("inf")
        for collider, axes, offset in self.parts:
            pose = coal.Transform3s(rot @ axes, xyz + rot @ offset)
            value = coal.distance(
                collider,
                pose,
                self.cylinder,
                volume_pose,
                self.request,
                coal.DistanceResult(),
            )
            if not np.isfinite(value):
                raise ValueError("nonfinite tool-to-adjustment-volume distance")
            d = min(d, float(value))
        return d
