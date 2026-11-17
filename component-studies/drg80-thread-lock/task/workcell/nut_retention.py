"""Measure axial guide engagement from native nut and screw poses."""

import numpy as np
from scipy.spatial.transform import Rotation

# SmoothThreadProxy runs from −24 to −6 mm; the contiguous outer surfaces
# of the 24 socket wall sectors continue the cylindrical guide up to z=0.
GUIDE = (-0.024, 0.0)
NUT_HALF_HEIGHT = 0.002
INSTALLED_RETAINER = np.array([0.0, 0.0, -0.002])


def nut_retention(screw, nut, retainer):
    inverse = Rotation.from_quat(screw[3:]).as_matrix().T
    overlap = []
    radial = []
    for obj in (nut, retainer):
        center = inverse @ (obj[:3] - screw[:3])
        axis = inverse @ Rotation.from_quat(obj[3:]).as_matrix()[:, 2]
        half_span = NUT_HALF_HEIGHT * abs(axis[2])
        overlap.append(
            float(
                max(
                    0.0,
                    min(GUIDE[1], center[2] + half_span)
                    - max(GUIDE[0], center[2] - half_span),
                )
            )
        )
        radial.append(float(np.linalg.norm(center[:2])))
    installed_error = float(
        np.linalg.norm(inverse @ (retainer[:3] - screw[:3]) - INSTALLED_RETAINER)
    )
    retained = bool(
        all(x >= 0.003 for x in overlap)
        and max(radial) <= 0.003
        and installed_error <= 0.002
    )
    return dict(
        nuts_retained=retained,
        nut_guide_overlap_m=overlap,
        guide_radial_m=radial,
        retainer_installed_error_m=installed_error,
    )
