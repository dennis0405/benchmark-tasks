"""Universal upper bounds for protected hardware versus tools/robot contacts."""

import numpy as np


def contact_bounds(forces, separations, counts, starts):
    """Read only populated PhysX manifold slices. Force values use dt, in N."""
    counts = np.asarray(counts).ravel()
    used = counts > 0
    if not used.any():
        return 0.0, 0.0
    starts = np.asarray(starts).ravel()[used].astype(int)
    ends = starts + counts[used]
    force = np.asarray(forces).ravel()
    sep = np.asarray(separations).ravel()
    if np.any(starts < 0) or np.any(ends > len(force)) or np.any(ends > len(sep)):
        raise ValueError("incomplete protected contact buffer")
    maximum_force = 0.0
    penetration = 0.0
    for a, b in zip(starts, ends):
        pair = force[a:b]
        gaps = sep[a:b]
        if not np.isfinite(pair).all() or not np.isfinite(gaps).all():
            raise ValueError("nonfinite protected contact")
        maximum_force = max(maximum_force, float(np.maximum(pair, 0.0).sum()))
        penetration = max(penetration, float(-gaps.min()))
    return maximum_force, penetration
