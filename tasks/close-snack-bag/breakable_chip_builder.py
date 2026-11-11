"""Build pre-fragmented chip surrogates and irreversible seam springs."""
import numpy as np
import warp as wp


def add_breakable_chips(builder, centers=((.055,-.025),(.055,.025),(.105,-.025),(.105,.025))):
    """Insert 4 chips as pairs of tetrahedral grids; return particle/spring indices.

    Geometry and bond stiffness are uncalibrated. Fixed, parallel seam is a
    discrete fracture approximation, not general mesh cutting.
    """
    groups, seam_springs = [], []
    for cx, cy in centers:
        chip_parts = []
        for x0 in (cx-.01425, cx+.00025):
            start = len(builder.particle_q)
            builder.add_soft_grid(pos=wp.vec3(x0, cy-.012, .01875), rot=wp.quat_identity(),
                vel=wp.vec3(0,0,0), dim_x=2, dim_y=2, dim_z=1,
                cell_x=.007, cell_y=.012, cell_z=.0025,
                density=300., k_mu=1.e6, k_lambda=1.e6, k_damp=10., particle_radius=.0001)
            chip_parts.append(np.arange(start, len(builder.particle_q)))
        points = np.asarray(builder.particle_q)
        xleft, xright = cx-.00025, cx+.00025
        left = chip_parts[0][np.isclose(points[chip_parts[0],0], xleft, atol=1e-7)]
        right = chip_parts[1][np.isclose(points[chip_parts[1],0], xright, atol=1e-7)]
        if len(left) != 6 or len(right) != 6:
            raise RuntimeError('unexpected chip-seam node count')
        chip_springs = []
        for i in left:
            distances = np.linalg.norm(points[right,1:] - points[i,1:], axis=1)
            if distances.min() > 1e-6:
                raise RuntimeError('no matching point across chip seam')
            chip_springs.append(len(builder.spring_rest_length))
            builder.add_spring(int(i), int(right[np.argmin(distances)]), ke=20., kd=.002, control=0)
        groups.append(np.concatenate(chip_parts))
        seam_springs.append(np.asarray(chip_springs, dtype=np.int32))
    return groups, seam_springs



@wp.kernel
def _check_and_break(q: wp.array(dtype=wp.vec3), indices: wp.array(dtype=int),
                     rest: wp.array(dtype=float), stiffness: wp.array(dtype=float),
                     damping: wp.array(dtype=float), groups: wp.array2d(dtype=int),
                     flags: wp.array(dtype=int), peaks: wp.array(dtype=float), threshold: float):
    chip = wp.tid()
    peak = peaks[chip]
    for j in range(groups.shape[1]):
        spring = groups[chip, j]
        a, b = indices[2*spring], indices[2*spring+1]
        extension = wp.length(q[b]-q[a]) - rest[spring]
        peak = wp.max(peak, extension)
    peaks[chip] = peak
    if peak >= threshold:
        flags[chip] = 1
        for j in range(groups.shape[1]):
            spring = groups[chip, j]
            stiffness[spring] = 0.0
            damping[spring] = 0.0


class ChipFractureMonitor:
    """Persistent fracture flag evaluated before each solver substep on GPU."""
    def __init__(self, model, groups, threshold):
        if threshold <= 0 or not groups or len({len(g) for g in groups}) != 1:
            raise ValueError('positive threshold and equal nonempty seam groups required')
        self.model, self.threshold = model, threshold
        self.groups = wp.array(np.stack(groups).astype(np.int32), dtype=int, device=model.device)
        self.flags = wp.zeros(len(groups), dtype=int, device=model.device)
        self.peaks = wp.zeros(len(groups), dtype=float, device=model.device)
    def apply(self, state):
        wp.launch(_check_and_break, dim=len(self.groups),
                  inputs=[state.particle_q, self.model.spring_indices, self.model.spring_rest_length,
                          self.model.spring_stiffness, self.model.spring_damping, self.groups,
                          self.flags, self.peaks, self.threshold], device=self.model.device)
