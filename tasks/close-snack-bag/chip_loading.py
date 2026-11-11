"""Balanced external couple for a controlled chip-bending diagnostic."""
import numpy as np
import warp as wp


@wp.kernel
def _apply(qf: wp.array(dtype=wp.vec3), left: wp.array(dtype=int), right: wp.array(dtype=int),
           load: wp.array(dtype=float)):
    i = wp.tid()
    f = wp.vec3(0., 0., load[0] / float(left.shape[0]))
    wp.atomic_add(qf, left[i], -f)
    wp.atomic_add(qf, right[i], f)


class ChipBendLoad:
    def __init__(self, model, left, right):
        if len(left) == 0 or len(left) != len(right):
            raise ValueError('matching nonempty force faces required')
        self.device = model.device
        self.left = wp.array(np.asarray(left, dtype=np.int32), dtype=int, device=self.device)
        self.right = wp.array(np.asarray(right, dtype=np.int32), dtype=int, device=self.device)
        self.load = wp.zeros(1, dtype=float, device=self.device)
    def set_force(self, force):
        wp.copy(self.load, wp.array([force], dtype=float, device=self.device))
    def apply(self, state):
        wp.launch(_apply, dim=len(self.left), inputs=[state.particle_f, self.left, self.right, self.load], device=self.device)
