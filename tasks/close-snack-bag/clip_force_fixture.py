"""Experimental bounded wrench fixture for a free, articulated clip."""
import numpy as np
import warp as wp


@wp.kernel
def _drive(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
           body_f: wp.array(dtype=wp.spatial_vector), body_id: int,
           target: wp.array(dtype=wp.vec3), enabled: wp.array(dtype=float), limit: float):
    if enabled[0] > 0.5:
        pose = body_q[body_id]
        v = body_qd[body_id]
        pos = wp.transform_get_translation(pose)
        rot = wp.transform_get_rotation(pose)
        velocity = wp.vec3(v[0], v[1], v[2])
        omega = wp.vec3(v[3], v[4], v[5])
        linear = 25.0 * (target[0] - pos) - 0.6 * velocity
        size = wp.length(linear)
        if size > limit:
            linear = linear * (limit / size)
        wanted = wp.quat(0.0, 0.0, 1.0, 0.0)
        error = wanted * wp.quat_inverse(rot)
        sign = wp.where(error[3] >= 0.0, 1.0, -1.0)
        torque = 0.04 * sign * wp.vec3(error[0], error[1], error[2]) - .003 * omega
        torque_size = wp.length(torque)
        if torque_size > .04:
            torque = torque * (.04 / torque_size)
        wp.atomic_add(body_f, body_id, wp.spatial_vector(linear, torque))


class ClipForceFixture:
    def __init__(self, model, body_id, limit=1.0):
        if limit <= 0:
            raise ValueError("force limit must be positive")
        self.device, self.body_id, self.limit = model.device, body_id, limit
        self.target = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.enabled = wp.zeros(1, dtype=float, device=self.device)
    def set(self, pos, active):
        wp.copy(self.target, wp.array(np.asarray([pos], dtype=np.float32), dtype=wp.vec3, device=self.device))
        wp.copy(self.enabled, wp.array([float(active)], dtype=float, device=self.device))
    def apply(self, state):
        wp.launch(_drive, dim=1, inputs=[state.body_q, state.body_qd, state.body_f,
                                         self.body_id, self.target, self.enabled, self.limit], device=self.device)
