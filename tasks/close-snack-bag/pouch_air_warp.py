"""GPU pressure callback: experimental isothermal gas, not CFD or calibrated."""
import numpy as np
import warp as wp
from pouch_air import R, enclosed_volume, mouth_cap


@wp.kernel
def _accumulate_geometry(q: wp.array(dtype=wp.vec3), faces: wp.array(dtype=wp.vec3i),
                         origin_index: int, stats: wp.array(dtype=float)):
    i = wp.tid()
    tri = faces[i]
    origin = q[origin_index]
    a = q[tri[0]] - origin
    b = q[tri[1]] - origin
    c = q[tri[2]] - origin
    wp.atomic_add(stats, 0, wp.dot(a, wp.cross(b, c)) / 6.0)


@wp.kernel
def _mouth_geometry(q: wp.array(dtype=wp.vec3), cap: wp.array(dtype=wp.vec3i),
                    stats: wp.array(dtype=float)):
    tri = cap[wp.tid()]
    a, b, c = q[tri[0]], q[tri[1]], q[tri[2]]
    wp.atomic_add(stats, 1, wp.length(wp.cross(b-a, c-a)) * 0.5)


@wp.kernel
def _gas_update(stats: wp.array(dtype=float), moles: wp.array(dtype=float),
                dt: float, temperature: float, ambient: float, opening_scale: float):
    volume = stats[0]
    if volume <= 1.e-10:
        stats[2] = -1.0e9  # invalid geometry sentinel; never apply a huge force
    else:
        p = moles[0] * 8.314462618 * temperature / volume
        diff = p - ambient
        rho = wp.max(wp.where(diff >= 0.0, p, ambient), 1.e-9) * 0.02897 / (8.314462618 * temperature)
        flux = 0.6 * stats[1] * opening_scale * wp.sqrt(2.0 * rho * wp.abs(diff)) / 0.02897
        equilibrium = ambient * volume / (8.314462618 * temperature)
        change = wp.min(flux * dt, wp.abs(moles[0] - equilibrium))
        moles[0] = moles[0] - wp.sign(diff) * change
        stats[2] = moles[0] * 8.314462618 * temperature / volume - ambient


@wp.kernel
def _film_force(q: wp.array(dtype=wp.vec3), forces: wp.array(dtype=wp.vec3),
                faces: wp.array(dtype=wp.vec3i), stats: wp.array(dtype=float), scale: float):
    tri = faces[wp.tid()]
    pressure = stats[2] * scale
    if pressure > -1.0e8:
        a, b, c = q[tri[0]], q[tri[1]], q[tri[2]]
        force = pressure * wp.cross(b-a, c-a) / 6.0
        wp.atomic_add(forces, tri[0], force)
        wp.atomic_add(forces, tri[1], force)
        wp.atomic_add(forces, tri[2], force)


class LumpedGas:
    def __init__(self, model, packet_points, packet_faces, grid, start, solver_dt, opening_scale=1.0):
        if not 0.0 <= opening_scale <= 1.0:
            raise ValueError('opening scale must be between 0 and 1')
        cap = mouth_cap(grid)
        volume = enclosed_volume(packet_points, packet_faces, cap)
        self.device = model.device
        self.start = start
        self.dt = solver_dt
        self.opening_scale = opening_scale
        self.film = wp.array((packet_faces + start).astype(np.int32), dtype=wp.vec3i, device=self.device)
        self.cap = wp.array((cap + start).astype(np.int32), dtype=wp.vec3i, device=self.device)
        self.closed = wp.array((np.vstack((packet_faces, cap)) + start).astype(np.int32),
                               dtype=wp.vec3i, device=self.device)
        self.stats = wp.zeros(3, dtype=float, device=self.device)
        self.moles = wp.array([101325.0 * volume / (R * 293.15)], dtype=float, device=self.device)

    def apply(self, state):
        self.stats.zero_()
        wp.launch(_accumulate_geometry, dim=len(self.closed), inputs=[state.particle_q, self.closed, self.start, self.stats], device=self.device)
        wp.launch(_mouth_geometry, dim=len(self.cap), inputs=[state.particle_q, self.cap, self.stats], device=self.device)
        wp.launch(_gas_update, dim=1, inputs=[self.stats, self.moles, self.dt, 293.15, 101325.0, self.opening_scale], device=self.device)
        wp.launch(_film_force, dim=len(self.film), inputs=[state.particle_q, state.particle_f, self.film, self.stats, 1.0], device=self.device)
        # A seal control uses a virtual closure traction to conserve net force.
        wp.launch(_film_force, dim=len(self.cap), inputs=[state.particle_q, state.particle_f, self.cap, self.stats, 1.0 - self.opening_scale], device=self.device)
