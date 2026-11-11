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
def _contents_volume(q: wp.array(dtype=wp.vec3), tets: wp.array(dtype=wp.vec4i),
                     orientation: wp.array(dtype=float), stats: wp.array(dtype=float)):
    i = wp.tid()
    t = tets[i]
    a, b, c, d = q[t[0]], q[t[1]], q[t[2]], q[t[3]]
    volume = orientation[i] * wp.dot(b-a, wp.cross(c-a, d-a)) / 6.0
    wp.atomic_add(stats, 3, volume)
    if volume <= 0.0:
        wp.atomic_add(stats, 4, 1.0)


@wp.kernel
def _init_channel(stats: wp.array(dtype=float)):
    stats[5] = 1.e10


@wp.kernel
def _channel_area(q: wp.array(dtype=wp.vec3), grid: wp.array3d(dtype=int),
                  first: int, gap: float, stats: wp.array(dtype=float)):
    i = wp.tid() + first
    area = float(0.0)
    for j in range(grid.shape[2] - 1):
        a, b = q[grid[0,i,j]], q[grid[1,i,j]]
        c, d = q[grid[0,i,j+1]], q[grid[1,i,j+1]]
        midpoint0 = .5 * (a+b)
        midpoint1 = .5 * (c+d)
        g0 = wp.max(wp.length(b-a) - gap, 0.0)
        g1 = wp.max(wp.length(d-c) - gap, 0.0)
        area += wp.length(midpoint1-midpoint0) * (g0+g1) * .5
    wp.atomic_min(stats, 5, area)


@wp.kernel
def _gas_update(stats: wp.array(dtype=float), moles: wp.array(dtype=float),
                dt: float, temperature: float, ambient: float, opening_scale: float, use_channel: int):
    volume = stats[0] - stats[3]
    if volume <= 1.e-10 or stats[4] > 0.0:
        stats[2] = -1.0e9  # invalid geometry sentinel; never apply a huge force
    else:
        p = moles[0] * 8.314462618 * temperature / volume
        diff = p - ambient
        rho = wp.max(wp.where(diff >= 0.0, p, ambient), 1.e-9) * 0.02897 / (8.314462618 * temperature)
        area = wp.where(use_channel != 0, wp.min(stats[1], stats[5]), stats[1])
        flux = 0.6 * area * opening_scale * wp.sqrt(2.0 * rho * wp.abs(diff)) / 0.02897
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
    def __init__(self, model, packet_points, packet_faces, grid, start, solver_dt, opening_scale=1.0, channel=False, contact_gap=.0002):
        if not 0.0 <= opening_scale <= 1.0:
            raise ValueError('opening scale must be between 0 and 1')
        cap = mouth_cap(grid)
        volume = enclosed_volume(packet_points, packet_faces, cap)
        self.device = model.device
        self.start = start
        self.dt = solver_dt
        self.opening_scale = opening_scale
        self.channel = channel
        self.first_column = int(round(.7*(grid.shape[1]-1)))
        self.contact_gap = contact_gap
        self.grid = wp.array((grid + start).astype(np.int32), dtype=int, device=self.device)
        self.film = wp.array((packet_faces + start).astype(np.int32), dtype=wp.vec3i, device=self.device)
        self.cap = wp.array((cap + start).astype(np.int32), dtype=wp.vec3i, device=self.device)
        self.closed = wp.array((np.vstack((packet_faces, cap)) + start).astype(np.int32),
                               dtype=wp.vec3i, device=self.device)
        self.stats = wp.zeros(6, dtype=float, device=self.device)
        self.tets, self.orientation = None, None
        if getattr(model, "tet_indices", None) is not None and model.tet_count:
            tets = model.tet_indices.numpy().astype(np.int32)
            xyz = model.particle_q.numpy()[tets].astype(np.float64)
            signed = np.einsum("ij,ij->i", xyz[:,1]-xyz[:,0],
                               np.cross(xyz[:,2]-xyz[:,0], xyz[:,3]-xyz[:,0])) / 6
            if np.any(np.abs(signed) < 1e-16):
                raise ValueError("contents contain a degenerate initial tetrahedron")
            volume -= float(np.abs(signed).sum())
            if volume <= 0:
                raise ValueError("contents exceed the gas cavity")
            self.tets = wp.array(tets, dtype=wp.vec4i, device=self.device)
            self.orientation = wp.array(np.sign(signed), dtype=float, device=self.device)
        self.moles = wp.array([101325.0 * volume / (R * 293.15)], dtype=float, device=self.device)

    def apply(self, state):
        self.stats.zero_()
        wp.launch(_accumulate_geometry, dim=len(self.closed), inputs=[state.particle_q, self.closed, self.start, self.stats], device=self.device)
        wp.launch(_mouth_geometry, dim=len(self.cap), inputs=[state.particle_q, self.cap, self.stats], device=self.device)
        if self.channel:
            wp.launch(_init_channel, dim=1, inputs=[self.stats], device=self.device)
            wp.launch(_channel_area, dim=self.grid.shape[1]-self.first_column,
                      inputs=[state.particle_q, self.grid, self.first_column, self.contact_gap, self.stats], device=self.device)
        if self.tets is not None:
            wp.launch(_contents_volume, dim=len(self.tets), inputs=[state.particle_q, self.tets, self.orientation, self.stats], device=self.device)
        wp.launch(_gas_update, dim=1, inputs=[self.stats, self.moles, self.dt, 293.15, 101325.0, self.opening_scale, int(self.channel)], device=self.device)
        wp.launch(_film_force, dim=len(self.film), inputs=[state.particle_q, state.particle_f, self.film, self.stats, 1.0], device=self.device)
        # A seal control uses a virtual closure traction to conserve net force.
        wp.launch(_film_force, dim=len(self.cap), inputs=[state.particle_q, state.particle_f, self.cap, self.stats, 1.0 - self.opening_scale], device=self.device)
