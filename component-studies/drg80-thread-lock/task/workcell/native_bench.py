"""Native DRG workcell. The mechanism reads positions and writes only internal efforts."""

import math
import numpy as np
import warp as wp
from isaaclab_physx.physics import PhysxManager
from thread_system import solve_internal


class Bench:
    def __init__(
        self,
        root,
        hz,
        device="cuda:0",
        tools=None,
        valve_force=5.0,
        rise_seconds=0.0,
        native_friction=False,
    ):
        self.valve_force = float(valve_force)
        self.rise_seconds = float(rise_seconds)
        self.h = 1 / hz
        self.device = device
        self.ids = wp.array([0], dtype=wp.uint32, device=device)
        physics = PhysxManager.get_physics_sim_view()
        physics.set_subspace_roots("/")
        self.art = physics.create_articulation_view(root)
        assert self.art.count == 1
        self.native_friction = native_friction
        if native_friction:
            self.friction_props = self.art.get_dof_friction_properties().numpy().copy()
            self.cpu_ids = wp.array([0], dtype=wp.uint32, device="cpu")
        self.names = list(self.art.get_metatype(0).dof_names)
        self.order = ["sz", "st", "nz", "nt", "p", "jl", "jr", "valve"]
        self.index = [self.names.index(s) for s in self.order]
        matrix = self.art.get_generalized_mass_matrices().numpy()[0]
        if matrix.shape != (len(self.order), len(self.order)):
            raise RuntimeError(f"unexpected fixed-base mass matrix {matrix.shape}")
        self.mass = matrix[np.ix_(self.index, self.index)].astype(float)
        print(
            "joint inertias",
            dict(zip(self.order, self.mass.diagonal().tolist())),
            flush=True,
        )
        self.unit = np.eye(len(self.order))
        sz, st, nz, nt, p, jl, jr, valve = self.unit
        lead = 0.001 / math.tau
        self.g = np.array([sz - lead * st, nz - lead * nt])
        self.k = np.array([1e6, 1e5])
        self.d = np.array([4000.0, 2500.0])
        self.axes = np.array([st, nt])
        self.rows = np.array([sz + nz, sz - p, p, -p, sz, -sz, nz, -nz])
        self.gap0 = np.array(
            [0.001, 0.0057, 0.0, 0.010297442586766544, 0.0048, 0.0048, 0.005, 0.005]
        )
        self.root = root
        self.bodies = {
            k: physics.create_rigid_body_view([root + "/" + k])
            for k in (
                "frame",
                "screw",
                "nut",
                "piston",
                "left_jaw",
                "right_jaw",
                "lever",
                "retainer",
            )
        }

        def pair(first, target):
            return physics.create_rigid_contact_view(
                [root + "/" + first],
                filter_patterns=[[target]],
                max_contact_data_count=512,
            )

        self.seat = pair("nut", root + "/frame")
        self.stop = pair("piston", root + "/screw")
        self.tool_pairs = (
            []
            if not tools
            else [
                (name, pair(name, tool)) for name in ("nut", "screw") for tool in tools
            ]
        )
        self.trace = []
        self.actuator = 0.0
        self.peak_external = 0.0
        self.last_internal = np.zeros(8)
        self.loads = np.zeros(len(self.g))

    def state(self):
        return (
            self.art.get_dof_positions().numpy()[0, self.index].astype(float),
            self.art.get_dof_velocities().numpy()[0, self.index].astype(float),
        )

    def tool_efforts(self):
        if self.native_friction:
            # The PhysX articulation exposes static/dynamic friction EFFORT per
            # DOF, independently of the finite diagnostic drive. Let the same
            # solver resolve contact and the load-dependent angular stick/slip.
            measured = self.k * (self.g @ q) + self.d * (self.g @ v)
            for i, idx in enumerate((self.index[1], self.index[3])):
                force = abs(measured[i]) * 0.0035
                self.friction_props[0, idx, :] = (
                    (0.08, 0.35)[i] * force,
                    (0.064, 0.28)[i] * force,
                    0.0,
                )
            self.art.set_dof_friction_properties(
                wp.array(self.friction_props, device="cpu"), self.cpu_ids
            )
            internal -= self.axes.T @ friction
        efforts = np.zeros(8)
        for name, view in self.tool_pairs:
            forces, points, normals, sep, count, start = view.get_contact_data(
                dt=self.h
            )
            n = int(count.numpy().ravel()[0])
            if not n:
                continue
            begin = int(start.numpy().ravel()[0])
            sl = slice(begin, begin + n)
            position = self.bodies[name].get_transforms().numpy()[0, :3]
            lever = points.numpy()[sl].reshape(-1, 3) - position
            vectors = forces.numpy()[sl].reshape(-1, 1) * normals.numpy()[sl].reshape(
                -1, 3
            )
            torque = float(np.cross(lever, vectors)[:, 2].sum())
            force = float(vectors[:, 2].sum())
            # The parent coordinates also move the distal nut.
            efforts[:2] += (force, torque)
            if name == "nut":
                efforts[2:4] += (force, torque)
        return efforts

    def bearing_torque(self):
        force, point, count, start = self.seat.get_friction_data(dt=self.h)
        n = int(count.numpy().ravel()[0])
        if n == 0:
            return 0.0
        begin = int(start.numpy().ravel()[0])
        indices = slice(begin, begin + n)
        center = self.bodies["nut"].get_transforms().numpy()[0, :3]
        radius = point.numpy()[indices].reshape(-1, 3) - center
        tangent = force.numpy()[indices].reshape(-1, 3)
        return float(np.cross(radius, tangent)[:, 2].sum())

    def step(self, diagnostic=None):
        q, v = self.state()
        degrees = math.degrees(q[7])
        target = (
            self.valve_force
            if degrees >= 25
            else -self.valve_force
            if degrees <= -25
            else 0.0
        )
        if self.rise_seconds and target and self.actuator * target >= 0:
            ramp = self.valve_force * self.h / self.rise_seconds
            self.actuator += min(ramp, max(-ramp, target - self.actuator))
        else:
            self.actuator = target
        applied = (
            np.zeros(8) if diagnostic is None else np.array(diagnostic, dtype=float)
        )
        applied[4] += self.actuator
        # Three valve detents, with a dissipative bearing.
        applied[7] += -0.025 * math.sin(12 * q[7]) - 0.002 * v[7]
        sensed = self.tool_efforts()
        self.bearing_nm = self.bearing_torque()
        sensed[1] += self.bearing_nm
        sensed[3] += self.bearing_nm
        gravity = -self.art.get_gravity_compensation_forces().numpy()[0, self.index]
        predicted_mass = self.mass.copy()
        predicted_mass[4, 4] += (
            150**2 * (self.mass[5, 5] + self.mass[6, 6]) + 350 * self.h
        )
        external = applied + sensed + gravity
        external[4] -= 350 * v[4]
        internal, self.loads, friction = solve_internal(
            predicted_mass,
            v,
            external,
            self.g,
            self.g @ q,
            self.k,
            self.d,
            self.rows,
            self.gap0 + self.rows @ q,
            self.h,
            self.axes,
        )
        if self.native_friction:
            # The PhysX articulation exposes static/dynamic friction EFFORT per
            # DOF, independently of the finite diagnostic drive. Let the same
            # solver resolve contact and the load-dependent angular stick/slip.
            measured = self.k * (self.g @ q) + self.d * (self.g @ v)
            for i, idx in enumerate((self.index[1], self.index[3])):
                force = abs(measured[i]) * 0.0035
                self.friction_props[0, idx, :] = (
                    (0.08, 0.35)[i] * force,
                    (0.064, 0.28)[i] * force,
                    0.0,
                )
            self.art.set_dof_friction_properties(
                wp.array(self.friction_props, device="cpu"), self.cpu_ids
            )
            internal -= self.axes.T @ friction
        efforts = np.zeros((1, self.art.max_dofs), dtype=np.float32)
        efforts[0, self.index] = applied + internal
        # Jaw gearing and piston damping live in the native constraints. The
        # thread contributes no piston effort, so the port can be read directly.
        self.art.set_dof_actuation_forces(
            wp.array(efforts, device=self.device), self.ids
        )
        self.last_internal = internal
        self.peak_external = max(self.peak_external, float(np.max(abs(sensed))))

    def poses(self):
        return {
            k: v.get_transforms().numpy()[0].tolist() for k, v in self.bodies.items()
        }

    def read(self):
        q, v = self.state()
        seat = float(
            abs(self.seat.get_contact_force_matrix(dt=self.h).numpy().reshape(3)[2])
        )
        stop = float(
            abs(self.stop.get_contact_force_matrix(dt=self.h).numpy().reshape(3)[2])
        )
        effort = float(self.art.get_dof_actuation_forces().numpy()[0, self.index[4]])
        return dict(
            sz=q[0],
            st=q[1],
            nz=q[2],
            nt=q[3],
            p=q[4],
            jl=q[5],
            jr=q[6],
            lever=q[7],
            angle=3 + math.degrees(q[6] - q[5]),
            screw_speed=v[1],
            nut_speed=v[1] + v[3],
            piston_speed=v[4],
            angular_speed=math.degrees(v[6] - v[5]),
            seat_n=seat,
            stop_n=stop,
            stop_gap=self.gap0[1] + q[0] - q[4],
            seat_gap=self.gap0[0] + q[0] + q[2],
            actuator_n=effort,
            piston_total_n=effort,
            bearing_nm=self.bearing_nm,
            thread_loads=self.loads.tolist(),
        )
