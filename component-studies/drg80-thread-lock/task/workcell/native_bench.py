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
        support_stiffness=1e6,
        measured_contact=False,
        implicit_axial=False,
        transmitted_reaction=False,
    ):
        self.implicit_axial = implicit_axial
        self.transmitted_reaction = transmitted_reaction
        self.old_axial_velocity = None
        self.previous_thread_angles = None
        if transmitted_reaction and not (implicit_axial and native_friction):
            raise ValueError(
                "reaction measurement requires physical axial drives and dry friction"
            )
        self.measured_contact = measured_contact
        self.previous_q = None
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
        self.k = np.array([support_stiffness, 1e5])
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
        self.driver_contact_force = np.zeros(3)
        self.tool_pairs = (
            []
            if not tools
            else [
                (name, pair(name, tool)) for name in ("nut", "screw") for tool in tools
            ]
        )
        self.driver_pair = self.tool_pairs[len(tools)][1] if tools else None
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
        efforts = np.zeros(8)
        self.driver_contact_force.fill(0)
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
            if (
                view is self.driver_pair
            ):  # Screw on driver is the opposite of the measured tool-on-screw load.
                self.driver_contact_force -= vectors.sum(axis=0)
            torque = float(np.cross(lever, vectors)[:, 2].sum())
            force = float(vectors[:, 2].sum())
            # Sidewall friction also loads a real driver axially. Normal
            # impulses alone omit this component of the measured tool wrench.
            tangent, anchors, patches, offsets = view.get_friction_data(dt=self.h)
            patch_count = int(patches.numpy().ravel()[0])
            if patch_count:
                head = int(offsets.numpy().ravel()[0])
                region = slice(head, head + patch_count)
                tangent = tangent.numpy()[region].reshape(-1, 3)
                arms = anchors.numpy()[region].reshape(-1, 3) - position
                if view is self.driver_pair:
                    self.driver_contact_force -= tangent.sum(axis=0)
                force += float(tangent[:, 2].sum())
                torque += float(np.cross(arms, tangent)[:, 2].sum())
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
        if self.transmitted_reaction:
            # Axial momentum balance in the two serial prismatic coordinates.
            # The native bearing and stop normals act upward on their workpieces.
            # Their opposite impulses act on the grounded frame and the piston.
            seat_z = self.seat.get_contact_force_matrix(dt=self.h).numpy().reshape(3)[2]
            stop_z = self.stop.get_contact_force_matrix(dt=self.h).numpy().reshape(3)[2]
            va = v[[0, 2]]
            accel = (
                np.zeros(2)
                if self.old_axial_velocity is None
                else (va - self.old_axial_velocity) / self.h
            )
            self.old_axial_velocity = va.copy()
            # The contact views use nut/frame and piston/screw body order.
            measured = (
                (gravity + applied + sensed)[[0, 2]]
                + np.array([seat_z - stop_z, seat_z])
                - self.mass[np.ix_([0, 2], [0, 2])] @ accel
            )
            self.loads = measured.copy()
            internal = self.axes.T @ (0.001 / math.tau * measured)
        else:
            predicted_mass = self.mass.copy()
            predicted_mass[4, 4] += (
                150**2 * (self.mass[5, 5] + self.mass[6, 6]) + 350 * self.h
            )
            external = applied + sensed + gravity
            external[4] -= 350 * v[4]
            rows = self.rows
            gaps = self.gap0 + self.rows @ q
            if self.measured_contact:
                if self.previous_q is not None:
                    v = v.copy()
                    v[:4] = (q[:4] - self.previous_q[:4]) / self.h
                self.previous_q = q.copy()
                normal = [
                    abs(view.get_contact_force_matrix(dt=self.h).numpy().reshape(3)[2])
                    for view in (self.seat, self.stop)
                ]
                keep = np.ones(len(rows), dtype=bool)
                for j, load in enumerate(normal):
                    if load > 0:
                        # A contact impulse has already been observed. Use it as the
                        # predictor's external load, and remove that predictive bound.
                        # PhysX remains the sole writer of real contact impulses.
                        external += load * rows[j]
                        keep[j] = False
                rows, gaps = rows[keep], gaps[keep]
            internal, self.loads, friction = solve_internal(
                predicted_mass,
                v,
                external,
                self.g,
                self.g @ q,
                self.k,
                self.d,
                rows,
                gaps,
                self.h,
                self.axes,
            )
            measured = self.k * (self.g @ q) + self.d * (self.g @ v)
            if self.native_friction:
                internal -= self.axes.T @ friction
        if self.native_friction:
            # The PhysX articulation exposes static/dynamic friction EFFORT per
            # DOF, independently of the finite diagnostic drive. Let the same
            # solver resolve contact and the load-dependent angular stick/slip.
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
        if self.implicit_axial:
            # The true contact solver acts on the passive axial spring damper
            # within the same step. Its moving datum is solely the thread angle.
            # The predictor still supplies the conjugate thread torques.
            lead = 0.001 / math.tau
            pos = self.art.get_dof_position_targets().numpy().copy()
            vel = self.art.get_dof_velocity_targets().numpy().copy()
            angles = q[[1, 3]]
            # Reported PhysX velocity can be nonzero while the angular pose is stationary.
            # Evaluate the moving axial datum from successive actual thread orientations.
            omega = (
                np.zeros(2)
                if self.previous_thread_angles is None
                else (angles - self.previous_thread_angles) / self.h
            )
            self.previous_thread_angles = angles.copy()
            for i, (axial, rotary) in enumerate(((0, 1), (2, 3))):
                pos[0, self.index[axial]] = lead * (q[rotary] + self.h * omega[i])
                vel[0, self.index[axial]] = lead * omega[i]
                internal[axial] = 0.0
            self.art.set_dof_position_targets(
                wp.array(pos, device=self.device), self.ids
            )
            self.art.set_dof_velocity_targets(
                wp.array(vel, device=self.device), self.ids
            )
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

    def seat_face_load(self):
        """Sum resolved normal forces on the N73/cover bearing face."""
        force, point, normal, separation, count, start = self.seat.get_contact_data(
            dt=self.h
        )
        n = int(count.numpy().ravel()[0])
        if not n:
            return 0.0, 0, 0.0
        first = int(start.numpy().ravel()[0])
        sl = slice(first, first + n)
        transform = self.bodies["frame"].get_transforms().numpy()[0]
        x, y, z, w = transform[3:]
        up = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )
        across = np.array(
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
        )
        center = transform[:3] - 0.018 * across + 0.090 * up
        rel = point.numpy()[sl].reshape(-1, 3) - center
        height = rel @ up
        local_radius = np.linalg.norm(rel - np.outer(height, up), axis=1)
        faces = (
            (np.abs(normal.numpy()[sl].reshape(-1, 3) @ up) > 0.95)
            & (np.abs(height) < 0.00025)
            & (local_radius <= 0.0076)
        )
        loads = np.abs(force.numpy()[sl].ravel())
        depths = np.maximum(0.0, -separation.numpy()[sl].ravel())
        return float(loads[faces].sum()), int(faces.sum()), float(depths.max())

    def read(self):
        q, v = self.state()
        seat = float(
            abs(self.seat.get_contact_force_matrix(dt=self.h).numpy().reshape(3)[2])
        )
        stop = float(
            abs(self.stop.get_contact_force_matrix(dt=self.h).numpy().reshape(3)[2])
        )
        face_n, points, penetration = self.seat_face_load()
        effort = float(self.art.get_dof_actuation_forces().numpy()[0, self.index[4]])
        return dict(
            projected_axial_n=self.art.get_dof_projected_joint_forces()
            .numpy()[0, [self.index[0], self.index[2]]]
            .tolist()
            if self.implicit_axial
            else None,
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
            seat_pair_axial_n=seat,
            seat_n=face_n,
            seat_face_points=points,
            seat_max_pair_penetration_m=penetration,
            stop_n=stop,
            stop_gap=self.gap0[1] + q[0] - q[4],
            seat_gap=self.gap0[0] + q[0] + q[2],
            actuator_n=effort,
            piston_total_n=effort,
            bearing_nm=self.bearing_nm,
            thread_loads=self.loads.tolist(),
        )
