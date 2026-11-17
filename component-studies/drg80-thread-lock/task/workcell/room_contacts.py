"""Native room-wide contact evidence with explicit uncertainty between damage bands."""

import itertools
from pathlib import Path
from runpy import run_path
import numpy as np
from scipy.spatial.transform import Rotation
from tool_drop import ToolDrop


def closing_speed(points, normals, center, velocity, other_center, other_velocity):
    first = velocity[:3] + np.cross(velocity[3:], points - center)
    other = other_velocity[:3] + np.cross(other_velocity[3:], points - other_center)
    # PhysX contact normals point from the filter body toward the sensor.
    return np.sum((first - other) * normals, axis=1)


class RoomContacts:
    def __init__(self, world, stage, hz):
        from pxr import UsdPhysics

        dyn = [
            str(p.GetPath())
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        static = [
            str(p.GetPath())
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            and not any(
                str(p.GetPath()) == root or str(p.GetPath()).startswith(root + "/")
                for root in dyn
            )
        ]
        self.paths = dyn + static
        self.tool_ids = [
            next(i for i, p in enumerate(dyn) if p.endswith("/" + name))
            for name in ("driver", "wrench")
        ]
        self.tips = {
            i: tuple(p.rsplit("/", 1))
            for i, p in enumerate(dyn)
            if p.endswith(("/tip_left", "/tip_right"))
        }
        self.impact_ids = {
            i
            for i, p in enumerate(self.paths)
            if any(
                k in p
                for k in (
                    "/gripper_bench/",
                    "_rack/",
                    "_support_",
                    "/stand_mast/",
                    "/stand_arm/",
                    "/stand_foot/",
                    "/camera_housing/",
                )
            )
        }
        from pxr import Usd, UsdGeom

        table_path = next(p for p in static if "/table/" in p)
        bounds = (
            UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_],
                useExtentsHint=False,
                ignoreVisibility=True,
            )
            .ComputeWorldBound(stage.GetPrimAtPath(table_path))
            .ComputeAlignedRange()
        )
        table_min, table_max = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        self.table_bounds = [table_min.tolist(), table_max.tolist()]
        self.drop = [
            ToolDrop(table_min[:2], table_max[:2], table_max[2]) for _ in self.tool_ids
        ]
        self.last_tool_observations = []
        self.ndyn = len(dyn)
        self.dt = 1 / hz
        self.capacity = 8192
        self.contact = world.create_rigid_contact_view(
            dyn,
            filter_patterns=[self.paths for _ in dyn],
            max_contact_data_count=self.capacity,
        )
        self.motion = world.create_rigid_body_view(dyn)
        lookup = {path: i for i, path in enumerate(self.paths)}
        self.sensors = [lookup[p] for p in self.contact.sensor_paths]
        self.motion_order = [list(self.motion.prim_paths).index(p) for p in dyn]
        if set(self.sensors) != set(range(self.ndyn)):
            raise ValueError("incomplete native contact sensor registry")
        local = self.motion.get_coms().numpy()
        self.com = local[self.motion_order, :3].copy()
        schema = run_path(
            str(Path(__file__).resolve().parent.parent / "contact_damage.py")
        )
        self.point = schema["ContactPoint"]
        ids = range(len(self.paths))
        pairs = set(itertools.combinations(ids, 2))
        self.exempt = {
            tuple(sorted((i, j)))
            for i, p in enumerate(dyn)
            if p.endswith("/Geometry/base")
            for j, st in enumerate(self.paths)
            if "/table/" in st
        }
        self.strict = schema["ContactDamage"](hz, ids, (), self.exempt)
        self.universal = schema["ContactDamage"](
            hz, ids, pairs - self.exempt, self.exempt
        )
        # A pad or arm pressing a moving jaw/piston is always unintended.
        # This makes a force between the two limits conclusive on those pairs.
        jaws = {
            i
            for i, p in enumerate(self.paths)
            if p.rsplit("/", 1)[-1] in ("left_jaw", "right_jaw", "piston")
        }
        robot = {
            i for i, p in enumerate(self.paths) if "/LeftArm/" in p or "/RightArm/" in p
        }
        self.known_unintended = {tuple(sorted((i, j))) for i in jaws for j in robot}
        self.known_damage = schema["ContactDamage"](
            hz, ids, pairs - self.exempt - self.known_unintended, self.exempt
        )
        self.min_signed_force = 0.0
        self.negative_normals = 0
        self.peak_force = 0.0
        self.peak_pen = 0.0
        self.pair_peaks = {}
        self.first_static_normal = None

    def observe(self, tick, contained):
        forces, positions, normals, seps, counts, starts = (
            self.contact.get_contact_data(dt=self.dt)
        )
        counts = counts.numpy()
        if (
            counts.shape != (self.ndyn, len(self.paths))
            or counts.sum() >= self.capacity
        ):
            raise ValueError("incomplete native room contact buffer")
        contacts = []
        transforms = self.motion.get_transforms().numpy()[self.motion_order]
        vels = self.motion.get_velocities().numpy()[self.motion_order]
        centers = transforms[:, :3] + Rotation.from_quat(transforms[:, 3:]).apply(
            self.com
        )
        finger_loads = {tool: {} for tool in self.tool_ids}
        impacts = set()
        if counts.any():
            starts = starts.numpy()
            forces, positions, normals, seps = [
                v.numpy() for v in (forces, positions, normals, seps)
            ]
            static_zeros = np.zeros(6)
            seen = set()
            for row, j in zip(*np.nonzero(counts)):
                i = self.sensors[row]
                pair = tuple(sorted((i, int(j))))
                if pair in self.exempt or pair in seen:
                    continue
                if i == j:
                    raise ValueError("unexpected self contact under same native ID")
                seen.add(pair)
                a = int(starts[row, j])
                b = a + int(counts[row, j])
                if a < 0 or b > len(forces):
                    raise ValueError("room contact slice exceeds buffer")
                force = forces[a:b].ravel()
                normal = normals[a:b].reshape(-1, 3)
                points = positions[a:b].reshape(-1, 3)
                gaps = seps[a:b].ravel()
                if not all(np.isfinite(t).all() for t in (force, normal, points, gaps)):
                    raise ValueError(
                        f"nonfinite contact on {self.paths[i]} / {self.paths[j]}"
                    )
                self.min_signed_force = min(self.min_signed_force, float(force.min()))
                self.negative_normals += int(np.count_nonzero(force < 0.0))
                # The task contract defines compression as max(0, signed impulse).
                # Tensile samples contribute no compressive work or normal load.
                force = np.maximum(force, 0.0)
                speed = closing_speed(
                    points,
                    normal,
                    centers[i],
                    vels[i],
                    centers[j] if j < self.ndyn else static_zeros[:3],
                    vels[j] if j < self.ndyn else static_zeros,
                )
                contacts.extend(
                    self.point(i, int(j), float(g), float(f * self.dt), float(u))
                    for g, f, u in zip(gaps, force, speed)
                )
                fn = float(np.maximum(force, 0.0).sum())
                if fn > 0.0 and (i in finger_loads or j in finger_loads):
                    for who, other, orient in ((i, j, 1), (j, i, -1)):
                        if who not in finger_loads:
                            continue
                        if other in self.impact_ids:
                            impacts.add(who)
                        if other in self.tips:
                            arm, tip = self.tips[other]
                            finger_loads[who].setdefault(arm, {})[tip] = orient * (
                                normal.T @ force
                            )
                pen = max(0.0, float(-gaps.min()))
                self.pair_peaks[pair] = max(self.pair_peaks.get(pair, 0.0), fn)
                self.peak_force = max(self.peak_force, fn)
                self.peak_pen = max(self.peak_pen, pen)
                if (
                    self.first_static_normal is None
                    and self.paths[i].endswith("/driver")
                    and "driver_support" in self.paths[j]
                    and fn > 0.01
                ):
                    self.first_static_normal = (normal.T @ force / fn).tolist()
        self.last_tool_observations = []
        for idx, who in enumerate(self.tool_ids):
            held = any(
                "tip_left" in tips
                and "tip_right" in tips
                and np.dot(tips["tip_left"], tips["tip_right"]) < 0.0
                for tips in finger_loads[who].values()
            )
            failure = self.drop[idx].observe(
                tick, centers[who], held or contained[idx], who in impacts
            )
            self.last_tool_observations.append(
                dict(
                    centre=centers[who].tolist(),
                    opposed_fingers=held,
                    supported=bool(held or contained[idx]),
                    equipment_impact=who in impacts,
                    failure=failure,
                )
            )
        self.strict.observe(tick, contacts)
        self.universal.observe(tick, contacts)
        self.known_damage.observe(tick, contacts)
        return self.known_damage.failure or next(
            (m.failure for m in self.drop if m.failure), None
        )

    def report(self):
        return dict(
            ticks=self.strict.tick,
            registered_actors=self.paths,
            tabletop_collider_bounds_m=self.table_bounds,
            last_tool_observations=self.last_tool_observations,
            tool_drop_or_loss=[m.failure for m in self.drop],
            peak_pair_force_n=self.peak_force,
            min_signed_normal_force_n=self.min_signed_force,
            negative_normal_samples=self.negative_normals,
            peak_penetration_m=self.peak_pen,
            strict_failure=self.strict.failure,
            universal_failure=self.universal.failure,
            confirmed_contact_failure=self.known_damage.failure,
            exempt_base_table_pairs=sorted(self.exempt),
            first_driver_support_normal=self.first_static_normal,
            active_pairs=[
                dict(actors=[self.paths[i], self.paths[j]], peak_force_n=f)
                for (i, j), f in sorted(self.pair_peaks.items())
            ],
            status="universal_failure"
            if self.universal.failure
            else "confirmed_unintended_damage"
            if self.known_damage.failure
            else "needs_surface_classification"
            if self.strict.failure
            else "below_both_damage_bands",
        )
