"""Live sieve outcome monitor. Sample after each completed PhysX substep."""

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from yam_arena.robot import asset_link_paths
from passage import Passage, rotation
from outcome import Outcome


def tip_paths_and_positions(base):
    """Actual current physics link origins; USD Xforms are static under Fabric."""
    paths = asset_link_paths()
    names, positions = [], []
    for robot_key, arm in (("left_robot", "LeftArm"), ("right_robot", "RightArm")):
        robot = base.scene.articulations[robot_key]
        for tip in ("tip_left", "tip_right"):
            indices = [
                i
                for i, name in enumerate(robot.body_names)
                if name.rsplit("/", 1)[-1] == tip
            ]
            if len(indices) != 1:
                raise RuntimeError(
                    f"expected one native body for {robot_key}/{tip}: {indices}"
                )
            value = robot.data.body_link_pos_w.torch[0, indices[0]]
            position = value.detach().cpu().numpy().astype(np.float64, copy=True)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise RuntimeError(
                    f"invalid native body-link position for {robot_key}/{tip}"
                )
            names.append(f"/World/envs/env_0/{arm}" + paths[tip])
            positions.append(position)
    return names, positions


def tip_collision_bounding_radii(tip_paths):
    """Conservative radii enclosing every collision shape below each tip link.

    The radius is derived from the authored collision geometry's world bounds,
    not from a visual mesh or a guessed fingertip size.  Subtracting it from
    link-origin clearance gives a conservative collision-geometry clearance.
    """
    stage = sim_utils.get_current_stage()
    cache = UsdGeom.XformCache()
    radii = []
    collider_paths = []
    for path in tip_paths:
        root = stage.GetPrimAtPath(path)
        origin_m = cache.GetLocalToWorldTransform(root)
        origin = np.array([float(origin_m[3][i]) for i in range(3)])
        radius = 0.0
        paths = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            paths.append(prim.GetPath().pathString)
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() if mesh else None
            if not points:
                raise RuntimeError(
                    f"unsupported non-mesh fingertip collider {prim.GetPath()}"
                )
            transform = cache.GetLocalToWorldTransform(prim)
            for point in points:
                world = np.asarray(transform.Transform(Gf.Vec3d(point)), dtype=float)
                if not np.all(np.isfinite(world)):
                    raise RuntimeError(
                        f"non-finite authored collider vertex at {prim.GetPath()}"
                    )
                radius = max(radius, float(np.linalg.norm(world - origin)))
        if not paths or radius <= 0:
            raise RuntimeError(f"no collision geometry below fingertip {path}")
        radii.append(radius)
        collider_paths.append(paths)
    return radii, collider_paths


class NativeGrade:
    def __init__(self, base, scene):
        self.base, self.scene = base, scene
        self.passage = Passage(scene)
        self.step = 0
        self.dwell = 0
        self.events = {}
        self.recovery = 0
        self.score = 0.0
        self.max_score = 0.0
        names, _ = tip_paths_and_positions(base)
        self.tip_radii, _ = tip_collision_bounding_radii(names)
        self.support = base.scene.sensors["screen_support_contacts"]
        self.dt = base.sim.get_physics_dt()
        self.outcome = Outcome(self.dt)
        self.last = {}
        self.report_counts = []
        self.history = []
        self.washer_local = np.array(
            [
                [scene.WASHER_OD / 2 * np.cos(a), scene.WASHER_OD / 2 * np.sin(a), z]
                for a in np.arange(32) * 2 * np.pi / 32
                for z in (-scene.WASHER_T / 2, scene.WASHER_T / 2)
            ]
        )

    def observe(self):
        self.step += 1
        s = self.scene
        poses = {
            n: b.root_view.get_transforms().numpy()[0].copy()
            for n, b in self.base.scene.rigid_objects.items()
        }
        self.history.append(np.stack(list(poses.values())).astype(np.float32))
        self.body_names = list(poses)
        self.passage.advance(poses, self.step)
        screen = poses["screen"]
        basis = rotation(screen)
        center = np.array([s.SCREEN_X, s.SCREEN_Y])
        top = s.TABLETOP_HEIGHT + 0.001 + s.RECEIVER_FLOOR_T + s.RECEIVER_WALL_H
        for name, body in poses.items():
            if name == "screen":
                continue
            is_washer = name.startswith("washer")
            vertices = (
                self.washer_local if is_washer else self.passage.media
            ) @ rotation(body).T + body[:3]
            local = (vertices - screen[:3]) @ basis
            if name not in self.passage.passed:
                if np.linalg.norm(local[:, :2], axis=1).max() > s.INNER_R + 0.0005 or (
                    is_washer and local[:, 2].max() < -s.FLOOR_T - 0.0005
                ):
                    self.events.setdefault(
                        name, "left the screen rim or passed below its floor"
                    )
            if is_washer:
                inside_receiver = np.all(
                    np.abs(vertices[:, :2] - center) < s.RECEIVER_INNER_XY / 2 - 0.0005,
                    axis=1,
                )
                if np.any(inside_receiver & (vertices[:, 2] < top - 0.0005)):
                    self.events.setdefault(name, "washer entered receiver")
            if (
                name in self.passage.passed
                and vertices[:, 2].max() < top
                and np.abs(vertices[:, :2] - center).max()
                > s.RECEIVER_INNER_XY / 2 + 0.0005
            ):
                self.events.setdefault(name, "missed receiver")
        self.events.update(self.passage.invalid)
        r = len(self.passage.received) / 8
        if r != self.recovery:
            self.report_counts.append((self.step, r))
        self.recovery = r
        supported = released = False
        if r >= 0.98 and not self.events:
            self.support.update(self.dt, force_recompute=True)
            forces = (
                self.support.data.normal_force_matrix_w.torch[0, 0]
                .cpu()
                .numpy()
                .reshape(2, 2, 3)
                .sum(axis=1)
            )
            supported = (
                np.all(np.linalg.norm(forces, axis=1) > 1e-8)
                and abs(screen[2] - s.SCREEN_ROOT_Z) <= 0.002
                and basis[2, 2] >= np.cos(np.deg2rad(3))
                and 0.04 <= screen[0] <= 0.56
                and abs(screen[1]) <= 0.1
            )
            if supported:
                # Conservative lower bound on clearance to the complete screen and payload.
                # Enclose the sieve in its local bounding box, including rim and handles.
                _, tips = tip_paths_and_positions(self.base)
                half = np.array(
                    [s.OUTER_R, s.TAB_Y + s.TAB_SIZE[1] / 2, (s.DEPTH + s.FLOOR_T) / 2]
                )
                local = (np.asarray(tips) - screen[:3]) @ basis - np.array(
                    [0, 0, (s.DEPTH - s.FLOOR_T) / 2]
                )
                dist = (
                    np.linalg.norm(np.maximum(np.abs(local) - half, 0), axis=1)
                    - self.tip_radii
                )
                released = bool(
                    np.min(dist) >= 0.060
                    and np.min(np.asarray(tips)[:, 2] - self.tip_radii - top) >= 0.060
                )
        self.score = self.outcome.advance(r, supported, released, bool(self.events))
        self.dwell = self.outcome.dwell_steps
        self.max_score = max(self.max_score, self.score)
        self.last = {
            "step": self.step,
            "score": self.score,
            "received": sorted(self.passage.received),
            "passages": self.passage.passed,
            "faults": self.events,
            "dwell_s": self.dwell * self.dt,
            "changes": self.report_counts,
        }
