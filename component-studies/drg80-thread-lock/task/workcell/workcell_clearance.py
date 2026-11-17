"""A conservative certificate for tool/arm clearance from the mounted fixture."""

import numpy as np
from scipy.spatial.transform import Rotation


class WorkcellClearance:
    def __init__(self, world, stage, fixture, tools, arms):
        from pxr import Usd, UsdGeom, UsdPhysics

        moving = set(tools)
        for links in arms:
            moving.update(p for p in links if p.rsplit("/", 1)[-1] != "base")
        equipment = {
            fixture + "/" + name
            for name in (
                "frame",
                "screw",
                "nut",
                "retainer",
                "piston",
                "left_jaw",
                "right_jaw",
                "lever",
            )
        }
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        names = sorted(moving | equipment)
        self.view = world.create_rigid_body_view(names)
        self.paths = list(self.view.prim_paths)
        if set(names) != set(self.paths):
            raise ValueError("clearance observer missed a moving body")
        indices = {path: i for i, path in enumerate(self.paths)}
        actors = {str(p.GetPath()) for p in bodies}
        ordered = sorted(actors, key=len, reverse=True)
        c = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=False,
            ignoreVisibility=True,
        )
        points = []
        sizes = []
        owner = []
        for p in stage.Traverse():
            if (
                not p.HasAPI(UsdPhysics.CollisionAPI)
                or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            ):
                continue
            path = str(p.GetPath())
            body = next(
                (
                    root
                    for root in ordered
                    if path == root or path.startswith(root + "/")
                ),
                None,
            )
            if body not in indices:
                continue
            bounds = c.ComputeRelativeBound(
                p, stage.GetPrimAtPath(body)
            ).ComputeAlignedRange()
            low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
            if not np.isfinite([low, high]).all() or np.any(high < low):
                raise ValueError(f"missing active collider bounds {path}")
            owner.append(indices[body])
            points.append((low + high) / 2)
            sizes.append((high - low) / 2)
        self.owner = np.array(owner)
        if set(self.owner) != set(indices.values()):
            raise ValueError("a required body has no active collision geometry")
        self.points, self.sizes = np.array(points), np.array(sizes)
        self.by_body = [np.flatnonzero(self.owner == i) for i in range(len(self.paths))]
        self.moving = [indices[p] for p in sorted(moving)]
        self.equipment = [indices[p] for p in sorted(equipment)]

    def lower_bound(self):
        poses = self.view.get_transforms().numpy()
        rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()[self.owner]
        centres = (
            np.einsum("nij,nj->ni", rotations, self.points) + poses[self.owner, :3]
        )
        half = np.einsum("nij,nj->ni", abs(rotations), self.sizes)
        lo, hi = centres - half, centres + half
        actor_lo = np.array([lo[ids].min(axis=0) for ids in self.by_body])
        actor_hi = np.array([hi[ids].max(axis=0) for ids in self.by_body])
        best = float("inf")
        for i in self.moving:
            delta = np.maximum(
                actor_lo[self.equipment] - actor_hi[i],
                actor_lo[i] - actor_hi[self.equipment],
            )
            rough = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
            for j, k in sorted(zip(self.equipment, rough), key=lambda pair: pair[1]):
                if k >= best:
                    break
                ai, bi = self.by_body[i], self.by_body[j]
                gaps = np.maximum(lo[ai, None] - hi[bi], lo[bi] - hi[ai, None])
                best = min(
                    best, float(np.linalg.norm(np.maximum(gaps, 0.0), axis=2).min())
                )
                if best == 0.0:
                    return best
        return best
