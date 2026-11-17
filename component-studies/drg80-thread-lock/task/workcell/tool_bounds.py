"""Compute exact axis-aligned extents of each composed rigid tool, in metres."""

from itertools import product
import numpy as np


class ToolBounds:
    def __init__(self, usd):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(usd))
        root = stage.GetDefaultPrim()
        cache = UsdGeom.XformCache()
        points = []
        self.handle = None
        self.cylinders = []
        self.convex_parts = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            local, _ = cache.ComputeRelativeTransform(prim, root)
            if prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                vertices = product((-half, half), repeat=3)
            elif prim.IsA(UsdGeom.Mesh):
                vertices = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            elif prim.IsA(UsdGeom.Cylinder):
                cylinder = UsdGeom.Cylinder(prim)
                axial = "XYZ".index(cylinder.GetAxisAttr().Get())
                radius = float(cylinder.GetRadiusAttr().Get())
                half = float(cylinder.GetHeightAttr().Get()) / 2
                basis = np.array([local.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)])
                radial = np.delete(basis, axial, axis=0) * radius
                self.cylinders.append(
                    (
                        np.array(local.Transform(Gf.Vec3d(0))),
                        half * basis[axial],
                        radial,
                    )
                )
                continue
            else:
                raise ValueError(f"unhandled tool collider {prim.GetPath()}")
            part = np.array([tuple(local.Transform(Gf.Vec3d(*v))) for v in vertices])
            if prim.GetName() in ("Grip", "Handle"):
                self.handle = part
            self.convex_parts.append(part)
            points.extend(part)
        if not (points or self.cylinders):
            raise ValueError(f"no active collision geometry in {usd}")
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)

    def world(self, position, rotation):
        samples = list(self.points @ rotation.T)
        for centre, axis, radial in self.cylinders:
            c = rotation @ centre
            extent = np.abs(rotation @ axis) + np.linalg.norm(
                radial @ rotation.T, axis=0
            )
            samples.extend([c - extent, c + extent])
        return np.array([np.min(samples, axis=0), np.max(samples, axis=0)]) + position

    def in_cradle(self, position, rotation, cradle):
        bounds = self.world(position, rotation)
        return bool(np.all(bounds[0] >= cradle[0]) and np.all(bounds[1] <= cradle[1]))
