"""Conservative box proxies for YAM/deformable contact in Newton experiments.

Original robot collision meshes retain rigid-body collision; only their soft
particle collision is replaced by one local AABB per affected link. This is a
coarse approximation and must be validated against gripper/bag manipulation.
"""
import numpy as np
import warp as wp
import newton
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull


def add_yam_soft_proxies(builder, kind="box"):
    if kind not in ("box", "hull"):
        raise ValueError("unknown proxy kind")
    groups = {}
    for i, body in enumerate(builder.shape_body):
        if body < 0 or not any(f'/World/{side}_robot/' in builder.body_label[body] for side in ('left','right')):
            continue
        if not int(builder.shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_PARTICLES):
            continue
        src = builder.shape_source[i]
        if src is None or not hasattr(src, 'vertices'):
            continue  # analytic primitives already support full-surface contact
        vertices = np.asarray(src.vertices, dtype=float) * np.asarray(builder.shape_scale[i])
        transform = np.asarray(builder.shape_transform[i],dtype=float)
        local = Rotation.from_quat(transform[3:7]).apply(vertices) + transform[:3]
        if not len(local) or not np.isfinite(local).all():
            raise ValueError(f'invalid collision vertices for shape {i}')
        lo, hi = local.min(axis=0), local.max(axis=0)
        if body in groups:
            groups[body][0] = np.minimum(groups[body][0], lo)
            groups[body][1] = np.maximum(groups[body][1], hi)
            groups[body][2].append(local)
        else:
            groups[body] = [lo, hi, [local]]
        builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
    if not groups:
        raise RuntimeError('no YAM collision mesh shapes found for soft proxies')
    result=[]
    for body,(lo,hi,point_groups) in groups.items():
        center=(lo+hi)/2; half=(hi-lo)/2
        if np.min(half) < 1e-5:
            raise ValueError(f'degenerate proxy for {builder.body_label[body]}')
        cfg = newton.ModelBuilder.ShapeConfig(density=0., has_shape_collision=False,
                has_particle_collision=True, is_visible=False, ke=2500.,kd=100.,mu=1., gap=.0005)
        if kind == "box":
            builder.add_shape_box(body, xform=wp.transform(wp.vec3(*center),wp.quat_identity()),
                                  hx=float(half[0]),hy=float(half[1]),hz=float(half[2]),cfg=cfg,
                                  label=f'{builder.body_label[body]}/soft_proxy')
        else:
            points = np.concatenate(point_groups)
            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            tri = points[faces]
            normal = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
            inward = np.sum(normal*(tri.mean(axis=1)-points[hull.vertices].mean(axis=0)),axis=1) < 0
            faces[inward] = faces[inward][:,[0,2,1]]
            used, remapped = np.unique(faces, return_inverse=True)
            mesh = newton.Mesh(points[used].astype(np.float32), remapped.astype(np.int32), compute_inertia=False)
            cfg.configure_sdf(force_sdf=True)
            builder.add_shape_mesh(body, mesh=mesh, cfg=cfg, label=f'{builder.body_label[body]}/soft_proxy')
        result.append({'body':builder.body_label[body], 'kind':kind, 'local_min':lo.tolist(),'local_max':hi.tolist()})
    return result
