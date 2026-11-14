"""Independent geometry checks at physics-step cadence, not solver contact flags.

Triangle/box SAT and strict segment/triangle crossings. Coplanar self-overlap
and motion between steps are outside these tests' scope. Warp BVH API follows
warp/tests/geometry/test_mesh_query_aabb.py at Warp 1.18.
"""
import numpy as np
import warp as wp


@wp.kernel
def stamp(slot: int, checked: wp.array[int]):
    checked[slot] += 1


@wp.func
def _lo(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def _hi(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def _separates(a: wp.vec3, b: wp.vec3, c: wp.vec3, h: wp.vec3, axis: wp.vec3):
    r = wp.dot(wp.abs(axis), h)
    pa, pb, pc = wp.dot(a, axis), wp.dot(b, axis), wp.dot(c, axis)
    return wp.min(wp.min(pa, pb), pc) > r or wp.max(wp.max(pa, pb), pc) < -r


@wp.func
def _box_hit(a: wp.vec3, b: wp.vec3, c: wp.vec3, h: wp.vec3):
    lower, upper = _lo(_lo(a, b), c), _hi(_hi(a, b), c)
    for j in range(3):
        if lower[j] >= h[j] or upper[j] <= -h[j]:
            return False
    e0, e1, e2 = b-a, c-b, a-c
    if _separates(a, b, c, h, wp.cross(e0, e1)):
        return False
    for j in range(3):
        edge = e0
        if j == 1:
            edge = e1
        elif j == 2:
            edge = e2
        for k in range(3):
            axis = wp.vec3(0.)
            axis[k] = 1.
            if _separates(a, b, c, h, wp.cross(edge, axis)):
                return False
    return True


@wp.kernel
def rigid_boxes(body_q: wp.array[wp.transform], shape_body: wp.array[int],
                shape_tf: wp.array[wp.transform], shape_scale: wp.array[wp.vec3],
                pairs: wp.array[wp.vec2i], slot: int, counts: wp.array[int], flags: wp.array[int]):
    tid = wp.tid()
    pair = pairs[tid]
    ta = wp.transform_multiply(body_q[shape_body[pair[0]]], shape_tf[pair[0]])
    tb = wp.transform_multiply(body_q[shape_body[pair[1]]], shape_tf[pair[1]])
    rel = wp.transform_multiply(wp.transform_inverse(tb), ta)
    center, rotation = wp.transform_get_translation(rel), wp.transform_get_rotation(rel)
    ha, hb = shape_scale[pair[0]]-wp.vec3(1.e-7), shape_scale[pair[1]]-wp.vec3(1.e-7)
    a0 = wp.quat_rotate(rotation, wp.vec3(1., 0., 0.))
    a1 = wp.quat_rotate(rotation, wp.vec3(0., 1., 0.))
    a2 = wp.quat_rotate(rotation, wp.vec3(0., 0., 1.))
    overlap = True
    for j in range(15):
        axis = wp.vec3(0.)
        if j < 3:
            axis[j] = 1.
        elif j < 6:
            axis = a0
            if j == 4:axis = a1
            elif j == 5:axis = a2
        else:
            k = (j-6)//3
            edge = a0
            if k == 1:edge = a1
            elif k == 2:edge = a2
            base = wp.vec3(0.)
            base[(j-6)%3] = 1.
            axis = wp.cross(edge, base)
        if wp.dot(axis, axis) > 1.e-12:
            radius = ha[0]*wp.abs(wp.dot(a0, axis))+ha[1]*wp.abs(wp.dot(a1, axis))+ha[2]*wp.abs(wp.dot(a2, axis))+wp.dot(hb, wp.abs(axis))
            if wp.abs(wp.dot(center, axis)) >= radius:
                overlap = False
    flags[tid] = int(overlap)
    if overlap:wp.atomic_add(counts, slot, 1)


@wp.kernel
def plane_points(points: wp.array[wp.vec3], slot: int, counts: wp.array[int]):
    if points[wp.tid()][2] < -1.e-7:
        wp.atomic_add(counts, slot, 1)


@wp.kernel
def plane_boxes(body_q: wp.array[wp.transform], shape_body: wp.array[int],
                shape_tf: wp.array[wp.transform], shape_scale: wp.array[wp.vec3],
                box_ids: wp.array[int], slot: int, counts: wp.array[int]):
    tid = wp.tid()
    sid, corner = box_ids[tid//8], tid%8
    sign = wp.vec3(float(2*(corner%2)-1), float(2*((corner//2)%2)-1), float(2*(corner//4)-1))
    point = wp.cw_mul(shape_scale[sid], sign)
    transform = wp.transform_multiply(body_q[shape_body[sid]], shape_tf[sid])
    if wp.transform_point(transform, point)[2] < -1.e-7:
        wp.atomic_add(counts, slot, 1)


@wp.kernel
def boxes(points: wp.array[wp.vec3], faces: wp.array[wp.vec3i],
          body_q: wp.array[wp.transform], shape_body: wp.array[int],
          shape_tf: wp.array[wp.transform], shape_scale: wp.array[wp.vec3],
          box_ids: wp.array[int], nfaces: int, slot: int, tolerance: float,
          counts: wp.array[int], flags: wp.array[int]):
    tid = wp.tid()
    face = faces[tid % nfaces]
    sid = box_ids[tid // nfaces]
    transform = wp.transform_inverse(wp.transform_multiply(body_q[shape_body[sid]], shape_tf[sid]))
    a = wp.transform_point(transform, points[face[0]])
    b = wp.transform_point(transform, points[face[1]])
    c = wp.transform_point(transform, points[face[2]])
    hit = _box_hit(a, b, c, shape_scale[sid]-wp.vec3(tolerance))
    flags[tid] = int(hit)
    if hit:
        wp.atomic_add(counts, slot, 1)


@wp.func
def _segment_hit(a: wp.vec3, b: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    e1, e2, d = v1-v0, v2-v0, b-a
    h = wp.cross(d, e2)
    det = wp.dot(e1, h)
    if wp.abs(det) < 1.e-14:
        return False
    inv = 1./det
    s = a-v0
    u = inv*wp.dot(s, h)
    w = wp.cross(s, e1)
    v, t = inv*wp.dot(d, w), inv*wp.dot(e2, w)
    return u > 1.e-7 and v > 1.e-7 and u+v < 1.-1.e-7 and t > 1.e-7 and t < 1.-1.e-7


@wp.kernel
def self_crossings(points: wp.array[wp.vec3], faces: wp.array[wp.vec3i],
                   edges: wp.array[wp.vec2i], mesh: wp.uint64, slot: int,
                   counts: wp.array[int], flags: wp.array[int]):
    tid = wp.tid()
    edge = edges[tid]
    a, b = points[edge[0]], points[edge[1]]
    query = wp.mesh_query_aabb(mesh, _lo(a, b)-wp.vec3(1.e-8), _hi(a, b)+wp.vec3(1.e-8))
    hits = int(0)
    for face_id in query:
        face = faces[face_id]
        adjacent = False
        for j in range(3):
            if face[j] == edge[0] or face[j] == edge[1]:
                adjacent = True
        if not adjacent and _segment_hit(a, b, points[face[0]], points[face[1]], points[face[2]]):
            hits += 1
    flags[tid] = hits
    if hits:
        wp.atomic_add(counts, slot, hits)


class GeometryChecks:
    def __init__(self, model, states, faces, steps, ground_plane=False):
        self.model, self.device = model, model.device
        self.ground_plane = ground_plane
        faces = np.asarray(faces, dtype=np.int32)
        edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
        self.faces = wp.array(faces, dtype=wp.vec3i, device=self.device)
        self.indices = wp.array(faces.ravel(), dtype=int, device=self.device)
        self.edges = wp.array(edges, dtype=wp.vec2i, device=self.device)
        self.meshes = {state.particle_q.ptr: wp.Mesh(points=state.particle_q, indices=self.indices) for state in states}
        owners = model.shape_body.numpy()
        self.box_ids = wp.array(np.flatnonzero(owners >= 0).astype(np.int32), dtype=int, device=self.device)
        pair_ids=[(i,j) for i in np.flatnonzero(owners>=0) for j in np.flatnonzero(owners>=0) if i<j and owners[i]!=owners[j]]
        self.rigid_pairs=wp.array(np.asarray(pair_ids,dtype=np.int32).reshape(-1,2),dtype=wp.vec2i,device=self.device)
        self.rigid_counts=wp.zeros(steps,dtype=int,device=self.device)
        self.rigid_flags=wp.zeros(len(pair_ids),dtype=int,device=self.device)
        self.box_counts, self.self_counts = (wp.zeros(steps, dtype=int, device=self.device) for _ in range(2))
        self.ground_counts = wp.zeros(steps, dtype=int, device=self.device)
        self.checked_counts = wp.zeros(steps, dtype=int, device=self.device)
        self.box_flags = wp.zeros(len(faces)*len(self.box_ids), dtype=int, device=self.device)
        self.self_flags = wp.zeros(len(edges), dtype=int, device=self.device)

    def check(self, state, slot):
        if len(self.rigid_pairs):
            wp.launch(rigid_boxes,dim=len(self.rigid_pairs),inputs=[state.body_q,self.model.shape_body,
                self.model.shape_transform,self.model.shape_scale,self.rigid_pairs,slot,self.rigid_counts,self.rigid_flags],device=self.device)
        if self.ground_plane:
            wp.launch(plane_points, dim=len(state.particle_q), inputs=[state.particle_q, slot, self.ground_counts], device=self.device)
            if len(self.box_ids):
                wp.launch(plane_boxes, dim=8*len(self.box_ids), inputs=[state.body_q, self.model.shape_body,
                    self.model.shape_transform, self.model.shape_scale, self.box_ids, slot, self.ground_counts], device=self.device)
        if len(self.box_ids):
            wp.launch(boxes, dim=len(self.box_flags), inputs=[state.particle_q, self.faces, state.body_q,
                self.model.shape_body, self.model.shape_transform, self.model.shape_scale, self.box_ids,
                len(self.faces), slot, 1.e-7, self.box_counts, self.box_flags], device=self.device)
        mesh = self.meshes[state.particle_q.ptr]
        mesh.refit()
        wp.launch(self_crossings, dim=len(self.edges), inputs=[state.particle_q, self.faces, self.edges,
            mesh.id, slot, self.self_counts, self.self_flags], device=self.device)
        wp.launch(stamp,dim=1,inputs=[slot,self.checked_counts],device=self.device)
