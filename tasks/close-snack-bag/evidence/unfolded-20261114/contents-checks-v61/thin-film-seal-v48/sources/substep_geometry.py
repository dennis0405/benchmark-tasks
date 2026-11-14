"""Observe every solved substep; never move a particle or rigid object.

Strict transverse self-crossings and film/BOX/capsule/sphere intersections.
Coplanar overlap, other colliders and within-substep motion are outside scope.
"""
import numpy as np
import newton
import warp as wp
from gpu_contact_checks import _box_hit, self_crossings, stamp
from round_contact_checks import scene_round,scene_ellipsoids


@wp.kernel
def scene_boxes(points:wp.array[wp.vec3],faces:wp.array[wp.vec3i],
                body_q:wp.array[wp.transform],owners:wp.array[int],
                local:wp.array[wp.transform],scale:wp.array[wp.vec3],
                ids:wp.array[int],nfaces:int,count:wp.array[int]):
    tid=wp.tid()
    face=faces[tid%nfaces]
    sid=ids[tid//nfaces]
    transform=local[sid]
    if owners[sid]>=0:
        transform=wp.transform_multiply(body_q[owners[sid]],transform)
    inverse=wp.transform_inverse(transform)
    hit=_box_hit(wp.transform_point(inverse,points[face[0]]),
                 wp.transform_point(inverse,points[face[1]]),
                 wp.transform_point(inverse,points[face[2]]),scale[sid]-wp.vec3(1.e-7))
    if hit:
        wp.atomic_add(count,sid,1)


class SubstepGeometry:
    def __init__(self,env,entity):
        self.model=env.model
        self.device=env.model.device
        faces=np.asarray(entity.particles)[np.asarray(entity.faces,dtype=np.int32)]
        edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        self.faces=wp.array(faces,dtype=wp.vec3i,device=self.device)
        self.edges=wp.array(edges,dtype=wp.vec2i,device=self.device)
        indices=wp.array(faces.ravel(),dtype=int,device=self.device)
        self.meshes={state.particle_q.ptr:wp.Mesh(points=state.particle_q,indices=indices)
                     for state in (env.pipeline.state_0,env.pipeline.state_1)}
        mask=(self.model.shape_type.numpy()==int(newton.GeoType.BOX)) & (
            (self.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_PARTICLES))!=0)
        self.ids=wp.array(np.flatnonzero(mask).astype(np.int32),dtype=int,device=self.device)
        self.self_count=wp.zeros(1,dtype=int,device=self.device)
        self.box_count=wp.zeros(self.model.shape_count,dtype=int,device=self.device)
        round_mask=np.isin(self.model.shape_type.numpy(),[int(newton.GeoType.CAPSULE),int(newton.GeoType.SPHERE)]) & (
            (self.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_PARTICLES))!=0)
        self.round_ids=wp.array(np.flatnonzero(round_mask).astype(np.int32),dtype=int,device=self.device)
        self.round_count=wp.zeros(self.model.shape_count,dtype=int,device=self.device)
        ellipsoid_mask=(self.model.shape_type.numpy()==int(newton.GeoType.ELLIPSOID)) & (
            (self.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_PARTICLES))!=0)
        self.ellipsoid_ids=wp.array(np.flatnonzero(ellipsoid_mask).astype(np.int32),dtype=int,device=self.device)
        self.ellipsoid_count=wp.zeros(self.model.shape_count,dtype=int,device=self.device)
        self.checked=wp.zeros(1,dtype=int,device=self.device)
        self.flags=wp.zeros(len(edges),dtype=int,device=self.device)

    def check(self,state,dt):
        del dt
        mesh=self.meshes[state.particle_q.ptr]
        mesh.refit()
        wp.launch(self_crossings,len(self.edges),inputs=[state.particle_q,self.faces,self.edges,
            mesh.id,0,self.self_count,self.flags],device=self.device)
        if len(self.ids):
            wp.launch(scene_boxes,len(self.faces)*len(self.ids),inputs=[state.particle_q,self.faces,
                state.body_q,self.model.shape_body,self.model.shape_transform,self.model.shape_scale,
                self.ids,len(self.faces),self.box_count],device=self.device)
        if len(self.round_ids):
            wp.launch(scene_round,len(self.round_ids),inputs=[state.particle_q,self.faces,mesh.id,
                state.body_q,self.model.shape_body,self.model.shape_transform,self.model.shape_scale,
                self.model.shape_type,self.round_ids,self.round_count],device=self.device)
        if len(self.ellipsoid_ids):
            wp.launch(scene_ellipsoids,len(self.ellipsoid_ids),inputs=[state.particle_q,self.faces,mesh.id,
                state.body_q,self.model.shape_body,self.model.shape_transform,self.model.shape_scale,
                self.ellipsoid_ids,self.ellipsoid_count],device=self.device)
        wp.launch(stamp,1,inputs=[0,self.checked],device=self.device)

    def report(self):
        box_counts=self.box_count.numpy()
        round_counts=self.round_count.numpy()
        ellipsoid_counts=self.ellipsoid_count.numpy()
        return dict(checked_substeps=int(self.checked.numpy()[0]),
                    transverse_edge_crossings=int(self.self_count.numpy()[0]),
                    film_box_intersections=int(box_counts.sum()),
                    checked_round_shapes=len(self.round_ids),
                    film_round_intersections=int(round_counts.sum()),
                    checked_ellipsoid_shapes=len(self.ellipsoid_ids),
                    film_ellipsoid_intersections=int(ellipsoid_counts.sum()),
                    hit_ellipsoid_shapes=[dict(shape=int(i),label=str(self.model.shape_label[i]),
                        intersections=int(ellipsoid_counts[i])) for i in np.flatnonzero(ellipsoid_counts)],
                    hit_round_shapes=[dict(shape=int(i),label=str(self.model.shape_label[i]),
                                    intersections=int(round_counts[i])) for i in np.flatnonzero(round_counts)],
                    hit_boxes=[dict(shape=int(i),label=str(self.model.shape_label[i]),
                                    intersections=int(box_counts[i])) for i in np.flatnonzero(box_counts)])
