"""Strict film penetration into capsule/sphere solids at solved snapshots.

Uses triangle/axis-segment distance, including interior face intersections.
Contact skins are not solid geometry. A 0.1 micrometre inset rejects numerical
surface equality, consistently with the BOX observer. Newton's pinned
triangle closest-point function is shared, not its contact flags.
"""
import warp as wp
from newton._src.geometry.kernels import triangle_closest_point
from gpu_contact_checks import _segment_hit, _lo, _hi


@wp.func
def segment_distance_sq(p:wp.vec3,q:wp.vec3,a:wp.vec3,b:wp.vec3):
    d1=q-p;d2=b-a;r=p-a
    aa=wp.dot(d1,d1);ee=wp.dot(d2,d2);ff=wp.dot(d2,r)
    ss=float(0.);tt=float(0.)
    if aa<=1.e-18:
        if ee>1.e-18:tt=wp.clamp(ff/ee,0.,1.)
    else:
        cc=wp.dot(d1,r)
        if ee<=1.e-18:
            ss=wp.clamp(-cc/aa,0.,1.)
        else:
            bb=wp.dot(d1,d2);denom=aa*ee-bb*bb
            if denom>1.e-18:ss=wp.clamp((bb*ff-cc*ee)/denom,0.,1.)
            tt=(bb*ss+ff)/ee
            if tt<0.:
                tt=0.;ss=wp.clamp(-cc/aa,0.,1.)
            elif tt>1.:
                tt=1.;ss=wp.clamp((bb-cc)/aa,0.,1.)
    delta=p+ss*d1-a-tt*d2
    return wp.dot(delta,delta)


@wp.func
def triangle_segment_distance_sq(a:wp.vec3,b:wp.vec3,c:wp.vec3,p:wp.vec3,q:wp.vec3):
    if _segment_hit(p,q,a,b,c):return float(0.)
    cp,_,__=triangle_closest_point(a,b,c,p)
    cq,_,__=triangle_closest_point(a,b,c,q)
    distance=wp.min(wp.dot(p-cp,p-cp),wp.dot(q-cq,q-cq))
    distance=wp.min(distance,segment_distance_sq(p,q,a,b))
    distance=wp.min(distance,segment_distance_sq(p,q,b,c))
    return wp.min(distance,segment_distance_sq(p,q,c,a))


@wp.kernel
def scene_round(points:wp.array[wp.vec3],faces:wp.array[wp.vec3i],mesh:wp.uint64,
                body_q:wp.array[wp.transform],owners:wp.array[int],
                local:wp.array[wp.transform],scale:wp.array[wp.vec3],
                kinds:wp.array[int],ids:wp.array[int],counts:wp.array[int]):
    sid=ids[wp.tid()]
    transform=local[sid]
    if owners[sid]>=0:transform=wp.transform_multiply(body_q[owners[sid]],transform)
    half=float(0.)
    if kinds[sid]==4:half=scale[sid][1] # Newton 1.7 CAPSULE
    p=wp.transform_point(transform,wp.vec3(0.,0.,-half))
    q=wp.transform_point(transform,wp.vec3(0.,0.,half))
    radius=scale[sid][0]-1.e-7
    query=wp.mesh_query_aabb(mesh,_lo(p,q)-wp.vec3(radius),_hi(p,q)+wp.vec3(radius))
    hits=int(0)
    for face_id in query:
        f=faces[face_id]
        distance=triangle_segment_distance_sq(points[f[0]],points[f[1]],points[f[2]],p,q)
        if distance<radius*radius:hits+=1
    if hits:wp.atomic_add(counts,sid,hits)
