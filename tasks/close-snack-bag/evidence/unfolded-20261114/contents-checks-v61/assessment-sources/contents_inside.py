"""Read-only fragment-centre containment in the measured pouch surface.

A temporary triangle fan closes the open mouth for classification only. It
never enters the physical model. Three forward-facing rays vote on parity;
the rays point toward the sealed end to avoid the folded mouth when possible.
"""
import numpy as np


def contained_centres(points,faces,grid,centres):
    q=np.asarray(points,dtype=float);centres=np.asarray(centres,dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(centres).all():
        return np.zeros(len(centres),dtype=bool)
    ring=np.r_[grid[0,-1,:],grid[1,-1,::-1]].tolist()
    ring=[v for i,v in enumerate(ring) if i==0 or v!=ring[i-1]]
    if ring[0]==ring[-1]:ring.pop()
    cap=len(q);q=np.vstack([q,q[ring].mean(axis=0)])
    cap_faces=np.asarray([(cap,ring[i],ring[(i+1)%len(ring)]) for i in range(len(ring))])
    triangles=q[np.vstack([faces,cap_faces])]
    axis=q[grid[:,0,:]].mean(axis=(0,1))-q[grid[:,-1,:]].mean(axis=(0,1))
    if np.linalg.norm(axis)<1e-8:return np.zeros(len(centres),dtype=bool)
    axis/=np.linalg.norm(axis)
    reference=np.array([0.,0.,1.]) if abs(axis[2])<.9 else np.array([0.,1.,0.])
    u=np.cross(axis,reference);u/=np.linalg.norm(u);v=np.cross(axis,u)
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    offset=centres[:,None,:]-a[None,:,:]
    votes=np.zeros(len(centres),dtype=int)
    for du,dv in ((.123,.231),(-.137,.193),(.219,-.113)):
        direction=axis+du*u+dv*v;direction/=np.linalg.norm(direction)
        h=np.cross(direction,e2);det=(e1*h).sum(axis=1)
        valid=abs(det)>1e-12;inverse=np.zeros_like(det);inverse[valid]=1/det[valid]
        bary_u=(offset*h).sum(axis=2)*inverse
        cross=np.cross(offset,e1);bary_v=(cross*direction).sum(axis=2)*inverse
        distance=(cross*e2).sum(axis=2)*inverse
        hit=valid & (bary_u>=0.) & (bary_v>=0.) & (bary_u+bary_v<=1.) & (distance>1e-7)
        votes+=(hit.sum(axis=1)%2).astype(int)
    return votes>=2
