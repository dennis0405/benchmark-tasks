"""Detect strict, transverse edge/triangle crossings; excludes coplanar overlap."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def crossings(q, faces):
    tri = q[faces]
    centers = tri.mean(axis=1)
    radii = np.linalg.norm(tri-centers[:,None], axis=2).max(axis=1)
    tree = cKDTree(centers)
    edges = np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])), axis=1),axis=0)
    start,end = q[edges[:,0]],q[edges[:,1]]
    candidates = tree.query_ball_point((start+end)/2, np.linalg.norm(end-start,axis=1)/2+radii.max()+1e-9)
    hits = []
    for edge,a,b,candidate in zip(edges,start,end,candidates):
        ids = np.asarray(candidate,dtype=int)
        ids = ids[~np.any(np.isin(faces[ids],edge),axis=1)]
        if not len(ids): continue
        v0,v1,v2 = tri[ids,0],tri[ids,1],tri[ids,2]
        e1,e2,d = v1-v0,v2-v0,b-a
        h = np.cross(np.broadcast_to(d,e2.shape),e2)
        det = np.einsum("ij,ij->i",e1,h)
        valid = np.abs(det)>1e-14
        inv=np.divide(1,det,out=np.zeros_like(det),where=valid)
        s=a-v0
        u=inv*np.einsum("ij,ij->i",s,h)
        w=np.cross(s,e1)
        v=inv*np.einsum("j,ij->i",d,w)
        t=inv*np.einsum("ij,ij->i",e2,w)
        eps=1e-7
        ok=valid & (u>eps)&(v>eps)&(u+v<1-eps)&(t>eps)&(t<1-eps)
        hits.extend((int(edge[0]),int(edge[1]),int(i)) for i in ids[ok])
    return hits

def sanity():
    q=np.array([[-1,-1,0],[1,-1,0],[0,1,0],[0,0,-1],[0,0,1],[.2,0,1]],float)
    f=np.array([[0,1,2],[3,4,5]])
    assert crossings(q,f), "missed a strict crossing with no vertex inside the other triangle"
    q[3:,0]+=4
    assert not crossings(q,f), "false crossing for separated triangles"

if __name__=="__main__":
    sanity()
    folder=Path(sys.argv[1])
    reports={}
    for path in sorted(folder.glob("*-snapshots.npz")):
        data=np.load(path); faces=data["faces"]
        rows=[]
        for key in sorted(k for k in data.files if k.startswith("q_")):
            hits=crossings(data[key],faces)
            rows.append({"snapshot":key,"strict_crossings":len(hits),"first_hits":hits[:10]})
        reports[path.stem]={"snapshots":len(rows),"frames_with_crossings":sum(r["strict_crossings"]>0 for r in rows),"rows":rows}
        print(path.name,"frames",len(rows),"crossed_frames",reports[path.stem]["frames_with_crossings"],flush=True)
    (folder/"transverse-crossings.json").write_text(json.dumps({"limitation":"sampled strict transverse crossings; coplanar overlaps and between-sample events are not covered","sanity_checks_passed":True,"reports":reports},indent=2))
