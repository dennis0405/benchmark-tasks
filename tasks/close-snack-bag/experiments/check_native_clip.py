"""Sampled clip-box intrusion checks using triangle/box separating axes."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def penetrating_triangles(q,faces,half,tolerance=1e-7):
    half=np.asarray(half)-tolerance
    tri=q[faces]
    valid=np.all(tri.min(1)<half,axis=1)&np.all(tri.max(1)>-half,axis=1)
    ids=np.flatnonzero(valid);v=tri[ids]
    if not len(ids):return ids
    edges=np.stack((v[:,1]-v[:,0],v[:,2]-v[:,1],v[:,0]-v[:,2]),axis=1)
    axes=[np.cross(edges[:,0],edges[:,1])]
    for j in range(3):
        for axis in np.eye(3):axes.append(np.cross(edges[:,j],axis))
    initial_ids=ids
    survives=np.ones(len(v),dtype=bool)
    for axis in axes:
        r=np.abs(axis)@half;proj=np.einsum("nij,nj->ni",v,axis)
        survives&=(proj.min(1)<=r)&(proj.max(1)>=-r)
    return initial_ids[survives]

def sanity():
    q=np.array([[-2,0,0],[2,0,0],[0,2,0]],float);f=np.array([[0,1,2]])
    assert len(penetrating_triangles(q,f,[.5,.5,.5]))==1
    assert len(penetrating_triangles(q+10,f,[.5,.5,.5]))==0
    q[:,2]=.5
    assert len(penetrating_triangles(q,f,[.5,.5,.5]))==0

def inspect_state(q,faces,body,shape):
    boxes=[]
    for i,(owner,loc,half) in enumerate(zip(shape["shape_body"],shape["shape_local"],shape["shape_scale"])):
        if owner<0:continue
        br=Rotation.from_quat(body[owner,3:]).as_matrix()
        sr=Rotation.from_quat(loc[3:]).as_matrix()
        pos=body[owner,:3]+br@loc[:3];rot=br@sr
        hits=penetrating_triangles((q-pos)@rot,faces,half)
        boxes.append({"shape":i,"body":owner,"penetrating_triangles":len(hits)})
    return boxes

if __name__=="__main__":
    sanity();folder=Path(sys.argv[1]);data=np.load(folder/"pouch-snapshots.npz")
    shape=json.loads((folder/"clip-shapes.json").read_text())
    reports=[]
    for key in sorted(k for k in data.files if k.startswith("q_")):
        reports.append({"snapshot":key,"boxes":inspect_state(data[key],data["faces"],data["b_"+key[2:]],shape)})
    summary={"sanity_checks_passed":True,"limitation":"sampled triangle/box intrusion with 0.1 micrometre boundary tolerance; not continuous collision detection",
        "sampled_states":len(reports),"states_with_intrusion":sum(any(b["penetrating_triangles"] for b in r["boxes"]) for r in reports),"rows":reports}
    (folder/"clip-intrusions.json").write_text(json.dumps(summary,indent=2))
    print("CLIP_INTRUSIONS",summary["states_with_intrusion"],"/",len(reports),flush=True)
