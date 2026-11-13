"""Sampled independent film/self and film/rigid-box crossing checks.

Does not cover coplanar overlap, between-sample events, or non-box robot
colliders. Never report this as complete continuous collision certification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import newton
import warp as wp
from gpu_contact_checks import boxes, self_crossings


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--stride",type=int,default=30)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    d=np.load(args.run/"states.npz")
    c=np.load(args.run/"colliders.npz")
    f=d["faces"].astype(np.int32)
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    device=args.device
    faces=wp.array(f,dtype=wp.vec3i,device=device)
    edge_ids=wp.array(edges,dtype=wp.vec2i,device=device)
    ids=np.flatnonzero((c["kind"]==int(newton.GeoType.BOX)) &
                       ((c["flags"] & int(newton.ShapeFlags.COLLIDE_PARTICLES))!=0)).astype(np.int32)
    owners=c["body"].copy()
    nbody=len(d["b_00000"])
    owners[owners<0]=nbody
    owner=wp.array(owners,dtype=int,device=device)
    local=wp.array(c["local"],dtype=wp.transform,device=device)
    scale=wp.array(c["scale"],dtype=wp.vec3,device=device)
    box_ids=wp.array(ids,dtype=int,device=device)
    q=wp.array(d["q_00000"],dtype=wp.vec3,device=device)
    mesh=wp.Mesh(points=q,indices=wp.array(f.ravel(),dtype=int,device=device))
    count_self=wp.zeros(1,dtype=int,device=device);count_box=wp.zeros(1,dtype=int,device=device)
    flags_self=wp.zeros(len(edges),dtype=int,device=device)
    flags_box=wp.zeros(len(f)*len(ids),dtype=int,device=device)
    keys=sorted(k for k in d.files if k.startswith("q_"))
    chosen=sorted(set(keys[::args.stride]+keys[-1:]))
    reports=[]
    for key in chosen:
        q.assign(d[key].astype(np.float32));mesh.refit();count_self.zero_();count_box.zero_()
        b=np.concatenate([d["b_"+key[2:]],np.array([[0,0,0,0,0,0,1]])]).astype(np.float32)
        body=wp.array(b,dtype=wp.transform,device=device)
        wp.launch(self_crossings,dim=len(edges),inputs=[q,faces,edge_ids,mesh.id,0,count_self,flags_self],device=device)
        if len(ids):
            wp.launch(boxes,dim=len(f)*len(ids),inputs=[q,faces,body,owner,local,scale,
                box_ids,len(f),0,1e-7,count_box,flags_box],device=device)
        row=dict(step=int(key[2:]),self_crossings=int(count_self.numpy()[0]),
                 box_intersections=int(count_box.numpy()[0]))
        reports.append(row);print(row,flush=True)
    report=dict(scope=__doc__.strip(),box_shape_ids=ids.tolist(),samples=reports,
                passed=all(row["self_crossings"]==row["box_intersections"]==0 for row in reports))
    (args.run/"sampled-geometry.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
