"""Check recorded reset finger BOX colliders against static BOX fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import newton
import warp as wp
from gpu_contact_checks import rigid_boxes


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);a=p.parse_args()
    c=np.load(a.run/"colliders.npz")
    with np.load(a.run/"states.npz") as d:b=d["b_00000"]
    parts=json.loads((a.run/"render-geometry.json").read_text())
    fingers={r["body"] for r in parts if "finger" in r["name"] and r["body"]>=0}
    owners=c["body"].copy()
    eligible=(c["kind"]==int(newton.GeoType.BOX))&((c["flags"]&int(newton.ShapeFlags.COLLIDE_SHAPES))!=0)
    selected=np.flatnonzero(eligible&np.isin(owners,list(fingers)))
    fixtures=np.flatnonzero(eligible&(owners<0))
    pairs=np.asarray([(int(i),int(j)) for i in selected for j in fixtures],np.int32).reshape(-1,2)
    owners[owners<0]=len(b);b=np.vstack([b,[0,0,0,0,0,0,1]])
    counts=wp.zeros(1,dtype=int,device="cpu");flags=wp.zeros(len(pairs),dtype=int,device="cpu")
    if len(pairs):
        wp.launch(rigid_boxes,len(pairs),inputs=[wp.array(b,dtype=wp.transform,device="cpu"),
            wp.array(owners,dtype=int,device="cpu"),wp.array(c["local"],dtype=wp.transform,device="cpu"),
            wp.array(c["scale"],dtype=wp.vec3,device="cpu"),wp.array(pairs,dtype=wp.vec2i,device="cpu"),
            0,counts,flags],device="cpu")
    result=dict(scope=__doc__,finger_bodies=sorted(fingers),checked_pairs=len(pairs),
                intersections=pairs[np.flatnonzero(flags.numpy())].tolist(),
                limitation="Reset snapshot and BOX colliders only; excludes robot mesh/capsules and later motion",
                colliders_sha256=hashlib.sha256((a.run/"colliders.npz").read_bytes()).hexdigest())
    (a.run/"finger-fixture-preflight.json").write_text(json.dumps(result,indent=2));print(result)


if __name__=="__main__":main()
