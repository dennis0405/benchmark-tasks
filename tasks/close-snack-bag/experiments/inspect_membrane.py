"""Report measured edge stretch in saved material coordinates."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path)
    p.add_argument("--steps",type=int,nargs="+",default=None)
    a=p.parse_args();data=np.load(a.run/"states.npz");q0=data["q_00000"];f=data["faces"]
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    rest=np.linalg.norm(q0[edges[:,1]]-q0[edges[:,0]],axis=1)
    mouth=np.median(q0[data["grid"][:,-1],0])
    neck=(q0[edges].mean(1)[:,0]<mouth+.07)&(rest>1e-8)
    rows=[]
    for key in sorted(k for k in data.files if k.startswith("q_")):
        step=int(key[2:])
        if a.steps is not None and step not in a.steps:continue
        q=data[key];length=np.linalg.norm(q[edges[:,1]]-q[edges[:,0]],axis=1)
        strain=length[neck]/rest[neck]-1
        rows.append(dict(step=step,neck_max_extension=float(strain.max()),
                         neck_p95_extension=float(np.quantile(strain,.95)),
                         neck_min_extension=float(strain.min())))
    result=dict(scope="Actual mesh edge extension relative to the initial sewn pouch; not calibrated tearing",rows=rows)
    print(json.dumps(result,indent=2))
    if a.steps is None:(a.run/"membrane-strain.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()
