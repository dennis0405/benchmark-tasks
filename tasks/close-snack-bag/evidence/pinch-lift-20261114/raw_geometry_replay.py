"""Read-only check of original material motion against recorded physical pads.

Uses raw q/body poses, rather than commanded cursors or logged lift/slip
summaries. Opposed force presence remains the recorded solver observation.
The cavity is the same main-pad prism approximation, in the measured pad
frame instead of the end-effector frame. Existing geometry exclusions remain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    parser.add_argument("experiment_sources",type=Path);args=parser.parse_args()
    sys.path.insert(0,str(args.experiment_sources))
    from pinch_lift_metrics import mouth_capture
    run=args.run;raw=(run/"states.npz").read_bytes();trace_bytes=(run/"trace.json").read_bytes()
    data=np.load(run/"states.npz");colliders=np.load(run/"colliders.npz")
    result=json.loads((run/"result.json").read_text());trace=json.loads(trace_bytes)
    refs=result["controller_diagnostics"]["pinch_lift_reference"]
    parts=json.loads((run/"render-geometry.json").read_text());shapes={}
    for arm in refs:
        owners={p["body"] for p in parts if f"/{arm}_arm/" in p["name"] and "finger" in p["name"]}
        shapes[arm]=np.flatnonzero(np.isin(colliders["body"],list(owners)) &
            (colliders["kind"]==7)&(abs(colliders["scale"][:,2]-.04)<1e-5))
        assert len(shapes[arm])==2
    def frame(step,arm):
        q=data[f"q_{step:05d}"];b=data[f"b_{step:05d}"];centres=[];rotations=[]
        for sid in shapes[arm]:
            body=b[colliders["body"][sid]];r=Rotation.from_quat(body[3:])
            centres.append(body[:3]+r.apply(colliders["local"][sid,:3]))
            rotations.append(r*Rotation.from_quat(colliders["local"][sid,3:]))
        origin=np.mean(centres,axis=0);r=rotations[0]
        local=r.inv().apply(q-origin);pads=r.inv().apply(np.asarray(centres)-origin)
        low,high=sorted(pads[:,1]);return q,local,(low+.002,high-.002)
    candidates=[row["step"] for row in trace]
    def reference_error(step):
        q=data[f"q_{step:05d}"]
        return max(float(abs(q[ids,2]-z).max()) for ref in refs.values()
                   for ids,z in zip(ref["vertices"],ref["world_z"]))
    reference_step=min(candidates,key=reference_error);assert reference_error(reference_step)<1e-8
    held=[row for row in trace if row["phase"]=="pinch_lift_hold"]
    rows=[]
    if held:
        hold_entry=held[0]["step"]-1
        for row in held:
            measured={};good=True
            for arm,ref in refs.items():
                q,local,bounds=frame(row["step"],arm);_,base,_=frame(hold_entry,arm)
                capture=mouth_capture(local,data["faces"],data["grid"],bounds)
                lifts=[];slips=[];retained=[]
                for ids,initial,z in zip(ref["vertices"],ref["local"],ref["world_z"]):
                    initial=np.asarray(initial);ids=np.asarray(ids)
                    central=np.argsort(np.linalg.norm(initial-np.median(initial,axis=0),axis=1))[:5]
                    p=local[ids[central]]
                    retained.append(bool(((abs(p[:,0])<=.006)&(abs(p[:,2])<=.04)&(p[:,1]>=bounds[0])&(p[:,1]<=bounds[1])).all()))
                    lifts.append(float(np.median(q[ids,2]-z)))
                    slips.append(float(np.quantile(np.linalg.norm(local[ids]-base[ids],axis=1),.9)))
                good&=all(retained) and min(lifts)>=.04 and max(slips)<=.003 and min(capture["area_mm2"])>=50.
                measured[arm]=dict(wall_lift_m=lifts,wall_hold_slip_p90_m=slips,wall_area_mm2=capture["area_mm2"],original_landmarks_retained=retained)
            rows.append(dict(step=row["step"],passed=good,hands=measured))
    assessment=json.loads((run/"pinch-lift-assessment.json").read_text())
    terminal={}
    for arm,ref in refs.items():
        _,local,bounds=frame(result["frames"],arm);counts=[];overruns=[]
        for ids,initial in zip(ref["vertices"],ref["local"]):
            initial=np.asarray(initial);ids=np.asarray(ids)
            central=np.argsort(np.linalg.norm(initial-np.median(initial,axis=0),axis=1))[:5]
            p=local[ids[central]]
            inside=(abs(p[:,0])<=.006)&(abs(p[:,2])<=.04)&(p[:,1]>=bounds[0])&(p[:,1]<=bounds[1])
            counts.append(int(inside.sum()))
            overruns.append(float(max(0.,(abs(p[:,0])-.006).max(),(abs(p[:,2])-.04).max())))
        terminal[arm]=dict(original_landmarks_inside_per_wall=counts,per_wall_count=5,maximum_pad_footprint_overrun_m=overruns)
    hashes_match=(hashlib.sha256(raw).hexdigest()==assessment["states_sha256"] and
                  hashlib.sha256(trace_bytes).hexdigest()==assessment["trace_sha256"])
    continuous=bool(rows) and all(b["step"]==a["step"]+1 for a,b in zip(rows,rows[1:]))
    duration=(rows[-1]["step"]-rows[0]["step"])/result["control_hz"] if rows else 0.
    report=dict(passed=bool(assessment["passed"] and hashes_match and continuous and duration>=3. and all(r["passed"] for r in rows)),
        reference_step=reference_step,raw_hold_duration_s=duration,hashes_match=hashes_match,terminal_landmarks=terminal,rows=rows,scope=__doc__)
    (run/"raw-pinch-lift-replay.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
