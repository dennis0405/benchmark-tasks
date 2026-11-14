"""Audit a v3 contact-retention recording from raw film and physical finger poses.

Cursor targets and pad-footprint landmark flags do not establish retention.
Recorded mean force vectors provide qualitative opposed-contact presence;
the contact readout and film properties remain uncalibrated.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pinch_lift_metrics import primitive_surface_distance


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    args=parser.parse_args();p=args.run
    recording=(p/"states.npz").read_bytes();trace_bytes=(p/"trace.json").read_bytes()
    data=np.load(io.BytesIO(recording));trace=json.loads(trace_bytes)
    result=json.loads((p/"result.json").read_text())
    assessment=json.loads((p/"pinch-lift-assessment.json").read_text())
    refs=result["controller_diagnostics"]["pinch_lift_reference"]
    shapes=np.load(p/"colliders.npz");parts=json.loads((p/"render-geometry.json").read_text())
    selected={}
    for arm in refs:
        bodies={part["body"] for part in parts if f"/{arm}_arm/" in part["name"] and "finger" in part["name"]}
        selected[arm]=np.flatnonzero(np.isin(shapes["body"],list(bodies))&np.isin(shapes["kind"],[3,4,7]))
        if not len(selected[arm]):raise ValueError(f"No physical finger primitives for {arm}")
    def ref_error(step):
        q=data[f"q_{step:05d}"]
        return max(float(abs(q[ids,2]-z).max()) for base in refs.values()
            for ids,z in zip(base["vertices"],base["world_z"]))
    reference_step=min((row["step"] for row in trace),key=ref_error)
    if ref_error(reference_step)>1e-8:raise ValueError("Original reference material is not in the raw recording")
    rows=[];longest=0.;start=None;previous=None;consistent=True
    for row in trace:
        if row["phase"]!="pinch_lift_hold":continue
        step=row["step"];q=data[f"q_{step:05d}"];b=data[f"b_{step:05d}"]
        measured={};good=abs(row.get("force_window_s",0.)-1/result["control_hz"])<1e-5
        for arm,base in refs.items():
            distances=np.full(len(q),np.inf)
            for sid in selected[arm]:
                body=b[shapes["body"][sid]];rotation=Rotation.from_quat(body[3:])
                center=body[:3]+rotation.apply(shapes["local"][sid,:3])
                rotation=rotation*Rotation.from_quat(shapes["local"][sid,3:])
                distances=np.minimum(distances,primitive_surface_distance(
                    rotation.inv().apply(q-center),int(shapes["kind"][sid]),shapes["scale"][sid]))
            wall_distance=[float(distances[np.unique(data["grid"][layer,-5:,:])].min()) for layer in range(2)]
            lift=[float(np.median(q[ids,2]-z)) for ids,z in zip(base["vertices"],base["world_z"])]
            loads=np.linalg.norm(np.asarray(row["finger_mean_force_world_N"][arm]),axis=1)
            hand_good=bool(min(lift)>=.04 and max(wall_distance)<=.003 and min(loads)>=.02)
            logged=row["pinch_lift"]["hands"][arm]
            consistent&=bool(np.allclose(lift,logged["wall_lift_m"],atol=1e-7,rtol=0.) and
                             np.allclose(wall_distance,logged["wall_finger_distance_m"],atol=1e-7,rtol=0.))
            measured[arm]=dict(passed=hand_good,original_wall_rise_m=lift,
                actual_finger_to_wall_distance_m=wall_distance,opposed_mean_load_N=loads.tolist())
            good&=hand_good
        if not good:start=None
        elif start is None or previous is None or step!=previous+1:start=step
        if good:longest=max(longest,(step-start)/result["control_hz"])
        rows.append(dict(step=step,passed=bool(good),hands=measured));previous=step
    hashes_match=(assessment["states_sha256"]==hashlib.sha256(recording).hexdigest() and
                  assessment["trace_sha256"]==hashlib.sha256(trace_bytes).hexdigest())
    report=dict(passed=bool(assessment["passed"] and hashes_match and consistent and longest>=3.),
        protocol=assessment["protocol"],reference_step=reference_step,
        longest_raw_verified_hold_s=longest,raw_measurements_match_trace=consistent,
        hashes_match=hashes_match,hold_rows=len(rows),rows=rows,scope=__doc__)
    (p/"raw-contact-hold-assessment.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
