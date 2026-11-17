"""Locate measured motion beyond the DRG energized-adjustment limits in a probe.

A proof force is an external test input. Only the full workcell can produce the
piston's measured actuator effort and the official task score.
"""
import argparse
import json
import math
from pathlib import Path


def inspect(trace, hz):
    rows = json.loads(trace.read_text())
    limits = {"screw_angle_rad": .01, "angle_rad": .01, "screw_z_m": .00001}
    max_rates = {k: 0. for k in limits}
    first = None
    count = 0
    for before, after in zip(rows,rows[1:]):
        if after["step"]!=before["step"]+1:
            raise ValueError("discontinuous native samples")
        if abs(after["proof_force_n"])<=1:
            continue
        rates={k:abs(after[k]-before[k])*hz for k in limits}
        if not all(math.isfinite(x) for x in rates.values()):
            raise ValueError("non-finite native pose")
        max_rates={k:max(max_rates[k],rates[k]) for k in limits}
        over = [k for k in limits if rates[k]>limits[k]]
        if over:
            count+=1
            if first is None:
                first={"step":after["step"],"time_s":after["step"]/hz,"exceeded":over,"rates":rates}
    return {"probe":trace.name,"hz":hz,"thresholds":limits,"first_failure":first,
            "over_limit_steps":count,"peak_rates":max_rates}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("--hz",type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(inspect(args.trace,args.hz),indent=2))
