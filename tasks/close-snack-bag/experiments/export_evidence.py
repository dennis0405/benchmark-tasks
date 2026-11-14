"""Preserve compact evidence without treating selected states as full coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--steps",type=int,nargs="+",required=True)
    parser.add_argument("--scope",required=True)
    args=parser.parse_args()
    if not (args.run/"result.json").exists():
        parser.error("run has no final result.json; finish or interrupt it cleanly first")
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ("result.json","trace.json","arguments.json","preflight.json","rigid-tool-preflight.json",
                 "sampled-geometry.json","crossing-details.json","controller-marks.json",
                 "core-assessment.json","render-manifest.json","finger-fixture-preflight.json",
                 "fixed-open-clip.xml","free-open-clip.xml","blade-clearance-audit.json","rigid-contact-audit.json"):
        path=args.run/name
        if path.exists():
            shutil.copy2(path,args.output/name)
    shutil.copytree(args.run/"sources",args.output/"sources")
    with np.load(args.run/"states.npz") as states:
        selected={"faces":states["faces"]} if "faces" in states else {}
        if "grid" in states:
            selected["grid"]=states["grid"]
        for name in ("edge_rows","neck_edge_mask","edge_rest_initial","edge_rest_current"):
            if name in states:
                selected[name]=states[name]
        for step in sorted(set(args.steps)):
            if f"b_{step:05d}" not in states:
                raise KeyError(f"No recorded step {step}")
            for prefix in ("q","b","j","d","w","v","r"):
                key=f"{prefix}_{step:05d}"
                if key in states:
                    selected[key]=states[key]
        np.savez_compressed(args.output/"selected-states.npz",**selected)
    files={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob("*") if p.is_file()}
    manifest=dict(scope=args.scope,selected_steps=sorted(set(args.steps)),
                  original_states_sha256=digest(args.run/"states.npz"),sha256=files,
                  limitation="Selected snapshots do not certify intervening motion or continuous contact")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(args.output)


if __name__=="__main__":
    main()
