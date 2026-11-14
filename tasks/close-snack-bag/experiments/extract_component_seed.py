"""Extract one recorded physical state for an explicitly reset component test.

Never stitches episodes into a demo. Initial contact reliefs, if desired,
must be declared to the runner and pass its unchanged intersection preflight.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--step",type=int);a=p.parse_args()
    args=json.loads((a.run/"arguments.json").read_text())
    states=np.load(a.run/"states.npz");trace=json.loads((a.run/"trace.json").read_text())
    step=a.step if a.step is not None else max(int(k[2:]) for k in states.files if k.startswith("b_"))
    row=next(t for t in trace if t["step"]==step)
    if row["phase"] not in ("lay_flap","clip_stage","clip_descend_outside","clip_insert","clip_release","hands_free_hold","press_left","press_right","table_fold_ready"):
        raise ValueError("extract an observed pressing or post-grasp component state")
    layout_path=a.run/"seeds"/"clip_insert"/"clip-entry.json"
    if not layout_path.exists():
        candidates=list((a.run/"seeds").glob("*/clip-entry.json"))
        if candidates:layout_path=max(candidates,key=lambda path:json.loads(path.read_text())["step"])
        elif args.get("clip_component_seed"):layout_path=Path(args["clip_component_seed"])/"clip-entry.json"
        else:
            import ast
            log=a.run.parent/(a.run.name+".log")
            lines=[line for line in log.read_text().splitlines() if line.startswith("LAYOUT ")]
            if not lines:raise ValueError("no observed layout metadata")
            layout=ast.literal_eval(lines[-1][len("LAYOUT "):].split(" rim_z ",1)[0])
            meta=dict(layout=layout,layout_source=str(log),layout_source_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    if layout_path.exists():meta=json.loads(layout_path.read_text())
    data={k:states[f"{k}_{step:05d}"] for k in ("q","b","j","d","v","r")}
    if f"w_{step:05d}" in states:data["w"]=states[f"w_{step:05d}"]
    for key in ("e","cp","cg"):
        if f"{key}_{step:05d}" in states:data[key]=states[f"{key}_{step:05d}"]
    data.update({k:states[k] for k in ("faces","grid","edge_rows")})
    if not all(np.isfinite(v).all() for v in data.values()):raise ValueError("non-finite seed")
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/"clip-entry.npz",**data)
    # The runner reconstructs actuated controls from this exact frame's
    # recorded targets, using its rebuilt model's coordinate mapping.
    (a.output/"trace.json").write_text(json.dumps([row]))
    result=json.loads((a.run/"result.json").read_text())
    (a.output/"clip-entry.json").write_text(json.dumps(dict(step=step,layout=meta["layout"],
        arguments=args,support_holds={},left_cleared_for_clip=meta.get("left_cleared_for_clip",False),
        source=str(a.run),phase=row["phase"],scope=__doc__,
        source_controller_failure=result.get("controller_failure"),
        source_states_sha256=hashlib.sha256((a.run/"states.npz").read_bytes()).hexdigest()),indent=2))
    print(a.output)


if __name__=="__main__":main()
