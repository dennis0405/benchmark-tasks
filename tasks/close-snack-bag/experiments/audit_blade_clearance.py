"""Read-only control-frame bound for lower-blade clearance above the board.

This is not a swept/substep collision proof. A positive minimum vertex height
is a conservative clearance bound for the convex blade boxes. Heights below
the board plane while the tool is outside its XY bounds are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    parser.add_argument("--phases",nargs="+",default=["clip_insert","clip_grasp_watch","hands_free_hold"])
    a=parser.parse_args()
    raw=(a.run/"trace.json").read_bytes();trace=json.loads(raw)
    geometry=np.load(a.run/"render-geometry.npz")
    parts=json.loads((a.run/"render-geometry.json").read_text())
    boards=[];blades=[];bodies=set()
    for i,part in enumerate(parts):
        name=part["name"]
        if name!="board" and not (name.endswith("/lower_jaw") or "/lower_tip_" in name):continue
        points=np.c_[geometry[f"v_{i}"],np.ones(len(geometry[f"v_{i}"]))]
        points=(points@np.asarray(part["local"]).T)[:,:3]
        if name=="board":
            if part["body"]>=0:raise ValueError("audit requires a static board")
            boards.extend(points)
        else:blades.extend(points);bodies.add(part["body"])
    if not boards or not blades or len(bodies)!=1:raise ValueError("missing/ambiguous blade geometry")
    board=np.asarray(boards);blade=np.asarray(blades)
    low,high=board.min(0),board.max(0);rows=[]
    for row in trace:
        if row["phase"] not in a.phases:continue
        pose=np.asarray(row["clip_root_pose"])
        points=Rotation.from_quat(pose[3:]).apply(blade)+pose[:3]
        overlaps=bool((points[:,:2].max(0)>=low[:2]).all() and
                      (points[:,:2].min(0)<=high[:2]).all())
        if overlaps:
            rows.append(dict(step=row["step"],phase=row["phase"],
                             clearance_lower_bound_m=float(points[:,2].min()-high[2])))
    report=dict(scope=__doc__,phases=a.phases,board_aabb_m=[low.tolist(),high.tolist()],
                lower_blade_body=int(next(iter(bodies))),blade_body_local_vertices_m=blade.tolist(),
                minimum_clearance_over_board_m=min((r["clearance_lower_bound_m"] for r in rows),default=None),
                checked_control_frames=len(rows),rows=rows,
                trace_sha256=hashlib.sha256(raw).hexdigest(),
                render_geometry_sha256=hashlib.sha256((a.run/"render-geometry.npz").read_bytes()).hexdigest())
    (a.run/"blade-clearance-audit.json").write_text(json.dumps(report,indent=2))
    print({k:report[k] for k in ("minimum_clearance_over_board_m","checked_control_frames")})


if __name__=="__main__":main()
