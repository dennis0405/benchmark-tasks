"""Assess recorded folding, clip retention and optional cracker integrity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from clip_capture import clip_capture
from contents_score import assess_contents


def contact_capacity_valid(report, expected_substeps):
    if not report or report.get("checked_substeps")!=expected_substeps:
        return False
    fields=("rigid_body_overflow_events","soft_body_overflow_events",
            "rigid_total_overflow_substeps","soft_total_overflow_substeps")
    return all(report.get(name)==0 for name in fields)


def summarize_hold(rows,hz,required_width=.096):
    longest=0.;start=None;previous=None
    for row in rows:
        good=(row["phase"]=="hands_free_hold" and row["fold_ok"] and
              row["hands_clear"] and row["coverage_m"]>=required_width and
              -.15<row["hinge_rad"]<-.005 and row["bag_on_table"] and row.get("jaw_load_ok",False))
        if not good:start=None
        elif start is None or previous is None or row["step"]!=previous+1:start=row["step"]
        if good:longest=max(longest,(row["step"]-start)/hz)
        previous=row["step"]
    return longest


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    args=parser.parse_args();p=args.run
    recording=(p/"states.npz").read_bytes();trace_bytes=(p/"trace.json").read_bytes()
    import io
    data=np.load(io.BytesIO(recording));trace=json.loads(trace_bytes)
    result=json.loads((p/"result.json").read_text())
    parts=json.loads((p/"render-geometry.json").read_text())
    lower=next(r["body"] for r in parts if r["name"].endswith("/lower_jaw"))
    upper=next(r["body"] for r in parts if r["name"].endswith("/upper_jaw"))
    hz=result["control_hz"];rows=[]
    for t in trace:
        if t["phase"]!="hands_free_hold":continue
        step=t["step"];q=data[f"q_{step:05d}"];b=data[f"b_{step:05d}"]
        captured=clip_capture(q,data["faces"],data["grid"],b[lower],b[upper])
        fold=t["fold"]["folds"][0]
        distances={a:float(np.linalg.norm(q-np.asarray(t["jaws"][a]),axis=1).min())
                   for a in ("left","right")}
        mean_loads={label:abs(float(np.dot(t.get("clip_mean_force_world_N",{}).get(label,[0,0,0]),
                         Rotation.from_quat(b[body,3:]).apply([0,0,1]))))
                    for label,body in (("lower",lower),("upper",upper))}
        valid_window=abs(t.get("force_window_s",0.)-1/hz)<1e-5
        robot_loads=[np.linalg.norm(v) for label,v in t.get("body_mean_force_world_N",{}).items() if "yam" in label]
        rows.append(dict(step=step,phase=t["phase"],
                         fold_ok=bool(fold["valid"] and fold["median_angle_rad"]>=2.5 and fold["folded_fraction"]>=.8),
                         hands_clear=bool(robot_loads) and sum(robot_loads)<.005 and min(distances.values())>.045 and all(
                             sum(np.linalg.norm(v) for v in t.get("finger_mean_force_world_N",{}).get(arm,[[float("inf"),0,0]]))<.005
                             for arm in ("left","right")),
                         jaw_center_film_distances_m=distances,
                         coverage_m=captured["common_coverage_m"],capture=captured,
                         hinge_rad=t["clip_angle_rad"],
                         mean_abs_jaw_normal_load_N=mean_loads,
                         jaw_load_ok=valid_window and min(mean_loads.values())>.005,
                         bag_on_table=bool(q[:,2].min()>.72+result.get("world_z_offset_m",0.) and (q[:,0]>.2).all() and (q[:,0]<1.2).all() and (abs(q[:,1])<.6).all())))
    geometry=result.get("substep_geometry") or {}
    checks=dict(controller_completed=result["controller_completed"] and result["controller_failure"] is None,
                full_episode_without_component_reset=not result.get("initial_fold_imposed",False) and result.get("component_reset") is None,
                uninterrupted=not result["interrupted"],all_finite=result["all_finite"],
                initial_unfolded=trace[0]["fold"]["folds"][0]["median_angle_rad"]<.1,
                all_substeps_observed=geometry.get("checked_substeps")==result["frames"]*result["physics_substeps"],
                contact_capacity_observed_without_overflow=contact_capacity_valid(
                    result.get("contact_capacity"),result["frames"]*result["physics_substeps"]),
                no_observed_transverse_crossings=geometry.get("transverse_edge_crossings")==0,
                no_observed_box_intersections=geometry.get("film_box_intersections")==0,
                round_shapes_checked=geometry.get("checked_round_shapes",0)>0,
                no_observed_round_intersections=geometry.get("film_round_intersections")==0,
                no_observed_ellipsoid_intersections=geometry.get("film_ellipsoid_intersections",0)==0,
                every_hold_force_window_observed=bool(rows) and all(
                    abs(t.get("force_window_s",0.)-1/hz)<1e-5 for t in trace if t["phase"]=="hands_free_hold"),
                hands_free_hold_3s=summarize_hold(rows,hz)>=3.)
    held=[t for t in trace if t["phase"]=="hands_free_hold"]
    loads={jaw:float(np.mean([r["mean_abs_jaw_normal_load_N"][jaw] for r in rows])) if rows else 0.
           for jaw in ("lower","upper")}
    checks["both_jaw_mean_observed_load"]=min(loads.values())>.005
    mechanics_passed=all(checks.values())
    contents=assess_contents(p,data,trace,result)
    checks.update(contents_evaluation_in_scope=contents["in_scope"],
                  contents_observation_complete=contents["observation_complete"],
                  contents_initial_valid=contents["initial_valid"],
                  contents_state_consistent=contents["state_consistent"],
                  contents_retained=contents["no_spill"],contents_intact=contents["no_fracture"])
    scored=contents["in_scope"] and contents["observation_complete"] and contents["state_consistent"] and contents["initial_valid"]
    report=dict(core_passed=all(checks.values()),checks=checks,
                mechanics_passed=mechanics_passed,contents=contents,
                experimental_score=float(mechanics_passed)*contents["multiplier"] if scored else None,
                longest_valid_hold_s=summarize_hold(rows,hz),mean_abs_jaw_normal_load_N=loads,
                required_width_m=.096,rows=rows,submission_score=None,
                limitation="Experimental mechanics and optional predefined-seam cracker integrity; no gas leakage, calibrated material fracture or Arena integration. Hand clearance uses a 45 mm pad-centre bound. Geometry observer covers BOX/capsule/sphere/ellipsoid but excludes coplanar overlap, other colliders and motion within a substep.",
                states_sha256=hashlib.sha256(recording).hexdigest(),trace_sha256=hashlib.sha256(trace_bytes).hexdigest())
    (p/"core-assessment.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
