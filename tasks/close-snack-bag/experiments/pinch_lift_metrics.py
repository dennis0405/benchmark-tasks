"""Reject empty-hand motion using two-wall capture and fixed material landmarks."""
import numpy as np
from stack_grasp import clipped_surfaces


MIN_AREA_MM2=50.
MIN_LOAD_N=.02  # Presence threshold only; the readout is not force calibrated.
MAX_SLIP_M=.003
MIN_LIFT_M=.040


def mouth_capture(local_points,faces,grid,inner_faces):
    """Clip the last 10 mm of both walls to the measured main-pad cavity."""
    q=np.asarray(local_points);low,high=sorted(map(float,inner_faces))
    if high<=low:return dict(area_mm2=[0.,0.],vertices=[[],[]])
    bounds=[(0,-.006,.006),(2,-.04,.04),(1,low,high)]
    nx,ny=grid.shape[1]-1,grid.shape[2]-1
    cells=np.asarray(faces).reshape(2,nx,ny,2,3)
    areas=[];vertices=[]
    for layer in range(2):
        triangles=cells[layer,max(0,nx-2):].reshape(-1,3)
        areas.append(clipped_surfaces(q[triangles],bounds)[0]*1e6)
        ids=np.unique(grid[layer,max(0,nx-2):,:]);p=q[ids]
        inside=np.ones(len(ids),dtype=bool)
        for axis,a,b in bounds:inside&=(p[:,axis]>=a)&(p[:,axis]<=b)
        vertices.append(ids[inside].tolist())
    return dict(area_mm2=areas,vertices=vertices)


def hand_valid(hand,*,lifted):
    try:
        area=np.asarray(hand["wall_area_mm2"],dtype=float)
        load=np.asarray(hand["opposed_load_N"],dtype=float)
        slip=np.asarray(hand["wall_slip_p90_m"],dtype=float)
        lift=np.asarray(hand["wall_lift_m"],dtype=float)
        if any(a.shape!=(2,) or not np.isfinite(a).all() for a in (area,load,slip,lift)):return False
        return bool((area>=MIN_AREA_MM2).all() and (load>=MIN_LOAD_N).all() and
                    (slip<=MAX_SLIP_M).all() and (not lifted or (lift>=MIN_LIFT_M).all()))
    except (KeyError,TypeError,ValueError):return False


def evaluate(trace,result):
    hz=float(result["control_hz"]);longest=0.;start=None;previous=None
    for row in trace:
        measurement=row.get("pinch_lift",{})
        good=(row["phase"]=="pinch_lift_hold" and measurement.get("reference_locked",False) and
              all(hand_valid(measurement.get("hands",{}).get(a,{}),lifted=True) for a in ("left","right")))
        if not good:start=None
        elif start is None or previous is None or row["step"]!=previous+1:start=row["step"]
        if good:longest=max(longest,(row["step"]-start)/hz)
        previous=row["step"]
    expected=result["frames"]*result["physics_substeps"]
    geometry=result.get("substep_geometry") or {};capacity=result.get("contact_capacity") or {}
    checks=dict(complete=result["controller_completed"] and result["controller_failure"] is None,
        uninterrupted=not result["interrupted"],finite=result["all_finite"],
        no_imposed_object_motion=not result["object_pose_writes_during_rollout"] and result.get("component_reset") is None,
        no_clip=not result["clip"],no_fold_phase=not any(r["phase"] in ("fold","lay_flap") for r in trace),
        all_physics_geometry_observed=geometry.get("checked_substeps")==expected,
        geometry_clear=all(geometry.get(k)==0 for k in ("transverse_edge_crossings","film_box_intersections","film_round_intersections","film_ellipsoid_intersections")),
        capacity_observed=capacity.get("checked_substeps")==expected and all(capacity.get(k)==0 for k in (
            "rigid_body_overflow_events","soft_body_overflow_events","rigid_total_overflow_substeps","soft_total_overflow_substeps")),
        bilateral_material_lift_and_hold_3s=longest>=3.)
    return dict(passed=all(checks.values()),checks=checks,longest_valid_hold_s=longest,
        scope="Two-wall mouth grasp; each wall rises at least 40 mm, material slip <=3 mm, both opposed contacts present for 3 s. Not whole-bag suspension, folding or sealing.",
        thresholds=dict(area_mm2=MIN_AREA_MM2,load_N=MIN_LOAD_N,slip_m=MAX_SLIP_M,lift_m=MIN_LIFT_M),
        limitation="Uncalibrated simulated material and qualitative force presence; main-pad prism approximation and existing geometry-observer exclusions remain.")
