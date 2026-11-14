"""Experimental contents integrity; not an official Arena submission score."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from contents_inside import contained_centres


def integrity_multiplier(broken,spilled,masses,groups):
    masses=np.asarray(masses,dtype=float);broken=np.asarray(broken,dtype=bool)
    spilled=np.asarray(spilled,dtype=bool)
    if masses.sum()<=0:raise ValueError("positive contents mass required")
    intact=sum(masses[group].sum() for bad,group in zip(broken,groups) if not bad)/masses.sum()
    retained=masses[~spilled].sum()/masses.sum()
    return dict(intact_mass_fraction=float(intact),retained_mass_fraction=float(retained),
                multiplier=float(intact*retained))


def assess_contents(run,data,trace,result):
    info=result.get("contents") or {}
    if not info.get("enabled"):
        return dict(in_scope=True,observation_complete=True,initial_valid=True,
                    state_consistent=True,no_spill=True,no_fracture=True,multiplier=1.,enabled=False)
    rejected=dict(enabled=True,in_scope=False,observation_complete=False,initial_valid=False,
                  state_consistent=False,no_spill=False,no_fracture=False,multiplier=None)
    if not info.get("fracture") or info.get("coordinate_model")!="brittle":
        return dict(rejected,reason="Rigid contents have no damage model")
    bodies=list(map(int,info["bodies"]));lookup={body:i for i,body in enumerate(bodies)}
    groups=[np.asarray([lookup[b] for b in group],dtype=int) for group in info["chip_groups"]]
    seams=np.asarray(info["seams"],dtype=int)
    if not bodies or len(groups)!=len(seams):return dict(rejected,reason="Incomplete fragment/seam mapping")
    colliders=np.load(Path(run)/"colliders.npz")
    from newton import GeoType
    shapes=[]
    for body in bodies:
        ids=np.flatnonzero(colliders["body"]==body)
        if len(ids)!=1 or colliders["kind"][ids[0]]!=int(GeoType.BOX):
            return dict(rejected,reason="Contents geometry is outside the box-fragment evaluator")
        shapes.append(int(ids[0]))
    if "body_mass_kg" in info:
        masses=np.asarray(info["body_mass_kg"],dtype=float);mass_source="recorded model body masses"
    else:
        # Earlier frozen cracker runs recorded total mass and identical boxes.
        scales=colliders["scale"][shapes]
        if not np.allclose(scales,scales[0],atol=1e-12,rtol=0.):
            return dict(rejected,reason="Nonuniform fragments have no recorded mass weights")
        masses=np.full(len(bodies),float(info["total_mass_kg"])/len(bodies))
        mass_source="uniform identical fragments and recorded total mass"
    if len(masses)!=len(bodies) or not np.isfinite(masses).all() or (masses<=0).any():
        return dict(rejected,reason="Invalid fragment masses")
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],dtype=float)
    local_points=[]
    for sid in shapes:
        transform=colliders["local"][sid]
        corners=Rotation.from_quat(transform[3:]).apply(signs*colliders["scale"][sid])+transform[:3]
        local_points.append(np.vstack([transform[:3],corners]))
    spilled=np.zeros(len(bodies),dtype=bool);broken=np.zeros(len(groups),dtype=bool)
    first_spill={};complete=True;consistent=True;initial_valid=True
    for index,row in enumerate(trace):
        step=row["step"];q=data[f"q_{step:05d}"];poses=data[f"b_{step:05d}"]
        queries=np.concatenate([Rotation.from_quat(poses[body,3:]).apply(local)+poses[body,:3]
                                for body,local in zip(bodies,local_points)])
        contained=contained_centres(q,data["faces"],data["grid"],queries).reshape(len(bodies),9).all(axis=1)
        if index==0:initial_valid=bool(contained.all())
        for fragment in np.flatnonzero(~contained & ~spilled):first_spill[str(bodies[fragment])]=int(step)
        spilled|=~contained
        damage=row.get("contents_damage") or {}
        current=np.asarray(damage.get("broken",[]),dtype=bool)
        if len(current)!=len(groups):complete=False;continue
        complete&=damage.get("observed_substeps")==step*result["physics_substeps"]
        if f"e_{step:05d}" not in data:
            consistent=False
        else:
            consistent&=bool(np.array_equal(current,~data[f"e_{step:05d}"][seams].astype(bool)))
        consistent&=not bool((broken & ~current).any())
        if index==0:initial_valid&=not bool(current.any())
        broken|=current
    score=integrity_multiplier(broken,spilled,masses,groups)
    return dict(enabled=True,in_scope=True,observation_complete=bool(complete),initial_valid=bool(initial_valid),
        state_consistent=bool(consistent),no_spill=not bool(spilled.any()),no_fracture=not bool(broken.any()),
        checked_control_frames=len(trace),broken_chip_indices=np.flatnonzero(broken).tolist(),
        spilled_fragment_bodies=[bodies[i] for i in np.flatnonzero(spilled)],first_spill_step=first_spill,
        mass_source=mass_source,**score,
        limitation="Nine box points per fragment at control cadence; temporary mouth cap and three-ray parity. Predefined seam damage is uncalibrated; no gas leakage or general material fracture assessment.")
