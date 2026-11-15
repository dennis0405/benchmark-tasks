"""Read-only post-analysis of genuine native logs; no simulator or model calls.

SAT is evaluated at the recorded 20 Hz control samples, not at PhysX substeps.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys

import numpy as np

FACES=((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
UNIT=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)
TOP=.7475

def rotation(quat):
    x,y,z,w=np.asarray(quat,dtype=float)/np.linalg.norm(quat)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def shape_axes(vertices):
    normals=[];edges=[]
    for face in FACES:
        a,b,c=vertices[list(face[:3])]
        normals.append(np.cross(b-a,c-a))
        for i in range(4):edges.append(vertices[face[(i+1)%4]]-vertices[face[i]])
    def unique(vectors):
        result={}
        for v in vectors:
            length=np.linalg.norm(v)
            if length<1e-12:continue
            v=v/length
            first=np.flatnonzero(np.abs(v)>1e-8)[0]
            if v[first]<0:v=-v
            result[tuple(np.round(v,9))]=v
        return list(result.values())
    return unique(normals),unique(edges)

def penetration(a,b):
    if np.any(a.max(0)<=b.min(0)) or np.any(b.max(0)<=a.min(0)):
        return 0.0
    na,ea=shape_axes(a);nb,eb=shape_axes(b)
    depth=float("inf")
    for axis in [*na,*nb,*(np.cross(x,y) for x in ea for y in eb)]:
        length=np.linalg.norm(axis)
        if length<1e-10:continue
        axis=axis/length
        pa,pb=a@axis,b@axis
        overlap=min(pa.max(),pb.max())-max(pa.min(),pb.min())
        if overlap<=0:return 0.0
        depth=min(depth,float(overlap))
    return depth

def components(path):
    text=path.read_text();out=[]
    for kind,name,body in re.findall(r'def (Cube|Mesh) "([^"]+)"[^\n]*\{\n(.*?)\n\}',text,re.S):
        if kind=="Cube":
            scale=np.array(ast.literal_eval(re.search(r'double3 xformOp:scale = (\([^\n]+\))',body)[1]))
            xyz=np.array(ast.literal_eval(re.search(r'double3 xformOp:translate = (\([^\n]+\))',body)[1]))
            vertices=UNIT*scale/2+xyz
        else:
            vertices=np.array(ast.literal_eval(re.search(r'point3f\[\] points = (\[[^\n]+\])',body)[1]))
        if vertices.shape!=(8,3):raise ValueError("unexpected collider: "+name)
        out.append((name,vertices))
    if len(out)!=6:raise ValueError("expected six authored convex colliders")
    return out

def main():
    phase=Path(sys.argv[1]).resolve()
    body=[json.loads(x) for x in (phase/"evidence/raw/body_states.jsonl").read_text().splitlines()]
    joints=[json.loads(x) for x in (phase/"evidence/raw/joint_states.jsonl").read_text().splitlines()]
    parts=components(phase/"source/assets/alignment_jig.usda")
    all_vertices=np.concatenate([v for _,v in parts])
    centre=(all_vertices.min(0)+all_vertices.max(0))/2
    maximum=0.0;worst=None;floor_max=0.0
    for index,row in enumerate(body):
        jig=row["bodies"]["alignment_jig"];r=rotation(jig["quat_xyzw"])
        origin=np.array(jig["pos_m"])-r@centre
        world_parts=[(name,v@r.T+origin) for name,v in parts]
        floor_max=max(floor_max,TOP-min(v[:,2].min() for _,v in world_parts))
        for name in ("lm_rail","master_rail"):
            state=row["bodies"][name];rr=rotation(state["quat_xyzw"])
            vertices=(UNIT*np.array([.160,.014,.008]))@rr.T+np.array(state["pos_m"])
            floor_max=max(floor_max,TOP-float(vertices[:,2].min()))
            for part,verts in world_parts:
                depth=penetration(verts,vertices)
                if depth>maximum:maximum=depth;worst={"sample":index,"pair":[part,name],"depth_m":depth}
    aligned=[]
    for row in joints:
        index=round(float(row["state_time_s"])*20)-1
        if 0<=index<len(body):aligned.append((row,body[index]))
    window=aligned[-41:]
    if len(window)!=41:raise ValueError("not enough actual joint observations for two seconds")
    minimum_open=min(min(j["actual_joint_pos"][6],j["actual_joint_pos"][13]) for j,_ in window)
    clearance=float("inf")
    for j,b in window:
        for arm in ("left","right"):
            tcp=np.array(j["arms"][arm]["actual_tcp_m"])
            for name in ("alignment_jig","lm_rail"):
                obj=b["bodies"][name];low=np.array(obj["aabb_low_m"]);high=np.array(obj["aabb_high_m"])
                delta=np.maximum(np.maximum(low-tcp,tcp-high),0)
                clearance=min(clearance,float(np.linalg.norm(delta)))
    span=window[-1][0]["state_time_s"]-window[0][0]["state_time_s"]
    from yam_arena.task_file import load,ObjectState
    task=load(phase/"source/task.py");last_score=0.0;last_success=False
    for row in body:
        states={name:ObjectState(np.array(s["pos_m"]),np.array(s["quat_xyzw"]),np.array(s["lin_vel_m_s"]),np.array(s["ang_vel_rad_s"]),np.array(s["aabb_low_m"]),np.array(s["aabb_high_m"])) for name,s in row["bodies"].items()}
        last_success=task.module.success(states)
        last_score=task.module.score(states)
    sdk=json.loads((phase/"evidence/native/trial.json").read_text())["results"]["metrics"]["task_score"]
    checks={"source_scorer_replay_matches_native":(1.0 if last_success else last_score)==sdk==1.0,
            "sampled_jig_rail_penetration_le_1mm":maximum<=.001,
            "sampled_floor_penetration_le_1mm":floor_max<=.001,
            "last_two_seconds_both_grippers_open":minimum_open>=.90 and span>=2-1e-6,
            "last_two_seconds_tcp_clearance_ge_30mm":clearance>=.030}
    result={"kind":"analysis-of-actual-native-trial","checks":checks,"all_passed":all(checks.values()),
            "control_sample_hz":20,"physics_substeps_observed":False,"samples":len(body),
            "source_task_sha256":hashlib.sha256((phase/"source/task.py").read_bytes()).hexdigest(),
            "measured_max_penetration_m":maximum,"worst_pair":worst,"measured_floor_penetration_m":max(0.0,floor_max),
            "release_window_s":span,"minimum_actual_gripper_opening":minimum_open,"minimum_tcp_aabb_clearance_m":clearance,
            "native_score":sdk,"replayed_score":1.0 if last_success else last_score}
    (phase/"evidence/measured-validation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result))
    return 0 if result["all_passed"] else 2

if __name__=="__main__":raise SystemExit(main())
