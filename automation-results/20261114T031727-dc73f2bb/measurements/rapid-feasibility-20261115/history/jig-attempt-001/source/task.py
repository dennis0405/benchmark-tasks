"""Align an unfastened subsidiary LM rail with its fixed master using a travelling rigid jig.

THK's official LM Guide mounting procedure describes using an alignment jig to establish
parallelism at successive mounting holes before bolt tightening (Fig. 17). This prototype
models the pre-fastening alignment operation, with millimetre-scale educational clearances,
not bearing preload, ball recirculation, micrometre metrology, or bolt tightening.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import numpy as np
from isaaclab_arena.assets.registries import AssetRegistry
from yam_arena.task_file import TABLE_TOP, Box, Asset
from yam_arena.demo import Waypoint
from yam_arena.constants import FINGER_OPEN, FINGER_CLOSED

SCENE = "station"
TOP = TABLE_TOP[SCENE]
REGISTRY_KEY = "rcauto_parallel_jig_20261115"
class _JigDescriptor:
    name = REGISTRY_KEY
    tags = []
    usd_path = str(Path(__file__).resolve().parent / "assets/alignment_jig.usda")
registry = AssetRegistry()
if not registry.is_registered(REGISTRY_KEY, ensure_loaded=False):
    registry.register(_JigDescriptor, key=REGISTRY_KEY)

MASTER = Box("master_rail", pos=(.280, .080, TOP+.008), size=(.320,.028,.016), color=(.25,.30,.35))
SLAVE = Box("lm_rail", pos=(.280,-.087,TOP+.008), size=(.320,.028,.016), mass=.80,
            yaw=math.radians(-3.0), color=(.62,.65,.70))
JIG = Asset("alignment_jig", asset=REGISTRY_KEY, xy=(.200,.220), extent=.206,
            mass=.25, color=(.90,.42,.05), convex=False)
PARK = Box("jig_parking_mark",pos=(.200,.220,TOP-.0005),size=(.050,.220,.001),color=(.05,.22,.32))
OBJECTS = [MASTER,SLAVE,JIG,PARK]
SUCCESS_HOLD_S = 2.0
SCORE_OVER = "end"
_sweep_bins = set()
_sample_index = 0
_invalid = False
_trace_path = os.environ.get("RAPID_BODY_TRACE")

def _yaw(q):
    x,y,z,w = [float(v) for v in q]
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def _errors(objects):
    rail,jig = objects["lm_rail"],objects["alignment_jig"]
    yaw = _yaw(rail.quat_xyzw)
    ends = [float(rail.pos[1]) + side*.160*math.sin(yaw) for side in (-1,1)]
    jig_yaw = _yaw(jig.quat_xyzw)
    return {
        "centre_xy_m":float(np.linalg.norm(rail.pos[:2]-np.array([.280,-.080]))),
        "height_m":abs(float(rail.pos[2])-(TOP+.008)),
        "yaw_rad":abs(yaw),
        "datum_gap_m":max(abs(y+.080) for y in ends),
        "speed_m_s":float(np.linalg.norm(rail.lin_vel)),
        "spin_rad_s":float(np.linalg.norm(rail.ang_vel)),
        "jig_x_m":float(jig.pos[0]),
        "jig_centre_y_error_m":abs(float(jig.pos[1])),
        "jig_yaw_rad":abs(jig_yaw),
        "jig_bottom_error_m":abs(float(jig.low[2])-TOP),
        "jig_park_error_m":float(np.linalg.norm(jig.pos[:2]-np.array([.200,.220]))),
        "jig_speed_m_s":float(np.linalg.norm(jig.lin_vel)),
    }

def _update(objects,e):
    global _sample_index,_invalid
    rail,jig = objects["lm_rail"],objects["alignment_jig"]
    # Persist only actual process outcomes. No action, grasp or scripted phase is used.
    if rail.low[2] < TOP-.020 or jig.low[2] < TOP-.020 or abs(rail.pos[1])>.35 or rail.pos[0]>.60:
        _invalid=True
    yaw = _yaw(rail.quat_xyzw)
    rail_y_at_jig = float(rail.pos[1]) + (float(jig.pos[0])-float(rail.pos[0]))*math.tan(yaw)
    engaged = (e["jig_centre_y_error_m"]<.0018 and e["jig_yaw_rad"]<math.radians(1.5)
               and e["jig_bottom_error_m"]<.002 and abs(rail_y_at_jig+.080)<.002
               and rail.low[2]>=TOP-.001 and rail.high[2]<TOP+.020)
    if engaged:
        for i,x in enumerate((.150,.200,.250,.300,.350,.400)):
            if abs(e["jig_x_m"]-x)<.012:
                _sweep_bins.add(i)
    if _trace_path:
        rec={"sample_index":_sample_index,"errors":e,"sweep_bins":sorted(_sweep_bins),"invalid":_invalid,
             "bodies":{k:{"pos_m":v.pos.tolist(),"quat_xyzw":v.quat_xyzw.tolist(),
                           "lin_vel_m_s":v.lin_vel.tolist(),"ang_vel_rad_s":v.ang_vel.tolist(),
                           "aabb_low_m":v.low.tolist(),"aabb_high_m":v.high.tolist()} for k,v in objects.items()}}
        with Path(_trace_path).open("a") as f: f.write(json.dumps(rec,separators=(",",":"))+"\n")
    _sample_index+=1

def success(objects):
    e=_errors(objects)
    return bool(not _invalid and len(_sweep_bins)==6 and e["datum_gap_m"]<=.002
                and e["centre_xy_m"]<=.012 and e["height_m"]<=.0015
                and e["speed_m_s"]<.005 and e["spin_rad_s"]<.10
                and e["jig_park_error_m"]<=.015 and e["jig_bottom_error_m"]<=.002
                and e["jig_speed_m_s"]<.005)

def score(objects):
    e=_errors(objects); _update(objects,e)
    if _invalid:return 0.0
    if success(objects):return 1.0
    return min(.75,.10*len(_sweep_bins)+(.15 if e["datum_gap_m"]<=.002 else 0))

def distance(objects):
    return _errors(objects)["datum_gap_m"]

def demo():
    z=TOP+.047
    hover=TOP+.135
    left=[
        Waypoint(2.0,(.200,.220,hover),FINGER_OPEN,"approach jig post"),
        Waypoint(1.5,(.200,.220,z),FINGER_OPEN,"descend"),
        Waypoint(1.0,(.200,.220,z),FINGER_CLOSED,"grasp jig"),
        Waypoint(1.5,(.200,.220,hover),FINGER_CLOSED,"lift"),
        Waypoint(2.0,(.150,0,hover),FINGER_CLOSED,"over rail starts"),
        Waypoint(2.0,(.150,0,z+.001),FINGER_CLOSED,"engage both shoes"),
        Waypoint(8.0,(.400,0,z+.001),FINGER_CLOSED,"sweep full alignment length"),
        Waypoint(1.5,(.380,0,z+.035),FINGER_CLOSED,"lift off aligned rails"),
        Waypoint(3.0,(.150,0,z+.035),FINGER_CLOSED,"return over rails"),
        Waypoint(1.0,(.150,0,hover),FINGER_CLOSED,"clear fixture"),
        Waypoint(2.0,(.200,.220,hover),FINGER_CLOSED,"return jig"),
        Waypoint(1.5,(.200,.220,z+.002),FINGER_CLOSED,"lower on parking mark"),
        Waypoint(1.0,(.200,.220,z+.002),FINGER_OPEN,"release jig"),
        Waypoint(1.5,(.180,.250,hover),FINGER_OPEN,"clear workpiece"),
    ]
    right=[Waypoint(1.5,(.18,-.24,hover),FINGER_OPEN,"clear fixture"),
           Waypoint(29.0,(.18,-.24,hover),FINGER_OPEN,"remain clear")]
    return {"left":left,"right":right}
