"""Scripted expert using permitted simulator poses and standard guarded joint actions.

The official review instructions explicitly allow simulator state in a scripted demo.
This policy reads task-local ObjectState snapshots and never writes simulator state.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path

import numpy as np
from inspect_robots.registry import policy
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import STATE_KEY, pack
from yam_arena import ik
from yam_arena.constants import READY, FINGER_OPEN
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy
from yam_arena.urdf import quat_xyzw_to_matrix


def rotation_error(a, b):
    q = ik.matrix_to_quat_xyzw(a @ b.T)
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return np.zeros(3) if length < 1e-10 else q[:3] / length * (2 * math.atan2(length, q[3]))


def limited(vector, length):
    return vector * min(1.0, length / max(float(np.linalg.norm(vector)), 1e-12))


def pose_of(state):
    return np.array(state.pos, copy=True), quat_xyzw_to_matrix(np.array(state.quat_xyzw, copy=True))


def grasp_transform(tcp_position, tcp_rotation, object_position, object_rotation):
    return tcp_rotation.T @ (object_position-tcp_position), tcp_rotation.T @ object_rotation


def tcp_for_object(position, rotation, relative_position, relative_rotation):
    tcp_rotation = rotation @ relative_rotation.T
    return position-tcp_rotation @ relative_position, tcp_rotation


class OracleJigDemo(YamScriptedPolicy):
    def __init__(self, hold_s=5.0, ki=.5):
        super().__init__(hold_s=float(hold_s))
        self.info = replace(self.info, name="rcauto_oracle_jig_demo")
        self.ki = float(ki)
        if not 0 <= self.ki <= 1:
            raise ValueError("tracking gain outside the validated interface range")

    def reset(self, scene):
        super().reset(scene)
        self.cursor = 0.0
        self.grasps = {}
        self.object_goals = {}
        self.last_commands = {}
        self.corrections = {name:np.zeros(6) for name in self._plans}
        self.release_commands = {}
        self.left_recalibrated = False
        self.boundaries = []
        for arm, plan in self._plans.items():
            end = 0.0
            for waypoint in plan.waypoints:
                end += waypoint.duration
                self.boundaries.append((end, arm, waypoint.label))
        self.boundaries.sort()
        name = os.environ.get("RAPID_JOINT_TRACE")
        self.trace = Path(name) if name else None
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            self.trace.write_text("")

    def pregrasp_target(self, arm, states):
        obj = states["alignment_jig" if arm == "left" else "lm_rail"]
        pos, rot = pose_of(obj)
        offset = np.array([0,.040,.0135]) if arm == "left" else np.array([-.150,0,.008])
        return pos+rot@offset, rot@grip_rotation(0)

    def fit(self, states):
        jig,rail = states["alignment_jig"],states["lm_rail"]
        jp,jr = pose_of(jig)
        rp,rr = pose_of(rail)
        endpoint_error = max(abs(float((rp+rr@np.array([s*.160,0,0]))[1])+.080) for s in (-1,1))
        tilt = math.acos(float(np.clip(jr[2,2],-1,1)))
        yaw = abs(math.atan2(jr[1,0],jr[0,0]))
        top = self._task.table_top
        return bool(abs(jp[1]) < .0017 and yaw < math.radians(1.4)
                    and tilt < math.radians(2.8) and abs(float(jig.low[2])-(top+.003)) < .0018
                    and endpoint_error < .002 and abs(rp[2]-(top+.008)) < .0015)

    def ready(self, arm, label, states, actual):
        offset = 0 if arm == "left" else 7
        grip = actual[offset+6]
        if label in {"descend", "descend to rail end"}:
            p,r = self.pregrasp_target(arm,states)
            a,b = ik.tcp_pose(actual[offset:offset+6],self._bases[arm])
            return bool(np.linalg.norm(p-a)<.003 and np.linalg.norm(rotation_error(r,b))<math.radians(3))
        if label in {"grasp jig", "stabilize near end"}:
            return bool(.07<grip<.65)
        if label == "engage both shoes":
            return self.fit(states)
        if label == "traverse held alignment jig":
            return self.fit(states) and abs(float(states["alignment_jig"].pos[0])-.380)<.006
        if label == "lower on parking mark":
            j=states["alignment_jig"]
            return bool(np.linalg.norm(j.pos[:2]-np.array([.200,.220]))<.008
                        and abs(float(j.low[2])-self._task.table_top)<.006)
        if label in {"release jig", "release aligned rail"}:
            return bool(grip>.9)
        return True

    def act(self, observation):
        actual=np.asarray(observation.state[STATE_KEY],dtype=float)
        if actual.shape!=(14,) or not np.isfinite(actual).all():
            raise ValueError("invalid standard joint observation")
        states=getattr(self._task.module,"_latest_objects",None)
        advance=True
        if states:
            left_label=self._plans["left"].target(self.cursor)[3]
            if left_label=="traverse held alignment jig":
                advance=self.fit(states)
                if "left" in self.object_goals:
                    advance = advance and np.linalg.norm(states["alignment_jig"].pos-self.object_goals["left"][0])<.006
        next_cursor=self.cursor+(self._dt if advance else 0.0)
        if states:
            for end,arm,label in self.boundaries:
                if self.cursor < end <= next_cursor+1e-9 and not self.ready(arm,label,states,actual):
                    next_cursor=min(next_cursor,end-1e-7)
        self.cursor=next_cursor
        outputs=[]
        info={}
        oracle_detail={}
        for arm,offset in (("left",0),("right",7)):
            if arm not in self._plans:
                outputs.append(np.r_[READY,1.0]);continue
            plan=self._plans[arm]
            p,r,finger,label=plan.target(self.cursor)
            p,r=np.array(p,copy=True),np.array(r,copy=True)
            nominal_grip=float(np.clip(finger/FINGER_OPEN,0,1))
            tcp_actual,tcp_rot_actual=ik.tcp_pose(actual[offset:offset+6],self._bases[arm])
            obj_name="alignment_jig" if arm=="left" else "lm_rail"
            release=label in {"release jig","release aligned rail"}
            if release:
                if arm not in self.release_commands:
                    self.release_commands[arm]=self.last_commands.get(arm,actual[offset:offset+6]).copy()
                q=self.release_commands[arm].copy()
                outputs.append(np.r_[q,nominal_grip])
                release_target,_=ik.tcp_pose(q,self._bases[arm])
                info[arm]={"phase":label,"actual_tcp_m":tcp_actual.tolist(),"target_tcp_m":release_target.tolist(),"tcp_error_m":float(np.linalg.norm(release_target-tcp_actual)),"release_hold":True}
                if actual[offset+6]>.9:self.grasps.pop(arm,None)
                continue
            before_grasp=label in ({"approach jig post","descend","grasp jig"} if arm=="left" else {"approach near rail end","descend to rail end","stabilize near end"})
            held=not before_grasp and nominal_grip<.4 and label not in {"clear workpiece","clear aligned rail"}
            if states and before_grasp:
                goal,goal_rotation=self.pregrasp_target(arm,states)
                fixed=np.array([.200,.260,self._task.table_top+.038]) if arm=="left" else np.array([.110,-.079,self._task.table_top+.016])
                p += goal-fixed
                if label in {"descend","grasp jig","descend to rail end","stabilize near end"}:r=goal_rotation
            if states and held:
                obj_p,obj_r=pose_of(states[obj_name])
                if arm not in self.grasps and .07<actual[offset+6]<.65:
                    self.grasps[arm]=grasp_transform(tcp_actual,tcp_rot_actual,obj_p,obj_r)
                    self.corrections[arm][:]=0
                if arm=="left" and label=="settle before engagement" and arm in self.grasps and not self.left_recalibrated:
                    body=states[obj_name]
                    if float(body.low[2])>self._task.table_top+.05 and np.linalg.norm(body.lin_vel)<.01 and np.linalg.norm(body.ang_vel)<.2:
                        self.grasps[arm]=grasp_transform(tcp_actual,tcp_rot_actual,obj_p,obj_r)
                        self.left_recalibrated=True
                if arm in self.grasps:
                    goal=p-np.array([0,.040,.0135]) if arm=="left" else np.array([.260,-.080,self._task.table_top+.008])
                    goal_r=np.eye(3)
                    self.object_goals[arm]=(goal.copy(),goal_r.copy())
                    p,r=tcp_for_object(goal,goal_r,*self.grasps[arm])
                    p += .35*limited(goal-obj_p,.010)
                    correction_rotation=ik.slerp_matrix(np.eye(3),goal_r@obj_r.T,.25)
                    r=correction_rotation@r
                    oracle_detail[arm]={"object":obj_name,"actual_m":obj_p.tolist(),"goal_m":goal.tolist(),"error_m":float(np.linalg.norm(goal-obj_p))}
            q,pe,re=ik.solve(p,ik.matrix_to_quat_xyzw(r),actual[offset:offset+6],base_pos=self._bases[arm],iters=80)
            self.corrections[arm]=np.clip(self.corrections[arm]+self.ki*self._dt*(q-actual[offset:offset+6]),-.10,.10)
            command=q+self.corrections[arm]
            self.last_commands[arm]=command.copy()
            outputs.append(np.r_[command,nominal_grip])
            info[arm]={"phase":label,"actual_tcp_m":tcp_actual.tolist(),"target_tcp_m":p.tolist(),"tcp_error_m":float(np.linalg.norm(p-tcp_actual)),"ik_position_error_m":pe,"ik_rotation_error_rad":re}
        command=pack(*outputs)
        if not np.isfinite(command).all():raise ValueError("non-finite expert command")
        if self.trace:
            record={"kind":"oracle-scripted-expert","state_time_s":float(observation.state_time),"plan_time_s":self.cursor,"actual_joint_pos":actual.tolist(),"command":command.tolist(),"arms":info,"object_targets":oracle_detail,"physical_fit":bool(states and self.fit(states))}
            with self.trace.open("a") as stream:stream.write(json.dumps(record,allow_nan=False,separators=(",",":"))+"\n")
        meta={"request_stop":True} if self.cursor>self._end+self.hold_s else {}
        return ActionChunk(actions=[Action(data=command,meta=meta)],control_hz=1/self._dt)


@policy("rcauto_oracle_jig_demo")
def make_oracle_demo(**kwargs):
    return OracleJigDemo(**kwargs)


if __name__=="__main__":
    from inspect_robots.cli import main
    main()
