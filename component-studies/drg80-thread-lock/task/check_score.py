"""Deterministic grader scenarios only. Inputs are synthetic, not robot traces."""
import json
from dataclasses import replace
from math import radians
from stop_lock_score import Sample, StopLockScore


def verify(hz):
    base=Sample(angle=90,angular_speed=0,piston=radians(87)/300,piston_speed=0,
                opening_force=0,stop_force=0,screw_angle=0,screw_height=.004,
                screw_speed=0,nut_angle=0,nut_speed=0,nut_seat_gap=0,nut_preload=3,
                lever_angle=0,robot_tool_clearance=.04,released=True,nuts_retained=True,
                tools_outside_adjustment=True,tools_racked=True,fingers_open=True,damage=False)
    g=StopLockScore(hz)
    def hold(s,n):
        for _ in range(n):
            g.observe(s,g.tick+1)
        return g.score
    def endpoint(a):
        return replace(base,angle=a,piston=radians(a-3)/300,lever_angle=30,opening_force=5,stop_force=5)
    def motion(a,closing):
        return replace(endpoint(a), stop_force=0,angular_speed=-50 if closing else 50,
                       opening_force=-5 if closing else 5,lever_angle=-30 if closing else 30)
    def cycle():
        for a in range(89,2,-1): hold(motion(a,True),1)
        for a in range(4,91): hold(motion(a,False),1)
        return hold(endpoint(90),hz//4)
    assert hold(endpoint(90),hz)==0
    assert hold(replace(base,nut_preload=1.99),hz)==0
    assert hold(base,hz//4)==0
    assert hold(endpoint(95),hz//4)==.25
    assert hold(endpoint(90),hz//4)==.5
    assert hold(endpoint(90),3*hz)==.5
    assert cycle()==.75
    assert cycle()==.75
    assert cycle()==.75
    assert hold(endpoint(90),hz)==1
    assert hold(replace(endpoint(90),robot_tool_clearance=.029),1)==.75
    assert hold(replace(endpoint(90),nut_seat_gap=.001),1)==0
    assert hold(base,hz//4)==0
    assert hold(endpoint(90),hz//4)==.5
    assert hold(replace(endpoint(90),nut_angle=1),1)==0 and g.damaged
    assert hold(endpoint(90),hz)==0
    return {"hz":hz,"checks":16,"score_before_cycles":.5,"score_after_three_and_clear":1,"damage_latched":True}

if __name__=="__main__":
    print(json.dumps([verify(hz) for hz in (240,480)],indent=2))
