"""Backward Euler reference for the two-DOF isolated N73 coupon with two rigid stops.

A unilateral normal reaction enforces each plate, and a single conjugate effort
acts on both coordinates. No data from the engine under test enter the solve.
"""
from math import tau
from statistics import median
import json
import sys

P = .001 / tau
K, D = 1e5, 2500.
M = .009001
I = 1 / (1/7e-8 + 1/2.6e-7)
GAP = .0001


def run(hz):
    h = 1/hz
    z=angle=v=w=0.
    result=[]
    for sign in (1,-1):
        held=[]
        unload=[]
        for i in range(4*hz):
            torque = sign*.005*(min(1,(i+1)*h) if i<3*hz else 0.)
            e=z-P*angle
            b=D+h*K
            w_free=w+h*torque/I
            spring=(K*e+b*(v-P*w_free))/(1+b*h*(1/M+P*P/I))
            vn=v-h*spring/M
            wn=w_free+h*P*spring/I
            reaction=0.
            side=1 if z+h*vn>GAP else -1 if z+h*vn < -GAP else 0
            if side:
                vn=(side*GAP-z)/h
                spring=(K*e+b*(vn-P*w_free))/(1+b*h*P*P/I)
                wn=w_free+h*P*spring/I
                reaction=(vn-v)*M/h+spring
                assert -side*reaction>=-1e-9
            z,angle=z+h*vn,angle+h*wn
            v,w=vn,wn
            assert abs(spring-K*(z-P*angle)-D*(v-P*w))<1e-7
            if 2*hz<=i<3*hz:
                held.append((z,angle,z-P*angle,abs(reaction)))
            if i>=4*hz-hz//4:
                unload.append(abs(z-P*angle))
        zmed,amed,emed,fmed=map(median,zip(*held))
        assert abs(emed+sign*.005/(K*P))<1e-7 and abs(fmed-.005/P)<.01
        assert max(unload)<20e-6
        result.append(dict(direction=sign,z_m=zmed,relative_rad=amed,extension_m=emed,
                           plate_force_N=fmed,released_max_abs_e_m=max(unload)))
    return dict(hz=hz,trials=result)


if __name__=="__main__":
    json.dump([run(240),run(480)],sys.stdout,indent=2)
    print()
