"""Native YAM joint actions with measured sleeve registration and bounded pose feedback."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from route import grounded_route
from yam_arena import ik
from yam_arena.constants import READY,LEFT_BASE_POS,RIGHT_BASE_POS

class Controller:
    def __init__(self,observer):
        self.o=observer
        p=json.loads(Path(__file__).with_name('motion.json').read_text())
        self.base=np.array(p['base'])
        self.B=R.from_euler('z',p['yaw'],degrees=True)
        self.start=p['start'];self.end=p['end']
        via=json.loads(Path(__file__).with_name('approach.json').read_text())
        self.rows=[dict(phase='pregrasp',q=q,jaw=[1.,1.],a=self.start,s=0.,f=0.,duration=2.3)
                   for q in (via,p['pre'])]+grounded_route(p['keys'], self.B.apply([1,0,0]), p.get('early_shift_control',False))
        self.i=0;self.t0=0.
        self.q0=np.tile(READY,(2,1));self.jaw0=np.ones(2)
        self.ref0=np.array([self.start,0.,0.]);self.last=self.q0.copy()
        self.bias=np.zeros((2,6))
        self.points=np.array([[.262128+v,0,.3556] for v in ([.172,.108] if p['swap'] else [.108,.172])])
    def geometry(self,ref):
        a,s,f=ref
        rotation=self.B*R.from_euler('x',a,degrees=True)*R.from_euler('y',f,degrees=True)
        return self.base-self.B.apply([s,0,0])+rotation.apply(self.points),rotation
    def action(self,t):
        row=self.rows[self.i];phase=row['phase']
        q1=np.array(row['q']);jaw1=np.array(row['jaw'])
        ref1=np.array([row['a'],row['s'],row['f']])
        elapsed=t-self.t0
        duration=row['duration']*(2.0 if phase=='grasp' else 1.)
        f=np.clip(elapsed/duration,0,1);u=f*f*(3-2*f)
        q=self.q0+(q1-self.q0)*u;jaw=self.jaw0+(jaw1-self.jaw0)*u
        ref=self.ref0+(ref1-self.ref0)*u
        _,_,_,a,s,fold,_=self.o.read()
        actual=np.array([np.rad2deg(a),s,np.rad2deg(fold)])
        if phase not in ('pregrasp','open','withdraw','clearance','released'):
            if phase in ('approach','grasp'):
                cmd=actual.copy()
            else:
                # Follow a moving sleeve while preserving a small lead along the commanded motion.
                limits=np.array([4.,.009,4.])
                cmd=actual+np.clip(ref-actual,-limits,limits)
                if phase=='shift':cmd[1]=min(.032,ref[1]+.004)
                if phase=='high':cmd[0]+=1.5
            nominal,old=self.geometry(ref);desired,new=self.geometry(cmd)
            corr=new*old.inv()
            for i,b in enumerate((LEFT_BASE_POS,RIGHT_BASE_POS)):
                if i==0 and phase in ('handoff_open','handoff_clear','fold','seat') and ref[2]>25.01:continue
                pos,rot=ik.tcp_pose(q[i],b)
                goal=desired[i]+corr.apply(pos-nominal[i])
                q[i],_,_=ik.solve(goal,ik.matrix_to_quat_xyzw(corr.as_matrix()@rot),q[i],base_pos=b,iters=12,damping=.022)
        if self.o.measured is not None:
            self.bias=np.clip(self.bias+.20*(self.last-self.o.measured),-.22,.22)
        self.last=q.copy()
        if f>=1 and self.i<len(self.rows)-1:
            self.q0=q1;self.jaw0=jaw1;self.ref0=ref1
            self.i+=1;self.t0=t
        return np.column_stack((q+self.bias,jaw)).ravel()
