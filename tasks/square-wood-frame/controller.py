import math
import numpy as np
from yam_arena import station, ik
from yam_arena.constants import READY
from layout import BASE, HALF, START_X, GRIP_Z

class Controller:
    def __init__(self):
        self.last=np.array([READY, READY]).copy()
        self.bias=np.zeros((2,6))
        self.sides=('left','right')
        self.coord=0.
        self.stop=None
        self.rots=[]
        for side in self.sides:
            jaw=np.array([0.,-1.,0.])
            a=np.array([.46,0.,-math.sqrt(1-.46**2)])
            self.rots.append(ik.matrix_to_quat_xyzw(np.column_stack([jaw,np.cross(a,jaw),a])))
    def action(self,t,delta,measured):
        # The two ends travel together until the jaws surround the frame.
        # During tightening the commanded advance uses the observed diagonal difference.
        if t>=14 and self.stop is None:
            if abs(delta)<.00085:
                self.stop=t
            else:
                self.coord+=np.clip(delta*.22,-.006,.006)*.05
                self.coord=np.clip(self.coord,0,.047)
        since=0 if self.stop is None else max(0,t-self.stop)
        rows=[
          (0,START_X,.092,0.,.75), (2.6,START_X,.067,0.,.75),
          (4.,START_X,0.,0.,.65), (5.,START_X,0.,0.,0.),
          (7.5,START_X,.073,0.,0.), (10.,0.,.073,0.,0.),
          (12.,0.,.002,0.,0.), (13.,0.,0.,0.,0.)]
        if t>13:
            x=0.; z=min(.12,max(0,since-1.35)*.075); close=self.coord
            jaw=min(.9,max(0,since-.45)*1.3)
        else:
            b=next(i for i in range(1,len(rows)) if rows[i][0]>=t)
            start,end=np.array(rows[b-1]),np.array(rows[b])
            u=np.clip((t-start[0])/(end[0]-start[0]),0,1);u=u*u*(3-2*u)
            _,x,z,close,jaw=start+u*(end-start)
        qs=[]
        for i,side in enumerate(self.sides):
            sign=1-2*i
            goal=BASE+np.array([x,sign*(HALF-close/2),GRIP_Z+z])
            q,err,ang=ik.solve(goal,self.rots[i],self.last[i],station.BASES[side],iters=18,damping=.012)
            if t<2.6:
                u=t/2.6;v=u*u*(3-2*u);q=(1-v)*np.array(READY)+v*q
            qs.append(q)
        self.bias=np.clip(self.bias+.12*(self.last-measured),-.16,.16)
        self.last=np.array(qs)
        return np.column_stack([self.last+self.bias,[jaw,jaw]]).ravel()
