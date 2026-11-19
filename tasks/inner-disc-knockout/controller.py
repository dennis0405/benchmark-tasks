"""Fourteen YAM joint/gripper actions. No object state writes or applied forces."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from yam_arena import ik
from yam_arena.station import BASES
from yam_arena.constants import READY
from task import BASE
def make_path():
    # Points are relative to the concentric mark, outside (+Y) and inside (-Y).
    L=(0,.090,.048);R=(0,-.10,.058)
    initial=(0,.066,-.010);press=(0,-.013,-.012)
    aim=(0,-.060,-.007);edge=(0,.009,-.007);rock=(0,.024,-.007)
    return [(1.8,L,R),(1.0,initial,R),(.95,press,R),(.30,press,R),
            (.75,initial,R),(.65,L,aim),(1.30,L,edge),(1.15,L,rock),(.25,L,rock),
            (.95,L,aim),(.8,L,R),(2.0,L,R)]
class Controller:
    def __init__(self,observer):
        self.o=observer;self.path=make_path(); self.ends=np.cumsum([r[0] for r in self.path])
        self.targets=[];self.feedback=np.zeros((2,6));self.last=None
        for j,side in enumerate(('left','right')):
            sign=1 if j==0 else -1
            incline=.33
            z=np.array([0,-sign*np.cos(incline),-np.sin(incline)])
            x=np.array([1.,0,0]); rot=np.column_stack((x,np.cross(z,x),z))
            qs=[];q=np.array(READY);errs=[]
            for row in self.path:
                pos=np.array(BASE)+row[j+1]
                def residual(v):
                    pt,r=ik.tcp_pose(v,BASES[side])
                    return np.r_[pt-pos,.17*Rotation.from_matrix(rot@r.T).as_rotvec()]
                q=least_squares(residual,q,bounds=(ik.LOWER,ik.UPPER),max_nfev=75).x
                qs.append(q.copy());errs.append(round(float(np.linalg.norm(residual(q))),4))
            print('IK residuals',side,errs,flush=True);self.targets.append(qs)
    def action(self,t):
        i=min(len(self.path)-1,int(np.searchsorted(self.ends,t)));d=self.path[i][0]
        u=np.clip((t-self.ends[i]+d)/d,0,1);u=u*u*(3-2*u)
        prev=np.array([self.targets[j][i-1] if i else READY for j in range(2)])
        goal=np.array([self.targets[j][i] for j in range(2)])
        q=prev+(goal-prev)*u
        if self.o.q is not None and self.last is not None:
            self.feedback=np.clip(self.feedback+.095*(self.last-self.o.q),-.095,.095)
        if 'InnerTab' in self.o.breaks.names and i>=6:
            if not hasattr(self,'release'):
                self.release=(t,self.o.q[1].copy())
            when,at=self.release
            f=min(1.,(t-when+.05)/.45);f=f*f*(3-2*f)
            q[1]=(1-f)*at+f*self.targets[1][-1]
            self.feedback[1]=0.
        self.last=q.copy()
        return np.column_stack((q+self.feedback,[0.,0.])).ravel()

