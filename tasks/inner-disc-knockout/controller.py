import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yam_arena import ik
from yam_arena.station import BASES
from yam_arena.constants import READY
from vision import DropDetector
from task import BASE
class Controller:
    def __init__(self,observer):
        self.o=observer;self.vision=DropDetector(BASE[2]);self.release=None;self.bias=np.zeros(6);self.last=None
        # The opposite gripper stays at its initial posture. Only the exterior
        # blade enters the work envelope. Targets locate the actual blade tip.
        # Advance the flat blade briskly against the lower edge. The short
        # stroke leaves space for the slug to travel freely into the enclosure.
        self.times=np.array([2.,3.1,4.15,4.45,4.70,5.10,5.50])
        self.points=[(0,-.065,.040),(0,-.027,.007),(0,-.013,.008),
                     (0,.014,.007),(0,.023,.004),(0,.011,.004),(0,-.001,.004)]
        z=np.array([0,np.cos(.12),np.sin(.12)])
        x=np.array([1.,0,0]);rot=np.column_stack((x,np.cross(z,x),z))
        self.targets=[];q=np.array(READY)
        for offset in self.points:
            target=np.array(BASE)+offset-.078*z
            def residual(v):
                p,r=ik.tcp_pose(v,BASES['right'])
                return np.r_[p-target,.18*Rotation.from_matrix(rot@r.T).as_rotvec()]
            q=least_squares(residual,q,bounds=(ik.LOWER,ik.UPPER),max_nfev=90).x
            self.targets.append(q.copy())
    def action(self,t):
        signal=self.vision.sample(self.o.rgb())
        self.o.visual=signal
        if signal['confirmed'] and self.release is None:
            self.release=(t,self.o.q[1].copy() if self.o.q is not None else np.array(READY))
        if self.release is not None:
            start,at=self.release
            elapsed=t-start+.05
            if elapsed<.85:
                u=elapsed/.85;u=u*u*(3-2*u)
                q=at*(1-u)+self.targets[1]*u
            else:
                u=min(1.,(elapsed-.85)/1.1);u=u*u*(3-2*u)
                q=self.targets[1]*(1-u)+np.array(READY)*u
        else:
            i=min(len(self.times)-1,int(np.searchsorted(self.times,t)))
            lo=self.times[i-1] if i else 0.
            u=np.clip((t-lo)/(self.times[i]-lo),0.,1.);u=u*u*(3-2*u)
            prev=self.targets[i-1] if i else np.array(READY)
            q=prev*(1-u)+self.targets[i]*u
        if self.o.q is not None and self.last is not None:
            self.bias=np.clip(self.bias+.08*(self.last-self.o.q[1]),-.09,.09)
        self.last=q.copy()
        return np.r_[READY,1.,q+self.bias,0.]

