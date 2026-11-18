"""Bounded physical-contact prototype controller; no apparatus state writes."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from yam_arena import ik,station
from yam_arena.constants import READY

DOWN=np.diag([1.,-1.,-1.])
PUMP_ROT=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
VALVE_ROT=Rotation.from_euler('z',180,degrees=True).as_matrix()@DOWN

def solve(arm,p,R,seed):
    def residual(q):
        pos,rot=ik.tcp_pose(q,station.BASES[arm])
        return np.r_[pos-p,.18*Rotation.from_matrix(R.T@rot).as_rotvec()]
    best=None
    for start in (seed,np.array(READY),np.array([0.,1.5,1.6,.5,0.,-1.8]),np.array([.4,.8,1.2,-1.,-.3,1.8])):
        fit=least_squares(residual,np.clip(start,ik.LOWER,ik.UPPER),bounds=(ik.LOWER,ik.UPPER),max_nfev=70)
        if best is None or fit.cost<best.cost:best=fit
        if np.linalg.norm(fit.fun[:3])<.002 and np.linalg.norm(fit.fun[3:])<.01:break
    if np.linalg.norm(best.fun[:3])>.003 or np.linalg.norm(best.fun[3:])>.016:
        raise RuntimeError(f'Unreachable {arm} waypoint {p}: {best.fun}')
    return best.x

class Controller:
    def __init__(self,observer):
        self.o=observer;self.q={s:np.array(READY,float) for s in ('left','right')}
        self.g={s:1. for s in self.q};self.queue=[];self.segment=None;self.phase='approach'
        self.adjustments=0;self.test_pressure=None;self.t0=None;self.close_index=0
        self.pump_target=25.
        self.reopen_attempts=0

    def move(self,arm,p,R,g,seconds):self.queue.append((arm,p,R,g,seconds))
    def wait(self,seconds):self.queue.append((None,None,None,None,seconds))
    def park(self,arm):self.queue.append(('park',arm,None,None,1.1))

    def lever(self,angle,back,g,seconds):
        _,poses=self.o.states();pivot=np.array(poses['lever'][:3])
        R=Rotation.from_euler('y',angle,degrees=True).as_matrix()
        p=pivot+R@np.array([-.30,-.035,0.])-back*(R@PUMP_ROT)[:,2]
        self.move('right',p,R@PUMP_ROT,g,seconds)

    def valve(self,name,target):
        _,poses=self.o.states();pivot=np.array(poses[name][:3]);start=math.degrees(self.o.angle(name))
        reopening = name == 'isolation' and target == 0
        radius = .063 if reopening else .047
        approach = .47 if reopening else 1.
        height = .005 if reopening else 0.
        for angle,z,g,dt in [(start,.035,approach,1.2),(start,height,approach,.7),(start,height,.12,.5)]+[(a,height,.12,.45) for a in np.linspace(start,target,7)[1:]]+[(target,height,approach,.4),(target,.035,approach,.7)]:
            R=Rotation.from_euler('z',float(angle),degrees=True).as_matrix()
            self.move('left',pivot+R@[-radius,0.,0.]+[0.,0.,z],R@(DOWN if name=='selector' else VALVE_ROT),g,dt)
        self.park('left')

    def plan(self):
        pressure=self.o.model.test_bar
        angle=math.degrees(self.o.angle('lever'))
        if self.phase=='approach':
            self.lever(angle,.035,1.,1.6);self.phase='insert'
        elif self.phase=='insert':
            self.lever(angle,0.,1.,.8);self.phase='close'
        elif self.phase=='close':
            openings=(.75,.5,.3,.1)
            self.lever(angle,0.,openings[self.close_index],.2)
            self.close_index+=1
            if self.close_index==len(openings):self.phase='prime'
        elif self.phase=='prime':
            # Pulling up opens the inlet check and fills the cylinder. Starting
            # at its lower stop avoids an unsupported raised handle at reset.
            self.lever(25.,0.,.1,1.8);self.phase='pump'
        elif self.phase=='pump':
            if pressure>=3.04:
                self.valve('isolation',90);self.phase='release_pump'
            elif self.adjustments>=45:raise RuntimeError('Pump pressure not reached in 45 measured adjustments')
            else:
                # Small downstrokes driven by measured gauge pressure, not a
                # precomputed amount of artificial volume per action.
                delta=min(1.,max(.12,(3.30-pressure)*.6))
                # Integrate the command to overcome finite servo deflection;
                # cap command/measured separation to avoid an uncontrolled push.
                self.pump_target=max(.5,angle-14.,min(self.pump_target,angle)-delta)
                self.lever(self.pump_target,0.,.1,.7);self.adjustments+=1
        elif self.phase=='release_pump':
            self.lever(angle,0.,1.,.4);self.lever(angle,.035,1.,.8);self.park('right')
            self.phase='isolate'
        elif self.phase=='isolate':
            self.test_pressure=pressure;self.wait(61.);self.phase='report'
        elif self.phase=='report':
            self.valve('selector',35 if self.test_pressure-pressure>.2 else -35)
            self.phase='reopen'
        elif self.phase=='reopen':
            if abs(math.degrees(self.o.angle('isolation'))) <= 8.:
                self.valve('drain',90);self.phase='finish'
            elif self.reopen_attempts < 4:
                self.valve('isolation',0)
                self.reopen_attempts += 1
            else:
                raise RuntimeError('Could not park isolation valve at open stop after 4 physical turns')
        else:self.wait(4.)

    def action(self,clock):
        if self.segment is None:
            if not self.queue:self.plan()
            arm,p,R,g,seconds=self.queue.pop(0)
            if arm=='park':arm,p=p,None;q1=np.array(READY);g=1.
            elif arm is not None:q1=solve(arm,p,R,self.q[arm])
            if arm is None:self.segment=(None,None,None,None,None,seconds)
            else:self.segment=(arm,self.q[arm].copy(),q1,self.g[arm],g,seconds)
            self.t0=clock
        arm,q0,q1,g0,g1,seconds=self.segment
        t=min(1.,max(0.,(clock-self.t0)/seconds));w=t*t*(3-2*t)
        if arm is not None:self.q[arm]=q0+w*(q1-q0);self.g[arm]=g0+w*(g1-g0)
        if t>=1.:self.segment=None
        return np.r_[self.q['left'],self.g['left'],self.q['right'],self.g['right']]
