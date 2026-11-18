"""Scripted feedback trial. Only 14 public YAM position/gripper targets leave this module."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from yam_arena import ik, station
from yam_arena.constants import READY

RIGHT_GRIP=np.array([-.019,-.083,0.])
TILT_HANDLE=np.array([.055,-.089,-.053])
LEFT_GRIP=np.array([-.105,.066,-.037])
RIGHT_ROT=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
LEFT_ROT=Rotation.from_euler('y',-32,degrees=True).as_matrix()@np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]])

def rotated(deg):
    return Rotation.from_euler('y',deg,degrees=True).as_matrix()

def solve_pose(arm,position,orient,seed):
    def residual(q):
        xyz,R=ik.tcp_pose(q, station.BASES[arm])
        return np.r_[xyz-position, .18*Rotation.from_matrix(orient.T@R).as_rotvec()]
    best=None
    for start in [seed,np.array(READY),[0.,1.5,1.6,.5,0.,-1.8],[.6,.7,1.1,-1.,-.4,1.8]]:
        start=np.clip(start,ik.LOWER,ik.UPPER)
        fit=least_squares(residual,start,bounds=(ik.LOWER,ik.UPPER),max_nfev=60)
        if best is None or fit.cost<best.cost:best=fit
        if np.linalg.norm(fit.fun[:3])<.002 and np.linalg.norm(fit.fun[3:])<.008:break
    return best.x,float(np.linalg.norm(best.fun[:3])),float(np.linalg.norm(best.fun[3:])/.18)

class Controller:
    def __init__(self,observer):
        self.observer=observer
        self.q={arm:np.array(READY,float) for arm in ('left','right')}
        self.grips={'left':1.,'right':1.}
        self.plan=[]
        self.start_t=None
        self.segment=None
        self.target_slide=-.019
        self.last_probe=None
        self.initial=True
        self.verified=set()

    def read(self):
        objects,poses=self.observer.states()
        return np.array(poses['cradle'][:3]),self.observer.measure(objects,poses)

    def grip_pose(self,arm,pitch,slide,back=0.):
        R=rotated(pitch)
        if arm=='left':return self.pivot+R@(LEFT_GRIP-back*LEFT_ROT[:,2]), R@LEFT_ROT
        return self.pivot+R@(RIGHT_GRIP+[slide,-back,0]), R@RIGHT_ROT

    def move(self,arm,angle,slide,back,opening,seconds):
        pos,rot=self.grip_pose(arm,angle,slide,back)
        self.plan.append((arm,pos,rot,opening,seconds))

    def hold(self,seconds):
        self.plan.append((None,None,None,None,seconds))

    def next_sequence(self):
        self.pivot,m=self.read()
        pitch,slide=m['pitch_deg'],m['slide_m']
        if self.last_probe is not None:
            direction, requested=self.last_probe
            residual=pitch-requested
            if abs(residual)<=2. and abs(m['pitch_speed_deg_s'])<2:
                self.verified.add(direction)
            else:
                self.verified.clear()
                # Gravity rotates a lens-heavy assembly nose down (positive pitch).
                error=math.radians(residual)
                self.target_slide=float(np.clip(slide-.003*error,-.045,.036))
            if len(self.verified)==2:
                self.hold(2.)
                return
        adjust=self.initial or not self.verified
        self.initial=False
        if adjust:
            self.move('right',pitch,slide,.045,1.,1.4)
            self.move('right',pitch,slide,0.,1.,.8)
            self.move('right',pitch,slide,0.,.08,.45)
            self.move('right',0,slide,0.,.08,1.6)
            self.move('left',0,0,.055,1.,1.3)
            self.move('left',0,0,0.,1.,.8)
            self.move('left',0,0,0.,.12,.6)
            self.trim_command=self.target_slide
            self.move('right',0,self.trim_command,0.,.08,.9)
            self.plan.extend([('trim',None,None,None,None)]*7)
            self.move('left',0,0,0.,1.,.5)
            self.plan.append(('park','left',None,None,1.1))
        target=30. if -1 in self.verified else -30.
        # Re-measure slide before setting the angle, after the rail has relocked.
        self.plan.append(('test',target,'camera' if adjust else 'cradle',None,None))
        self.last_probe=(-1 if target<0 else 1,target)

    def action(self,clock):
        if self.start_t is None:self.start_t=clock
        if self.segment is None:
            if not self.plan:self.next_sequence()
            arm,pos,orient,opening,seconds=self.plan.pop(0)
            if arm=='trim':
                _,measured=self.read()
                error=self.target_slide-measured['slide_m']
                self.trim_command+=float(np.clip(error,-.006,.006))
                pos,orient=self.grip_pose('right',0,self.trim_command,0.)
                arm,opening,seconds='right',.08,.32
            if arm=='test':
                _,m=self.read()
                a=pos; pitch=m['pitch_deg']
                if orient=='camera':
                    s=m['slide_m']
                    for back,g,dt in [(0.,.08,1.2),(0.,.08,.35),(0.,1.,.28),(.045,1.,.55)]:
                        self.move('right',a,s,back,g,dt)
                else:
                    for angle,back,g,dt in [(pitch,.045,1.,1.4),(pitch,0.,1.,.8),(pitch,0.,.12,.5),
                                            (a,0.,.12,1.7),(a,0.,.12,.5),(a,0.,1.,.4),(a,.045,1.,.6)]:
                        R=rotated(angle)
                        xyz=self.pivot+R@(TILT_HANDLE+[0,-back,0])
                        self.plan.append(('right',xyz,R@RIGHT_ROT,g,dt))
                self.plan.append(('park','right',None,None,.75))
                self.hold(3.6)
                return self.action(clock)
            if arm=='park':
                arm=pos
                self.segment=(arm,self.q[arm].copy(),np.array(READY),self.grips[arm],1.,seconds)
            elif arm is not None:
                start=self.q[arm].copy()
                end,err,ang=solve_pose(arm,pos,orient,start)
                if err>.012 or ang>.13:raise RuntimeError(f'Unreachable {arm} pose at {pos}: {err:.3f} m, {ang:.3f} rad')
                self.segment=(arm,start,end,self.grips[arm],opening,seconds)
            else:self.segment=(None,None,None,0.,0.,seconds)
            self.start_t=clock
        arm,q0,q1,g0,g1,seconds=self.segment
        t=min(1.,max(0.,(clock-self.start_t)/seconds))
        h=t*t*(3-2*t)
        if arm is not None:
            self.q[arm]=q0+h*(q1-q0)
            self.grips[arm]=g0+h*(g1-g0)
        if t>=1.:self.segment=None
        return np.r_[self.observer.compensate('left',self.q['left']),self.grips['left'],
                     self.observer.compensate('right',self.q['right']),self.grips['right']]
