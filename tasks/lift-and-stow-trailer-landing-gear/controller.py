"""Full-stroke YAM crank controller. Progress comes from the passive measured input shaft."""
import json
from pathlib import Path
import numpy as np
from limits import HIGH_LEAD, STROKE
from yam_arena import ik
from yam_arena.constants import READY,RIGHT_BASE_POS
class Controller:
    def __init__(self,observer):
        self.o=observer
        self.path=json.loads(Path(__file__).with_name('orbit.json').read_text())
        self.start=self.path['start']; self.end=self.path['fold_angle']
        self.shift_angle=262.3
        self.next_regrip=None
        self.stage='approach';self.t0=0.;self.last=np.array(READY);self.qfrom=self.last.copy()
        self.bias=np.zeros(6);self.regrips=0;self.clock=0.
    def orbit(self,angle,mode):
        a=(angle%360)/5
        i=int(a);u=a-i
        pts=self.path[mode]
        return (1-u)*np.array(pts[i])+u*np.array(pts[i+1])
    def enter(self,state,t):
        self.stage=state;self.t0=t;self.qfrom=self.last.copy()
        if state in ('brake','shift','parkalign'):
            self.stop_angle=np.degrees(self.o.read()[3])
        if state=='release':
            self.correction=np.zeros(3)
            self.bias[:]=0.
            self.center_stable=0
    def action(self,t):
        phase=self.stage;elapsed=t-self.t0
        angle=np.degrees(self.o.read()[3])
        low=np.degrees(self.o.low);high=np.degrees(self.o.high)
        jaw=.09
        def move(goal,dt):
            u=np.clip(elapsed/dt,0.,1.);u=u*u*(3-2*u)
            return self.qfrom*(1-u)+np.asarray(goal)*u
        if phase=='approach':
            q=move(self.path['pre'],2.4);jaw=1.
            if elapsed>=2.4:self.enter('align',t)
        elif phase=='align':
            q=move(self.orbit(angle,'low'),1.0);jaw=1.
            if elapsed>=1.0:self.enter('grasp',t)
        elif phase=='grasp':
            q=self.orbit(angle,'low')
            if elapsed>=.9:
                self.low_goal=low+(self.shift_angle-angle)%360
                while self.low_goal*.79375/360<3.05:self.low_goal+=360
                self.enter('low',t)
        elif phase in ('low','high'):
            value=low if phase=='low' else high
            if phase=='low':
                goal=self.low_goal
            else:
                if self.next_regrip is None:
                    offset=(self.end-angle)%360
                    self.next_regrip=high+(offset if offset>=30 else offset+360)
                # Aim 0.07 mm into the compliant end stop to allow the shoe to settle.
                remaining=(STROKE+.00007-self.o.travel)/HIGH_LEAD*360.
                end_value=high+remaining
                braking_for_regrip=self.next_regrip<end_value-18.
                goal=self.next_regrip if braking_for_regrip else end_value
            if not hasattr(self,'ref') or self.ref_phase != phase or elapsed<=.051:
                self.ref=value
            self.ref_phase=phase
            self.ref=min(goal+2.5,value+3.2,self.ref+(20. if phase=='low' else 26.)*.05)
            lead=np.clip(self.ref-value,-3.2,3.2)
            q=self.orbit(angle+lead,phase)
            if value>=goal:
                if phase=='low':self.enter('shift',t)
                elif braking_for_regrip:
                    self.regrips+=1;self.next_regrip=None;self.enter('brake',t)
                else:self.enter('parkalign',t)
        elif phase=='brake':
            q=self.orbit(angle,'high')
            if elapsed>=.80:self.enter('release',t)
        elif phase=='shift':
            q=move(self.orbit(self.shift_angle,'high'),1.2)
            pos,rot=ik.tcp_pose(q,RIGHT_BASE_POS)
            pos=pos+np.array([np.clip((elapsed-.65)*.025,0,.031),0.,0.])
            q,_,_=ik.solve(pos,ik.matrix_to_quat_xyzw(rot),q,base_pos=RIGHT_BASE_POS,iters=16,damping=.015)
            if elapsed>=1.55 and self.o.read()[4]>.025:self.enter('high',t)
        elif phase in ('release','reclose'):
            q=self.orbit(angle,'high')
            jaw=.88 if phase=='release' else .09
            if phase=='release' and elapsed>=1.25 and self.center_stable>=4:self.enter('reclose',t)
            if phase=='reclose' and elapsed>=.85 and len(self.o.contacts)>=2:self.enter('high',t)
        elif phase=='parkalign':
            err=(self.end-angle+180.)%360.-180.
            q=self.orbit(angle+np.clip(err,-2.8,2.8),'high')
            if elapsed>=.85 and abs(err)<1.4:self.enter('fold',t)
        elif phase=='fold':
            a=np.clip(elapsed/4.5*30,0,30);i=min(29,int(a));u=a-i
            q=(1-u)*np.array(self.path['fold'][i])+u*np.array(self.path['fold'][i+1])
            if elapsed>=5.5:self.enter('open',t)
        elif phase=='open':
            q=np.array(self.path['fold'][-1]);jaw=1.
            if elapsed>=.65:self.enter('withdraw',t)
        else:
            q=move(self.path['withdraw'],1.5);jaw=1.
        if phase in ('low','high','brake','release','reclose'):
            # Follow the true sleeve center; a slightly bent fold hinge displaces it axially.
            p,w,h,theta,selector,phi,_=self.o.read()
            from scipy.spatial.transform import Rotation as R
            grip=np.array([.404628,0.,.3556])
            target_a=(self.stop_angle if phase=='brake' else angle)+(lead if phase in ('low','high') else 0)
            desired=(R.from_euler('z',180,degrees=True)*R.from_euler('x',target_a,degrees=True)).as_matrix()
            axial_shift=.028 if phase in ('high','brake','release','reclose') else 0.
            preload=min(.012,max(0.,(.028-selector)*1.2)) if axial_shift else 0.
            center=p['gearbox'][:3]+[axial_shift+preload,0.,0.]
            tip=center+desired@(R.from_euler('y',phi).as_matrix()@grip)
            if phase in ('release','reclose'):
                tcp,_=ik.tcp_pose(self.o.measured[1],RIGHT_BASE_POS)
                current=p['crank'][:3]+h@grip
                error=current-tcp
                self.center_stable=self.center_stable+1 if np.linalg.norm(error)<.005 else 0
                if phase=='release':self.correction=np.clip(self.correction+.18*error,-.035,.035)
                tip=current+self.correction
            _,rot=ik.tcp_pose(q,RIGHT_BASE_POS)
            q,_,_=ik.solve(tip,ik.matrix_to_quat_xyzw(rot),q,base_pos=RIGHT_BASE_POS,iters=10,damping=.021)
        if self.o.measured is not None:
            self.bias=np.clip(self.bias+.055*(self.last-self.o.measured[1]),-.07,.07)
        self.last=q.copy()
        self.o.pilot=dict(phase=phase,regrips=self.regrips)
        return np.r_[READY,1.,q+self.bias,jaw]
