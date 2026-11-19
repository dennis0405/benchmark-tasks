"""All tool and weight motion is caused by YAM joint and finger actions."""
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik
from yam_arena.constants import READY
from yam_arena.station import BASES
from task import BASE
L=np.array([.015,.049,-.002])
R=np.array([0.,-.065,.010])
# Carry both objects from their open, separate tool rests.
def path():
    wg=np.array([-.085,.112,.12215]); hg=np.array([-.085,-.075,.142])
    mark=np.array([0.,0.,.1068]); poised=np.array([0.,0.,.16])
    z=np.array([0.,0.,1.]); park=hg.copy()
    return [
        (1.8,'approach',wg+.065*z,hg+.065*z,1,1),
        (1.6,'align',wg,hg,1,1),(.7,'grasp',wg,hg,0,0),
        (1.4,'lift',wg+.035*z,hg+.042*z,0,0),
        (1.7,'carry',mark+.033*z,poised+.035*z,0,0),
        (1.4,'hook',mark,poised+.027*z,0,0),
        (.8,'rest_hook',mark,np.array([0,0,.145]),0,0),
        (.9,'aim',mark,np.array([0,0,.134]),.125,0),
        (.14,'strike',mark,np.array([.004,-.002,.076]),.125,0),
        (.32,'seat',mark,np.array([.002,-.002,.080]),.125,0),
        (.50,'rebound',mark,np.array([0,0,.138]),.115,0),
        (.14,'tap_again',mark,np.array([.004,-.002,.072]),.125,0),
        (.32,'press_after_tap',mark,np.array([.002,-.002,.080]),.125,0),
        (.8,'withdraw',mark,np.array([0,0,.155]),.09,0),
        (.8,'release_clip',mark,poised,1,0),
        (1.,'clear_left',mark+[0,.055,.06],poised,1,0),
        (1.5,'return',wg+.065*z,park+.04*z,1,0),
        (.95,'lower',wg+.065*z,park,1,0),
        (.55,'release_hammer',wg+.065*z,park,1,1),
        (1.0,'hands_clear',wg+.085*z,park+.065*z,1,1),
        (6.,'proof',wg+.085*z,park+.065*z,1,1)]
class Controller:
    def __init__(self,o):
        self.o=o;self.q=np.tile(READY,(2,1)); self.bias=np.zeros((2,6))
        self.points=path();self.ends=np.cumsum([r[0] for r in self.points]);self.targets=[]
        import json
        from pathlib import Path
        raw=json.loads(Path(__file__).with_name('plan.json').read_text())
        self.targets=list(np.stack([raw[k]['q'] for k in ('left','right')],axis=1))
    def action(self,t):
        high=np.tile(READY,(2,1));high[:,1]-=.48
        if t<1.0:
            f=np.clip(t,0,1);q=np.tile(READY,(2,1))+(high-READY)*f*f*(3-2*f)
            self.q=q.copy()
            return np.column_stack((q,np.ones(2))).ravel()
        t-=1.
        i=min(len(self.ends)-1,int(np.searchsorted(self.ends,t)))
        row=self.points[i];d=row[0];f=np.clip((t-(self.ends[i]-d))/d,0,1);u=f*f*(3-2*f)
        start=self.targets[i-1] if i else high
        q=start+(self.targets[i]-start)*u
        init=np.array(self.points[i-1][-2:] if i else (1,1)); jaws=init+(np.array(row[-2:])-init)*u
        holding=('carry','hook','rest_hook','aim','strike','seat','rebound','tap_again','press_after_tap','withdraw')
        active=(holding+('release_clip',),holding+('release_clip','clear_left','return','lower'))
        if row[1]=='rebound':
            c=self.o.bodies['Crown'].get_transforms()[0]
            if f>.75:self.skip_second=self.o.ever_hit and c[2]-BASE[2]<.1045 and abs(c[1]-BASE[1])<.006
        for k,(phases,n,side) in enumerate(zip(active,('Crown','Hammer'),('left','right'))):
            if self.o.q is None or row[1] not in phases:continue
            current=self.o.bodies[n].get_transforms()[0]
            wp=np.array(BASE)+np.array(row[k+2])
            prev=np.array(BASE)+np.array(self.points[i-1][k+2])
            goal=prev+(wp-prev)*u
            if k==0 and row[1] in ('strike','seat','rebound','tap_again','press_after_tap','withdraw') and self.o.ever_hit:
                goal[2]=min(goal[2],float(current[2]))
            if k==1 and getattr(self,'skip_second',False):
                if row[1] in ('tap_again','press_after_tap'):
                    goal=np.array(BASE)+[0,0,.138]
                if row[1]=='withdraw':
                    goal=np.array(BASE)+[0,0,.138+(.155-.138)*u]
            point,ori=ik.tcp_pose(self.o.q[k],BASES[side])
            delta=Rotation.from_quat(current[3:]).inv()
            rotvec=delta.as_rotvec();norm=np.linalg.norm(rotvec)
            delta=Rotation.from_rotvec(rotvec*min(1.,.28/max(norm,1e-9)))
            if np.linalg.norm(current[:3]-point)>(.067 if k==0 else .16):continue
            xyzw=ik.matrix_to_quat_xyzw(delta.as_matrix()@ori)
            target=goal+delta.apply(point-current[:3])
            if row[1]=='release_clip' and k==0 and hasattr(self,'hold_left'):
                q[k]=self.hold_left.copy();continue
            q[k],_,_=ik.solve(target,xyzw,self.o.q[k],base_pos=BASES[side],iters=22,damping=.025)
        if row[1]=='withdraw':self.hold_left=q[0].copy()
        if self.o.q is not None:
            # Slow bias compensation for gravity when holding the tool at a waypoint.
            if row[1] not in ('strike','seat','release_clip'):
                self.bias=np.clip(self.bias+.10*(self.q-self.o.q),-.15,.15)
        self.q=q.copy()
        return np.column_stack((q+self.bias,jaws)).ravel()
