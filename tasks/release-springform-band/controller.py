"""Track presolved bimanual joint waypoints. The hands are the only actuated bodies."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena.constants import READY

class Controller:
    def __init__(self,observer):
        self.observer=observer
        self.rows=json.loads(Path(__file__).with_name('waypoints.json').read_text())
        self.initial=dict(left=READY,right=READY,time=0.,jaws=[1.,1.])
        self.bias=np.zeros((2,6))
        self.previous=np.array([READY,READY])
        self.center_trim=np.zeros((2,3))

    def action(self,t):
        i=min(sum(t>r['time'] for r in self.rows),len(self.rows)-1)
        a=self.initial if i==0 else self.rows[i-1]
        b=self.rows[i]
        u=np.clip((t-a['time'])/(b['time']-a['time']),0,1)
        u=u*u*(3-2*u)
        q=np.array([np.array(a[k])*(1-u)+np.array(b[k])*u for k in ('left','right')])
        jaw=np.array(a['jaws'])*(1-u)+np.array(b['jaws'])*u
        pose=getattr(self.observer,'band_pose',None)
        if pose is not None and 12.35<t<19.5:
            # Use the fixture pose to correct both fingertips together and keep the rim level.
            # The same increments go to physical robot joints; the pan has no pose drive.
            goal_center=np.array([.32,-.097])
            error_xy=goal_center-pose[:2]
            rot=Rotation.from_quat(pose[3:])
            offsets=np.array([[0,.082,.075],[-.119,-.025,.055]])
            droop=(rot.apply(offsets)-offsets)[:,2]
            error=np.column_stack([np.tile(error_xy,(2,1)),-droop])
            self.center_trim=np.clip(self.center_trim+.065*error,-.014,.014)
            delta=np.clip(.42*error+self.center_trim,-.014,.014)
            q+=np.einsum('ijk,ik->ij',np.array(b['correction']),delta)
        elif t>=19.5 and t<23.:
            fade=(23.-t)/3.5
            q+=fade*np.einsum('ijk,ik->ij',np.array(b['correction']),self.center_trim)
        measured=self.observer.joints
        if measured is not None:
            self.bias=np.clip(self.bias+.16*(self.previous-measured),-.20,.20)
        self.previous=q
        return np.column_stack((q+self.bias,jaw)).ravel()
