"""Track presolved bimanual joint waypoints. The hands are the only actuated bodies."""
import json
from pathlib import Path
import numpy as np
from yam_arena.constants import READY

class Controller:
    def __init__(self,observer):
        self.observer=observer
        self.rows=json.loads(Path(__file__).with_name('waypoints.json').read_text())
        self.initial=dict(left=READY,right=READY,time=0.,jaw=1.)
        self.bias=np.zeros((2,6))
        self.previous=np.array([READY,READY])

    def action(self,t):
        i=min(sum(t>r['time'] for r in self.rows),len(self.rows)-1)
        a=self.initial if i==0 else self.rows[i-1]
        b=self.rows[i]
        u=np.clip((t-a['time'])/(b['time']-a['time']),0,1)
        u=u*u*(3-2*u)
        q=np.array([np.array(a[k])*(1-u)+np.array(b[k])*u for k in ('left','right')])
        jaw=a['jaw']*(1-u)+b['jaw']*u
        measured=self.observer.joints
        if measured is not None:
            self.bias=np.clip(self.bias+.16*(self.previous-measured),-.20,.20)
        self.previous=q
        return np.column_stack((q+self.bias,[jaw,jaw])).ravel()
