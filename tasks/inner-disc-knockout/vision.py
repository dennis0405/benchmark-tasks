"""Triangulate an inspection paint dot using two calibrated physical RGB views."""
import numpy as np
from scipy.spatial.transform import Rotation
class DropDetector:
    def __init__(self,z0):
        self.z0=z0;self.streak=0;self.detected=False
    def sample(self,views):
        points=[];pixels=[];counts=[]
        for rgb,eye,quat,K in views:
            r,g,b=np.moveaxis(rgb.astype(np.int16),-1,0)
            yy,xx=np.nonzero((r>80)&(r>1.32*g+10)&(b>1.25*g+8)&(r>.75*b))
            counts.append(len(xx))
            if len(xx)<3:
                pixels.append(None);continue
            u,v=float(np.median(xx)),float(np.median(yy));pixels.append([u,v])
            ray=Rotation.from_quat(quat).apply(np.linalg.solve(K,[u,v,1.]))
            points.append((eye,ray/np.linalg.norm(ray)))
        delta=None;err=None;xyz=None
        if len(points)==2:
            (a,ra),(b,rb)=points
            lengths=np.linalg.lstsq(np.column_stack((ra,-rb)),b-a,rcond=None)[0]
            pa=a+lengths[0]*ra;pb=b+lengths[1]*rb
            err=float(np.linalg.norm(pa-pb));p=(pa+pb)/2;xyz=p.tolist()
            if min(lengths)>0 and err<.006:delta=float(self.z0-p[2])
        self.streak=self.streak+1 if delta is not None and delta>.023 else 0
        self.detected |= self.streak>=2
        return dict(dot_pixels=counts,centroids=pixels,stereo_xyz=xyz,ray_error_m=err,
                    drop_m=delta,confirmed=self.detected)

