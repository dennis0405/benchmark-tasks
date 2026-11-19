"""Score visible marks, an unobstructed aperture and a hands-off final hold."""
import math
import numpy as np
from scipy.spatial.transform import Rotation

class Scorer:
    def __init__(self, radius=.01415, hole=.0143, thickness=.00152, outer=.0173, final_step=240, witness_sign=-1):
        self.radius=radius
        self.hole=hole
        self.thickness=thickness
        self.outer=outer
        self.final_step=final_step
        self.origin=None
        self.witness_sign=witness_sign
        self.partial=False
        self.damage=False
        self.frames=0
        self.axes=np.stack((np.cos(np.arange(180)*math.pi/180),np.zeros(180),
                            np.sin(np.arange(180)*math.pi/180)),axis=1)
    def update(self,*,step,objects,root_poses,robot,mechanics):
        d=np.asarray(root_poses['Disc']);r=np.asarray(root_poses['Ring'])
        if self.origin is None:self.origin=r[:3].copy()
        nd=Rotation.from_quat(d[3:]).apply([0.,1.,0.])
        nr=Rotation.from_quat(r[3:]).apply([0.,1.,0.])
        ring_deg=math.degrees(math.acos(float(np.clip(nr[1],-1,1))))
        ring_shift=float(np.linalg.norm(r[:3]-self.origin))
        # A flush, detached ring can jam in its kerf. In the simulator the
        # retention inspection is exact; on a bench check the ring after the run.
        retention=mechanics.get('physics_audit',mechanics)
        self.damage |= ring_deg>8. or ring_shift>.002 or bool(retention.get('outer_broken',False))
        # Separating-axis test in the frontal inspection view. Project the full
        # thickness and rim of the slug, including a slug parallel to the opening.
        dot=self.axes@nd
        reach=self.radius*np.sqrt(np.maximum(0.,1.-dot**2))+self.thickness/2*np.abs(dot)
        gap=float(np.max(np.abs(self.axes@(d[:3]-r[:3]))-reach)-self.hole)
        tilt=math.degrees(math.acos(float(np.clip(abs(nd@nr),0,1))))
        # Compare the paired visible paint marks at the ligament end of the disk.
        top=np.array([0.,0.,self.witness_sign*self.radius])
        witness=float(np.linalg.norm(d[:3]+Rotation.from_quat(d[3:]).apply(top)-
                                     r[:3]-Rotation.from_quat(r[3:]).apply(top)))
        self.partial = tilt>14 and witness<.004 and gap<=0. and not retention.get('inner_broken',False)
        clear=bool(mechanics['hands_clear'])
        self.frames=self.frames+1 if gap>.002 and clear and not self.damage else 0
        full=self.frames>=20
        score=0. if self.damage else 1. if full else .5 if self.partial else 0.
        return dict(score=score,success=bool(full and step>=self.final_step),clear_frames=self.frames,
                    projected_gap_mm=round(1000*gap,3),ring_shift_mm=round(1000*ring_shift,3),
                    ring_tilt_deg=round(ring_deg,3),slug_tilt_deg=round(tilt,2),
                    witness_gap_mm=round(witness*1000,3),outer_damage=bool(self.damage))

