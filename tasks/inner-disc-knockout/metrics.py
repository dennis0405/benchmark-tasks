"""Classify measured poses, including separate inner and outer tab anchors."""
import math
def anchor(p,offset):
    x,y,z,w=p[3:]; h=offset
    return (p[0]+2*h*(x*z+w*y),p[1]+2*h*(y*z-w*x),p[2]+h*(1-2*(x*x+y*y)))
def pitch(p):
    x,y,z,w=p[3:]
    return math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)))
def measure(disc,ring,base):
    fixed=tuple(base)+(0.,0.,0.,1.)
    return dict(inner_angle_deg=pitch(disc)-pitch(ring),outer_angle_deg=pitch(ring),
                inner_gap_m=math.dist(anchor(disc,.0325),anchor(ring,.0325)),
                outer_gap_m=math.dist(anchor(ring,.053),anchor(fixed,.053)),
                center_clearance_m=math.dist(disc[:3],ring[:3]))
class Assessment:
    def __init__(self):
        self.bent=self.damaged=self.detached=False
        self.clear_frames=0
    def sample(self,m,clear=True,contact=True):
        self.damaged |= m['outer_broken'] or abs(m['outer_angle_deg'])>8 or m['outer_gap_m']>.005
        self.bent |= contact and abs(m['inner_angle_deg'])>14 and m['inner_gap_m']<.004
        self.detached |= self.bent and m['inner_broken'] and m['inner_gap_m']>.009
        eligible=self.detached and m['center_clearance_m']>.045 and clear and not self.damaged
        self.clear_frames=self.clear_frames+1 if eligible else 0
        success=self.clear_frames>=20 and not self.damaged
        return dict(score=0. if self.damaged else 1. if success else .5 if self.bent else 0.,
                    state='outer_damaged' if self.damaged else 'inner_removed' if success else 'detached' if self.detached else 'bent_attached' if self.bent else 'untouched',
                    success=success,clear_frames=self.clear_frames)

