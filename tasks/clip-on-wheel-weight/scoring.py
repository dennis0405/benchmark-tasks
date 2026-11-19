"""Check the strike, initial engagement and native retention during the tensile test."""
import math

MIN_CLOSING_SPEED = 0.12
class Scorer:
    def __init__(self):
        self.held=self.hooked=self.struck=self.engaged=False
        self.frames=0
        self.proof_reference=None
    def update(self,*,step,objects,root_poses,robot,mechanics):
        m=mechanics
        self.held |= m['weight_held'] and m['off_mark']
        self.hooked |= self.held and m['on_mark'] and m['rim_contact'] and m['weight_held']
        self.struck |= self.hooked and m['hammer_held'] and m['fast_contact_impulse']>.002
        self.engaged |= self.struck and m['on_mark'] and m['under_lip'] and m['pull_N']==0.
        # Both toes pass the underside at installation. The clip takes up clearance when
        # tensioned. Test restraint under load with measured rim contact and displacement.
        at_load=(self.engaged and m['hammer_parked'] and m['both_clear'] and m['on_mark'] and m['full_load']
                 and m['rim_contact'] and max(t[2] for t in m['toes'])<.098)
        if at_load and self.proof_reference is None:self.proof_reference=tuple(m['crown'])
        drift=math.dist(self.proof_reference,m['crown']) if self.proof_reference else None
        stable=at_load and drift<=.001
        self.frames=self.frames+1 if stable else 0
        success=self.frames>=60
        return dict(score=.25*(int(self.hooked)+int(self.struck))+.5*int(success),success=success,
                    held_frames=self.frames,hooked=self.hooked,struck=self.struck,engaged=self.engaged,loaded_drift_m=drift)
