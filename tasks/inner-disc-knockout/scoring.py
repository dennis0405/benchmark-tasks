from metrics import Assessment
class Scorer:
    def __init__(self):
        self.assessment=Assessment()
    def update(self,*,step,objects,root_poses,robot,mechanics):
        return self.assessment.sample(mechanics,clear=mechanics['hands_clear'],contact=mechanics['disc_contact'])

