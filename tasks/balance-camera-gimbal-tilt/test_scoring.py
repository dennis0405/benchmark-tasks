import unittest
from scoring import Scorer

class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.s=Scorer()
        self.k=0

    def sample(self, pitch=30., slide=-.019, brake=0., clear=True, speed=0., contact=0.):
        self.k+=1
        return self.s.update(step=self.k,mechanics=dict(pitch_deg=pitch,pitch_speed_deg_s=speed,
                   slide_m=slide,brake_m=brake,both_hands_clear=clear,hand_impulse_Ns=contact))['score']

    def adjust(self):
        self.sample(slide=0.,brake=.010,clear=False)
        self.sample(slide=-.019,brake=.010,clear=False)
        self.sample(clear=False)

    def hold(self,angle,steps):
        for _ in range(steps):self.sample(pitch=angle)

    def test_two_actual_releases(self):
        self.adjust()
        self.hold(30.,60);self.assertEqual(self.s.score,0.)
        self.hold(30.,1);self.assertEqual(self.s.score,.5)
        self.sample(pitch=-30,clear=False)
        self.hold(-30.,61);self.assertEqual(self.s.score,1.)
        self.sample(brake=.01,clear=False);self.assertEqual(self.s.score,0.)

    def test_no_motion_and_clamped_support(self):
        self.hold(30.,65);self.assertEqual(self.s.score,0.)
        self.adjust()
        for _ in range(65):self.sample(contact=.01)
        self.assertEqual(self.s.score,0.)

    def test_drift_and_repeated_pose(self):
        self.adjust();self.hold(30.,35);self.sample(pitch=24);self.hold(30.,65)
        self.assertEqual(self.s.score,0.)
        self.sample(clear=False);self.hold(30,65);self.sample(clear=False);self.hold(30,65)
        self.assertEqual(self.s.score,.5)
        self.sample(slide=-.017);self.assertEqual(self.s.score,0.)
