"""Recovery, steady-load drift, and tamper controls at both supported sample rates."""
import unittest
from math import radians
from stop_lock_score import ElasticLockMotion


class ElasticLockTest(unittest.TestCase):
    def sequence(self,hz,parts):
        m=ElasticLockMotion(hz)
        tick=0
        for n,force,stop,z,angle,released in parts:
            for _ in range(n):
                tick+=1
                m.observe(tick,force,z,angle,0.,stop,released,stopped=True)
        return m

    def test_native_rate_controls(self):
        for hz in (240,480):
            baseline=(hz//4,0,0,0,0,True)
            load=(hz//3,5,6,30e-6,0,True)
            recover=(hz//2,-5,0,0,0,True)
            m=self.sequence(hz,[baseline,load,recover,load,recover,load,recover,load])
            self.assertIsNone(m.first_failure)
            self.assertEqual(m.recoveries,3)
            cases=[
                ([load,(hz//2,-5,0,2.5e-6,0,True)],"permanent_axial_displacement"),
                ([(1,5,0,2e-6,0,True)],"axial_motion_without_stop_load"),
                ([(1,5,6,30e-6,0,False)],"externally_supported_shaft"),
                ([(1,5,6,110e-6,0,True)],"axial_travel_exceeds_setting_band"),
                ([load,(1,5,6,30e-6,radians(.011),True)],"screw_angle_rad"),
                ([load,(hz,5,6,34e-6,0,True)],"axial_drift_at_equal_load"),
            ]
            for seq,channel in cases:
                with self.subTest(hz=hz,channel=channel):
                    self.assertEqual(self.sequence(hz,[baseline,*seq]).first_failure[1],channel)
            m=self.sequence(hz,[baseline,load,(2,-5,0,0,0,True),load])
            self.assertEqual(m.recoveries,0)


if __name__=="__main__":
    unittest.main()
