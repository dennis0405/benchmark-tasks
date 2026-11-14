"""Preserve both orientations and nonzero offsets of authored mimic rows."""
import unittest
from types import SimpleNamespace
import numpy as np
from native_joint_mimics import configure


class Array:
    def __init__(self,v):self.v=np.asarray(v)
    def numpy(self):return self.v.copy()
    def assign(self,v):self.v=np.asarray(v)


class MimicTests(unittest.TestCase):
    def test_passive_follower_orientation(self):
        for gains in ([10.,0.],[0.,10.]):
            with self.subTest(gains=gains):
                m=SimpleNamespace(constraint_mimic_count=1,**{k:Array(v) for k,v in dict(
                    joint_mimic_joint=[-1,-1],joint_mimic_coeffs=[[0.,1.],[0.,1.]],
                    joint_target_ke=gains,joint_qd_start=[0,1],constraint_mimic_joint0=[0],
                    constraint_mimic_joint1=[1],constraint_mimic_coef0=[.02],
                    constraint_mimic_coef1=[-2.],constraint_mimic_enabled=[True]).items()})
                rows=configure(m);self.assertEqual(len(rows),1)
                for q1 in (-.01,0.,.03):
                    q=np.array([.02-2*q1,q1]);r=rows[0]
                    self.assertAlmostEqual(q[r['follower']],r['offset']+r['multiplier']*q[r['reference']])
                self.assertEqual(rows[0]['reference'],int(gains[0]==0))


if __name__=='__main__':unittest.main()
