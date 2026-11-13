import unittest
import numpy as np
from clip_capture import capture_groups


class CaptureGeometryTests(unittest.TestCase):
    def setUp(self):
        self.root=np.array([0.,0.,0.,0.,0.,0.,1.])
        self.groups=[]
        for z in (-.00012,-.00004,.00004,.00012):
            q=np.array([[.005,-.05,z],[.040,-.05,z],[.040,.05,z],[.005,.05,z]])
            self.groups.append(q[[[0,1,2],[0,2,3]]])

    def test_four_flat_plies_are_captured_across_full_width(self):
        r=capture_groups(self.groups,self.root,self.root,tolerance=0.)
        self.assertAlmostEqual(r["common_span_m"],.1)
        self.assertTrue(all(abs(p["area_mm2"]-3500)<1e-6 for p in r["plies"]))

    def test_empty_clip_cannot_pass(self):
        displaced=[g+[.1,0,0] for g in self.groups]
        r=capture_groups(displaced,self.root,self.root)
        self.assertEqual(r["common_span_m"],0.)

    def test_missing_one_ply_cannot_pass(self):
        self.groups[2]=self.groups[2]+[0,0,.01]
        r=capture_groups(self.groups,self.root,self.root)
        self.assertEqual(r["common_span_m"],0.)


if __name__=="__main__":unittest.main()
