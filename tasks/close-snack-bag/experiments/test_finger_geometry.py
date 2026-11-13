import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from finger_geometry import low_offset


class FingerEnvelopeTests(unittest.TestCase):
    def test_round_shapes_and_proposed_orientation_are_included(self):
        def array(x):return SimpleNamespace(numpy=lambda:np.asarray(x))
        origin=np.array([.7,.1,1.])
        model=SimpleNamespace(
            shape_body=array([0,0,0,1]),shape_type=array([7,4,3,3]),
            shape_scale=array([[.006,.002,.04],[.01,.01,0],[.005,0,0],[.1,0,0]]),
            shape_transform=array([[0,0,0,0,0,0,1],[.02,0,-.05,0,0,0,1],
                                   [.03,0,-.1,0,0,0,1],[0,0,-1,0,0,0,1]]))
        sim=SimpleNamespace(model=model,state=SimpleNamespace(body_q=array([
            [*origin,0,0,0,1],[0,0,0,0,0,0,1]])),
            scene=SimpleNamespace(embodiment=SimpleNamespace(arms={
                "left":SimpleNamespace(finger_bodies=[[0]])})))
        ik=SimpleNamespace(ee_pose=lambda _: (origin,np.array([0,0,0,1])))
        np.testing.assert_allclose(low_offset(sim,"left",[0,0,0,1],ik),[.03,0,-.105],atol=1e-8)
        rotated=low_offset(sim,"left",Rotation.from_euler("x",90,degrees=True).as_quat(),ik)
        self.assertAlmostEqual(rotated[2],-.01,places=7)
        self.assertAlmostEqual(rotated[0],.02,places=7)


if __name__=="__main__":unittest.main()
