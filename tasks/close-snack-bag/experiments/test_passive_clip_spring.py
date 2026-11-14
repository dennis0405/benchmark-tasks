"""Check restoring torque, world axes, damping and equal/opposite wrenches."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from passive_clip_spring import spring_torque


class PassiveSpringTests(unittest.TestCase):
    def test_rotated_restoring_spring_and_damper(self):
        devices=["cpu"]+(["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            for angle,rate in ((0.,0.),(-.3,0.),(0.,5.),(-.3,-2.)):
                with self.subTest(device=device,angle=angle,rate=rate):
                    root=Rotation.from_euler("xyz",[.8,-.4,.6])
                    axis=root.apply([0,1,0])
                    poses=wp.array([np.r_[[.6,0,.9],root.as_quat()],
                        np.r_[[.6,0,.9],(root*Rotation.from_euler("y",angle)).as_quat()]],
                        dtype=wp.transform,device=device)
                    velocity=np.zeros((2,6));velocity[1,3:]=axis*rate
                    rates=wp.array(velocity,dtype=wp.spatial_vector,device=device)
                    forces=wp.zeros(2,dtype=wp.spatial_vector,device=device)
                    observed=wp.zeros(1,dtype=float,device=device)
                    wp.launch(spring_torque,1,inputs=[poses,rates,0,1,.6,.01],
                              outputs=[forces,observed],device=device)
                    force=forces.numpy();expected=-.6*angle-.01*rate
                    np.testing.assert_allclose(force[:,:3],0,atol=1e-7)
                    np.testing.assert_allclose(force.sum(0),0,atol=1e-7)
                    np.testing.assert_allclose(force[1,3:],axis*expected,atol=2e-6)
                    self.assertAlmostEqual(float(observed.numpy()[0]),expected,places=5)
                    if angle==0 and rate:self.assertLess(float(force[1,3:]@(axis*rate)),0.)


if __name__=="__main__":unittest.main()
