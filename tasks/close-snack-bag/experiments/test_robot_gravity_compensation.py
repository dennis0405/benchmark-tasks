"""Compensation respects authored coefficients, world gravity, and COM torque."""
import unittest
import numpy as np
import warp as wp
from robot_gravity_compensation import compensate


class GravityCompensationTests(unittest.TestCase):
    def test_force_only_and_uncompensated_tool(self):
        for device in ["cpu"]+(["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                mass=wp.array([2.,3.,.024],dtype=float,device=device)
                coefficients=wp.array([1.,.5,0.],dtype=float,device=device)
                world=wp.array([0,1,0],dtype=int,device=device)
                gravity=wp.array([[0,0,-9.81],[0,-10,0]],dtype=wp.vec3,device=device)
                force=wp.zeros(3,dtype=wp.spatial_vector,device=device)
                wp.launch(compensate,3,inputs=[mass,coefficients,world,gravity],outputs=[force],device=device)
                np.testing.assert_allclose(force.numpy(),[[0,0,19.62,0,0,0],[0,15,0,0,0,0],[0,0,0,0,0,0]],atol=2e-6)


if __name__=="__main__":unittest.main()
