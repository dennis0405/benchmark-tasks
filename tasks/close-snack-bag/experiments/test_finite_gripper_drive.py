"""The experimental transmission applies bounded forces, never poses."""
import unittest
import numpy as np
import warp as wp
from finite_gripper_drive import drive,geared_drive


class FiniteDriveTests(unittest.TestCase):
    def test_opposed_targets_damping_and_force_limits(self):
        for device in ['cpu']+(['cuda:0'] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                q=wp.array([0.,0.,.02,-.02],dtype=float,device=device)
                qd=wp.array([0.,0.,.1,-.1],dtype=float,device=device)
                target=wp.array([.0475,0.,.02,0.],dtype=float,device=device)
                index=wp.array([[0,0,0],[1,1,0],[2,2,2],[3,3,2]],dtype=wp.vec3i,device=device)
                coeff=wp.array([[0,1],[0,-1],[0,1],[0,-1]],dtype=wp.vec2,device=device)
                force=wp.zeros(4,dtype=float,device=device)
                wp.launch(drive,4,inputs=[q,qd,target,index,coeff,10000.,10.,5.],outputs=[force],device=device)
                np.testing.assert_allclose(force.numpy(),[5,-5,-1,1],atol=1e-5)
                np.testing.assert_allclose(q.numpy(),[0,0,.02,-.02],atol=1e-7)


    def test_centering_transmission_is_passive_and_actuator_limited(self):
        for device in ['cpu']+(['cuda:0'] if wp.is_cuda_available() else []):
            for a,b in ((.02,-.02),(.021,-.02)):
                with self.subTest(device=device,a=a):
                    q=wp.array([a,b],dtype=float,device=device)
                    qd=wp.array([.1,.1],dtype=float,device=device)
                    target=wp.array([.0475],dtype=float,device=device)
                    pairs=wp.array([[0,0,1,1]],dtype=wp.vec4i,device=device)
                    targets=wp.array([0],dtype=int,device=device)
                    force=wp.zeros(2,dtype=float,device=device)
                    observed=wp.zeros(1,dtype=wp.vec2,device=device)
                    wp.launch(geared_drive,1,inputs=[q,qd,target,pairs,targets],outputs=[force,observed],device=device)
                    actuator,reaction=observed.numpy()[0]
                    self.assertLessEqual(abs(actuator),5.)
                    self.assertAlmostEqual(reaction,-25000*(a+b)-10*.2,places=3)
                    np.testing.assert_allclose(force.numpy(),[actuator+reaction,-actuator+reaction],atol=1e-6)
                    # After subtracting the spring energy rate, remaining power is dissipative.
                    self.assertLessEqual(reaction*.2+25000*(a+b)*.2,1e-4)


if __name__=='__main__':unittest.main()
