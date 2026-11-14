"""Capacity events remain visible after later quiet substeps."""
import unittest
import numpy as np
import warp as wp
from contact_capacity import observe


class CapacityTests(unittest.TestCase):
    def test_persistent_overflow_history(self):
        for device in ['cpu']+(['cuda:0'] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                r=wp.array([10,65],dtype=int,device=device)
                s=wp.array([100,101],dtype=int,device=device)
                rt=wp.array([11],dtype=int,device=device);st=wp.array([201],dtype=int,device=device)
                stats=wp.zeros(7,dtype=int,device=device)
                inputs=[2,r,s,64,100,rt,st,10,200]
                wp.launch(observe,2,inputs=inputs,outputs=[stats],device=device)
                np.testing.assert_array_equal(stats.numpy(),[1,65,101,1,1,1,1])
                r.assign([10,64]);s.assign([100,100]);rt.assign([10]);st.assign([200])
                wp.launch(observe,2,inputs=inputs,outputs=[stats],device=device)
                np.testing.assert_array_equal(stats.numpy(),[2,65,101,1,1,1,1])


if __name__=='__main__':unittest.main()
