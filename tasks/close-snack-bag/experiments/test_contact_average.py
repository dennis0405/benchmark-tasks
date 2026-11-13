import unittest
import numpy as np
import warp as wp
from contact_average import accumulate,elapsed


class ContactAverageTests(unittest.TestCase):
    def test_impulse_weighting_and_rigid_record_offset(self):
        for device in ("cpu",*(["cuda:0"] if wp.is_cuda_available() else [])):
            with self.subTest(device=device):
                impulse=wp.zeros(3,dtype=wp.vec3,device=device)
                duration=wp.zeros(1,dtype=float,device=device)
                count=wp.array([3],dtype=int,device=device)
                shape=wp.array([0,0,1,-1],dtype=int,device=device)
                vectors=np.array([[99,99,99,0,0,0],[99,99,99,0,0,0],
                    [1,2,3,4,5,6],[2,0,0,4,5,6],[0,4,0,4,5,6],[999]*6],np.float32)
                for dt,scale in ((.02,1.),(.03,2.)):
                    forces=wp.array(vectors*scale,dtype=wp.spatial_vector,device=device)
                    wp.launch(accumulate,4,inputs=[count,shape,2,forces,dt,impulse],device=device)
                    wp.launch(elapsed,1,inputs=[dt,duration],device=device)
                np.testing.assert_allclose(impulse.numpy()/duration.numpy()[0],
                    [[4.8,3.2,4.8],[0,6.4,0],[0,0,0]],rtol=1e-6)


if __name__=="__main__":unittest.main()
