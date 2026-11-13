"""Static fixtures and body-local boxes must share the same crossing rule."""
import unittest
import numpy as np
import warp as wp
from substep_geometry import scene_boxes


class SceneBoxTests(unittest.TestCase):
    def test_static_and_moving_owner_transforms(self):
        triangle=np.array([[-.2,-.05,0],[.2,-.05,0],[0,.2,0]],np.float32)
        points=np.concatenate([triangle+[x,0,0] for x in (0,2,4)])
        device="cpu"
        faces=wp.array(np.arange(9).reshape(3,3),dtype=wp.vec3i,device=device)
        owner=wp.array([-1,0],dtype=int,device=device)
        local=wp.array([[0,0,0,0,0,0,1]]*2,dtype=wp.transform,device=device)
        body=wp.array([[2,0,0,0,0,0,1]],dtype=wp.transform,device=device)
        scale=wp.array([[.1,.1,.1]]*2,dtype=wp.vec3,device=device)
        ids=wp.array([0,1],dtype=int,device=device)
        count=wp.zeros(2,dtype=int,device=device)
        for shift,expected in ((0,2),(10,0)):
            with self.subTest(shift=shift):
                count.zero_()
                q=wp.array(points+[shift,0,0],dtype=wp.vec3,device=device)
                wp.launch(scene_boxes,6,inputs=[q,faces,body,owner,local,scale,ids,3,count],device=device)
                self.assertEqual(int(count.numpy().sum()),expected)
                if expected:
                    np.testing.assert_array_equal(count.numpy(),[1,1])


if __name__=="__main__":
    unittest.main()
