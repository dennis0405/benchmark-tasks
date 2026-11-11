import unittest
import numpy as np
from check_clip_intersections import inspect


class ClipIntersectionTest(unittest.TestCase):
    def test_triangle_crossing_detected_without_vertex_inside(self):
        # Lower jaw box spans x=[0,.044], y=[-.05,.05], z=[-.007,-.001].
        faces=np.array([[0,1,2]],dtype=np.int32)
        points=np.array([[-.01,0,-.004],[.06,0,-.004],[.02,.08,-.004]])
        pose=[[0,0,0,0,0,0,1],[0,0,0,0,0,0,1]]
        result=inspect(points,faces,pose)
        self.assertTrue(result['lower_jaw']['surface_intersection_or_touch'])
        self.assertEqual(result['lower_jaw']['vertices_inside'],0)
        far=inspect(points+[0,0,1],faces,pose)
        self.assertFalse(far['lower_jaw']['surface_intersection_or_touch'])


if __name__ == '__main__':
    unittest.main()
