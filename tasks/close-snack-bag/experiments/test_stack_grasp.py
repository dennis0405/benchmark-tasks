"""Geometry checks that do not require a GPU or simulator."""
import unittest
import numpy as np
from stack_grasp import clipped_surfaces


class TestTriangleClipping(unittest.TestCase):
    def test_surface_between_vertices_counts(self):
        # No vertex lies inside this rectangle, but half the triangle does.
        triangles=np.array([[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]])
        area,vertices=clipped_surfaces(triangles,[(0,.5,1.5),(1,0.,2.)])
        self.assertAlmostEqual(area,1.)
        self.assertTrue(all(.5<=v[0]<=1.5 for v in vertices))

    def test_projection_and_real_surface_are_distinct(self):
        triangles=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,1.]]])
        area,_=clipped_surfaces(triangles,[])
        projected,_=clipped_surfaces(triangles,[],True)
        self.assertAlmostEqual(area,np.sqrt(2)/2)
        self.assertAlmostEqual(projected,.5)

    def test_disjoint_box_cannot_capture_a_ply(self):
        triangles=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        area,vertices=clipped_surfaces(triangles,[(0,2.,3.)])
        self.assertEqual(area,0.)
        self.assertEqual(vertices,[])


if __name__=="__main__":unittest.main()
