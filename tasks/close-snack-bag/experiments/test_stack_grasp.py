"""Geometry checks that do not require a GPU or simulator."""
import unittest
import numpy as np
from stack_grasp import clipped_surfaces,select_press_patch
from pouch_geometry import open_pouch


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

    def folded_fixture(self):
        q,f,g=open_pouch(nx=40,ny=32)
        x=q[:,0].copy();q[:,0]=np.where(x>.17,.63+x-.17,.8-x)
        for layer in range(2):
            for row in range(41):
                q[g[layer,row],2]=.0003+layer*.0005+(.001 if row>34 else 0.)
        return q,f,g

    def test_press_patch_contains_four_plies_on_the_support(self):
        q,f,g=self.folded_fixture()
        r=select_press_patch(q,g,f,0.,.620)
        self.assertGreater(min(r["ply_area_mm2"]),500.)
        self.assertTrue(.6405<r["x"]<.6445)
        self.assertGreaterEqual(r["x"]-.006,.620)

    def test_press_plan_is_translation_invariant(self):
        q,f,g=self.folded_fixture()
        a=select_press_patch(q,g,f,0.,.620)
        b=select_press_patch(q+[.08,.02,.1],g,f,.02,.700)
        self.assertAlmostEqual(b["x"]-a["x"],.08,places=6)
        np.testing.assert_allclose(a["ply_area_mm2"],b["ply_area_mm2"],rtol=1e-5)


if __name__=="__main__":unittest.main()
