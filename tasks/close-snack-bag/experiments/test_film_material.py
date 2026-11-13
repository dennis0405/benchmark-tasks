import unittest
from dataclasses import replace
from film_material import FilmMaterial


class MaterialScalingTest(unittest.TestCase):
    def test_thickness_changes_bending_cubically(self):
        base = FilmMaterial()
        a, b = base.coefficients(.0025), replace(base, thickness=2*base.thickness).coefficients(.0025)
        self.assertAlmostEqual(b['edge_ke']/a['edge_ke'], 8)
        self.assertAlmostEqual(b['tri_ke']/a['tri_ke'], 2)
        self.assertAlmostEqual(b['areal_density']/a['areal_density'], 2)

    def test_mesh_refinement_preserves_yield_curvature(self):
        a, b = FilmMaterial().coefficients(.005), FilmMaterial().coefficients(.0025)
        self.assertAlmostEqual(b['yield_angle']/a['yield_angle'], .5)
        self.assertAlmostEqual(b['edge_ke']/a['edge_ke'], 2)

    def test_invalid_physical_inputs_fail(self):
        for material in (FilmMaterial(thickness=0), FilmMaterial(poisson=.5), FilmMaterial(young=float('nan'))):
            with self.assertRaises(ValueError):
                material.coefficients(.0025)


if __name__ == '__main__':
    unittest.main()
