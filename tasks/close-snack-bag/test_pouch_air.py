import unittest
import numpy as np
from pouch_geometry import open_pouch
from pouch_air import R, enclosed_volume, film_pressure_forces, flow_step, mouth_area, mouth_cap


class AirTest(unittest.TestCase):
    def test_volume_translation_invariance_and_pressure(self):
        p, f, grid = open_pouch(nx=40, ny=32)
        cap = mouth_cap(grid)
        volume = enclosed_volume(p, f, cap)
        self.assertGreater(volume, 0)
        self.assertAlmostEqual(volume, enclosed_volume(p + [.3, -.2, .1], f, cap), places=8)
        self.assertGreater(mouth_area(p, cap), 0)
        np.testing.assert_allclose(film_pressure_forces(p, f, 0), 0)
        n = 101325 * volume / (R * 293.15)
        self.assertAlmostEqual(flow_step(n, volume, 0, .1), n)
        compressed = volume / 2
        self.assertGreater(n * R * 293.15 / compressed, 101325)
        after = flow_step(n, compressed, 1e-4, .1)
        self.assertLess(after, n)
        self.assertGreaterEqual(after, n / 2)

    def test_force_direction_on_top_face(self):
        p = np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        force = film_pressure_forces(p, np.array([[0,1,2]]), 6.)
        np.testing.assert_allclose(force.sum(axis=0), [0,0,3])

    def test_flow_from_both_directions(self):
        v = .001; n = 101325 * v / (R * 293.15)
        self.assertGreater(flow_step(n/2, v, .001, .001), n/2)
        self.assertLessEqual(flow_step(n/2, v, .001, 100), n)


if __name__ == '__main__':
    unittest.main()
