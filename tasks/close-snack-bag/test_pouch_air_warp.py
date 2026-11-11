"""GPU kernels checked against the NumPy reference; no simulator coupling test."""
import unittest
import numpy as np
import warp as wp
from types import SimpleNamespace
from pouch_geometry import open_pouch
from pouch_air import enclosed_volume, film_pressure_forces, mouth_area, mouth_cap
from pouch_air_warp import LumpedGas


class GasGpuTest(unittest.TestCase):
    def test_sealed_compression_force_and_volume(self):
        wp.init()
        p, f, grid = open_pouch(nx=40, ny=32)
        cap = mouth_cap(grid)
        gas = LumpedGas(SimpleNamespace(device='cuda:0'), p, f, grid, 0, 1/960, 0)
        q = wp.array(p, dtype=wp.vec3, device='cuda:0')
        force = wp.zeros(len(p), dtype=wp.vec3, device='cuda:0')
        state = SimpleNamespace(particle_q=q, particle_f=force)
        compressed = p.copy(); compressed[:, 2] *= .9
        wp.copy(q, wp.array(compressed, dtype=wp.vec3, device='cuda:0'))
        gas.apply(state); wp.synchronize()
        volume, area, pressure = gas.stats.numpy()
        self.assertAlmostEqual(float(volume), enclosed_volume(compressed, f, cap), delta=1e-8)
        self.assertAlmostEqual(float(area), mouth_area(compressed, cap), delta=1e-8)
        self.assertGreater(pressure, 10000)
        expected = film_pressure_forces(compressed, np.vstack((f, cap)), float(pressure))
        np.testing.assert_allclose(force.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(force.numpy().astype(np.float64).sum(axis=0), 0, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
