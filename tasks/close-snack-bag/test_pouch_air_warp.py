"""GPU kernels checked against the NumPy reference; no simulator coupling test."""
import unittest
import numpy as np
import warp as wp
from types import SimpleNamespace
from pouch_geometry import open_pouch
from pouch_air import enclosed_volume, film_pressure_forces, mouth_area, mouth_cap, mouth_channel_area
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
        volume, area, pressure = gas.stats.numpy()[:3]
        self.assertAlmostEqual(float(volume), enclosed_volume(compressed, f, cap), delta=1e-8)
        self.assertAlmostEqual(float(area), mouth_area(compressed, cap), delta=1e-8)
        self.assertGreater(pressure, 10000)
        expected = film_pressure_forces(compressed, np.vstack((f, cap)), float(pressure))
        np.testing.assert_allclose(force.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(force.numpy().astype(np.float64).sum(axis=0), 0, atol=1e-4)

    def test_channel_area_matches_cpu_reference(self):
        wp.init()
        p, f, grid = open_pouch(nx=20, ny=16)
        model = SimpleNamespace(device='cuda:0')
        gas = LumpedGas(model, p, f, grid, 0, 1/960, 1, channel=True, contact_gap=.0002)
        q = wp.array(p, dtype=wp.vec3, device='cuda:0')
        state = SimpleNamespace(particle_q=q, particle_f=wp.zeros(len(p), dtype=wp.vec3, device='cuda:0'))
        gas.apply(state); wp.synchronize()
        self.assertAlmostEqual(float(gas.stats.numpy()[5]), mouth_channel_area(p, grid, 14, .0002), delta=1e-8)
        closed = p.copy()
        for j in range(grid.shape[2]):
            closed[grid[1,-1,j]] = closed[grid[0,-1,j]]
        wp.copy(q,wp.array(closed,dtype=wp.vec3,device='cuda:0'))
        gas.apply(state); wp.synchronize()
        self.assertAlmostEqual(float(gas.stats.numpy()[5]), 0., delta=1e-9)

    def test_excludes_contents_and_reports_inversion(self):
        wp.init()
        p, f, grid = open_pouch(nx=20, ny=16)
        t = np.array([[.06,0,0],[.07,0,0],[.06,.01,0],[.06,0,.01]], dtype=np.float32)
        all_points = np.vstack((p,t))
        tet = np.array([[len(p),len(p)+1,len(p)+2,len(p)+3]],dtype=np.int32)
        model = SimpleNamespace(device='cuda:0', tet_count=1,
            particle_q=wp.array(all_points,dtype=wp.vec3,device='cuda:0'),
            tet_indices=wp.array(tet,dtype=wp.vec4i,device='cuda:0'))
        gas = LumpedGas(model,p,f,grid,0,1/960,0)
        force = wp.zeros(len(all_points),dtype=wp.vec3,device='cuda:0')
        state = SimpleNamespace(particle_q=model.particle_q,particle_f=force)
        gas.apply(state); wp.synchronize()
        stats=gas.stats.numpy()
        self.assertAlmostEqual(float(stats[3]),1e-6/6,delta=1e-11)
        self.assertEqual(stats[4],0)
        inverted=all_points.copy(); inverted[[len(p)+1,len(p)+2]]=inverted[[len(p)+2,len(p)+1]]
        wp.copy(model.particle_q,wp.array(inverted,dtype=wp.vec3,device='cuda:0'))
        gas.apply(state); wp.synchronize()
        stats=gas.stats.numpy()
        self.assertEqual(stats[4],1)
        self.assertEqual(stats[2],-1e9)


if __name__ == '__main__':
    unittest.main()
