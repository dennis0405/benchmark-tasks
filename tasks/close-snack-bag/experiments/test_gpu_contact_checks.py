"""Cross-check the independent GPU checks against CPU geometry references."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from check_native_clip import penetrating_triangles
from gpu_contact_checks import boxes, GeometryChecks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pouch_geometry import open_pouch
from bag_surface_mesh import folded_positions


class GPUChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = 'cuda:0'

    def box_flags(self, points, faces, body, local, half):
        n = len(faces)
        counts = wp.zeros(1, dtype=int, device=self.device)
        flags = wp.zeros(n, dtype=int, device=self.device)
        wp.launch(boxes, dim=n, inputs=[
            wp.array(points, dtype=wp.vec3, device=self.device),
            wp.array(faces, dtype=wp.vec3i, device=self.device),
            wp.array([body], dtype=wp.transform, device=self.device),
            wp.array([0], dtype=int, device=self.device),
            wp.array([local], dtype=wp.transform, device=self.device),
            wp.array([half], dtype=wp.vec3, device=self.device),
            wp.array([0], dtype=int, device=self.device), n, 0, 1.e-7, counts, flags], device=self.device)
        return flags.numpy()

    def test_random_transformed_triangles_match_cpu_sat(self):
        rng = np.random.default_rng(31)
        points = rng.uniform(-.1, .1, (1500, 3)).astype(np.float32)
        faces = np.arange(1500, dtype=np.int32).reshape(-1, 3)
        body = np.r_[[.01, .02, .03], Rotation.from_euler('xyz', [.2, -.4, .7]).as_quat()].astype(np.float32)
        local = np.array([.022, 0, -.004, 0, 0, 0, 1], dtype=np.float32)
        half = np.array([.022, .05, .003], dtype=np.float32)
        rot = Rotation.from_quat(body[3:]).as_matrix()
        relative = (points-(body[:3]+rot@local[:3]))@rot
        expected = np.zeros(len(faces), dtype=int)
        expected[penetrating_triangles(relative, faces, half)] = 1
        np.testing.assert_array_equal(expected, self.box_flags(points, faces, body, local, half))

    def test_crossing_without_inside_vertices_and_boundary_touch(self):
        points = np.array([[-2, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        identity = [0, 0, 0, 0, 0, 0, 1]
        self.assertEqual(int(self.box_flags(points, faces, identity, identity, [.5]*3)[0]), 1)
        points[:, 2] = .5
        self.assertEqual(int(self.box_flags(points, faces, identity, identity, [.5]*3)[0]), 0)

    def checker(self, points, faces, steps):
        state = SimpleNamespace(particle_q=wp.array(points, dtype=wp.vec3, device=self.device))
        model = SimpleNamespace(device=wp.get_device(self.device),
            shape_body=wp.empty(0, dtype=int, device=self.device))
        return state, GeometryChecks(model, [state], faces, steps)

    def test_bvh_detects_crossing_and_rejects_separated_triangles(self):
        points = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0],
                           [0, 0, -1], [0, 0, 1], [.2, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        state, checks = self.checker(points, faces, 2)
        checks.check(state, 0)
        self.assertGreater(checks.self_counts.numpy()[0], 0)
        points[3:, 0] += 4
        state.particle_q.assign(points)
        checks.check(state, 1)
        self.assertEqual(checks.self_counts.numpy()[1], 0)

    def test_real_folded_pouch_initialization_has_no_strict_crossing(self):
        q, faces, _ = open_pouch(nx=80, ny=64, mouth_half_gap=.0003, foldable_mouth=True)
        initial = folded_positions(q+[0, 0, .019], radius=.0018)-[0, 0, .019]+[0, 0, .025]
        state, checks = self.checker(initial, faces, 1)
        checks.check(state, 0)
        self.assertEqual(checks.self_counts.numpy()[0], 0)


if __name__ == '__main__':
    unittest.main()
