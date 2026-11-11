"""Topology checks for the experimental packet's surface mesh."""
import collections
import unittest

import numpy as np
from bag_surface_mesh import folded_positions, packet_mesh


class SurfaceMeshTest(unittest.TestCase):
    def test_open_mouth_and_watertight_seams(self):
        for flat in (False, True):
            vertices, faces = packet_mesh(flat_mouth=flat)
            edges = collections.defaultdict(list)
            for triangle in faces:
                for i in range(3):
                    a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
                    edges[tuple(sorted((a, b)))].append((a, b))
            boundary = [edge for edge, directions in edges.items() if len(directions) == 1]
            self.assertEqual(len(boundary), 34)
            self.assertTrue(all(np.allclose(vertices[list(edge), 0], 0.2) for edge in boundary))
            self.assertTrue(all(len(directions) <= 2 for directions in edges.values()))
            self.assertTrue(all(len(directions) < 2 or directions[0] != directions[1]
                                for directions in edges.values()))
            self.assertTrue(np.isfinite(vertices).all())

    def test_fold_changes_only_mouth_initial_positions(self):
        rest, _ = packet_mesh(nx=80, flat_mouth=True)
        folded = folded_positions(rest, radius=0.0022)
        self.assertTrue(np.isfinite(folded).all())
        self.assertTrue(np.allclose(folded[rest[:, 0] <= 0.16], rest[rest[:, 0] <= 0.16]))
        self.assertLess(np.ptp(folded[rest[:, 0] >= 0.16, 0]), 0.025)
        self.assertGreater(np.ptp(folded[rest[:, 0] >= 0.16, 2]), 0.008)


if __name__ == "__main__":
    unittest.main()
