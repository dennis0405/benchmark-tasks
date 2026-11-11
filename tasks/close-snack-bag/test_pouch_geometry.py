"""Topology and dimension checks for the experimental open pouch."""
import unittest
from collections import Counter
import numpy as np
from pouch_geometry import open_pouch


class OpenPouchGeometryTest(unittest.TestCase):
    def test_sealed_edges_and_only_open_mouth(self):
        points, faces, ix = open_pouch(nx=40, ny=32)
        self.assertTrue(np.array_equal(ix[0, 0], ix[1, 0]))
        self.assertTrue(np.array_equal(ix[0, :, 0], ix[1, :, 0]))
        self.assertTrue(np.array_equal(ix[0, :, -1], ix[1, :, -1]))
        self.assertFalse(np.any(ix[0, -1, 1:-1] == ix[1, -1, 1:-1]))
        edges = Counter(tuple(sorted(e)) for a, b, c in faces for e in ((a,b), (b,c), (c,a)))
        self.assertTrue(all(count in (1, 2) for count in edges.values()))
        boundary = [edge for edge, count in edges.items() if count == 1]
        self.assertEqual(len(boundary), 64)
        self.assertTrue(all(np.allclose(points[list(edge), 0], 0.20) for edge in boundary))
        area2 = np.linalg.norm(np.cross(points[faces[:, 1]]-points[faces[:, 0]],
                                         points[faces[:, 2]]-points[faces[:, 0]]), axis=1)
        self.assertGreater(area2.min(), 1e-10)

    def test_foldable_mouth_is_narrow_after_taper(self):
        points, _, ix = open_pouch(nx=40, ny=32, mouth_half_gap=0.0003, foldable_mouth=True)
        for column in range(28, 41):
            gap = points[ix[1, column, 16], 2] - points[ix[0, column, 16], 2]
            self.assertAlmostEqual(float(gap), 0.0006, places=6)

    def test_body_bulge_and_mouth(self):
        points, _, ix = open_pouch(nx=40, ny=32)
        self.assertAlmostEqual(float(np.ptp(points[:, 0])), 0.20, places=6)
        self.assertAlmostEqual(float(np.ptp(points[:, 1])), 0.16, places=6)
        self.assertGreater(points[ix[1, 18, 16], 2] - points[ix[0, 18, 16], 2], 0.04)
        self.assertAlmostEqual(float(points[ix[1, -1, 16], 2] - points[ix[0, -1, 16], 2]), 0.004, places=6)


if __name__ == '__main__':
    unittest.main()
