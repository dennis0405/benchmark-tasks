import unittest
import numpy as np
from inspect_pouch_contents import ray_diagnostic


class RayDiagnosticTest(unittest.TestCase):
    def test_inside_and_crossings(self):
        vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                             [0,0,1], [1,0,1], [1,1,1], [0,1,1]], dtype=float)
        faces = np.array([[0,1,2], [0,2,3], [4,5,6], [4,6,7]])
        points = np.array([[0.5,0.5,0.4], [0.3,0.6,1.1], [2,2,0.5]])
        result = ray_diagnostic(vertices, faces, points)
        self.assertEqual([x['status'] for x in result], ['inside', 'outside', 'ambiguous'])
        self.assertAlmostEqual(result[0]['clearance_m'], 0.4)
        self.assertAlmostEqual(result[1]['clearance_m'], -0.1)


if __name__ == '__main__':
    unittest.main()
