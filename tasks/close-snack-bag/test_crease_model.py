import unittest
import numpy as np
from crease_model import hinge_angles, yielded_rest_angles


class CreaseTest(unittest.TestCase):
    def test_yield_is_irreversible_and_restricted(self):
        rest = np.zeros(3)
        updated = yielded_rest_angles(rest, np.array([.2, 1., -1.]), np.array([1, 1, 0], bool), .3)
        np.testing.assert_allclose(updated, [0, .7, 0], atol=1e-12)
        no_more = yielded_rest_angles(updated, np.array([0, .7, 0]), np.ones(3, bool), .3)
        np.testing.assert_allclose(no_more, updated, atol=1e-12)

    def test_angle_and_degenerate(self):
        p = np.array([[0,1,0], [0,-1,0], [0,0,0], [1,0,0]], float)
        a, valid = hinge_angles(p, np.array([[0,1,2,3], [-1,1,2,3]]))
        self.assertTrue(valid[0]); self.assertFalse(valid[1]); self.assertAlmostEqual(a[0], 0)
        p[1] = [0, 0, 1]
        a, _ = hinge_angles(p, np.array([[0,1,2,3]]))
        self.assertAlmostEqual(abs(a[0]), np.pi/2)

    def test_branch_crossing_preserves_physical_angle(self):
        old = np.array([-3.])
        updated = yielded_rest_angles(old, np.array([3.]), np.array([True]), .3)
        self.assertAlmostEqual(updated[0], -3. + 2*np.pi)
        recovered = yielded_rest_angles(updated, np.array([-3.]), np.array([True]), .3)
        self.assertAlmostEqual(recovered[0], -3.)


if __name__ == '__main__':
    unittest.main()
