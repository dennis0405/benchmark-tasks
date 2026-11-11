import unittest
import numpy as np
from chip_damage import reference_inverse, max_strain


class ChipStrainTest(unittest.TestCase):
    def test_strain_invariant_to_translation_and_rotation(self):
        p = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        t = np.array([[0,1,2,3]])
        inverse = reference_inverse(p, t)
        rot = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.assertAlmostEqual(max_strain(p @ rot.T + [2,3,4], t, inverse), 0)
        self.assertAlmostEqual(max_strain(p * [1,1,.8], t, inverse), .2)

    def test_reject_degenerate(self):
        with self.assertRaises(ValueError):
            reference_inverse(np.zeros((4,3)), np.array([[0,1,2,3]]))

if __name__ == '__main__':
    unittest.main()
