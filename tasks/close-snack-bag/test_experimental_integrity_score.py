import unittest
from experimental_integrity_score import apply_integrity_penalty


class IntegrityScoreTest(unittest.TestCase):
    def test_clean_partial_and_full(self):
        self.assertEqual(apply_integrity_penalty(1, 4, 0, 0), 1)
        self.assertEqual(apply_integrity_penalty(.5, 4, 0, 0), .5)
    def test_damage_and_spill_override(self):
        self.assertEqual(apply_integrity_penalty(1, 4, 0, 1), .5)
        self.assertEqual(apply_integrity_penalty(1, 4, 0, 2), 0)
        self.assertEqual(apply_integrity_penalty(1, 4, 1, 0), 0)
        self.assertEqual(apply_integrity_penalty(0, 4, 0, 1), 0)
    def test_missing_observations_rejected(self):
        for count in (None, -1, 5, True):
            with self.assertRaises(ValueError):
                apply_integrity_penalty(1, 4, count, 0)

if __name__ == '__main__':
    unittest.main()
