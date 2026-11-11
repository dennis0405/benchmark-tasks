import unittest
from dataclasses import replace
from deformable_scoring import DeformableScorer, Observation


def obs(t, **kw):
    base = Observation(t, True, True, True, .10, True, True, 0, 0, 4)
    return replace(base, **kw)


class DeformableScoringTest(unittest.TestCase):
    def test_hold_duration_and_damage(self):
        s = DeformableScorer()
        for i in range(60):
            self.assertEqual(s.update(obs(i*.05)), .5)
        self.assertEqual(s.update(obs(3.)), 1.)
        self.assertEqual(s.update(obs(3.05, broken_count=1)), .5)
        self.assertEqual(s.update(obs(3.10, broken_count=2)), 0.)
        self.assertEqual(s.update(obs(3.15, broken_count=0)), 0.)

    def test_spill_and_ground_are_irreversible(self):
        s = DeformableScorer()
        self.assertEqual(s.update(obs(0, spilled_count=1)), 0)
        self.assertEqual(s.update(obs(.05, spilled_count=0)), 0)
        s = DeformableScorer()
        self.assertEqual(s.update(obs(0, bag_on_table=False)), 0)
        self.assertEqual(s.update(obs(.05)), 0)

    def test_dropped_or_invalid_observations_reset_hold(self):
        s = DeformableScorer()
        s.update(obs(0))
        self.assertIsNone(s.update(obs(.05, valid=False)))
        self.assertEqual(s.update(obs(3.1)), .5)
        self.assertEqual(s.update(obs(3.2)), .5)
        self.assertEqual(s.update(obs(3.4)), .5)
        with self.assertRaises(ValueError):
            s.update(obs(3.4))

    def test_partial_requires_contact(self):
        s = DeformableScorer()
        self.assertEqual(s.update(obs(0, both_jaws_contact=False)), 0)
        self.assertEqual(s.update(obs(.05, width_covered_m=.02)), .5)
        with self.assertRaises(ValueError):
            s.update(obs(.10, broken_count=None))

if __name__ == '__main__':
    unittest.main()
