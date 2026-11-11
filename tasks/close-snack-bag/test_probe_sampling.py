import unittest

from probe_sampling import sample_steps


class SamplingTests(unittest.TestCase):
    def test_20_hz_monitor_has_no_missing_regular_samples(self):
        steps = sample_steps(240 * 4, every=12)
        self.assertTrue(set(range(11, 240 * 4, 12)).issubset(steps))
        self.assertIn(0, steps)
        self.assertIn(959, steps)

    def test_invalid_input_is_rejected(self):
        for total, every in ((0, 12), (10, -1)):
            with self.assertRaises(ValueError):
                sample_steps(total, every)
