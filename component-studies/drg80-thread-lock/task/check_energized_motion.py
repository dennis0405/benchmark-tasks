"""Boundary, segmentation, and full-grader latching regressions."""
from dataclasses import replace
from math import inf, nan, nextafter
from stop_lock_score import EnergizedMotion, Sample, StopLockScore
import unittest


class PoweredMotionTest(unittest.TestCase):
    def test_equal_and_next_float_on_each_axis(self):
        for axis, limit in enumerate(EnergizedMotion.limits):
            for sign in (-1, 1):
                g = EnergizedMotion()
                self.assertFalse(g.observe(1, 2, 0, 0, 0))
                pose = [0., 0., 0.]
                pose[axis] = sign * limit
                self.assertFalse(g.observe(2, 2, *pose))
                pose[axis] = sign * nextafter(limit, inf)
                self.assertTrue(g.observe(3, 2, *pose))
                self.assertEqual(g.first_failure, (3, g.channels[axis]))
                self.assertTrue(g.observe(4, 0, 0, 0, 0))

    def test_reversing_path(self):
        g = EnergizedMotion()
        for tick, z in enumerate((0., .6e-6, -.6e-6, 0.), 1):
            g.observe(tick, -2, z, 0., 0.)
        self.assertEqual(g.first_failure, (3, g.channels[0]))

    def test_slow_creep(self):
        g = EnergizedMotion()
        for i in range(400):
            g.observe(i + 1, 2., i * 4e-9, 0., 0.)
        self.assertIsNotNone(g.first_failure)

    def test_vent_and_signed_force(self):
        g = EnergizedMotion()
        for i, (force, z) in enumerate(((0, 0), (-2, .9e-6), (-1, 1.), (2, 1.+.9e-6)), 1):
            self.assertFalse(g.observe(i, force, z, 0, 0))
        self.assertEqual(g.windows, 2)
        self.assertTrue(g.observe(5, 2, 1.+1.2e-6, 0, 0))

    def test_reset_and_invalid_input(self):
        for axis in range(4):
            pose = [2., 0., 0., 0.]
            pose[axis] = nan
            self.assertTrue(EnergizedMotion().observe(1, *pose))
        g = EnergizedMotion()
        self.assertFalse(g.observe(1, 2, 1000, 700, 900))
        self.assertFalse(g.observe(2, 2, 1000, 700, 900))

    def test_task_score_latches_after_gradual_change(self):
        sample = Sample(90, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 30, .04,
                        True, True, True, True, True, False)
        for hz in (240, 480):
            g = StopLockScore(hz)
            for _ in range(hz//4):
                g.observe(sample, g.tick+1)
            for _ in range(hz//4):
                g.observe(replace(sample, opening_force=2), g.tick+1)
            self.assertEqual(g.score, .5)
            for i in range(200):
                g.observe(replace(sample, opening_force=2, screw_height=i*1e-8), g.tick+1)
            self.assertTrue(g.damaged)
            self.assertEqual(g.score, 0.)
            self.assertEqual(g.observe(sample, g.tick+1), 0.)


if __name__ == "__main__":
    unittest.main()
