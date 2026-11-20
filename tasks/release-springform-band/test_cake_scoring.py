"""Boundary and regression checks for the damage-aware scorer. Run with unittest."""
import unittest
from types import SimpleNamespace

from cake_contact import manifold_peaks, touches_exposed_cake
from scoring import Scorer


def frame(height=3.0, **changes):
    m = dict(band_xyz=[.32, .097, .85], lip_above_plate_mm=[height] * 40,
             lip_radial_margin_mm=[1.0] * 40, lip_above_cake_mm=height,
             motion_mm=0., tilt_deg=0., on_destination_mat=True, speed=0.,
             spin=0., robot_contacts=[], cake_contact_impulse_ns=0.,
             cake_penetration_mm=0.)
    return dict(m, **changes)


class CakeChecks(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer()
        self.scorer.update(mechanics=frame(-1.0))

    def settle(self):
        for _ in range(21):
            result = self.scorer.update(mechanics=frame())
        return result['score']

    def test_contact_filter(self):
        for layer in ('filling', 'crust'):
            cake = '/world/pan_fixture/cake/' + layer
            base = '/world/pan_fixture/base/plate'
            for other in ('/world/left_robot/pad', '/world/pan_fixture/sector1/wall2'):
                self.assertTrue(touches_exposed_cake((cake, other)))
                self.assertTrue(touches_exposed_cake((other, cake)))
            self.assertFalse(touches_exposed_cake((base, cake)))
        self.assertFalse(touches_exposed_cake((base, '/world/left_robot/pad')))

    def test_manifold_offset_normal_and_contact_depth(self):
        def point(x, z, depth):
            return SimpleNamespace(normal=(0, 0, 1), impulse=(x, 0, z), separation=depth)
        points = [point(99., 99., -1.), point(1., -.009, .001), point(0., -.011, -.0015)]
        impulse, depth = manifold_peaks(points, 1, 2)
        self.assertAlmostEqual(impulse, .020)
        self.assertAlmostEqual(depth, 1.5)

    def test_undamaged_within_limits(self):
        self.scorer.update(mechanics=frame(motion_mm=3., tilt_deg=3.,
                                           cake_contact_impulse_ns=.0249,
                                           cake_penetration_mm=1.99))
        self.assertEqual(self.settle(), 1.)

    def test_damage_after_removal(self):
        for cause in (dict(cake_contact_impulse_ns=.025), dict(cake_penetration_mm=2.)):
            with self.subTest(cause=cause):
                self.setUp()
                self.assertEqual(self.scorer.update(mechanics=frame())['score'], .5)
                damaged = self.scorer.update(mechanics=frame(**cause))
                self.assertTrue(damaged['cake_damaged'])
                self.assertEqual(damaged['score'], 0.)
                self.assertEqual(self.settle(), 0.)

    def test_pose_violation_persists(self):
        for cause in (dict(motion_mm=3.001), dict(tilt_deg=3.001)):
            with self.subTest(cause=cause):
                self.setUp()
                self.scorer.update(mechanics=frame(**cause))
                self.assertEqual(self.settle(), 0.)

    def test_old_trace_requires_new_observations(self):
        m = frame()
        del m['cake_contact_impulse_ns']
        with self.assertRaisesRegex(ValueError, 'Missing cake-contact measurements'):
            self.scorer.update(mechanics=m)


if __name__ == '__main__':
    unittest.main()
