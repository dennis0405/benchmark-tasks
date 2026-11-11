import unittest

from support_clearance import initial_overlap


class SupportClearanceTests(unittest.TestCase):
    def test_wide_support_intersects_bag_but_corner_support_does_not(self):
        self.assertTrue(initial_overlap(80, 64, .0018, (.12, 0, .004), (.025, .13, .008)))
        self.assertFalse(initial_overlap(80, 64, .0018, (.145, .068, .004), (.03, .024, .008)))
        self.assertFalse(initial_overlap(80, 64, .0018, (.145, .068, .014), (.03, .024, .008)))
        self.assertTrue(initial_overlap(80, 64, .0018, (.145, .068, .016), (.03, .024, .008)))
