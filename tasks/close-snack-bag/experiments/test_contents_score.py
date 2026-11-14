import unittest
import numpy as np
from contents_score import integrity_multiplier


class IntegrityScoreTests(unittest.TestCase):
    def test_all_broken_or_all_spilled_scores_zero(self):
        groups=[np.array([0,1]),np.array([2,3])]
        self.assertEqual(integrity_multiplier([True,True],[False]*4,[1]*4,groups)["multiplier"],0.)
        self.assertEqual(integrity_multiplier([False,False],[True]*4,[1]*4,groups)["multiplier"],0.)

    def test_partial_damage_and_spill_both_reduce_score(self):
        groups=[np.array([0,1]),np.array([2,3])]
        value=integrity_multiplier([True,False],[False,False,True,False],[1]*4,groups)
        self.assertEqual(value["intact_mass_fraction"],.5)
        self.assertEqual(value["retained_mass_fraction"],.75)
        self.assertEqual(value["multiplier"],.375)

    def test_undamaged_retained_contents_do_not_reduce_score(self):
        self.assertEqual(integrity_multiplier([False],[False,False],[.5,.5],[np.array([0,1])])["multiplier"],1.)


if __name__=="__main__":unittest.main()
