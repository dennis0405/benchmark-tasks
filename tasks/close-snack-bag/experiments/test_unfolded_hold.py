import unittest
from assess_unfolded import summarize_hold


class HoldChecks(unittest.TestCase):
    def rows(self,n=62):
        return [dict(step=i,phase="hands_free_hold",fold_ok=True,hands_clear=True,
                     coverage_m=.1,hinge_rad=-.04,bag_on_table=True) for i in range(n)]

    def test_three_seconds_requires_observed_interval(self):
        self.assertLess(summarize_hold(self.rows(60),20),3.)
        self.assertEqual(summarize_hold(self.rows(61),20),3.)

    def test_released_empty_clip_does_not_pass(self):
        rows=self.rows()
        for row in rows:row["coverage_m"]=0.
        self.assertEqual(summarize_hold(rows,20),0.)

    def test_missing_frame_breaks_continuity(self):
        rows=self.rows(100);del rows[45]
        self.assertLess(summarize_hold(rows,20),3.)


if __name__=="__main__":unittest.main()
