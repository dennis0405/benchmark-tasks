import unittest
from assess_unfolded import summarize_hold,contact_capacity_valid


class HoldChecks(unittest.TestCase):
    def rows(self,n=62):
        return [dict(step=i,phase="hands_free_hold",fold_ok=True,hands_clear=True,
                     coverage_m=.1,hinge_rad=-.04,bag_on_table=True,jaw_load_ok=True) for i in range(n)]

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

    def test_unloaded_jaw_breaks_continuity(self):
        rows=self.rows(100);rows[45]["jaw_load_ok"]=False
        self.assertLess(summarize_hold(rows,20),3.)

    def test_missing_or_partial_contact_observation_cannot_pass(self):
        self.assertFalse(contact_capacity_valid(None,100))
        self.assertFalse(contact_capacity_valid(dict(checked_substeps=100),100))

    def test_transient_contact_overflow_cannot_pass(self):
        report=dict(checked_substeps=100,rigid_body_overflow_events=0,soft_body_overflow_events=0,
                    rigid_total_overflow_substeps=0,soft_total_overflow_substeps=0)
        self.assertTrue(contact_capacity_valid(report,100))
        self.assertFalse(contact_capacity_valid(report,101))
        for field in list(report)[1:]:
            with self.subTest(field=field):
                self.assertFalse(contact_capacity_valid(dict(report,**{field:1}),100))


if __name__=="__main__":unittest.main()
