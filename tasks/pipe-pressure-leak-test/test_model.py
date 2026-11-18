"""Causal/shortcut tests, not robot or real-apparatus validation."""
import math
import unittest
from hydraulics import Hydraulics, Parameters
from scoring import Scorer


class HydraulicTests(unittest.TestCase):
    def test_idle_does_not_pump(self):
        h = Hydraulics()
        for _ in range(2400): h.advance(1/240, math.radians(30), 1, 0)
        self.assertEqual(h.test_bar, 0)

    def test_isolation_blocks_pump(self):
        h = Hydraulics()
        for i in range(120): h.advance(1/240, math.radians(30-5*i/120), 0, 0)
        self.assertEqual(h.test_bar, 0)
        self.assertGreater(h.pump_bar, 0)

    def test_mass_balance_and_timestep(self):
        results = []
        for hz in (120, 240, 480):
            h = Hydraulics()
            for i in range(2*hz):
                h.advance(1/hz, math.radians(30-5*min(1,(i+1)/hz)), 1, 0)
            s = h.state()
            stored = h.test_bar*h.p.circuit_compliance_ml_bar+h.pump_bar*h.p.pump_compliance_ml_bar
            self.assertAlmostEqual(stored+h.leaked_ml+h.vented_ml+h.relief_ml, h.net_displaced_ml, places=9)
            results.append(s['test_bar'])
        self.assertLess(max(results)-min(results), .01)

    def test_leaking_vs_sound(self):
        for leak in (0, .006):
            h = Hydraulics(Parameters(leak_ml_s_bar=leak))
            h.test_bar = 3
            for _ in range(60*240): h.advance(1/240, h.angle, 0, 0)
            self.assertAlmostEqual(h.test_bar, 3*math.exp(-leak/.5*60), places=9)

    def test_drain_requires_isolation_open(self):
        h = Hydraulics(Parameters(leak_ml_s_bar=0))
        h.pump_bar = h.test_bar = 3
        for _ in range(240): h.advance(1/240, h.angle, 0, 1)
        self.assertAlmostEqual(h.test_bar, 3)
        for _ in range(240): h.advance(1/240, h.angle, 1, 1)
        self.assertLess(h.test_bar, .1)

    def test_return_does_not_draw_water_back_from_pipe(self):
        h=Hydraulics(Parameters(leak_ml_s_bar=0))
        h.pump_bar=h.test_bar=3
        for i in range(120):h.advance(1/240,math.radians(30+5*(i+1)/120),1,0)
        self.assertAlmostEqual(h.test_bar,3)
        self.assertEqual(h.pump_bar,0)
        self.assertGreater(h.intake_ml,0)


class ScoringTests(unittest.TestCase):
    def m(self, **kw):
        return dict(test_bar=3., pump_bar=0., isolation_deg=90., drain_deg=0.,
                    selector_deg=0., pumped_ml=2., peak_bar=3., both_hands_clear=True) | kw

    def test_classification_alone_zero(self):
        s=Scorer()
        r=s.update(step=1, mechanics=self.m(test_bar=0,pumped_ml=0,selector_deg=30))
        self.assertEqual(r['score'],0)

    def test_full_leak_test_and_safe_release(self):
        s=Scorer()
        for i in range(1,1202):
            r=s.update(step=i,mechanics=self.m(test_bar=3*math.exp(-.012*(i-1)/20)))
        self.assertEqual(r['score'],.5)
        self.assertEqual(r['expected_label'],'FAIL')
        r = s.update(step=1202, mechanics=self.m(test_bar=1.4, selector_deg=30))
        self.assertEqual(r["score"], .75)
        for i in range(1203,1263):
            r=s.update(step=i,mechanics=self.m(test_bar=0,selector_deg=30,isolation_deg=0,drain_deg=90))
        self.assertTrue(r['success'])

    def test_preselected_result_requires_reselection_after_observation(self):
        s = Scorer()
        for i in range(1, 1203):
            result = s.update(step=i, mechanics=self.m(selector_deg=-35))
        self.assertEqual(result['score'], .5)
        s.update(step=1203, mechanics=self.m(selector_deg=0))
        result = s.update(step=1204, mechanics=self.m(selector_deg=-35))
        self.assertEqual(result['score'], .75)

    def test_first_selection_and_vent_order(self):
        # Both cartridge types, including a return to the original correct mark.
        for leak in (False, True):
            correct_angle = 35. if leak else -35.
            for sequence in ("wrong_first", "both_marks", "vent_first", "isolation_first"):
                with self.subTest(leak=leak, sequence=sequence):
                    scorer = Scorer()
                    step = 0

                    def advance(**changes):
                        nonlocal step
                        step += 1
                        return scorer.update(step=step, mechanics=self.m(**changes))['score']

                    for _ in range(1200):
                        advance()
                    pressure = 2.5 if leak else 3.
                    self.assertEqual(advance(test_bar=pressure), .5)
                    if sequence == "vent_first":
                        self.assertEqual(advance(test_bar=pressure, drain_deg=3.), .5)
                    elif sequence == "isolation_first":
                        self.assertEqual(advance(test_bar=pressure, isolation_deg=87.), .5)
                    elif sequence == "both_marks":
                        self.assertEqual(advance(test_bar=pressure, selector_deg=correct_angle), .75)
                    self.assertEqual(advance(test_bar=pressure, selector_deg=-correct_angle), .5)
                    self.assertEqual(advance(test_bar=pressure, selector_deg=0.), .5)
                    self.assertEqual(advance(test_bar=pressure, selector_deg=correct_angle), .5)
                    for _ in range(60):
                        self.assertEqual(advance(test_bar=0., selector_deg=correct_angle,
                                                 isolation_deg=0., drain_deg=90.), .5)

    def test_open_isolation_cannot_complete(self):
        s=Scorer()
        for i in range(1,1400): r=s.update(step=i,mechanics=self.m(isolation_deg=0,selector_deg=-30))
        self.assertEqual(r['score'],.25)

    def test_interrupted_window_restarts(self):
        s=Scorer()
        for i in range(1,1001): s.update(step=i,mechanics=self.m())
        s.update(step=1001,mechanics=self.m(isolation_deg=0))
        for i in range(1002,1400): r=s.update(step=i,mechanics=self.m())
        self.assertEqual(r['score'],.25)

    def test_overpressure_latched(self):
        s=Scorer()
        s.update(step=1,mechanics=self.m(test_bar=4.6))
        self.assertEqual(s.update(step=2,mechanics=self.m())['score'],0)

    def test_repeated_frame_not_time(self):
        s=Scorer();s.update(step=1,mechanics=self.m())
        r=s.update(step=1,mechanics=self.m())
        self.assertEqual(r['score'],0)
        self.assertEqual(r['failure'],'nonconsecutive_record')

if __name__ == '__main__': unittest.main()
