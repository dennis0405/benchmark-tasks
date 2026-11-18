"""Scoring from recorded pressure, native valve/selector poses and robot state."""
import math

HZ = 20
TARGET_BAR = 3.0
TARGET_TOL = .15
OBSERVE_SECONDS = 60.0
DROP_LIMIT_BAR = .20


class Scorer:
    def __init__(self):
        self.last_step = 0
        self.start_step = None
        self.start_pressure = None
        self.completed_drop = None
        self.expected_label = None
        self.selection_armed = False
        self.first_selection = None
        self.diagnosis_invalid = False
        self.charged = False
        self.released_frames = 0
        self.failure = None

    def update(self, *, step, objects=None, root_poses=None, robot=None, mechanics=None):
        m = mechanics or {}
        required = ('test_bar', 'pump_bar', 'isolation_deg', 'drain_deg',
                    'selector_deg', 'pumped_ml', 'peak_bar', 'both_hands_clear')
        if step != self.last_step+1:
            self.failure = 'nonconsecutive_record'
        self.last_step = step
        if any(k not in m for k in required):
            self.failure = 'missing_measured_evidence'
        if any(not math.isfinite(float(m[k])) for k in required if k in m):
            self.failure = 'nonfinite_measured_evidence'
        if self.failure:
            return dict(score=0., success=False, failure=self.failure)
        pressure = m['test_bar']
        if max(m['peak_bar'], pressure, m['pump_bar']) > 4.5:
            self.failure = 'excess_pressure'
            return dict(score=0., success=False, failure=self.failure)
        in_band = abs(pressure-TARGET_BAR) <= TARGET_TOL
        isolated = m['isolation_deg'] >= 88.
        drain_closed = m['drain_deg'] <= 2.
        charged = in_band and m['pumped_ml'] >= .2
        self.charged |= charged
        if self.completed_drop is None:
            if isolated and drain_closed:
                if self.start_step is None and charged:
                    self.start_step, self.start_pressure = step, pressure
                if self.start_step is not None and (step-self.start_step)/HZ >= OBSERVE_SECONDS:
                    self.completed_drop = self.start_pressure-pressure
                    self.expected_label = 'FAIL' if self.completed_drop > DROP_LIMIT_BAR else 'PASS'
            else:
                self.start_step = self.start_pressure = None
        label = 'FAIL' if m['selector_deg'] >= 25 else ('PASS' if m['selector_deg'] <= -25 else 'UNSET')
        if self.completed_drop is not None and not self.diagnosis_invalid:
            if self.first_selection is None:
                if not isolated or not drain_closed:
                    self.diagnosis_invalid = True
                elif label == 'UNSET':
                    self.selection_armed = True
                elif self.selection_armed:
                    self.first_selection = label
                    self.diagnosis_invalid = label != self.expected_label
            elif label != 'UNSET' and label != self.first_selection:
                self.diagnosis_invalid = True
        correct = (not self.diagnosis_invalid and self.first_selection is not None
                   and label == self.first_selection)
        released = (correct and pressure <= .1 and m['pump_bar'] <= .1 and
                    m['isolation_deg'] <= 10. and m['drain_deg'] >= 80. and
                    m['both_hands_clear'] is True)
        self.released_frames = self.released_frames+1 if released else 0
        success = self.released_frames >= 3*HZ
        score = 1. if success else .75 if correct else .5 if self.completed_drop is not None else .25 if self.charged else 0.
        return dict(score=score, success=success, failure=None, label=label,
                    expected_label=self.expected_label, test_drop_bar=self.completed_drop,
                    isolated_seconds=0. if self.start_step is None else min(OBSERVE_SECONDS,(step-self.start_step)/HZ),
                    released_frames=self.released_frames)
