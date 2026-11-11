"""Experimental fold/clip/integrity scorer over validated observations.

This is not wired to the Arena task. A caller must independently observe all
fields; unknown containment or fracture status is invalid, never treated as a
clean trial. Gap tolerance protects the continuous 3-second requirement.
"""
from dataclasses import dataclass
import math
from experimental_integrity_score import apply_integrity_penalty


@dataclass(frozen=True)
class Observation:
    time_s: float
    jaw_closed: bool
    both_jaws_contact: bool
    fold_retained: bool
    width_covered_m: float
    hands_clear: bool
    bag_on_table: bool
    spilled_count: int
    broken_count: int
    total_chips: int
    valid: bool = True


class DeformableScorer:
    def __init__(self, required_width_m=.096, required_hold_s=3., max_gap_s=.1):
        if min(required_width_m, required_hold_s, max_gap_s) <= 0:
            raise ValueError('scoring thresholds must be positive')
        self.required_width_m = required_width_m
        self.required_hold_s = required_hold_s
        self.max_gap_s = max_gap_s
        self.previous = None
        self.hold_start = None
        self.violation = False
        self.peak_broken = 0
        self.peak_spilled = 0

    def update(self, obs):
        if not obs.valid:
            self.hold_start = None
            self.previous = None
            return None
        flags = (obs.jaw_closed, obs.both_jaws_contact, obs.fold_retained, obs.hands_clear, obs.bag_on_table)
        if any(type(x) is not bool for x in flags) or not math.isfinite(obs.time_s) or not math.isfinite(obs.width_covered_m) or obs.width_covered_m < 0:
            raise ValueError('observation contains missing or invalid sensor values')
        # Enforce observed, bounded integrity counts even when the seal fails.
        apply_integrity_penalty(0., obs.total_chips, obs.spilled_count, obs.broken_count)
        if self.previous is not None and obs.time_s <= self.previous:
            raise ValueError('observation timestamps must strictly increase')
        gap = self.previous is not None and obs.time_s - self.previous > self.max_gap_s + 1e-9
        self.previous = obs.time_s
        self.violation |= not obs.bag_on_table
        self.peak_broken = max(self.peak_broken, obs.broken_count)
        self.peak_spilled = max(self.peak_spilled, obs.spilled_count)
        partial = obs.jaw_closed and obs.both_jaws_contact and obs.fold_retained
        full_now = partial and obs.width_covered_m >= self.required_width_m and obs.hands_clear
        if not full_now:
            self.hold_start = None
        elif self.hold_start is None or gap:
            self.hold_start = obs.time_s
        held = full_now and obs.time_s - self.hold_start >= self.required_hold_s - 1e-9
        score = 1. if held else .5 if partial else 0.
        if self.violation:
            return 0.
        return apply_integrity_penalty(score, obs.total_chips, self.peak_spilled, self.peak_broken)
