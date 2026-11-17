"""Objective credit from measured outcomes at the native physics rate."""

from dataclasses import dataclass


@dataclass
class Outcome:
    interval: float
    dwell_steps: int = 0
    failed: bool = False

    def advance(self, recovered_fraction, both_supported, released, fault):
        self.failed |= bool(fault)
        qualifies = (
            recovered_fraction >= 0.98
            and both_supported
            and released
            and not self.failed
        )
        self.dwell_steps = self.dwell_steps + 1 if qualifies else 0
        if self.failed:
            return 0.0
        return 0.90 * min(max(recovered_fraction, 0) / 0.98, 1.0) + 0.10 * (
            self.dwell_steps * self.interval >= 3.0
        )
