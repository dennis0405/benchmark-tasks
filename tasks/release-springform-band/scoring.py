"""Ordered release of the real lip, followed by independent support on the second mat."""
import numpy as np

class Scorer:
    def __init__(self):
        self.violation = False
        self.removed = False
        self.hold = 0
        self.previous = None
        self.passed = np.zeros(40, dtype=bool)

    def update(self, *, mechanics, **kwargs):
        m = mechanics
        x,y,_ = m['band_xyz']
        height = np.asarray(m['lip_above_plate_mm'])
        margin = np.asarray(m['lip_radial_margin_mm'])
        self.violation |= m['motion_mm'] > 3 or m['tilt_deg'] > 3
        if self.previous is not None:
            old_height,old_margin = self.previous
            # Determine the radial clearance when each facet actually crossed the
            # disk top. Opening the lever later cannot rewrite that crossing.
            rising = (old_height <= 0) & (height > 0)
            fraction = -old_height[rising] / (height[rising]-old_height[rising])
            at_crossing = old_margin[rising]+fraction*(margin[rising]-old_margin[rising])
            self.passed[rising] = at_crossing >= 0
        self.passed[height <= 0] = False
        self.previous = (height,margin)
        self.removed |= bool(self.passed.all() and m['lip_above_cake_mm'] > 0)
        if max(height) <= 0 and np.hypot(x-.32,y+.097) < .015:
            self.removed = False
        on_target = m['on_destination_mat'] and abs(x-.32) < .105 and abs(y-.097) < .0945
        stable = m['speed'] < .015 and m['spin'] < .15
        partial = self.removed and not self.violation
        resting = partial and on_target and stable and not m['robot_contacts']
        self.hold = self.hold+1 if resting else 0
        success = self.hold >= 21  # 21 samples at 20 Hz span one full second.
        return dict(score=1.0 if success else .5 if partial else 0.0,success=success,
                    held_frames=self.hold,disturbed=self.violation,facets_passed=int(self.passed.sum()),
                    removed=bool(self.removed))
