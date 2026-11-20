"""Score visible foot, selector, crank and hand positions at 20 Hz."""


class Scorer:
    def __init__(self):
        self.grounded = False
        self.low_lift = False
        self.early_shift = False
        self.shifted = False
        self.high_start_mm = None
        self.retracted = False
        self.hold = 0
        self.last_step = 0

    def update(self, *, step, objects, root_poses, robot, mechanics):
        height = mechanics["clearance_mm"]
        selector = mechanics["slide_mm"]
        low = selector <= 5.6
        high = selector >= 22.4
        if step == 1:
            self.grounded = bool(abs(height) <= 2 and low)

        if self.grounded and low and height >= 3:
            self.low_lift = True
        if not self.shifted and not low and (not self.low_lift or height < 3):
            self.early_shift = True

        in_band = 300 <= height <= 310
        if high and self.low_lift and not self.early_shift:
            self.shifted = True
            if self.high_start_mm is None:
                self.high_start_mm = height
            if in_band and self.high_start_mm <= 295:
                self.retracted = True
        else:
            self.high_start_mm = None
        if not in_band:
            self.retracted = False

        parked = (self.retracted and mechanics["fold"] >= 80
                  and mechanics["gap_mm"] <= 20 and min(mechanics["distance"]) >= 0.050)
        if step != self.last_step + 1:
            self.hold = 0
        self.hold = self.hold + 1 if parked else 0
        self.last_step = step
        success = bool(self.hold >= 61 and step <= 24000)
        partial = self.retracted and step <= 24000
        return dict(score=1.0 if success else 0.5 if partial else 0.0,
                    success=success, held_frames=self.hold, grounded=self.grounded,
                    low_lift=self.low_lift, early_shift=self.early_shift,
                    shifted=self.shifted, full_travel=self.retracted)
