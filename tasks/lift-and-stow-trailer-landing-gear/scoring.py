"""Score a grounded start, supported low turn, lift, high turn, and hands-free parking."""
from limits import CLEAR_MM, STROKE

class Scorer:
    def __init__(self):
        self.prev = self.hold = self.air_steps = 0
        self.started_loaded = self.early = self.low = self.shift = self.full = False
        self.grip_angle = self.high_start = None

    def update(self, *, step, objects, root_poses, robot, mechanics):
        if step != self.prev + 1:
            raise ValueError('Missing consecutive physics observation')
        self.prev = step
        m = mechanics
        if step == 1:
            self.started_loaded = (m['ground_points'] > 0 and m['ground_force_n'] > .005
                                   and abs(m['clearance_mm']) < .02 and m['slide_mm'] < 5.6)
        both = len(set(m['pads']) & {'left:tip_left','left:tip_right','right:tip_left','right:tip_right'}) == 4
        clear = m['ground_points'] == 0 and m['clearance_mm'] >= CLEAR_MM
        in_low = m['slide_mm'] < 5.6
        if not self.shift and not in_low and (not self.low or not clear):
            self.early = True
        if not self.shift and not clear:
            self.low = False
        if both and in_low and self.started_loaded:
            if m['ground_points'] > 0 and m['ground_force_n'] > .005 and abs(m['clearance_mm']) < .02:
                self.grip_angle = m['low_deg'] if self.grip_angle is None else min(self.grip_angle, m['low_deg'])
            self.air_steps = self.air_steps + 1 if clear else 0
            if self.air_steps >= 3 and self.grip_angle is not None and m['low_deg'] - self.grip_angle >= 15:
                self.low = True
        else:
            self.grip_angle = None
            self.air_steps = 0
        if self.low and both and clear and m['slide_mm'] >= 22.4 and not self.shift:
            self.shift = True
            self.high_start = m['high_deg']
        retracted = m['clearance_mm'] >= STROKE*1000-.045
        if self.shift and m['high_deg']-self.high_start >= 40 and retracted:
            self.full = True
        parked = (self.full and clear and retracted and m['caught'] and m['fold'] >= 85
                  and m['gap_mm'] < 20 and m['slide_mm'] >= 22.4 and m['hands_clear'])
        self.hold = self.hold + 1 if parked else 0
        valid = self.started_loaded and not self.early and step <= 1400
        success = valid and self.hold >= 61
        score = .25 * (self.low + self.shift + self.full + success) if valid else 0.
        return dict(score=score, success=success, started_loaded=self.started_loaded,
                    early_shift=self.early, held_frames=self.hold)
