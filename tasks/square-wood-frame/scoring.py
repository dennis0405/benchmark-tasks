"""Cumulative milestones from observed jaw placement, frame deformation and hand clearance."""
class Scorer:
    def __init__(self):
        self.handled = False
        self.seated_span = None
        self.corrected = False
        self.hold_start = None

    def update(self, *, step, objects, root_poses, robot, mechanics):
        m = mechanics
        self.handled |= m['hand_on_clamp']
        seated = m['both_seated']
        if self.seated_span is None and self.handled and seated and m['from_rest_mm'] >= 60:
            self.seated_span = m['jaw_span_mm']
        placed = self.seated_span is not None
        if placed and seated and self.seated_span - m['jaw_span_mm'] >= 5:
            self.corrected |= bool(abs(m['difference_mm']) <= m['initial_error_mm'] / 2)
        square = abs(m['difference_mm']) <= 1.0 and m['max_angle_error_deg'] <= .2
        stable = self.corrected and seated and square and m['hands_clear']
        if not stable:
            self.hold_start = None
        elif self.hold_start is None:
            self.hold_start = step
        frames = 0 if self.hold_start is None else step - self.hold_start + 1
        success = frames >= 61
        return dict(score=1.0 if success else .5 if self.corrected else .25 if placed else 0.,
                    placed=placed, corrected=self.corrected, held_frames=frames, success=success)
