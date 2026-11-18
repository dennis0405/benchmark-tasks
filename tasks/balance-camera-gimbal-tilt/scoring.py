"""At 20 Hz, score real, sequential hands-off releases with the spring clamp engaged."""
import math

HZ = 20
HOLD = 3 * HZ
TARGET = 30.0
START_WINDOW = 2.0
DRIFT = 3.0

class Scorer:
    def __init__(self):
        self.visited = set()
        self.anchor = None
        self.frames = 0
        self.last_step = 0
        self.open_start = None
        self.adjusted = False
        self.free_before = False
        self.validated_slide = None
        self.score = 0.0

    def update(self, *, step, mechanics, **unused):
        if step != self.last_step + 1:
            raise ValueError('A release needs consecutive observations from reset.')
        self.last_step = step
        angle, speed, slide, brake = (float(mechanics[k]) for k in
                                     ('pitch_deg','pitch_speed_deg_s','slide_m','brake_m'))
        impulse = float(mechanics['hand_impulse_Ns'])
        tcp_clear = bool(mechanics['both_hands_clear'])
        if not all(map(math.isfinite, (angle, speed, slide, brake, impulse))):
            raise ValueError('Invalid mechanism measurement')
        opened = brake > .003
        closed = abs(brake) < .0008
        if opened:
            if self.open_start is None:
                self.open_start = slide
            if abs(slide - self.open_start) >= .008:
                self.adjusted = True
        elif closed:
            self.open_start = None
        if opened or (self.validated_slide is not None and abs(slide-self.validated_slide) > .0005):
            self.visited.clear()
            self.validated_slide = None
        free = closed and tcp_clear and impulse < .0001
        if not free:
            self.frames = 0
            self.anchor = None
        elif not self.free_before and self.adjusted:
            side = 1 if angle >= 0 else -1
            if abs(angle-side*TARGET) <= START_WINDOW and abs(speed) <= 5:
                self.anchor = (side,angle,slide)
                self.frames = 0
        if free and self.anchor is not None:
            side, start, start_slide = self.anchor
            if abs(angle-start) > DRIFT or abs(angle-side*TARGET) > DRIFT or abs(slide-start_slide) > .0005:
                self.anchor = None
                self.frames = 0
            else:
                self.frames += 1
                # First sample marks release time; sixty successive intervals are three seconds.
                if self.frames > HOLD:
                    self.visited.add(side)
                    self.validated_slide = slide
        self.free_before = free
        self.score = len(self.visited)/2
        return dict(score=self.score,success=self.score==1.0,held_frames=self.frames,
                    verified_sides=sorted(self.visited),clear=free,adjusted=self.adjusted)
