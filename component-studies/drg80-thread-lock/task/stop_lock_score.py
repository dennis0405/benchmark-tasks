"""Read-only DRG stop grader. Call once after every physics substep.

The scene adapter must measure each field, including the per-pair damage latch,
surface clearances and rack containment. No action or expert phase is an input.
Angles are degrees, lengths metres, time seconds and forces newtons.
"""
from dataclasses import dataclass, fields
from math import degrees, isfinite


@dataclass(frozen=True)
class Sample:
    angle: float
    angular_speed: float
    piston: float
    piston_speed: float
    opening_force: float
    stop_force: float
    screw_angle: float
    screw_height: float
    screw_speed: float
    nut_angle: float
    nut_speed: float
    nut_seat_gap: float
    nut_preload: float
    lever_angle: float
    robot_tool_clearance: float
    released: bool
    nuts_retained: bool
    tools_outside_adjustment: bool
    tools_racked: bool
    fingers_open: bool
    damage: bool


class StopLockScore:
    def __init__(self, hz):
        if hz not in (240, 480):
            raise ValueError("Expected a verified 240 or 480 Hz physics stream")
        self.hz = hz
        self.previous = None
        self.tick = 0
        self.damaged = False
        self.preloaded = False
        self.preload_ticks = self.coarse_ticks = self.clear_ticks = 0
        self.precise_angles = []
        self.certificate = None
        self.cycle = "idle"
        self.endpoints = []
        self.bad_endpoint_ticks = 0
        self.score = 0.0

    def invalidate(self):
        self.certificate = None
        self.endpoints.clear()
        self.cycle = "idle"
        self.clear_ticks = 0
        self.precise_angles.clear()

    def observe(self, sample, tick):
        if tick != self.tick + 1:
            raise ValueError("Missing, duplicate, or reordered physics substep")
        self.tick = tick
        prior, self.previous = self.previous, sample
        s = sample
        finite = all(isfinite(getattr(s, f.name)) for f in fields(s))
        self.damaged |= not finite or s.damage or not s.nuts_retained
        if prior is not None and abs(s.opening_force) > 1:
            self.damaged |= (max(abs(s.screw_angle-prior.screw_angle), abs(s.nut_angle-prior.nut_angle))
                             * self.hz > degrees(.01) or abs(s.screw_height-prior.screw_height)*self.hz > .00001)
        if self.damaged:
            self.score = 0.0
            return self.score

        window = self.hz // 4
        seated = -.00010 <= s.nut_seat_gap <= .00050
        loaded = seated and s.released and s.nut_preload >= 2
        self.preload_ticks = self.preload_ticks + 1 if (
            loaded and abs(s.opening_force) <= .10 and max(abs(s.screw_speed), abs(s.nut_speed)) <= 1
        ) else 0
        self.preloaded = loaded and (self.preloaded or self.preload_ticks >= window)
        opened = 25 <= s.lever_angle <= 35
        stopped = (opened and s.opening_force > 1 and s.stop_force >= .1
                   and abs(s.angular_speed) <= 2 and abs(s.piston_speed) <= .005)
        endpoint = self.preloaded and stopped
        coarse = endpoint and abs(s.angle - 90) <= 6
        self.coarse_ticks = self.coarse_ticks + 1 if coarse else 0
        if not seated or not s.released or (self.certificate and (
            abs(s.screw_angle-self.certificate[0]) > 1 or abs(s.screw_height-self.certificate[1]) > .0001
        )):
            self.invalidate()
        self.bad_endpoint_ticks = self.bad_endpoint_ticks+1 if stopped and abs(s.angle-90)>3 else 0
        if self.bad_endpoint_ticks >= window:
            self.invalidate()
        if endpoint and abs(s.angle-90) <= 3:
            self.precise_angles.append(s.screw_angle)
            self.precise_angles = self.precise_angles[-window:]
        else:
            self.precise_angles.clear()
        precise = len(self.precise_angles) == window and abs(self.precise_angles[-1]-self.precise_angles[0]) <= 1
        if not self.certificate and precise:
            self.certificate = s.screw_angle, s.screw_height

        if self.certificate and prior is not None:
            dp = s.piston-prior.piston
            closed = -35 <= s.lever_angle <= -25
            coherent = abs(s.angle - (3+degrees(300*s.piston))) <= 1 and s.released
            if not coherent:
                self.cycle = "idle"
            elif self.cycle == "idle" and closed and prior.angle > 10 >= s.angle and dp < -1e-7:
                self.cycle = "closing"
            if self.cycle == "closing":
                if dp > 1e-7:
                    self.cycle = "idle"
                elif s.piston <= .00060:
                    self.cycle = "closed"
            elif self.cycle == "closed" and dp > 1e-7:
                self.cycle = "opening" if opened else "idle"
            if self.cycle in ("opening", "past80"):
                if dp < -1e-7:
                    self.cycle = "idle"
                elif prior.angle < 80 <= s.angle:
                    self.cycle = "past80"
                if self.cycle == "past80" and precise:
                    self.endpoints.append(s.angle)
                    if max(self.endpoints)-min(self.endpoints)>2:
                        self.endpoints.clear()
                    self.certificate = s.screw_angle, s.screw_height
                    self.cycle = "idle"
        clear = (self.certificate and len(self.endpoints)>=3 and precise and s.tools_racked
                 and s.robot_tool_clearance >= .030 and s.fingers_open)
        self.clear_ticks = self.clear_ticks + 1 if clear else 0
        self.score = max(.25 if self.coarse_ticks>=window else 0, .5 if self.certificate else 0,
                         .75 if self.certificate and self.endpoints and s.tools_outside_adjustment else 0,
                         1 if self.clear_ticks>=self.hz else 0)
        return self.score
