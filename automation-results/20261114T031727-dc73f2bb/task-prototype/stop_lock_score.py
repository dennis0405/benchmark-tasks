"""Read-only DRG stop grader. Call once after every physics substep.

The scene adapter must measure each field, including the per-pair damage latch,
surface clearances and rack containment. No action or expert phase is an input.
Angles are degrees, lengths metres, time seconds and forces newtons.
"""
from dataclasses import dataclass, fields
from math import degrees, isfinite, radians


class EnergizedMotion:
    """One tracker per episode. Supply unwrapped absolute screw and nut angles in radians."""

    limits = (1e-6, radians(.01), radians(.01))
    channels = ("screw_axial_m", "screw_angle_rad", "nut_angle_rad")

    def __init__(self):
        self.previous = None
        self.low = self.high = None
        self.maximum = [0.] * 3
        self.first_failure = None
        self.windows = 0

    def observe(self, tick, force_n, screw_z_m, screw_rad, nut_rad):
        pose = (screw_z_m, screw_rad, nut_rad)
        if not all(map(isfinite, (force_n, *pose))):
            if self.first_failure is None:
                self.first_failure = (tick, "nonfinite")
            return True
        if abs(force_n) <= 1:
            self.low = self.high = None
        elif self.previous is not None:
            if self.low is None:
                self.windows += 1
                self.low = list(self.previous)
                self.high = list(self.previous)
            for i, value in enumerate(pose):
                self.low[i] = min(value, self.low[i])
                self.high[i] = max(value, self.high[i])
                span = self.high[i] - self.low[i]
                self.maximum[i] = max(self.maximum[i], span)
                if span > self.limits[i] and self.first_failure is None:
                    self.first_failure = (tick, self.channels[i])
        self.previous = pose
        return self.first_failure is not None


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
        self.energized_motion = EnergizedMotion()
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
        self.damaged |= self.energized_motion.observe(
            tick, s.opening_force, s.screw_height, radians(s.screw_angle), radians(s.nut_angle)
        )
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
