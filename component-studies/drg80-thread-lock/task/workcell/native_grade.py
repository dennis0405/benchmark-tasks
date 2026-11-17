"""Bound the local score using native observations while hard-failure coverage is incomplete."""

from math import degrees


class NativeGradeBound:
    """The optimistic grader assumes unresolved damage is absent.

    A zero upper bound establishes zero credit. A positive bound still needs the
    remaining damage rules and exact clearance when the collider lower bound
    straddles 30 mm. No competition score is published by this diagnostic.
    """

    def __init__(self, scoring, hz):
        self.grader = scoring["StopLockScore"](hz)
        self.sample = scoring["Sample"]
        self.peak = 0.0

    def observe(self, r, tick, definitive_damage):
        # Clearance only enters the grader as a comparison, gated by both
        # tools being racked. An unresolved distance is optimistic here.
        clear_for_upper = 0.030 if r["tools_racked"] else 0.0
        record = self.sample(
            angle=r["angle"],
            angular_speed=r["angular_speed"],
            piston=r["p"],
            piston_speed=r["piston_speed"],
            opening_force=r["actuator_n"],
            stop_force=r["stop_n"],
            screw_angle=degrees(r["st"]),
            screw_height=r["sz"],
            screw_speed=degrees(r["screw_speed"]),
            nut_angle=degrees(r["st"] + r["nt"]),
            nut_speed=degrees(r["nut_speed"]),
            nut_seat_gap=r["seat_gap"],
            nut_preload=r["seat_n"],
            lever_angle=degrees(r["lever"]),
            robot_tool_clearance=clear_for_upper,
            released=r["released"],
            nuts_retained=r["nuts_retained"],
            tools_outside_adjustment=r["tools_outside_adjustment"],
            tools_racked=r["tools_racked"],
            fingers_open=r["fingers_open"],
            damage=definitive_damage,
        )
        score = self.grader.observe(record, tick)
        self.peak = max(score, self.peak)
        return score

    def report(self):
        return dict(
            native_ticks=self.grader.tick,
            current_upper=self.grader.score,
            peak_upper=self.peak,
            certain_current_zero=bool(self.grader.tick and self.grader.score == 0),
            official_submission=False,
        )
