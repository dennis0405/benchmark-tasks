"""Provisional integrity penalty applied only to independently validated events.

Not wired to Arena or the current rigid-packet scorer. `broken_count` must mean
actual breakage, not an uncalibrated strain reading; callers must not turn
unavailable measurements into zero events.
"""
import math


def apply_integrity_penalty(seal_score, total_chips, spilled_count, broken_count):
    if seal_score not in (0.0, 0.5, 1.0):
        raise ValueError('seal_score must be 0, 0.5, or 1')
    if (not isinstance(total_chips, int) or isinstance(total_chips, bool) or total_chips < 1 or
        not isinstance(spilled_count, int) or isinstance(spilled_count, bool) or
        not isinstance(broken_count, int) or isinstance(broken_count, bool) or
        not 0 <= spilled_count <= total_chips or not 0 <= broken_count <= total_chips):
        raise ValueError('chip counts must be observed integers within bounds')
    if spilled_count or broken_count >= math.ceil(total_chips / 2):
        return 0.0
    if broken_count:
        return min(seal_score, 0.5)
    return seal_score
