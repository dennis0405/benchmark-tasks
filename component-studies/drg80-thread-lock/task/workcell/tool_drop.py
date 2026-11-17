"""Detect unsupported tool falls and tabletop loss on successive native samples."""

from math import isfinite


class ToolDrop:
    def __init__(self, table_min_xy, table_max_xy, tabletop_z):
        self.lo = tuple(float(v) - 0.050 for v in table_min_xy)
        self.hi = tuple(float(v) + 0.050 for v in table_max_xy)
        self.low_z = tabletop_z - 0.050
        self.support_z = None
        self.fell = False
        self.failure = None

    def observe(self, tick, centre, supported, equipment_impact):
        x, y, z = centre
        reason = None
        if not all(map(isfinite, centre)):
            reason = "nonfinite_tool_centroid"
        elif z < self.low_z or not (
            self.lo[0] <= x <= self.hi[0] and self.lo[1] <= y <= self.hi[1]
        ):
            reason = "tool_outside_table"
        elif supported:
            self.support_z = z
            self.fell = False
        elif self.support_z is not None:
            self.fell |= self.support_z - z >= 0.050
            if self.fell and equipment_impact:
                reason = "unsupported_fall_with_impact"
        if reason is not None and self.failure is None:
            self.failure = (tick, reason)
        return self.failure
