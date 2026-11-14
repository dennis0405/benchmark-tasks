"""Pure-Python outcome scorer for native observer snapshots.

The simulator adapter owns observation and ordering.  This module only scores
an immutable, already validated POST snapshot; absent or stale evidence fails
closed.  It deliberately cannot infer a fracture from a tang pose.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def score_snapshot(snapshot: Mapping | None) -> float:
    if not snapshot or snapshot.get("phase") != "POST":
        return 0.0
    required = ("generation_id", "interval_id", "timestamp_s", "removed_ids",
                "collected_ids", "hard_failure", "terminal_hold_count",
                "terminal_hold_span_s", "camera_valid")
    if any(k not in snapshot for k in required):
        return 0.0
    if snapshot["hard_failure"] or not math.isfinite(float(snapshot["timestamp_s"])):
        return 0.0
    removed = set(snapshot["removed_ids"])
    collected = set(snapshot["collected_ids"])
    originals = {0, 1, 2, 3}
    if not removed <= originals or not collected <= removed:
        return 0.0
    terminal = (
        removed == originals
        and collected == originals
        and bool(snapshot["camera_valid"])
        and int(snapshot["terminal_hold_count"]) >= 41
        and float(snapshot["terminal_hold_span_s"]) >= 2.0
    )
    return 0.125 * len(removed) + 0.10 * len(collected) + 0.10 * int(terminal)


def success_snapshot(snapshot: Mapping | None) -> bool:
    return abs(score_snapshot(snapshot) - 1.0) <= 1e-12
