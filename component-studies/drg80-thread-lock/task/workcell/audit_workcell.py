"""Read-only necessary-condition audit for commissioning traces, never a YAM grade."""

import argparse
import json
import math
from pathlib import Path


def audit(rows, hz):
    from runpy import run_path

    grader = run_path(
        str(Path(__file__).resolve().parent.parent / "stop_lock_score.py")
    )
    motion = grader["EnergizedMotion"]()
    first = None
    for tick, r in enumerate(rows, 1):
        violation = motion.observe(
            tick, r["actuator_n"], r["sz"], r["st"], r["st"] + r["nt"]
        )
        if violation and first is None:
            first = {
                "tick": tick,
                "phase": r["phase"],
                "channel": motion.first_failure[1],
            }
    peak = [
        motion.maximum[0] * 1e6,
        math.degrees(motion.maximum[1]),
        math.degrees(motion.maximum[2]),
    ]

    def last(phase):
        return [r for r in rows if r["phase"] == phase][-hz // 4 :]

    release = last("released")
    opens = [last("open" + str(i)) for i in (2, 3, 4)]
    closes = [last("close" + str(i)) for i in (1, 2, 3)]
    release_n = min((r["seat_n"] for r in release), default=None)
    endpoints = [s[-1]["angle"] for s in opens if s]
    checks = {
        "retained_preload": bool(release and release_n >= 2),
        "three_open_endpoints": all(
            len(s) == hz // 4
            and all(abs(r["angle"] - 90) <= 3 and r["stop_n"] >= 0.1 for r in s)
            for s in opens
        ),
        "three_closed": all(
            len(s) == hz // 4 and all(r["angle"] <= 10 for r in s) for s in closes
        ),
        "preload_at_open_endpoints": all(
            len(s) == hz // 4 and min(r["seat_n"] for r in s) >= 2 for s in opens
        ),
        "powered_motion": first is None and motion.windows > 0,
    }
    return {
        "scope": "instrumented fixture only; these are necessary conditions, no task points",
        "rate_hz": hz,
        "checks": checks,
        "release_min_n": release_n,
        "open_endpoints_deg": endpoints,
        "peak_powered_spans_um_deg_deg": peak,
        "first_powered_violation": first,
        "all_necessary_pass": all(checks.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("trace", type=Path)
    p.add_argument("--hz", type=int, default=240)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = audit(json.loads(a.trace.read_text()), a.hz)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["checks"]))
