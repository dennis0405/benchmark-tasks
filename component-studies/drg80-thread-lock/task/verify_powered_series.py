"""Check four fresh native traces, compare against the actual grader, and record their hashes."""
import argparse
import hashlib
import json
from dataclasses import replace
from math import degrees
from pathlib import Path
from stop_lock_score import Sample, StopLockScore


def replay(root, hz, fault):
    tag = f"dual-rev{'fault' if fault else 'normal'}{hz}"
    result = json.loads((root/f"{tag}.json").read_bytes())
    raw = (root/f"{tag}.trace.json").read_bytes()
    rows = json.loads(raw)
    assert len(rows) == 14 * hz and result["hz"] == hz
    assert result["case"] == ("powered-adjustment" if fault else "normal")
    # Missing workcell hardware cannot earn points. Only the powered hazard is replayed.
    neutral = Sample(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,False,True,False,False,False,False)
    grader = StopLockScore(hz)
    onset = None
    for i, row in enumerate(rows, 1):
        assert row["step"] == i
        if row["phase"]=="proof_load" and row["driver_torque_nm"] and onset is None:
            onset = i
        grader.observe(replace(neutral,
                               opening_force=row["proof_force_n"], screw_height=row["screw_z_m"],
                               screw_angle=degrees(row["screw_angle_rad"]), nut_angle=degrees(row["angle_rad"])), i)
    motion = grader.energized_motion
    assert motion.windows == result["windows"] == 1
    assert list(motion.first_failure or ()) == (result["first_failure"] or [])
    for name, value in zip(motion.channels, motion.maximum):
        assert abs(value-result["span"][name]) < 1e-12
    assert len(result["checks"]) == 17
    assert result["overall_pass"] == (not fault)
    if fault:
        assert onset == int(13.25*hz)+1
        assert onset <= motion.first_failure[0] < onset + hz//4
        assert grader.damaged and grader.score == 0
        assert min(row["normal_n"] for row in rows[13*hz:onset-1]) >= 2
        assert min(row["piston_stop_n"] for row in rows[13*hz:onset-1]) >= .1
        assert min(row["proof_force_n"] for row in rows[onset-1:onset-1+hz//4]) > 1
        assert result["proof_min_n"] < 2
    else:
        assert onset is None and not grader.damaged
    return dict(case=result["case"], hz=hz, samples=len(rows), trace_sha256=hashlib.sha256(raw).hexdigest(),
                trace_bytes=len(raw), axial_um=1e6*motion.maximum[0], screw_deg=degrees(motion.maximum[1]),
                nut_deg=degrees(motion.maximum[2]), fault_injection_step=onset, first_failure=motion.first_failure,
                proof_seat_min_N=result["proof_min_n"], piston_stop_min_N=result["piston_stop_min_N"],
                checks_passed=sum(result["checks"].values()), checks_total=len(result["checks"]),
                expected_outcome_verified=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("measurements", type=Path)
    args = p.parse_args()
    outcomes = []
    for hz in (240, 480):
        normal = replay(args.measurements, hz, False)
        fault = replay(args.measurements, hz, True)
        original, injected = [json.loads((args.measurements/f"dual-rev{case}{hz}.trace.json").read_text())
                              for case in ("normal", "fault")]
        onset = fault["fault_injection_step"] - 1
        fault["identical_prefault_rows"] = original[:onset] == injected[:onset]
        outcomes.extend((normal, fault))
    print(json.dumps(outcomes, indent=2))
