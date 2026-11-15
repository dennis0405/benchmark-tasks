#!/usr/bin/env python3
"""CPU-only declaration and IK reachability checks; this is not native physics success."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "lm-rail-parallel-jig"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=f"measurements/{CANDIDATE}/material_check.json")
    args = parser.parse_args()
    report = (ROOT / args.report).resolve()
    if ROOT not in report.parents:
        raise SystemExit("--report must stay inside the experiment workspace")
    report.parent.mkdir(parents=True, exist_ok=True)
    task = load(ROOT / "task" / CANDIDATE / "task.py")
    demos = task.demo()
    plans = {
        "left": ArmPlan(demos["left"], LEFT_BASE_POS),
        "right": ArmPlan(demos["right"], RIGHT_BASE_POS),
    }
    samples = {}
    checks = [{"id": "taskfile_load", "passed": True, "actual": len(task.objects)}]
    for name, plan in plans.items():
        actions = []
        for t in np.linspace(0.0, plan.duration, round(plan.duration * 20) + 1):
            action, _ = plan.action7(float(t))
            actions.append(action.tolist())
        finite = bool(actions) and bool(np.isfinite(np.asarray(actions)).all())
        samples[name] = {
            "duration_s": plan.duration,
            "action_samples": len(actions),
            "worst_ik_position_error_mm": plan.worst_ik_mm,
            "worst_ik_rotation_error_deg": plan.worst_ik_deg,
        }
        checks.append({"id": f"{name}_finite_ik", "passed": finite and plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 0.5, "actual": plan.worst_ik_mm})
    output = {
        "schema": "robocurve.rapid_feasibility.material_check.v1",
        "candidate_id": CANDIDATE,
        "prototype_kind": "material_only_no_physics",
        "requirements": ["TaskFile declaration loads", "Both public waypoint plans are finite and IK-reachable"],
        "checks": checks,
        "actual_measurements": samples,
        "warning": "This report does not establish grasp, contact, shared-load stability, media, or native task success.",
    }
    report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if all(c["passed"] for c in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
