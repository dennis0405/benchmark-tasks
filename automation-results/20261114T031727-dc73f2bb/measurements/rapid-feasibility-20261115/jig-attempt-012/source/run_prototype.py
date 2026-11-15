#!/usr/bin/env python3
"""Trusted-host runner. Launches the pinned standard YAM prototype and writes one fresh report."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "lm-rail-parallel-jig"
DEFAULT_REPORT = ROOT / "measurements" / CANDIDATE / "measurement.json"


def _numbers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        yield float(value)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)


def _action_rows(node):
    rows = []
    if isinstance(node, dict):
        for key, value in node.items():
            low = str(key).lower()
            if "action" in low and isinstance(value, list):
                for item in value:
                    nums = list(_numbers(item))
                    if nums:
                        rows.extend(nums[start : start + 14] for start in range(0, len(nums) - 13, 14))
                    else:
                        rows.extend(_action_rows(item))
                continue
            if low == "data" and isinstance(value, list):
                nums = list(_numbers(value))
                rows.extend(nums[start : start + 14] for start in range(0, len(nums) - 13, 14))
                continue
            rows.extend(_action_rows(value))
    elif isinstance(node, list):
        for item in node:
            rows.extend(_action_rows(item))
    return rows


def _score_values(node):
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in {"task_score", "score"} and isinstance(value, (int, float)):
                out.append(float(value))
            out.extend(_score_values(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(_score_values(item))
    return out


def _run(cmd, *, env=None, stdout=None):
    return subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=subprocess.STDOUT, text=True, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--control-hz", type=int, default=20)
    args = parser.parse_args()
    report = (ROOT / args.report).resolve()
    if ROOT not in report.parents:
        raise SystemExit("--report must stay inside the experiment workspace")
    out = report.parent
    logs = out / "inspect_logs"
    raw = out / "raw"
    artifacts = out / "artifacts"
    for path in (out, logs, raw, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    for path in (report, raw / "body_states.jsonl", raw / "actions.json", raw / "runner.log", artifacts / "top_cam_h264.mp4", artifacts / "setup.png"):
        if path.exists():
            path.unlink()

    started = time.time()
    checks = []
    requirements = [
        "Pinned Isaac Sim/YAM runtime launches the unmodified yam_arena embodiment.",
        "The scripted expert reads permitted object poses and joint states, and commands standard guarded 14-D joint actions. It never writes simulator object state.",
        "The compound jig and the loose rail interact through native rigid contact; no pose or velocity writes occur after initialization.",
        "The jig traverses all six stations while engaged; final rail endpoint spacing errors are at most 2 mm, the jig is parked, and both bodies rest for two seconds.",
        "Raw actions, per-step body states, an H.264 top-camera video, and setup image are produced.",
    ]
    measurement = {
        "schema": "robocurve.rapid_feasibility.measurement.v1",
        "candidate_id": CANDIDATE,
        "prototype_kind": "actual_standard_yam",
        "fresh_started_unix_s": started,
        "requirements": requirements,
        "checks": checks,
        "actual_measurements": {},
        "artifacts": {},
        "certification_note": "A prototype task_score of 1.0 establishes only this run's physical feasibility; it is not final task certification.",
    }
    exit_code = 1
    try:
        env = os.environ.copy()
        env["RAPID_BODY_TRACE"] = str(raw / "body_states.jsonl")
        env["RAPID_JOINT_TRACE"] = str(raw / "joint_states.jsonl")
        task = ROOT / "task" / CANDIDATE / "task.py"
        cmd = [
            sys.executable, str(Path(__file__).with_name("oracle_demo_cli.py")), "run", "--embodiment", "yam_arena", "--policy", "rcauto_oracle_jig_demo", "-P", "ki=0.5", "-P", "hold_s=5",
            "-E", f"task={task}", "-E", "object_jitter=0", "-E", f"control_hz={args.control_hz}",
            "-E", "cam_width=640", "-E", "cam_height=352",
            "--instruction", "Hold the loose rail steady with one gripper. Use the orange jig with the other to align it parallel to the fixed rail, sweeping the jig from the near station to the far station. Leave the rails parallel and return the jig to its blue mark.",
            "--scorer", "task_score", "--max-steps", str(args.max_steps), "--epochs", "1", "--grader", "none", "--no-prompt", "--store-frames", "--log-dir", str(logs),
        ]
        with (raw / "runner.log").open("w", encoding="utf-8") as stream:
            _run(cmd, env=env, stdout=stream)
        checks.append({"id": "runner_exit", "passed": True, "actual": 0})

        log_files = sorted(logs.glob("adhoc_*.json"), key=lambda p: p.stat().st_mtime)
        if not log_files:
            raise RuntimeError("inspect-robots produced no JSON trial log")
        trial_log = log_files[-1]
        data = json.loads(trial_log.read_text(encoding="utf-8"))
        # The canonical eval log intentionally contains aggregates only. Raw policy
        # chunks live in default-on sidecars under the same log directory.
        action_relative = data["samples"][0]["trial_metadata"][0]["actions"]
        action_file = (logs / action_relative).resolve()
        if logs.resolve() not in action_file.parents:
            raise RuntimeError("action trace escapes the trial log directory")
        actions = [row["action"] for line in action_file.read_text().splitlines()
                   if (row := json.loads(line)).get("action") is not None]
        if any(len(row) != 14 for row in actions):
            raise RuntimeError("actual action journal has a non-14-D row")
        (raw / "actions.json").write_text(json.dumps(actions, indent=2) + "\n", encoding="utf-8")
        finite_actions = bool(actions) and all(math.isfinite(v) for row in actions for v in row)
        left_range = max((max(row[i] for row in actions) - min(row[i] for row in actions) for i in range(6)), default=0.0)
        right_range = max((max(row[i] for row in actions) - min(row[i] for row in actions) for i in range(7, 13)), default=0.0)
        checks.append({"id": "finite_action_trace", "passed": finite_actions, "actual": len(actions)})
        checks.append({"id": "both_arms_moved", "passed": left_range > 0.05 and right_range > 0.05, "actual": min(left_range, right_range)})

        body_lines = (raw / "body_states.jsonl").read_text(encoding="utf-8").splitlines()
        final_body = json.loads(body_lines[-1]) if body_lines else {}
        final_errors = final_body.get("errors", {})
        if not body_lines or not all(math.isfinite(float(final_errors[k])) for k in ("centre_xy_m","height_m","yaw_rad","datum_gap_m")):
            raise RuntimeError("missing/non-finite native task-state trace")
        checks.append({"id": "body_state_trace", "passed": bool(body_lines), "actual": len(body_lines)})

        score = float(data["results"]["metrics"]["task_score"])
        checks.append({"id": "native_task_score", "passed": score >= 1.0, "actual": score})

        _run(["inspect-robots", "video", str(trial_log)])
        videos = sorted((logs / "frames" / action_file.parent.name).glob("*top_cam.mp4"))
        if not videos:
            raise RuntimeError("inspect-robots video produced no MP4")
        source_video = videos[0]
        _run(["ffmpeg", "-y", "-i", str(source_video), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(artifacts / "top_cam_h264.mp4")])
        _run(["ffmpeg", "-y", "-i", str(artifacts / "top_cam_h264.mp4"), "-frames:v", "1", str(artifacts / "setup.png")])
        checks.append({"id": "media", "passed": (artifacts / "top_cam_h264.mp4").stat().st_size > 0 and (artifacts / "setup.png").stat().st_size > 0, "actual": (artifacts / "top_cam_h264.mp4").stat().st_size})

        measurement["actual_measurements"] = {
            "task_score": score,
            "action_rows": len(actions),
            "body_state_samples": len(body_lines),
            "left_action_span_proxy_rad": left_range,
            "right_action_span_proxy_rad": right_range,
            "final_centre_xy_error_m": float(final_errors["centre_xy_m"]),
            "final_height_error_m": float(final_errors["height_m"]),
            "final_yaw_error_rad": float(final_errors["yaw_rad"]),
            "final_datum_gap_m": float(final_errors["datum_gap_m"]),
            "elapsed_wall_s": time.time() - started,
        }
        measurement["artifacts"] = {
            "trial_log": str(trial_log.relative_to(ROOT)),
            "runner_log": str((raw / "runner.log").relative_to(ROOT)),
            "raw_actions": str((raw / "actions.json").relative_to(ROOT)),
            "raw_body_states": str((raw / "body_states.jsonl").relative_to(ROOT)),
            "h264_video": str((artifacts / "top_cam_h264.mp4").relative_to(ROOT)),
            "setup_image": str((artifacts / "setup.png").relative_to(ROOT)),
        }
        exit_code = 0 if all(item["passed"] for item in checks) else 2
    except Exception as exc:
        checks.append({"id": "runner_exception", "passed": False, "actual": 1, "detail": f"{type(exc).__name__}: {exc}"})
        measurement["actual_measurements"]["elapsed_wall_s"] = time.time() - started
    finally:
        measurement["completed"] = exit_code == 0
        measurement["fresh_completed_unix_s"] = time.time()
        report.write_text(json.dumps(measurement, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
