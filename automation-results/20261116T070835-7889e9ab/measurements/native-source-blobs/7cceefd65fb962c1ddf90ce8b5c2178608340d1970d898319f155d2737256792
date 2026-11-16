"""Run the frozen 37-case native aperture-release engineering exploration.

This is a preflight, not production qualification.  Each case is a fresh
Isaac Sim process and retains its own native report and gzip trace.  The
aggregate pass concerns complete, finite, source-defined data collection only;
physical release outcomes are reported separately and are never pass-filled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


# These are the child checks that establish that a case produced trustworthy
# native acquisition data.  Physical-outcome checks are intentionally absent:
# release, crossing, and even initial bridging may legitimately fail in this
# engineering grid without making the measurement invalid.
REQUIRED_ACQUISITION_CHECKS = (
    "physics_dt_absolute_error",
    "finite_native_samples",
    "medium_native_physx_ccd_enabled",
    "moving_aperture_native_physx_ccd_enabled",
    "contact_processing_enabled_readback_before_first_reset",
    "settings_manager_omniverse_mode",
    "fresh_physics_sample_count",
    "cuda_backed_raw_contact_points",
    "populated_raw_contact_points",
    "uniquely_mapped_raw_contact_points",
    "nonzero_signed_scalar_raw_contact_points",
    "verified_signed_scalar_times_normal_reaction_points",
    "positive_pair_force_samples_with_verified_vector_reconciliation",
    "maximum_signed_scalar_vector_sum_relative_residual",
    "minimum_signed_scalar_vector_sum_direction_cosine",
    "raw_contact_classification_sum_difference",
    "invalid_populated_raw_contact_fields",
    "invalid_populated_raw_contacts",
    "raw_contact_buffer_saturated_samples",
    "maximum_raw_contacts_strictly_below_capacity",
    "observed_roll_range",
    "maximum_hinge_frame_translation_residual",
    "maximum_hinge_axis_rotation_residual",
    "native_aperture_state_samples",
)

# An unsupported passage can legitimately have no medium/aperture raw contacts.
# Keep these checks mandatory in the child schema, but apply their strictly
# positive thresholds only when the native raw-contact buffer observed at
# least one populated point in that case.
CONTACT_CONDITIONAL_POSITIVE_CHECKS = frozenset((
    "populated_raw_contact_points",
    "uniquely_mapped_raw_contact_points",
    "nonzero_signed_scalar_raw_contact_points",
    "verified_signed_scalar_times_normal_reaction_points",
))


def check_passes(check):
    measured, threshold = check["measured"], check["threshold"]
    return {
        "<=": measured <= threshold,
        "<": measured < threshold,
        ">=": measured >= threshold,
        ">": measured > threshold,
        "==": measured == threshold,
    }[check["operator"]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-hz", type=int, choices=(240, 480), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def case_table():
    rows = []
    for aperture in (0.0127, 0.01905):
        for amplitude in (5.0, 25.0, 45.0):
            for frequency in (1.0, 2.0):
                for yaw in (0.0, 45.0, 90.0):
                    rows.append({"aperture_width_m": aperture, "amplitude_deg": amplitude,
                                 "frequency_hz": frequency, "roll_deg": 0.0,
                                 "pitch_deg": 90.0, "yaw_deg": yaw,
                                 "initial_clearance_m": 0.001, "kind": "transverse_grid"})
    rows.append({"aperture_width_m": 0.0127, "amplitude_deg": 5.0,
                 "frequency_hz": 1.0, "roll_deg": 0.0, "pitch_deg": 35.0,
                 "yaw_deg": 45.0, "initial_clearance_m": 1.0e-5,
                 "kind": "preserved_oblique_baseline"})
    return rows


def main():
    args = parse_args()
    report = args.report if args.report.is_absolute() else (Path.cwd() / args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(__file__).resolve().parent.parent
    worker = Path(__file__).with_name("run_moving_aperture_crossing_probe.py")
    case_dir = report.parent / f"aperture-release-grid-{args.physics_hz}hz-cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, spec in enumerate(case_table(), 1):
        case_id = f"case-{index:02d}"
        child_report = case_dir / f"{case_id}.json"
        command = [sys.executable, str(worker), "--physics-hz", str(args.physics_hz),
                   "--report", str(child_report), "--device", args.device,
                   "--aperture-width-m", str(spec["aperture_width_m"]),
                   "--drive-amplitude-deg", str(spec["amplitude_deg"]),
                   "--drive-frequency-hz", str(spec["frequency_hz"]),
                   "--roll-deg", str(spec["roll_deg"]),
                   "--pitch-deg", str(spec["pitch_deg"]),
                   "--yaw-deg", str(spec["yaw_deg"]),
                   "--initial-clearance-m", str(spec["initial_clearance_m"])]
        # This pinned AppLauncher no longer registers a --headless CLI option.
        # Its source-supported HEADLESS environment setting is resolved before
        # launch, so keep the child invocation parser-valid and deterministic.
        child_env = os.environ.copy()
        child_env["HEADLESS"] = "1"
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False,
                                   env=child_env)
        row = {"case_id": case_id, **spec, "process_exit_code": completed.returncode,
               "report": child_report.relative_to(workspace).as_posix(),
               "log_tail": completed.stdout[-4000:]}
        if completed.returncode == 0 and child_report.is_file():
            data = json.loads(child_report.read_text(encoding="utf-8"))
            checks_by_name = {check["name"]: check for check in data.get("checks", [])}
            missing_acquisition_checks = [
                name for name in REQUIRED_ACQUISITION_CHECKS if name not in checks_by_name
            ]
            observed_raw_contact = (
                data.get("raw_contact_accounting", {}).get("total", 0) > 0
            )
            failed_acquisition_checks = [
                name for name in REQUIRED_ACQUISITION_CHECKS
                if name in checks_by_name
                and (observed_raw_contact or
                     name not in CONTACT_CONDITIONAL_POSITIVE_CHECKS)
                and not check_passes(checks_by_name[name])
            ]
            row.update({
                "initial_native_pose": data.get("initial_native_pose"),
                "final_native_pose": data.get("final_native_pose"),
                "initial_bridge_qualified": data.get("initial_bridge_qualified"),
                "initial_support_outcome": data.get("initial_support_outcome"),
                "initial_gravity_passage_observed": data.get(
                    "initial_gravity_passage_observed"),
                "physical_outcome": data.get("physical_outcome"),
                "sampled_crossing_observed": data.get("sampled_crossing_observed"),
                "patch_edge_exit_or_side_loss_observed": data.get(
                    "patch_edge_exit_or_side_loss_observed"),
                "rebound_or_resupport_observed": data.get(
                    "rebound_or_resupport_observed"),
                "maximum_fresh_consecutive_support_loss_samples": data.get(
                    "maximum_fresh_consecutive_support_loss_samples"),
                "observed_roll_range_deg": data.get("observed_roll_range_deg"),
                "raw_contact_accounting": data.get("raw_contact_accounting"),
                "raw_force_vector_reconciliation_summary": data.get(
                    "raw_force_vector_reconciliation_summary"),
                "native_contact_processing_setting": data.get(
                    "native_contact_processing_setting"),
                "child_artifacts": data.get("artifacts", []),
                "acquisition_checks_present": not missing_acquisition_checks,
                "acquisition_checks_pass": not (missing_acquisition_checks or
                                                  failed_acquisition_checks),
                "observed_raw_contact": observed_raw_contact,
                "contact_conditional_positive_checks_applied": observed_raw_contact,
                "missing_acquisition_checks": missing_acquisition_checks,
                "failed_acquisition_checks": failed_acquisition_checks,
            })
        results.append(row)

    completed_rows = [r for r in results if r["process_exit_code"] == 0 and
                      r.get("initial_native_pose") is not None and
                      r.get("final_native_pose") is not None]
    integrity_rows = [r for r in completed_rows if
                      r.get("acquisition_checks_present") is True and
                      r.get("acquisition_checks_pass") is True and
                      r.get("raw_contact_accounting", {}).get("invalid_fields") == 0 and
                      r.get("raw_contact_accounting", {}).get("saturated") == 0 and
                      r.get("raw_force_vector_reconciliation_summary", {}).get(
                          "positive_pair_force_samples", 0) ==
                      r.get("raw_force_vector_reconciliation_summary", {}).get(
                          "verified_positive_pair_force_samples", -1)]
    releases = sum(r.get("sampled_crossing_observed") is True for r in completed_rows
                   if r.get("initial_bridge_qualified") is True)
    qualified = sum(r.get("initial_bridge_qualified") is True for r in completed_rows)
    initial_support_outcome_counts = {
        outcome: sum(r.get("initial_support_outcome") == outcome
                     for r in completed_rows)
        for outcome in (
            "initial_gravity_passage",
            "qualified_two_or_more_rib_bridge",
            "one_rib_support",
            "unstable_support",
            "no_support",
        )
    }
    checks = [
        {"name": "predeclared_case_count", "measured": len(results), "operator": "==",
         "threshold": 37, "unit": "cases"},
        {"name": "completed_case_count", "measured": len(completed_rows), "operator": "==",
         "threshold": 37, "unit": "cases"},
        {"name": "data_integrity_case_count", "measured": len(integrity_rows), "operator": "==",
         "threshold": 37, "unit": "cases"},
    ]
    artifacts = [r["report"] for r in completed_rows]
    artifacts.extend(path for r in completed_rows for path in r.get("child_artifacts", []))
    payload = {
        "scope": "frozen 37-case engineering-design exploration only",
        "qualification_claimed": False,
        "coverage": "sampled endpoint classifier; no continuous sweep, full screen, washer retention, robot, or scorer claim",
        "physics_hz": args.physics_hz,
        "case_order": "aperture-major, amplitude-major, frequency-major, yaw-minor; preserved baseline last",
        "physical_result_summary": {"initial_bridge_qualified_cases": qualified,
                                    "sampled_crossings_among_qualified_bridges": releases,
                                    "initial_support_outcome_counts": initial_support_outcome_counts},
        "diagnostics": {
            "initial_bridge_qualified_case_count": qualified,
            "sampled_crossing_case_count_among_qualified_bridges": releases,
        },
        "cases": results,
        "checks": checks,
        "artifacts": artifacts,
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
