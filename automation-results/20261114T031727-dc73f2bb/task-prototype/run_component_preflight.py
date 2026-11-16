#!/usr/bin/env python3
"""Native DRG80 GPU-constraint isolation probe (not qualification).

The previous host run established that PhysX rack-and-pinion and gear constraints
cannot be added to a direct-GPU scene.  This bounded probe deactivates exactly
those three unsupported constraints in the composed in-memory stage, retains the
actual fixture rigid bodies and its GPU-compatible fixed/revolute/prismatic/D6
joints, and measures whether that remainder loads, settles, and responds to a
reversible physical lever torque.  It also verifies that no lever motion leaks to
the now deliberately uncoupled piston or jaws.  This does not validate the final
mechanism; its result decides whether the production asset can be repaired by
replacing only the three unsupported couplers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import uuid

from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
DEFAULT_REPORT = WORKSPACE_DIR / "measurements/component-preflight.json"
DEFAULT_TRACE = WORKSPACE_DIR / "measurements/component-preflight-trace.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(DEFAULT_REPORT))
parser.add_argument("--trace", default=str(DEFAULT_TRACE))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Invalidate prior products before Kit startup.  In particular, a launcher or
# import failure must not leave a stale report that looks like this invocation.
for stale_path in (Path(args_cli.output).resolve(), Path(args_cli.trace).resolve()):
    stale_path.unlink(missing_ok=True)

TRIAL_ID = f"drg80-component-{uuid.uuid4()}"
STARTED_AT = datetime.now(timezone.utc).isoformat()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import IsaacEvents, PhysxManager


ROOT = SCRIPT_DIR
PHYSICS_DT = 1.0 / 240.0
FIXTURE_TRANSLATION = (0.36, 0.0, 0.75)
BODY_NAMES = (
    "frame",
    "left_jaw",
    "right_jaw",
    "piston",
    "adjustment_screw",
    "nut73",
    "nut74",
    "lever",
)
ASSET_PATHS = (
    ROOT / "assets/fixture/drg80_fixture.usda",
    ROOT / "assets/fixture/parts/frame.usda",
    ROOT / "assets/fixture/parts/left_jaw.usda",
    ROOT / "assets/fixture/parts/lever.usda",
    ROOT / "assets/fixture/parts/piston.usda",
    ROOT / "assets/fixture/parts/right_jaw.usda",
    ROOT / "assets/tools/socket_adjuster.usda",
    ROOT / "assets/tools/thin_nut.usda",
)
UNSUPPORTED_COUPLER_PATHS = (
    "/World/Fixture/lever_to_piston",
    "/World/Fixture/jaw_pair",
    "/World/Fixture/piston_to_jaw",
)


def _cpu(array) -> torch.Tensor:
    return wp.to_torch(array).detach().cpu().clone()


def _quat_multiply_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    return torch.cat((aw * bv + bw * av + torch.linalg.cross(av, bv), (aw * bw - torch.dot(av, bv)).view(1)))


def _quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[:3], q[3:].clone()))


def _rotate_vector_xyzw(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((vector, torch.zeros(1, dtype=q.dtype)))
    return _quat_multiply_xyzw(_quat_multiply_xyzw(q, pure), _quat_conjugate_xyzw(q))[:3]


def _signed_axis_delta_deg(a: torch.Tensor, b: torch.Tensor, local_axis: torch.Tensor) -> float:
    """Signed shortest twist from orientation a to b about a's local axis."""
    a = a / torch.linalg.vector_norm(a)
    b = b / torch.linalg.vector_norm(b)
    relative = _quat_multiply_xyzw(b, _quat_conjugate_xyzw(a))
    if relative[3] < 0.0:
        relative = -relative
    world_axis = _rotate_vector_xyzw(a, local_axis)
    signed_sine = float(torch.dot(relative[:3], world_axis).item())
    signed_angle = 2.0 * math.atan2(signed_sine, float(relative[3].item()))
    return math.degrees(math.atan2(math.sin(signed_angle), math.cos(signed_angle)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(name: str, measured: float, operator: str, threshold: float, unit: str) -> dict:
    measured = float(measured)
    threshold = float(threshold)
    if not math.isfinite(measured):
        raise RuntimeError(f"non-finite component measurement: {name}={measured}")
    comparisons = {
        "==": lambda x, y: x == y,
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
        ">": lambda x, y: x > y,
        "<": lambda x, y: x < y,
    }
    passed = bool(comparisons[operator](measured, threshold))
    return {
        "name": name,
        "measured": measured,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "passed": passed,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _spawn_fixture() -> None:
    table_cfg = sim_utils.CuboidCfg(
        size=(0.8, 1.2, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.25, 0.18)),
    )
    table_cfg.func("/World/Table", table_cfg, translation=(0.35, 0.0, 0.73))
    fixture_cfg = sim_utils.UsdFileCfg(usd_path=str(ROOT / "assets/fixture/drg80_fixture.usda"))
    fixture_cfg.func("/World/Fixture", fixture_cfg, translation=FIXTURE_TRANSLATION)


def main() -> None:
    missing_assets = [str(path) for path in ASSET_PATHS if not path.is_file()]
    if missing_assets:
        raise RuntimeError(f"component asset files missing: {missing_assets}")
    source_hash = _sha256(Path(__file__).resolve())
    asset_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in ASSET_PATHS}

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=12, device=args_cli.device)
    )
    _spawn_fixture()
    deactivated_couplers = 0
    for prim_path in UNSUPPORTED_COUPLER_PATHS:
        prim = sim.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"expected composed coupler prim is missing: {prim_path}")
        prim.SetActive(False)
        if not prim.IsActive():
            deactivated_couplers += 1
    if deactivated_couplers != len(UNSUPPORTED_COUPLER_PATHS):
        raise RuntimeError(
            f"failed to deactivate all unsupported couplers: {deactivated_couplers}/"
            f"{len(UNSUPPORTED_COUPLER_PATHS)}"
        )
    sim.reset()

    physics_view = PhysxManager.get_physics_sim_view()
    if physics_view is None:
        raise RuntimeError("native PhysX simulation view is unavailable after reset")
    physics_view.set_subspace_roots("/")
    views = {
        name: physics_view.create_rigid_body_view([f"/World/Fixture/{name}"])
        for name in BODY_NAMES
    }
    bad_counts = {name: int(view.count) for name, view in views.items() if int(view.count) != 1}
    if bad_counts:
        raise RuntimeError(f"fixture rigid-body view count mismatch: {bad_counts}")

    samples: list[dict] = []
    callback_errors: list[str] = []
    driven_step = {"index": -1, "phase": "startup"}

    def post_step(dt: float) -> None:
        try:
            bodies = {}
            for name, view in views.items():
                transform = _cpu(view.get_transforms())[0]
                velocity = _cpu(view.get_velocities())[0]
                bodies[name] = {
                    "position_m": transform[:3].tolist(),
                    "quaternion_xyzw": transform[3:7].tolist(),
                    "linear_velocity_m_s": velocity[:3].tolist(),
                    "angular_velocity_rad_s": velocity[3:6].tolist(),
                }
            samples.append(
                {
                    "step": int(driven_step["index"]),
                    "phase": str(driven_step["phase"]),
                    "dt_s": float(dt),
                    "bodies": bodies,
                }
            )
        except Exception as exc:
            callback_errors.append(f"{type(exc).__name__}: {exc}")
            PhysxManager.store_callback_exception(exc)
            raise

    callback_handle = PhysxManager.register_callback(
        post_step,
        IsaacEvents.POST_PHYSICS_STEP,
        name="drg80_component_preflight_post",
        wrap_weak_ref=False,
    )

    # The pinned tensor frontend documents optional force/position buffers, but
    # indices are not optional: as_contiguous_uint32(None) dereferences
    # ``None.dtype`` before reaching PhysX.  Select the probe's sole lever body
    # explicitly.  This also makes the target immutable if the view API later
    # changes its handling of an omitted selection.
    lever_indices = wp.array([0], dtype=wp.uint32, device=args_cli.device)

    def drive_phase(name: str, steps: int, lever_torque_y_nm: float) -> None:
        torque = torch.tensor([[0.0, lever_torque_y_nm, 0.0]], dtype=torch.float32, device=args_cli.device)
        torque_wp = wp.from_torch(torque.contiguous(), dtype=wp.float32)
        for _ in range(steps):
            driven_step["index"] += 1
            driven_step["phase"] = name
            views["lever"].apply_forces_and_torques_at_position(
                force_data=None,
                torque_data=torque_wp,
                position_data=None,
                indices=lever_indices,
                is_global=True,
            )
            sim.step(render=False)
            PhysxManager.raise_callback_exception_if_any()

    # Two seconds untouched, then reversible guarded lever excitation.  These
    # are physical wrench inputs; no body pose, velocity, or joint state is set.
    # Piston and jaw response is expected to remain negligible because this
    # probe intentionally deactivated only their unsupported coupling joints.
    drive_phase("settle", 480, 0.0)
    baseline = samples[-1]
    drive_phase("positive_lever_torque", 240, 0.030)
    positive = samples[-1]
    drive_phase("negative_lever_torque", 480, -0.030)
    negative = samples[-1]
    drive_phase("released", 120, 0.0)
    final = samples[-1]

    callback_handle.deregister()
    sample_count_at_deregister = len(samples)
    for _ in range(4):
        sim.step(render=False)
    callbacks_after_deregister = len(samples) - sample_count_at_deregister

    def position(sample: dict, name: str) -> torch.Tensor:
        return torch.tensor(sample["bodies"][name]["position_m"], dtype=torch.float64)

    def quaternion(sample: dict, name: str) -> torch.Tensor:
        return torch.tensor(sample["bodies"][name]["quaternion_xyzw"], dtype=torch.float64)

    def speed(sample: dict, name: str, angular: bool) -> float:
        key = "angular_velocity_rad_s" if angular else "linear_velocity_m_s"
        return float(torch.linalg.vector_norm(torch.tensor(sample["bodies"][name][key])).item())

    # Exactly the final fresh second of the two-second untouched settle.
    settle_tail = [sample for sample in samples if sample["phase"] == "settle"][-240:]
    frame_origin = position(settle_tail[0], "frame")
    frame_translation_mm = max(
        1000.0 * float(torch.linalg.vector_norm(position(sample, "frame") - frame_origin).item())
        for sample in settle_tail
    )
    dynamic = tuple(name for name in BODY_NAMES if name != "frame")
    settle_linear_speed = max(speed(sample, name, False) for sample in settle_tail for name in dynamic)
    settle_angular_speed = max(speed(sample, name, True) for sample in settle_tail for name in dynamic)

    hinge_axis = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float64)
    piston_axis = _rotate_vector_xyzw(
        quaternion(baseline, "frame"), torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)
    )

    def signed_angle(start: dict, end: dict, name: str) -> float:
        return _signed_axis_delta_deg(quaternion(start, name), quaternion(end, name), hinge_axis)

    def signed_piston_mm(start: dict, end: dict) -> float:
        displacement = position(end, "piston") - position(start, "piston")
        return 1000.0 * float(torch.dot(displacement, piston_axis).item())

    lever_positive_deg = signed_angle(baseline, positive, "lever")
    piston_positive_mm = signed_piston_mm(baseline, positive)
    left_positive_deg = signed_angle(baseline, positive, "left_jaw")
    right_positive_deg = signed_angle(baseline, positive, "right_jaw")
    lever_reversal_deg = signed_angle(positive, negative, "lever")
    piston_reversal_mm = signed_piston_mm(positive, negative)
    left_reversal_deg = signed_angle(positive, negative, "left_jaw")
    right_reversal_deg = signed_angle(positive, negative, "right_jaw")

    # Use every fresh post-physics sample in each driven phase.  Endpoints alone
    # can miss a transient excursion that returns before the phase completes.
    # Isolation is measured against the single settled baseline across both
    # driven phases.  Resetting the reference at torque reversal could hide
    # cumulative same-direction drift split across the positive and negative
    # phases.  Phase-relative endpoint values above remain useful diagnostics,
    # but they are not used to pass the isolation claim.
    positive_samples = [sample for sample in samples if sample["phase"] == "positive_lever_torque"]
    negative_samples = [sample for sample in samples if sample["phase"] == "negative_lever_torque"]
    if len(positive_samples) != 240 or len(negative_samples) != 480:
        raise RuntimeError(
            "driven-phase post-step sample count mismatch: "
            f"positive={len(positive_samples)}, negative={len(negative_samples)}"
        )
    driven_samples = (*positive_samples, *negative_samples)

    lever_reversal_product = lever_positive_deg * lever_reversal_deg
    uncoupled_piston_motion_mm = max(
        abs(signed_piston_mm(baseline, sample))
        for sample in driven_samples
    )
    uncoupled_jaw_motion_deg = max(
        abs(signed_angle(baseline, sample, name))
        for sample in driven_samples
        for name in ("left_jaw", "right_jaw")
    )
    jaw_anchor_translation_mm = max(
        1000.0 * float(torch.linalg.vector_norm(position(sample, name) - position(baseline, name)).item())
        for sample in driven_samples
        for name in ("left_jaw", "right_jaw")
    )
    finite_values = []
    for sample in samples:
        for body in sample["bodies"].values():
            for key in (
                "position_m",
                "quaternion_xyzw",
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
            ):
                finite_values.extend(body[key])
    finite_fraction = sum(math.isfinite(float(value)) for value in finite_values) / len(finite_values)
    dt_error = max(abs(float(sample["dt_s"]) - PHYSICS_DT) for sample in samples)

    checks = [
        _check(
            "unsupported_gpu_coupler_deactivation_count",
            deactivated_couplers,
            "==",
            len(UNSUPPORTED_COUPLER_PATHS),
            "composed prims",
        ),
        _check(
            "fixture_native_rigid_body_count",
            sum(int(view.count) for view in views.values()),
            "==",
            len(BODY_NAMES),
            "bodies",
        ),
        _check("post_callback_failure_count", len(callback_errors), "==", 0, "exceptions"),
        _check("post_callback_sample_count", len(samples), "==", 1320, "physics substeps"),
        _check("post_callback_dt_max_error", dt_error, "<=", 1.0e-9, "s"),
        _check("callbacks_after_deregister", callbacks_after_deregister, "==", 0, "callbacks"),
        _check("native_state_finite_fraction", finite_fraction, "==", 1.0, "fraction"),
        _check("untouched_frame_translation_max", frame_translation_mm, "<=", 1.0, "mm over final settle second"),
        _check("untouched_dynamic_linear_speed_max", settle_linear_speed, "<=", 0.02, "m/s"),
        _check("untouched_dynamic_angular_speed_max", settle_angular_speed, "<=", 0.10, "rad/s"),
        _check("lever_signed_rotation_magnitude_under_native_torque", abs(lever_positive_deg), ">=", 5.0, "degree"),
        _check("lever_reversal_rotation_magnitude_under_native_torque", abs(lever_reversal_deg), ">=", 5.0, "degree"),
        _check("lever_reversal_direction_product", lever_reversal_product, "<", 0.0, "degree^2"),
        _check("uncoupled_piston_motion_max", uncoupled_piston_motion_mm, "<=", 0.10, "mm"),
        _check("uncoupled_jaw_rotation_max", uncoupled_jaw_motion_deg, "<=", 0.10, "degree"),
        _check("jaw_hinge_anchor_translation_max", jaw_anchor_translation_mm, "<=", 1.0, "mm"),
    ]

    trace_path = Path(args_cli.trace).resolve()
    _write_json(
        trace_path,
        {
            "native_trial_id": TRIAL_ID,
            "started_at_utc": STARTED_AT,
            "source_sha256": source_hash,
            "asset_sha256": asset_hashes,
            "physics_dt_s": PHYSICS_DT,
            "fixture_translation_m": list(FIXTURE_TRANSLATION),
            "deactivated_gpu_incompatible_couplers": list(UNSUPPORTED_COUPLER_PATHS),
            "applied_lever_torque_nm": {"positive": 0.030, "negative": -0.030},
            "signed_endpoint_motion": {
                "positive": {
                    "lever_deg": lever_positive_deg,
                    "piston_mm": piston_positive_mm,
                    "left_jaw_deg": left_positive_deg,
                    "right_jaw_deg": right_positive_deg,
                },
                "reversal": {
                    "lever_deg": lever_reversal_deg,
                    "piston_mm": piston_reversal_mm,
                    "left_jaw_deg": left_reversal_deg,
                    "right_jaw_deg": right_reversal_deg,
                },
            },
            "samples": samples,
        },
    )
    report = {
        "native_trial_id": TRIAL_ID,
        "started_at_utc": STARTED_AT,
        "source_sha256": source_hash,
        "asset_sha256": asset_hashes,
        "checks": checks,
        "artifacts": ["measurements/component-preflight-trace.json"],
        "overall_pass": all(check["passed"] for check in checks),
    }
    _write_json(Path(args_cli.output).resolve(), report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # A native Python-side failure still produces a fresh, trial-linked
        # diagnostic rather than leaving a stale report from an earlier run.
        # Capture the complete traceback while the exception is active and
        # before SimulationApp shutdown can replace or obscure its context.
        error_traceback = traceback.format_exc()
        print(error_traceback, file=sys.stderr, flush=True)
        failure_report = {
            "native_trial_id": TRIAL_ID,
            "started_at_utc": STARTED_AT,
            "source_sha256": _sha256(Path(__file__).resolve()),
            "checks": [
                _check("native_probe_completed", 0.0, "==", 1.0, "boolean"),
            ],
            "artifacts": [],
            "overall_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": error_traceback,
        }
        _write_json(Path(args_cli.output).resolve(), failure_report)
        raise
    finally:
        simulation_app.close()
