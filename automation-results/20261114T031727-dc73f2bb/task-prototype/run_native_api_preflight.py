#!/usr/bin/env python3
"""Bounded native PhysX callback/contact probe for the DRG80 task.

This is deliberately not an environment qualification.  It creates one falling
rigid cube and a static floor, samples the pinned PhysX tensor contact API from a
POST_PHYSICS_STEP callback, and verifies callback removal across reset.  The
trusted controller must run it with the pinned IsaacLab-Arena Python.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
from typing import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import IsaacEvents, PhysxManager


PHYSICS_DT = 1.0 / 240.0
DRIVEN_STEPS = 180
POST_DEREGISTER_STEPS = 8


def _torch(array) -> torch.Tensor:
    """Copy a Warp-backed PhysX tensor to finite CPU storage immediately."""

    return wp.to_torch(array).detach().cpu().clone()


def _finite_fraction(*values: torch.Tensor) -> float:
    elements = sum(value.numel() for value in values)
    finite = sum(int(torch.isfinite(value).sum().item()) for value in values)
    return float(finite / elements) if elements else 0.0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(name: str, measured: float, operator: str, threshold: float, unit: str) -> dict:
    if not math.isfinite(float(measured)):
        raise RuntimeError(f"non-finite preflight measurement: {name}={measured}")
    comparisons = {
        "==": lambda value, limit: value == limit,
        ">": lambda value, limit: value > limit,
        ">=": lambda value, limit: value >= limit,
        "<": lambda value, limit: value < limit,
        "<=": lambda value, limit: value <= limit,
    }
    if operator not in comparisons:
        raise ValueError(f"unsupported preflight operator: {operator}")
    passed = bool(comparisons[operator](float(measured), float(threshold)))
    return {
        "name": name,
        "measured": float(measured),
        "operator": operator,
        "threshold": float(threshold),
        "unit": unit,
        "passed": passed,
    }


def _spawn_coupon() -> None:
    floor = sim_utils.CuboidCfg(
        size=(1.0, 1.0, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
    )
    floor.func("/World/Floor", floor, translation=(0.0, 0.0, -0.02))

    cube = sim_utils.CuboidCfg(
        size=(0.10, 0.10, 0.10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
    )
    cube.func("/World/Coupon", cube, translation=(0.0, 0.0, 0.20))


def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "measurements"
    report_path = output_dir / "native-api-preflight.json"
    trace_path = output_dir / "native-api-preflight-trace.json"

    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    _spawn_coupon()
    sim.reset()

    simulation_view = PhysxManager.get_physics_sim_view()
    if simulation_view is None:
        raise RuntimeError("PhysxManager did not expose a simulation view after reset")
    simulation_view.set_subspace_roots("/")
    body_view = simulation_view.create_rigid_body_view("/World/Coupon")
    contact_view = simulation_view.create_rigid_contact_view(
        "/World/Coupon", max_contact_data_count=64
    )
    if body_view.count != 1 or contact_view.sensor_count != 1:
        raise RuntimeError(
            f"unexpected coupon views: bodies={body_view.count}, sensors={contact_view.sensor_count}"
        )

    sequence: list[tuple[str, int, float]] = []
    pre_body_samples: dict[int, dict] = {}
    post_samples: list[dict] = []
    current_step = {"value": -1}
    callback_failures: list[dict] = []
    peak = {
        "force_n": 0.0,
        "normal_impulse_magnitude_sum_ns": 0.0,
        "points": [],
        "normals": [],
        "separations_m": [],
        "cube_transform": [],
        "cube_velocity": [],
        "point_normal_velocity_m_s": 0.0,
    }
    tuple_lengths: list[int] = []

    def guard_callback(name: str, callback: Callable[[float], None]) -> Callable[[float], None]:
        """Surface callback failures that Omniverse event dispatch may swallow.

        The pinned PhysxManager Isaac-event wrapper does not catch callback
        exceptions itself.  Store the original exception before re-raising it
        so the immediately following raise_callback_exception_if_any() cannot
        miss a failure, including one raised after a sample was appended.
        Retain a separate local latch so a future backend behavior change also
        makes the report fail closed.
        """

        @functools.wraps(callback)
        def guarded(dt: float) -> None:
            try:
                callback(dt)
            except Exception as exc:
                callback_failures.append(
                    {
                        "callback": name,
                        "step": int(current_step["value"]),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                PhysxManager.store_callback_exception(exc)
                raise

        return guarded

    def pre_callback(dt: float) -> None:
        sequence.append(("pre", current_step["value"], float(dt)))
        transforms = _torch(body_view.get_transforms())
        velocities = _torch(body_view.get_velocities())
        pre_body_samples[current_step["value"]] = {
            "cube_z_m": float(transforms[0, 2].item()),
            "cube_vertical_velocity_m_s": float(velocities[0, 2].item()),
        }

    def post_callback(dt: float) -> None:
        sequence.append(("post", current_step["value"], float(dt)))
        contact_data = contact_view.get_contact_data(dt=float(dt))
        tuple_lengths.append(len(contact_data))
        if len(contact_data) != 6:
            raise RuntimeError(f"unexpected get_contact_data tuple length: {len(contact_data)}")
        forces, points, normals, separations, counts, starts = map(_torch, contact_data)

        # The backend reuses its buffers, so copy the force-valued result before
        # requesting the scalar per-contact normal impulse magnitudes with a
        # one-second divisor.  This is not a resolved impulse vector.
        impulse_data = contact_view.get_contact_data(dt=1.0)
        if len(impulse_data) != 6:
            raise RuntimeError(f"unexpected impulse get_contact_data tuple length: {len(impulse_data)}")
        impulses = _torch(impulse_data[0])
        transforms = _torch(body_view.get_transforms())
        velocities = _torch(body_view.get_velocities())
        count = int(counts.to(torch.int64).sum().item())
        start = int(starts.flatten()[0].item()) if count else 0
        active_slice = slice(start, start + count)
        active_forces = forces.flatten()[active_slice]
        active_impulses = impulses.flatten()[active_slice]
        active_points = points.reshape(-1, 3)[active_slice]
        active_normals = normals.reshape(-1, 3)[active_slice]
        active_separations = separations.flatten()[active_slice]
        force_sum = float(active_forces.sum().item()) if count else 0.0
        impulse_sum = float(active_impulses.sum().item()) if count else 0.0

        post_samples.append(
            {
                "step": current_step["value"],
                "dt_s": float(dt),
                "contact_count": count,
                "force_sum_n": force_sum,
                "normal_impulse_magnitude_sum_ns": impulse_sum,
                "cube_z_m": float(transforms[0, 2].item()),
                "cube_vertical_velocity_m_s": float(velocities[0, 2].item()),
                "cube_linear_speed_m_s": float(torch.linalg.vector_norm(velocities[0, :3]).item()),
            }
        )
        if force_sum > peak["force_n"]:
            point_normal_velocity = 0.0
            if count:
                position = transforms[0, :3]
                linear = velocities[0, :3]
                angular = velocities[0, 3:6]
                point_velocity = linear[None, :] + torch.linalg.cross(
                    angular.expand_as(active_points), active_points - position[None, :]
                )
                point_normal_velocity = float(
                    torch.mean(torch.sum(point_velocity * active_normals, dim=1)).item()
                )
            peak.update(
                {
                    "force_n": force_sum,
                    "normal_impulse_magnitude_sum_ns": impulse_sum,
                    "points": active_points.tolist(),
                    "normals": active_normals.tolist(),
                    "separations_m": active_separations.tolist(),
                    "cube_transform": transforms[0].tolist(),
                    "cube_velocity": velocities[0].tolist(),
                    "point_normal_velocity_m_s": point_normal_velocity,
                }
            )

    pre_handle = PhysxManager.register_callback(
        guard_callback("pre", pre_callback),
        IsaacEvents.PRE_PHYSICS_STEP,
        name="drg80_preflight_pre",
        wrap_weak_ref=False,
    )
    post_handle = PhysxManager.register_callback(
        guard_callback("post", post_callback),
        IsaacEvents.POST_PHYSICS_STEP,
        name="drg80_preflight_post",
        wrap_weak_ref=False,
    )

    initial_z = float(_torch(body_view.get_transforms())[0, 2].item())
    for step in range(DRIVEN_STEPS):
        current_step["value"] = step
        sim.step(render=False)
        PhysxManager.raise_callback_exception_if_any()
        external_transforms = _torch(body_view.get_transforms())
        external_velocities = _torch(body_view.get_velocities())
        post_samples[-1]["external_cube_z_m"] = float(external_transforms[0, 2].item())
        post_samples[-1]["external_cube_vertical_velocity_m_s"] = float(
            external_velocities[0, 2].item()
        )
    final_z = float(_torch(body_view.get_transforms())[0, 2].item())

    pre_count = sum(kind == "pre" for kind, _, _ in sequence)
    post_count = sum(kind == "post" for kind, _, _ in sequence)
    exact_ordered_steps = 0
    malformed_step_events: dict[int, list[str]] = {}
    for step in range(DRIVEN_STEPS):
        events = [kind for kind, sample_step, _ in sequence if sample_step == step]
        if events == ["pre", "post"]:
            exact_ordered_steps += 1
        else:
            malformed_step_events[step] = events
    callback_dts = [dt for _, _, dt in sequence]
    dt_error = max(abs(dt - PHYSICS_DT) for dt in callback_dts)

    pre_handle.deregister()
    post_handle.deregister()
    sequence_count_at_deregister = len(sequence)
    sim.reset()
    for _ in range(POST_DEREGISTER_STEPS):
        sim.step(render=False)
        PhysxManager.raise_callback_exception_if_any()
    callbacks_after_deregister = len(sequence) - sequence_count_at_deregister

    if len(post_samples) != DRIVEN_STEPS or len(pre_body_samples) != DRIVEN_STEPS:
        raise RuntimeError(
            "callback sample cardinality mismatch: "
            f"pre={len(pre_body_samples)}, post={len(post_samples)}, expected={DRIVEN_STEPS}"
        )
    post_external_z_error = max(
        abs(sample["cube_z_m"] - sample["external_cube_z_m"]) for sample in post_samples
    )
    post_external_velocity_error = max(
        abs(
            sample["cube_vertical_velocity_m_s"]
            - sample["external_cube_vertical_velocity_m_s"]
        )
        for sample in post_samples
    )
    integrated_transition_steps = sum(
        (
            abs(post_samples[step]["cube_z_m"] - pre_body_samples[step]["cube_z_m"])
            > 1.0e-8
            or abs(
                post_samples[step]["cube_vertical_velocity_m_s"]
                - pre_body_samples[step]["cube_vertical_velocity_m_s"]
            )
            > 1.0e-8
        )
        for step in range(DRIVEN_STEPS)
    )

    active_points = torch.tensor(peak["points"], dtype=torch.float64).reshape(-1, 3)
    active_normals = torch.tensor(peak["normals"], dtype=torch.float64).reshape(-1, 3)
    active_separations = torch.tensor(peak["separations_m"], dtype=torch.float64)
    if active_points.numel() == 0:
        raise RuntimeError("falling coupon produced no native contact samples")
    normal_unit_error = float(torch.max(torch.abs(torch.linalg.vector_norm(active_normals, dim=1) - 1.0)).item())
    contact_finite_fraction = _finite_fraction(active_points, active_normals, active_separations)
    force_impulse_scale_error = abs(
        peak["force_n"] * PHYSICS_DT - peak["normal_impulse_magnitude_sum_ns"]
    )

    trace = {
        "probe": "falling rigid cube on static floor",
        "device": str(args_cli.device),
        "physics_dt_s": PHYSICS_DT,
        "driven_steps": DRIVEN_STEPS,
        "initial_cube_z_m": initial_z,
        "final_cube_z_m": final_z,
        "callback_sequence": sequence,
        "malformed_step_events": malformed_step_events,
        "pre_body_samples": pre_body_samples,
        "post_samples": post_samples,
        "peak_contact": peak,
        "contact_tuple_lengths": tuple_lengths,
        "callback_failures": callback_failures,
        "contact_tuple_semantics": [
            "scalar_normal_force_or_normal_impulse_magnitude (not a vector)",
            "point_world",
            "normal_world",
            "signed_separation",
            "count",
            "start_index",
        ],
    }
    _write_json(trace_path, trace)

    checks = [
        _check("native_physics_dt", float(sim.get_physics_dt()), "==", PHYSICS_DT, "s"),
        _check("pre_callback_count", pre_count, "==", DRIVEN_STEPS, "callbacks"),
        _check("post_callback_count", post_count, "==", DRIVEN_STEPS, "callbacks"),
        _check("callback_failure_count", len(callback_failures), "==", 0, "exceptions"),
        _check(
            "exactly_one_pre_then_one_post_fraction",
            exact_ordered_steps / DRIVEN_STEPS,
            "==",
            1.0,
            "fraction of driven steps",
        ),
        _check("callback_dt_max_error", dt_error, "<=", 1.0e-9, "s"),
        _check("callbacks_after_deregister_and_reset", callbacks_after_deregister, "==", 0, "callbacks"),
        _check("contact_data_tuple_length", min(tuple_lengths), "==", 6, "arrays"),
        _check(
            "contacting_post_sample_count",
            sum(sample["contact_count"] > 0 for sample in post_samples),
            ">=",
            1,
            "post-physics samples",
        ),
        _check("peak_manifold_normal_force", peak["force_n"], ">", 0, "N"),
        _check(
            "peak_manifold_normal_impulse_magnitude_sum",
            peak["normal_impulse_magnitude_sum_ns"],
            ">",
            0,
            "N*s",
        ),
        _check("force_impulse_dt_scaling_error", force_impulse_scale_error, "<=", 1.0e-5, "N*s"),
        _check("contact_point_normal_separation_finite_fraction", contact_finite_fraction, "==", 1, "fraction"),
        _check("contact_normal_unit_length_error_max", normal_unit_error, "<=", 1.0e-4, "dimensionless"),
        _check("body_pose_vertical_change", abs(final_z - initial_z), ">=", 0.10, "m"),
        _check(
            "post_callback_external_pose_match_max_error",
            post_external_z_error,
            "<=",
            1.0e-7,
            "m",
        ),
        _check(
            "post_callback_external_vertical_velocity_match_max_error",
            post_external_velocity_error,
            "<=",
            1.0e-7,
            "m/s",
        ),
        _check(
            "pre_to_post_integrated_transition_step_count",
            integrated_transition_steps,
            ">=",
            1,
            "driven steps",
        ),
    ]
    overall_pass = all(check["passed"] for check in checks)
    _write_json(
        report_path,
        {
            "checks": checks,
            "artifacts": [str(trace_path.relative_to(output_dir.parent))],
            "overall_pass": overall_pass,
        },
    )
    if not overall_pass:
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"native API preflight failed checks: {failed}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
