#!/usr/bin/env python3
"""Native bounded probe of loaded reset and 20 Hz no-op observations.

This discovery probe intentionally does not claim the frozen 240 Hz stability or
continuous aperture-passage predicates.  It records the state needed to plan
the next physical separation trial without turning sampled endpoints into a
qualification result.
"""
from __future__ import annotations

import argparse, gzip, json, math, os, subprocess, sys, tempfile, traceback
from pathlib import Path


def arg_path(name, default):
    for i, value in enumerate(sys.argv[1:]):
        if value == name and i + 2 <= len(sys.argv) - 1:
            return Path(sys.argv[i + 2])
        if value.startswith(name + "="):
            return Path(value.split("=", 1)[1])
    return Path(default)


def failure_receipt(path, message, detail=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "scope": "failed loaded-reset native preflight; no qualification claim",
        "checks": [{"name": "native_program_completed_without_exception", "measured": 0,
                    "operator": "==", "threshold": 1, "unit": "boolean", "passed": False}],
        "artifacts": [], "execution_exception": {"message": message, "traceback": detail[-20000:]}
    }, indent=2))


if __name__ == "__main__" and os.environ.get("ROBOCURVE_NATIVE_LOADED_CHILD") != "1":
    report = arg_path("--report", "../measurements/loaded-reset-native-probe.json")
    env = os.environ.copy(); env["ROBOCURVE_NATIVE_LOADED_CHILD"] = "1"
    with tempfile.NamedTemporaryFile(prefix="loaded-reset-", suffix=".stderr", delete=False) as stream:
        stderr_path = Path(stream.name)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env, stderr=stream)
    stderr = stderr_path.read_text(errors="replace"); stderr_path.unlink(missing_ok=True)
    if stderr: sys.stderr.write(stderr)
    if completed.returncode and not report.is_file():
        failure_receipt(report, f"native child exited with status {completed.returncode}", stderr)
    raise SystemExit(completed.returncode)


from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, default=Path("../measurements/loaded-reset-native-probe.json"))
parser.add_argument("--screenshot", type=Path, default=Path("../measurements/loaded-reset-native-probe.png"))
parser.add_argument("--seed", type=int, default=8261)
parser.add_argument("--duration", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.enable_cameras = True
launcher = AppLauncher(args); simulation_app = launcher.app

import imageio.v2 as imageio
import numpy as np
import torch
from yam_arena.constants import FINGER_OPEN, READY
from full_screen_scene import (CONTROL_HZ, MEDIA_DENSITY, MEDIA_DIAMETER, MEDIA_LENGTH,
    SCREEN_ROOT_Z, WASHER_BORE, WASHER_DENSITY, WASHER_OD, WASHER_T, build_loaded_environment)


def evaluate(row):
    a, b = float(row["measured"]), float(row["threshold"])
    if not math.isfinite(a) or not math.isfinite(b): return False
    return {"==": a == b, "<=": a <= b, ">=": a >= b, "<": a < b, ">": a > b}[row["operator"]]


def main():
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = ArenaEnvBuilder(build_loaded_environment(), ArenaEnvBuilderCfg(
        num_envs=1, solve_relations=False, device=args.device)).make_registered()
    base = env.unwrapped; env.reset(seed=args.seed)
    dt = float(base.sim.get_physics_dt())
    names = [f"media_{i:02d}" for i in range(8)] + [f"washer_{i:02d}" for i in range(2)]
    bodies = {name: base.scene.rigid_objects[name] for name in names}
    expected_media_mass = MEDIA_DENSITY * math.pi * (MEDIA_DIAMETER/2)**2 * MEDIA_LENGTH
    expected_washer_mass = WASHER_DENSITY * math.pi * ((WASHER_OD/2)**2-(WASHER_BORE/2)**2) * WASHER_T
    native_masses = {}
    for name, body in bodies.items():
        body.update(dt); native_masses[name] = float(body.data.body_mass.torch[0,0].detach().cpu())
    screen = base.scene.rigid_objects["screen"]
    screen.update(dt)
    initial = {name: body.data.root_pos_w.torch[0].detach().cpu().numpy().astype(float).copy() for name, body in bodies.items()}
    action = torch.tensor(np.r_[READY, FINGER_OPEN, READY, FINGER_OPEN], dtype=torch.float32,
                          device=args.device).unsqueeze(0)
    samples = int(math.ceil(args.duration * CONTROL_HZ)); trace = []
    max_speed = 0.0; max_displacement = 0.0; min_z = math.inf
    final_rgb = None
    for sample in range(samples):
        env.step(action)
        row = {"control_sample": sample + 1, "time_s": (sample + 1) / CONTROL_HZ, "objects": {}}
        for name, body in bodies.items():
            body.update(dt)
            pos = body.data.root_pos_w.torch[0].detach().cpu().numpy().astype(float)
            vel = body.data.root_lin_vel_w.torch[0].detach().cpu().numpy().astype(float)
            ang = body.data.root_ang_vel_w.torch[0].detach().cpu().numpy().astype(float)
            speed = float(np.linalg.norm(vel)); disp = float(np.linalg.norm(pos - initial[name]))
            max_speed = max(max_speed, speed); max_displacement = max(max_displacement, disp); min_z = min(min_z, float(pos[2]))
            row["objects"][name] = {"position_w_m": pos.tolist(), "linear_velocity_w_m_s": vel.tolist(),
                                    "angular_velocity_w_rad_s": ang.tolist()}
        screen.update(dt)
        row["screen"] = {
            "root_position_w_m": screen.data.root_pos_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "root_quaternion_w_xyzw": screen.data.root_quat_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "root_linear_velocity_w_m_s": screen.data.root_lin_vel_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "root_angular_velocity_w_rad_s": screen.data.root_ang_vel_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
        }
        trace.append(row)
        cam = base.scene.sensors.get("overview_cam", base.scene.sensors.get("top_cam"))
        rgb = cam.data.output["rgb"].torch[0].detach().cpu().numpy()[...,:3]
        if sample == 0:
            min_finite_fraction = 1.0
            min_in_range_fraction = 1.0
        finite = np.isfinite(rgb)
        min_finite_fraction = min(min_finite_fraction, float(np.mean(finite)))
        min_in_range_fraction = min(min_in_range_fraction, float(np.mean(finite & (rgb >= 0) & (rgb <= 255))))
        final_rgb = np.nan_to_num(rgb, nan=0, posinf=255, neginf=0).clip(0,255).astype(np.uint8)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True); imageio.imwrite(args.screenshot, final_rgb)
    trace_path = args.report.with_name(args.report.stem + "-trace.json.gz")
    with gzip.open(trace_path, "wt") as stream: json.dump(trace, stream, separators=(",", ":"))
    media_error = max(abs(native_masses[n]-expected_media_mass)/expected_media_mass for n in names[:8])
    washer_error = max(abs(native_masses[n]-expected_washer_mass)/expected_washer_mass for n in names[8:])
    final_centers_below = sum(int(trace[-1]["objects"][n]["position_w_m"][2] < SCREEN_ROOT_Z-0.001) for n in names)
    checks = [
      {"name":"native_physics_timestep_240hz","measured":abs(dt-1/240),"operator":"<=","threshold":1e-12,"unit":"s"},
      {"name":"free_rigid_media_body_count","measured":len(names[:8]),"operator":"==","threshold":8,"unit":"bodies"},
      {"name":"free_rigid_washer_body_count","measured":len(names[8:]),"operator":"==","threshold":2,"unit":"bodies"},
      {"name":"authored_media_mass_propagation_readback_relative_error_max","measured":media_error,"operator":"<=","threshold":1e-4,"unit":"fraction"},
      {"name":"authored_washer_mass_propagation_readback_relative_error_max","measured":washer_error,"operator":"<=","threshold":1e-4,"unit":"fraction"},
      {"name":"noop_final_object_centers_below_screen_reference_height_at_20hz_endpoint","measured":final_centers_below,"operator":"==","threshold":0,"unit":"body_centers"},
      {"name":"noop_max_loose_body_translation_from_20hz_observations","measured":max_displacement,"operator":"<=","threshold":0.010,"unit":"m"},
      {"name":"noop_max_loose_body_linear_speed_from_20hz_observations","measured":max_speed,"operator":"<=","threshold":0.25,"unit":"m/s"},
      {"name":"native_rgb_minimum_finite_pixel_fraction_before_sanitization","measured":min_finite_fraction,"operator":"==","threshold":1.0,"unit":"fraction"},
      {"name":"native_rgb_minimum_finite_in_range_pixel_fraction_before_sanitization","measured":min_in_range_fraction,"operator":"==","threshold":1.0,"unit":"fraction"},
      {"name":"rendered_rgb_standard_deviation","measured":float(np.std(final_rgb.astype(np.float32))),"operator":">=","threshold":2.0,"unit":"uint8_level"},
    ]
    for row in checks: row["passed"] = evaluate(row)
    workspace = Path(__file__).resolve().parents[1]
    artifacts = [p.resolve().relative_to(workspace).as_posix() for p in (args.screenshot, trace_path)]
    report = {"scope":"loaded reset and 20 Hz endpoint-observation native preflight only; not 240 Hz stability, grasp, passage, retention, scorer, or full qualification",
      "component_preflight_passed":all(r["passed"] for r in checks), "seed":args.seed,
      "physics_dt_s":dt, "control_hz":CONTROL_HZ, "duration_s":args.duration,
      "native_body_masses_kg":native_masses, "checks":checks, "artifacts":artifacts,
      "mass_readback_scope":"comparison to the explicitly authored analytical mass formulas; verifies propagation only, not cooked-geometry-derived mass, constitutive fidelity, or inertia qualification",
      "endpoint_height_scope":"final body-center height only; not passage history, whole-collider crossing, aperture identity, swept clearance, or invalid-crossing detection",
      "sampling_scope":"loose-body translation and linear speed are observed after 20 Hz control steps; they do not establish the frozen 240 Hz <=0.01 m/s stability criterion",
      "unqualified_observables":["240 Hz loaded-scene stability","genuine aperture passage","washer retention under agitation","loaded bilateral grasp","receiver containment","frozen aggregate contract"]}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2))
    env.close(); simulation_app.close()


try:
    main()
except BaseException as exc:
    failure_receipt(args.report, f"{type(exc).__name__}: {exc}", traceback.format_exc())
    try: simulation_app.close()
    except BaseException: pass
    raise
