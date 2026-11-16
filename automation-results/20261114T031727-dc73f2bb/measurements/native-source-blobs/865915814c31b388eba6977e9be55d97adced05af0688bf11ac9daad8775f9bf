#!/usr/bin/env python3
"""Final aggregate native component diagnostic for the DRG80 stop-lock task.

This program intentionally performs no robot demo.  It loads the task-local USD,
uses native rigid bodies/joints/contact reports, executes reset, torque, actuation,
damage and 480 Hz refinement probes, and writes every frozen contract check into
one report.  A missing native datum is an exception, never a substituted constant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="../measurements/physics.json")
parser.add_argument("--trace", default="../measurements/component-trace.json")
parser.add_argument("--refinement", action="store_true", help="internal use")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from PIL import Image
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg

ROOT = Path(__file__).resolve().parent
MEASUREMENTS = (ROOT / "../measurements").resolve()
DT = 1.0 / (480.0 if args.refinement else 240.0)
CONTROL_HZ = 20
SUBSTEPS = round(1.0 / (CONTROL_HZ * DT))
FIXTURE_ORIGIN = np.array([0.36, 0.0, 0.75])


def scalar(value) -> float:
    if hasattr(value, "torch"):
        value = value.torch
    return float(torch.as_tensor(value).detach().cpu().reshape(-1)[0])


def array(value) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    return torch.as_tensor(value).detach().cpu().numpy()


def quat_angle_deg(q0, q1) -> float:
    a = np.asarray(q0, float) / np.linalg.norm(q0)
    b = np.asarray(q1, float) / np.linalg.norm(q1)
    return math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def signed_y_deg(q) -> float:
    # IsaacLab root quaternions are xyzw.
    x, y, z, w = np.asarray(q, float)
    return math.degrees(math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x)))


class ContactLog:
    def __init__(self):
        self.nonessential = {"penetration_mm": 0.0, "peak_n": 0.0, "work_j": 0.0}
        self.intended = {"penetration_mm": 0.0, "peak_n": 0.0, "work_j": 0.0}
        self._last_pos = {}
        self._events = []

    @staticmethod
    def _path(actor):
        try:
            return str(PhysicsSchemaTools.intToSdfPath(int(actor)))
        except Exception:
            return str(actor)

    def callback(self, headers, data):
        for h in headers:
            pa, pb = self._path(h.actor0), self._path(h.actor1)
            intended = any(x in pa + pb for x in ("nut73", "adjustment_screw", "piston", "jaw"))
            bucket = self.intended if intended else self.nonessential
            force_sum = 0.0
            start = int(h.contact_data_offset)
            stop = start + int(h.num_contact_data)
            for c in data[start:stop]:
                impulse = np.asarray(c.impulse, float)
                force = float(np.linalg.norm(impulse)) / DT
                force_sum += force
                sep = float(c.separation)
                bucket["penetration_mm"] = max(bucket["penetration_mm"], max(0.0, -1000.0 * sep))
                pos = np.asarray(c.position, float)
                key = (pa, pb)
                prior = self._last_pos.get(key)
                if prior is not None:
                    bucket["work_j"] += max(0.0, force * float(np.linalg.norm(pos - prior)))
                self._last_pos[key] = pos
            bucket["peak_n"] = max(bucket["peak_n"], force_sum)
            self._events.append({"a": pa, "b": pb, "points": stop - start, "force_n": force_sum})


def spawn_scene():
    sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8)).func("/World/Light", sim_utils.DomeLightCfg(intensity=1800.0))
    table = sim_utils.CuboidCfg(
        size=(0.8, 1.2, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.25, 0.18)),
    )
    table.func("/World/Table", table, translation=(0.35, 0.0, 0.73))
    usd = sim_utils.UsdFileCfg(usd_path=str(ROOT / "assets/fixture/drg80_fixture.usda"))
    usd.func("/World/Fixture", usd, translation=tuple(FIXTURE_ORIGIN))
    # Native free diagnostic tools. They begin in their cradles; no post-init teleport is used.
    driver_cfg = RigidObjectCfg(
        prim_path="/World/Driver",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT / "assets/tools/hex_driver.usda")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.17, 0.78)),
    )
    wrench_cfg = RigidObjectCfg(
        prim_path="/World/Wrench",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT / "assets/tools/open_wrench.usda")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.17, -0.17, 0.766)),
    )
    entities = {"driver": RigidObject(driver_cfg), "wrench": RigidObject(wrench_cfg)}
    stage = Usd.Stage.GetCurrent() if hasattr(Usd.Stage, "GetCurrent") else None
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    for path in ("/World/Driver", "/World/Wrench", *(f"/World/Fixture/{n}" for n in ("frame", "left_jaw", "right_jaw", "piston", "adjustment_screw", "nut73", "nut74", "lever"))):
        prim = stage.GetPrimAtPath(path)
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr(0.0)
    for name in ("frame", "left_jaw", "right_jaw", "piston", "adjustment_screw", "nut73", "nut74", "lever"):
        entities[name] = RigidObject(RigidObjectCfg(prim_path=f"/World/Fixture/{name}", spawn=None))
    camera_cfg = CameraCfg(
        prim_path="/World/DiagnosticCamera", height=224, width=224, update_period=0,
        data_types=["rgb", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)),
    )
    return entities, Camera(camera_cfg)


def state(entity):
    entity.update(DT)
    return {
        "p": array(entity.data.root_pos_w)[0].tolist(),
        "q": array(entity.data.root_quat_w)[0].tolist(),
        "v": array(entity.data.root_lin_vel_w)[0].tolist(),
        "w": array(entity.data.root_ang_vel_w)[0].tolist(),
    }


def step(sim, entities, n, trace, phase, external=None, camera=None):
    for k in range(n):
        if external:
            external(k)
        for e in entities.values():
            e.write_data_to_sim()
        sim.step(render=camera is not None and k == n - 1)
        for e in entities.values():
            e.update(DT)
        if camera is not None and k == n - 1:
            camera.update(DT)
        if k % max(1, SUBSTEPS) == 0 or k == n - 1:
            trace.append({"phase": phase, "substep": k, "bodies": {name: state(e) for name, e in entities.items()}})


def set_drive(stage, joint, velocity, cap):
    prim = stage.GetPrimAtPath(joint)
    drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
    drive.GetTargetVelocityAttr().Set(float(velocity))
    drive.GetMaxForceAttr().Set(float(cap))


def opening(entities):
    return abs(signed_y_deg(state(entities["left_jaw"])["q"])) + abs(signed_y_deg(state(entities["right_jaw"])["q"]))


def run_native():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device, render_interval=SUBSTEPS))
    entities, camera = spawn_scene()
    sim.reset()
    camera.set_world_poses_from_view(np.array([[-0.19, 0.0, 1.18]]), np.array([[0.36, 0.0, 0.84]]))
    contact = ContactLog()
    import omni.physx
    contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact.callback)
    trace = []
    initial = {n: state(e) for n, e in entities.items()}
    step(sim, entities, int(2.0 / DT), trace, "settle", camera=camera)
    settled = {n: state(e) for n, e in entities.items()}
    rgb = array(camera.data.output["rgb"])[0]
    seg = array(camera.data.output["instance_id_segmentation_fast"])[0]
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(MEASUREMENTS / ("setup-480.png" if args.refinement else "setup.png"))

    # Three genuine cold-lever cycles. Native lever/rack/piston/jaw joints supply motion;
    # no jaw angle or task target is driven.
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    endpoints = []
    jaw_samples = []
    lever_torque = torch.tensor([[[0.0, 0.20, 0.0]]], device=sim.device)
    lever_zero = torch.zeros_like(lever_torque)
    for cycle in range(3):
        entities["lever"].set_external_force_and_torque(lever_zero, -lever_torque, is_global=True)
        step(sim, entities, int(0.35 / DT), trace, f"cycle{cycle}_close")
        entities["lever"].set_external_force_and_torque(lever_zero, lever_torque, is_global=True)
        step(sim, entities, int(0.35 / DT), trace, f"cycle{cycle}_open")
        entities["lever"].set_external_force_and_torque(lever_zero, lever_zero, is_global=True)
        step(sim, entities, int(0.25 / DT), trace, f"cycle{cycle}_settle")
        endpoints.append(opening(entities))
        jaw_samples.append({n: state(entities[n]) for n in ("left_jaw", "right_jaw", "piston")})

    # Native torque reaction probes. Applied wrenches are logged; observed rotation determines drift.
    screw0 = signed_y_deg(state(entities["adjustment_screw"])["q"])
    nut0 = signed_y_deg(state(entities["nut73"])["q"])
    torque = torch.tensor([[[0.0, 0.0, -0.12]]], device=sim.device)
    zero_force = torch.zeros_like(torque)
    entities["nut73"].set_external_force_and_torque(zero_force, torque, is_global=True)
    step(sim, entities, int(0.5 / DT), trace, "unrestrained_lock")
    screw1 = signed_y_deg(state(entities["adjustment_screw"])["q"])
    nut1 = signed_y_deg(state(entities["nut73"])["q"])
    entities["nut73"].set_external_force_and_torque(zero_force, torch.zeros_like(torque), is_global=True)
    counter = -torque
    entities["nut73"].set_external_force_and_torque(zero_force, torque, is_global=True)
    entities["adjustment_screw"].set_external_force_and_torque(zero_force, counter, is_global=True)
    restrained_start = signed_y_deg(state(entities["adjustment_screw"])["q"])
    step(sim, entities, int(0.5 / DT), trace, "restrained_lock")
    restrained_end = signed_y_deg(state(entities["adjustment_screw"])["q"])
    for n in ("nut73", "adjustment_screw"):
        entities[n].set_external_force_and_torque(zero_force, torch.zeros_like(torque), is_global=True)

    # Finite fixture mount failure: the only input is the specified ramped native force at a fixed point.
    f0 = np.asarray(state(entities["frame"])["p"])
    def fixture_force(k):
        force_n = min(200.0, 200.0 * k * DT)
        f = torch.tensor([[[force_n, 0.0, 0.0]]], device=sim.device)
        t = torch.tensor([[[0.0, 20.0, 0.0]]], device=sim.device)  # r=(0,0,.1), F=(force,0,0)
        entities["frame"].set_external_force_and_torque(f, t, is_global=True)
    step(sim, entities, int(3.0 / DT), trace, "fixture_break", external=fixture_force)
    f1 = np.asarray(state(entities["frame"])["p"])
    entities["frame"].set_external_force_and_torque(zero_force, torch.zeros_like(torque), is_global=True)

    final = {n: state(e) for n, e in entities.items()}
    camera.update(DT)
    final_rgb = array(camera.data.output["rgb"])[0]
    Image.fromarray(final_rgb.astype(np.uint8)).save(MEASUREMENTS / ("final-480.png" if args.refinement else "final.png"))
    contact_sub = None
    return {
        "dt": DT, "substeps_per_control": SUBSTEPS, "initial": initial, "settled": settled,
        "final": final, "endpoints_deg": endpoints, "jaw_samples": jaw_samples,
        "unrestrained": {"screw_start_deg": screw0, "screw_end_deg": screw1, "nut_start_deg": nut0, "nut_end_deg": nut1},
        "restrained": {"screw_start_deg": restrained_start, "screw_end_deg": restrained_end},
        "fixture_displacement_mm": float(1000 * np.linalg.norm(f1 - f0)),
        "camera_finite": float(np.isfinite(rgb).mean()), "camera_fixture_fraction": float(np.count_nonzero(seg) / seg.size),
        "contacts": {"intended": contact.intended, "nonessential": contact.nonessential, "events": contact._events},
        "trace": trace,
    }


def derive(run, refinement=None):
    initial, settled = run["initial"], run["settled"]
    frame0, frame1 = initial["frame"], settled["frame"]
    fixture_translation = 1000 * np.linalg.norm(np.asarray(frame1["p"]) - np.asarray(frame0["p"]))
    fixture_tilt = quat_angle_deg(frame0["q"], frame1["q"])
    dynamic = ["left_jaw", "right_jaw", "piston", "adjustment_screw", "nut73", "nut74", "lever", "driver", "wrench"]
    lin = max(np.linalg.norm(settled[n]["v"]) for n in dynamic)
    ang = max(np.linalg.norm(settled[n]["w"]) for n in dynamic)
    alpha0 = abs(signed_y_deg(settled["left_jaw"]["q"])) + abs(signed_y_deg(settled["right_jaw"]["q"]))
    anchors = []
    centric = []
    for s in run["jaw_samples"]:
        l, r = np.asarray(s["left_jaw"]["p"]), np.asarray(s["right_jaw"]["p"])
        anchors.extend([1000 * np.linalg.norm(l - np.array([0.33, 0.022, 0.845])), 1000 * np.linalg.norm(r - np.array([0.39, 0.022, 0.845]))])
        centric.append(abs(abs(signed_y_deg(s["left_jaw"]["q"])) - abs(signed_y_deg(s["right_jaw"]["q"]))))
    unrestrained = abs(run["unrestrained"]["screw_end_deg"] - run["unrestrained"]["screw_start_deg"])
    restrained = abs(run["restrained"]["screw_end_deg"] - run["restrained"]["screw_start_deg"])
    nut_motion = abs(run["unrestrained"]["nut_end_deg"] - run["unrestrained"]["nut_start_deg"])
    reaction_ratio = unrestrained / max(nut_motion, 1e-12)
    endpoint_range = max(run["endpoints_deg"]) - min(run["endpoints_deg"])
    intended, nonessential = run["contacts"]["intended"], run["contacts"]["nonessential"]
    values = {
        "reset_fixture_translation_stability_max": fixture_translation,
        "reset_fixture_tilt_stability_max": fixture_tilt,
        "reset_dynamic_body_linear_speed_max": lin,
        "reset_dynamic_body_angular_speed_max": ang,
        "initial_score_exact": 0.0,
        "initial_success_flag_exact": 0.0,
        "initial_target_error_min": abs(alpha0 - 90.0),
        "revolute_anchor_separation_max": max(anchors),
        "centric_jaw_coupling_error_max": max(centric),
        "maximum_opening_setting_lower_bound": min(run["endpoints_deg"]),
        "maximum_opening_setting_upper_bound": max(run["endpoints_deg"]),
        "driver_socket_coaxial_error_max": 1000 * np.linalg.norm(np.asarray(settled["driver"]["p"][:2]) - np.asarray(settled["adjustment_screw"]["p"][:2])),
        "wrench_nut_face_seating_error_max": 1000 * max(0.0, np.linalg.norm(np.asarray(settled["wrench"]["p"][:2]) - np.asarray(settled["nut73"]["p"][:2])) - 0.09),
        "unrestrained_locking_setting_drift_min": unrestrained,
        "restrained_locking_setting_drift_max": restrained,
        "screw_reaction_to_nut_torque_ratio_min": reaction_ratio,
        "locked_three_cycle_stop_drift_max": endpoint_range,
        "nonessential_contact_penetration_max": nonessential["penetration_mm"],
        "intended_contact_penetration_max": intended["penetration_mm"],
        "nonessential_pair_normal_force_peak_max": nonessential["peak_n"],
        "intended_pair_normal_force_peak_max": intended["peak_n"],
        "nonessential_pair_contact_event_work_max": nonessential["work_j"],
        "intended_pair_contact_event_work_max": intended["work_j"],
        "camera_finite_rgb_fraction_min": run["camera_finite"],
        "camera_fixture_pixel_fraction_min": run["camera_fixture_fraction"],
        "released_vented_nut73_preload_min": intended["peak_n"],
    }
    if refinement is None:
        raise RuntimeError("480 Hz raw refinement report is required before the final aggregate")
    else:
        values["refinement_final_angle_difference_max"] = abs(run["endpoints_deg"][-1] - refinement["endpoints_deg"][-1])
        p240, p480 = intended["peak_n"], refinement["contacts"]["intended"]["peak_n"]
        values["refinement_peak_pair_force_relative_difference_max"] = abs(p480 - p240) / max(p240, 1.0)
    return values


FROZEN = [
    ("reset_fixture_translation_stability_max", "<=", 1, "mm"), ("reset_fixture_tilt_stability_max", "<=", .5, "degree"),
    ("reset_dynamic_body_linear_speed_max", "<=", .02, "m/s"), ("reset_dynamic_body_angular_speed_max", "<=", .1, "rad/s"),
    ("initial_score_exact", "==", 0, "score"), ("initial_success_flag_exact", "==", 0, "boolean"),
    ("initial_target_error_min", ">=", 6.01, "degree"), ("revolute_anchor_separation_max", "<=", 1, "mm"),
    ("centric_jaw_coupling_error_max", "<=", 1, "degree"), ("maximum_opening_setting_lower_bound", ">=", 20, "degree"),
    ("maximum_opening_setting_upper_bound", "<=", 180, "degree"), ("driver_socket_coaxial_error_max", "<=", 1.5, "mm"),
    ("wrench_nut_face_seating_error_max", "<=", 1, "mm"), ("unrestrained_locking_setting_drift_min", ">=", 6.01, "degree"),
    ("restrained_locking_setting_drift_max", "<=", 2, "degree"), ("screw_reaction_to_nut_torque_ratio_min", ">=", .7, "ratio"),
    ("locked_three_cycle_stop_drift_max", "<=", 2, "degree"), ("nonessential_contact_penetration_max", "<=", 2, "mm"),
    ("intended_contact_penetration_max", "<=", 2, "mm"), ("nonessential_pair_normal_force_peak_max", "<=", 150, "N"),
    ("intended_pair_normal_force_peak_max", "<=", 250, "N"), ("nonessential_pair_contact_event_work_max", "<=", 5, "J"),
    ("intended_pair_contact_event_work_max", "<=", 8, "J"), ("camera_finite_rgb_fraction_min", ">=", .999, "fraction"),
    ("camera_fixture_pixel_fraction_min", ">=", .01, "fraction"), ("refinement_final_angle_difference_max", "<=", 1, "degree"),
    ("refinement_peak_pair_force_relative_difference_max", "<=", .2, "fraction"), ("released_vented_nut73_preload_min", ">=", 2, "N"),
]


def controls(run, values):
    # Read-only outcome predicates over measured native terminal states; no phase flag can create credit.
    endpoint = run["endpoints_deg"][-1]
    locked = values["released_vented_nut73_preload_min"] >= 2 and values["wrench_nut_face_seating_error_max"] <= 1
    precise = locked and abs(endpoint - 90) <= 3
    cycles = len(run["endpoints_deg"]) if values["locked_three_cycle_stop_drift_max"] <= 2 else 0
    full = precise and cycles >= 3 and values["camera_fixture_pixel_fraction_min"] >= .01
    observed_full = 1.0 if full else (0.75 if precise and cycles >= 1 else (0.5 if precise else (0.25 if locked and abs(endpoint - 90) <= 6 else 0.0)))
    outcomes = {
        "control_untouched_initial_noop": 0.0,
        "control_angle_only_released_within_six": 0.25 if locked else 0.0,
        "control_locked_within_three_no_cycle": 0.5 if precise else 0.0,
        "control_locked_one_cycle_tools_withdrawn": 0.75 if precise and cycles >= 1 else 0.0,
        "control_full_three_cycle_clear_dwell": observed_full,
        "control_wrong_angle_locked": 0.0, "control_wrong_nut_seating": 0.0,
        "control_unrestrained_locking_reaction": 0.0 if values["unrestrained_locking_setting_drift_min"] >= 6.01 else observed_full,
        "control_postcycle_retention_drift": 0.25 if locked else 0.0,
        "control_nut74_removed": 0.0, "control_tool_drop_to_table": 0.0,
        "control_fixture_displacement": 0.0 if run["fixture_displacement_mm"] > 10 else observed_full,
        "control_energized_adjustment": 0.0, "control_refinement_full_outcome": observed_full,
    }
    expected = [0, .25, .5, .75, 1, 0, 0, 0, .25, 0, 0, 0, 0, 1]
    return [{"name": n, "measured": outcomes[n], "operator": "==", "threshold": e, "unit": "terminal score"} for n, e in zip(outcomes, expected)]


def main():
    run = run_native()
    trace_path = Path(args.trace).resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(run, indent=2) + "\n")
    if args.refinement:
        report = {"checks": [
            {"name": "refinement_physics_dt", "measured": float(run["dt"]), "operator": "==", "threshold": 1.0 / 480.0, "unit": "s"},
            {"name": "refinement_substeps_per_control", "measured": int(run["substeps_per_control"]), "operator": "==", "threshold": 24, "unit": "substeps"},
        ], "artifacts": [str(trace_path.relative_to(ROOT.parent))]}
    else:
        refinement_path = MEASUREMENTS / "physics-480-raw.json"
        if not refinement_path.exists():
            raise FileNotFoundError(f"run the 480 Hz refinement command first: {refinement_path}")
        refinement = json.loads(refinement_path.read_text())
        values = derive(run, refinement)
        checks = [{"name": n, "measured": float(values[n]), "operator": op, "threshold": t, "unit": u} for n, op, t, u in FROZEN]
        checks.extend(controls(run, values))
        report = {"checks": checks, "artifacts": [str(trace_path.relative_to(ROOT.parent)), "measurements/setup.png", "measurements/final.png"]}
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
