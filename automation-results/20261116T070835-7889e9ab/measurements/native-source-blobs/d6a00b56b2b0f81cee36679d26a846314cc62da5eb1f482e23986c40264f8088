#!/usr/bin/env python3
"""Bounded native RigidObject API probe; this is not task qualification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe native cuboid pose, XYZW rotation, gravity, CUDA state, and dt.")
parser.add_argument("--physics-hz", type=int, choices=(240, 480), required=True)
parser.add_argument("--steps", type=int, default=24, help="Integrated physics steps after the reset sample.")
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0", visualizer=["none"])
ARGS = parser.parse_args()

APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


# Nontrivial normalized XYZW quaternion: 0.7 rad about normalized (1, 2, 3).
AUTHORED_QUAT_XYZW = (
    0.0916432938695913,
    0.1832865877391826,
    0.2749298816087739,
    0.9393727128473789,
)


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    cuboid: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ProbeCuboid",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.05, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=AUTHORED_QUAT_XYZW,
        ),
    )


def proxy_to_torch(value):
    """Use the pinned public ProxyArray torch accessor, retaining native device."""
    tensor = value.torch
    return tensor() if callable(tensor) else tensor


def pose_sample(cuboid) -> tuple[list[float], object, torch.Tensor]:
    proxy = cuboid.data.root_pose_w
    tensor = proxy_to_torch(proxy)
    # Synchronize before copying so each receipt is a fresh completed native read.
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    values = tensor[0].detach().cpu().to(torch.float64).tolist()
    return values, proxy, tensor


def main() -> None:
    if ARGS.steps < 2:
        raise ValueError("--steps must be at least 2")

    requested_dt = 1.0 / float(ARGS.physics_hz)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=requested_dt, device=ARGS.device, use_fabric=True)
    )
    scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    reset_pose, reset_proxy, reset_tensor = pose_sample(scene["cuboid"])
    trace = [{"step": 0, "time_s": 0.0, "pose_xyzw": reset_pose}]

    sim_dt = float(sim.get_physics_dt())
    sim_dt_error = abs(sim_dt - requested_dt)
    for step in range(1, ARGS.steps + 1):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)
        current_pose, _, _ = pose_sample(scene["cuboid"])
        trace.append({"step": step, "time_s": step * sim_dt, "pose_xyzw": current_pose})

    final_pose, final_proxy, final_tensor = pose_sample(scene["cuboid"])
    all_values = [x for sample in trace for x in sample["pose_xyzw"]]
    finite_fraction = sum(math.isfinite(x) for x in all_values) / len(all_values)
    reset_quat = reset_pose[3:7]
    final_quat = final_pose[3:7]
    reset_quat_norm = math.sqrt(sum(x * x for x in reset_quat))
    final_quat_norm = math.sqrt(sum(x * x for x in final_quat))
    authored_linf_error = max(abs(a - b) for a, b in zip(reset_quat, AUTHORED_QUAT_XYZW))
    delta_z = final_pose[2] - reset_pose[2]
    device_index = -1 if final_tensor.device.index is None else int(final_tensor.device.index)
    cuda_backed = float(final_tensor.is_cuda and str(final_tensor.device) == ARGS.device)

    report_path = ARGS.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = report_path.with_name(report_path.stem + "-trace.json")
    trace_path.write_text(json.dumps({"physics_hz": ARGS.physics_hz, "samples": trace}, indent=2) + "\n")

    workspace = Path(__file__).resolve().parents[1]
    artifact_path = trace_path.relative_to(workspace).as_posix()
    report = {
        "scope": "API-only RigidObject cuboid preflight; no aperture, retention, spill, YAM, or 20 Hz claim",
        "physics_hz": ARGS.physics_hz,
        "steps": ARGS.steps,
        "state": {
            "reset_proxy_type": f"{type(reset_proxy).__module__}.{type(reset_proxy).__name__}",
            "final_proxy_type": f"{type(final_proxy).__module__}.{type(final_proxy).__name__}",
            "tensor_type": f"{type(final_tensor).__module__}.{type(final_tensor).__name__}",
            "tensor_device": str(final_tensor.device),
            "tensor_device_index": device_index,
            "tensor_shape": list(final_tensor.shape),
            "quaternion_order": "XYZW",
            "authored_quaternion_xyzw": list(AUTHORED_QUAT_XYZW),
            "reset_pose_xyzw": reset_pose,
            "final_pose_xyzw": final_pose,
        },
        "checks": [
            # Keep the report contract machine-evaluable: operators are simple
            # comparators and the measured value is the numerical error itself.
            {"name": "sim_dt_absolute_error", "measured": sim_dt_error, "operator": "<=", "threshold": 1.0e-9, "unit": "s"},
            {"name": "native_cuda_state", "measured": cuda_backed, "operator": "==", "threshold": 1.0, "unit": "bool"},
            {"name": "finite_pose_fraction", "measured": finite_fraction, "operator": "==", "threshold": 1.0, "unit": "fraction"},
            {"name": "reset_quaternion_norm_error", "measured": abs(reset_quat_norm - 1.0), "operator": "<=", "threshold": 1.0e-5, "unit": "1"},
            {"name": "final_quaternion_norm_error", "measured": abs(final_quat_norm - 1.0), "operator": "<=", "threshold": 1.0e-5, "unit": "1"},
            {"name": "authored_xyzw_readback_linf_error", "measured": authored_linf_error, "operator": "<=", "threshold": 1.0e-5, "unit": "1"},
            {"name": "gravity_delta_z", "measured": delta_z, "operator": "<", "threshold": -1.0e-4, "unit": "m"},
            {"name": "fresh_pose_sample_count", "measured": float(len(trace)), "operator": ">=", "threshold": float(ARGS.steps + 1), "unit": "samples"},
        ],
        "artifacts": [artifact_path],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    SIMULATION_APP.close()


if __name__ == "__main__":
    main()
