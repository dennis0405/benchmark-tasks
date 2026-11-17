"""Apply the declared horizontal overload to the actual mounted fixture in isolation."""

import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument(
    "--external-mount",
    action="store_true",
    help="Screen a floating articulation supported by an external fixed joint.",
)
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--hz", type=int, choices=(240, 480), default=480)
args = p.parse_args()
app = AppLauncher(headless=True).app

import numpy as np
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.physics.physx_manager import PhysxManager
from pxr import PhysicsSchemaTools, UsdGeom, UsdPhysics
from omni.physx import get_physx_interface
from omni.physx.bindings._physx import SimulationEvent
from build_bench import build


def main():
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asset = args.out.with_suffix(".usda")
    build(
        asset,
        args.assets,
        continuous_seat=True,
        implicit_axial=True,
        external_mount=args.external_mount,
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / args.hz,
            device="cuda:0",
            physics=PhysxCfg(
                solver_type=0, enable_external_forces_every_iteration=False
            ),
        )
    )
    UsdGeom.Xform.Define(
        sim.stage, "/World/Press"
    ).GetPrim().GetReferences().AddReference(str(asset.resolve()))
    # A support surface catches detached equipment. Its top matches the native Arena station.
    cfg = sim_utils.CuboidCfg(
        size=(1.1, 1.3, 0.06), collision_props=sim_utils.CollisionPropertiesCfg()
    )
    cfg.func("/World/table", cfg, translation=(0.36, 0, 0.7175))
    breaks = []

    def observe(event):
        if event.type == int(SimulationEvent.JOINT_BREAK):
            breaks.append(
                str(PhysicsSchemaTools.decodeSdfPath(*event.payload["jointPath"]))
            )

    stream = get_physx_interface().get_simulation_event_stream_v2()
    subscription = stream.create_subscription_to_pop(observe)
    sim.reset()
    view = PhysxManager.get_physics_sim_view().create_rigid_body_view(
        ["/World/Press/frame"]
    )
    zero = view.get_transforms().numpy()[0, :3].copy()
    joint = UsdPhysics.FixedJoint(sim.stage.GetPrimAtPath("/World/Press/mount"))
    report = dict(
        hz=args.hz,
        external_mount=args.external_mount,
        authored_break_force_n=joint.GetBreakForceAttr().Get(),
        authored_break_torque_nm=joint.GetBreakTorqueAttr().Get(),
        displacement_m=0.0,
        break_paths=breaks,
        peak_applied_force_n=0.0,
    )
    ids = wp.array([0], dtype=wp.uint32, device="cuda:0")
    for k in range(3 * args.hz):
        force = min(200.0, 200 * (k + 1) / args.hz)
        f = wp.array(np.array([[force, 0, 0]], dtype=np.float32), device="cuda:0")
        point = wp.array(
            np.array([zero + [0, 0, 0.1]], dtype=np.float32), device="cuda:0"
        )
        view.apply_forces_and_torques_at_position(f, None, point, ids, True)
        sim.step(render=False)
        current = view.get_transforms().numpy()[0, :3]
        report["peak_applied_force_n"] = force
        report["displacement_m"] = max(
            report["displacement_m"], float(np.linalg.norm(current - zero))
        )
        if breaks and report["displacement_m"] > 0.010:
            break
    report["substeps"] = k + 1
    report["passed"] = (
        "/World/Press/mount" in breaks and report["displacement_m"] > 0.010
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("MOUNT", report, flush=True)
    subscription = None


try:
    main()
finally:
    app.close()
