"""Isolated deformable-packet indentation experiment, not a YAM demo.

The kinematic plate presses and releases the packet. This is a physics probe:
its movement is not a robot action, and its material is not calibrated.
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("output", type=Path)
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"])
parser.add_argument("--press-x", type=float, default=0.15)
parser.add_argument("--press-z", type=float, default=0.009)
parser.add_argument("--dynamic-friction", type=float, default=0.4)
parser.add_argument("--thickness", type=float, default=0.001)
parser.add_argument("--contact-offset", type=float, default=0.001)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args = parser.parse_args()

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg
from bag_surface_mesh import write_usd


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    usd = args.output / "packet-surface.usda"
    if usd.exists():
        raise FileExistsError(f"{usd} already exists; choose a fresh output directory")
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        write_usd(usd)
        dt = 1 / 240
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics=physics_cfg))
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        # Fixed, rigid contents proxy keeps the packet partly filled. It is a
        # deliberate test fixture, not a model of individual potato chips.
        contents = sim_utils.CuboidCfg(size=(0.075, 0.08, 0.014),
                                       collision_props=sim_utils.CollisionPropertiesCfg())
        contents.func("/World/contents", contents, translation=(0.10, 0, 0.019))
        material = PhysxSurfaceDeformableBodyMaterialCfg(
            density=900.0, youngs_modulus=2e6, poissons_ratio=0.35,
            surface_thickness=args.thickness, surface_bend_stiffness=0.001,
            static_friction=0.5, dynamic_friction=args.dynamic_friction, elasticity_damping=0.02, bend_damping=0.02)
        cfg = DeformableObjectCfg(
            prim_path="/World/packet",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd.resolve()),
                deformable_props=PhysxDeformableBodyPropertiesCfg(self_collision=True,
                    solver_position_iteration_count=32, linear_damping=1.0),
                physics_material=material),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0.003)))
        packet = cfg.class_type(cfg)
        sim_utils.modify_collision_properties(
            "/World/packet", sim_utils.CollisionPropertiesCfg(contact_offset=args.contact_offset, rest_offset=0.0))
        plate_cfg = RigidObjectCfg(
            prim_path="/World/plate",
            spawn=sim_utils.CuboidCfg(size=(0.05, 0.08, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(args.press_x, 0, 0.09)))
        plate = plate_cfg.class_type(plate_cfg)
        sim.reset()
        packet.update(dt)
        initial = packet.data.nodal_pos_w.torch[0].detach().cpu().numpy().copy()
        print("INITIAL_BOUNDS", initial.min(axis=0), initial.max(axis=0), flush=True)
        sample = ((initial[:, 0] > args.press_x - 0.02) & (initial[:, 0] < args.press_x + 0.02) &
                  (np.abs(initial[:, 1]) < 0.025) & (initial[:, 2] > 0.025))
        if not sample.any():
            raise RuntimeError("Unable to select the upper packet surface")
        records = []
        for step in range(720):
            # settle, descend, hold, ascend, recover; kinematic plate only.
            if step < 240:
                z = 0.09
            elif step < 360:
                z = 0.09 - (0.09 - args.press_z) * (step - 240) / 120
            elif step < 480:
                z = args.press_z
            elif step < 600:
                z = args.press_z + (0.09 - args.press_z) * (step - 480) / 120
            else:
                z = 0.09
            pose = torch.tensor([[args.press_x, 0, z, 1, 0, 0, 0]], dtype=torch.float32, device=args.device)
            plate.write_root_pose_to_sim(pose)
            sim.step(render=False)
            packet.update(dt)
            if step in (0, 239, 359, 479, 599, 719):
                nodes = packet.data.nodal_pos_w.torch[0].detach().cpu().numpy()
                row = {"step": step + 1, "plate_z": z, "upper_center_z": float(np.median(nodes[sample, 2])),
                       "bounds_min": nodes.min(axis=0).tolist(), "bounds_max": nodes.max(axis=0).tolist(),
                       "finite": bool(np.isfinite(nodes).all())}
                records.append(row)
                print("SURFACE", json.dumps(row), flush=True)
        report = {"note": "uncalibrated kinematic indentation; not a robot demo", "dt": dt,
                  "dynamic_friction": args.dynamic_friction, "surface_thickness_m": args.thickness,
                  "contact_offset_m": args.contact_offset,
                  "selected_nodes": int(sample.sum()), "samples": records}
        (args.output / "surface-press.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
