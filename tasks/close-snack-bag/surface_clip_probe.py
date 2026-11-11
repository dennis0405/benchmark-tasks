"""Direct-placement surface/clip contact probe; not a YAM demonstration.

This intentionally initializes an opened spring clip around the initially
flat mouth. It does not fold the packet, and is not a benchmark rollout.
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("output", type=Path)
parser.add_argument("--empty", action="store_true", help="Run the same clip without the packet")
parser.add_argument("--folded", action="store_true", help="Initialize a flat-rest packet with two folded turns")
parser.add_argument("--no-clip", action="store_true", help="Keep clip away from the folded packet")
parser.add_argument("--clip-x", type=float, default=0.198, help="Initial clip hinge x in the folded placement test")
parser.add_argument("--clip-z", type=float, default=0.0215)
parser.add_argument("--fold-radius", type=float, default=0.003)
parser.add_argument("--thickness", type=float, default=0.001)
parser.add_argument("--contact-offset", type=float, default=0.001)
parser.add_argument("--hold-fixture", action="store_true", help="Press one mouth edge and initially hold the jaw open")
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"])
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args = parser.parse_args()

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg
from bag_surface_mesh import folded_positions, packet_mesh, write_usd


def rotation(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def jaw_distances(nodes, body_poses, body_names, mouth_nodes=None):
    """Approximate proximity to the two jaw boxes, not force/contact reports."""
    results = {}
    if mouth_nodes is not None:
        nodes = nodes[mouth_nodes]
    for body, center_z in (("clip", -0.004), ("upper", 0.004)):
        pose = body_poses[body_names.index(body)]
        local = (nodes - pose[:3]) @ rotation(pose[3:])
        delta = np.abs(local - [0.022, 0, center_z]) - [0.022, 0.05, 0.003]
        signed = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)
        near = np.abs(signed) <= 0.001
        results[body] = {"min_signed_m": float(signed.min()), "nodes_within_1mm": int(near.sum()),
                         "near_y_span_m": float(np.ptp(local[near, 1])) if near.any() else 0.0}
    return results


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    usd = args.output / "packet-surface.usda"
    if usd.exists():
        raise FileExistsError(f"{usd} already exists; choose a fresh output directory")
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        nx = 80 if args.folded else 20
        write_usd(usd, flat_mouth=True, nx=nx)
        dt = 1 / 240
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics=physics_cfg))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        contents = sim_utils.CuboidCfg(size=(0.075, 0.08, 0.014),
                                       collision_props=sim_utils.CollisionPropertiesCfg())
        contents.func("/World/contents", contents, translation=(0.10, 0, 0.019))
        packet = None
        if not args.empty:
            material = PhysxSurfaceDeformableBodyMaterialCfg(
                density=900, youngs_modulus=2e6, poissons_ratio=0.35, surface_thickness=args.thickness,
                surface_bend_stiffness=0.001, static_friction=0.5, dynamic_friction=0.4,
                elasticity_damping=0.02, bend_damping=0.02)
            cfg = DeformableObjectCfg(prim_path="/World/packet", spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd.resolve()),
                deformable_props=PhysxDeformableBodyPropertiesCfg(self_collision=True,
                    solver_position_iteration_count=32, linear_damping=1.0), physics_material=material),
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0.003)))
            packet = cfg.class_type(cfg)
            sim_utils.modify_collision_properties(
                "/World/packet", sim_utils.CollisionPropertiesCfg(contact_offset=args.contact_offset, rest_offset=0.0))
        clip_cfg = ArticulationCfg(
            prim_path="/World/clip",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).with_name("assets") / "clip_v5/clip_v5.usda"),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    solver_position_iteration_count=16, enabled_self_collisions=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.4, 0, 0.08) if args.folded else (0.205, 0, 0.021),
                                                       rot=(0, 0, 1, 0),
                                                       joint_pos={"clip_hinge": -0.55 if args.folded else -0.15}),
            actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=["clip_hinge"], stiffness=0.13,
                                                     damping=0.003, friction=0.0)})
        clip = clip_cfg.class_type(clip_cfg)
        fixture = None
        if args.hold_fixture:
            fixture_cfg = RigidObjectCfg(prim_path="/World/fixture", spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.02, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.069, 0.06)))
            fixture = fixture_cfg.class_type(fixture_cfg)
        sim.reset()
        ids, _ = clip.find_joints(["clip_hinge"], preserve_order=True)
        q = torch.tensor([[-0.55 if args.folded else -0.15]], dtype=torch.float32, device=args.device)
        clip.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), joint_ids=ids)
        clip.set_joint_position_target_index(target=q if fixture else torch.zeros_like(q), joint_ids=ids)
        mouth_nodes = None
        if args.folded and packet:
            from scipy.spatial import cKDTree
            rest, _ = packet_mesh(nx=nx, flat_mouth=True)
            rest_world = rest + np.array([0, 0, 0.003], dtype=np.float32)
            state = packet.data.default_nodal_state_w.torch.clone()
            default_nodes = state[0, :, :3].detach().cpu().numpy()
            distance, mapping = cKDTree(rest_world).query(default_nodes)
            # Initialization may advance gravity slightly before this read.
            if len(mapping) != len(rest) or np.max(distance) > 5e-4 or len(np.unique(mapping)) != len(rest):
                raise RuntimeError(f"Cannot match nodes: simulation={len(mapping)}, authored={len(rest)}, "
                                   f"max_distance={np.max(distance)}, unique={len(np.unique(mapping))}")
            folded = folded_positions(rest, radius=args.fold_radius)[mapping] + np.array([0, 0, 0.003], dtype=np.float32)
            mouth_nodes = rest[mapping, 0] >= 0.16
            packet.write_nodal_pos_to_sim_index(torch.tensor(folded[None], device=args.device))
            packet.write_nodal_velocity_to_sim_index(torch.zeros_like(state[..., 3:]))
            if not args.no_clip:
                clip.write_root_pose_to_sim_index(root_pose=torch.tensor([[args.clip_x, 0, args.clip_z, 0, 0, 1, 0]],
                                                  device=args.device, dtype=torch.float32))
                clip.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=args.device))
        sim.forward()
        rows = []
        for step in range(1440 if fixture else 960):
            if fixture:
                # A kinematic fixture presses an outer edge; the hinged jaw is
                # commanded open for one second, then released. Neither is a robot.
                if step < 120:
                    z = 0.06 - 0.025 * step / 120
                elif step < 480:
                    z = 0.035
                elif step < 600:
                    z = 0.035 + 0.025 * (step - 480) / 120
                else:
                    z = 0.06
                fixture.write_root_pose_to_sim_index(root_pose=torch.tensor([[0.15, 0.069, z, 0, 0, 0, 1]],
                                                     dtype=torch.float32, device=args.device))
                if step == 240:
                    clip.set_joint_position_target_index(target=torch.zeros_like(q), joint_ids=ids)
            clip.write_data_to_sim()
            sim.step(render=False)
            clip.update(dt)
            if packet:
                packet.update(dt)
            if step in (0, 59, 119, 239, 479, 719, 959, 1199, 1439):
                root = clip.data.root_link_pose_w.torch[0].detach().cpu().numpy()
                row = {"step": step + 1,
                       "jaw_radians": float(clip.data.joint_pos.torch[0, ids[0]].item()),
                       "clip_xyz": root[:3].tolist()}
                if packet:
                    nodes = packet.data.nodal_pos_w.torch[0].detach().cpu().numpy()
                    row.update(packet_min=nodes.min(axis=0).tolist(), packet_max=nodes.max(axis=0).tolist(),
                               finite=bool(np.isfinite(nodes).all()),
                               jaw_proximity=jaw_distances(nodes,
                                   clip.data.body_link_pose_w.torch[0].detach().cpu().numpy(), clip.body_names,
                                   mouth_nodes))
                    if mouth_nodes is not None:
                        mouth = nodes[mouth_nodes]
                        row["mouth_span_x"] = float(np.ptp(mouth[:, 0]))
                        row["mouth_span_z"] = float(np.ptp(mouth[:, 2]))
                rows.append(row)
                print("CLIP_PROBE", json.dumps(row), flush=True)
        (args.output / "surface-clip.json").write_text(json.dumps(
            {"note": "directly initialized clip and optional folded nodes; not a robot demonstration",
             "empty": args.empty, "folded": args.folded, "no_clip": args.no_clip,
             "hold_fixture": args.hold_fixture,
             "clip_x": args.clip_x, "clip_z": args.clip_z, "fold_radius": args.fold_radius,
             "surface_thickness_m": args.thickness, "contact_offset_m": args.contact_offset,
             "dt": dt, "samples": rows}, indent=2))


if __name__ == "__main__":
    main()
