"""Newton surface and optional articulated-clip probes; not a robot demo."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("output", type=Path)
parser.add_argument("--folded", action="store_true")
parser.add_argument("--fold-radius", type=float, default=0.0018)
parser.add_argument("--duration", type=float, default=0.5)
parser.add_argument("--nx", type=int, default=80)
parser.add_argument("--ny", type=int, default=64)
parser.add_argument("--clip", action="store_true", help="Try the articulated clip in the Newton VBD backend")
parser.add_argument("--clip-x", type=float, default=0.194)
parser.add_argument("--clip-z", type=float, default=0.044)
parser.add_argument("--coupled", action="store_true", help="Use experimental two-way MJWarp/VBD coupling")
parser.add_argument("--proxy", action="store_true", help="Use the Newton proxy-coupling preset")
parser.add_argument("--coupling-mode", choices=("one_way", "two_way"), default="two_way")
parser.add_argument("--soft-ke", type=float, default=1e5)
parser.add_argument("--soft-kd", type=float, default=1.0)
parser.add_argument("--contact-capacity", type=int, default=4096)
parser.add_argument("--soft-contact-margin", type=float, default=0.0003)
parser.add_argument("--shape-gap", type=float, default=0.0005)
parser.add_argument("--physics", default="newton_vbd", choices=["newton_vbd"])
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args = parser.parse_args()

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.physics import PhysicsCfg
from isaaclab_newton.physics import NewtonSoftContactCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from pouch_geometry import open_pouch, write_usd


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    usd = args.output / "packet-surface.usda"
    if usd.exists():
        raise FileExistsError(f"{usd} already exists; choose a fresh output directory")
    if args.proxy and args.coupled:
        raise ValueError("choose either --proxy or --coupled")
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        if args.proxy:
            from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg, VBDSolverCfg
            rigid_paths = [r"/World/clip/Geometry/.*"]
            physics_cfg = NewtonCfg(
                solver_cfg=CouplerProxyCfg(entries=[
                    CouplerEntryCfg(name="rigid", solver_cfg=MJWarpSolverCfg(njmax=1024, nconmax=512,
                        ls_iterations=20, integrator="implicitfast"), bodies=rigid_paths),
                    CouplerEntryCfg(name="soft", solver_cfg=VBDSolverCfg(iterations=20,
                        particle_enable_self_contact=True, particle_self_contact_radius=0.0001,
                        particle_self_contact_margin=0.0002, particle_collision_detection_interval=2,
                        rigid_body_particle_contact_buffer_size=args.contact_capacity), all_particles=True, include_static_shapes=True),
                ], proxies=[CouplerProxyMappingCfg(source="rigid", destination="soft", bodies=rigid_paths,
                    collide_interval=1, collision_pipeline=NewtonCollisionPipelineCfg(
                        soft_contact_margin=args.soft_contact_margin, enable_rigid_soft_full_surface_contact=True))], iterations=1),
                num_substeps=4, collision_decimation=1,
                default_shape_cfg=NewtonShapeCfg(gap=args.shape_gap),
                soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=args.soft_ke, soft_contact_kd=args.soft_kd,
                                                      soft_contact_mu=0.4))
        elif args.coupled:
            from isaaclab_contrib.custom_coupling import CoupledMJWarpVBDSolverCfg
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg, VBDSolverCfg
            physics_cfg = NewtonCfg(
                solver_cfg=CoupledMJWarpVBDSolverCfg(
                    coupling_mode=args.coupling_mode,
                    rigid_solver_cfg=MJWarpSolverCfg(njmax=1024, nconmax=512, ls_iterations=20,
                                                    integrator="implicitfast"),
                    soft_solver_cfg=VBDSolverCfg(iterations=20, integrate_with_external_rigid_solver=True,
                        particle_enable_self_contact=True, particle_self_contact_radius=0.0001,
                        particle_self_contact_margin=0.0002, particle_collision_detection_interval=2,
                        rigid_body_particle_contact_buffer_size=args.contact_capacity)),
                num_substeps=4, collision_decimation=1,
                collision_cfg=NewtonCollisionPipelineCfg(soft_contact_margin=args.soft_contact_margin,
                                                        enable_rigid_soft_full_surface_contact=True),
                default_shape_cfg=NewtonShapeCfg(gap=args.shape_gap),
                soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=args.soft_ke, soft_contact_kd=args.soft_kd,
                                                      soft_contact_mu=0.4))
        else:
            from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonShapeCfg
            physics_cfg.solver_cfg.iterations = 20
            physics_cfg.solver_cfg.particle_enable_self_contact = True
            physics_cfg.solver_cfg.particle_self_contact_radius = 0.0001
            physics_cfg.solver_cfg.particle_self_contact_margin = 0.0002
            physics_cfg.solver_cfg.particle_collision_detection_interval = 2
            physics_cfg.solver_cfg.rigid_body_particle_contact_buffer_size = args.contact_capacity
            physics_cfg.num_substeps = 4
            physics_cfg.collision_decimation = 1
            physics_cfg.collision_cfg = NewtonCollisionPipelineCfg(soft_contact_margin=args.soft_contact_margin)
            physics_cfg.default_shape_cfg = NewtonShapeCfg(gap=args.shape_gap)
            physics_cfg.soft_contact_cfg = NewtonSoftContactCfg(soft_contact_ke=args.soft_ke, soft_contact_kd=args.soft_kd,
                                                                soft_contact_mu=0.4)
        dt = 1 / 240
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics=physics_cfg))
        write_usd(usd, nx=args.nx, ny=args.ny)
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        cfg = DeformableObjectCfg(prim_path="/World/packet", spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd.resolve()), deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0, particle_radius=0.0001, tri_ke=500, tri_ka=500, edge_ke=0.5)),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0.025)))
        packet = cfg.class_type(cfg)
        clip = None
        if args.clip:
            clip_cfg = ArticulationCfg(prim_path="/World/clip", spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).with_name("assets") / "clip_v5/clip_v5.usda"),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    solver_position_iteration_count=16, enabled_self_collisions=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0)),
                # Spawn clear of the as-yet-unfolded packet during sim.reset.
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.4, 0, 0.1), rot=(0, 0, 1, 0),
                                                           joint_pos={"clip_hinge": -0.55}),
                actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=["clip_hinge"], stiffness=0.13,
                                                         damping=0.003, friction=0.0)})
            clip = clip_cfg.class_type(clip_cfg)
        sim.reset()
        packet.update(dt)
        if clip:
            clip.update(dt)
            print("CLIP_INERTIA", "names", clip.body_names, "mass", clip.data.body_mass.torch.cpu().tolist(),
                  "inertia", clip.data.body_inertia.torch.cpu().tolist(), flush=True)
            ids, _ = clip.find_joints(["clip_hinge"], preserve_order=True)
            opened = torch.tensor([[-0.55]], dtype=torch.float32, device=args.device)
            clip.write_joint_state_to_sim_index(position=opened, velocity=torch.zeros_like(opened), joint_ids=ids)
            clip.set_joint_position_target_index(target=torch.zeros_like(opened), joint_ids=ids)
            clip.write_root_pose_to_sim_index(root_pose=torch.tensor(
                [[args.clip_x, 0, args.clip_z, 0, 0, 1, 0]], device=args.device, dtype=torch.float32))
            clip.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=args.device))
        rest, faces, grid = open_pouch(nx=args.nx, ny=args.ny)
        initial = packet.data.default_nodal_state_w.torch[0, :, :3].detach().cpu().numpy()
        if initial.shape != rest.shape or np.max(np.linalg.norm(initial - rest - [0, 0, 0.025], axis=1)) > 0.001:
            raise RuntimeError("Newton node ordering differs from authored mesh")
        if args.folded or args.clip:
            raise ValueError("pouch geometry probe does not support folding or clip")
        snapshots, samples = {}, []
        total = round(args.duration / dt)
        sample_steps = {0, 59, 119, 239, 479, total - 1}
        def column(x):
            i = round(x / 0.20 * args.nx)
            return np.concatenate([grid[0, i], grid[1, i]])
        columns = [column(x) for x in (0.14, 0.16, 0.18, 0.20)]
        for step in range(total):
            if clip:
                clip.write_data_to_sim()
            sim.step(render=False)
            packet.update(dt)
            if clip:
                clip.update(dt)
            if step in sample_steps:
                nodes = packet.data.nodal_pos_w.torch[0].detach().cpu().numpy()
                snapshots[f"nodes_{step + 1}"] = nodes.copy()
                positions = [nodes[indices] for indices in columns]
                vecs = [positions[i + 1] - positions[i] for i in range(3)]
                cosines = [np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) *
                           np.linalg.norm(b, axis=1), 1e-12) for a, b in zip(vecs, vecs[1:])]
                folded_fraction = float(np.mean((cosines[0] < -0.5) & (cosines[1] < -0.5)))
                row = {"step": step + 1, "time_s": (step + 1) * dt,
                       "bounds_min": nodes.min(axis=0).tolist(), "bounds_max": nodes.max(axis=0).tolist(),
                       "finite": bool(np.isfinite(nodes).all()), "fold_cosine_mean": [float(c.mean()) for c in cosines],
                       "both_folds_reversed_fraction": folded_fraction}
                if clip:
                    clip_pos = clip.data.root_link_pose_w.torch[0, :3].detach().cpu().numpy()
                    jaw = float(clip.data.joint_pos.torch[0, 0].item())
                    row["clip_finite"] = bool(np.isfinite(clip_pos).all() and np.isfinite(jaw))
                    row["clip_plausible"] = bool(row["clip_finite"] and np.max(np.abs(clip_pos)) < 2 and abs(jaw) < 1)
                    row["clip_xyz"] = clip_pos.tolist() if np.isfinite(clip_pos).all() else None
                    row["jaw_radians"] = jaw if np.isfinite(jaw) else None
                samples.append(row)
                print("NEWTON_SURFACE", json.dumps(row), flush=True)
        np.savez_compressed(args.output / "surface-snapshots.npz", faces=faces, **snapshots)
        (args.output / "newton-result.json").write_text(json.dumps({"dt": dt, "folded": args.folded, "clip": args.clip,
            "coupled": args.coupled,
            "proxy": args.proxy,
            "coupling_mode": args.coupling_mode if args.coupled else None,
            "nx": args.nx, "ny": args.ny,
            "soft_ke": args.soft_ke, "soft_kd": args.soft_kd,
            "contact_capacity": args.contact_capacity,
            "soft_contact_margin_m": args.soft_contact_margin, "shape_gap_m": args.shape_gap,
            "clip_x": args.clip_x, "clip_z": args.clip_z,
            "fold_radius": args.fold_radius, "samples": samples}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
