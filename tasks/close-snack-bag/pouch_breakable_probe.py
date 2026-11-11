"""Newton surface and optional articulated-clip probes; not a robot demo."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("output", type=Path)
parser.add_argument("--folded", action="store_true")
parser.add_argument("--initial-crease-yield", type=float, default=None, help="Experimental reset-time plasticity diagnostic, radians")
parser.add_argument("--dynamic-crease-yield", type=float, default=None, help="Update permanent bend after each simulation step")
parser.add_argument("--foldable-mouth", action="store_true")
parser.add_argument("--damage-force", type=float, default=0.0, help="Diagnostic balanced bend force on chip 0")
parser.add_argument("--break-extension", type=float, default=.00015, help="Uncalibrated chip bond failure extension in metres")
parser.add_argument("--air", action="store_true", help="Experimental lumped isothermal gas")
parser.add_argument("--air-opening-scale", type=float, default=1.0)
parser.add_argument("--air-channel", action="store_true", help="Experimental narrowest-mouth leak proxy")
parser.add_argument("--fold-radius", type=float, default=0.0018)
parser.add_argument("--duration", type=float, default=0.5)
parser.add_argument("--contents-stiffness", type=float, default=1000.0)
parser.add_argument("--contents-damping", type=float, default=1.0)
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
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--self-radius", type=float, default=0.0001)
parser.add_argument("--self-margin", type=float, default=0.0002)
parser.add_argument("--collision-interval", type=int, default=2)
parser.add_argument("--soft-contact-margin", type=float, default=0.0003)
parser.add_argument("--shape-gap", type=float, default=0.0005)
parser.add_argument("--physics", default="newton_vbd", choices=["newton_vbd"])
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args = parser.parse_args()
if args.clip or args.coupled or args.proxy:
    parser.error("This probe supports surface and deformable contents only; clip and rigid coupling are not implemented.")
if args.folded and not args.foldable_mouth:
    parser.error("--folded requires --foldable-mouth")
if args.initial_crease_yield is not None and not args.folded:
    parser.error("--initial-crease-yield requires --folded")
if args.dynamic_crease_yield is not None and args.initial_crease_yield is not None:
    parser.error("choose either dynamic or initial crease plasticity")

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.physics import PhysicsCfg
from isaaclab_newton.physics import NewtonSoftContactCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg, NewtonDeformableBodyMaterialCfg
from pouch_geometry import open_pouch, write_usd
from bag_surface_mesh import folded_positions


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
            physics_cfg.solver_cfg.iterations = args.iterations
            physics_cfg.solver_cfg.particle_enable_self_contact = True
            physics_cfg.solver_cfg.particle_self_contact_radius = args.self_radius
            physics_cfg.solver_cfg.particle_self_contact_margin = args.self_margin
            physics_cfg.solver_cfg.particle_collision_detection_interval = args.collision_interval
            physics_cfg.solver_cfg.rigid_body_particle_contact_buffer_size = args.contact_capacity
            physics_cfg.num_substeps = args.substeps
            physics_cfg.collision_decimation = 1
            physics_cfg.collision_cfg = NewtonCollisionPipelineCfg(soft_contact_margin=args.soft_contact_margin)
            physics_cfg.default_shape_cfg = NewtonShapeCfg(gap=args.shape_gap)
            physics_cfg.soft_contact_cfg = NewtonSoftContactCfg(soft_contact_ke=args.soft_ke, soft_contact_kd=args.soft_kd,
                                                                soft_contact_mu=0.4)
        dt = 1 / 240
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics=physics_cfg))
        mouth_gap = 0.0003 if args.foldable_mouth else 0.002
        write_usd(usd, nx=args.nx, ny=args.ny, mouth_half_gap=mouth_gap, foldable_mouth=args.foldable_mouth)
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
        import warp as wp
        from isaaclab.physics import PhysicsEvent
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from breakable_chip_builder import add_breakable_chips, ChipFractureMonitor
        chip_groups, seam_groups = [], []
        def add_chips(_event):
            groups, seams = add_breakable_chips(NewtonManager._builder)
            chip_groups.extend(groups)
            seam_groups.extend(seams)
            NewtonManager._builder.color()
        chip_handle = NewtonManager.register_callback(add_chips, PhysicsEvent.MODEL_INIT,
                                                       name="pouch_breakable_chips", wrap_weak_ref=False)
        sim.reset()
        packet.update(dt)
        chip_model = NewtonManager.get_model()
        if len(chip_groups) != 4:
            raise RuntimeError("builder did not register the four chips")
        chip_broken = [False] * len(chip_groups)
        events = []
        fracture = ChipFractureMonitor(chip_model, seam_groups, args.break_extension)
        NewtonManager.register_state_force_callback(fracture.apply)
        seam_peak = [0.0] * len(chip_groups)
        loader = None
        if args.damage_force:
            from chip_loading import ChipBendLoad
            all_points = chip_model.particle_q.numpy()
            indices = chip_groups[0]
            x = all_points[indices, 0]
            left = indices[np.isclose(x, x.min(), atol=1e-7)]
            right = indices[np.isclose(x, x.max(), atol=1e-7)]
            loader = ChipBendLoad(chip_model, left, right)
            NewtonManager.register_state_force_callback(loader.apply)
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
        rest, faces, grid = open_pouch(nx=args.nx, ny=args.ny, mouth_half_gap=mouth_gap, foldable_mouth=args.foldable_mouth)
        initial = packet.data.default_nodal_state_w.torch[0, :, :3].detach().cpu().numpy()
        if initial.shape != rest.shape or np.max(np.linalg.norm(initial - rest - [0, 0, 0.025], axis=1)) > 0.001:
            raise RuntimeError("Newton node ordering differs from authored mesh")
        if args.clip or (args.folded and not args.foldable_mouth):
            raise ValueError("clip unsupported; --folded requires --foldable-mouth")
        if args.folded:
            shift = np.array([0, 0, 0.019], dtype=np.float32)
            folded = folded_positions(rest + shift, radius=args.fold_radius) - shift
            folded += np.array([0, 0, 0.025], dtype=np.float32)
            packet.write_nodal_pos_to_sim_index(torch.tensor(folded[None], device=args.device))
            packet.write_nodal_velocity_to_sim_index(torch.zeros((1, len(rest), 3), device=args.device))
            sim.forward()
        crease_count, gas = 0, None
        if args.initial_crease_yield is not None or args.dynamic_crease_yield is not None or args.air:
            from isaaclab_newton.physics.newton_manager import NewtonManager
            model = NewtonManager.get_model()
            model_points = model.particle_q.numpy()
            candidates = np.flatnonzero(np.linalg.norm(model_points - initial[0], axis=1) < 1e-6)
            offsets = [int(i) for i in candidates if i + len(initial) <= len(model_points)
                       and np.max(np.abs(model_points[i:i + len(initial)] - initial)) < 1e-6]
            if len(offsets) != 1:
                raise RuntimeError(f"Cannot uniquely locate pouch particles: {offsets}")
            start = offsets[0]
        if args.dynamic_crease_yield is not None:
            import warp as wp
            from crease_model import hinge_angles, yielded_rest_angles
            edges = model.edge_indices.numpy()
            safe = np.clip(edges[:, 2:4], 0, len(model_points) - 1)
            is_pouch = np.all((edges >= start) & (edges < start + len(initial)), axis=1)
            crease_region = is_pouch & (model_points[safe, 0].mean(axis=1) >= 0.145)
            dynamic_changed = np.zeros(len(edges), dtype=bool)
        if args.air:
            from pouch_air_warp import LumpedGas
            gas = LumpedGas(model, initial, faces, grid, start, dt / args.substeps, args.air_opening_scale, channel=args.air_channel, contact_gap=2*args.self_radius)
            NewtonManager.register_state_force_callback(gas.apply)
        if args.initial_crease_yield is not None:
            import warp as wp
            from crease_model import hinge_angles, yielded_rest_angles
            edges = model.edge_indices.numpy()
            old_angles = model.edge_rest_angle.numpy()
            posed = model_points.copy()
            posed[start:start + len(folded)] = folded
            angles, valid = hinge_angles(posed, edges)
            is_pouch = np.all((edges >= start) & (edges < start + len(initial)), axis=1)
            safe = np.clip(edges[:, 2:4], 0, len(model_points) - 1)
            eligible = valid & is_pouch & (model_points[safe, 0].mean(axis=1) >= 0.145)
            updated = yielded_rest_angles(old_angles, angles, eligible, args.initial_crease_yield)
            crease_count = int(np.count_nonzero(np.abs(updated - old_angles) > 1e-6))
            if crease_count == 0:
                raise RuntimeError("No yielding edges; check crease region and node mapping")
            wp.copy(model.edge_rest_angle, wp.array(updated, dtype=wp.float32, device=model.device))
            print("RESET_CREASE_EDGES", crease_count, flush=True)
        snapshots, samples = {}, []
        total = round(args.duration / dt)
        sample_steps = {0, 59, 119, 239, 479, total - 1}
        def column(x):
            i = round(x / 0.20 * args.nx)
            return np.concatenate([grid[0, i], grid[1, i]])
        columns = [column(x) for x in (0.14, 0.16, 0.18, 0.20)]
        for step in range(total):
            if loader is not None:
                loader.set_force(args.damage_force if step * dt >= .15 and not chip_broken[0] else 0.0)
            if clip:
                clip.write_data_to_sim()
            sim.step(render=False)
            packet.update(dt)
            particle_q = NewtonManager.get_state_0().particle_q.numpy()
            flags = fracture.flags.numpy().astype(bool).tolist()
            seam_peak = fracture.peaks.numpy().tolist()
            for chip_id, (was, now) in enumerate(zip(chip_broken, flags)):
                if now and not was:
                    events.append({"step": step + 1, "chip_id": chip_id})
            chip_broken = flags
            if args.dynamic_crease_yield is not None:
                # Update after each 240 Hz step. This is a numerical prototype;
                # it does not yet account for plastic work or substep history.
                current = packet.data.nodal_pos_w.torch[0].detach().cpu().numpy()
                posed = model_points.copy()
                posed[start:start + len(current)] = current
                angles, valid = hinge_angles(posed, edges)
                old_angles = model.edge_rest_angle.numpy()
                eligible = valid & crease_region
                updated = yielded_rest_angles(old_angles, angles, eligible, args.dynamic_crease_yield)
                dynamic_changed |= np.abs(updated - old_angles) > 1e-6
                wp.copy(model.edge_rest_angle, wp.array(updated, dtype=wp.float32, device=model.device))
                crease_count = int(dynamic_changed.sum())
            if clip:
                clip.update(dt)
            if step in sample_steps:
                nodes = packet.data.nodal_pos_w.torch[0].detach().cpu().numpy()
                snapshots[f"nodes_{step + 1}"] = nodes.copy()
                content_nodes = np.concatenate([particle_q[group] for group in chip_groups])
                for chip_id, group in enumerate(chip_groups):
                    snapshots[f"chip_{chip_id}_{step+1}"] = particle_q[group].copy()
                snapshots[f"contents_{step + 1}"] = content_nodes.copy()
                positions = [nodes[indices] for indices in columns]
                vecs = [positions[i + 1] - positions[i] for i in range(3)]
                cosines = [np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) *
                           np.linalg.norm(b, axis=1), 1e-12) for a, b in zip(vecs, vecs[1:])]
                folded_fraction = float(np.mean((cosines[0] < -0.5) & (cosines[1] < -0.5)))
                row = {"step": step + 1, "time_s": (step + 1) * dt,
                       "bounds_min": nodes.min(axis=0).tolist(), "bounds_max": nodes.max(axis=0).tolist(),
                       "finite": bool(np.isfinite(nodes).all()),
                       "contents_min": content_nodes.min(axis=0).tolist(),
                       "contents_max": content_nodes.max(axis=0).tolist(),
                       "contents_finite": bool(np.isfinite(content_nodes).all()),
                       "chip_broken": chip_broken.copy(), "seam_peak_extension_m": seam_peak.copy(), "fold_cosine_mean": [float(c.mean()) for c in cosines],
                       "both_folds_reversed_fraction": folded_fraction}
                if gas is not None:
                    gas_stats, gas_moles = gas.stats.numpy(), gas.moles.numpy()
                    row["air"] = {"volume_m3": float(gas_stats[0] - gas_stats[3]), "gross_volume_m3": float(gas_stats[0]), "contents_volume_m3": float(gas_stats[3]), "inverted_tets": int(gas_stats[4]), "opening_area_m2": float(gas_stats[1]), "channel_area_m2": float(gas_stats[5]) if args.air_channel else None,
                                  "gauge_pressure_pa": float(gas_stats[2]), "moles": float(gas_moles[0])}
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
        (args.output / "newton-result.json").write_text(json.dumps({"break_events": events, "break_extension_m": args.break_extension, "seam_peak_extension_m": seam_peak, "damage_force": args.damage_force, "dt": dt, "air_enabled": args.air, "air_channel": args.air_channel, "air_opening_scale": args.air_opening_scale, "initial_crease_yield": args.initial_crease_yield, "dynamic_crease_yield": args.dynamic_crease_yield, "crease_count": crease_count, "substeps": args.substeps, "iterations": args.iterations, "self_radius": args.self_radius, "self_margin": args.self_margin, "collision_interval": args.collision_interval, "foldable_mouth": args.foldable_mouth, "contents_stiffness": args.contents_stiffness, "contents_damping": args.contents_damping, "folded": args.folded, "clip": args.clip,
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
