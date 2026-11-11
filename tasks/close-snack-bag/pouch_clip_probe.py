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
parser.add_argument("--fixture-seat", action="store_true", help="Move open clip with a scripted kinematic fixture before release")
parser.add_argument("--force-fixture", action="store_true", help="Move clip by bounded external wrench")
parser.add_argument("--press-fixture", action="store_true", help="Press fold with a controlled plate")
parser.add_argument("--seat-x", type=float, default=.18)
parser.add_argument("--seat-z", type=float, default=.012)
parser.add_argument("--fixture-move-s", type=float, default=.3)
parser.add_argument("--fixture-release-s", type=float, default=.4)
parser.add_argument("--fixture-max-force", type=float, default=1.)
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
parser.add_argument("--full-surface-contact", action="store_true")
parser.add_argument("--shape-gap", type=float, default=0.0005)
parser.add_argument("--physics", default="newton_vbd", choices=["newton_vbd"])
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args = parser.parse_args()
if args.coupled or args.proxy:
    parser.error("This probe supports VBD coupling only.")
if args.folded and not args.foldable_mouth:
    parser.error("--folded requires --foldable-mouth")
if args.initial_crease_yield is not None and not args.folded:
    parser.error("--initial-crease-yield requires --folded")
if args.dynamic_crease_yield is not None and args.initial_crease_yield is not None:
    parser.error("choose either dynamic or initial crease plasticity")

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
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
            physics_cfg.collision_cfg = NewtonCollisionPipelineCfg(soft_contact_margin=args.soft_contact_margin, enable_rigid_soft_full_surface_contact=args.full_surface_contact)
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
        contents_cfg = DeformableObjectCfg(prim_path="/World/contents", spawn=sim_utils.MeshCuboidCfg(
            size=(0.070, 0.060, 0.018), deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=NewtonDeformableBodyMaterialCfg(density=100.0, particle_radius=0.0001,
                k_mu=args.contents_stiffness, k_lambda=args.contents_stiffness, k_damp=args.contents_damping)),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.080, 0, 0.025)))
        contents = contents_cfg.class_type(contents_cfg)
        plate = None
        if args.press_fixture:
            plate_cfg = RigidObjectCfg(prim_path="/World/press_plate", spawn=sim_utils.CuboidCfg(
                size=(.035,.035,.004), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=.04)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(.145,.058,.05), rot=(0,0,0,1)))
            plate = plate_cfg.class_type(plate_cfg)
        sim.reset()
        packet.update(dt)
        contents.update(dt)
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
            sim.forward()
            from isaaclab_newton.physics.newton_manager import NewtonManager
            labels = list(NewtonManager.get_model().body_label)
            clip_body_ids = [labels.index("/World/clip/Geometry/clip"),
                             labels.index("/World/clip/Geometry/clip/upper")]
        rest, faces, grid = open_pouch(nx=args.nx, ny=args.ny, mouth_half_gap=mouth_gap, foldable_mouth=args.foldable_mouth)
        initial = packet.data.default_nodal_state_w.torch[0, :, :3].detach().cpu().numpy()
        if initial.shape != rest.shape or np.max(np.linalg.norm(initial - rest - [0, 0, 0.025], axis=1)) > 0.001:
            raise RuntimeError("Newton node ordering differs from authored mesh")
        if args.folded and not args.foldable_mouth:
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
        plate_fixture = None
        if plate is not None:
            from clip_force_fixture import ClipForceFixture
            from isaaclab_newton.physics.newton_manager import NewtonManager
            model_labels = list(NewtonManager.get_model().body_label)
            print("PLATE_LABELS",model_labels,flush=True)
            plate_id = next(i for i, name in enumerate(model_labels) if name == "/World/press_plate")
            plate_fixture = ClipForceFixture(NewtonManager.get_model(), plate_id, 1.0)
            plate_fixture.set([.145,.058,.05], True)
            NewtonManager.register_state_force_callback(plate_fixture.apply)
        fixture = None
        if clip and args.force_fixture:
            from clip_force_fixture import ClipForceFixture
            fixture = ClipForceFixture(NewtonManager.get_model(), clip_body_ids[0], args.fixture_max_force)
            fixture.set([args.clip_x, 0, args.clip_z], True)
            NewtonManager.register_state_force_callback(fixture.apply)
        snapshots, samples = {}, []
        total = round(args.duration / dt)
        sample_steps = {0, 59, 95, 119, 143, 191, 239, 479, total - 1}
        def column(x):
            i = round(x / 0.20 * args.nx)
            return np.concatenate([grid[0, i], grid[1, i]])
        columns = [column(x) for x in (0.14, 0.16, 0.18, 0.20)]
        for step in range(total):
            if plate_fixture is not None:
                t_plate = min(1., (step+1)*dt/.2)
                smooth_plate = t_plate*t_plate*(3-2*t_plate)
                z_plate = .05 + (.014-.05)*smooth_plate
                if step*dt >= args.fixture_release_s:
                    withdraw = min(1., (step*dt-args.fixture_release_s)/.5)
                    withdraw = withdraw*withdraw*(3-2*withdraw)
                    z_plate = .014 + (.12-.014)*withdraw
                # Keep the fixture well above the bag after withdrawal rather
                # than allowing the plate to fall back and hold the fold.
                plate_fixture.set([.145,.058,z_plate], True)
            if fixture is not None:
                # An external-force fixture holds the clip open and moves it.
                # The fixture is released after 0.4 s; this is not a robot demo.
                t = min(1., (step + 1)*dt / args.fixture_move_s)
                smooth = t*t*(3 - 2*t)
                pos = [args.clip_x + (args.seat_x - args.clip_x)*smooth, 0,
                       args.clip_z + (args.seat_z - args.clip_z)*smooth]
                fixture.set(pos, step*dt < args.fixture_release_s)
                target = opened if step*dt < args.fixture_move_s else torch.zeros_like(opened)
                clip.set_joint_position_target_index(target=target, joint_ids=ids)
            if clip and args.fixture_seat and step < 48:
                # Experimental kinematic fixture, not a YAM action. It holds the
                # jaws open while moving towards the folded mouth for 0.2 s.
                t = (step + 1) / 48
                smooth = t*t*(3 - 2*t)
                x = .225 + (args.clip_x - .225) * smooth
                z = .040 + (args.clip_z - .040) * smooth
                clip.write_joint_state_to_sim_index(position=opened, velocity=torch.zeros_like(opened), joint_ids=ids)
                clip.write_root_pose_to_sim_index(root_pose=torch.tensor([[x, 0, z, 0, 0, 1, 0]], device=args.device))
                clip.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6), device=args.device))
                sim.forward()
            if clip:
                clip.write_data_to_sim()
            sim.step(render=False)
            packet.update(dt)
            contents.update(dt)
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
                content_nodes = contents.data.nodal_pos_w.torch[0].detach().cpu().numpy()
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
                       "contents_finite": bool(np.isfinite(content_nodes).all()), "fold_cosine_mean": [float(c.mean()) for c in cosines],
                       "both_folds_reversed_fraction": folded_fraction}
                if gas is not None:
                    gas_stats, gas_moles = gas.stats.numpy(), gas.moles.numpy()
                    row["air"] = {"volume_m3": float(gas_stats[0] - gas_stats[3]), "gross_volume_m3": float(gas_stats[0]), "contents_volume_m3": float(gas_stats[3]), "inverted_tets": int(gas_stats[4]), "opening_area_m2": float(gas_stats[1]), "channel_area_m2": float(gas_stats[5]) if args.air_channel else None,
                                  "gauge_pressure_pa": float(gas_stats[2]), "moles": float(gas_moles[0])}
                if plate_fixture is not None:
                    row["plate_xyz"] = NewtonManager.get_state_0().body_q.numpy()[plate_id,:3].tolist()
                if clip:
                    body_pose = NewtonManager.get_state_0().body_q.numpy()[clip_body_ids]
                    clip_pos = body_pose[0, :3]
                    x0, y0, z0, w0 = body_pose[0, 3:7]
                    x1, y1, z1, w1 = body_pose[1, 3:7]
                    rel_y = w0*y1 - x0*z1 - y0*w1 + z0*x1
                    rel_w = x0*x1 + y0*y1 + z0*z1 + w0*w1
                    jaw = float(-((2 * np.arctan2(rel_y, rel_w) + np.pi) % (2*np.pi) - np.pi))
                    contacts = NewtonManager.get_contacts()
                    count = int(contacts.soft_contact_count.numpy()[0])
                    shape_ids, counts = np.unique(contacts.soft_contact_shape.numpy()[:min(count, contacts.soft_contact_max)], return_counts=True)
                    shape_labels = NewtonManager.get_model().shape_label
                    row["soft_contact_count"] = count
                    row["soft_contact_capacity"] = int(contacts.soft_contact_max)
                    row["soft_contacts_by_shape"] = {str(shape_labels[int(i)]): int(n) for i, n in zip(shape_ids, counts) if i >= 0}
                    # Width is an approximate world-Y extent of contacted particles,
                    # not yet a validated seal or continuous contact measurement.
                    n = min(count, contacts.soft_contact_max)
                    shape_values = contacts.soft_contact_shape.numpy()[:n]
                    features = contacts.soft_contact_indices.numpy()[:n]
                    positions = NewtonManager.get_state_0().particle_q.numpy()
                    bounds = {}
                    for shape_id in shape_ids:
                        if shape_id < 0 or "jaw" not in str(shape_labels[int(shape_id)]):
                            continue
                        particle_ids = np.unique(features[shape_values == shape_id].reshape(-1))
                        particle_ids = particle_ids[(particle_ids >= 0) & (particle_ids < len(positions))]
                        if len(particle_ids):
                            y = positions[particle_ids,1]
                            bounds[str(shape_labels[int(shape_id)])] = [float(y.min()), float(y.max())]
                    row["jaw_contact_y_bounds"] = bounds
                    row["clip_pose_source"] = "newton_state_body_q"
                    row["clip_body_poses_xyzw"] = body_pose.tolist()
                    row["clip_finite"] = bool(np.isfinite(clip_pos).all() and np.isfinite(jaw))
                    row["clip_plausible"] = bool(row["clip_finite"] and np.max(np.abs(clip_pos)) < 2 and abs(jaw) < 1)
                    row["clip_xyz"] = clip_pos.tolist() if np.isfinite(clip_pos).all() else None
                    row["jaw_radians"] = jaw if np.isfinite(jaw) else None
                samples.append(row)
                print("NEWTON_SURFACE", json.dumps(row), flush=True)
        np.savez_compressed(args.output / "surface-snapshots.npz", faces=faces, **snapshots)
        (args.output / "newton-result.json").write_text(json.dumps({"dt": dt, "full_surface_contact": args.full_surface_contact, "air_enabled": args.air, "air_channel": args.air_channel, "air_opening_scale": args.air_opening_scale, "initial_crease_yield": args.initial_crease_yield, "dynamic_crease_yield": args.dynamic_crease_yield, "crease_count": crease_count, "substeps": args.substeps, "iterations": args.iterations, "self_radius": args.self_radius, "self_margin": args.self_margin, "collision_interval": args.collision_interval, "foldable_mouth": args.foldable_mouth, "contents_stiffness": args.contents_stiffness, "contents_damping": args.contents_damping, "folded": args.folded, "clip": args.clip, "fixture_seat": args.fixture_seat, "force_fixture": args.force_fixture, "press_fixture": args.press_fixture, "fixture_move_s": args.fixture_move_s, "fixture_release_s": args.fixture_release_s, "fixture_max_force": args.fixture_max_force, "seat_x": args.seat_x, "seat_z": args.seat_z,
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
