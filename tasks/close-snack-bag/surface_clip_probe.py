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
parser.add_argument("--rest-only", action="store_true", help="Keep authored rest pose for no-fold control")
parser.add_argument("--no-clip", action="store_true", help="Keep clip away from the folded packet")
parser.add_argument("--clip-x", type=float, default=0.198, help="Initial clip hinge x in the folded placement test")
parser.add_argument("--clip-z", type=float, default=0.0215)
parser.add_argument("--fold-radius", type=float, default=0.003)
parser.add_argument("--crease-rest-fraction", type=float, default=0.0)
parser.add_argument("--thickness", type=float, default=0.001)
parser.add_argument("--contact-offset", type=float, default=0.001)
parser.add_argument("--youngs-modulus", type=float, default=2e6)
parser.add_argument("--bend-stiffness", type=float, default=0.001)
parser.add_argument("--linear-damping", type=float, default=1.0)
parser.add_argument("--bend-damping", type=float, default=0.02)
parser.add_argument("--hold-fixture", action="store_true", help="Press one mouth edge and initially hold the jaw open")
parser.add_argument("--nx", type=int, default=80)
parser.add_argument("--ny", type=int, default=16)
parser.add_argument("--mouth-half-gap", type=float, default=0.0015)
parser.add_argument("--taper-start", type=float, default=0.14)
parser.add_argument("--taper-end", type=float, default=0.16)
parser.add_argument("--board-height", type=float, default=0.0)
parser.add_argument("--contents-center-x", type=float, default=0.10)
parser.add_argument("--contents-length", type=float, default=0.075)
parser.add_argument("--edge-support-top", type=float, default=0.0)
parser.add_argument("--edge-support-width", type=float, default=0.04)
parser.add_argument("--fixture-offset", type=float, default=0.035)
parser.add_argument("--fixture-x", type=float, default=0.15)
parser.add_argument("--fixture-start-offset", type=float, default=0.06)
parser.add_argument("--fixture-descent-steps", type=int, default=120)
parser.add_argument("--force-insert", action="store_true", help="Guide the open clip using a temporary force controller")
parser.add_argument("--snapshots", action="store_true", help="Save sampled surface nodes and topology for offline checks")
parser.add_argument("--physics-hz", type=int, default=240)
parser.add_argument("--duration", type=float, default=None)
parser.add_argument("--solver-iterations", type=int, default=32)
parser.add_argument("--self-filter-distance", type=float, default=None)
parser.add_argument("--collision-updates", type=int, default=None)
parser.add_argument("--collision-multiplier", type=float, default=None)
parser.add_argument("--speculative-ccd", action="store_true")
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
    if args.force_insert and (not args.folded or not args.hold_fixture or args.no_clip):
        raise ValueError("--force-insert requires --folded --hold-fixture and a clip")
    args.output.mkdir(parents=True, exist_ok=True)
    usd = args.output / "packet-surface.usda"
    if usd.exists():
        raise FileExistsError(f"{usd} already exists; choose a fresh output directory")
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        nx, ny = (args.nx, args.ny) if args.folded else (20, 16)
        write_usd(usd, flat_mouth=True, nx=nx, ny=ny, mouth_half_gap=args.mouth_half_gap,
                  crease_rest_fraction=args.crease_rest_fraction, fold_radius=args.fold_radius,
                  taper_start=args.taper_start, taper_end=args.taper_end)
        if args.physics_hz < 240 or args.physics_hz % 240:
            raise ValueError("physics-hz must be an integer multiple of 240")
        dt = 1 / args.physics_hz
        scale = args.physics_hz // 240
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics=physics_cfg))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        if args.board_height:
            board = sim_utils.CuboidCfg(size=(0.14, 0.16, args.board_height),
                                       collision_props=sim_utils.CollisionPropertiesCfg())
            board.func("/World/board", board, translation=(0.07, 0, args.board_height / 2))
        if args.edge_support_top:
            support = sim_utils.CuboidCfg(size=(args.edge_support_width, 0.018, args.edge_support_top),
                                          collision_props=sim_utils.CollisionPropertiesCfg())
            support.func("/World/edge_support", support,
                         translation=(args.fixture_x, 0.069, args.edge_support_top / 2))
        contents = sim_utils.CuboidCfg(size=(args.contents_length, 0.08, 0.014),
                                       collision_props=sim_utils.CollisionPropertiesCfg())
        contents.func("/World/contents", contents, translation=(args.contents_center_x, 0, 0.019 + args.board_height))
        packet = None
        if not args.empty:
            material = PhysxSurfaceDeformableBodyMaterialCfg(
                density=900, youngs_modulus=args.youngs_modulus, poissons_ratio=0.35,
                surface_thickness=args.thickness, surface_bend_stiffness=args.bend_stiffness,
                static_friction=0.5, dynamic_friction=0.4,
                elasticity_damping=0.02, bend_damping=args.bend_damping)
            cfg = DeformableObjectCfg(prim_path="/World/packet", spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd.resolve()),
                deformable_props=PhysxDeformableBodyPropertiesCfg(self_collision=True,
                    solver_position_iteration_count=args.solver_iterations,
                    linear_damping=args.linear_damping, self_collision_filter_distance=args.self_filter_distance,
                    collision_pair_update_frequency=args.collision_updates,
                    collision_iteration_multiplier=args.collision_multiplier,
                    enable_speculative_c_c_d=True if args.speculative_ccd else None), physics_material=material),
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0.003 + args.board_height)))
            packet = cfg.class_type(cfg)
            sim_utils.modify_collision_properties(
                "/World/packet", sim_utils.CollisionPropertiesCfg(contact_offset=args.contact_offset, rest_offset=0.0))
            from isaaclab.sim.utils.stage import get_current_stage
            for prim in get_current_stage().Traverse():
                if str(prim.GetPath()).startswith("/World/packet"):
                    values = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                              if "selfCollision" in a.GetName() or "collisionPairUpdate" in a.GetName()}
                    if values:
                        print("DEFORMABLE_COLLISION_CONFIG", str(prim.GetPath()), values, flush=True)
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
                init_state=RigidObjectCfg.InitialStateCfg(pos=(args.fixture_x, 0.069, 0.06 + args.board_height)))
            fixture = fixture_cfg.class_type(fixture_cfg)
        sim.reset()
        ids, _ = clip.find_joints(["clip_hinge"], preserve_order=True)
        q = torch.tensor([[-0.55 if args.folded else -0.15]], dtype=torch.float32, device=args.device)
        clip.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), joint_ids=ids)
        clip.set_joint_position_target_index(target=q if fixture else torch.zeros_like(q), joint_ids=ids)
        mouth_nodes = None
        neighbor_edges = None
        fold_columns = None
        if args.folded and packet:
            from scipy.spatial import cKDTree
            rest, faces = packet_mesh(nx=nx, ny=ny, flat_mouth=True, mouth_half_gap=args.mouth_half_gap,
                                      taper_start=args.taper_start, taper_end=args.taper_end)
            offset = np.array([0, 0, 0.003 + args.board_height], dtype=np.float32)
            rest_shape = ((1 - args.crease_rest_fraction) * rest +
                          args.crease_rest_fraction * folded_positions(rest, radius=args.fold_radius))
            rest_world = rest_shape + offset
            state = packet.data.default_nodal_state_w.torch.clone()
            default_nodes = state[0, :, :3].detach().cpu().numpy()
            # Match by (x, y) first, then by lower/upper skin. A gravity step
            # can exceed half of the very narrow gap, so a 3-D nearest-node
            # query alone can map both skins to the same authored vertex.
            span = (nx + 1) * (ny + 1)
            if len(default_nodes) == len(rest_world):
                direct_distance = np.linalg.norm(default_nodes - rest_world, axis=1)
                print("REST_MAPPING_DIRECT", "max", float(direct_distance.max()),
                      "median", float(np.median(direct_distance)), flush=True)
            if len(default_nodes) == len(rest_world) and np.max(direct_distance) <= 6e-4:
                mapping = np.arange(len(default_nodes))
            elif args.crease_rest_fraction:
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import min_weight_full_bipartite_matching
                distances, candidates = cKDTree(rest_world).query(default_nodes, k=12)
                n = len(default_nodes)
                graph = coo_matrix((distances.ravel() + 1e-9,
                                    (np.repeat(np.arange(n), 12), candidates.ravel())), shape=(n, len(rest))).tocsr()
                rows, cols = min_weight_full_bipartite_matching(graph)
                mapping = np.empty(n, dtype=int)
                mapping[rows] = cols
            else:
                distance_xy, sites = cKDTree(rest_world[:span, :2]).query(default_nodes[:, :2])
                if (len(sites) != len(rest) or np.max(distance_xy) > 5e-4 or
                        not np.all(np.bincount(sites, minlength=span) == 2)):
                    raise RuntimeError("Cannot establish a unique pair of surface nodes for each (x, y) site")
                mapping = np.empty(len(sites), dtype=int)
                order = np.lexsort((default_nodes[:, 2], sites))
                ordered_sites = sites[order].reshape(-1, 2)
                if not np.all(ordered_sites[:, 0] == ordered_sites[:, 1]):
                    raise RuntimeError("Surface-node pairing failed")
                mapping[order[::2]] = ordered_sites[:, 0]
                mapping[order[1::2]] = ordered_sites[:, 1] + span
            if np.max(np.linalg.norm(default_nodes - rest_world[mapping], axis=1)) > 6e-4:
                raise RuntimeError("Surface-node rest positions shifted too far during initialization")
            print("REST_MAPPING", "identity_fraction", float(np.mean(mapping == np.arange(len(mapping)))), flush=True)
            folded = (rest_shape if args.rest_only else folded_positions(rest, radius=args.fold_radius))[mapping] + offset
            mouth_nodes = rest[mapping, 0] >= 0.15999
            neighbor_edges = {tuple(sorted((int(triangle[i]), int(triangle[(i + 1) % 3]))))
                              for triangle in faces for i in range(3)}
            def edge_indices(x_value):
                indices = np.flatnonzero(np.isclose(rest[mapping, 0], x_value, atol=1e-6))
                return indices[np.argsort((mapping[indices] // span) * (ny + 1) + mapping[indices] % (ny + 1))]
            fold_columns = [edge_indices(x) for x in (0.14, 0.16, 0.18, 0.20)]
            if any(len(column) != 2 * (ny + 1) for column in fold_columns):
                raise RuntimeError("Mesh resolution must include node columns at 0.14, 0.16, 0.18, 0.20 m")
            if not args.rest_only:
                packet.write_nodal_pos_to_sim_index(torch.tensor(folded[None], device=args.device))
                packet.write_nodal_velocity_to_sim_index(torch.zeros_like(state[..., 3:]))
            if not args.no_clip:
                start_x = args.clip_x + (0.045 if args.force_insert else 0.0)
                clip.write_root_pose_to_sim_index(root_pose=torch.tensor([[start_x, 0, args.clip_z, 0, 0, 1, 0]],
                                                  device=args.device, dtype=torch.float32))
                clip.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=args.device))
        if fixture:
            fixture.write_root_pose_to_sim_index(root_pose=torch.tensor(
                [[args.fixture_x, 0.069, args.fixture_start_offset + args.board_height, 0, 0, 0, 1]],
                dtype=torch.float32, device=args.device))
        sim.forward()
        rows, snapshots = [], {}
        total_steps = round(args.duration * args.physics_hz) if args.duration else (1440 if fixture else 960) * scale
        sample_steps = {0, *(n * scale - 1 for n in (60, 120, 240, 480, 720, 960, 1200, 1440))}
        sample_steps.add(total_steps - 1)
        for step in range(total_steps):
            base_step = step / scale
            if fixture:
                # A kinematic fixture presses an outer edge; the hinged jaw is
                # commanded open for one second, then released. Neither is a robot.
                if base_step < args.fixture_descent_steps:
                    z = args.fixture_start_offset - (args.fixture_start_offset - args.fixture_offset) * base_step / args.fixture_descent_steps
                elif base_step < 480:
                    z = args.fixture_offset
                elif base_step < 600:
                    z = args.fixture_offset + (0.06 - args.fixture_offset) * (base_step - 480) / 120
                else:
                    z = 0.06
                z += args.board_height
                fixture.write_root_pose_to_sim_index(root_pose=torch.tensor([[args.fixture_x, 0.069, z, 0, 0, 0, 1]],
                                                     dtype=torch.float32, device=args.device))
                if step == (320 if args.force_insert else 240) * scale:
                    clip.set_joint_position_target_index(target=torch.zeros_like(q), joint_ids=ids)
            if args.force_insert and base_step < 480:
                # Temporary external-wrench fixture emulates holding and
                # inserting the clip. It is removed entirely at step 480; no
                # object pose is written after reset, and this is not a robot.
                pose = clip.data.root_link_pose_w.torch[0]
                velocity = clip.data.root_link_vel_w.torch[0]
                progress = np.clip((base_step - 80) / 240, 0, 1)
                desired = torch.tensor([args.clip_x + 0.045 * (1 - progress), 0, args.clip_z],
                                       device=args.device, dtype=torch.float32)
                force = 30 * (desired - pose[:3]) - 1.5 * velocity[:3]
                force = force + torch.tensor([0, 0, 0.2354], device=args.device)
                force = force * torch.clamp(2.0 / torch.linalg.vector_norm(force).clamp(min=1e-6), max=1.0)
                x, y, z, w = pose[3:7]
                orientation_error = 2 * torch.stack((y, -x, w)) * torch.where(z < 0, -1.0, 1.0)
                torque = 0.04 * orientation_error - 0.004 * velocity[3:6]
                torque = torque * torch.clamp(0.025 / torch.linalg.vector_norm(torque).clamp(min=1e-6), max=1.0)
                clip.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=force.view(1, 1, 3), torques=torque.view(1, 1, 3),
                    body_ids=[clip.body_names.index("clip")], is_global=True)
            elif args.force_insert and step == 480 * scale:
                clip.permanent_wrench_composer.reset()
            clip.write_data_to_sim()
            sim.step(render=False)
            clip.update(dt)
            if packet:
                packet.update(dt)
            if step in sample_steps:
                root = clip.data.root_link_pose_w.torch[0].detach().cpu().numpy()
                row = {"step": step + 1, "time_s": (step + 1) * dt,
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
                        columns = [nodes[indices] for indices in fold_columns]
                        distances = np.linalg.norm(columns[1] - columns[3], axis=1)
                        row["edge_distance_mean_m"] = float(distances.mean())
                        row["edge_distance_max_m"] = float(distances.max())
                        vectors = [columns[i + 1] - columns[i] for i in range(3)]
                        cosines = [np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) *
                                   np.linalg.norm(b, axis=1), 1e-12)
                                   for a, b in zip(vectors, vectors[1:])]
                        row["fold_cosine_mean"] = [float(c.mean()) for c in cosines]
                        reversed_folds = (cosines[0] < -0.5) & (cosines[1] < -0.5)
                        row["both_folds_reversed_fraction"] = float(np.mean(reversed_folds))
                        # Two skins and matched width samples. Count only
                        # contiguous intervals where both skins pass at both ends.
                        passing_columns = reversed_folds.reshape(2, ny + 1).all(axis=0)
                        passing_intervals = passing_columns[:-1] & passing_columns[1:]
                        longest = run = 0
                        for passed in passing_intervals:
                            run = run + 1 if passed else 0
                            longest = max(longest, run)
                        row["reversed_fold_contiguous_width_m"] = longest * 0.16 / ny
                        # Node-nearness is a warning, not a triangle-intersection test.
                        close_pairs = cKDTree(nodes).query_pairs(r=args.thickness * 0.5, output_type="ndarray")
                        row["nonneighbor_node_pairs_under_half_thickness"] = sum(
                            tuple(sorted((int(mapping[a]), int(mapping[b])))) not in neighbor_edges
                            for a, b in close_pairs)
                        if args.snapshots:
                            snapshots[f"nodes_{step + 1}"] = nodes[np.argsort(mapping)].copy()
                rows.append(row)
                print("CLIP_PROBE", json.dumps(row), flush=True)
        (args.output / "surface-clip.json").write_text(json.dumps(
            {"note": "directly initialized clip and optional folded nodes; not a robot demonstration",
             "empty": args.empty, "folded": args.folded, "no_clip": args.no_clip,
             "rest_only": args.rest_only,
             "hold_fixture": args.hold_fixture,
             "force_insert": args.force_insert,
             "clip_x": args.clip_x, "clip_z": args.clip_z, "fold_radius": args.fold_radius,
             "crease_rest_fraction": args.crease_rest_fraction,
             "surface_thickness_m": args.thickness, "contact_offset_m": args.contact_offset,
             "solver_iterations": args.solver_iterations,
             "self_filter_distance_m": args.self_filter_distance,
             "collision_updates": args.collision_updates, "collision_multiplier": args.collision_multiplier,
             "speculative_ccd": args.speculative_ccd,
             "youngs_modulus_pa": args.youngs_modulus, "surface_bend_stiffness": args.bend_stiffness,
             "linear_damping": args.linear_damping, "bend_damping": args.bend_damping,
             "mouth_half_gap_m": args.mouth_half_gap, "board_height_m": args.board_height, "nx": nx, "ny": ny,
             "contents_center_x_m": args.contents_center_x, "contents_length_m": args.contents_length,
             "taper_start_m": args.taper_start, "taper_end_m": args.taper_end,
             "edge_support_top_m": args.edge_support_top,
             "edge_support_width_m": args.edge_support_width,
             "fixture_offset_m": args.fixture_offset, "fixture_x_m": args.fixture_x,
             "fixture_start_offset_m": args.fixture_start_offset,
             "fixture_descent_steps": args.fixture_descent_steps,
             "dt": dt, "samples": rows}, indent=2))
        if snapshots:
            np.savez_compressed(args.output / "surface-snapshots.npz", faces=faces, **snapshots)


if __name__ == "__main__":
    main()
