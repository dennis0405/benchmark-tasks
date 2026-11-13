"""Record contact-driven YAM folding experiments; no object pose writes."""
import argparse
import json
import sys
import time
import shutil
import signal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment, read_object_states, export_render_geometry
from fold_metrics import fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--drive", action="store_true")
    parser.add_argument("--coupling", choices=("one_way", "proxy", "unified"), default="one_way")
    parser.add_argument("--ready-start", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--checkpoint-every",type=int,default=120)
    parser.add_argument("--material-profile",choices=("derived","permanent","laminate","paper_matched"),default="derived")
    parser.add_argument("--stage",choices=("fold","fold_hold","regrasp","seal","clip_grasp","rest"),default="fold")
    parser.add_argument("--self-filter-rings",type=int,choices=(0,2),default=0)
    parser.add_argument("--self-radius-mm",type=float,default=.5)
    parser.add_argument("--rebase-z",action="store_true")
    parser.add_argument("--hz",type=int,choices=(20,60),default=60)
    parser.add_argument("--physics-hz",type=int,choices=(600,1200,2400),default=None)
    parser.add_argument("--check-substeps",action="store_true")
    parser.add_argument("--side-access",action="store_true")
    parser.add_argument("--rigid-cpu",action="store_true")
    parser.add_argument("--workspace",choices=("original","raised_forward","raised_flat"),default="original")
    parser.add_argument("--clip-style",choices=("standard","thin_lower","thin_strong","thin_broad","thin_aligned"),default="standard")
    parser.add_argument("--override-clip-spring",action="store_true")
    parser.add_argument("--override-clip-geometry",action="store_true")
    parser.add_argument("--reset-tool",action="store_true",
                        help="At a component reset only, restore the selected tool's authored parking pose")
    parser.add_argument("--table-insert",action="store_true")
    parser.add_argument("--insert-mode",choices=("overhang","wedge"),default="overhang")
    parser.add_argument("--force-readout",choices=("last","mean"),default="last")
    parser.add_argument("--record-force-means",action="store_true")
    parser.add_argument("--release-style",choices=("wide","sliding"),default="sliding")
    parser.add_argument("--clip-component-seed",type=Path,default=None)
    parser.add_argument("--component-start",choices=("clip","centre_press","raw_release","edge_drag","press_drag","insert","prepare"),default="clip")
    parser.add_argument("--mouth-support",choices=("narrow","full"),default="narrow")
    parser.add_argument("--support-extension-mm",type=float,default=0.)
    parser.add_argument("--support-hold",choices=("static","admittance"),default="static")
    parser.add_argument("--press-floor-mm",type=float,default=2.5)
    parser.add_argument("--board-friction",type=float,default=1.)
    parser.add_argument("--finger-friction",type=float,default=None)
    parser.add_argument("--override-contact-materials",action="store_true")
    args = parser.parse_args()
    if args.reset_tool and not args.clip_component_seed:
        parser.error("reset-tool is only for an explicit clip component seed")
    if args.stage=="clip_grasp" and not (args.clip and args.drive):
        parser.error("clip_grasp requires --clip --drive")
    if args.table_insert and (args.clip_style not in ("thin_lower","thin_strong","thin_broad","thin_aligned") or args.workspace!="raised_flat"):
        parser.error("table-insert requires the thin lower blade and raised_flat workspace")
    if args.clip_component_seed:
        if not (args.stage=="seal" and args.table_insert and args.clip and args.drive):
            parser.error("a clip component seed requires --stage seal --table-insert --clip --drive")
        seed_meta=json.loads((args.clip_component_seed/"clip-entry.json").read_text())
        for key in ("workspace","clip_style","material_profile","hz","physics_hz","coupling"):
            if seed_meta["arguments"][key]!=getattr(args,key):
                if key=="clip_style" and args.override_clip_geometry and args.reset_tool and {
                    seed_meta["arguments"][key],args.clip_style}.issubset({"thin_lower","thin_strong","thin_broad","thin_aligned"}):
                    continue
                if key=="clip_style" and args.override_clip_spring and {
                    seed_meta["arguments"][key],args.clip_style}=={"thin_lower","thin_strong"}:
                    continue
                parser.error(f"checkpoint scene mismatch: {key}")
        for key,default in (("mouth_support","narrow"),("support_extension_mm",0.),("rebase_z",False)):
            if seed_meta["arguments"].get(key,default)!=getattr(args,key):
                parser.error(f"checkpoint support geometry mismatch: {key}")
        for key,default in (("board_friction",1.),("finger_friction",None),("self_filter_rings",0),("self_radius_mm",.5)):
            if seed_meta["arguments"].get(key,default)!=getattr(args,key) and not args.override_contact_materials:
                parser.error(f"checkpoint material change requires --override-contact-materials: {key}")
    interrupted = [False]
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output / "sources"
    sources.mkdir()
    for source in (Path(__file__), Path(__file__).with_name("fold_controller.py"),
                   Path(__file__).resolve().parents[1]/"deformable_scene.py",
                   Path(__file__).resolve().parents[1]/"pouch_geometry.py",
                   Path(__file__).with_name("film_material.py"),
                   Path(__file__).with_name("stack_grasp.py"),
                   Path(__file__).resolve().parents[1]/"fold_metrics.py",
                   Path(__file__).with_name("assets")/"yam_ready_pose.json"):
        shutil.copy2(source, sources/source.name)
    if args.workspace!="original":
        source=Path(__file__).with_name("assets")/f"yam_{args.workspace}_pose.json"
        shutil.copy2(source,sources/source.name)
    if args.clip:
        # Eagerly import optional controllers before stepping. A later rsync
        # must not change a running experiment at its first delayed import.
        import clip_controller
        import check_rigid_tool
        import stack_grasp
        import plan_clip_poses
        for source in (Path(__file__).resolve().parents[1]/"deformable_clip.py",
                       Path(__file__).with_name("tool_feedback.py"),
                       Path(__file__).with_name("clip_controller.py"),
                       Path(__file__).with_name("check_rigid_tool.py"),
                       Path(__file__).with_name("gpu_contact_checks.py"),
                       Path(__file__).with_name("stack_grasp.py"),
                       Path(__file__).with_name("plan_clip_poses.py"),
                       Path(__file__).with_name("assets")/"clip_spring_v2.xml"):
            shutil.copy2(source, sources/source.name)
        if args.clip_style!="standard":
            source=Path(__file__).with_name("assets")/{"thin_lower":"clip_thin_lower.xml","thin_strong":"clip_thin_strong.xml","thin_broad":"clip_thin_broad.xml","thin_aligned":"clip_thin_aligned.xml"}[args.clip_style]
            shutil.copy2(source,sources/source.name)
    (args.output/"arguments.json").write_text(json.dumps(vars(args), default=str, indent=2))
    if args.clip_component_seed:
        for name in ("clip-entry.npz","clip-entry.json"):
            shutil.copy2(args.clip_component_seed/name,sources/name)
        shutil.copy2(args.clip_component_seed/"trace.json",sources/"seed-trace.json")
        shutil.copy2(Path(__file__).with_name("resume_clip_component.py"),sources/"resume_clip_component.py")
    started = time.monotonic()
    sim = make_environment(coupling=args.coupling, ready_start=args.ready_start,
                           with_clip=args.clip, material_profile=args.material_profile, control_hz=args.hz,
                           side_access=args.mouth_support=="narrow" and (args.side_access or args.stage in ("regrasp","seal")),
                           physics_hz=args.physics_hz,rigid_cpu=args.rigid_cpu,workspace=args.workspace,
                           clip_style=args.clip_style,support_extension=args.support_extension_mm/1000.,
                           board_friction=args.board_friction,finger_friction=args.finger_friction,
                           self_filter_rings=args.self_filter_rings,self_radius=args.self_radius_mm/1000.,
                           rebase_z=args.rebase_z)
    timings = dict(build_s=time.monotonic()-started, driver_tick_s=0., readback_and_metrics_s=0., recording_s=0.)
    entity = sim.scene.by_name["pouch"]
    export_render_geometry(sim, args.output)
    session = controller = drive = None
    component_reset=None
    if args.drive:
        import stack_grasp
        from tests.fixtures.grasp import open_session, tick
        from fold_controller import FoldController
        shutil.copy2(Path(__file__).with_name("finger_geometry.py"),sources/"finger_geometry.py")
        session = open_session(sim, 9764)
        if args.clip_component_seed:
            from resume_clip_component import restore
            component_reset=restore(sim,session,args.clip_component_seed,reset_tool=args.reset_tool)
            component_reset["clip_asset_change"]=dict(from_style=seed_meta["arguments"]["clip_style"],
                                                     to_style=args.clip_style)
        controller = FoldController(sim, session)
        controller.table_insert=args.table_insert
        controller.insert_mode=args.insert_mode
        controller.release_style=args.release_style
        controller.support_hold_enabled=args.support_hold=="admittance"
        controller.press_floor_m=args.press_floor_mm/1000.
        if args.stage=="fold_hold":
            controller.after_fold="hold"
        elif args.stage in ("regrasp","seal"):
            controller.after_fold=args.stage
        if args.clip_component_seed:
            from clip_controller import seal_fold,centre_press_component,raw_release_component,edge_drag_component,press_drag_component,insert_component,sequential_component
            controller.layout=component_reset["layout"]
            controller.marks["left_press_hold"]=True
            if component_reset.get("left_cleared_for_clip"):
                controller.marks["left_cleared_for_clip"]=True
                controller.marks["left_press_hold"]=False
            controller.support_holds=component_reset.get("support_holds",{})
            if not controller.support_holds and not controller.marks.get("left_cleared_for_clip"):
                controller.start_support("left")
            controller.phase="clip_component_reset"
            stages={"clip":seal_fold,"centre_press":centre_press_component,"raw_release":raw_release_component,"edge_drag":edge_drag_component,"press_drag":press_drag_component,"insert":insert_component,"prepare":sequential_component}
            drive=stages[args.component_start](controller)
        elif args.stage=="clip_grasp":
            from clip_controller import clip_grasp_drive
            drive=clip_grasp_drive(controller)
        elif args.stage=="rest":
            controller.phase="rest"
            for panel in session.arms.values():panel.grip_request=.9052631579
            drive=controller.wait((args.frames+10)*60/args.hz)
        else:
            drive = controller.drive()
    snapshots, trace = {}, []
    rest_initial=entity._rest.numpy()[entity.rows].copy()
    stepping=entity.stepping_model if entity.stepping_model is not None else sim.model
    edges=stepping.edge_indices.numpy()[entity.rows,2:4]
    midpoints=sim.state.particle_q.numpy()[edges].mean(axis=1)
    q_initial=sim.state.particle_q.numpy()[entity.particles]
    mouth_x=float(np.median(q_initial[entity.grid[:,-1,:],0]))
    neck_edges=(midpoints[:,0]<mouth_x+.070)&(abs(midpoints[:,1])<.07)
    watcher=None
    if args.check_substeps:
        from substep_geometry import SubstepGeometry
        shutil.copy2(Path(__file__).with_name("substep_geometry.py"),sources/"substep_geometry.py")
        shutil.copy2(Path(__file__).with_name("round_contact_checks.py"),sources/"round_contact_checks.py")
        shutil.copy2(Path(__file__).with_name("gpu_contact_checks.py"),sources/"gpu_contact_checks.py")
        watcher=SubstepGeometry(sim,entity)
        sim.pipeline.hooks.add("post_solve",watcher.check,name="independent-film-geometry",cadence=1)
        sim.pipeline.invalidate_graph()
    average=None
    if args.force_readout=="mean" or args.record_force_means:
        from contact_average import ContactAverage
        shutil.copy2(Path(__file__).with_name("contact_average.py"),sources/"contact_average.py")
        average=ContactAverage(sim.pipeline.program.solver)
        if args.force_readout=="mean":sim.snack_force_average=average
        sim.pipeline.hooks.add("post_solve",average.observe,name="substep-contact-average",cadence=1)
        sim.pipeline.invalidate_graph()
    controller_completed = False
    clip_entry_saved=False
    saved_stages=set()
    def save_component_seed(directory,step):
        directory.mkdir(parents=True,exist_ok=True)
        seed={name:snapshots[f"{name}_{step:05d}"] for name in ("q","b","j","d","v","r")}
        seed["targets"]=sim.control.joint_target_q.numpy().copy()
        seed.update(faces=entity.faces,grid=entity.grid,edge_rows=entity.rows)
        np.savez_compressed(directory/"clip-entry.npz",**seed)
        (directory/"clip-entry.json").write_text(json.dumps(dict(
            step=step,layout=controller.layout,arguments=vars(args),
            support_holds=controller.support_holds,source=str(args.output),
            left_cleared_for_clip=controller.marks.get("left_cleared_for_clip",False),
            phase=controller.phase,
            scope="Observed physical component reset; velocities and plastic rest preserved, solver caches rebuilt; not an uninterrupted episode"),default=str,indent=2))
        if directory!=args.output:
            (directory/"trace.json").write_text(json.dumps([trace[-1]]))
    def save_record():
        before=time.monotonic()
        temporary=args.output/"states.tmp.npz"
        np.savez_compressed(temporary,faces=entity.faces,grid=entity.grid,
                            edge_rows=entity.rows,neck_edge_mask=neck_edges,
                            edge_rest_initial=rest_initial,
                            edge_rest_current=entity._rest.numpy()[entity.rows],**snapshots)
        temporary.replace(args.output/"states.npz")
        temporary=args.output/"trace.tmp.json"
        temporary.write_text(json.dumps(trace));temporary.replace(args.output/"trace.json")
        timings["recording_s"]+=time.monotonic()-before
    for step in range(args.frames + 1):
        if interrupted[0]:
            break
        if step:
            before=time.monotonic()
            if drive:
                try:
                    next(drive)
                except StopIteration:
                    controller_completed = True
                    break
                controller.update_support_holds()
                if average:average.reset()
                tick(session, 1)
            else:
                if average:average.reset()
                sim.run(1).sync()
            timings["driver_tick_s"]+=time.monotonic()-before
        before=time.monotonic()
        state = read_object_states(sim)
        snapshots[f"q_{step:05d}"] = state["pouch"]
        snapshots[f"b_{step:05d}"] = state["bodies"]
        snapshots[f"j_{step:05d}"] = sim.state.joint_q.numpy().copy()
        snapshots[f"d_{step:05d}"] = sim.state.joint_qd.numpy().copy()
        snapshots[f"v_{step:05d}"] = sim.state.particle_qd.numpy()[entity.particles].copy()
        trace.append(dict(step=step, finite=bool(sim.state_is_finite()),
                          phase=controller.phase if controller else "rest",
                          z_min=float(state["pouch"][:, 2].min()),
                          z_max=float(state["pouch"][:, 2].max())))
        trace[-1]["fold"] = fold_metrics(state["pouch"], entity.grid, [(34, 28, 40)])
        rest=entity._rest.numpy()[entity.rows]
        snapshots[f"r_{step:05d}"]=rest.copy()
        change=np.abs(np.arctan2(np.sin(rest-rest_initial),np.cos(rest-rest_initial)))[neck_edges]
        trace[-1]["neck_plastic_rest_change"]=dict(edges_above_005_rad=int((change>.05).sum()),
            max_rad=float(change.max(initial=0.)),p95_rad=float(np.quantile(change,.95)) if len(change) else 0.)
        if not trace[-1]["finite"]:
            if controller:
                controller.failure="non-finite simulation state"
            break
        if watcher:
            trace[-1]["substep_geometry"]=watcher.report()
        if args.clip:
            from scipy.spatial.transform import Rotation
            clip_entity=sim.scene.by_name["clip"]
            root, upper=(state["bodies"][body] for body in clip_entity.bodies)
            trace[-1]["clip_root_pose"] = root.tolist()
            trace[-1]["clip_angle_rad"] = float((Rotation.from_quat(root[3:]).inv()*
                                                  Rotation.from_quat(upper[3:])).as_rotvec()[1])
        if step == 0:
            import newton
            from check_native_clip import inspect_state
            c = np.load(args.output/"colliders.npz")
            owners = c["body"].copy()
            nbody = len(state["bodies"])
            owners[owners<0] = nbody
            eligible = (c["kind"]==int(newton.GeoType.BOX)) & (
                (c["flags"] & int(newton.ShapeFlags.COLLIDE_PARTICLES))!=0)
            owners[~eligible] = -1
            shape = dict(shape_body=owners.tolist(), shape_local=c["local"], shape_scale=c["scale"])
            bodies=np.vstack([state["bodies"], [0,0,0,0,0,0,1]])
            preflight=inspect_state(state["pouch"], entity.faces, bodies, shape)
            (args.output/"preflight.json").write_text(json.dumps(preflight,indent=2))
            if any(row["penetrating_triangles"] for row in preflight):
                save_record()
                raise RuntimeError("Initial pouch/box intersection; reject before robot motion")
            if args.clip:
                from check_rigid_tool import inspect_tool
                tool_check=inspect_tool(sim)
                if tool_check["intersections"] and args.clip_component_seed:
                    strict=tool_check
                    tool_check=inspect_tool(sim,resting_fixture_tolerance=1e-5)
                    tool_check["strict_without_rack_inset"]=strict
                (args.output/"rigid-tool-preflight.json").write_text(json.dumps(tool_check,indent=2))
                if tool_check["intersections"]:
                    save_record()
                    raise RuntimeError("Initial rigid clip/BOX intersection; reject before motion")
        if controller:
            target_q=sim.control.joint_target_q.numpy()
            trace[-1]["robot_targets"]={a:dict(joints=target_q[sim.arm(a).arm_coords].tolist(),
                gripper_joint_targets=target_q[sim.scene.embodiment.arms[a].gripper_coords].tolist(),
                grip_request=(None if controller.session.arms[a].grip_request is None else
                              float(controller.session.arms[a].grip_request))) for a in ("left","right")}
            trace[-1]["jaws"] = {a:controller.takes._fold_jaws(controller.film,a).tolist()
                                 for a in ("left", "right")}
            trace[-1]["hand_tracking"]={
                a:dict(tcp_m=panel.ik.ee_pose(sim.state)[0].tolist(),
                       cursor_m=np.asarray(panel.cursor_pos).tolist(),
                       moving=bool(panel.dragging))
                for a,panel in controller.session.arms.items()}
            solver = sim.pipeline.program.solver
            if hasattr(solver, "shape_contact_forces"):
                forces = solver.shape_contact_forces()
                owners = sim.model.shape_body.numpy()
                trace[-1]["fixture_soft_force_world_N"]={
                    str(sim.model.shape_label[i]):forces[i].tolist() for i in np.flatnonzero(owners<0)}
                if average:
                    mean_forces=average.forces()
                    trace[-1]["force_window_s"]=float(average.duration.numpy()[0])
                    trace[-1]["all_shape_mean_force_world_N"]=mean_forces.sum(0).tolist()
                    trace[-1]["body_mean_force_world_N"]={str(sim.model.body_label[body]):mean_forces[owners==body].sum(0).tolist()
                        for body in np.unique(owners[owners>=0])}
                    trace[-1]["finger_mean_force_world_N"]={a:[mean_forces[np.isin(owners,side)].sum(0).tolist()
                        for side in index.finger_bodies] for a,index in sim.scene.embodiment.arms.items()}
                    trace[-1]["fixture_mean_force_world_N"]={str(sim.model.shape_label[i]):mean_forces[i].tolist()
                                                              for i in np.flatnonzero(owners<0)}
                trace[-1]["finger_soft_force_N"] = {
                    a:[float(np.linalg.norm(forces[np.isin(owners, side)].sum(axis=0)))
                       for side in index.finger_bodies]
                    for a,index in sim.scene.embodiment.arms.items()}
                trace[-1]["finger_soft_force_world_N"]={
                    a:[forces[np.isin(owners,side)].sum(0).tolist() for side in index.finger_bodies]
                    for a,index in sim.scene.embodiment.arms.items()}
                trace[-1]["support_hold"]={a:dict(filtered_force_z_N=h["filtered"],floor_cursor_z_m=h["floor_z"])
                                           for a,h in controller.support_holds.items()}
                if args.clip:
                    clip_forces={}
                    for label,body in zip(("lower","upper"),clip_entity.bodies):
                        force=forces[owners==body].sum(axis=0)
                        normal=Rotation.from_quat(state["bodies"][body,3:]).apply([0,0,1])
                        clip_forces[label]=dict(world_N=force.tolist(),normal_N=float(force@normal))
                    trace[-1]["clip_film_force"]=clip_forces
                    if average:
                        trace[-1]["clip_mean_force_world_N"]={label:mean_forces[owners==body].sum(0).tolist()
                            for label,body in zip(("lower","upper"),clip_entity.bodies)}
        timings["readback_and_metrics_s"]+=time.monotonic()-before
        if controller and hasattr(controller,"layout"):
            geometry=watcher.report() if watcher else None
            clean=geometry and not (geometry["transverse_edge_crossings"] or geometry["film_box_intersections"] or geometry["film_round_intersections"])
            phase=controller.phase
            if clean and phase not in saved_stages and phase in (
                    "release_left","release_right","press_left","press_right",
                    "lift_press_right","fold_drag_to_edge","table_fold_ready",
                    "clip_approach","clip_carry_high","clip_stage","clip_descend_outside","clip_insert","clip_release","hands_free_hold"):
                save_component_seed(args.output/"seeds"/phase,step)
                saved_stages.add(phase)
            if clean and phase=="table_fold_ready" and not clip_entry_saved:
                save_component_seed(args.output,step)
                print("CLIP_ENTRY_SAVED",step,flush=True)
                clip_entry_saved=True
        if step % 30 == 0:
            print("PROGRESS", {k:v for k,v in trace[-1].items() if k!="jaws"}, flush=True)
            if step == 0 and "jaws" in trace[-1]:
                print("RESET_JAWS", trace[-1]["jaws"], flush=True)
        if step % args.checkpoint_every == 0:
            save_record()
        if watcher and (trace[-1]["substep_geometry"]["transverse_edge_crossings"] or
                        trace[-1]["substep_geometry"]["film_box_intersections"] or
                        trace[-1]["substep_geometry"]["film_round_intersections"]):
            if controller:
                controller.failure="independent substep geometry observer detected an intersection"
            break
    save_record()
    result = dict(frames=trace[-1]["step"], wall_seconds=time.monotonic()-started,
                  all_finite=all(row["finite"] for row in trace),
                  initial_fold_imposed=bool(args.clip_component_seed), object_pose_writes_during_rollout=False,
                  component_reset=component_reset,
                  coupling=args.coupling, ready_start=args.ready_start,
                  clip=args.clip, physics_substeps=sim.request.physics.substeps,
                  control_hz=args.hz,
                  rigid_backend=sim.snack_rigid_backend,
                  workspace=sim.snack_workspace,
                  world_z_offset_m=sim.snack_world_z_offset,
                  clip_style=args.clip_style,table_insert=args.table_insert,
                  material_profile=args.material_profile,
                  material=sim.snack_material_parameters,
                  timings=timings,
                  interrupted=interrupted[0],
                  controller_failure=controller.failure if controller else None,
                  controller_completed=controller_completed,
                  controller_diagnostics=({k:controller.marks[k] for k in (
                      "press_plans","press_diagnostics","fold_drag_m","clip_grasp_transform",
                      "left_clear_kinematic_plan","left_cleared_for_clip","last_move_error")
                      if k in controller.marks} if controller else {}),
                  substep_geometry=watcher.report() if watcher else None,
                  limitation="Uncalibrated material; contents/air not integrated; ready-start places open hands around the unfolded rim at reset; controller completion is not a physical success assessment")
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT", result, flush=True)
    if session:
        session.teardown()


if __name__ == "__main__":
    main()
