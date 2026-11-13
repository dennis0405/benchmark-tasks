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
    parser.add_argument("--material-profile",choices=("derived","paper_matched"),default="derived")
    parser.add_argument("--stage",choices=("fold","clip_grasp"),default="fold")
    parser.add_argument("--hz",type=int,choices=(20,60),default=60)
    args = parser.parse_args()
    if args.stage=="clip_grasp" and not (args.clip and args.drive):
        parser.error("clip_grasp requires --clip --drive")
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
                   Path(__file__).with_name("assets")/"yam_ready_pose.json"):
        shutil.copy2(source, sources/source.name)
    if args.clip:
        for source in (Path(__file__).resolve().parents[1]/"deformable_clip.py",
                       Path(__file__).with_name("tool_feedback.py"),
                       Path(__file__).with_name("clip_controller.py"),
                       Path(__file__).with_name("assets")/"clip_spring_v2.xml"):
            shutil.copy2(source, sources/source.name)
    (args.output/"arguments.json").write_text(json.dumps(vars(args), default=str, indent=2))
    started = time.monotonic()
    sim = make_environment(coupling=args.coupling, ready_start=args.ready_start,
                           with_clip=args.clip, material_profile=args.material_profile, control_hz=args.hz)
    timings = dict(build_s=time.monotonic()-started, driver_tick_s=0., readback_and_metrics_s=0., recording_s=0.)
    entity = sim.scene.by_name["pouch"]
    export_render_geometry(sim, args.output)
    session = controller = drive = None
    if args.drive:
        from tests.fixtures.grasp import open_session, tick
        from fold_controller import FoldController
        session = open_session(sim, 9764)
        controller = FoldController(sim, session)
        if args.stage=="clip_grasp":
            from clip_controller import clip_grasp_drive
            drive=clip_grasp_drive(controller)
        else:
            drive = controller.drive()
    snapshots, trace = {}, []
    def save_record():
        before=time.monotonic()
        temporary=args.output/"states.tmp.npz"
        np.savez_compressed(temporary, faces=entity.faces, **snapshots)
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
                    break
                tick(session, 1)
            else:
                sim.run(1).sync()
            timings["driver_tick_s"]+=time.monotonic()-before
        before=time.monotonic()
        state = read_object_states(sim)
        snapshots[f"q_{step:05d}"] = state["pouch"]
        snapshots[f"b_{step:05d}"] = state["bodies"]
        trace.append(dict(step=step, finite=bool(sim.state_is_finite()),
                          phase=controller.phase if controller else "rest",
                          z_min=float(state["pouch"][:, 2].min()),
                          z_max=float(state["pouch"][:, 2].max())))
        trace[-1]["fold"] = fold_metrics(state["pouch"], entity.grid, [(34, 28, 40)])
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
        if controller:
            target_q=sim.control.joint_target_q.numpy()
            trace[-1]["robot_targets"]={a:dict(joints=target_q[sim.arm(a).arm_coords].tolist(),
                gripper_joint_targets=target_q[sim.scene.embodiment.arms[a].gripper_coords].tolist(),
                grip_request=(None if controller.session.arms[a].grip_request is None else
                              float(controller.session.arms[a].grip_request))) for a in ("left","right")}
            trace[-1]["jaws"] = {a:controller.takes._fold_jaws(controller.film,a).tolist()
                                 for a in ("left", "right")}
            solver = sim.pipeline.program.solver
            if hasattr(solver, "shape_contact_forces"):
                forces = solver.shape_contact_forces()
                owners = sim.model.shape_body.numpy()
                trace[-1]["finger_soft_force_N"] = {
                    a:[float(np.linalg.norm(forces[np.isin(owners, side)].sum(axis=0)))
                       for side in index.finger_bodies]
                    for a,index in sim.scene.embodiment.arms.items()}
        timings["readback_and_metrics_s"]+=time.monotonic()-before
        if step % 30 == 0:
            print("PROGRESS", {k:v for k,v in trace[-1].items() if k!="jaws"}, flush=True)
            if step == 0 and "jaws" in trace[-1]:
                print("RESET_JAWS", trace[-1]["jaws"], flush=True)
        if step % args.checkpoint_every == 0:
            save_record()
    save_record()
    result = dict(frames=trace[-1]["step"], wall_seconds=time.monotonic()-started,
                  all_finite=all(row["finite"] for row in trace),
                  initial_fold_imposed=False, object_pose_writes_during_rollout=False,
                  coupling=args.coupling, ready_start=args.ready_start,
                  clip=args.clip, physics_substeps=sim.request.physics.substeps,
                  control_hz=args.hz,
                  material_profile=args.material_profile,
                  timings=timings,
                  interrupted=interrupted[0],
                  controller_failure=controller.failure if controller else None,
                  limitation="Uncalibrated material; contents/air not integrated; ready-start places open hands around the unfolded rim at reset; clip insertion not yet demonstrated")
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT", result, flush=True)
    if session:
        session.teardown()


if __name__ == "__main__":
    main()
