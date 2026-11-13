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
    args = parser.parse_args()
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
    (args.output/"arguments.json").write_text(json.dumps(vars(args), default=str, indent=2))
    started = time.monotonic()
    sim = make_environment(coupling=args.coupling, ready_start=args.ready_start)
    entity = sim.scene.by_name["pouch"]
    export_render_geometry(sim, args.output)
    session = controller = drive = None
    if args.drive:
        from tests.fixtures.grasp import open_session, tick
        from fold_controller import FoldController
        session = open_session(sim, 9764)
        controller = FoldController(sim, session)
        drive = controller.drive()
    snapshots, trace = {}, []
    for step in range(args.frames + 1):
        if interrupted[0]:
            break
        if step:
            if drive:
                try:
                    next(drive)
                except StopIteration:
                    break
                tick(session, 1)
            else:
                sim.run(1).sync()
        state = read_object_states(sim)
        snapshots[f"q_{step:05d}"] = state["pouch"]
        snapshots[f"b_{step:05d}"] = state["bodies"]
        trace.append(dict(step=step, finite=bool(sim.state_is_finite()),
                          phase=controller.phase if controller else "rest",
                          z_min=float(state["pouch"][:, 2].min()),
                          z_max=float(state["pouch"][:, 2].max())))
        trace[-1]["fold"] = fold_metrics(state["pouch"], entity.grid, [(35, 30, 40)])
        if controller:
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
        if step % 30 == 0:
            print("PROGRESS", {k:v for k,v in trace[-1].items() if k!="jaws"}, flush=True)
            if step == 0 and "jaws" in trace[-1]:
                print("RESET_JAWS", trace[-1]["jaws"], flush=True)
        if step % 30 == 0:
            np.savez_compressed(args.output/"states.npz", faces=entity.faces, **snapshots)
            (args.output/"trace.json").write_text(json.dumps(trace))
    np.savez_compressed(args.output / "states.npz", faces=entity.faces, **snapshots)
    (args.output / "trace.json").write_text(json.dumps(trace))
    result = dict(frames=trace[-1]["step"], wall_seconds=time.monotonic()-started,
                  all_finite=all(row["finite"] for row in trace),
                  initial_fold_imposed=False, object_pose_writes_during_rollout=False,
                  coupling=args.coupling, ready_start=args.ready_start,
                  interrupted=interrupted[0],
                  limitation="Uncalibrated film; no clip or food yet; ready-start places open hands around the unfolded rim at reset")
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT", result, flush=True)
    if session:
        session.teardown()


if __name__ == "__main__":
    main()
