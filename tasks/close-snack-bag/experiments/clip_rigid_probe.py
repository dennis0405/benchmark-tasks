"""Isolate YAM / clip / rack dynamics. No film or sealing claim."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deformable_scene import make_environment,export_render_geometry
from fold_controller import FoldController
from clip_controller import clip_grasp_drive,clip_pose,clip_angle
from check_rigid_tool import inspect_tool


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--frames",type=int,default=1000)
    parser.add_argument("--physics-hz",type=int,default=1200)
    parser.add_argument("--workspace",choices=("original","raised_forward","raised_flat"),default="original")
    parser.add_argument("--clip-style",choices=("standard","thin_lower"),default="standard")
    parser.add_argument("--carry-root",type=float,nargs=3,default=None)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sources=args.output/"sources";sources.mkdir()
    root=Path(__file__).resolve().parents[1]
    for name in ("deformable_scene.py","deformable_clip.py","pouch_geometry.py"):
        shutil.copy2(root/name,sources/name)
    for name in ("clip_rigid_probe.py","fold_controller.py","clip_controller.py","check_rigid_tool.py",
                 "gpu_contact_checks.py","film_material.py","plan_clip_poses.py","assets/clip_spring_v2.xml","assets/yam_ready_pose.json"):
        shutil.copy2(root/"experiments"/name,sources/Path(name).name)
    if args.workspace!="original":
        name=f"assets/yam_{args.workspace}_pose.json"
        shutil.copy2(root/"experiments"/name,sources/Path(name).name)
    if args.clip_style=="thin_lower":
        name="assets/clip_thin_lower.xml"
        shutil.copy2(root/"experiments"/name,sources/Path(name).name)
    (args.output/"arguments.json").write_text(json.dumps(dict(vars(args),hz=20),default=str,indent=2))
    started=time.monotonic()
    sim=make_environment(ready_start=True,with_clip=True,rigid_cpu=args.cpu,
                         rigid_only=True,control_hz=20,physics_hz=args.physics_hz,workspace=args.workspace,
                         clip_style=args.clip_style)
    assert sim.model.particle_count==0
    solver=sim.pipeline.program.solver
    actual=getattr(solver,"rigid",solver)
    assert bool(actual.use_mujoco_cpu)==args.cpu
    from tests.fixtures.grasp import open_session,tick
    session=open_session(sim,9766)
    controller=FoldController(sim,session)
    controller.plan_clip=args.plan
    if args.carry_root is not None:
        controller.carry_root=args.carry_root
    drive=clip_grasp_drive(controller)
    export_render_geometry(sim,args.output)
    preflight=inspect_tool(sim)
    (args.output/"rigid-tool-preflight.json").write_text(json.dumps(preflight,indent=2))
    if preflight["intersections"]:
        raise RuntimeError("Initial rigid tool interference")
    build_s=time.monotonic()-started
    interrupted=[False]
    signal.signal(signal.SIGINT,lambda *_:interrupted.__setitem__(0,True))
    signal.signal(signal.SIGTERM,lambda *_:interrupted.__setitem__(0,True))
    snapshots,trace={},[]
    completed=False
    def save():
        path=args.output/"states.tmp.npz"
        np.savez_compressed(path,**snapshots);path.replace(args.output/"states.npz")
        path=args.output/"trace.tmp.json"
        path.write_text(json.dumps(trace));path.replace(args.output/"trace.json")
    for step in range(args.frames+1):
        if interrupted[0]:break
        if step:
            try:next(drive)
            except StopIteration:
                completed=True;break
            tick(session,1)
        snapshots[f"b_{step:05d}"]=sim.state.body_q.numpy().copy()
        snapshots[f"j_{step:05d}"]=sim.state.joint_q.numpy().copy()
        snapshots[f"d_{step:05d}"]=sim.state.joint_qd.numpy().copy()
        trace.append(dict(step=step,phase=controller.phase,finite=bool(sim.state_is_finite()),
                          clip_root_pose=clip_pose(controller).tolist(),clip_angle_rad=clip_angle(controller),
                          jaws={a:controller.takes._fold_jaws(controller.film,a).tolist() for a in ("left","right")}))
        if step%30==0:print("PROGRESS",trace[-1],flush=True)
        if step%60==0:save()
        if not trace[-1]["finite"]:
            controller.failure="non-finite rigid state";break
    save()
    result=dict(component="rigid YAM / clip / rack only; no pouch or sealing",
                rigid_backend=sim.snack_rigid_backend,frames=trace[-1]["step"],
                control_hz=20,physics_hz=args.physics_hz,completed=completed,
                controller_failure=controller.failure,interrupted=interrupted[0],
                all_finite=all(r["finite"] for r in trace),
                build_s=build_s,rollout_s=time.monotonic()-started-build_s)
    (args.output/"result.json").write_text(json.dumps(result,indent=2))
    (args.output/"controller-marks.json").write_text(json.dumps(controller.marks,indent=2))
    print("RESULT",result,flush=True)
    session.teardown();sim.close()


if __name__=="__main__":main()
