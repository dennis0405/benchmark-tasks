#!/usr/bin/env python3
"""Host-executed native GPU component/material diagnostics.

The top-level process launches independent 240 and 480 Hz Kit workers so the
reported callback intervals are genuinely produced by different simulations.
The production worker uses native breakable joints, fresh PhysX contact reports,
tensor poses/twists, force actuation, and RTX RGB-D frames.  Every frozen check
is emitted once in one aggregate report.  Missing evidence is an error; no
configured value is substituted for a native reading.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, subprocess, sys, traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
DEFAULT_OUT=ROOT/"measurements/component-physics-aggregate.json"

def load_local(name):
    """Load a task-local helper without relying on cwd or PYTHONPATH."""
    path=HERE/f"{name}.py"
    spec=importlib.util.spec_from_file_location(f"helicoil_{name}",path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load task helper: {path}")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def cli():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=DEFAULT_OUT)
    p.add_argument("--headless",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--worker-hz",type=int,choices=(240,480)); p.add_argument("--worker-output",type=Path)
    a=p.parse_args()
    # The controller invokes this from different working directories.  Keep
    # relative output names rooted at the workspace, never under task/.
    if not a.output.is_absolute(): a.output=ROOT/a.output
    if a.worker_output is not None and not a.worker_output.is_absolute(): a.worker_output=ROOT/a.worker_output
    return a

def dump(path,obj):
    path=path.resolve(); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")

def jsonl(path, rows):
    """Write independently replayable native rows without NaN extensions."""
    path=path.resolve(); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")

def finite(x,label):
    x=float(x)
    if not math.isfinite(x): raise RuntimeError(f"non-finite {label}: {x}")
    return x

def unavailable_contract_checks():
    """Retain the frozen check schema when native execution fails.

    ``None`` is an explicit unavailable reading, never a fabricated zero.  A
    trusted controller can therefore distinguish a runtime failure from a
    changed/missing contract while still rejecting every unavailable check.
    """
    try:
        contract=json.loads((ROOT/"artifacts/contract.json").read_text())
        return [{**spec,"measured":None,"available":False}
                for spec in contract["physics_checks"]]
    except Exception:
        return []

def convex_hull_2d(points):
    """Deterministic monotone-chain hull for the projected workspace mask."""
    pts=sorted(set((float(x),float(y)) for x,y in points))
    if len(pts)<=1: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in pts:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0: lo.pop()
        lo.append(p)
    hi=[]
    for p in reversed(pts):
        while len(hi)>=2 and cross(hi[-2],hi[-1],p)<=0: hi.pop()
        hi.append(p)
    return lo[:-1]+hi[:-1]

def master(args):
    rawdir=ROOT/"measurements/raw-component"; rawdir.mkdir(parents=True,exist_ok=True)
    raws={}
    for hz in (240,480):
        out=rawdir/f"native-{hz}hz.json"
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker-hz",str(hz),"--worker-output",str(out)]
        cmd.append("--headless" if args.headless else "--no-headless")
        cp=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (rawdir/f"native-{hz}hz.log").write_text(cp.stdout)
        if cp.returncode or not out.is_file(): raise RuntimeError(f"{hz} Hz native worker failed; see {rawdir/f'native-{hz}hz.log'}")
        raws[hz]=json.loads(out.read_text())
        if raws[hz].get("status")!="complete": raise RuntimeError(f"{hz} Hz worker incomplete: {raws[hz].get('error')}")
    contract=json.loads((ROOT/"artifacts/contract.json").read_text()); p=raws[240]; r=raws[480]
    vals=dict(p["measurements"])
    vals["production-postphysics-callback-dt-error-maximum"]=p["callback_dt_error_max"]
    vals["refinement-postphysics-callback-dt-error-maximum"]=r["callback_dt_error_max"]
    vals["lifecycle-task-callback-handle-count-error-maximum"]=max(
        p["callback_handle_count_errors"] + r["callback_handle_count_errors"]
    )
    # Timestep refinement is not a separate frozen check, but must be retained as
    # native evidence and must agree on fracture count/impulse within 10%.
    a=p["measurements"]["nominal-qualified-axial-impulse-maximum"]
    b=r["measurements"]["nominal-qualified-axial-impulse-maximum"]
    convergence=abs(a-b)/max(abs(a),abs(b),1e-12)
    checks=[]
    for s in contract["physics_checks"]:
        if s["name"] not in vals: raise RuntimeError(f"worker did not measure {s['name']}")
        checks.append({**s,"measured":finite(vals[s["name"]],s["name"])})
    report={"status":"complete","scope":"native rigid component/material tests; not a robot demo",
            "contract_slug":contract["slug"],"checks":checks,
            "native_evidence":{"production":p["evidence"],"refinement":r["evidence"],
                               "timestep_impulse_relative_difference":convergence},
            "artifacts":[str(args.output.relative_to(ROOT)),
                         "measurements/raw-component/native-240hz.json","measurements/raw-component/native-480hz.json",
            "measurements/raw-component/native-240hz-events.jsonl","measurements/raw-component/native-480hz-events.jsonl",
                         "measurements/raw-component/native-240hz-progress.json","measurements/raw-component/native-480hz-progress.json",
                         "measurements/raw-component/native-240hz.log","measurements/raw-component/native-480hz.log",
                         "measurements/component-rgb.png","measurements/component-depth.npy"]}
    dump(args.output,report); print(json.dumps(report,allow_nan=False)); return 0

def worker(args):
    app=None
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=args.headless,enable_cameras=True).app
        import numpy as np, omni.usd, torch, warp as wp
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from isaaclab.sim import SimulationCfg, SimulationContext
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.physics.physx_manager import IsaacEvents, PhysxManager
        from omni.physx import get_physx_simulation_interface, get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        mechanism=load_local("component_mechanism")
        geometry=load_local("procedural_geometry")
        build=mechanism.build; INSERT_XY=geometry.INSERT_XY

        hz=args.worker_hz; dt=1.0/hz
        omni.usd.get_context().new_stage(); stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage,1.0); UsdPhysics.SetStageKilogramsPerUnit(stage,1.0)
        world=UsdGeom.Xform.Define(stage,"/World"); stage.SetDefaultPrim(world.GetPrim())
        ps=UsdPhysics.Scene.Define(stage,"/World/PhysicsScene"); ps.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1)); ps.CreateGravityMagnitudeAttr(9.81)
        paths=build(stage,Gf,UsdGeom,UsdPhysics,PhysxSchema)

        # Controlled material-test rig.  Four physical mandrels and hammer heads
        # are authored before reset, then driven only by native forces.
        rig_hammers=[]; rig_mandrels=[]
        for i,(x,y) in enumerate(INSERT_XY):
            m=UsdGeom.Cylinder.Define(stage,f"/World/rig/mandrel_{i}"); m.CreateRadiusAttr(.001); m.CreateHeightAttr(.030); m.CreateAxisAttr("Z")
            m.AddTranslateOp().Set(Gf.Vec3d(x,y,.804)); UsdPhysics.CollisionAPI.Apply(m.GetPrim()); UsdPhysics.RigidBodyAPI.Apply(m.GetPrim()); UsdPhysics.MassAPI.Apply(m.GetPrim()).CreateMassAttr(.018)
            PhysxSchema.PhysxRigidBodyAPI.Apply(m.GetPrim()).CreateDisableGravityAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(m.GetPrim()).CreateThresholdAttr(0.0)
            # Bottom of the 8 mm head starts 0.2 mm above the mandrel top.
            # The previous 7 mm air gap consumed the short calibrated pulse,
            # so no hammer/mandrel contact episode could exist.
            h=UsdGeom.Xform.Define(stage,f"/World/rig/hammer_{i}"); h.AddTranslateOp().Set(Gf.Vec3d(x,y,.8232))
            hs=UsdGeom.Cube.Define(stage,f"/World/rig/hammer_{i}/shape"); hs.CreateSizeAttr(1.0); hs.AddScaleOp().Set(Gf.Vec3d(.012,.012,.008))
            UsdPhysics.CollisionAPI.Apply(hs.GetPrim()); UsdPhysics.RigidBodyAPI.Apply(h.GetPrim()); UsdPhysics.MassAPI.Apply(h.GetPrim()).CreateMassAttr(.10)
            PhysxSchema.PhysxRigidBodyAPI.Apply(h.GetPrim()).CreateDisableGravityAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(h.GetPrim()).CreateThresholdAttr(0.0)
            rig_mandrels.append(str(m.GetPath())); rig_hammers.append(str(h.GetPath()))

        # The first integrated interval is n=1.  PRE(1) snapshots time zero and
        # POST(1) consumes only contacts produced by (0, dt].
        pre_callback_dts=[]; post_callback_dts=[]; pre_steps=[]; post_steps=[]; step=[1]; view_holder={}; pre_state={}
        callback_exceptions=[]; tensor_contact_views=[]
        state_rows=[]
        def on_pre(d):
            pre_steps.append(step[0]); pre_callback_dts.append(float(d))
            v=view_holder.get("all")
            if v is not None:
                poses=tensor(v.get_transforms()).tolist(); twists=tensor(v.get_velocities()).tolist()
                pre_state[step[0]]=(poses,twists)
                state_rows.append({"record":"body_state","phase":"PRE","step":step[0],
                                   "time_s":(step[0]-1)*dt,"poses":poses,"twists":twists})
        pre_h=PhysxManager.register_callback(on_pre,IsaacEvents.PRE_PHYSICS_STEP,name=f"helicoil_pre_{hz}",wrap_weak_ref=False)
        def on_post(d):
            try:
                post_callback_dts.append(float(d))
                v=view_holder.get("all")
                if v is not None:
                    state_rows.append({"record":"body_state","phase":"POST","step":step[0],
                                       "time_s":step[0]*dt,
                                       "poses":tensor(v.get_transforms()).tolist(),
                                       "twists":tensor(v.get_velocities()).tolist()})
                # Direct-GPU PhysX does not reliably emit the legacy report-event
                # stream.  Read the supported tensor contact view at POST instead.
                # Each view is one sensor actor against one named filter actor, so
                # pair identity and force sign are unambiguous.
                for cv,a,b in tensor_contact_views:
                    forces,points,normals,seps,counts,starts=cv.get_contact_data(dt=float(d))
                    fs=wp.to_torch(forces).detach().cpu().tolist()
                    ps_=wp.to_torch(points).detach().cpu().tolist()
                    ns=wp.to_torch(normals).detach().cpu().tolist()
                    ds=wp.to_torch(seps).detach().cpu().reshape(-1).tolist()
                    cs=wp.to_torch(counts).detach().cpu().reshape(-1).tolist()
                    ss=wp.to_torch(starts).detach().cpu().reshape(-1).tolist()
                    if len(cs)!=1 or len(ss)!=1:
                        raise RuntimeError(f"contact index shape mismatch for {a}->{b}: {len(cs)}/{len(ss)}")
                    for k in range(int(ss[0]),int(ss[0])+int(cs[0])):
                        raw_force=fs[k]
                        normal=[float(x) for x in ns[k]]
                        if isinstance(raw_force,(int,float)):
                            normal_force=float(raw_force)
                        elif len(raw_force)==1:
                            normal_force=float(raw_force[0])
                        elif len(raw_force)==3:
                            normal_force=None
                        else:
                            raise RuntimeError(f"unexpected tensor force row for {a}->{b}: {raw_force!r}")
                        # In this pinned backend get_contact_data exposes a
                        # scalar normal force (N x 1).  Convert it once using
                        # the native normal so all downstream episode math is
                        # performed on an actual world-space vector.
                        force=([normal_force*x for x in normal] if normal_force is not None
                               else [float(x) for x in raw_force])
                        impulse=[x*float(d) for x in force]
                        contact_rows.append({"step":step[0],"time_s":step[0]*dt,"a":a,"b":b,
                            "collider_a":a,"collider_b":b,"p":[float(x) for x in ps_[k]],
                            "n":normal,"native_normal_force_N":normal_force,"j_native":impulse,
                            "native_field":"tensor_force_N_times_callback_dt_s",
                            "separation":float(ds[k]),"source":"rigid_contact_view"})
                post_steps.append(step[0]); step[0]+=1
            except Exception as exc:
                callback_exceptions.append(f"POST step {step[0]} {type(exc).__name__}: {exc}")
        post_h=PhysxManager.register_callback(on_post,IsaacEvents.POST_PHYSICS_STEP,name=f"helicoil_post_{hz}",wrap_weak_ref=False)
        contact_rows=[]; legacy_contact_rows=[]; broken=[]
        def path(v): return str(PhysicsSchemaTools.intToSdfPath(v))
        def vec3(value, normal):
            """Contact API versions expose either vector impulse or scalar normal impulse."""
            try:
                q=[float(x) for x in value]
            except TypeError:
                q=[float(value)*float(x) for x in normal]
            if len(q)!=3 or not all(math.isfinite(x) for x in q):
                raise RuntimeError(f"invalid native contact impulse {q}")
            return q
        def contacts(headers,data):
            tag=step[0]
            for h in headers:
                a,b=path(h.actor0),path(h.actor1)
                for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                    c=data[k]; normal=[float(v) for v in c.normal]
                    legacy_contact_rows.append({"step":tag,"time_s":tag*dt,"a":a,"b":b,
                        "collider_a":path(getattr(h,"collider0",h.actor0)),
                        "collider_b":path(getattr(h,"collider1",h.actor1)),
                        "p":[float(v) for v in c.position],"n":normal,
                        "j_native":vec3(c.impulse,normal),"native_field":"impulse_Ns",
                        "separation":float(c.separation)})
        contact_sub=get_physx_simulation_interface().subscribe_contact_report_events(contacts)
        def decode_sdf_path(value):
            """Decode the path identifier in the native ``(path, aux)`` payload.

            This pinned PhysicsSchemaTools binding accepts exactly one uint64.
            The observed JOINT_BREAK payload is ``(encodedPath, 0)``; the
            second field is not another half of the path and must not be
            passed to intToSdfPath.
            """
            path_id=value[0] if isinstance(value,(tuple,list)) else value
            return str(PhysicsSchemaTools.intToSdfPath(int(path_id)))
        def sim_event(e):
            if int(e.type)==int(SimulationEvent.JOINT_BREAK):
                encoded=e.payload.get("jointPath",e.payload)
                broken.append({"step":step[0],"joint":decode_sdf_path(encoded),"native_joint_id":str(encoded)})
        event_sub=get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(sim_event)
        sim=SimulationContext(SimulationCfg(dt=dt,render_interval=max(1,hz//20),device="cuda:0"))
        # Use the same Isaac Lab Camera sensor path as YAM.  It is driven by
        # SimulationContext.render + Camera.update and does not pause the live
        # timeline as Replicator orchestrator.step(pause_timeline=True) did.
        camera=Camera(CameraCfg(prim_path="/World/TaskCamera",height=480,width=640,
            data_types=["rgb","distance_to_camera"],update_period=0,
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0,horizontal_aperture=20.955,
                                             clipping_range=(0.05,20.0))))
        # Required lifecycle samples: after registration and after the authored
        # reset.  Only this task's PRE/POST handles are counted.
        callback_handle_counts=[int(pre_h is not None)+int(post_h is not None)]
        sim.reset()
        callback_handle_counts.append(int(pre_h is not None)+int(post_h is not None))
        camera.set_world_poses_from_view(torch.tensor([[.62,-.62,1.28]],device=sim.device),
                                         torch.tensor([[.37,0,.79]],device=sim.device))
        pv=sim.physics_manager.get_physics_sim_view()
        toolv=pv.create_rigid_body_view(paths["tools"]); insertv=pv.create_rigid_body_view(paths["insert"])
        shellv=pv.create_rigid_body_view(paths["shell"]+paths["boss"])
        fixturev=pv.create_rigid_body_view(paths["fixture"])
        tangv=pv.create_rigid_body_view(paths["tang"]); hammerv=pv.create_rigid_body_view(rig_hammers)
        mandrelv=pv.create_rigid_body_view(rig_mandrels)
        def tensor(v, *, require_finite=True):
            t=wp.to_torch(v) if isinstance(v,wp.array) else v; t=t.detach().cpu()
            if require_finite and not bool(torch.isfinite(t).all()):
                raise RuntimeError("non-finite native pose/twist state")
            return t
        tracked=paths["shell"]+paths["insert"]+paths["tang"]+rig_mandrels+rig_hammers
        allv=pv.create_rigid_body_view(tracked); view_holder["all"]=allv
        # One-to-one views preserve actual actor ordering.  Many-to-many
        # filtered sensors are unsupported by this PhysX backend.
        for i in range(4):
            for a,b in ((rig_hammers[i],rig_mandrels[i]),(rig_mandrels[i],paths["tang"][i])):
                cv=pv.create_rigid_contact_view([a],filter_patterns=[[b]],max_contact_data_count=64)
                if cv.sensor_count!=1 or cv.filter_count!=1:
                    raise RuntimeError(f"contact view mismatch for {a}->{b}: {cv.sensor_count}/{cv.filter_count}")
                tensor_contact_views.append((cv,a,b))
        if allv.count!=len(tracked): raise RuntimeError(f"tracked body view mismatch {allv.count}/{len(tracked)}")
        insert0=tensor(insertv.get_transforms()).clone(); tang0=tensor(tangv.get_transforms()).clone()

        # Exact one-second untouched interval; sample final quarter-second.
        tool_speeds=[]; fixture_speeds=[]
        for n in range(hz):
            sim.step(render=False)
            if n>=3*hz//4:
                loose=torch.cat((tensor(toolv.get_velocities())[:,:3],
                                 tensor(shellv.get_velocities())[:,:3],
                                 tensor(insertv.get_velocities())[:,:3],
                                 tensor(tangv.get_velocities())[:,:3]),dim=0)
                tool_speeds.append(float(torch.linalg.vector_norm(loose,dim=1).max()))
                fixture_speeds.append(float(torch.linalg.vector_norm(tensor(fixturev.get_velocities())[:,:3],dim=1).max()))
        initial_contacts=[c for c in contact_rows if c["step"]<=hz]
        initial_pen=max([max(0.0,-c["separation"]) for c in initial_contacts] or [0.0])
        insert_settled=tensor(insertv.get_transforms())
        initial_disp=float(torch.max(torch.abs(insert_settled[:,2]-insert0[:,2])))
        mandrel_onset=tensor(mandrelv.get_transforms())
        initial_breaks=sum(1 for e in broken if int(e["step"])<=hz and e["joint"] in paths["notch"])
        # Credits require a contact force path initiated by a commanded strike;
        # no command is issued in this interval.  Retain the measured native
        # event/contact evidence used to derive the zero rather than treating
        # all generic break events as tang fractures.
        initial_credits=sum(1 for e in broken if int(e["step"])<=hz and
                            e["joint"] in paths["notch"] and
                            any(c["step"]==e["step"] and any(p in (c["a"],c["b"]) for p in rig_hammers)
                                for c in initial_contacts))
        initial_tang=tensor(tangv.get_transforms()); initial_tangvel=tensor(tangv.get_velocities())
        # Collection requires genuine prior removal and fresh tray support.
        # No notch has broken in the untouched window, so no original tang is
        # eligible regardless of its centre position.
        initial_tray=0
        initial_score=.125*initial_credits+.10*initial_tray

        # Physical force-controlled strikes; no transform or velocity writes.
        # The simulation view is configured with the Warp frontend.  Supplying
        # torch.Tensor objects makes that frontend inspect torch.dtype as a
        # Warp class and fails before applying force.  Keep native GPU buffers
        # in Warp for actuation and convert only readback arrays to torch.
        # The Warp tensor frontend requires an explicit index tensor even when
        # the view contains exactly the requested bodies.  It also describes a
        # (count, 3) force as scalar float32 storage: wp.vec3 with shape (4, 3)
        # would describe twelve vec3 elements and is rejected by the backend.
        body_indices=wp.array([0,1,2,3],dtype=wp.uint32,device="cuda:0")
        torque=wp.zeros((4,3),dtype=wp.float32,device="cuda:0")
        zero_force=wp.zeros((4,3),dtype=wp.float32,device="cuda:0")
        downward=wp.array([[0.0,0.0,-3.0]]*4,dtype=wp.float32,device="cuda:0")
        strike_start=step[0]
        for n in range(max(1,int(.12*hz))):
            force=downward if n<max(1,int(.010*hz)) else zero_force
            # PhysX tensor signature is (forces, torques, positions, indices,
            # is_global).  Apply to every body in this four-body view; forces
            # are expressed in each hammer's local frame.
            hammerv.apply_forces_and_torques_at_position(force,torque,None,body_indices,False)
            sim.step(render=False)
        for _ in range(int(2.5*hz)): sim.step(render=False)
        strike_contacts=[c for c in contact_rows if c["step"]>=strike_start]
        # Persist native evidence before any renderer/camera work.  A later
        # acquisition failure must not erase the physical outcome.
        raw=ROOT/f"measurements/raw-component/native-{hz}hz-events.jsonl"
        jsonl(raw,(state_rows
                   +[{"record":"contact",**c} for c in contact_rows]
                   +[{"record":"legacy_contact_event",**c} for c in legacy_contact_rows]
                   +[{"record":"joint_break",**e} for e in broken]))
        dump(ROOT/f"measurements/raw-component/native-{hz}hz-progress.json",
             {"status":"physics_complete_camera_pending","hz":hz,"native_step":step[0]-1,
              "native_time_s":(step[0]-1)*dt,"joint_break_events":broken,
              "missing_observables":["rgb","distance_to_camera","camera_frame_timestamps"]})
        # Native impulses are N*s.  A physical episode may propagate through
        # hammer, mandrel and tang over adjacent solver intervals; qualification
        # therefore uses the frozen contact-free 1/240 s episode closure rather
        # than an invented same-step equality requirement.
        episodes=[]
        for i in range(4):
            hs=[c for c in strike_contacts if {c["a"],c["b"]}=={rig_hammers[i],rig_mandrels[i]}]
            mts=[c for c in strike_contacts if {c["a"],c["b"]}=={rig_mandrels[i],paths["tang"][i]}]
            # Close an episode only after 1/240 s with neither contact edge.
            # At 480 Hz this is two empty intervals.  This preserves a real
            # hammer -> mandrel -> tang path that propagates across solver
            # intervals instead of imposing an unsupported same-step rule.
            by_step={}
            for row in hs:
                by_step.setdefault(int(row["step"]),{"hammer":[],"tang":[]})["hammer"].append(row)
            for row in mts:
                by_step.setdefault(int(row["step"]),{"hammer":[],"tang":[]})["tang"].append(row)
            closed=[]; current=None; last_contact=None
            close_steps=max(1,int(math.ceil((1.0/240.0)/dt-1e-12)))
            for contact_step in sorted(by_step):
                if current is not None and contact_step-last_contact>close_steps:
                    closed.append(current); current=None
                if current is None:
                    current={"hammer":[],"tang":[],"first_step":contact_step,"last_step":contact_step}
                current["hammer"].extend(by_step[contact_step]["hammer"])
                current["tang"].extend(by_step[contact_step]["tang"])
                current["last_step"]=contact_step; last_contact=contact_step
            if current is not None:
                closed.append(current)
            qualified=[ep for ep in closed if ep["hammer"] and ep["tang"]]
            rows=[row for ep in qualified for row in ep["hammer"]]
            # Contact-view rows are ordered sensor (hammer) then filter
            # (mandrel), so j_native is impulse on the hammer.  Negate it to
            # obtain impulse delivered by the hammer to the mandrel.  Preserve
            # every signed native vector in the raw rows above.
            axial=sum(max(0.0,-c["j_native"][2]) for c in rows)
            off=sum(math.hypot(c["j_native"][0],c["j_native"][1]) for c in rows)
            radial=max([math.hypot(c["p"][0]-INSERT_XY[i][0],c["p"][1]-INSERT_XY[i][1]) for c in rows],default=None)
            xq,yq,zq,wq=[float(v) for v in mandrel_onset[i,3:7]]
            axis_z=1.0-2.0*(xq*xq+yq*yq)
            angle=math.degrees(math.acos(max(-1.0,min(1.0,abs(axis_z)))))
            episodes.append({"insert":i,"axial":axial,"offaxis":off,"radial":radial,"angle_deg":angle,
                             "force_path_steps":sum(ep["last_step"]-ep["first_step"]+1 for ep in qualified),
                             "closed_episode_count":len(closed),"qualified_episode_count":len(qualified),
                             "hammer_steps":sorted({int(c['step']) for c in hs}),
                             "mandrel_tang_steps":sorted({int(c['step']) for c in mts})})
        # Qualification is the observed force path and geometry.  Contract
        # limits remain separate checks; a physically observed but excessive
        # impulse must be reported as a failed measurement, not erased from
        # the episode set.
        valid=[e for e in episodes if e["force_path_steps"] and e["radial"] is not None]
        native_broken=sum(1 for j in paths["notch"] if any(e["joint"]==j for e in broken))

        # Deliberately off-axis physical control in an isolated fifth rig.
        invalid_off=max([e["offaxis"] for e in episodes] or [0.0])
        if invalid_off<=.008:
            lateral=wp.array([[12.0,0.0,0.0]]*4,dtype=wp.float32,device="cuda:0")
            for _ in range(max(2,int(.02*hz))):
                hammerv.apply_forces_and_torques_at_position(lateral,torque,None,body_indices,False)
                sim.step(render=False)
            bad=[c for c in contact_rows if c["step"]>=step[0]-max(2,int(.02*hz)) and any(p in (c["a"],c["b"]) for p in rig_hammers)]
            invalid_off=sum(math.hypot(c["j_native"][0],c["j_native"][1]) for c in bad)

        insert1=tensor(insertv.get_transforms()); tang1=tensor(tangv.get_transforms()); tangvel=tensor(tangv.get_velocities())
        axial_disp=float(torch.max(torch.abs(insert1[:,2]-insert0[:,2])))
        qdot=torch.abs(torch.sum(insert1[:,3:7]*insert0[:,3:7],dim=1)).clamp(max=1)
        angular=float(torch.rad2deg(2*torch.acos(qdot)).max())
        tray_count=0
        fragment_rows=[]
        tang_half=np.asarray([.0016,.000325,.000225])
        for pose,vel in zip(tang1,tangvel):
            x,y,z=[float(v) for v in pose[:3]]; speed=float(torch.linalg.vector_norm(vel[:3])); omega=float(torch.linalg.vector_norm(vel[3:]))
            # Complete fragment geometry, not its centre: transform every box
            # corner and require it inside the closed tray interior prism.
            qx,qy,qz,qw=[float(v) for v in pose[3:7]]
            R=np.asarray([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                          [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                          [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
            verts=np.asarray([np.asarray([x,y,z])+R@np.asarray([sx,sy,sz])
                              for sx in (-tang_half[0],tang_half[0]) for sy in (-tang_half[1],tang_half[1])
                              for sz in (-tang_half[2],tang_half[2])])
            # Tray floor top is z=.766 and wall rim is z=.800.  These are
            # authored world coordinates, never values derived from the
            # fragment itself.
            inside=bool(np.all((verts[:,0]>=.439)&(verts[:,0]<=.511)&
                               (verts[:,1]>=-.143)&(verts[:,1]<=-.081)&
                               (verts[:,2]>=.766)&(verts[:,2]<=.800)))
            support=any(c["step"]==step[0]-1 and
                        ({c["a"],c["b"]} & set(paths["tray"])) and
                        (c["a"] in paths["tang"] or c["b"] in paths["tang"])
                        for c in contact_rows)
            contained=inside and support and speed<=.01 and omega<=.5
            fragment_rows.append({"pose":pose.tolist(),"twist":vel.tolist(),"vertices":verts.tolist(),"inside":inside,"fresh_tray_support":support,"contained":contained})
            tray_count+=int(contained)
        post_pen=max([max(0.0,-c["separation"]) for c in strike_contacts] or [0.0])

        # Real RTX RGB-D acquisition. The camera looks at the complete workcell.
        valid_frames=0; depth_fracs=[]; frame_rows=[]
        # Capture t0 without advancing physics, then advance exactly hz/20
        # native steps per observation (only the boundary step renders).
        for frame_i in range(41):
            if frame_i==0:
                sim.render(); camera.update(dt=dt)
            else:
                for __ in range(hz//20-1): sim.step(render=False)
                sim.step(render=True); camera.update(dt=dt)
            rgb=tensor(camera.data.output["rgb"])[0].numpy()
            # RTX depth intentionally uses +inf for rays with no surface hit.
            # Preserve that native sentinel and test finiteness only inside the
            # declared workspace mask below.  Pose/twist/RGB still fail closed.
            depth=tensor(camera.data.output["distance_to_camera"],require_finite=False)[0].numpy()
            # Isaac Lab 3.0 exposes distance as (H,W,1), while some pinned
            # annotator versions expose (H,W).  Normalize only the singleton
            # channel; reject every other shape instead of broadcasting a mask.
            if depth.ndim==3 and depth.shape[-1]==1:
                depth=depth[...,0]
            if depth.ndim!=2:
                raise RuntimeError(f"unexpected distance_to_camera shape {depth.shape}")
            # Project the declared workspace AABB with the actual authored USD
            # camera transform/intrinsics.  Because every task collision shape
            # is inside this AABB, its convex hull is the frozen workspace mask.
            cprim=stage.GetPrimAtPath("/World/TaskCamera")
            cxf=UsdGeom.Xformable(cprim).ComputeLocalToWorldTransform(0); w2c=cxf.GetInverse()
            focal=float(cprim.GetAttribute("focalLength").Get()); ha=float(cprim.GetAttribute("horizontalAperture").Get())
            va=float(cprim.GetAttribute("verticalAperture").Get()); H,W=depth.shape[:2]
            projections=[]; positive=True
            for xx in (.245,.520):
              for yy in (-.155,.150):
                for zz in (.740,.910):
                  q=w2c.Transform(Gf.Vec3d(xx,yy,zz)); d=-float(q[2]); positive &= d>0
                  projections.append((W*(.5+float(q[0])*focal/(ha*d)),H*(.5-float(q[1])*focal/(va*d))))
            corners_in=positive and all(0<=u<W and 0<=v<H for u,v in projections)
            from PIL import Image, ImageDraw
            mask_im=Image.new("1",(W,H),0); ImageDraw.Draw(mask_im).polygon(convex_hull_2d(projections),fill=1)
            mask=np.asarray(mask_im,dtype=bool); denom=int(mask.sum())
            frac=float(np.count_nonzero(mask&np.isfinite(depth)&(depth>0))/denom) if denom else 0.0
            depth_fracs.append(frac)
            fresh_time=(step[0]-1)*dt
            frame_id=int(tensor(camera.frame)[0])
            if frame_rows and (frame_id<=frame_rows[-1]["camera_frame"] or
                               fresh_time<=frame_rows[-1]["native_time_s"]):
                raise RuntimeError("stale RGB-D frame or non-advancing native timestamp")
            frame_rows.append({"native_time_s":fresh_time,"target_index":len(frame_rows),"mask_pixels":denom,
                               "valid_depth_fraction":frac,"corners_visible":bool(corners_in),
                               "camera_frame":frame_id})
            if rgb.size and np.isfinite(rgb[mask]).all() and frac>=.99 and corners_in: valid_frames+=1
        from PIL import Image
        Image.fromarray(np.asarray(rgb)[...,:3].astype(np.uint8)).save(ROOT/"measurements/component-rgb.png")
        np.save(ROOT/"measurements/component-depth.npy",depth)
        corners_visible=int(all(f["corners_visible"] for f in frame_rows))

        credited=len(valid); jaxs=[e["axial"] for e in valid]; joffs=[e["offaxis"] for e in valid]
        shell_initial_breaks=sum(1 for e in broken if int(e["step"])<=hz and e["joint"] in paths["shell_joints"])
        seat_initial_breaks=sum(1 for e in broken if int(e["step"])<=hz and e["joint"] in paths["seat"])
        if not valid:
            raise RuntimeError("nominal qualified episode set is empty; measurements are unavailable")
        if not contact_rows:
            raise RuntimeError("native contact stream is empty; contact-dependent measurements are unavailable")
        if callback_exceptions:
            raise RuntimeError("native callback exceptions: "+"; ".join(callback_exceptions))
        # The aggregate contract also requires three isolated adversarial
        # probes and a YAM robot self-contact probe.  This material worker does
        # not manufacture either result from unrelated joint events or from an
        # absent robot.  Until those native probes are wired, fail closed after
        # persisting all available raw evidence.
        missing_observables=[]
        adversarial_damage_by_probe={}
        if set(adversarial_damage_by_probe)!={"seat-break","shell-break","forbidden-penetration"}:
            missing_observables.append("isolated adversarial seat/shell/penetration event streams")
        robot_contact_impulses=None
        if robot_contact_impulses is None:
            missing_observables.append("YAM unsafe self-contact impulse stream")
        if missing_observables:
            dump(ROOT/f"measurements/raw-component/native-{hz}hz-progress.json",
                 {"status":"incomplete","hz":hz,"native_step":step[0]-1,
                  "missing_observables":missing_observables,
                  "available":{"contact_rows":len(contact_rows),"joint_break_events":broken,
                               "qualified_nominal_episodes":len(valid),"camera_frames":len(frame_rows)}})
            raise RuntimeError("required native probes unavailable: "+"; ".join(missing_observables))

        measurements={
          "initial-settle-station-fixture-linear-speed-maximum":max(fixture_speeds),"initial-settle-loose-body-linear-speed-maximum":max(tool_speeds),
          "initial-untouched-task-score":initial_score,"initial-native-fractured-tang-count":initial_breaks,
          "initial-tray-contained-fragment-count":initial_tray,"initial-shell-integrity-joint-break-count":shell_initial_breaks,
          "initial-insert-seat-joint-break-count":seat_initial_breaks,
          "initial-insert-case-relative-translation-maximum":initial_disp,
          "initial-rigid-body-penetration-maximum":initial_pen,
          "nominal-all-inserts-closed-contact-episode-count-minimum":min(int(e["force_path_steps"]>0) for e in episodes),
          "nominal-native-fractured-tang-count":native_broken,"nominal-released-tang-dynamic-body-count":native_broken,
          "nominal-contact-stream-row-count-total":len(contact_rows),
          "nominal-impact-axis-angle-maximum":max(e["angle_deg"] for e in valid),
          "nominal-impact-radial-offset-maximum":max(e["radial"] for e in valid),
          "nominal-qualified-axial-impulse-minimum":min(jaxs),"nominal-qualified-axial-impulse-maximum":max(jaxs),
          "nominal-qualified-offaxis-impulse-maximum":max(joffs),
          "adversarial-offaxis-impulse-minimum":invalid_off,
          "adversarial-visible-damage-event-count-minimum":min(adversarial_damage_by_probe.values()),
          "retained-insert-case-relative-translation-maximum":axial_disp,
          "retained-insert-case-relative-rotation-maximum":angular,
          "postfracture-forbidden-penetration-maximum":post_pen,"camera-workspace-valid-depth-fraction-minimum":min(depth_fracs),
          "camera-workspace-corners-visible":corners_visible,"terminal-fresh-valid-camera-frame-count-minimum":valid_frames,
          "terminal-continuous-rest-duration-minimum":(frame_rows[-1]["native_time_s"]-frame_rows[0]["native_time_s"]) if valid_frames==41 else 0.0,
          "terminal-tray-contained-fragment-count":tray_count,
          "terminal-loose-body-linear-speed-maximum":float(torch.linalg.vector_norm(tangvel[:,:3],dim=1).max()),
          "terminal-loose-body-angular-speed-maximum":float(torch.linalg.vector_norm(tangvel[:,3:],dim=1).max()),
          "reset-stale-event-count":0,
          "unsafe-robot-self-contact-impulse-maximum":max(robot_contact_impulses)}
        # Frozen incident-work proxy from native impulse and PRE contact-point
        # velocities, including angular terms.
        index={p:i for i,p in enumerate(tracked)}; work=0.0; work_samples=0
        for c in strike_contacts:
            vulnerable=actor=None
            if c["a"] in paths["shell"]+paths["insert"]: vulnerable,actor=c["a"],c["b"]
            elif c["b"] in paths["shell"]+paths["insert"]: vulnerable,actor=c["b"],c["a"]
            if vulnerable not in index or actor not in index or c["step"] not in pre_state: continue
            poses,vels=pre_state[c["step"]]; iv=index[vulnerable]; ia=index[actor]; p=np.asarray(c["p"])
            def point_v(k):
                return np.asarray(vels[k][:3])+np.cross(np.asarray(vels[k][3:6]),p-np.asarray(poses[k][:3]))
            # Preserve sign: opposing motion cannot be turned into incident
            # work by abs().  Orient the stored actor0 impulse onto V.
            j=np.asarray(c["j_native"])
            j_on_v=j if c["a"]==vulnerable else -j
            work+=max(0.0,float(np.dot(j_on_v,point_v(ia)-point_v(iv)))); work_samples+=1
        measurements["shell-incident-contact-work-maximum"]=work
        control_rows=[{"record":"control_outcome","name":"four-force-driven-material-strikes",
                       "start_step":strike_start,"end_step":step[0]-1,
                       "episodes":episodes,"credited_count":credited,
                       "native_broken_count":native_broken,"tray_count":tray_count}]
        jsonl(raw,(state_rows
                   +[{"record":"contact",**c} for c in contact_rows]
                   +[{"record":"legacy_contact_event",**c} for c in legacy_contact_rows]
                   +[{"record":"joint_break",**e} for e in broken]
                   +[{"record":"camera_frame",**e} for e in frame_rows]
                   +[{"record":"fragment_terminal","fragment":i,**e} for i,e in enumerate(fragment_rows)]
                   +control_rows))
        result={"status":"complete","hz":hz,
                "callback_dt_mean":sum(post_callback_dts)/len(post_callback_dts),
                "callback_dt_error_max":max(abs(x-1.0/hz) for x in post_callback_dts),
                "callback_handle_count_errors":[abs(x-2) for x in callback_handle_counts],"measurements":measurements,
                "evidence":{"pre_count":len(pre_steps),"post_count":len(post_steps),"contact_count":len(contact_rows),
                            "legacy_contact_event_count":len(legacy_contact_rows),
                            "joint_break_events":broken,"episodes":episodes,"native_broken":native_broken,
                            "tray_count":tray_count,"incident_work_samples":work_samples,
                            "rgb_shape":list(rgb.shape),"depth_shape":list(depth.shape),
                            "raw_events":str(raw.relative_to(ROOT)),
                            "camera_frames":frame_rows,
                            "initial_insert_poses":insert0.tolist(),"terminal_insert_poses":insert1.tolist(),
                            "initial_tang_poses":tang0.tolist(),"terminal_fragments":fragment_rows},
                "checks":[],"artifacts":[str(raw.relative_to(ROOT))]}
        PhysxManager.deregister_callback(pre_h); PhysxManager.deregister_callback(post_h)
        dump(args.worker_output,result); return 0
    except Exception as e:
        result={"status":"error","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}
        if args.worker_output: dump(args.worker_output,result)
        print(json.dumps(result),file=sys.stderr); return 2
    finally:
        if app is not None: app.close()

def main():
    a=cli()
    try: return worker(a) if a.worker_hz else master(a)
    except Exception as e:
        report={"status":"error","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc(),
                "checks":unavailable_contract_checks(),
                "artifacts":["measurements/raw-component/native-240hz.log",
                             "measurements/raw-component/native-240hz-progress.json"]}
        dump(a.output,report); print(json.dumps(report),file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
