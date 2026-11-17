"""Measure the relative screw/nut compliant thread against two load plates.

The native fixed tendon represents e = z_rel - pitch_per_rad * theta_rel.
Uses the fixture masses, stiffness and damping. This diagnostic earns no task points.
"""
import argparse
import json
import math
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--mechanism", choices=("tendon", "implicit-ports", "coupled-effort"), default="tendon")
p.add_argument("--gap",type=float,default=.0001)
p.add_argument("--hz", type=int, choices=(240,480), default=240)
p.add_argument("--magnitude", type=float, default=.005)
p.add_argument("--rotary-limit-deg", type=float, default=1440.)
p.add_argument("--observed-load",action="store_true")
p.add_argument("--gearing", choices=("per-degree", "per-radian"), default="per-degree")
args=p.parse_args()
app=AppLauncher(headless=True).app
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
import numpy as np
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxManager
def create_body(
    stage,
    path: str,
    mass_kg: float,
    inertia: tuple[float, float, float],
    *,
    collision: bool = False,
):
    body = UsdGeom.Cylinder.Define(stage, path)
    body.CreateAxisAttr("Z")
    body.CreateRadiusAttr(0.004)
    body.CreateHeightAttr(0.012)
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    if collision:
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateDisableGravityAttr(True)
    return body.GetPath()


def create_plate(stage, name: str, z_m: float):
    plate = UsdGeom.Cube.Define(stage, f"/World/{name}")
    plate.CreateSizeAttr(1.0)
    plate.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_m))
    plate.AddScaleOp().Set(Gf.Vec3f(0.025, 0.025, 0.002))
    rigid = UsdPhysics.RigidBodyAPI.Apply(plate.GetPrim())
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    UsdPhysics.MassAPI.Apply(plate.GetPrim()).CreateMassAttr(100.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(plate.GetPrim()).CreateDisableGravityAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(plate.GetPrim()).CreateThresholdAttr(0.0)
    return plate.GetPath()


def create_prismatic(stage, path: str, body0, body1, lower: float, upper: float):
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    return joint


def create_revolute(stage, path: str, body0, body1):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-args.rotary_limit_deg)
    joint.CreateUpperLimitAttr(args.rotary_limit_deg)
    return joint


def main():
    hz=args.hz
    lead=.001/(2*math.pi)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/hz,device="cuda:0",render_interval=12))
    stage=sim.stage
    path="/World/Coupling"
    root=UsdGeom.Xform.Define(stage,path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    solver=PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    solver.CreateSolverPositionIterationCountAttr(128)
    solver.CreateSolverVelocityIterationCountAttr(8)
    frame=create_body(stage,path+"/frame",1.,(.001,)*3)
    screw=create_body(stage,path+"/screw",.009,(5e-7,5e-7,7e-8))
    carrier=create_body(stage,path+"/carrier",.000001,(1e-12,)*3)
    nut=create_body(stage,path+"/nut",.009,(1.4e-7,1.4e-7,2.6e-7),collision=True)
    lower=create_plate(stage,"LowerPlate",.5-.007-args.gap)
    upper=create_plate(stage,"UpperPlate",.5+.007+args.gap)
    mount=UsdPhysics.FixedJoint.Define(stage,path+"/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0,0,.5))
    create_revolute(stage,path+"/screw_rotation",frame,screw)
    slide=create_prismatic(stage,path+"/axial",screw,carrier,-.004,.004)
    turn=create_revolute(stage,path+"/relative_turn",carrier,nut)
    if args.mechanism=="tendon":
        name="nut_thread"
        tendon=PhysxSchema.PhysxTendonAxisRootAPI.Apply(slide.GetPrim(),name)
        tendon.CreateStiffnessAttr(1e5)
        tendon.CreateDampingAttr(2500.)
        tendon.CreateRestLengthAttr(0.)
        axis=PhysxSchema.PhysxTendonAxisAPI(tendon,name)
        axis.CreateGearingAttr([1.])
        axis.CreateForceCoefficientAttr([1.])
        axis=PhysxSchema.PhysxTendonAxisAPI.Apply(turn.GetPrim(),name)
        axis.CreateGearingAttr([-.001/360. if args.gearing=="per-degree" else -lead])
        axis.CreateForceCoefficientAttr([-lead])
    elif args.mechanism=="implicit-ports":
        # Core angular DriveAPI uses force per degree; native tensor targets use radians.
        for joint,kind,scale in ((slide,"linear",1.),(turn,"angular",lead*lead*math.pi/180)):
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),kind)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(1e5*scale)
            drive.CreateDampingAttr(2500.*scale)
            drive.CreateMaxForceAttr(float("inf"))
    sim.reset()
    view=PhysxManager.get_physics_sim_view()
    view.set_subspace_roots("/")
    art=view.create_articulation_view(path)
    names=list(art.get_metatype(0).dof_names)
    assert art.max_fixed_tendons==int(args.mechanism=="tendon") and art.count==1
    pair=view.create_rigid_contact_view([str(nut)],filter_patterns=[[str(lower),str(upper)]],max_contact_data_count=64)
    ids=wp.array([0],dtype=wp.uint32,device="cuda:0")
    rr=names.index("relative_turn")
    zz=names.index("axial")
    k_native=art.get_dof_stiffnesses().numpy()[0].tolist()
    d_native=art.get_dof_dampings().numpy()[0].tolist()
    screw_body=view.create_rigid_body_view([str(screw)])
    nut_body=view.create_rigid_body_view([str(nut)])
    z_reset=float(nut_body.get_transforms().numpy()[0,2]-screw_body.get_transforms().numpy()[0,2])
    from compliant_thread import Thread
    law=Thread(lead,1e5,2500.)
    top=view.create_rigid_body_view([str(upper)])
    bottom=view.create_rigid_body_view([str(lower)])
    previous_yaw=None
    world_yaw=0.
    previous_speed=0.
    previous_thread_force=0.
    tr=[]
    for direction, label in ((1,"positive"),(-1,"negative")):
        for phase,steps in (("ramp",hz),("hold",2*hz),("release",hz)):
            for i in range(steps):
                torque=direction*args.magnitude*(min(1,(i+1)/hz) if phase=="ramp" else int(phase=="hold"))
                efforts=np.zeros((1,3),dtype=np.float32)
                efforts[0,rr]=torque
                thread_force=0.
                if args.mechanism=="coupled-effort":
                    now=art.get_dof_positions().numpy()[0]
                    speeds=art.get_dof_velocities().numpy()[0]
                    inv_m=1/.009001
                    inv_i=1/7e-8+1/2.6e-7
                    zpos=float(nut_body.get_transforms().numpy()[0,2])
                    upper_face=float(top.get_transforms().numpy()[0,2])-.001
                    lower_face=float(bottom.get_transforms().numpy()[0,2])+.001
                    w_now=float(speeds[rr])
                    external_delta_w=inv_i*torque/hz
                    if args.observed_load:
                        external_delta_w=w_now-previous_speed-inv_i*lead*previous_thread_force/hz
                    free_w=w_now+external_delta_w
                    thread_force=law.effort(float(now[zz]-lead*now[rr]),float(speeds[zz]),free_w,
                        inv_m,inv_i,1/hz,(lower_face+.006-zpos)*hz,(upper_face-.006-zpos)*hz)
                    previous_speed=w_now
                    previous_thread_force=thread_force
                    efforts[0,zz]=-thread_force
                    efforts[0,rr]+=lead*thread_force
                if args.mechanism=="implicit-ports":
                    # Each port follows the *other measured coordinate*. No task endpoint or
                    # contact classification enters the drive. The native contact solve
                    # determines both joint positions. This is a first-order split scheme.
                    q_old=art.get_dof_positions().numpy()[0]
                    v_old=art.get_dof_velocities().numpy()[0]
                    q_ref=np.zeros_like(efforts)
                    v_ref=np.zeros_like(efforts)
                    q_ref[0,zz]=lead*(q_old[rr]+v_old[rr]/hz)
                    q_ref[0,rr]=(q_old[zz]+v_old[zz]/hz)/lead
                    v_ref[0,zz]=lead*v_old[rr]
                    v_ref[0,rr]=v_old[zz]/lead
                    art.set_dof_position_targets(wp.array(q_ref,device="cuda:0"),ids)
                    art.set_dof_velocity_targets(wp.array(v_ref,device="cuda:0"),ids)
                art.set_dof_actuation_forces(wp.array(efforts,device="cuda:0"),ids)
                sim.step(render=False)
                PhysxManager.raise_callback_exception_if_any()
                q=art.get_dof_positions().numpy()[0]
                native=art.get_dof_actuation_forces().numpy()[0,rr]
                force=np.linalg.norm(pair.get_contact_force_matrix(dt=1/hz).numpy().reshape(2,3),axis=1)
                sx=screw_body.get_transforms().numpy()[0]
                nx=nut_body.get_transforms().numpy()[0]
                z_world=float(nx[2]-sx[2])-z_reset
                yaw=2*(math.atan2(float(nx[5]),float(nx[6]))-math.atan2(float(sx[5]),float(sx[6])))
                world_yaw+=math.remainder(yaw-(previous_yaw or 0.),math.tau)
                previous_yaw=yaw
                vz=art.get_dof_velocities().numpy()[0]
                tr.append(dict(direction=label,phase=phase,torque_nm=torque,native_turn_effort_nm=float(native),thread_force_n=thread_force,z=float(q[zz]),
                               theta=float(q[rr]),z_world_m=z_world,theta_world_rad=world_yaw,
                               screw_rad=float(q[names.index("screw_rotation")]),edot=float(vz[zz]-lead*vz[rr]),e=float(q[zz]-lead*q[rr]),plate_n=float(sum(force))))
    medians={}
    checks={}
    for direction,label in ((1,"positive"),(-1,"negative")):
        rows=[r for r in tr if r["phase"]=="hold" and r["direction"]==label][-hz:]
        m={key:float(np.median([r[key] for r in rows])) for key in ("z","theta","e","plate_n","torque_nm")}
        medians[label]=m
        checks[label+"_travel"]=direction*m["z"]>5e-5
        checks[label+"_rotation"]=direction*m["theta"]>.05
        checks[label+"_opposing_torque"]=-direction*m["e"]>5e-5
        checks[label+"_plate_force"]=m["plate_n"]>=.1
        checks[label+"_off_rotary_limit"]=abs(m["theta"])<math.radians(args.rotary_limit_deg-1)
        checks[label+"_static_band"]=abs(abs(m["e"])-args.magnitude/(lead*1e5))<=.35*args.magnitude/(lead*1e5)
        if args.mechanism!="tendon":
            checks[label+"_force_band"]=abs(m["plate_n"]-args.magnitude/lead)<=.35*args.magnitude/lead
            release=[abs(r["e"]) for r in tr if r["phase"]=="release" and r["direction"]==label][-hz//4:]
            checks[label+"_unloads"]=max(release)<=20e-6
    for row in tr:
        if not all(math.isfinite(value) for value in row.values() if isinstance(value,float)):
            raise ValueError("nonfinite native output")
    checks["physical_axis_agrees_with_joint"]=max(abs(r["z_world_m"]-r["z"]) for r in tr)<5e-6
    checks["physical_yaw_agrees_with_joint"]=max(abs(r["theta_world_rad"]-r["theta"]) for r in tr)<.002
    args.output.with_suffix(".trace.json").write_text(json.dumps(tr,allow_nan=False))
    args.output.write_text(json.dumps(dict(observed_load=args.observed_load,mechanism=args.mechanism,hz=hz,dof_names=names,native_stiffness=k_native,native_damping=d_native,torque=args.magnitude,rotary_limit_deg=args.rotary_limit_deg,gearing=args.gearing,medians=medians,checks=checks,all_pass=all(checks.values()),
        scope="independent bidirectional component diagnostic",expected_static_abs_e_m=args.magnitude/(lead*1e5)),indent=2))


try:
    main()
finally:
    app.close()
