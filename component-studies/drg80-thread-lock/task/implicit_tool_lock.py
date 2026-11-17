"""M8x1 locking seat test under tightening torque then complete tool release.

Diagnostic fixture only: fixed screw, original bearing carrier and no YAM. The
grounded cover has four contact pads and, in elastic mode, a flexure spring.
Neither cover position nor retained preload is controlled by the task scorer.
"""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cover", choices=["rigid", "elastic"], required=True)
parser.add_argument("--fixture", choices=["coupon","task"], default="task")
parser.add_argument("--origin-shift",type=float,default=0.0)
parser.add_argument("--solver",choices=["tgs","pgs"],default="tgs")
parser.add_argument("--velocity-iters", type=int, default=1)
parser.add_argument("--torque", type=float, default=.012)
parser.add_argument("--hz", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import warp as wp
from isaaclab_physx.physics import PhysxManager
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
from shared_frame_fixture import make_fixture


def run():
    sim, stage, nut, nut_material = make_fixture(args.hz, args.device, args.fixture, args.velocity_iters, int(args.solver=="tgs"),args.origin_shift)
    path = "/World/ScrewGuide/frame"
    top_local=.0144-.00030
    for i, (x, y) in enumerate(((.006, 0), (-.006, 0), (0,.006), (0,-.006))):
        pad = UsdGeom.Cube.Define(stage, path + "/Pad" + str(i))
        pad.CreateSizeAttr(1.)
        pad.AddTranslateOp().Set(Gf.Vec3d(x,y,top_local-.0005))
        pad.AddScaleOp().Set(Gf.Vec3f(.002,.002,.001))
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        contact = PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
        contact.CreateContactOffsetAttr(.00015)
        contact.CreateRestOffsetAttr(0.)
    # Include the 30 g representative piston with its actual stop-rod profile.
    from shared_frame_fixture import create_body
    # Positive-volume convex contact land projects 0.60 mm below the original
    # mesh. This is an authored bearing face of the adjuster, not an end limit.
    land=UsdGeom.Cylinder.Define(stage,"/World/ScrewGuide/screw/StopLand")
    land.CreateAxisAttr("Z");land.CreateRadiusAttr(.0032);land.CreateHeightAttr(.0006)
    land.AddTranslateOp().Set(Gf.Vec3d(0,0,-.0003))
    UsdPhysics.CollisionAPI.Apply(land.GetPrim())
    face=PhysxSchema.PhysxCollisionAPI.Apply(land.GetPrim())
    face.CreateContactOffsetAttr(.0001);face.CreateRestOffsetAttr(0.)
    piston=create_body(stage,"/World/ScrewGuide/piston",Gf.Vec3d(0,0,.4992-args.origin_shift),
                       .03,Gf.Vec3f(1.5e-6,1.5e-6,1e-6))
    for name,rad,height,center in (("Head",.008,.020,-.015),("Rod",.003,.006,-.003)):
        part=UsdGeom.Cylinder.Define(stage,"/World/ScrewGuide/piston/"+name)
        part.CreateAxisAttr("Z");part.CreateRadiusAttr(rad);part.CreateHeightAttr(height)
        part.AddTranslateOp().Set(Gf.Vec3d(0,0,center))
        UsdPhysics.CollisionAPI.Apply(part.GetPrim())
        c=PhysxSchema.PhysxCollisionAPI.Apply(part.GetPrim())
        c.CreateContactOffsetAttr(.0001);c.CreateRestOffsetAttr(0.)
    PhysxSchema.PhysxContactReportAPI.Apply(piston.GetPrim()).CreateThresholdAttr(0.)
    pj=UsdPhysics.PrismaticJoint.Define(stage,"/World/ScrewGuide/piston_axis")
    pj.CreateBody0Rel().SetTargets(["/World/ScrewGuide/frame"])
    pj.CreateBody1Rel().SetTargets([piston.GetPath()])
    pj.CreateAxisAttr("Z");pj.CreateLocalPos0Attr(Gf.Vec3f(0,0,-.0008))
    pj.CreateLowerLimitAttr(-.005);pj.CreateUpperLimitAttr(.005)
    # Bounded external socket-driver compliance while a tool is present.
    # PhysX integrates this spring implicitly. Its maximum force is zero
    # before tool engagement and after complete withdrawal.
    hold_drive=UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath("/World/ScrewGuide/rotation"),"angular")
    hold_drive.CreateTypeAttr("force")
    hold_drive.CreateStiffnessAttr(.2*math.pi/180)
    hold_drive.CreateDampingAttr(.00025*math.pi/180)
    hold_drive.CreateTargetPositionAttr(0.)
    hold_drive.CreateTargetVelocityAttr(0.)
    hold_drive.CreateMaxForceAttr(0.)
    sim.reset()
    phys = PhysxManager.get_physics_sim_view()
    phys.set_subspace_roots("/")
    art = phys.create_articulation_view("/World/ScrewGuide")
    dofs = list(art.get_metatype(0).dof_names)
    ri, zi = dofs.index("nut_rotation"), dofs.index("nut_axial")
    screw = art
    screw_dofs = list(screw.get_metatype(0).dof_names)
    si, saz = screw_dofs.index("rotation"), screw_dofs.index("axial")
    screw_body = phys.create_rigid_body_view(["/World/ScrewGuide/screw"])
    nut_body = phys.create_rigid_body_view([str(nut.GetPath())])
    assert screw.count == nut_body.count == screw_body.count == 1
    seat = phys.create_rigid_body_view([path])
    contact = phys.create_rigid_contact_view([str(nut.GetPath())],filter_patterns=[[path]],max_contact_data_count=4096)
    assert contact.sensor_count == contact.filter_count == seat.count == 1
    ids = wp.array([0],dtype=wp.uint32,device=args.device)
    if art.max_dofs!=5 or art.max_fixed_tendons!=2:
        raise RuntimeError(f"Expected two native fixed tendons in the four-DOF articulation, got {art.max_dofs} DOFs and {art.max_fixed_tendons} tendons")
    pi=dofs.index("piston_axis")
    stop_contact=phys.create_rigid_contact_view(["/World/ScrewGuide/piston"],filter_patterns=[["/World/ScrewGuide/screw"]],max_contact_data_count=1024)
    drive_limits=art.get_dof_max_forces().numpy().copy()
    observed_limit=float(drive_limits[0,si])
    assert observed_limit==0.
    cpu_indices=wp.array([0],dtype=wp.uint32,device="cpu")
    samples = []
    last_speed = 0.
    last_screw_angle, last_screw_speed = 0.,0.
    for k in range(14*args.hz):
        t = k/args.hz
        phase = "settle" if t < .5 else "tighten" if t < 9 else "released" if t<12 else "proof_load"
        # Speed-dependent bounded wrench torque emulates a tool with a finite
        # no-load speed. Removing the wrench sets BOTH applied efforts to zero.
        applied = max(-args.torque, min(0., args.torque*(-3.-last_speed)/3.)) if phase == "tighten" else 0.
        forces = np.zeros((1,art.max_dofs),dtype=np.float32)
        forces[0,ri] = applied
        requested_limit=.025 if phase=="tighten" else 0.
        if observed_limit!=requested_limit and abs(observed_limit-requested_limit)>1e-8:
            drive_limits[0,si]=requested_limit
            art.set_dof_max_forces(wp.array(drive_limits,dtype=wp.float32,device="cpu"),cpu_indices)
            observed_limit=float(art.get_dof_max_forces().numpy()[0,si])
            if abs(observed_limit-requested_limit)>1e-7:
                raise RuntimeError("native driver effort bound mismatch")
        driver_torque=0.  # external generalized torque below is wrench-only
        # Absolute nut wrench does work on theta_S and the relative nut coordinate.
        forces[0,si] = driver_torque + applied
        forces[0,pi] = min(2.,max(0.,(t-12.)*4.)) if phase=="proof_load" else 0.
        screw.set_dof_actuation_forces(wp.array(forces,device=args.device),ids)
        sim.step(render=False)
        qs=screw.get_dof_positions().numpy()[0]
        vs=screw.get_dof_velocities().numpy()[0]
        last_screw_angle, last_screw_speed=float(qs[si]),float(vs[si])
        def world_yaw(body):
            _,_,z,w=map(float,body.get_transforms().numpy()[0,3:])
            return 2*math.atan2(z,w)

        q = art.get_dof_positions().numpy()[0]
        v = art.get_dof_velocities().numpy()[0]
        z = float(seat.get_transforms().numpy()[0,2])+top_local
        normal = float(abs(contact.get_contact_force_matrix(dt=1./args.hz).numpy().reshape(3)[2]))
        effort=art.get_dof_actuation_forces().numpy()[0,pi]
        stop_force=float(abs(stop_contact.get_contact_force_matrix(dt=1/args.hz).numpy().reshape(3)[2]))
        last_speed = float(v[ri]+v[si])
        samples.append(dict(step=k+1,phase=phase,torque_nm=applied,angle_rad=float(q[ri]+q[si]),
                            omega_rad_s=last_speed,screw_angle_rad=last_screw_angle,screw_z_m=float(qs[saz]),
                            screw_world_yaw=world_yaw(screw_body),nut_world_yaw=world_yaw(nut_body),
                            driver_limit_nm=observed_limit,driver_torque_nm=driver_torque,proof_force_n=float(effort),piston_stop_n=stop_force,stop_face_gap_m=.0002+float(q[saz])-float(q[pi]),piston_q_m=float(q[pi]),axial_m=float(q[zi]+q[saz]),cover_z_m=z,
                            seat_gap_m=.5144-args.origin_shift+float(q[zi]+q[saz])-z,normal_n=normal))
    released = [x for x in samples if x["phase"]=="released"][-args.hz:]
    proof = [x for x in samples if x["phase"]=="proof_load"]
    def rate(rows,field):
        diff=np.diff(np.array([x[field] for x in rows]))
        return float(np.max(abs(diff))*args.hz)
    nut_rate=rate(released,"nut_world_yaw")
    screw_rate=rate(released,"screw_world_yaw")
    above=[i for i,row in enumerate(samples) if row["proof_force_n"]>1]
    if not above: raise ValueError("missing energized piston window")
    proof_all=samples[above[0]-1:above[-1]+1]
    settling=[x for x in samples if x["phase"]=="settle"]
    checks = dict(unpowered_reset_seat_force=max(x["normal_n"] for x in settling)<.05,
                  unpowered_shaft_reset_drift=max(abs(x["screw_z_m"]) for x in settling)<.00005,
                  unpowered_reset_angle_motion=max(abs(x["screw_angle_rad"]) for x in settling)<math.radians(1),
                  both_parts_seated=min(x["normal_n"] for x in released)>=2,
                  released_driver_limit_zero=all(x["driver_limit_nm"]==0 for x in released+proof),
                  released_no_tool=all(x["torque_nm"]==0 and x["driver_torque_nm"]==0 for x in released+proof),
                  nut_stationary=nut_rate<=math.radians(1),
                  screw_stationary=screw_rate<=math.radians(1),
                  zero_drive_release=all(x["proof_force_n"]==0 for x in released),
                  real_load=all(abs(x["proof_force_n"]-2)<1e-5 for x in proof[-args.hz:]),
                  piston_not_at_travel_limit=max(abs(x["piston_q_m"]) for x in proof)<.0048,
                  no_stop_face_penetration=all(x["stop_face_gap_m"]>=-.00010 for x in samples),
                  native_piston_stop=min(x["piston_stop_n"] for x in proof[-args.hz:])>=.1,
                  seat_gap=all(-.0001<=x["seat_gap_m"]<=.00050 for x in released+proof),
                  powered_screw_angular_rate=rate(proof_all,"screw_angle_rad")<=.01,
                  powered_nut_angular_rate=rate(proof_all,"angle_rad")<=.01,
                  powered_screw_axial_rate=rate(proof_all,"screw_z_m")<=.00001,
                  maintains_preload=min(x["normal_n"] for x in proof)>=2)
    result=dict(piston_stop_min_N=min(x["piston_stop_n"] for x in proof[-args.hz:]),origin_shift=args.origin_shift,solver=args.solver,hz=args.hz,torque_limit_nm=args.torque,checks=checks,overall_pass=all(checks.values()),
                released_min_n=min(x["normal_n"] for x in released),proof_min_n=min(x["normal_n"] for x in proof),
                peak_seat_n=max(x["normal_n"] for x in samples),
                released_world_deg_s=[math.degrees(screw_rate),math.degrees(nut_rate)],
                proof_rad_s=[rate(proof_all,"screw_angle_rad"),rate(proof_all,"angle_rad")],
                proof_screw_m_s=rate(proof_all,"screw_z_m"),final=samples[-1])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix(".trace.json").write_text(json.dumps(samples,allow_nan=False))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)

try:
    run()
finally:
    app.close()
