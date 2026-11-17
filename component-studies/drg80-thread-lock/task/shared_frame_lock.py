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
    if art.max_dofs!=4 or art.max_fixed_tendons!=2:
        raise RuntimeError(f"Expected two native fixed tendons in the four-DOF articulation, got {art.max_dofs} DOFs and {art.max_fixed_tendons} tendons")
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
        driver_torque = float(np.clip(-.025*last_screw_angle-.0004*last_screw_speed,-.025,.025)) if phase=="tighten" else 0.
        # Absolute nut wrench does work on theta_S and the relative nut coordinate.
        forces[0,si] = driver_torque + applied
        forces[0,saz] = 2.0 if phase=="proof_load" else 0.0
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
        last_speed = float(v[ri]+v[si])
        samples.append(dict(step=k+1,phase=phase,torque_nm=applied,angle_rad=float(q[ri]+q[si]),
                            omega_rad_s=last_speed,screw_angle_rad=last_screw_angle,screw_z_m=float(qs[saz]),
                            screw_world_yaw=world_yaw(screw_body),nut_world_yaw=world_yaw(nut_body),
                            driver_torque_nm=driver_torque,proof_force_n=float(forces[0,saz]),axial_m=float(q[zi]+q[saz]),cover_z_m=z,
                            seat_gap_m=.5144-args.origin_shift+float(q[zi]+q[saz])-z,normal_n=normal))
    released = [x for x in samples if x["phase"]=="released"][-args.hz:]
    proof = [x for x in samples if x["phase"]=="proof_load"]
    def rate(rows,field):
        diff=np.diff(np.array([x[field] for x in rows]))
        return float(np.max(abs(diff))*args.hz)
    nut_rate=rate(released,"nut_world_yaw")
    screw_rate=rate(released,"screw_world_yaw")
    proof_all=[released[-1],*proof]
    settling=[x for x in samples if x["phase"]=="settle"]
    checks = dict(unpowered_reset_seat_force=max(x["normal_n"] for x in settling)<.05,
                  unpowered_shaft_reset_drift=max(abs(x["screw_z_m"]) for x in settling)<.00005,
                  unpowered_reset_angle_motion=max(abs(x["screw_angle_rad"]) for x in settling)<math.radians(1),
                  both_parts_seated=min(x["normal_n"] for x in released)>=2,
                  released_no_tool=all(x["torque_nm"]==0 and x["driver_torque_nm"]==0 for x in released+proof),
                  nut_stationary=nut_rate<=math.radians(1),
                  screw_stationary=screw_rate<=math.radians(1),
                  zero_drive_release=all(x["proof_force_n"]==0 for x in released),
                  real_load=all(x["proof_force_n"]==2 for x in proof),
                  seat_gap=all(-.0001<=x["seat_gap_m"]<=.00050 for x in released+proof),
                  powered_screw_angular_rate=rate(proof_all,"screw_angle_rad")<=.01,
                  powered_nut_angular_rate=rate(proof_all,"angle_rad")<=.01,
                  powered_screw_axial_rate=rate(proof_all,"screw_z_m")<=.00001,
                  maintains_preload=min(x["normal_n"] for x in proof)>=2)
    result=dict(origin_shift=args.origin_shift,solver=args.solver,hz=args.hz,torque_limit_nm=args.torque,checks=checks,overall_pass=all(checks.values()),
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
