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
    joint.CreateLowerLimitAttr(-1440.0)
    joint.CreateUpperLimitAttr(1440.0)
    return joint


def main():
    hz=240
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
    lower=create_plate(stage,"LowerPlate",.5-.0071)
    upper=create_plate(stage,"UpperPlate",.5+.0071)
    mount=UsdPhysics.FixedJoint.Define(stage,path+"/mount")
    mount.CreateBody1Rel().SetTargets([frame])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0,0,.5))
    create_revolute(stage,path+"/screw_rotation",frame,screw)
    slide=create_prismatic(stage,path+"/axial",screw,carrier,-.004,.004)
    turn=create_revolute(stage,path+"/relative_turn",carrier,nut)
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
    sim.reset()
    view=PhysxManager.get_physics_sim_view()
    view.set_subspace_roots("/")
    art=view.create_articulation_view(path)
    names=list(art.get_metatype(0).dof_names)
    assert art.max_fixed_tendons==1 and art.count==1
    pair=view.create_rigid_contact_view([str(nut)],filter_patterns=[[str(lower),str(upper)]],max_contact_data_count=64)
    ids=wp.array([0],dtype=wp.uint32,device="cuda:0")
    rr=names.index("relative_turn")
    zz=names.index("axial")
    tr=[]
    for direction, label in ((1,"positive"),(-1,"negative")):
        for phase,steps in (("ramp",240),("hold",480),("release",240)):
            for i in range(steps):
                torque=direction*.005*(min(1,(i+1)/240) if phase=="ramp" else int(phase=="hold"))
                efforts=np.zeros((1,3),dtype=np.float32)
                efforts[0,rr]=torque
                art.set_dof_actuation_forces(wp.array(efforts,device="cuda:0"),ids)
                sim.step(render=False)
                PhysxManager.raise_callback_exception_if_any()
                q=art.get_dof_positions().numpy()[0]
                native=art.get_dof_actuation_forces().numpy()[0,rr]
                force=np.linalg.norm(pair.get_contact_force_matrix(dt=1/hz).numpy().reshape(2,3),axis=1)
                tr.append(dict(direction=label,phase=phase,torque_nm=float(native),z=float(q[zz]),
                               theta=float(q[rr]),e=float(q[zz]-lead*q[rr]),plate_n=float(sum(force))))
    medians={}
    checks={}
    for direction,label in ((1,"positive"),(-1,"negative")):
        rows=[r for r in tr if r["phase"]=="hold" and r["direction"]==label][-240:]
        m={key:float(np.median([r[key] for r in rows])) for key in ("z","theta","e","plate_n","torque_nm")}
        medians[label]=m
        checks[label+"_travel"]=direction*m["z"]>5e-5
        checks[label+"_rotation"]=direction*m["theta"]>.05
        checks[label+"_opposing_torque"]=-direction*m["e"]>5e-5
        checks[label+"_plate_force"]=m["plate_n"]>=.1
        checks[label+"_off_rotary_limit"]=abs(m["theta"])<math.radians(1439)
        checks[label+"_static_band"]=abs(abs(m["e"])-.005/(lead*1e5))<=.35*.005/(lead*1e5)
    args.output.with_suffix(".trace.json").write_text(json.dumps(tr,allow_nan=False))
    args.output.write_text(json.dumps(dict(gearing=args.gearing,medians=medians,checks=checks,all_pass=all(checks.values()),
        scope="independent bidirectional component diagnostic",expected_static_abs_e_m=.005/(lead*1e5)),indent=2))


try:
    main()
finally:
    app.close()
