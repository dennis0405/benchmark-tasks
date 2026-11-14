"""Native USD/PhysX construction shared by diagnostics and eventual YAM wiring."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().with_name("procedural_geometry.py")
_s = importlib.util.spec_from_file_location(f"{__name__}_geometry", _p)
if _s is None or _s.loader is None:
    raise ImportError(f"cannot load task geometry: {_p}")
_g = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_g)
INSERT_XY = _g.INSERT_XY
NOTCH_BREAK_FORCE_N = _g.NOTCH_BREAK_FORCE_N
NOTCH_BREAK_TORQUE_NM = _g.NOTCH_BREAK_TORQUE_NM
SEAT_BREAK_FORCE_N = _g.SEAT_BREAK_FORCE_N
SHELL_BREAK_FORCE_N = _g.SHELL_BREAK_FORCE_N

def build(stage, Gf, UsdGeom, UsdPhysics, PhysxSchema, z=0.76):
    paths={"shell":[],"fixture":[],"boss":[],"insert":[],"tang":[],"notch":[],"seat":[],
           "shell_joints":[],"jig_joints":[],"tools":[],"tray":[]}
    def box(path,p,s,mass=None,report=False):
        if mass is None:
            x=UsdGeom.Cube.Define(stage,path); x.CreateSizeAttr(1.0)
            x.AddTranslateOp().Set(Gf.Vec3d(*p)); x.AddScaleOp().Set(Gf.Vec3d(*s))
            UsdPhysics.CollisionAPI.Apply(x.GetPrim()); return x.GetPrim()
        # Rigid/joint frames live on an unscaled Xform.  Collision scale is
        # confined to a child shape so PhysX never has to interpret a joint
        # local frame through a scaled rigid-body root.
        root=UsdGeom.Xform.Define(stage,path); root.AddTranslateOp().Set(Gf.Vec3d(*p))
        shape=UsdGeom.Cube.Define(stage,f"{path}/shape"); shape.CreateSizeAttr(1.0)
        shape.AddScaleOp().Set(Gf.Vec3d(*s)); UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()); UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
        if report: PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.0)
        return root.GetPrim()
    def cylinder(path,p,r,h,mass):
        x=UsdGeom.Cylinder.Define(stage,path); x.CreateRadiusAttr(r); x.CreateHeightAttr(h); x.CreateAxisAttr("Z")
        x.AddTranslateOp().Set(Gf.Vec3d(*p)); UsdPhysics.CollisionAPI.Apply(x.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()); UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr(0.0); return x.GetPrim()
    def joint(path,a,b,force,torque):
        j=UsdPhysics.FixedJoint.Define(stage,path); j.CreateBody0Rel().SetTargets([a.GetPath()]); j.CreateBody1Rel().SetTargets([b.GetPath()])
        # A fixed joint's default frames are both body origins.  Leaving those
        # defaults on separated bodies makes PhysX snap the bodies together at
        # reset.  Author coincident world-space frames from the bodies' local
        # translations so the assembled geometry starts stress free.
        ma=UsdGeom.Xformable(a).ComputeLocalToWorldTransform(0)
        mb=UsdGeom.Xformable(b).ComputeLocalToWorldTransform(0)
        pa=ma.ExtractTranslation()
        pb=mb.ExtractTranslation()
        anchor=(pa+pb)*0.5
        # Joint frames are expressed in each body's complete local frame.  A
        # world-space subtraction is incorrect for scaled Cube prims and made
        # nominally coincident frames disjoint.  Transform the shared anchor
        # through each full inverse local-to-world matrix instead.
        la=ma.GetInverse().Transform(anchor)
        lb=mb.GetInverse().Transform(anchor)
        j.CreateLocalPos0Attr(Gf.Vec3f(*la))
        j.CreateLocalPos1Attr(Gf.Vec3f(*lb))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0,Gf.Vec3f(0.0)))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0,Gf.Vec3f(0.0)))
        j.CreateBreakForceAttr(force); j.CreateBreakTorqueAttr(torque); return str(j.GetPath())
    def world_joint(path, body, force, torque):
        """Attach an unscaled rigid root to world at its authored pose.

        The pinned Isaac Lab implementation authors body0 as the world by
        leaving that relationship empty.  Pointing body0 at the scaled static
        table made every shell joint report a native break on step one.
        """
        j=UsdPhysics.FixedJoint.Define(stage,path)
        j.CreateBody1Rel().SetTargets([body.GetPath()])
        world_pose=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0).RemoveScaleShear()
        j.CreateLocalPos0Attr(Gf.Vec3f(*world_pose.ExtractTranslation()))
        j.CreateLocalRot0Attr(Gf.Quatf(world_pose.ExtractRotationQuat()))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0,Gf.Vec3f(0.0)))
        j.CreateBreakForceAttr(force); j.CreateBreakTorqueAttr(torque)
        return str(j.GetPath())
    def hollow(path,p,outer,wall,height,mass=None,report=False):
        root=UsdGeom.Xform.Define(stage,path); root.AddTranslateOp().Set(Gf.Vec3d(*p))
        if mass is not None:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()); UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
            if report: PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.0)
        half=(outer-wall)/2.0
        for n,lp,s in (("n",(0,half,0),(outer,wall,height)),("s",(0,-half,0),(outer,wall,height)),
                       ("e",(half,0,0),(wall,outer-2*wall,height)),("w",(-half,0,0),(wall,outer-2*wall,height))):
            c=UsdGeom.Cube.Define(stage,f"{path}/{n}"); c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(*lp)); c.AddScaleOp().Set(Gf.Vec3d(*s)); UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        return root.GetPrim()
    # A bounded work surface is also the optical background for every ray in
    # the declared camera workspace hull.  The earlier 0.70 x 0.55 m slab
    # ended underneath the high rear-corner rays, producing native +inf depth
    # for roughly one quarter of the frozen mask.
    table=box("/World/table",(.38,0,z-.02),(1.10,.90,.04))
    box("/World/workcell_back",(.69,0,z+.18),(.02,.90,.40))
    # The jig is a stationary assembly, but it only supports the portable
    # workpiece by collision.  Its named world joints constrain the jig bodies,
    # never the case, and make fixture motion observable through a native rigid
    # body view (a static collider has no tensor velocity state).
    for n,p,s in (("left",(.340,.047,z+.004),(.190,.004,.008)),
                  ("right",(.340,-.047,z+.004),(.190,.004,.008)),
                  ("back",(.247,0,z+.004),(.004,.086,.008))):
        fixture=box(f"/World/jig/{n}",p,s,.08,True)
        paths["fixture"].append(str(fixture.GetPath()))
        paths["jig_joints"].append(
            world_joint(f"/World/joints/J_jig_{n}",fixture,1000.0,25.0)
        )
    shell_parts=[]
    for n,p,s in (("floor",(.340,0,z+.003),(.180,.090,.006)),("wl",(.340,.044,z+.012),(.180,.003,.024)),
                  ("wr",(.340,-.044,z+.012),(.180,.003,.024)),("wf",(.429,0,z+.012),(.003,.084,.024)),
                  ("wb",(.251,0,z+.012),(.003,.084,.024))):
        shell_parts.append(box(f"/World/shell/{n}",p,s,.025 if n=="floor" else .006,True)); paths["shell"].append(str(shell_parts[-1].GetPath()))
    # Four real structural joints make the five-part 49 g shell portable.
    # The floor is the case root; no case body is attached to world.
    for k,p in enumerate(shell_parts[1:]):
        paths["shell_joints"].append(joint(f"/World/joints/J_shell_{k}",shell_parts[0],p,SHELL_BREAK_FORCE_N,3.0))
    for i,(x,y) in enumerate(INSERT_XY):
        boss=hollow(f"/World/boss_{i}",(x,y,z+.015),.014,.002,.018,.0015,True); paths["boss"].append(str(boss.GetPath()))
        paths["shell_joints"].append(joint(f"/World/joints/J_shell_boss_{i}",shell_parts[0],boss,SHELL_BREAK_FORCE_N,3.0))
        # Retained insert is dynamic and seated by a high-strength native joint.
        ins=hollow(f"/World/insert_{i}",(x,y,z+.025),.0048,.0006,.00375,.00022,True); paths["insert"].append(str(ins.GetPath()))
        tang=box(f"/World/tang_{i}",(x,y,z+.0271),(.0032,.00065,.00045),.000035,True); paths["tang"].append(str(tang.GetPath()))
        paths["seat"].append(joint(f"/World/joints/J_seat_{i}",shell_parts[0],ins,SEAT_BREAK_FORCE_N,.12))
        paths["notch"].append(joint(f"/World/joints/J_tang_{i}",ins,tang,NOTCH_BREAK_FORCE_N,NOTCH_BREAK_TORQUE_NM))
    # Compound tools: parts are separate collision bodies joined natively.
    ms=cylinder("/World/mandrel_shaft",(.255,-.135,z+.036),.001,.070,.025)
    mh=cylinder("/World/mandrel_handle",(.255,-.135,z+.091),.007,.040,.030)
    joint("/World/joints/J_mandrel",ms,mh,500,10)
    hh=box("/World/hammer_head",(.350,.145,z+.024),(.030,.018,.016),.105,True)
    hd=cylinder("/World/hammer_handle",(.350,.145,z+.083),.006,.105,.075)
    joint("/World/joints/J_hammer",hh,hd,500,10); paths["tools"]=[str(ms.GetPath()),str(mh.GetPath()),str(hh.GetPath()),str(hd.GetPath())]
    for n,p,s in (("floor",(.475,-.112,z+.003),(.080,.070,.006)),("l",(.475,-.078,z+.020),(.080,.006,.040)),
                  ("r",(.475,-.146,z+.020),(.080,.006,.040)),("f",(.514,-.112,z+.020),(.006,.064,.040)),
                  ("b",(.436,-.112,z+.020),(.006,.064,.040))): paths["tray"].append(str(box(f"/World/tray/{n}",p,s).GetPath()))
    return paths
