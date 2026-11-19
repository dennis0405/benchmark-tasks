"""A raised removable base, cooled cake, and a split metal ring with a rolled lip. SI units."""
import math
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Sdf

CX=.32
TOP=.76
R=.075
BODY=(.37,.43,.46)
CAKE=(.96,.80,.55)
CRUST=(.56,.30,.12)
def collision(shape, friction=.48):
    p=shape.GetPrim()
    UsdPhysics.CollisionAPI.Apply(p)
    p.AddAppliedSchema('PhysxCollisionAPI')
    p.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.00010)
    p.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
    stage=p.GetStage()
    mat=UsdShade.Material.Define(stage,'/Materials/m'+str(round(friction*100)))
    api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction);api.CreateDynamicFrictionAttr(friction*.9)
    api.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,materialPurpose='physics')
def body(stage,path,pos,mass,kin=False):
    b=UsdGeom.Xform.Define(stage,path)
    tr=b.AddTranslateOp();tr.Set(Gf.Vec3d(*pos))
    a=UsdPhysics.RigidBodyAPI.Apply(b.GetPrim());a.CreateKinematicEnabledAttr(kin)
    UsdPhysics.MassAPI.Apply(b.GetPrim()).CreateMassAttr(mass)
    return tr
def box(stage,path,pos,dims,color,rotation=0,collide=True):
    b=UsdGeom.Cube.Define(stage,path);b.CreateSizeAttr(1)
    b.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rotation:b.AddRotateZOp().Set(rotation)
    b.AddScaleOp().Set(Gf.Vec3f(*dims));b.CreateDisplayColorAttr([color])
    if collide:collision(b)
    return b
def disk(stage,path,pos,r,h,color,collide=True):
    b=UsdGeom.Cylinder.Define(stage,path);b.CreateRadiusAttr(r);b.CreateHeightAttr(h)
    b.AddTranslateOp().Set(Gf.Vec3d(*pos));b.CreateDisplayColorAttr([color])
    if collide:collision(b)
    return b
def contents(stage,root):
    base=body(stage,root+'/base',(CX,0,TOP+.009),.115)
    disk(stage,root+'/base/plate',(0,0,0),R,.002,BODY)
    disk(stage,root+'/base/raised_foot',(0,0,-.005),.062,.008,BODY)
    cake=body(stage,root+'/cake',(CX,0,TOP+.034),.48)
    disk(stage,root+'/cake/filling',(0,0,.0025),.073,.043,CAKE)
    disk(stage,root+'/cake/crust',(0,0,-.0215),.073,.005,CRUST)
    return base,cake
def sector(stage,path,i):
    # Four 90-degree bands, each including ten straight facets and the lower rolled bead.
    # The seam opens with the band. Each chord is tangent to the stated inside radius.
    for j in range(10):
        deg=90*i+(j-4.5)*9
        u,v=math.cos(math.radians(deg)),math.sin(math.radians(deg))
        for label,r,t,h,z in [('wall',.077,.0016,.060,.0315),('bead',.074,.0046,.002,.006)]:
            rad=r+t/2
            box(stage,path+f'/{label}{j}',(rad*u,rad*v,z),
                (t,2*rad*math.tan(math.radians(4.5))+.0001,h),BODY,deg)
