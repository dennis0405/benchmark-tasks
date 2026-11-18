"""Build the task-local USD from the dimensions of the tested rigid gimbal."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT=Path(__file__).resolve().parent
ASSET=ROOT/'gimbal.usda'
PIVOT=Gf.Vec3d(0,0,.145)

def numbers(x):
    return tuple(float(v) for v in x.split())

def build():
    ASSET.parent.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(ASSET))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1)
    root=UsdGeom.Xform.Define(stage,'/Gimbal')
    stage.SetDefaultPrim(root.GetPrim())
    material=UsdShade.Material.Define(stage,'/Gimbal/GripMaterial')
    friction=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction.CreateStaticFrictionAttr(1.1)
    friction.CreateDynamicFrictionAttr(.85)
    friction.CreateRestitutionAttr(0.)
    stand=ET.parse(ROOT/'gimbal.xml').find('.//body[@name="stand"]')
    for node in stand.iter('body'):
        name=node.attrib['name'];path='/Gimbal/'+name
        obj=UsdGeom.Xform.Define(stage,path)
        obj.AddTranslateOp().Set(PIVOT)
        rb=UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        obj.GetPrim().AddAppliedSchema('PhysxContactReportAPI')
        obj.GetPrim().CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
        if name=='stand':rb.CreateKinematicEnabledAttr(True)
        else:
            inert=node.find('inertial')
            mass=UsdPhysics.MassAPI.Apply(obj.GetPrim())
            if inert is not None:
                mass.CreateMassAttr(float(inert.attrib['mass']))
                mass.CreateCenterOfMassAttr(Gf.Vec3f(*numbers(inert.attrib['pos'])))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*numbers(inert.attrib['diaginertia'])))
            else:
                mass.CreateMassAttr(.036)
                mass.CreateCenterOfMassAttr(Gf.Vec3f(-.0525,.0325,-.037))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.000011,.000107,.000112))
        for i,geo in enumerate(node.findall('geom')):
            a=geo.attrib; shape=path+'/'+a.get('name',f'part_{i}');dims=numbers(a['size'])
            if a['type']=='box':
                g=UsdGeom.Cube.Define(stage,shape);g.CreateSizeAttr(2.)
                g.AddTranslateOp().Set(Gf.Vec3d(*numbers(a['pos'])))
                g.AddScaleOp().Set(Gf.Vec3f(*dims))
            else:
                endpoints=numbers(a['fromto'])
                start=Gf.Vec3d(*endpoints[:3]);end=Gf.Vec3d(*endpoints[3:])
                g=UsdGeom.Cylinder.Define(stage,shape)
                g.CreateRadiusAttr(dims[0]);g.CreateHeightAttr((end-start).GetLength())
                g.CreateAxisAttr('X' if end[0]!=start[0] else 'Y')
                g.AddTranslateOp().Set((start+end)*.5)
            g.CreateDisplayColorAttr([Gf.Vec3f(*numbers(a['rgba'])[:3])])
            if a.get('contype')!='0':
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material,materialPurpose='physics')
                # Millimeter scale pads: PhysX's generic centimeter-scale skin merges the trigger gap.
                g.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
                g.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.00045)
                g.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
    for n,kind,parent,child,axis,lo,hi in [
        ('tilt',UsdPhysics.RevoluteJoint,'stand','cradle','Y',-55.,55.),
        ('slide',UsdPhysics.PrismaticJoint,'cradle','camera','X',-.048,.04),
        ('release',UsdPhysics.PrismaticJoint,'cradle','brake','Y',-.001,.017)]:
        joint=kind.Define(stage,'/Gimbal/'+n);joint.CreateAxisAttr(axis);joint.CreateLowerLimitAttr(lo);joint.CreateUpperLimitAttr(hi)
        joint.CreateBody0Rel().SetTargets(['/Gimbal/'+parent]);joint.CreateBody1Rel().SetTargets(['/Gimbal/'+child])
        for attr in (joint.CreateLocalPos0Attr(),joint.CreateLocalPos1Attr()):attr.Set(Gf.Vec3f(0))
        joint.CreateCollisionEnabledAttr(False)
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular' if n=='tilt' else 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(600. if n=='release' else 0.)
        drive.CreateTargetPositionAttr(-.030 if n=='release' else 0.)
        drive.CreateDampingAttr({'tilt':.04*math.pi/180,'slide':.055,'release':1.}[n])
    stage.GetRootLayer().Save()
if __name__=='__main__':build()
