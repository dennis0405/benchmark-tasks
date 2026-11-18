"""Original full-size bench hydrostatic apparatus. PXR-only asset construction."""
from pathlib import Path
import math
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT=Path(__file__).parent
PIVOTS={'lever':(.58,-.05,.18), 'isolation':(.39,.24,.15),
        'drain':(.25,.24,.15), 'selector':(.27,.43,.13)}
COLORS={'blue':(.10,.30,.63),'steel':(.58,.62,.65),'brass':(.63,.43,.13),
        'black':(.07,.08,.09),'red':(.65,.08,.05),'copper':(.70,.37,.19),
        'white':(.87,.87,.82)}


def build():
    stage=Usd.Stage.CreateNew(str(ROOT/'pump.usda'))
    UsdGeom.SetStageUpAxis(stage,'Z'); UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Hydro'); stage.SetDefaultPrim(root.GetPrim())
    mat=UsdShade.Material.Define(stage,'/Hydro/Material')
    api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(.9);api.CreateDynamicFrictionAttr(.7);api.CreateRestitutionAttr(0.)

    def body(name, pos, mass=None, axis=None, degrees=0):
        obj=UsdGeom.Xform.Define(stage,'/Hydro/'+name)
        obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if axis: obj.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(*axis),degrees).GetQuat()))
        rigid=UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        if mass is None: rigid.CreateKinematicEnabledAttr(True)
        else: UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(mass)
        obj.GetPrim().AddAppliedSchema('PhysxContactReportAPI')
        obj.GetPrim().CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
        return '/Hydro/'+name

    def finish(g,color,collide=True):
        g.CreateDisplayColorAttr([Gf.Vec3f(*COLORS.get(color,color))])
        if collide:
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat,materialPurpose='physics')
            g.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
            g.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0005)
            g.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
        return g

    def box(path,pos,size,color,collide=True):
        g=UsdGeom.Cube.Define(stage,path);g.CreateSizeAttr(1.)
        g.AddTranslateOp().Set(Gf.Vec3d(*pos));g.AddScaleOp().Set(Gf.Vec3f(*size))
        return finish(g,color,collide)

    def tube(path,a,b,r,color,collide=True):
        a,b=Gf.Vec3d(*a),Gf.Vec3d(*b);d=b-a
        g=UsdGeom.Cylinder.Define(stage,path);g.CreateRadiusAttr(r);g.CreateHeightAttr(d.GetLength())
        g.AddTranslateOp().Set((a+b)/2)
        g.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),d.GetNormalized()).GetQuat()))
        return finish(g,color,collide)

    def label(name, text, center, size, vertical=False):
        from PIL import Image, ImageDraw, ImageFont
        image=Image.new('RGB',(768,512),(238,238,223));d=ImageDraw.Draw(image)
        lines=text.split('\n')
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        for font_size in range(300,20,-2):
            font=ImageFont.truetype(font_path,font_size)
            if max(d.textlength(line,font=font) for line in lines)<700 and font_size*1.3*len(lines)<440:break
        for i,line in enumerate(lines):d.text((384,256+(i-(len(lines)-1)/2)*font_size*1.3),line,font=font,fill=(20,25,30),anchor='mm')
        image.save(ROOT/(name+'.png'))
        shader_mat=UsdShade.Material.Define(stage,'/Hydro/'+name+'_material')
        surf=UsdShade.Shader.Define(stage,'/Hydro/'+name+'_material/surface');surf.CreateIdAttr('UsdPreviewSurface')
        tex=UsdShade.Shader.Define(stage,'/Hydro/'+name+'_material/tex');tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(name+'.png'))
        reader=UsdShade.Shader.Define(stage,'/Hydro/'+name+'_material/uv');reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        surf.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
        surf.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8)
        shader_mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(),'surface')
        x,y,z=center;w,h=size
        pts=[(x,y+w/2,z-h/2),(x,y-w/2,z-h/2),(x,y-w/2,z+h/2),(x,y+w/2,z+h/2)] if vertical else [(x-w/2,y-h/2,z),(x+w/2,y-h/2,z),(x+w/2,y+h/2,z),(x-w/2,y+h/2,z)]
        mesh=UsdGeom.Mesh.Define(stage,base+'/'+name)
        mesh.CreatePointsAttr(pts);mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([(0,0),(1,0),(1,1),(0,1)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(shader_mat)

    base=body('base',(0,0,0))
    # 440 x 200 mm, 4.5-litre reservoir envelope; double-wall reservoir is
    # represented by floor and four walls, not a solid block pretending to be a pump.
    box(base+'/tank_floor',(.44,-.05,.018),(.44,.20,.025),'blue')
    for j,y in enumerate((-.145,.045)): box(base+f'/tank_side{j}',(.44,y,.073),(.44,.01,.10),'blue')
    for j,x in enumerate((.225,.655)): box(base+f'/tank_end{j}',(x,-.05,.073),(.01,.20,.10),'blue')
    box(base+'/water',(.44,-.05,.08),(.41,.17,.002),(.15,.39,.56),False)
    tube(base+'/pump_cylinder',(.58,-.05,.04),(.58,-.05,.17),.025,'brass')
    for j,y in enumerate((-.075,-.025)): box(base+f'/fulcrum{j}',(.58,y,.155),(.035,.012,.085),'steel')
    tube(base+'/axle',(.58,-.095,.18),(.58,-.005,.18),.008,'steel')
    lever=body('lever',PIVOTS['lever'],.22,(0,1,0),0)
    tube(lever+'/shaft',(0,0,0),(-.30,0,0),.009,'steel')
    tube(lever+'/grip',(-.30,-.055,0),(-.30,.055,0),.013,'black')
    # Valve bodies and fixture are bolted to the bench as a service manifold.
    box(base+'/manifold_plate',(.32,.24,.025),(.29,.13,.035),'steel')
    tube(base+'/manifold',(.19,.24,.12),(.45,.24,.12),.013,'brass')
    for name in ('isolation','drain','selector'):
        x,y,z=PIVOTS[name]
        tube(base+'/'+name+'_stem',(x,y,.04),(x,y,z-.005),.013,'brass')
        b=body(name,(x,y,z),.06,(0,0,1),0)
        box(b+'/arm',(-.035,0,0),(.09,.019,.012),'red' if name=='drain' else 'blue')
        tube(b+'/hub',(0,0,-.008),(0,0,.008),.016,'steel')
        # Actual pin stops, plus joint limits; no pose-triggered valve switch.
    box(base+'/report_plate',(.27,.43,.018),(.15,.14,.022),'white')
    # 10 m OD35/ID32 copper coil. 10.25 turns, pitch38 mm, radius155 mm.
    # Pipe wall compliance is lumped in hydraulics.py; collision envelope is rigid.
    turns=10.25; steps=360
    for i in range(steps):
        def pt(j):
            t=turns*2*math.pi*j/steps
            return (.66+.155*math.cos(t),.32+.155*math.sin(t),.035+.038*t/(2*math.pi))
        tube(base+f'/coil_{i:03}',pt(i),pt(i+1),.0175,'copper').CreateVisibilityAttr('invisible')
    # Smooth continuous outer surface; segment cylinders above remain physical
    # collision envelopes only. No manipulation enters the pipe lumen.
    vertices=[];normals=[];faces=[];sides=16
    for i in range(steps+1):
        t=turns*2*math.pi*i/steps
        c=Gf.Vec3d(*pt(i));radial=Gf.Vec3d(math.cos(t),math.sin(t),0)
        tangent=Gf.Vec3d(-.155*math.sin(t),.155*math.cos(t),.038/(2*math.pi)).GetNormalized()
        binormal=Gf.Cross(tangent,radial).GetNormalized()
        for j in range(sides):
            a=2*math.pi*j/sides;n=radial*math.cos(a)+binormal*math.sin(a)
            vertices.append(c+.0175*n);normals.append(n)
    for i in range(steps):
        for j in range(sides):faces.extend([i*sides+j,i*sides+(j+1)%sides,(i+1)*sides+(j+1)%sides,(i+1)*sides+j])
    smooth=UsdGeom.Mesh.Define(stage,base+'/coil_surface');smooth.CreatePointsAttr(vertices)
    smooth.CreateFaceVertexCountsAttr([4]*(steps*sides));smooth.CreateFaceVertexIndicesAttr(faces)
    smooth.CreateNormalsAttr(normals);smooth.SetNormalsInterpolation('vertex');finish(smooth,'copper',False)
    for j,x in enumerate((.56,.76)): box(base+f'/coil_support{j}',(x,.32,.012),(.04,.36,.024),'black')
    # Braided hose path between fixed pump and fixed manifold; flexible motion is
    # irrelevant here because both fittings are already installed at reset.
    pts=[(.58,-.05,.12),(.58,0.,.12),(.52,.11,.12),(.36,.17,.12),(.27,.24,.12)]
    for i,(a,b) in enumerate(zip(pts,pts[1:])):tube(base+f'/hose{i}',a,b,.009,'black')
    feed=[(.45,.24,.12),(.45,.14,.035),(.815,.14,.035),(.815,.32,.035)]
    for i,(a,b) in enumerate(zip(feed,feed[1:])):tube(base+f'/feed{i}',a,b,.0175,'copper')
    end=pt(steps)
    tube(base+'/specimen_end_cap',end,(end[0]-.035,end[1],end[2]),.019,'brass')
    tube(base+'/return',(.25,.24,.105),(.30,.02,.105),.008,'black')
    tube(base+'/gauge_stem',(.46,.10,.12),(.46,.10,.25),.009,'brass')
    tube(base+'/gauge_branch',(.46,.10,.12),(.45,.24,.12),.009,'brass')
    # Gauge facing the robot. Its needle is an uncollidable visual indicator only.
    tube(base+'/gauge_back',(.46,.10,.25),(.48,.10,.25),.055,'steel')
    tube(base+'/gauge_face',(.455,.10,.25),(.459,.10,.25),.050,'white',False)
    for i in range(31):
        a=math.radians(-135+9*i)
        y,z=.10+.043*math.sin(a),.25+.043*math.cos(a)
        box(base+f'/tick{i}',(.453,y,z),(.001,.003 if i%5==0 else .0015,.003 if i%5==0 else .0015),'black',False)
    needle=UsdGeom.Xform.Define(stage,'/Hydro/needle');needle.AddTranslateOp().Set(Gf.Vec3d(.451,.10,.25))
    needle.AddRotateXOp().Set(135.)
    box('/Hydro/needle/pointer',(0,0,.02),(.001,.002,.040),'red',False)
    label('test_card','PIPE LEAK TEST\n3.0 bar / 60 s\nDROP > 0.2 bar: FAIL',(.12,.02,.035),(.19,.14))
    label('isolation_label','ISOLATE\nOPEN - SHUT',(.39,.31,.075),(.10,.07))
    label('drain_label','RETURN\nSHUT - OPEN',(.25,.31,.075),(.10,.07))
    label('result_label','RESULT',(.27,.49,.14),(.12,.04))
    label('pass_label','PASS',(.195,.475,.14),(.065,.024))
    label('fail_label','FAIL',(.195,.385,.14),(.065,.024))
    label('gauge_units','bar\n0 - 6',(.452,.10,.225),(.055,.025),True)
    for i in range(7):
        a=math.radians(-135+45*i)
        label('number_'+str(i),str(i),(.450,.10+.034*math.sin(a),.25+.034*math.cos(a)),(.012,.012),True)
    # Pump-side dial is upstream of isolation. Its own needle follows cylinder pressure.
    x,y,z=.55,-.115,.24
    tube(base+'/pump_gauge_stem',(x,y,.115),(x,y,z),.006,'brass')
    tube(base+'/pump_gauge_feed',(.58,-.05,.115),(x,y,.115),.006,'brass')
    tube(base+'/pump_gauge_back',(x,y,z),(x+.014,y,z),.035,'steel')
    tube(base+'/pump_gauge_face',(x-.004,y,z),(x-.001,y,z),.033,'white',False)
    for i in range(31):
        a=math.radians(-135+9*i)
        box(base+f'/pump_tick{i}',(x-.005,y+.029*math.sin(a),z+.029*math.cos(a)),
            (.001,.0018,.0018),'black',False)
        if i%5==0:
            label('pump_number_'+str(i//5),str(i//5),(x-.006,y+.023*math.sin(a),z+.023*math.cos(a)),(.007,.007),True)
    pump_needle=UsdGeom.Xform.Define(stage,'/Hydro/pump_needle')
    pump_needle.AddTranslateOp().Set(Gf.Vec3d(x-.008,y,z));pump_needle.AddRotateXOp().Set(135.)
    box('/Hydro/pump_needle/pointer',(0,0,.012),(.001,.0015,.024),'red',False)
    label('pump_units','PUMP bar',(x-.007,y,z-.016),(.029,.009),True)
    # Sealed specimen cartridge and low-pressure discharge line to the tank.
    tube(base+'/leak_cartridge',(.815,.14,.035),(.815,.09,.035),.009,'steel')
    tube(base+'/leak_return',(.815,.09,.035),(.64,.018,.09),.003,'black')
    for name in PIVOTS:
        j=UsdPhysics.RevoluteJoint.Define(stage,'/Hydro/'+name+'_joint')
        j.CreateBody0Rel().SetTargets([base]);j.CreateBody1Rel().SetTargets(['/Hydro/'+name])
        j.CreateAxisAttr('Y' if name=='lever' else 'Z')
        j.CreateLocalPos0Attr(Gf.Vec3f(*PIVOTS[name]));j.CreateLocalPos1Attr(Gf.Vec3f(0))
        j.CreateLowerLimitAttr(0. if name!='selector' else -35.)
        j.CreateUpperLimitAttr(50. if name=='lever' else 35. if name=='selector' else 90.)
        j.CreateCollisionEnabledAttr(False)
        drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular')
        drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.)
        drive.CreateDampingAttr(.04 if name == "lever" else .003)
    stage.GetRootLayer().Save()

if __name__=='__main__': build()
