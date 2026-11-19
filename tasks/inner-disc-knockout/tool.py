"""Flange-clamped flat blade, with native collision geometry and no moving fixture."""
from isaaclab.sim.utils import clone
from yam_arena.robot import spawn_in_dream_look, asset_link_paths
@clone
def spawn_arm_with_blade(prim_path,cfg,*args,**kwargs):
    from pxr import UsdGeom as G,UsdPhysics as P,Gf,PhysxSchema as X
    prim=spawn_in_dream_look.__wrapped__(prim_path,cfg,*args,**kwargs)
    stage=prim.GetStage()
    root=prim_path+asset_link_paths()['gripper']+'/FlatDriver'
    G.Xform.Define(stage,root)
    for name,z,length,width,depth in [('Holder',.143,.048,.017,.013),('Shaft',.181,.036,.004,.004),('Blade',.205,.014,.0055,.0012)]:
        g=G.Cube.Define(stage,root+'/'+name);g.CreateSizeAttr(1.)
        g.AddTranslateOp().Set(Gf.Vec3d(0,0,z));g.AddScaleOp().Set(Gf.Vec3f(width,depth,length))
        g.CreateDisplayColorAttr([Gf.Vec3f(*((.15,.23,.29) if name=='Holder' else (.42,.46,.49)))])
        P.CollisionAPI.Apply(g.GetPrim());a=X.PhysxCollisionAPI.Apply(g.GetPrim());a.CreateContactOffsetAttr(.00025);a.CreateRestOffsetAttr(0.)
    return prim

