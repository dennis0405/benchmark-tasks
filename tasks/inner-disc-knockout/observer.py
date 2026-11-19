import numpy as np
from yam_arena.task_file import ObjectState
from metrics import measure
from task import BASE

class Observer:
    def __init__(self,env):
        self.env=env; self.bodies=None; self.contact=None; self.q=None
    def reset(self):
        import carb,omni.physx,omni.physics.tensors as tensors
        from pxr import UsdUtils,PhysicsSchemaTools
        stage=self.env.unwrapped.sim.stage
        root=next(str(p.GetPath()).rsplit('/',1)[0] for p in stage.Traverse() if str(p.GetPath()).endswith('/knockout/Disc'))
        self.view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        self.view.set_subspace_roots('/')
        self.bodies={n:self.view.create_rigid_body_view(root+'/'+n) for n in ('Disc','Ring')}
        self.paths={}; self.to_path=lambda k:str(PhysicsSchemaTools.intToSdfPath(k)).lower()
        from joints import Breaks
        self.breaks=Breaks()
        self.before_step(0,None);self.q=None
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        self.contact=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.events)
    def close(self):self.contact=None
    def before_step(self,step,action):
        self.hits=set()
    def events(self,headers,data):
        for h in headers:
            if not h.num_contact_data:continue
            for k in (h.collider0,h.collider1):
                if k not in self.paths:self.paths[k]=self.to_path(k)
            a,b=self.paths[h.collider0],self.paths[h.collider1]
            for part,other in ((a,b),(b,a)):
                if '/knockout/' in part and ('_robot/' in other or 'arm/' in other):
                    self.hits.add((part.split('/knockout/')[-1].split('/')[0],other.split('/geometry/')[-1].split('/collisions/')[0]))
    def states(self):
        if self.bodies is None:self.reset()
        objects={};poses={}
        for n,v in self.bodies.items():
            p=v.get_transforms()[0].copy();vel=v.get_velocities()[0].copy();poses[n]=p
            objects[n]=ObjectState(p[:3],p[3:],vel[:3],vel[3:],p[:3]-.02,p[:3]+.02)
        return objects,poses
    def after_step(self,step,objects,poses,robot):
        self.q=np.array([robot[s]['joints'] for s in ('left','right')])
        tcp=np.array([robot[s]['tcp'] for s in ('left','right')])
        m=measure(poses['Disc'].tolist(),poses['Ring'].tolist(),BASE)
        m.update(self.breaks.fields())
        m['contacts']=sorted('/'.join(i) for i in self.hits)
        m['disc_contact']=any(i[0]=='disc' for i in self.hits)
        m['hands_clear']=not self.hits and min(np.linalg.norm(tcp-np.asarray(BASE),axis=1))>.072
        m['hands_clear']=bool(m['hands_clear'])
        return m

