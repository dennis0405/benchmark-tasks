import numpy as np
from yam_arena.task_file import ObjectState
from yam_arena import ik
from yam_arena.station import BASES
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
        self.audit_breaks=Breaks()
        self.visual={}
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
    def rgb(self):
        def array(x):
            x=x.torch if hasattr(x,'torch') else x
            return x[0].cpu().numpy()
        views=[]
        for name in ('overview_cam','top_cam'):
            cam=self.env.unwrapped.scene[name].data
            views.append((array(cam.output['rgb'])[:,:,:3],array(cam.pos_w),array(cam.quat_w_ros),
                          array(cam.intrinsic_matrices)))
        return views
    def after_step(self,step,objects,poses,robot):
        self.q=np.array([robot[s]['joints'] for s in ('left','right')])
        tcp=np.array([robot[s]['tcp'] for s in ('left','right')])
        wrist,rot=ik.tcp_pose(self.q[1],BASES['right'])
        tip=wrist+.078*rot[:,2]
        center=np.asarray(BASE)
        clear=min(np.linalg.norm(tcp-center,axis=1))>.072 and np.linalg.norm(tip-center)>.072
        return dict(hands_clear=bool(clear and not self.hits),contacts=sorted('/'.join(i) for i in self.hits),
                    blade_tip=(tip-center).tolist(),vision=self.visual,physics_audit=self.audit_breaks.fields())
