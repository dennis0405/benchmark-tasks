import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
from yam_arena.task_file import ObjectState
from task import BASE,OBJECT_NAMES
from scoring import MIN_CLOSING_SPEED

class Observer:
    def __init__(self,env):
        self.env=env;self.bodies=None;self.q=None;self.contact=None;self.sub=None
    def reset(self):
        import carb,omni.physx,omni.physics.tensors as tensors
        from pxr import UsdUtils,UsdPhysics,PhysicsSchemaTools
        self.contact=self.sub=None
        stage=self.env.unwrapped.sim.stage
        root=next(str(p.GetPath()).rsplit('/',1)[0] for p in stage.Traverse() if str(p.GetPath()).endswith('/wheel/Crown'))
        view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        view.set_subspace_roots('/');self.view=view
        self.bodies={n:view.create_rigid_body_view(root+'/'+n) for n in OBJECT_NAMES}
        self.names={}; self.load=0.;self.ramp=0.;self.proof_started=False
        self.speed_history=deque(maxlen=8);self.ever_hit=False;self.trial_over=False;self.start_z=None;self.peak_speed=0.;self.closing=0.
        self.force=np.zeros((1,3),np.float32);self.index=np.array([0],np.uint32)
        self.getpath=lambda ident:str(PhysicsSchemaTools.intToSdfPath(ident)).lower()
        self.before_step(0,None)
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        self.contact=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.events)
        self.sub=omni.physx.get_physx_interface().subscribe_physics_step_events(self.substep)
    def close(self):self.contact=self.sub=None
    def substep(self,dt):
        # A rolling 4 ms window at 2 kHz expires old samples even outside the scripted swing.
        crown_vz = self.bodies['Crown'].get_velocities()[0, 2]
        hammer_vz = self.bodies['Hammer'].get_velocities()[0, 2]
        self.speed_history.append(float(crown_vz - hammer_vz))
        if self.proof_started and not self.trial_over:
            z=float(self.bodies['Crown'].get_transforms()[0,2])
            if self.start_z is None:self.start_z=z
            if abs(z-self.start_z)>.012 or self.ramp>5.6:
                self.trial_over=True;self.load=0.;return
            self.ramp+=dt
            self.load=min(4.,self.ramp*2.)
            self.force[0,2]=self.load
            anchor=self.bodies['Crown'].get_transforms()[:,:3].copy()
            self.bodies['Crown'].apply_forces_and_torques_at_position(self.force,None,anchor,self.index,True)
            if self.load>=4.:self.loaded+=dt
    def before_step(self,step,action):
        self.step=step;self.impact_speed=0.
        self.fast_contact_impulse = 0.
        self.rack_supports = set()
        self.pads={'weight':set(),'hammer':set()};self.robot_hit=False;self.fixture_hits=set();self.strike=0.;self.rim=False;self.loaded=0.
    def events(self,headers,data):
        for h in headers:
            if not h.num_contact_data:continue
            ids=(h.collider0,h.collider1)
            for id_ in ids:
                if id_ not in self.names:self.names[id_]=self.getpath(id_)
            a,b=[self.names[id_] for id_ in ids]
            for obj,other in ((a,b),(b,a)):
                if '/wheel/' not in obj:continue
                if obj.endswith('/hammer/nose') and other.endswith('/crown/bridge'):
                    self.ever_hit=True
                    closing_speed = max(self.speed_history, default=0.)
                    impulse = sum(float(np.linalg.norm(data[k].impulse)) for k in
                                  range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data))
                    self.impact_speed = max(self.impact_speed, closing_speed)
                    self.strike += impulse
                    if closing_speed > MIN_CLOSING_SPEED:
                        self.fast_contact_impulse += impulse
                if '/hammer/' in obj and other.endswith(('/hammerheadrest', '/hammerrearrest')):
                    # Count load-bearing top contacts, ignoring grazing contact with a side.
                    for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                        contact = data[k]
                        if abs(contact.normal[2]) > 0.8 and abs(contact.impulse[2]) > 0:
                            self.rack_supports.add(other.rsplit('/', 1)[-1])
                            break
                if obj.endswith('/lip') and ('/toe' in other or '/spring' in other):self.rim=True
                if '_robot/' not in other and 'arm/' not in other:continue
                self.fixture_hits.add(other.split('/Geometry/')[-1].split('/collisions/')[0][-32:]+':'+obj.split('/wheel/')[-1])
                if '/crown/' in obj or '/hammer/' in obj or '/left/' in obj or '/right/' in obj:self.robot_hit=True
                for side,key,target in [('left','weight','/crown/'),('right','hammer','/hammer/handle')]:
                    if target in obj and side in other.split('/tip_')[0] and 'tip_' in other:
                        self.pads[key].add('left' if 'tip_left' in other else 'right')
    def states(self):
        if self.bodies is None:self.reset()
        out={};poses={}
        for n,b in self.bodies.items():
            p=b.get_transforms()[0].copy();v=b.get_velocities()[0].copy();poses[n]=p
            out[n]=ObjectState(p[:3],p[3:],v[:3],v[3:],p[:3]-.01,p[:3]+.01)
        return out,poses
    def after_step(self,step,objects,poses,robot):
        p=poses['Crown'][:3]-BASE;h=poses['Hammer'][:3]-BASE
        self.q=np.array([robot[s]['joints'] for s in ('left','right')])
        tcp=np.array([robot[s]['tcp'] for s in ('left','right')])-BASE
        toes=[]
        for name,sg in [('Left',-1),('Right',1)]:
            loc=poses[name]
            toes.append(loc[:3]-BASE+R.from_quat(loc[3:]).apply([-sg*.00205,0,-.0082]))
        on_mark=abs(p[0])<.006 and abs(p[1])<.006 and .098<p[2]<.111
        under=(toes[0][0]<0<toes[1][0] and all(t[2]<.0954 and abs(t[0])<.0037 for t in toes))
        handle=h+R.from_quat(poses['Hammer'][3:]).apply([0,-.065,.01])
        clear=not self.robot_hit and np.linalg.norm(tcp[0]-p)>.055 and np.linalg.norm(tcp[1]-handle)>.045 and np.linalg.norm(h-p)>.06
        # The tester starts on a fixed schedule even if installation failed.
        if step>=355 and clear:self.proof_started=True
        hammer = objects['Hammer']
        parked = (len(self.rack_supports) == 2 and np.linalg.norm(hammer.lin_vel) < 0.01
                  and np.linalg.norm(hammer.ang_vel) < 0.25)
        return dict(hammer_parked=bool(parked), rack_supports=sorted(self.rack_supports),
                    fast_contact_impulse=self.fast_contact_impulse, crown=p.tolist(),hammer=h.tolist(),tcp=tcp.tolist(),toes=np.asarray(toes).tolist(),fixture_hits=sorted(self.fixture_hits),angle=R.from_quat(poses['Crown'][3:]).as_euler('xyz',degrees=True).tolist(),weight_held=len(self.pads['weight'])==2,hammer_held=len(self.pads['hammer'])==2,pads={k:sorted(v) for k,v in self.pads.items()},rim_contact=self.rim,off_mark=bool(abs(p[1])>.045),on_mark=bool(on_mark),under_lip=bool(under),impact_speed=self.impact_speed,strike_impulse=self.strike,both_clear=bool(clear),pull_N=self.load,full_load=self.loaded>.049)
