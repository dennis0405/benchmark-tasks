"""Native mechanism telemetry and disclosed pressure/flow coupling."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from hydraulics import Hydraulics, Parameters
from task import OBJECT_NAMES


class Observer:
    def __init__(self,env):
        self.env=env;self.bodies=None;self.subscription=None;self.contact_subscription=None
        self.hand_impulse=0.
        self.model=Hydraulics();self.last_angle=None;self.frames=0

    def close(self):
        self.subscription=None;self.contact_subscription=None;self.bodies=None

    def reset(self):
        import omni.physics.tensors as tensors
        import omni.physx
        import carb
        from pxr import UsdUtils,UsdPhysics
        self.close()
        stage=self.env.unwrapped.sim.stage
        matches=[str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).endswith('/hydro/lever') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(matches)!=1:raise RuntimeError(f'Expected one pump lever: {matches}')
        self.root=matches[0].rsplit('/',1)[0]
        self.view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        self.view.set_subspace_roots('/')
        self.bodies={n:self.view.create_rigid_body_view(self.root+'/'+n) for n in OBJECT_NAMES}
        if any(b.count!=1 for b in self.bodies.values()):raise RuntimeError('Missing native body')
        self.model=Hydraulics(Parameters(leak_ml_s_bar=0. if self.env.unwrapped.pipe_sound_specimen else .006))
        self.model.reset(self.angle('lever'));self.frames=0
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        self.contact_subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)
        self.subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(self.physics_step)

    def on_contact(self,headers,data):
        from pxr import PhysicsSchemaTools
        for h in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))).lower() for k in ('collider0','collider1')]
            if any('/hydro/' in p for p in paths) and any(any(a in p for a in ('/leftarm/','/rightarm/','/left_robot/','/right_robot/')) for p in paths):
                self.hand_impulse+=sum(float(np.linalg.norm(c.impulse)) for c in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])

    def angle(self,name):
        pose=np.asarray(self.bodies[name].get_transforms()[0],float)
        R=Rotation.from_quat(pose[3:]).as_matrix()
        return math.atan2(-R[2,0],R[0,0]) if name=='lever' else math.atan2(R[1,0],R[0,0])

    def physics_step(self,dt):
        if not self.bodies:return
        q=self.angle('lever'); iso=math.degrees(self.angle('isolation'));drain=math.degrees(self.angle('drain'))
        # Physical seat closure at the end of quarter-turn ball-valve travel.
        iso_open=max(0.,min(1.,(88.-iso)/86.))
        drain_open=max(0.,min(1.,(drain-2.)/86.))
        self.model.advance(float(dt),q,iso_open,drain_open)
        # Cylinder pressure releases only its own stored energy on return.
        # The outlet check prevents the isolated pipe from motoring the lever.
        torque=self.model.resisting_torque(q)
        self.bodies['lever'].apply_forces_and_torques_at_position(
            None,np.array([[0.,torque,0.]],dtype=np.float32),None,
            np.array([0],dtype=np.uint32),True)
        self.frames+=1

    def before_step(self,step,action):
        self.hand_impulse=0.

    def states(self):
        from yam_arena.task_file import ObjectState
        if self.bodies is None:self.reset()
        out,poses={},{}
        bounds={'base':((.07,-.28,0),(.84,.50,.43)),
                'lever':((-.32,-.07,-.015),(.02,.07,.015)),
                'isolation':((-.08,-.02,-.01),(.02,.02,.01)),
                'drain':((-.08,-.02,-.01),(.02,.02,.01)),
                'selector':((-.08,-.02,-.01),(.02,.02,.01))}
        for n,body in self.bodies.items():
            pose=np.asarray(body.get_transforms()[0],float);v=np.asarray(body.get_velocities()[0],float)
            lo,hi=np.array(bounds[n]);R=Rotation.from_quat(pose[3:]).as_matrix()
            center=pose[:3]+R@((hi+lo)/2);half=np.abs(R)@((hi-lo)/2)
            out[n]=ObjectState(center,pose[3:],v[:3],v[3:],center-half,center+half);poses[n]=pose
        return out,poses

    def after_step(self,step,objects,poses,robot):
        m=self.model.state()
        m.update({n+'_deg':math.degrees(self.angle(n)) for n in ('lever','isolation','drain','selector')})
        def distance_to_bounds(tcp,obj):
            return float(np.linalg.norm(np.maximum(np.maximum(obj.low-tcp,tcp-obj.high),0.)))
        m['hand_impulse_Ns']=self.hand_impulse
        m['both_hands_clear']=bool(robot) and self.hand_impulse < 1e-5 and all(
            all(distance_to_bounds(np.array(robot[a]['tcp']),objects[n])>.05 for n in ('lever','isolation','drain','selector'))
            for a in ('left','right'))
        m['physics_frames']=self.frames
        # The needle is cosmetic: only a scalar pressure readout is animated.
        # All colliding mechanism poses remain exclusively PhysX-controlled.
        stage=self.env.unwrapped.sim.stage
        stage.GetPrimAtPath(self.root+'/needle').GetAttribute('xformOp:rotateX').Set(135.-45.*self.model.test_bar)
        stage.GetPrimAtPath(self.root+'/pump_needle').GetAttribute('xformOp:rotateX').Set(135.-45.*self.model.pump_bar)
        return m

    def compensate(self,arm,target):return target.copy()
