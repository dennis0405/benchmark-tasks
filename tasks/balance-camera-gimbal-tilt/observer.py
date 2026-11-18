"""Read-only telemetry for the four native bodies, robot contacts, and public scoring."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from task import OBJECT_NAMES

class Observer:
    def __init__(self,env):
        import omni.physx
        self.env=env
        self.bodies=None
        self.impulse=0.
        self.robot_contacts_seen=set()
        self.subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)

    def close(self):
        self.subscription=None
        self.bodies=None

    def reset(self):
        import carb
        import omni.physics.tensors as tensors
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        from pxr import UsdUtils,UsdPhysics
        stage=self.env.unwrapped.sim.stage
        matches=[str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).endswith('/gimbal/cradle') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(matches)!=1:raise RuntimeError(f'Expected one articulated cradle: {matches}')
        self.view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        self.view.set_subspace_roots('/')
        root=matches[0].rsplit('/',1)[0]
        self.bodies={n:self.view.create_rigid_body_view(root+'/'+n) for n in OBJECT_NAMES}
        if any(body.count!=1 for body in self.bodies.values()):
            raise RuntimeError('Expected one native rigid body per gimbal component')
        self.robot_contacts_seen=set()
        self.impulse=0.

    def before_step(self,step,action):
        self.impulse=0.

    def on_contact(self,headers,data):
        from pxr import PhysicsSchemaTools
        for h in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))).lower() for k in ('collider0','collider1')]
            if not any('/gimbal/' in p for p in paths):
                continue
            for side in ('left','right'):
                if any('/'+side+'arm/' in p or '/'+side+'_robot/' in p for p in paths):
                    impulse=sum(float(np.linalg.norm(c.impulse)) for c in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])
                    self.impulse+=impulse
                    if impulse>0:
                        self.robot_contacts_seen.add(side)

    def states(self):
        from yam_arena.task_file import ObjectState
        if self.bodies is None:self.reset()
        out,poses={},{}
        # Local bounding boxes of the authored solids, in their rigid-body frame.
        bounds={'stand':((-.110,-.105,-.145),(.110,.134,.049)),
                'cradle':((-.122,-.101,-.066),(.073,.121,.014)),
                'camera':((-.080,-.093,-.038),(.090,.038,.027)),
                'brake':((-.121,.014,-.058),(.018,.051,-.025))}
        for n in OBJECT_NAMES:
            body=self.bodies[n]
            pose=np.asarray(body.get_transforms()[0],dtype=float)
            v=np.asarray(body.get_velocities()[0],dtype=float)
            lo,hi=np.array(bounds[n]);R=Rotation.from_quat(pose[3:]).as_matrix()
            midpoint=pose[:3]+R@((hi+lo)/2);half=np.abs(R)@((hi-lo)/2)
            out[n]=ObjectState(midpoint,pose[3:],v[:3],v[3:],midpoint-half,midpoint+half)
            poses[n]=pose
        return out,poses

    def measure(self,objects,poses):
        c=np.array(poses['cradle']);R=Rotation.from_quat(c[3:]).as_matrix()
        base=Rotation.from_quat(poses['stand'][3:]).as_matrix()
        relative=base.T@R
        return dict(pitch_deg=math.degrees(math.atan2(-relative[2,0],relative[0,0])),
                    pitch_speed_deg_s=math.degrees((R.T@objects['cradle'].ang_vel)[1]),
                    slide_m=float((R.T@(np.array(poses['camera'][:3])-c[:3]))[0]),
                    brake_m=float((R.T@(np.array(poses['brake'][:3])-c[:3]))[1]))

    def after_step(self,step,objects,poses,robot):
        m=self.measure(objects,poses)
        # Both TCPs must withdraw outside the entire tool. Contact also checks non-TCP robot links.
        pivot=np.array(poses['cradle'][:3])
        m.update(robot_contacts_seen=sorted(self.robot_contacts_seen),
                 both_hands_clear=bool(self.robot_contacts_seen) and all(np.linalg.norm(np.array(robot[arm]['tcp'])-pivot)>.145 for arm in ('left','right')),
                 hand_impulse_Ns=self.impulse)
        return m

    def compensate(self,arm,target):
        # YamArmAction already supplies gravity feedforward at every physics substep.
        # Adding torque/stiffness to these targets would displace the commanded grasp.
        return target.copy()
