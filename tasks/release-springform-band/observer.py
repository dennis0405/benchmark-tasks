"""Read native pose, speed, lever angle and radial opening without moving any task body."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena.task_file import ObjectState
from separation import measure
from cake_contact import manifold_peaks, touches_exposed_cake
NAMES=('base','cake','band','lever','sector0','sector1','sector2','sector3')
class Observer:
    def __init__(self,env):self.env=env;self.bodies=None;self.joints=None
    def close(self):self.bodies=None;self.subscription=None
    def reset(self):
        import carb,omni.physx
        import omni.physics.tensors as tensors
        from pxr import UsdUtils
        st=self.env.unwrapped.sim.stage
        base=next(str(p.GetPath()) for p in st.Traverse() if str(p.GetPath()).endswith('/pan_fixture/band')).rsplit('/',1)[0]
        self.view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(st).ToLongInt())
        self.view.set_subspace_roots('/')
        self.bodies={n:self.view.create_rigid_body_view(base+'/'+n) for n in NAMES}
        self.joints=None;self.origin=None;self.band_pose=None
        self.contacts=set()
        self.active_pairs=set()
        self.cake_sensor_ready = False
        self.cake_impulse_ns = self.cake_penetration_mm = 0.0
        self.mat_touch=False
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        self.subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.contact)
    def contact(self,headers,data):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        for h in headers:
            paths=tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(getattr(h,key))).lower()
                               for key in ('collider0','collider1')))
            if h.type == ContactEventType.CONTACT_LOST:
                self.active_pairs.discard(paths)
            elif h.num_contact_data:
                if (not self.cake_sensor_ready
                        and any('/pan_fixture/cake/' in p for p in paths)
                        and any('/pan_fixture/base/' in p for p in paths)):
                    support, _ = manifold_peaks(data, h.contact_data_offset, h.num_contact_data)
                    self.cake_sensor_ready = support > 0.0
                if touches_exposed_cake(paths):
                    impulse, depth = manifold_peaks(data, h.contact_data_offset, h.num_contact_data)
                    self.cake_impulse_ns = max(self.cake_impulse_ns, impulse)
                    self.cake_penetration_mm = max(self.cake_penetration_mm, depth)
                if self.contact_label(paths):
                    self.active_pairs.add(paths)
                    self.record_pair(paths)

    @staticmethod
    def contact_label(paths):
        parts=('lever','band','sector0','sector1','sector2','sector3')
        part=next((n for n in parts if any('/pan_fixture/'+n+'/' in p for p in paths)),None)
        if part is None:return None
        if any(p.endswith('/pan_fixture/mat1') for p in paths):return 'destination'
        for arm in ('left','right'):
            if any('/'+arm+'arm/' in p or '/'+arm+'_robot/' in p for p in paths):
                return arm+':'+part
        return None

    def record_pair(self,paths):
        label=self.contact_label(paths)
        if label=='destination':self.mat_touch=True
        elif label:self.contacts.add(label)

    def before_step(self,step,action):
        self.cake_impulse_ns = self.cake_penetration_mm = 0.0
        self.contacts.clear()
        self.mat_touch=False

    def states(self):
        if self.bodies is None:self.reset()
        objs={};poses={}
        for n,body in self.bodies.items():
            p=body.get_transforms()[0].copy();v=body.get_velocities()[0].copy()
            poses[n]=p
            half=np.array([.075,.075,.025]) if n=='cake' else np.array([.075,.075,.001]) if n=='base' else np.array([.035,.035,.02])
            objs[n]=ObjectState(p[:3],p[3:],v[:3],v[3:],p[:3]-half,p[:3]+half)
        if self.origin is None:self.origin={n:p.copy() for n,p in poses.items()}
        return objs,poses
    def after_step(self,step,objects,poses,robot):
        if not self.cake_sensor_ready:
            raise RuntimeError('No measured cake/base support impulse after settling. Check PhysX contact reporting.')
        self.joints=np.array([robot[s]['joints'] for s in ('left','right')])
        self.band_pose=poses['band']
        rb=Rotation.from_quat(poses['band'][3:])
        angle=(rb.inv()*Rotation.from_quat(poses['lever'][3:])).as_rotvec()[2]
        rad=[]
        for i in range(4):
            v=rb.inv().apply(poses[f'sector{i}'][:3]-poses['band'][:3]);a=i*math.pi/2
            rad.append(float(v@np.array([math.cos(a),math.sin(a),0.])))
        movement=max(float(np.linalg.norm(poses[n][:3]-self.origin[n][:3])) for n in ('cake','base'))
        tilt=max(float(np.degrees(np.arccos(np.clip(Rotation.from_quat(poses[n][3:]).as_matrix()[2,2],-1,1)))) for n in ('cake','base'))
        for paths in self.active_pairs:
            self.record_pair(paths)
        return dict(**measure(poses),cake_contact_impulse_ns=self.cake_impulse_ns,
                    cake_penetration_mm=self.cake_penetration_mm,on_destination_mat=self.mat_touch,robot_contacts=sorted(self.contacts),angle_deg=float(np.degrees(angle)),expansion_mm=[round(1000*r,3) for r in rad],motion_mm=1000*movement,
                    tilt_deg=tilt,band_xyz=poses['band'][:3].tolist(),hands_away=not self.contacts,
                    speed=float(np.linalg.norm(objects['band'].lin_vel)),spin=float(np.linalg.norm(objects['band'].ang_vel)))
