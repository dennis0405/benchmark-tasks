"""Read PhysX contacts and poses. Only robot joint and gripper actions move the tools."""
import numpy as np
from scipy.spatial.transform import Rotation
from layout import WIDTH, GRIP_Z

NAMES = ('Rail0','Rail1','Rail2','Rail3','Fixed','Slide')

def geometry(poses):
    corners = []
    for name in ('Rail0', 'Rail2'):
        p = poses[name]
        half = Rotation.from_quat(p[3:]).apply([WIDTH / 2, 0, 0])
        corners.extend([p[:3]-half, p[:3]+half])
    diagonals = [np.linalg.norm(corners[i+2]-corners[i])*1000 for i in (0,1)]
    errors = []
    for i in range(4):
        u, v = corners[(i+1)%4]-corners[i], corners[(i-1)%4]-corners[i]
        angle = np.rad2deg(np.arccos(np.clip(u@v / (np.linalg.norm(u)*np.linalg.norm(v)), -1., 1.)))
        errors.append(abs(angle-90))
    return dict(diagonals_mm=diagonals, difference_mm=diagonals[0]-diagonals[1],
                max_angle_error_deg=max(errors)), corners

class Observer:
    def __init__(self,env):
        self.env = env
        self.bodies = None
        self.subscription = None

    def reset(self):
        import carb, omni.physx
        import omni.physics.tensors as tensors
        from pxr import PhysicsSchemaTools, UsdUtils
        stage = self.env.unwrapped.sim.stage
        root = next(str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).endswith('/frame/Fixed')).rsplit('/',1)[0]
        self.view = tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        self.view.set_subspace_roots('/')
        self.bodies = {n:self.view.create_rigid_body_view(root+'/'+n) for n in NAMES}
        self.paths = {}
        self.path_of = PhysicsSchemaTools.intToSdfPath
        self.contacts = set()
        self.origin = None
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)

    def on_contact(self,headers,data):
        for h in headers:
            if not h.num_contact_data:
                continue
            names = []
            for idx in (h.collider0,h.collider1):
                if idx not in self.paths:
                    self.paths[idx] = str(self.path_of(idx)).split('/env_0/')[-1].lower()
                names.append(self.paths[idx])
            if any('frame/fixed/' in p or 'frame/slide/' in p for p in names):
                self.contacts.add(tuple(sorted(names)))

    def before_step(self,step,action):
        self.contacts.clear()

    def poses(self):
        return {n:b.get_transforms()[0].copy().astype(float) for n,b in self.bodies.items()}

    def geometry(self,poses):
        return geometry(poses)[0]

    def states(self):
        from yam_arena.task_file import ObjectState
        if self.bodies is None:
            self.reset()
        poses = self.poses()
        if self.origin is None:
            self.origin = poses
        states = {}
        for n,p in poses.items():
            v = self.bodies[n].get_velocities()[0].copy()
            # Only poses and contact geometry enter scoring. Conservative enclosing bounds.
            radius = .62 if n=='Fixed' else .23 if n.startswith('Rail') else .14
            states[n] = ObjectState(p[:3],p[3:],v[:3],v[3:],p[:3]-radius,p[:3]+radius)
        return states,poses

    def after_step(self,step,objects,poses,robot):
        return measurements(poses,self.origin,self.contacts,robot)

    def close(self):
        self.subscription = None
        self.bodies = None

def measurements(poses,origin,contacts,robot):
    m,corners = geometry(poses)
    initial,_ = geometry(origin)
    ac = initial['difference_mm'] > 0
    ids = (0,2) if ac else (1,3)
    caps = ('rail0/start','rail0/end','rail2/start','rail2/end')
    ends = [poses[k] for k in ('Fixed','Slide')]
    grips = []
    def fits(j,i):
        p = ends[j]
        v = Rotation.from_quat(p[3:]).inv().apply(corners[i]-p[:3])*1000
        gap = (-1 if j==0 else 1)*v[1]-20.
        names = {'frame/'+('fixed','slide')[j]+'/face','frame/'+caps[i]}
        return (abs(gap)<=1 and abs(v[0])<12 and 8<v[2]<16 and tuple(sorted(names)) in contacts)
    for p in ends:
        grips.append(p[:3]+Rotation.from_quat(p[3:]).apply([0,0,GRIP_Z]))
    touched = [p for pair in contacts for p in pair if 'arm/geometry/' in p or '_robot/' in p]
    cleared = not touched and all(min(np.linalg.norm(np.array(v['tcp'])-g) for g in grips)>.060
                                  and min(abs(f) for f in v['fingers'])>.027 for v in robot.values())
    return dict(**m,initial_error_mm=abs(initial['difference_mm']),
                jaw_span_mm=1000*np.linalg.norm(ends[0][:3]-ends[1][:3]),
                from_rest_mm=min(1000*np.linalg.norm(poses[n][:3]-origin[n][:3]) for n in ('Fixed','Slide')),
                both_seated=bool((fits(0,ids[0]) and fits(1,ids[1])) or (fits(0,ids[1]) and fits(1,ids[0]))),
                hand_on_clamp=any('/gripper/' in name for name in touched),hands_clear=bool(cleared))
