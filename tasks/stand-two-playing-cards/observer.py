"""Read PhysX contacts and body states. No object pose or force writes."""
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena.task_file import ObjectState
import geometry as g

class Observer:
    def __init__(self, env):
        self.env=env
        self.bodies=None
        self.sub=None
        self.joints=None

    def close(self):
        self.sub=None
        self.bodies=None

    def reset(self):
        import carb,omni.physx,omni.physics.tensors as tensors
        from pxr import UsdUtils,UsdPhysics
        self.close()
        stage=self.env.unwrapped.sim.stage
        cards=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()).endswith('/card_pair/coral')]
        if len(cards)!=1:raise RuntimeError(f'Expected two-card assembly: {cards}')
        base=cards[0].rsplit('/',1)[0]
        self.view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        self.view.set_subspace_roots('/')
        self.bodies={n:self.view.create_rigid_body_view(base+'/'+n) for n in g.NAMES}
        self.contacts=set()
        self.joints=None
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        self.sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.contact)

    @staticmethod
    def tag(path):
        for n in g.NAMES:
            if path.endswith('/'+n+'/face'):return n
        if path.endswith('/card_pair/mat'):return 'mat'
        if '/card_pair/' in path:return 'holder'
        return 'robot' if any(piece in path for piece in ('_robot/','arm/')) else 'other'

    def contact(self,headers,data):
        from pxr import PhysicsSchemaTools
        for h in headers:
            if h.num_contact_data:
                a,b=[self.tag(str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))).lower()) for k in ('collider0','collider1')]
                self.contacts.add(frozenset((a,b)))

    def before_step(self,step,action):
        self.contacts=set()

    def states(self):
        if self.bodies is None:self.reset()
        states={};poses={}
        for n,b in self.bodies.items():
            p=b.get_transforms()[0].copy();v=b.get_velocities()[0].copy()
            half=abs(Rotation.from_quat(p[3:]).as_matrix())@np.array([g.WIDTH,g.THICKNESS,g.HEIGHT])*.5
            poses[n]=p
            states[n]=ObjectState(p[:3],p[3:],v[:3],v[3:],p[:3]-half,p[:3]+half)
        return states,poses

    def after_step(self,step,objects,poses,robot):
        self.joints=np.array([robot[s]['joints'] for s in ('left','right')])
        p=np.array([poses[n][:3] for n in g.NAMES])
        rot=np.array([Rotation.from_quat(poses[n][3:]).as_matrix() for n in g.NAMES])
        vertical=rot[:,:,2].copy()
        for i in range(2):
            if vertical[i,2]<0:vertical[i]*=-1
        bottom=p-vertical*g.HEIGHT/2;top=p+vertical*g.HEIGHT/2
        # Evaluate separation at the middle of the two top edges, allowing a stagger along their width.
        width=rot[0,:,0]
        ridge=top[1]-top[0]
        plane_gap=np.linalg.norm(ridge-width*(width@ridge))
        separation=bottom[1,:2]-bottom[0,:2]
        direction=separation/max(.000001,np.linalg.norm(separation))
        lean=[float(vertical[0,:2]@direction),float(-vertical[1,:2]@direction)]
        hand_distance={}
        for side,arm in robot.items():
            distances=[]
            for i in range(2):
                q=rot[i].T@(np.array(arm['tcp'])-p[i])
                distances.append(np.linalg.norm(np.maximum(abs(q)-np.array([g.WIDTH,g.THICKNESS,g.HEIGHT])*.5,0)))
            hand_distance[side]=float(min(distances))
        contacts=self.contacts
        return dict(on_mat=[frozenset((n,'mat')) in contacts for n in g.NAMES],
                    ridge_contact=frozenset(g.NAMES) in contacts, ridge_error_m=float(plane_gap),
                    edge_overlap_m=float(g.WIDTH*abs(rot[0,:,0]@rot[1,:,0])-abs(width@ridge)),
                    base_error_m=abs(bottom[:,2]-g.MAT_Z).tolist(),inward_lean=lean,
                    holders_clear=all(frozenset((n,'holder')) not in contacts for n in g.NAMES)
                        and all(frozenset((n,'other')) not in contacts for n in g.NAMES),
                    tcp_to_card_m=hand_distance,
                    hands_clear=all(frozenset((n,'robot')) not in contacts for n in g.NAMES),
                    vmax=float(max(np.linalg.norm(o.lin_vel) for o in objects.values())),
                    wmax=float(max(np.linalg.norm(o.ang_vel) for o in objects.values())),
                    contacts=sorted(':'.join(sorted(c)) for c in contacts))
