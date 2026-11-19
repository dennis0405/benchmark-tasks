"""Observe per-sector travel and native contact impulses. The grader cannot move a body."""
import numpy as np
from collections import deque
import paint as P
import rim
from strike_credit import StrikeCredit


class Dynamics:
    def __init__(self, env):
        import carb
        import omni.physx
        from pxr import PhysicsSchemaTools, UsdPhysics
        stage = env.unwrapped.sim.stage
        self.world = env.unwrapped
        self.drives = {}
        for prim in stage.Traverse():
            if prim.GetName()=='spring_joint' and prim.GetParent().GetName() in rim.SECTORS:
                self.drives[prim.GetParent().GetName()] = UsdPhysics.DriveAPI.Get(prim, 'linear')
        assert len(self.drives)==8,self.drives
        carb.settings.get_settings().set('/physics/disableContactProcessing', False)
        import omni.physics.tensors as tensors
        from pxr import UsdUtils
        self.view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        self.view.set_subspace_roots('/')
        env_root=str(next(iter(self.drives.values())).GetPrim().GetPath()).rsplit('/',2)[0]
        self.head=self.view.create_rigid_body_view(env_root+'/mallet')
        self.can=self.view.create_rigid_body_view(env_root+'/can')
        self.history=deque(maxlen=5)
        self.sub=omni.physx.get_physx_interface().subscribe_physics_step_events(self.substep)
        self.report = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.contact)
        self.path = PhysicsSchemaTools.intToSdfPath
        self.reset()

    def reset(self):
        self.closed = set()
        self.credit = StrikeCredit()
        self.engaged = np.zeros(8,dtype=bool)
        self.pairs = {}
        self.had_grasp = False
        self.history.clear()
        self.fast_impulses = np.zeros(8)
        self.peak_speeds = np.zeros(8)
        self.last_face_speed = 0.
        self.last_struck = set()
        for drive in self.drives.values():
            drive.GetTargetPositionAttr().Set(0.)

    def substep(self,dt):
        vel=self.head.get_velocities()[0]
        axis=P.mat(self.head.get_transforms()[0])[:,1]
        offset=-np.sign(axis[2])*axis*(P.HEAD_L/2)
        vz=(vel[:3]+np.cross(vel[3:],offset))[2]-self.can.get_velocities()[0,2]
        self.history.append(float(-vz) if abs(axis[2])>.85 else 0.)

    def contact(self, headers, points):
        for h in headers:
            if not h.num_contact_data: continue
            names = sorted(str(self.path(getattr(h,k))).split('/env_0/')[-1] for k in ('collider0','collider1'))
            key = '|'.join(names)
            impulse = sum(float(np.linalg.norm(points[i].impulse)) for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data))
            if names[0].startswith('mallet/geometry/') and names[1].startswith('rim'):
                i=rim.SECTORS.index(names[1].split('/')[0])
                speed=max(self.history,default=0.)
                self.peak_speeds[i]=max(self.peak_speeds[i],speed)
                if speed>.12:
                    self.fast_impulses[i]+=sum(abs(points[j].impulse[2]) for j in range(
                        h.contact_data_offset,h.contact_data_offset+h.num_contact_data) if abs(points[j].normal[2])>.6)
            self.pairs[key] = self.pairs.get(key,0.)+impulse

    def before_step(self, step, action):
        self.pairs.clear()
        self.fast_impulses.fill(0.)
        self.peak_speeds.fill(0.)

    def after_step(self, step, objects, root_poses, robot):
        from task import hand_points
        can = root_poses['can']
        up = P.mat(can)[:,2]
        depth = np.array([rim.CENTER_Z-P.H_CAN/2-np.dot(root_poses[n][:3]-can[:3],up) for n in rim.SECTORS])
        # Over-centre model: an unloaded spring initially resists travel at 4 kN/m.
        # Change wells only after the lip physically travels through the midpoint.
        # Reopening needs outward travel through the same midpoint.
        for i,(n,d) in enumerate(zip(rim.SECTORS,depth)):
            state=d>.002
            if state!=self.engaged[i]:
                self.drives[n].GetTargetPositionAttr().Set(-P.PROUD if state else 0.)
                self.engaged[i]=state
        pairs={k:round(v,6) for k,v in self.pairs.items() if v>1.e-6}
        def fingers(side, target):
            tips = set()
            for pair in pairs:
                a,b = pair.lower().split('|')
                for hand,part in ((a,b),(b,a)):
                    if target in part and side+'arm/' in hand:
                        for tip in ('tip_left','tip_right'):
                            if '/gripper/'+tip+'/' in hand:
                                tips.add(tip)
            return sorted(tips)
        left=fingers('left','can/')
        right=fingers('right','mallet/piece0')
        grip = len(left)==2 and len(right)==2
        self.had_grasp |= grip
        seated = list(map(bool,self.credit.update(step,depth*1000,self.fast_impulses,grip)))
        hands=hand_points(robot)
        cloth_support=any('cloth/geometry/' in k and 'mallet/geometry/' in k and v>.001 for k,v in pairs.items())
        return dict(cloth_support=cloth_support,sectors=seated,seated_n=sum(map(int,seated)),depth_mm=(depth*1000).tolist(),
                    rapid_normal_Ns=self.fast_impulses.tolist(),incoming_m_s=self.peak_speeds.tolist(),
                    tapped=sorted(self.credit.attributed),lid_proud_mm=float(4-min(depth)*1000),
                    grasp_evidence=bool(grip),left_contacts=left,right_contacts=right,pairs=pairs,
                    pad_boxes=[[c.tolist(),R.tolist(),h.tolist()] for hand in hands.values() for c,R,h in hand[2]])
