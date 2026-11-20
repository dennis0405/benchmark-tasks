"""Read native PhysX contacts and apply only passive forces on the fixture."""
import math
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena.task_file import ObjectState

NAMES=('gearbox','shoe','slider','spindle','crank')
PLAN=json.loads(Path(__file__).with_name('motion.json').read_text())
START,END=np.deg2rad([PLAN['start'],PLAN['end']])
from limits import LOW_LEAD as LOW, HIGH_LEAD as HIGH, STROKE as LIMIT, PAD_TOP, FOOT_HALF_WIDTHS, FOOT_CENTER_Z, PRELOAD
from limits import CARRIAGE_MASS, OUTPUT_STIFFNESS, OUTPUT_DAMPING, THRUST_FRICTION, THRUST_RADIUS
INDEX=np.array([0],dtype=np.uint32)
R0=Rotation.from_euler('z',PLAN['yaw'],degrees=True).as_matrix()

class Observer:
    def __init__(self,env):
        self.env=env;self.bodies=None;self.contact=None;self.subscription=None;self.measured=None
    def close(self):
        self.subscription=None;self.contact=None;self.bodies=None
    def reset(self):
        import carb,omni.physx,omni.physics.tensors as tensors
        from pxr import UsdUtils,UsdPhysics
        self.close();stage=self.env.unwrapped.sim.stage
        handles=[p for p in stage.Traverse() if str(p.GetPath()).endswith('/landing/crank') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(handles)!=1:raise RuntimeError(f'Expected one crank, got {len(handles)}')
        root=str(handles[0].GetPath()).rsplit('/',1)[0]
        self.view=tensors.create_simulation_view('numpy',stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
        self.view.set_subspace_roots('/')
        self.bodies={n:self.view.create_rigid_body_view(root+'/'+n) for n in NAMES}
        self.travel=0.;self.low=0.;self.high=0.;self.prev=START;self.rest=START;self.peak=0.;self.caught=False
        self.measured=None;self.contacts=set();self.other=set()
        self.warm=True
        self.pending_normal_impulse = 0.
        self.normal_load = 0.
        self.torque_sums = np.zeros(4)
        self.load_time = self.load_sum = 0.
        self.ground_hits=0;self.ground_impulse=0.;self.interval_dt=0.
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        self.contact=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)
        self.subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(self.advance)
    def on_contact(self,headers,data):
        from pxr import PhysicsSchemaTools
        for h in headers:
            if not h.num_contact_data:continue
            pair=[str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))).lower() for k in ('collider0','collider1')]
            if any(x.endswith('/shoe/foot_plate') for x in pair) and any(x.endswith('/ground_pad') for x in pair):
                for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    # The plate normal is world +Z. Ignore tangential friction impulses.
                    normal_impulse = abs(float(data[k].impulse[2]))
                    self.ground_impulse += normal_impulse
                    self.pending_normal_impulse += normal_impulse
                    self.ground_hits += 1
            for a,b in (pair,pair[::-1]):
                if '/landing/' not in a:continue
                for side in ('left','right'):
                    if not any(x in b for x in ('/'+side+'_robot/','/'+side+'arm/')):continue
                    if a.endswith('/roller/grip') and 'tip_' in b:
                        pad='tip_left' if 'tip_left' in b else 'tip_right'
                        self.contacts.add((side,pad))
                    else:self.other.add(side+':'+a.rsplit('/',1)[-1])
    def before_step(self,step,action):
        self.contacts=set();self.other=set()
        self.ground_hits=0;self.ground_impulse=0.; self.interval_dt=0.
        self.torque_sums[:] = 0.
        self.load_sum = self.load_time = 0.
        if step==1:
            self.warm=False
            self.prev=self.read()[3];self.low=self.high=self.travel=self.peak=0.
    def read(self):
        p={n:b.get_transforms()[0].copy() for n,b in self.bodies.items()}
        wheel=Rotation.from_quat(p['spindle'][3:]).as_matrix()
        handle=Rotation.from_quat(p['crank'][3:]).as_matrix()
        w=R0.T@wheel;rel=wheel.T@handle
        angle=math.atan2(w[2,1],w[1,1]);fold=math.atan2(rel[0,2],rel[2,2])
        stand=p['gearbox'][:3]
        slide=-float((p['slider'][:3]-stand)@R0[:,0])
        radial=np.array([0,-np.sin(END),np.cos(END)])
        slot=stand+R0@(np.array([-.028,0,0])-.205*radial)
        gap=float(np.linalg.norm(p['crank'][:3]+handle[:,0]*.205-slot))
        return p,wheel,handle,angle,slide,fold,gap
    def apply(self,n,f=None,t=None):
        def arr(v):return None if v is None else np.array([v],np.float32)
        self.bodies[n].apply_forces_and_torques_at_position(arr(f),arr(t),None,INDEX,True)
    def advance(self,dt):
        if self.bodies is None:return
        p,w,h,angle,slide,fold,gap=self.read()
        self.interval_dt+=dt
        v={n:b.get_velocities()[0].copy() for n,b in self.bodies.items()}
        axis=R0[:,0];fa=w[:,1]
        spin=v['spindle'][3:]@axis;omega=(v['crank'][3:]-v['spindle'][3:])@fa
        delta=math.remainder(angle-self.prev,2*math.pi);self.prev=angle
        if not self.warm and abs(fold)<math.radians(14):
            if slide<.0056:self.low+=delta;self.travel+=LOW*delta/(2*math.pi)
            if slide>.0224:self.high+=delta;self.travel+=HIGH*delta/(2*math.pi)
        self.peak=max(self.peak,self.travel)
        if self.caught and gap>.024:self.caught=False
        if gap<.014 and abs(fold-np.pi/2)<.10:self.caught=True
        error=np.clip(math.remainder(angle-self.rest,2*np.pi),-.012,.012);self.rest=angle-error
        tau=-np.clip(42*fold,-3,3)*np.exp(-min(abs(fold/.115),8)**4)-.22*omega
        # Reduced passive hinge balancer. Its generalized torque is the gravity
        # moment of an inboard weight along negative local X, with the opposite
        # spindle reaction. No action, gear state or score enters the force law.
        balance_arm = h[:,0] * (-.185)
        balance_torque = float(np.cross(balance_arm, [0.,0.,-.98*9.81]) @ fa)
        tau += balance_torque
        # The preceding PhysX substep supplies the normal impulse. It is consumed
        # once, even when a control sample ends between physics callbacks.
        load = self.pending_normal_impulse / dt
        self.pending_normal_impulse = 0.
        # Compliant thrust bearing. The short response time filters contact chatter.
        self.normal_load += -math.expm1(-dt/.016) * (load - self.normal_load)
        elastic_slip = error / .012
        thrust_torque = -THRUST_FRICTION * THRUST_RADIUS * self.normal_load * elastic_slip
        base_torque = -1.55 * elastic_slip - .97 * spin
        resistance = base_torque + thrust_torque
        self.torque_sums += dt * np.array([spin, base_torque, thrust_torque, resistance])
        self.load_sum += self.normal_load * dt
        self.load_time += dt
        drag = resistance - min(5.,max(0,self.travel-LIMIT)*2*np.pi/HIGH*18.)
        if self.caught:
            drag+=np.clip(-28*math.remainder(angle-END,2*np.pi)-.25*spin,-3.,3.)
            tau+=np.clip(35*(np.pi/2-fold)-.4*omega,-3.,3.)
        self.apply('crank',t=fa*tau)
        self.apply('spindle',t=axis*drag-fa*tau)
        speed=-float(v['slider'][:3]@axis)
        self.apply('slider',f=-axis*(16*np.tanh((slide-.014)/.002)-35*speed))
        z=p['shoe'][2]-(p['gearbox'][2]-.35)
        self.apply('shoe',f=[0,0,CARRIAGE_MASS*9.81+OUTPUT_STIFFNESS*(min(self.travel,LIMIT)-PRELOAD-z)-OUTPUT_DAMPING*v['shoe'][2]])
    def states(self):
        if self.bodies is None:self.reset()
        out={};poses={}
        for n,b in self.bodies.items():
            p=b.get_transforms()[0].copy();v=b.get_velocities()[0].copy();poses[n]=p
            out[n]=ObjectState(p[:3],p[3:],v[:3],v[3:],p[:3]-.01,p[:3]+.01)
        return out,poses
    def after_step(self,step,objects,poses,robot):
        p,w,h,angle,slide,fold,gap=self.read()
        self.measured=np.array([robot[s]['joints'] for s in ('left','right')])
        distances=[]
        for s in ('left','right'):
            q=h.T@(np.asarray(robot[s]['tcp'])-p['crank'][:3])
            nearest=np.array([np.clip(q[0],.262128+.100,.262128+.185),0,.3556])
            distances.append(float(np.linalg.norm(q-nearest)))
        # Lowest transformed corner of the actual box, measured independently of screw travel.
        foot_rot = Rotation.from_quat(p['shoe'][3:]).as_matrix()
        bottom = float(p['shoe'][2]) + foot_rot[2,2]*FOOT_CENTER_Z - abs(foot_rot[2]) @ FOOT_HALF_WIDTHS
        spin,base_torque,thrust_torque,resistance = self.torque_sums/max(self.load_time,1e-9)
        return dict(pilot=getattr(self,"pilot",{}),spin_rad_s=spin, bearing_load_n=self.load_sum/max(self.load_time,1e-9),
                    base_torque_nm=base_torque, thrust_torque_nm=thrust_torque,
                    drivetrain_torque_nm=resistance, ground_points=self.ground_hits, ground_force_n=self.ground_impulse/max(self.interval_dt, 1e-9),
                    clearance_mm=(bottom-PAD_TOP)*1000, angle=math.degrees(angle),fold=math.degrees(fold),slide_mm=slide*1000,
                    travel_mm=self.travel*1000,peak_mm=self.peak*1000,
                    foot_mm=float((p['shoe'][2]-p['gearbox'][2]+.35)*1000),
                    low_deg=math.degrees(self.low),high_deg=math.degrees(self.high),gap_mm=gap*1000,
                    caught=self.caught,pads=sorted(s+':'+j for s,j in self.contacts),other=sorted(self.other),
                    distance=distances,hands_clear=(min(distances)>.047 and not self.contacts and not self.other))
