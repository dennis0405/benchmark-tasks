"""Closed-loop tool tracking for the scripted demo. Only sends YAM joint targets."""
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN

from scoring import PAN_HANDLE_Y

class ToolController:
    def __init__(self, env, plans):
        self.world = env.unwrapped.arena_world
        self.plans = plans
        self.translation = {side: np.zeros(3) for side in plans}
        self.rotation = {side: np.zeros(3) for side in plans}

    def action(self, time):
        if time <= 4.0:
            return np.concatenate([self.plans[side].action7(time)[0] for side in ("left","right")])
        actions=[]
        for side,tool,offset in (("left","pan",(0.,PAN_HANDLE_Y,.107)),("right","broom",(0.,0.,.107))):
            plan = self.plans[side]
            target, nominal_rot, finger, phase = plan.target(time)
            measured = self.world.get_pose_w(tool)[0].cpu().numpy()
            target = np.asarray(target)
            root_goal = target - offset
            if side=="left":
                # Bias the rear upward while the front lip bears on the surface.
                # Lowering just the rear grip can rock a long pan onto its heel.
                floor_weight=np.clip(1.-(root_goal[2]-.7475)/.045,0.,1.)
                desired=Rotation.from_rotvec([.055*floor_weight,0.,0.])
                root_goal=root_goal-[0.,0.,.0008*floor_weight]
            else:
                # Toe the brush inwards to guide the two outlying pieces.
                sign=np.clip((target[0]-.30)/.038,-1.,1.)
                desired=Rotation.from_rotvec([0.,0.,.14*sign])
            target = root_goal + desired.apply(offset)
            error = root_goal-measured[:3]
            # Keep the tool level under gravity and against contact loads, including
            # while bristles climb the actual sloped entry to the pan.
            angle = (desired * Rotation.from_quat(measured[3:]).inv()).as_rotvec()
            self.translation[side] = np.clip(self.translation[side]+(.035 if side=="left" else .065)*error,-.035,.035)
            self.rotation[side] = np.clip(self.rotation[side]+.055*angle,-.32,.32)
            delta = np.clip(.32*error+self.translation[side],-.045,.045)
            rot = Rotation.from_rotvec(np.clip(.28*angle+self.rotation[side],-.38,.38)).as_matrix()@desired.as_matrix()@nominal_rot
            q, _, _ = ik.solve(target+delta,ik.matrix_to_quat_xyzw(rot),plan.q,
                               base_pos=station.BASES[side],iters=60)
            plan.q = q
            actions.extend([*q, float(finger/FINGER_OPEN)])
        return np.array(actions)
