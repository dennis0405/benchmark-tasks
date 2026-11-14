"""Internal restoring torque for a Y-axis clip hinge, in any world pose.

Used by the unified Newton experiment when its imported MuJoCo-only spring
is disabled. This is a passive angular spring/damper, not an object command.
Newton spatial wrenches store world force first, COM torque second.
"""
import warp as wp


@wp.kernel
def spring_torque(body_q:wp.array[wp.transform],body_qd:wp.array[wp.spatial_vector],
                  lower:int,upper:int,stiffness:float,damping:float,
                  body_f:wp.array[wp.spatial_vector],observed:wp.array[float]):
    root=wp.transform_get_rotation(body_q[lower])
    relative=wp.quat_inverse(root)*wp.transform_get_rotation(body_q[upper])
    angle=2.0*wp.atan2(relative[1],relative[3])
    angle=wp.atan2(wp.sin(angle),wp.cos(angle))
    axis=wp.quat_rotate(root,wp.vec3(0.0,1.0,0.0))
    rate=wp.dot(wp.spatial_bottom(body_qd[upper])-wp.spatial_bottom(body_qd[lower]),axis)
    scalar=-stiffness*angle-damping*rate
    torque=scalar*axis
    wp.atomic_add(body_f,lower,wp.spatial_vector(wp.vec3(),-torque))
    wp.atomic_add(body_f,upper,wp.spatial_vector(wp.vec3(),torque))
    observed[0]=scalar


class PassiveClipSpring:
    def __init__(self,sim,lower,upper,stiffness,damping):
        self.lower=int(lower);self.upper=int(upper)
        self.stiffness=float(stiffness);self.damping=float(damping)
        self.device=sim.model.device
        self.observed=wp.zeros(1,dtype=float,device=self.device)

    def before_solve(self,state,dt):
        del dt
        wp.launch(spring_torque,1,inputs=[state.body_q,state.body_qd,self.lower,self.upper,
                  self.stiffness,self.damping],outputs=[state.body_f,self.observed],device=self.device)
