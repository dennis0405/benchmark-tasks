"""Preserve imported YAM MuJoCo gravity compensation in Newton VBD.

Authored per-body coefficients are applied as an external force at each
body COM. No tool or cloth gravity is changed, and no pose is assigned.
"""
import numpy as np
import warp as wp


@wp.kernel
def compensate(mass:wp.array[float],coefficient:wp.array[float],world:wp.array[int],
               gravity:wp.array[wp.vec3],force:wp.array[wp.spatial_vector]):
    i=wp.tid()
    if coefficient[i]!=0.0:
        g=gravity[wp.max(world[i],0)]
        wp.atomic_add(force,i,wp.spatial_vector(-mass[i]*coefficient[i]*g,wp.vec3()))


class RobotGravityCompensation:
    def __init__(self,sim):
        self.model=sim.model
        authored=getattr(getattr(sim.model,"mujoco",None),"gravcomp",None)
        if authored is None:raise RuntimeError("Imported robot gravity compensation is missing")
        values=authored.numpy()
        if values.shape!=(sim.model.body_count,) or not np.isfinite(values).all():
            raise ValueError("Invalid per-body gravity compensation")
        for i,value in enumerate(values):
            if value and "clip" in str(sim.model.body_label[i]):
                raise ValueError("The passive tool must retain gravity")
        self.coefficient=wp.array(values,dtype=float,device=sim.model.device)
        self.metadata={str(sim.model.body_label[i]):float(v) for i,v in enumerate(values) if v}

    def before_solve(self,state,dt):
        del dt
        m=self.model
        wp.launch(compensate,m.body_count,inputs=[m.body_mass,self.coefficient,m.body_world,m.gravity],
                  outputs=[state.body_f],device=m.device)
