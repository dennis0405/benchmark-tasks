"""Integrate observed rigid-soft reaction over every solved physics substep.

Read-only: does not change contacts or apply forces. This avoids controlling
from a single aliased sample of a stiff contact at each 20 Hz command.
"""
import numpy as np
import warp as wp


@wp.kernel
def accumulate(count:wp.array[int],shape:wp.array[int],offset:int,
               forces:wp.array[wp.spatial_vector],dt:float,
               impulse:wp.array[wp.vec3]):
    i=wp.tid()
    if i<count[0] and shape[i]>=0:
        wp.atomic_add(impulse,shape[i],wp.spatial_top(forces[offset+i])*dt)


@wp.kernel
def elapsed(dt:float,duration:wp.array[float]):
    duration[0]+=dt


class ContactAverage:
    def __init__(self,solver):
        self.solver=solver;self.model=solver.model
        self.impulse=wp.zeros(self.model.shape_count,dtype=wp.vec3,device=self.model.device)
        self.duration=wp.zeros(1,dtype=float,device=self.model.device)

    def reset(self):
        self.impulse.zero_();self.duration.zero_()

    def observe(self,state,dt):
        del state
        s=self.solver;c=s.contacts
        wp.launch(accumulate,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_shape,
            int(c.rigid_contact_max),s.observables.contact_f,dt,self.impulse],device=self.model.device)
        wp.launch(elapsed,1,inputs=[dt,self.duration],device=self.model.device)

    def forces(self):
        duration=float(self.duration.numpy()[0])
        if duration<=0:return np.zeros((self.model.shape_count,3))
        return self.impulse.numpy().astype(np.float64)/duration
