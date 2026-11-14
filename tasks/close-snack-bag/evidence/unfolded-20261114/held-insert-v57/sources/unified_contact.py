"""Read-only contact observable adapter for a single Newton VBD solver.

The rigid tool and film are solved intrinsically together. Wrenches below
are recorded only; no delayed rigid-soft force feedback is added.
"""
import numpy as np
import warp as wp
import newton
from contact_average import accumulate


class UnifiedContactReadout:
    def __init__(self,inner,model,contacts):
        self.inner=inner;self.model=model;self.contacts=contacts;self.device=model.device
        self.observables=inner.observables({inner.ObservableKind.CONTACT_F},contacts=contacts)
        self._forces=wp.zeros(model.shape_count,dtype=wp.vec3,device=model.device)

    def __getattr__(self,name):return getattr(self.inner,name)

    def step(self,state_in,state_out,control,contacts,dt):
        self.inner.step(state_in,state_out,control,contacts,dt,observables=self.observables)

    def shape_contact_forces(self):
        self._forces.zero_();c=self.contacts
        if c.soft_contact_max:
            wp.launch(accumulate,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_shape,
                int(c.rigid_contact_max),self.observables.contact_f,1.,self._forces],device=self.device)
        return self._forces.numpy().astype(np.float64)


def install(sim):
    inner=sim.pipeline.solver
    reader=UnifiedContactReadout(inner,sim.model,sim.pipeline.contacts)
    sim.pipeline.solver=reader
    sim.pipeline.program.solver=reader
    sim.pipeline.invalidate_graph()
    return reader


def install_joint_encoder(sim):
    def encode(state,dt):
        del dt
        # VBD advances body poses. This derives encoder coordinates from
        # those physical poses; it never assigns particle or body poses.
        newton.eval_ik(sim.model,state,state.joint_q,state.joint_qd)
    sim.pipeline.hooks.add("post_solve",encode,name="vbd-joint-encoder",cadence=1)
    sim.pipeline.invalidate_graph()
