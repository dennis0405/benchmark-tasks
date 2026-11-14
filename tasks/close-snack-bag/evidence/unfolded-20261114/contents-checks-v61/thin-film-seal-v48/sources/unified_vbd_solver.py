"""Single Newton solver with solver-owned rigid/film collision checkpoints.

The native DAT truncation requires ownership: contact division planes and
both bodies' reference poses must come from the same detection instant.
"""
import newton
from newton.solvers import SolverVBD


class OwnedVBD(SolverVBD):
    def __init__(self,model,**kwargs):
        pipeline=newton.CollisionPipeline(model,soft_contact_gap=.008,
            enable_rigid_soft_full_surface_contact=True,contact_matching="latest")
        pipeline.contacts()  # establish capacities before solver allocation
        super().__init__(model,collision_pipeline=pipeline,**kwargs)

    def step(self,state_in,state_out,control,contacts,dt,*,observables=None):
        # RSS supplies its own main buffer. This solver intentionally has one
        # contact authority; the adapter below removes the duplicate detect.
        del contacts
        return super().step(state_in,state_out,control,None,dt,observables=observables)


def install_contact_authority(sim):
    solver=sim.pipeline.solver
    if not isinstance(solver,OwnedVBD):raise TypeError("expected single owned VBD solver")
    sim.pipeline.collision=None;sim.pipeline.program.collision=None
    sim.pipeline.contacts=solver.contacts;sim.pipeline.program.contacts=solver.contacts
    sim.pipeline.invalidate_graph()
