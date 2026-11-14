"""Observe the Newton film/tool island while the native robot stays MuJoCo.

The stock two-way proxy coupler sends contact reactions back to the robot.
This adapter records its last inner solve; it does not apply extra feedback.
"""
import numpy as np
import newton
from unified_contact import UnifiedContactReadout


class HybridReadout:
    def __init__(self,outer,reader,model):
        self.inner=outer;self.reader=reader;self.model=model
        self.contacts=reader.contacts;self.observables=reader.observables
    def __getattr__(self,name):return getattr(self.inner,name)
    def step(self,state_in,state_out,control,contacts,dt):
        self.inner.step(state_in,state_out,control,contacts,dt)
    def shape_contact_forces(self):return self.reader.shape_contact_forces()


def install(sim,newton_contact_ke=10000.):
    outer=sim.pipeline.solver
    paper=next(i for i in sim.pipeline.program.islands if i.particles.size)
    robot=next(i for i in sim.pipeline.program.islands if i.name!=paper.name)
    solver=outer.solver(paper.name)
    contacts=outer.get_proxy_contacts(robot.name,paper.name)
    if contacts is None:raise RuntimeError("No native robot to Newton contact channel")
    # The proxy factory owns contacts on its view. Publish those exact
    # capacities on the solver view before allocating native observables.
    capacity_pipeline=newton.CollisionPipeline(sim.model,soft_contact_gap=.008,
        enable_rigid_soft_full_surface_contact=True,rigid_contact_max=contacts.rigid_contact_max,
        soft_contact_max=contacts.soft_contact_max)
    reader=UnifiedContactReadout(solver,solver.model,contacts)
    reader.capacity_pipeline=capacity_pipeline
    original=solver.step
    def observed_step(state_in,state_out,control,contacts_arg,dt,*,observables=None):
        if observables is not None:raise RuntimeError("Contact reader already owns the inner observable request")
        if contacts_arg is not contacts:raise RuntimeError("Unexpected proxy contact buffer")
        original(state_in,state_out,control,contacts_arg,dt,observables=reader.observables)
    solver.step=observed_step
    # MuJoCo's rigid model was compiled with its original stiffness. Newton
    # uses the softer contact material on its film/tool/proxy interfaces.
    values=sim.model.shape_material_ke.numpy().copy()
    owners=sim.model.shape_body.numpy()
    tool=sim.scene.by_name["clip"]
    values[np.isin(owners,tool.bodies)]=newton_contact_ke
    # Preserve the validated stiff finger/film and board/film interfaces.
    # Only the light free clip receives the softer contact material.
    solver.model.shape_material_ke.assign(values)
    sim.model.shape_material_ke.assign(values)
    wrapper=HybridReadout(outer,reader,sim.model)
    sim.pipeline.solver=wrapper;sim.pipeline.program.solver=wrapper
    sim.pipeline.invalidate_graph()
    wrapper.metadata=dict(robot_island=robot.name,film_tool_island=paper.name,
        newton_clip_shape_ke_N_m=newton_contact_ke,
        newton_other_shape_ke_range_N_m=[float(values[~np.isin(owners,tool.bodies)].min()),
                                        float(values[~np.isin(owners,tool.bodies)].max())],manual_feedback=False,
        channels=[dict(source=c.source,destination=c.destination,coupling=c.coupling,
                       bodies=len(c.bodies),joints=len(c.joints)) for c in sim.pipeline.program.channels],
        note="stock two-way staggered proxy coupling; native robot controller retained")
    return wrapper
