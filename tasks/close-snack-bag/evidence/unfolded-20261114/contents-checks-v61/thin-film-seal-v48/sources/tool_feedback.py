"""Experimental staggered six-axis film reaction on selected rigid tools.

The pinned RSS coupler already feeds forces to YAM finger slides. This adds
the PREVIOUS substep's complete Newton contact wrench to clip bodies, so a
spring hinge responds to film contact. It is explicit, not an implicit
two-way solver: timestep convergence must be checked before use as evidence.
"""
import numpy as np
import warp as wp


@wp.kernel
def tool_reaction(count: wp.array[int], shapes: wp.array[int], offset: int,
                  forces: wp.array[wp.spatial_vector], shape_body: wp.array[int],
                  selected: wp.array[int], body_force: wp.array[wp.spatial_vector]):
    i = wp.tid()
    if i < count[0]:
        shape = shapes[i]
        if shape >= 0:
            body = shape_body[shape]
            if body >= 0 and selected[body] != 0:
                # CONTACT_F is force on the rigid shape, plus torque about
                # its body's COM, all in world coordinates (Newton 1.7).
                wp.atomic_add(body_force, body, forces[offset+i])


class ToolFeedback:
    def __init__(self, inner):
        self.inner = inner
        self.selected = wp.array(np.array([
            int("snack_clip" in str(label)) for label in inner.model.body_label], np.int32),
            dtype=wp.int32, device=inner.device)
        # Full wrench feedback must not also get slide-only feedback.
        if inner._body_slide is not None:
            slides = inner._body_slide.numpy().copy()
            slides[self.selected.numpy()!=0] = -1
            inner._body_slide.assign(slides)

    def before_solve(self, state, dt):
        del dt
        c, m = self.inner.contacts, self.inner.model
        wp.launch(tool_reaction, dim=c.soft_contact_max,
                  inputs=[c.soft_contact_count, c.soft_contact_shape, int(c.rigid_contact_max),
                          self.inner.observables.contact_f, m.shape_body, self.selected],
                  outputs=[state.body_f], device=m.device)
