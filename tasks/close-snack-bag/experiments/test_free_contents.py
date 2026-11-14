"""A chip without generalized coordinates must still be a dynamic body."""
import unittest
import numpy as np
import newton
from newton.solvers import SolverVBD
import warp as wp


@unittest.skipUnless(wp.is_cuda_available(),"VBD rigid integration check needs CUDA")
class FreeContentsTests(unittest.TestCase):
    def test_body_only_link_falls_under_gravity(self):
        builder=newton.ModelBuilder()
        # The installed VBD has a zero-joint table dimensionality issue.
        # The real scene has YAM joints; this tiny test uses a fixed anchor.
        anchor=builder.add_link(is_kinematic=True)
        joint=builder.add_joint_fixed(parent=-1,child=anchor)
        builder.add_articulation([joint])
        body=builder.add_link(xform=wp.transform(wp.vec3(0,0,1),wp.quat_identity()))
        builder.add_shape_box(body,hx=.01,hy=.01,hz=.01,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.))
        builder.color();model=builder.finalize(device="cuda:0")
        self.assertEqual(model.joint_coord_count,0)
        self.assertAlmostEqual(float(model.body_mass.numpy()[body]),.008,places=7)
        solver=SolverVBD(model,iterations=10,particle_enable_self_contact=False,rigid_compliant_alm=True)
        collision=newton.CollisionPipeline(model);contacts=collision.contacts()
        a,b=model.state(),model.state();control=model.control()
        for _ in range(100):
            a.clear_forces();collision.collide(a,contacts)
            solver.step(a,b,control,contacts,.001);a,b=b,a
        position=a.body_q.numpy();velocity=a.body_qd.numpy()
        self.assertTrue(np.isfinite(position).all())
        self.assertAlmostEqual(float(position[body,2]),1-.5*9.81*.1**2,delta=.002)
        self.assertAlmostEqual(float(velocity[body,2]),-9.81*.1,delta=.01)
        np.testing.assert_allclose(position[anchor,:3],0.,atol=1e-7)


if __name__=="__main__":unittest.main()
