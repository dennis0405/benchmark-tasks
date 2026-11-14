"""Synthetic seam poses test failure bookkeeping, not physical calibration."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import newton
from newton.solvers import SolverVBD
import warp as wp
from brittle_seams import BrittleSeams


@unittest.skipUnless(wp.is_cuda_available(),"Brittle seam observation needs CUDA")
class BrittleSeamTests(unittest.TestCase):
    def make(self):
        builder=newton.ModelBuilder();parts=[]
        for x in (-.012,.012):
            body=builder.add_link(xform=wp.transform(wp.vec3(x,0,.1),wp.quat_identity()))
            builder.add_shape_box(body,hx=.012,hy=.018,hz=.0012)
            parts.append(body)
        tangent=wp.quat_from_axis_angle(wp.vec3(0,1,0),np.pi/2)
        seam=builder.add_joint_rod(*parts,parent_xform=wp.transform(wp.vec3(.012,0,0),tangent),
            child_xform=wp.transform(wp.vec3(-.012,0,0),tangent),bend_stiffness=.05,
            stretch_stiffness=1e4,collision_filter_parent=False)
        builder.add_articulation([seam])
        anchor=builder.add_link(is_kinematic=True)
        unrelated=builder.add_joint_fixed(parent=-1,child=anchor);builder.add_articulation([unrelated])
        builder.color();model=builder.finalize(device="cuda:0")
        solver=SolverVBD(model,particle_enable_self_contact=False,rigid_compliant_alm=True)
        invalidations=[]
        sim=SimpleNamespace(model=model,scene=SimpleNamespace(by_name={"contents":SimpleNamespace(
            seams=[seam],chip_groups=[tuple(parts)])}),
            pipeline=SimpleNamespace(solver=solver,invalidate_graph=lambda:invalidations.append(1)),
            request=SimpleNamespace(physics=SimpleNamespace(control_hz=20)))
        return sim,model.state(),BrittleSeams(sim),unrelated,invalidations

    def test_bending_failure_is_irreversible_and_never_moves_fragments(self):
        sim,state,tracker,unrelated,invalidations=self.make()
        initial=state.body_q.numpy().copy();poses=initial.copy()
        rotation=Rotation.from_euler("y",.2)
        poses[1,:3]=[0,0,.1]+rotation.apply([.012,0,0]);poses[1,3:]=rotation.as_quat()
        state.body_q.assign(poses);tracker.observe(state,1/2400);tracker.commit(1)
        np.testing.assert_array_equal(state.body_q.numpy(),poses)
        self.assertFalse(bool(sim.model.joint_enabled.numpy()[0]))
        self.assertTrue(bool(sim.model.joint_enabled.numpy()[unrelated]))
        state.body_q.assign(initial);tracker.observe(state,1/2400);tracker.commit(2)
        self.assertEqual(tracker.report()["broken_count"],1)
        self.assertEqual(len(tracker.events),1);self.assertEqual(len(invalidations),1)
        tracker.reset()
        self.assertEqual(tracker.report()["broken_count"],1,"bookkeeping cannot heal disabled material")
        enabled=sim.model.joint_enabled.numpy().copy();enabled[0]=True;sim.model.joint_enabled.assign(enabled)
        tracker.reset()
        self.assertEqual(tracker.report()["broken_count"],0,"an explicit new intact episode clears damage")
        self.assertEqual(tracker.report()["observed_substeps"],0)

    def test_small_bending_does_not_break(self):
        sim,state,tracker,unrelated,invalidations=self.make()
        poses=state.body_q.numpy();rotation=Rotation.from_euler("y",.01)
        poses[1,:3]=[0,0,.1]+rotation.apply([.012,0,0]);poses[1,3:]=rotation.as_quat()
        state.body_q.assign(poses);tracker.observe(state,1/2400);tracker.commit(1)
        self.assertEqual(tracker.report()["broken_count"],0)
        self.assertTrue(sim.model.joint_enabled.numpy().all());self.assertFalse(invalidations)


if __name__=="__main__":unittest.main()
