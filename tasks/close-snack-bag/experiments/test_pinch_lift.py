"""A moving empty hand, single-wall grasp or slipping film must not pass."""
from copy import deepcopy
import unittest
import numpy as np
from pinch_lift_metrics import evaluate,mouth_capture,confirmed_release_evidence,primitive_surface_distance


class PinchLiftTests(unittest.TestCase):
    def example(self):
        hand=dict(wall_area_mm2=[100.,100.],opposed_load_N=[.2,.2],wall_slip_p90_m=[.001,.001],wall_hold_slip_p90_m=[.001,.001],wall_landmarks_retained=[True,True],wall_lift_m=[.05,.05],finger_film_distance_m=.0003,wall_finger_distance_m=[.0003,.0003])
        trace=[dict(step=i,phase="pinch_lift_hold",pinch_lift=dict(reference_locked=True,hands=dict(left=deepcopy(hand),right=deepcopy(hand)))) for i in range(61)]
        geometry=dict(checked_substeps=7200,transverse_edge_crossings=0,film_box_intersections=0,film_round_intersections=0,film_ellipsoid_intersections=0)
        capacity=dict(checked_substeps=7200,rigid_body_overflow_events=0,soft_body_overflow_events=0,rigid_total_overflow_substeps=0,soft_total_overflow_substeps=0)
        result=dict(control_hz=20,frames=60,physics_substeps=120,controller_completed=True,controller_failure=None,interrupted=False,all_finite=True,object_pose_writes_during_rollout=False,component_reset=None,clip=False,substep_geometry=geometry,contact_capacity=capacity)
        return trace,result

    def test_accepts_observed_three_seconds(self):
        self.assertTrue(evaluate(*self.example())["passed"])

    def test_empty_single_wall_slipping_and_no_material_lift_reject(self):
        for key,value in [("opposed_load_N",[0.,.2]),("wall_finger_distance_m",[.0003,.02]),("wall_lift_m",[0.,0.])]:
            trace,result=self.example();trace[30]["pinch_lift"]["hands"]["left"][key]=value
            with self.subTest(key=key):self.assertFalse(evaluate(trace,result)["passed"])

    def test_missing_frame_or_interruption_rejects(self):
        trace,result=self.example();del trace[30]
        self.assertFalse(evaluate(trace,result)["passed"])
        trace,result=self.example();result["interrupted"]=True
        self.assertFalse(evaluate(trace,result)["passed"])

    def test_load_seating_is_reported_separately_from_hold_creep(self):
        trace,result=self.example()
        for row in trace:row["pinch_lift"]["hands"]["left"]["wall_slip_p90_m"]=[.004,.004]
        self.assertTrue(evaluate(trace,result)["passed"])

    def test_boundary_escape_and_shift_do_not_reject_retained_contact(self):
        trace,result=self.example()
        for row in trace:
            hand=row["pinch_lift"]["hands"]["right"]
            hand.update(wall_landmarks_retained=[False,True],wall_area_mm2=[40.,40.],
                wall_slip_p90_m=[.009,.009],wall_hold_slip_p90_m=[.004,.004])
            self.assertFalse(confirmed_release_evidence(hand))
        self.assertTrue(evaluate(trace,result)["passed"])

    def test_physical_release_needs_separation_not_only_zero_load(self):
        hand=self.example()[0][0]["pinch_lift"]["hands"]["left"]
        hand["opposed_load_N"]=[0.,0.]
        self.assertFalse(confirmed_release_evidence(hand))
        hand["finger_film_distance_m"]=.006
        self.assertTrue(confirmed_release_evidence(hand))
        hand["opposed_load_N"]=[.2,.2];hand["finger_film_distance_m"]=.02
        self.assertTrue(confirmed_release_evidence(hand))

    def test_film_near_long_pad_end_is_not_a_distant_empty_hand(self):
        distance=primitive_surface_distance(np.array([[0.,.0023,.039],[.026,.0023,.039]]),7,[.006,.002,.04])
        self.assertAlmostEqual(distance[0],.0003)
        self.assertGreater(distance[1],.01)

    def test_between_vertex_surface_is_captured_but_outside_sheet_is_not(self):
        # Two triangles per wall; each crosses the pad without needing an
        # enclosed vertex. Surface clipping must not substitute nearest points.
        points=np.array([[-.01,-.0002,-.05],[.01,-.0002,-.05],[.01,-.0002,.05],[-.01,-.0002,.05],
                         [-.01,.0002,-.05],[.01,.0002,-.05],[.01,.0002,.05],[-.01,.0002,.05]])
        grid=np.arange(8).reshape(2,2,2);faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
        self.assertGreater(min(mouth_capture(points,faces,grid,[-.001,.001])["area_mm2"]),900.)
        points[4:,1]=.002
        self.assertEqual(mouth_capture(points,faces,grid,[-.001,.001])["area_mm2"][1],0.)


if __name__=="__main__":unittest.main()
