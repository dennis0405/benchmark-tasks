"""Gate regressions: finite motion or final-only geometry cannot imply success."""
from copy import deepcopy
import unittest
from verify_native_hold import evaluate


class HoldGateTest(unittest.TestCase):
    def example(self):
        result = {'all_finite': True, 'substeps': 10, 'geometry': {'states_checked': 1811,
                  'clip_intrusion_states': 0, 'self_crossing_states': 0}}
        shapes = {'shape_body': [0, 1], 'shape_local': [[.022, 0, -.004], [.022, 0, .004]],
                  'shape_scale': [[.022, .05, .003]]*2}
        trace = [{'step': i+1, 'shape_fold_contacts': {str(j): {'active_fold_contacts': 5,
            'fold_contact_y_span_m': .1} for j in (0, 1)}, 'fold_centroid_clip_local_m': [.025, 0, .002],
            'clip_root_pose': [.184, 0, .01], 'jaw_angle_rad': -.1} for i in range(181)]
        return result, trace, shapes

    def test_full_three_seconds_required(self):
        args = self.example()
        self.assertTrue(evaluate(*args)['passed'])
        args[0]['geometry']['states_checked'] = 1801
        self.assertFalse(evaluate(args[0], args[1][:-1], args[2])['passed'])

    def test_early_intrusion_rejects_final_stable_window(self):
        result, trace, shapes = self.example()
        result['geometry']['clip_intrusion_states'] = 1
        self.assertFalse(evaluate(result, trace, shapes)['passed'])

    def test_missing_jaw_contact_slip_and_joint_limit_each_reject(self):
        for change in ('contact', 'slip', 'joint', 'self_crossing'):
            result, trace, shapes = deepcopy(self.example())
            if change == 'contact':
                trace[90]['shape_fold_contacts']['1']['active_fold_contacts'] = 0
            elif change == 'slip':
                trace[90]['fold_centroid_clip_local_m'][0] += .01
            elif change == 'joint':
                trace[90]['jaw_angle_rad'] = .5
            else:
                result['geometry']['self_crossing_states'] = 1
            with self.subTest(change=change):
                self.assertFalse(evaluate(result, trace, shapes)['passed'])


if __name__ == '__main__':
    unittest.main()
