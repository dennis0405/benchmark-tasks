"""Synthetic scorer checks; these do not substitute for physics or a YAM demonstration."""
from types import SimpleNamespace
import unittest

import numpy as np
import arena_scene as task


def state(pos, quat, joints, still=True):
    return SimpleNamespace(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat, float), joints=joints,
                           resting=lambda **_kw: still)


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.objects = {
            "bag": state([0, 0, 0], [0, 0, 0, 1], {"fold": -3.14, "fold2": 3.14}),
            "clip": state([0.12, 0, 0.0213], [0, 0, 1, 0], {"clip_hinge": -0.10}),
        }
        task._STATE["arms_clear"] = True
        task._STATE["violation"] = False
        task._STATE["hold_steps"] = 60
        task._STATE["holding_tool"] = "clip"
        task._STATE["contacts"] = {"clip": True, "peg": True}

    def test_valid_geometry(self):
        self.assertEqual(task.score(self.objects), 1.0)

    def test_open_mouth_and_unaligned_clip(self):
        self.objects["bag"].joints["fold"] = 0
        self.assertEqual(task.score(self.objects), 0.0)
        self.objects["bag"].joints["fold"] = -3.14
        self.objects["clip"].pos[1] = 0.09
        self.assertEqual(task.score(self.objects), 0.5)

    def test_open_clip_and_nearby_arm_do_not_pass(self):
        self.objects["clip"].joints["clip_hinge"] = -0.43
        self.assertEqual(task.score(self.objects), 0.0)
        self.objects["clip"].joints["clip_hinge"] = -0.10
        task._STATE["arms_clear"] = False
        self.assertEqual(task.score(self.objects), 0.5)

    def test_spilled_contents_override_full_score(self):
        task._STATE["violation"] = True
        self.assertEqual(task.score(self.objects), 0.0)

    def test_no_contact_fails_closed(self):
        task._STATE["contacts"]["clip"] = False
        self.assertEqual(task.score(self.objects), 0.0)

    def test_full_score_requires_three_seconds(self):
        task._STATE["hold_steps"] = 59
        self.assertEqual(task.score(self.objects), 0.5)

    def test_wide_peg_is_scored_equally(self):
        self.objects["peg"] = self.objects.pop("clip")
        task._STATE["holding_tool"] = "peg"
        self.assertEqual(task.score(self.objects), 1.0)


if __name__ == "__main__":
    unittest.main()
