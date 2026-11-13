import unittest

from bridge_scoring import BridgeScorer


def evidence(tool="clip"):
    closed = dict(folded=True, aligned=True, jaw_on_layers=True,
                  still=True, contact=True, overlap=0.096)
    empty = dict(folded=False, aligned=False, jaw_on_layers=False,
                 still=True, contact=False, overlap=0.0)
    return {"violation": False, "clear": True,
            "tools": {"clip": closed if tool == "clip" else empty,
                      "peg": closed if tool == "peg" else empty}}


class TestBridgeScorer(unittest.TestCase):
    def score(self, scorer, step, sample):
        return scorer.update(step=step, objects={}, root_poses={}, robot={}, mechanics=sample)

    def test_both_tools_require_sixty_consecutive_frames(self):
        for tool in ("clip", "peg"):
            scorer = BridgeScorer()
            for step in range(1, 60):
                self.assertEqual(self.score(scorer, step, evidence(tool))["score"], 0.5)
            result = self.score(scorer, 60, evidence(tool))
            self.assertEqual((result["score"], result["held_frames"]), (1.0, 60))

    def test_missing_contact_or_clearance_resets_hold(self):
        scorer = BridgeScorer()
        self.score(scorer, 1, evidence())
        sample = evidence()
        sample["tools"]["clip"]["contact"] = False
        self.assertEqual(self.score(scorer, 2, sample)["score"], 0.0)
        sample = evidence()
        sample["clear"] = False
        self.assertEqual(self.score(scorer, 3, sample)["held_frames"], 0)
        self.assertEqual(self.score(scorer, 4, evidence())["held_frames"], 1)

    def test_gap_and_tool_change_reset_hold(self):
        scorer = BridgeScorer()
        self.score(scorer, 1, evidence())
        self.assertEqual(self.score(scorer, 3, evidence())["held_frames"], 1)
        self.assertEqual(self.score(scorer, 4, evidence("peg"))["held_frames"], 1)
        with self.assertRaises(ValueError):
            self.score(scorer, 4, evidence())

    def test_violation_latches_and_missing_evidence_fails_closed(self):
        scorer = BridgeScorer()
        self.assertFalse(self.score(scorer, 1, None)["valid"])
        sample = evidence()
        sample["violation"] = True
        self.assertEqual(self.score(scorer, 2, sample)["score"], 0.0)
        self.assertEqual(self.score(scorer, 3, evidence())["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
