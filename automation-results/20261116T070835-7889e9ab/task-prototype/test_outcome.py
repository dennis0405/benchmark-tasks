import unittest
from outcome import Outcome


class OutcomeTests(unittest.TestCase):
    def test_partial(self):
        score = Outcome(1 / 240)
        self.assertEqual(score.advance(0, True, True, False), 0)
        self.assertAlmostEqual(score.advance(0.5, False, True, False), 0.459183673469)

    def test_release_boundary(self):
        score = Outcome(1 / 240)
        for _ in range(719):
            self.assertEqual(score.advance(1, True, True, False), 0.9)
        self.assertEqual(score.advance(1, True, True, False), 1)
        self.assertEqual(score.advance(1, True, False, False), 0.9)
        self.assertEqual(score.dwell_steps, 0)

    def test_permanent_failure(self):
        score = Outcome(1 / 240)
        self.assertEqual(score.advance(0.5, True, True, True), 0)
        for _ in range(721):
            self.assertEqual(score.advance(1, True, True, False), 0)


if __name__ == "__main__":
    unittest.main()
