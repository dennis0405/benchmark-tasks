import unittest
import numpy as np
from pouch_geometry import open_pouch
from fold_metrics import fold_metrics


class FoldMeasurements(unittest.TestCase):
    def test_unfolded_bag_does_not_count_as_folded(self):
        q, _, grid = open_pouch(nx=40, ny=32, mouth_half_gap=.0006, foldable_mouth=True)
        report = fold_metrics(q, grid, [(35, 30, 40)])
        self.assertTrue(report["valid"])
        self.assertEqual(report["folds"][0]["folded_fraction"], 0.)

    def test_fold_and_world_rotation_are_measured_equally(self):
        q, _, grid = open_pouch(nx=40, ny=32, mouth_half_gap=.0006, foldable_mouth=True)
        flap = q[:, 0] > .175001
        q[flap, 0] = .35-q[flap, 0]
        q[flap, 2] += .002
        report = fold_metrics(q, grid, [(35, 30, 40)])
        self.assertGreater(report["folds"][0]["folded_fraction"], .95)
        rotated = q@np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) + [2, -3, 1]
        other = fold_metrics(rotated, grid, [(35, 30, 40)])
        self.assertAlmostEqual(report["folds"][0]["median_angle_rad"],
                               other["folds"][0]["median_angle_rad"], places=5)

    def test_missing_geometry_fails_closed(self):
        q, _, grid = open_pouch(nx=40, ny=32)
        q[0, 0] = np.nan
        self.assertFalse(fold_metrics(q, grid, [(35, 30, 40)])["valid"])


if __name__ == "__main__":
    unittest.main()
