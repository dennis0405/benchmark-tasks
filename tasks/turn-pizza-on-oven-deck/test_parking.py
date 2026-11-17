"""Regression checks for the last 3 seconds after a successful half-turn.

The synthetic contacts isolate scoring. The demo provides the physics evidence.
"""

import unittest
from types import SimpleNamespace
import numpy as np

from grading import Grade
from layout import MOUTH, OVEN_Y, PARK


class ForceView:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def get_contact_force_matrix(self, dt):
        return SimpleNamespace(numpy=lambda: self.values)


def pose(values):
    return SimpleNamespace(
        get_transforms=lambda: SimpleNamespace(numpy=lambda: values[None, :])
    )


class ParkingTest(unittest.TestCase):
    def setUp(self):
        self.grade = Grade.__new__(Grade)
        self.grade.dt = 1 / 240
        self.grade.nfilters = 4
        self.grade.reset()
        self.grade.had_grasp = True
        self.grade.angle = self.grade.contact_angle = 180.0
        self.grade.food = pose(
            np.array([MOUTH + 0.23, OVEN_Y, 1.002, 0.0, 0.0, 0.0, 1.0])
        )
        self.peel = np.r_[PARK, 0.0, 0.0, 0.0, 1.0]
        self.grade.tool = pose(self.peel)
        self.grade.contacts = ForceView([[0, 0, 2.7], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.grade.rack = ForceView([[0, 0, 1], [0, 0, 2], [0, 0, 0], [0, 0, 0]])

    def advance(self, n):
        for _ in range(n):
            self.grade.observe()

    def test_full_score_ends_when_rear_support_is_lost(self):
        self.advance(719)
        self.assertEqual(self.grade.score, 0.8)
        self.advance(1)
        self.assertEqual(self.grade.score, 1.0)
        self.grade.rack.values[1, 2] = 0.0
        self.advance(1)
        self.assertEqual(self.grade.score, 0.8)

    def test_robot_base_cannot_substitute_for_stand(self):
        self.grade.rack.values[1, 2] = 0.0
        self.grade.rack.values[2, 2] = 1.6
        self.advance(721)
        self.assertEqual(self.grade.score, 0.8)
        self.assertEqual(self.grade.dwell, 0.0)

    def test_shaft_motion_restarts_dwell_even_if_blade_center_stays(self):
        self.advance(718)
        # A one-degree yaw moves the shaft end by about six millimetres.
        half_angle = np.deg2rad(0.5)
        self.peel[5:] = [np.sin(half_angle), np.cos(half_angle)]
        self.advance(2)
        self.assertEqual(self.grade.score, 0.8)
        self.assertLess(self.grade.dwell, 0.01)
