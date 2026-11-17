"""Controlled 180-degree trajectories; the native contact trials are separate."""

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from grading import Grade
from layout import MOUTH, OVEN_Y
from food_contact import top_surface_load
from test_parking import ForceView, pose


class EngagementTest(unittest.TestCase):
    def trajectory(self, contact, sign=1):
        grade = Grade.__new__(Grade)
        grade.dt = 1 / 240
        grade.nfilters = 4
        grade.reset()
        grade.contacts = ForceView([[0, 0, 2.7], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
        grade.rack = ForceView([[0, 0, 0], [0, 0, 0], [0, 0, 3], [0, 0, 3]])
        center = np.array([MOUTH + 0.23, OVEN_Y, 1.002])
        for angle in np.linspace(0, sign * 180, 181):
            rot = Rotation.from_euler("z", angle, degrees=True)
            quat = rot.as_quat()
            top = contact in ("shaft", "top", "core_top")
            z = 0.016 if top else (0.009 if contact == "side" else 0.001)
            radius = 0.070 if contact == "core_top" else 0.142
            point = center + rot.apply([radius, 0, z])
            if contact == "shaft":
                tool = point - rot.apply([-0.365, 0, 0.042])
            else:
                tool = point - rot.apply([0.04, 0, 0.0008])
            rim = center + rot.apply([0.105, 0, 0.008])
            normal = [1.0, 0.0, 0.0] if contact == "side" else [0.0, 0.0, 1.0]
            core_id = 16 if contact == "core_top" else 0
            top_load = top_surface_load(
                point[None, :],
                np.array([normal]),
                np.array([1.0]),
                np.array([core_id]),
                np.r_[center, quat],
                np.array([np.r_[rim, quat]]),
            )
            loads = (1.0 - top_load, top_load)
            grade.food_contact = SimpleNamespace(loads=lambda food: loads)
            grade.food = pose(np.r_[center, quat])
            grade.tool = pose(np.r_[tool, quat])
            grade.observe()
        self.assertTrue(grade.had_grasp)
        self.assertIsNone(grade.fault)
        self.assertAlmostEqual(grade.angle, sign * 180)
        return grade

    def test_shaft_on_top_rotates_but_gets_no_credit(self):
        g = self.trajectory("shaft")
        self.assertEqual((g.score, g.contact_angle, g.engaged_steps), (0.0, 0.0, 0))

    def test_blade_on_top_gets_no_credit(self):
        self.assertEqual(self.trajectory("top").score, 0.0)

    def test_core_top_is_also_excluded(self):
        self.assertEqual(self.trajectory("core_top").score, 0.0)

    def test_side_turn_without_insertion_is_credited(self):
        self.assertEqual(self.trajectory("side").score, 0.8)

    def test_loaded_blade_under_rim_in_either_direction(self):
        for sign in (-1, 1):
            g = self.trajectory("under", sign)
            self.assertAlmostEqual(g.contact_angle, sign * 180)
            self.assertEqual(g.score, 0.8)  # tool still held
