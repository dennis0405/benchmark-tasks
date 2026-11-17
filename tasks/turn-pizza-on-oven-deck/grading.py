"""One outcome grader for the native demo and arbitrary policy actions."""

import math
import numpy as np
from food_contact import FoodContact, SUPPORT_N
from layout import MOUTH, OVEN_Y, PARK


class Grade:
    def __init__(
        self, physics_view, food_paths, arms, rate=240, tool_path="/World/Peel"
    ):
        self.dt = 1 / rate
        self.nfilters = 2 + len(arms)
        self.food = physics_view.create_rigid_body_view(food_paths[0])
        self.tool = physics_view.create_rigid_body_view(tool_path)
        sources = (
            ["/World/Food/*"]
            if food_paths[0] == "/World/Food/core"
            else [food_paths[0], "/World/Food/*"]
        )
        self.contacts = physics_view.create_rigid_contact_view(
            sources,
            filter_patterns=[["/World/Stone", tool_path, *arms]] * len(sources),
            max_contact_data_count=2048,
        )
        self.rack = physics_view.create_rigid_contact_view(
            [tool_path],
            filter_patterns=[["/World/Stand/front", "/World/Stand/rear", *arms]],
            max_contact_data_count=512,
        )
        self.food_contact = FoodContact(physics_view, tool_path, food_paths[0], self.dt)
        self.reset()

    def reset(self):
        self.fault = None
        self.last_yaw = None
        self.engaged_steps = 0
        self.top_steps = 0
        self.angle = self.contact_angle = self.shift = self.dwell = 0.0
        self.min_deck = math.inf
        self.anchor = None
        self.max_drift = 0.0
        self.min_supports = [math.inf, math.inf]
        self.had_grasp = False

    @property
    def score(self):
        if self.fault or not self.had_grasp:
            return 0.0
        progress = max(
            0.0,
            min(abs(self.angle), math.copysign(1.0, self.angle) * self.contact_angle),
        )
        if 175 <= abs(self.angle) <= 185 and progress >= 165:
            return 1.0 if self.dwell + 1e-8 >= 3.0 else 0.8
        return (
            0.6
            if progress >= 135
            else 0.4
            if progress >= 90
            else 0.2
            if progress >= 45
            else 0.0
        )

    def observe(self):
        food = self.food.get_transforms().numpy()[0]
        peel = self.tool.get_transforms().numpy()[0]
        forces = (
            self.contacts.get_contact_force_matrix(dt=self.dt)
            .numpy()
            .reshape(-1, self.nfilters, 3)
        )
        normal = float(np.abs(forces[:, 0, 2]).sum())
        tool_food = np.linalg.norm(forces[:, 1], axis=1).sum()
        rack = self.rack.get_contact_force_matrix(dt=self.dt).numpy().reshape(-1, 3)
        support = np.abs(rack[:2, 2])
        hand_load = np.linalg.norm(rack[2:], axis=1)
        self.had_grasp |= np.count_nonzero(hand_load > 2.0) >= 2
        shift = math.hypot(food[0] - (MOUTH + 0.23), food[1] - OVEN_Y)
        self.shift = max(self.shift, shift)
        self.min_deck = min(self.min_deck, normal)
        x, y, z, w = map(float, food[3:])
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        delta = (
            0.0
            if self.last_yaw is None
            else (yaw - self.last_yaw + 180.0) % 360.0 - 180.0
        )
        self.last_yaw = yaw
        self.angle += delta
        if tool_food > SUPPORT_N:
            non_top, top = self.food_contact.loads(food)
            if non_top > SUPPORT_N and top <= SUPPORT_N:
                self.contact_angle += delta
                self.engaged_steps += 1
            elif top > SUPPORT_N:
                self.top_steps += 1
        if self.fault is None:
            if shift > 0.05:
                self.fault = "pizza center exceeded 50 mm"
            elif normal < 0.001:
                self.fault = "lost deck support"
            elif 1.0 - 2 * (x * x + y * y) < math.cos(math.radians(15)):
                self.fault = "pizza tilt exceeded 15 degrees"
            elif np.any(np.linalg.norm(forces[:, 2:], axis=-1) > 0.1):
                self.fault = "robot touched pizza"
        x, y, z, w = map(float, peel[3:])
        direction = np.array(
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
        )
        shaft = peel[:3] - 0.34 * direction
        near = (
            np.linalg.norm(peel[:2] - PARK[:2]) <= 0.03
            and abs(peel[2] - PARK[2]) < 0.005
        )
        aligned = direction[0] >= math.cos(math.radians(8)) and 1.0 - 2 * (
            x * x + y * y
        ) > math.cos(math.radians(5))
        clear = not np.any(hand_load > 0.02) and tool_food < 0.02
        if near and aligned and clear and np.all(support > 0.10) and self.had_grasp:
            points = np.stack([peel[:3], shaft])
            drift = (
                0.0
                if self.anchor is None
                else float(np.linalg.norm(points - self.anchor, axis=1).max())
            )
            if self.anchor is None or drift > 0.002:
                self.anchor = points.copy()
                self.dwell = 0.0
                self.max_drift = 0.0
                self.min_supports = [math.inf, math.inf]
            else:
                self.max_drift = max(self.max_drift, drift)
            self.dwell += self.dt
            self.min_supports = np.minimum(self.min_supports, support).tolist()
        else:
            self.anchor = None
            self.dwell = 0.0
            self.max_drift = 0.0
            self.min_supports = [math.inf, math.inf]

    def result(self):
        return dict(
            score=self.score,
            fault=self.fault,
            rotation_deg=self.angle,
            max_shift_mm=self.shift * 1000,
            minimum_deck_N=self.min_deck,
            signed_non_top_deg=self.contact_angle,
            engaged_steps=self.engaged_steps,
            top_contact_steps=self.top_steps,
            parked_seconds=self.dwell,
            parked_drift_mm=self.max_drift * 1000,
            parked_minimum_support_N=[
                float(v) if math.isfinite(v) else None for v in self.min_supports
            ],
        )
