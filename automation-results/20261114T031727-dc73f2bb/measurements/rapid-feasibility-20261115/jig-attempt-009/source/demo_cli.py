"""Task-local public-policy extension; the SDK embodiment and guards stay intact.

Only the robot's public joint-position observation is read. No simulator object state,
pose writes, constraint edits, Dream credentials, or framework monkeypatching is used.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
from inspect_robots.registry import policy
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import STATE_KEY
from yam_arena import ik
from yam_arena.scripted import YamScriptedPolicy


class TrackingDemo(YamScriptedPolicy):
    """Play public task waypoints and optionally correct slow joint tracking error."""

    def __init__(self, hold_s=5.0, ki=0.0, correction_limit=0.10):
        super().__init__(hold_s=hold_s)
        self.info = replace(self.info, name="rcauto_joint_tracking_demo")
        self.ki = float(ki)
        self.limit = float(correction_limit)
        if not 0 <= self.ki <= 1 or not 0 < self.limit <= .15:
            raise ValueError("demo correction gain/limit out of range")
        self.correction = np.zeros(14)
        self.trace = None

    def reset(self, scene):
        super().reset(scene)
        self.correction[:] = 0
        trace = os.environ.get("RAPID_JOINT_TRACE")
        self.trace = Path(trace) if trace else None
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            self.trace.write_text("")

    def act(self, observation):
        nominal = super().act(observation)
        action = nominal.actions[0]
        desired = np.asarray(action.data, dtype=float)
        actual = np.asarray(observation.state[STATE_KEY], dtype=float)
        if desired.shape != (14,) or actual.shape != (14,) or not np.isfinite(actual).all():
            raise ValueError("expected finite standard 14-D joint-position observation")
        arm_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        error = desired - actual
        self.correction[arm_indices] = np.clip(
            self.correction[arm_indices] + self.ki * self._dt * error[arm_indices],
            -self.limit, self.limit,
        )
        command = desired + self.correction
        # Normalized gripper targets are unchanged. The normal CLI clamp and delta
        # guards still inspect every resulting absolute joint command.
        command[[6, 13]] = desired[[6, 13]]
        t = float(observation.state_time)
        per_arm = {}
        for name, offset in (("left", 0), ("right", 7)):
            if name not in self._plans:
                continue
            plan = self._plans[name]
            target, rotation, _, label = plan.target(t)
            position, actual_rotation = ik.tcp_pose(actual[offset:offset+6], self._bases[name])
            per_arm[name] = {
                "phase": label,
                "actual_tcp_m": position.tolist(),
                "target_tcp_m": np.asarray(target).tolist(),
                "tcp_error_m": float(np.linalg.norm(position-target)),
                "actual_tcp_rotation": actual_rotation.tolist(),
                "target_tcp_rotation": np.asarray(rotation).tolist(),
            }
        if self.trace:
            record = {"state_time_s": t, "actual_joint_pos": actual.tolist(),
                      "nominal_action": desired.tolist(), "command": command.tolist(),
                      "correction": self.correction.tolist(), "arms": per_arm}
            with self.trace.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":"))+"\n")
        if self.ki == 0:
            return nominal
        return ActionChunk(actions=[Action(data=command, meta=action.meta)], control_hz=nominal.control_hz)


@policy("rcauto_joint_tracking_demo")
def make_tracking_demo(**kwargs):
    return TrackingDemo(**kwargs)


if __name__ == "__main__":
    from inspect_robots.cli import main
    main()
