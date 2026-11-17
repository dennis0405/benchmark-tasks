"""Analytic controls for a moving perforated plate and an oblique ceramic rod."""

import math, unittest
from types import SimpleNamespace
import numpy as np
from passage import Passage

cfg = SimpleNamespace(
    HOLES=[(0, 0)],
    APERTURE=0.0127,
    MEDIA_DIAMETER=0.009525,
    MEDIA_LENGTH=0.022225,
    FLOOR_T=0.003,
    RECEIVER_INNER_XY=0.280,
    SCREEN_X=0.37,
    SCREEN_Y=0.0,
    TABLETOP_HEIGHT=0.75,
    RECEIVER_FLOOR_T=0.004,
    RECEIVER_WALL_H=0.05,
)


def trajectory(x, vertical=True):
    monitor = Passage(cfg)
    q = (
        [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)]
        if vertical
        else [0, 0, 0, 1]
    )
    for k, z in enumerate(np.linspace(0.88, 0.769, 450)):
        poses = {
            "screen": np.array([0.37, 0, 0.85, 0, 0, 0, 1]),
            "media_00": np.array([0.37 + x, 0, z, *q]),
        }
        monitor.advance(poses, k)
    return monitor


class ApertureTests(unittest.TestCase):
    def test_axis_aligned_crossing(self):
        m = trajectory(0)
        self.assertEqual(m.received, {"media_00"})
        self.assertFalse(m.invalid)

    def test_through_solid(self):
        m = trajectory(0.02)
        self.assertFalse(m.received)
        self.assertIn("media_00", m.invalid)

    def test_horizontal_rod_will_not_fit(self):
        m = trajectory(0, False)
        self.assertFalse(m.passed)

    def test_receiver_teleport(self):
        m = Passage(cfg)
        m.advance(
            {
                "screen": np.array([0.37, 0, 0.85, 0, 0, 0, 1]),
                "media_00": np.array([0.37, 0, 0.77, 0, 0, 0, 1]),
            },
            0,
        )
        self.assertFalse(m.received)
        self.assertIn("media_00", m.invalid)


if __name__ == "__main__":
    unittest.main()
