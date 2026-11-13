"""Exercise continuous cursor descent through the studio seating boundary."""
from types import SimpleNamespace
import unittest
import numpy as np
from fold_controller import FoldController


class PressHoldTests(unittest.TestCase):
    def test_force_hold_preserves_downward_cursor_integral(self):
        actual = np.array([.7, .09, 1.0])

        class Panel:
            cursor_pos = actual.copy()

            def aim(self, position, quat, *, moving):
                self.cursor_pos = position.copy()
                self.dragging = moving

            def studio_seat(self):
                # A submillimetre release is seated on the measured hand by the
                # upstream panel. Continuous contact control must remain a drag.
                if not self.dragging:
                    self.cursor_pos = actual.copy()

        panel = Panel()
        controller = FoldController.__new__(FoldController)
        controller.support_holds = {"left": dict(floor_z=.98, quat=[0,0,0,1], filtered=0.)}
        controller.session = SimpleNamespace(arms={"left": panel})
        controller.sim = SimpleNamespace(
            pipeline=SimpleNamespace(program=SimpleNamespace(solver=SimpleNamespace(
                shape_contact_forces=lambda: np.zeros((2,3))))),
            model=SimpleNamespace(shape_body=SimpleNamespace(numpy=lambda:np.array([6,7]))),
            request=SimpleNamespace(physics=SimpleNamespace(control_hz=20)),
            scene=SimpleNamespace(embodiment=SimpleNamespace(arms={
                "left":SimpleNamespace(finger_bodies=[[6],[7]])})))
        for _ in range(10):
            controller.update_support_holds()
            panel.studio_seat()
        assert panel.dragging
        np.testing.assert_allclose(panel.cursor_pos, actual-[0,0,.002])

if __name__ == "__main__":
    unittest.main()
