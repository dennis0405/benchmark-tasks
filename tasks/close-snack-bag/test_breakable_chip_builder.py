import unittest
import numpy as np
import warp as wp
import newton
from types import SimpleNamespace
from breakable_chip_builder import add_breakable_chips, ChipFractureMonitor


class ChipFractureMonitorTest(unittest.TestCase):
    def test_bond_breaks_only_target_and_never_heals(self):
        wp.init()
        builder = newton.ModelBuilder(gravity=wp.vec3(0,0,0))
        chips, seams = add_breakable_chips(builder)
        builder.color()
        model = builder.finalize(device='cuda:0')
        monitor = ChipFractureMonitor(model, seams, .0006)
        positions = model.particle_q.numpy()
        q = wp.array(positions, dtype=wp.vec3, device=model.device)
        state = SimpleNamespace(particle_q=q)
        monitor.apply(state); wp.synchronize()
        np.testing.assert_array_equal(monitor.flags.numpy(), [0,0,0,0])
        moved = positions.copy()
        moved[chips[0][18:], 0] += .001
        wp.copy(q, wp.array(moved, dtype=wp.vec3, device=model.device))
        monitor.apply(state); wp.synchronize()
        np.testing.assert_array_equal(monitor.flags.numpy(), [1,0,0,0])
        self.assertTrue(np.all(model.spring_stiffness.numpy()[seams[0]] == 0))
        self.assertTrue(np.all(model.spring_stiffness.numpy()[seams[1]] > 0))
        wp.copy(q, wp.array(positions, dtype=wp.vec3, device=model.device))
        monitor.apply(state); wp.synchronize()
        np.testing.assert_array_equal(monitor.flags.numpy(), [1,0,0,0])


if __name__ == '__main__':
    unittest.main()
