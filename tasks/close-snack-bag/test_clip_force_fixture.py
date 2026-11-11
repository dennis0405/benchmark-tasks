"""Force fixture must respect its bounds and release the clip."""
import unittest
import numpy as np
import warp as wp
from types import SimpleNamespace
from clip_force_fixture import ClipForceFixture


class ClipForceFixtureTest(unittest.TestCase):
    def test_force_limit_and_release(self):
        wp.init()
        m = SimpleNamespace(device='cuda:0')
        state = SimpleNamespace(body_q=wp.array([wp.transform(wp.vec3(0,0,0), wp.quat(0,0,1,0))], dtype=wp.transform,device=m.device),
                                body_qd=wp.zeros(1,dtype=wp.spatial_vector,device=m.device),
                                body_f=wp.zeros(1,dtype=wp.spatial_vector,device=m.device))
        fixture=ClipForceFixture(m,0,limit=.5)
        fixture.set([1,0,0],True);fixture.apply(state);wp.synchronize()
        np.testing.assert_allclose(state.body_f.numpy()[0,:3],[.5,0,0],atol=1e-6)
        state.body_f.zero_();fixture.set([1,0,0],False);fixture.apply(state);wp.synchronize()
        np.testing.assert_allclose(state.body_f.numpy(),0,atol=1e-8)


if __name__ == '__main__':
    unittest.main()
