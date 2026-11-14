"""Containment controls for the real open-pouch topology and its sealed end."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pouch_geometry import open_pouch
from contents_inside import contained_centres


class ContainmentTests(unittest.TestCase):
    def test_inside_outside_and_open_mouth_cap(self):
        q,f,g=open_pouch(nx=20,ny=16,mouth_half_gap=.004)
        points=np.array([[.08,0,0],[.198,0,0],[.22,0,0],[-.02,0,0],[.08,.10,0],[.08,0,.06]])
        np.testing.assert_array_equal(contained_centres(q,f,g,points),[True,True,False,False,False,False])

    def test_rotation_and_translation_do_not_change_classification(self):
        q,f,g=open_pouch(nx=20,ny=16,mouth_half_gap=.004)
        points=np.array([[.08,0,0],[.12,.12,0]])
        rotation=Rotation.from_euler('xyz',[20,35,140],degrees=True);translation=np.array([.8,-.1,.97])
        np.testing.assert_array_equal(contained_centres(rotation.apply(q)+translation,f,g,
            rotation.apply(points)+translation),[True,False])

    def test_non_finite_surface_never_passes(self):
        q,f,g=open_pouch(nx=8,ny=8);q[0,0]=np.nan
        self.assertFalse(contained_centres(q,f,g,[[.08,0,0]])[0])


if __name__=='__main__':unittest.main()
