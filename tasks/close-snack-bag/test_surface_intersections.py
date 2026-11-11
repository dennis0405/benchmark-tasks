"""Triangle-intersection regression for the reset-time folded geometry."""
import unittest

import numpy as np
import open3d as o3d
from bag_surface_mesh import folded_positions, packet_mesh
from check_surface_intersections import strict_crossing
from initial_clip_clearance import inspect
from pouch_geometry import open_pouch


class SurfaceIntersectionTest(unittest.TestCase):
    def test_crossing_triangles_are_detected(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.25, 0]], float)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        self.assertGreater(len(mesh.get_self_intersecting_triangles()), 0)
        self.assertTrue(strict_crossing(vertices[faces[0]], vertices[faces[1]]))

    def test_coplanar_contact_is_not_crossing(self):
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        b = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]], float)
        overlapping = np.array([[0.2, 0.2, 0], [0.6, 0.2, 0], [0.2, 0.6, 0]], float)
        self.assertFalse(strict_crossing(a, b))
        self.assertTrue(strict_crossing(a, overlapping))

    def test_fold_with_longer_flat_mouth_has_no_initial_triangle_crossings(self):
        vertices, faces = packet_mesh(nx=160, flat_mouth=True, mouth_half_gap=0.0003,
                                      taper_start=0.12, taper_end=0.14)
        folded = folded_positions(vertices, radius=0.001)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(folded.astype(np.float64)),
                                         o3d.utility.Vector3iVector(faces))
        self.assertEqual(len(mesh.get_self_intersecting_triangles()), 0)

    def test_new_pouch_fold_reset_has_no_initial_crossings(self):
        points, faces, _ = open_pouch(nx=80, ny=64, mouth_half_gap=0.0003, foldable_mouth=True)
        shift = np.array([0, 0, .019], dtype=np.float32)
        folded = folded_positions(points + shift, radius=.0018) - shift
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(folded.astype(np.float64)),
                                         o3d.utility.Vector3iVector(faces))
        self.assertEqual(len(mesh.get_self_intersecting_triangles()), 0)

    def test_initial_clip_clearance_distinguishes_a_penetrating_placement(self):
        clear = inspect(clip_x=0.18, clip_z=0.043)
        colliding = inspect(clip_x=0.18, clip_z=0.040)
        self.assertTrue(all(not value["mesh_intersects_box_surface"] and value["vertices_inside_box"] == 0
                            for value in clear.values()))
        self.assertTrue(colliding["upper_jaw"]["mesh_intersects_box_surface"])


if __name__ == "__main__":
    unittest.main()
