"""Cross-check the independent GPU checks against CPU geometry references."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from check_native_clip import penetrating_triangles,closing_face_mask
from gpu_contact_checks import boxes, plane_points, plane_boxes, rigid_boxes, GeometryChecks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pouch_geometry import open_pouch
from bag_surface_mesh import folded_positions


class GPUChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = 'cuda:0'

    def box_flags(self, points, faces, body, local, half):
        n = len(faces)
        counts = wp.zeros(1, dtype=int, device=self.device)
        flags = wp.zeros(n, dtype=int, device=self.device)
        wp.launch(boxes, dim=n, inputs=[
            wp.array(points, dtype=wp.vec3, device=self.device),
            wp.array(faces, dtype=wp.vec3i, device=self.device),
            wp.array([body], dtype=wp.transform, device=self.device),
            wp.array([0], dtype=int, device=self.device),
            wp.array([local], dtype=wp.transform, device=self.device),
            wp.array([half], dtype=wp.vec3, device=self.device),
            wp.array([0], dtype=int, device=self.device), n, 0, 1.e-7, counts, flags], device=self.device)
        return flags.numpy()

    def test_random_transformed_triangles_match_cpu_sat(self):
        rng = np.random.default_rng(31)
        points = rng.uniform(-.1, .1, (1500, 3)).astype(np.float32)
        faces = np.arange(1500, dtype=np.int32).reshape(-1, 3)
        body = np.r_[[.01, .02, .03], Rotation.from_euler('xyz', [.2, -.4, .7]).as_quat()].astype(np.float32)
        local = np.array([.022, 0, -.004, 0, 0, 0, 1], dtype=np.float32)
        half = np.array([.022, .05, .003], dtype=np.float32)
        rot = Rotation.from_quat(body[3:]).as_matrix()
        relative = (points-(body[:3]+rot@local[:3]))@rot
        expected = np.zeros(len(faces), dtype=int)
        expected[penetrating_triangles(relative, faces, half)] = 1
        np.testing.assert_array_equal(expected, self.box_flags(points, faces, body, local, half))

    def test_crossing_without_inside_vertices_and_boundary_touch(self):
        points = np.array([[-2, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        identity = [0, 0, 0, 0, 0, 0, 1]
        self.assertEqual(int(self.box_flags(points, faces, identity, identity, [.5]*3)[0]), 1)
        points[:, 2] = .5
        self.assertEqual(int(self.box_flags(points, faces, identity, identity, [.5]*3)[0]), 0)

    def checker(self, points, faces, steps):
        state = SimpleNamespace(particle_q=wp.array(points, dtype=wp.vec3, device=self.device))
        model = SimpleNamespace(device=wp.get_device(self.device),
            shape_body=wp.empty(0, dtype=int, device=self.device))
        return state, GeometryChecks(model, [state], faces, steps)

    def test_bvh_detects_crossing_and_rejects_separated_triangles(self):
        points = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0],
                           [0, 0, -1], [0, 0, 1], [.2, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        state, checks = self.checker(points, faces, 2)
        checks.check(state, 0)
        self.assertGreater(checks.self_counts.numpy()[0], 0)
        points[3:, 0] += 4
        state.particle_q.assign(points)
        checks.check(state, 1)
        self.assertEqual(checks.self_counts.numpy()[1], 0)
        np.testing.assert_array_equal(checks.checked_counts.numpy(),[1,1])

    def test_real_folded_pouch_initialization_has_no_strict_crossing(self):
        q, faces, _ = open_pouch(nx=80, ny=64, mouth_half_gap=.0003, foldable_mouth=True)
        initial = folded_positions(q+[0, 0, .019], radius=.0018)-[0, 0, .019]+[0, 0, .025]
        state, checks = self.checker(initial, faces, 1)
        checks.check(state, 0)
        self.assertEqual(checks.self_counts.numpy()[0], 0)

    def test_plane_vertex_crossing_and_boundary_tolerance(self):
        points = wp.array([[0, 0, -.001], [0, 0, -.5e-7], [0, 0, .001]], dtype=wp.vec3, device=self.device)
        counts = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(plane_points, dim=3, inputs=[points, 0, counts], device=self.device)
        self.assertEqual(counts.numpy()[0], 1)

    def test_rotated_box_plane_corners(self):
        quat = Rotation.from_euler('y', np.pi/4).as_quat()
        local = wp.array([[0, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device=self.device)
        owners = wp.array([0], dtype=int, device=self.device)
        half = wp.array([[.1, .1, .1]], dtype=wp.vec3, device=self.device)
        ids = wp.array([0], dtype=int, device=self.device)
        counts = wp.zeros(3, dtype=int, device=self.device)
        for slot, height in enumerate((.1, .2, np.sqrt(.02))):
            body = wp.array([np.r_[[0, 0, height], quat]], dtype=wp.transform, device=self.device)
            wp.launch(plane_boxes, dim=8, inputs=[body, owners, local, half, ids, slot, counts], device=self.device)
        np.testing.assert_array_equal(counts.numpy(), [2, 0, 0])

    def test_clip_handles_overlap_at_old_opening_but_not_revised_opening(self):
        owners=wp.array([0,0,1,1],dtype=int,device=self.device)
        local=wp.array([[.022,0,-.004,0,0,0,1],[-.018,0,-.008,0,0,0,1],
                        [.022,0,.004,0,0,0,1],[-.018,0,.008,0,0,0,1]],dtype=wp.transform,device=self.device)
        scale=wp.array([[.022,.05,.003],[.018,.012,.003]]*2,dtype=wp.vec3,device=self.device)
        pairs=wp.array([[0,2],[0,3],[1,2],[1,3]],dtype=wp.vec2i,device=self.device)
        counts=wp.zeros(3,dtype=int,device=self.device);flags=wp.zeros(4,dtype=int,device=self.device)
        for slot,angle in enumerate((-.55,-.27,0.)):
            body=wp.array([[0,0,0,0,0,0,1],np.r_[[0,0,0],Rotation.from_euler('y',angle).as_quat()]],
                          dtype=wp.transform,device=self.device)
            wp.launch(rigid_boxes,dim=4,inputs=[body,owners,local,scale,pairs,slot,counts,flags],device=self.device)
        values=counts.numpy()
        self.assertGreater(values[0],0)
        np.testing.assert_array_equal(values[1:],[0,0])

    def test_rigid_box_rotated_overlap_and_boundary(self):
        owners=wp.array([0,1],dtype=int,device=self.device)
        local=wp.array([[0,0,0,0,0,0,1]]*2,dtype=wp.transform,device=self.device)
        scale=wp.array([[.1,.05,.01]]*2,dtype=wp.vec3,device=self.device)
        pairs=wp.array([[0,1]],dtype=wp.vec2i,device=self.device)
        counts=wp.zeros(3,dtype=int,device=self.device);flags=wp.zeros(1,dtype=int,device=self.device)
        quat=Rotation.from_euler('z',np.pi/2).as_quat()
        for slot,x in enumerate((.14,.15,.4)):
            body=wp.array([[0,0,0,0,0,0,1],np.r_[[x,0,0],quat]],dtype=wp.transform,device=self.device)
            wp.launch(rigid_boxes,dim=1,inputs=[body,owners,local,scale,pairs,slot,counts,flags],device=self.device)
        np.testing.assert_array_equal(counts.numpy(),[1,0,0])

    def test_gripping_face_normal_uses_world_orientation(self):
        body=Rotation.from_euler('xyz',[.3,.7,-1.2]);shape=Rotation.from_euler('y',-.4)
        direction=(body*shape).apply([0,0,1]);side=(body*shape).apply([1,0,0])
        normals=np.array([direction,-direction,side])
        np.testing.assert_array_equal(closing_face_mask(normals,body.as_quat(),shape.as_quat(),1),[True,False,False])
        np.testing.assert_array_equal(closing_face_mask(normals,body.as_quat(),shape.as_quat(),-1),[False,True,False])


if __name__ == '__main__':
    unittest.main()
