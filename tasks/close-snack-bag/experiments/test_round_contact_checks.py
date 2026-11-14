"""Positive and negative cases for face, edge and capsule-axis penetration."""
import unittest
import numpy as np
import warp as wp
from round_contact_checks import triangle_segment_distance_sq,scene_round,scene_ellipsoids


@wp.kernel
def distances(segments:wp.array[wp.vec3],out:wp.array[float]):
    i=wp.tid()
    out[i]=triangle_segment_distance_sq(wp.vec3(-1.,-1.,0.),wp.vec3(1.,-1.,0.),wp.vec3(0.,1.,0.),
                                       segments[2*i],segments[2*i+1])


class RoundContactTests(unittest.TestCase):
    def test_ellipsoid_face_crossing_and_separation(self):
        # Every triangle vertex is outside the solid; its face crosses it.
        for device in ("cpu",*(["cuda:0"] if wp.is_cuda_available() else [])):
            with self.subTest(device=device):
                q=wp.array([[-1,-1,0],[1,-1,0],[0,1,0]],dtype=wp.vec3,device=device)
                mesh=wp.Mesh(points=q,indices=wp.array([0,1,2],dtype=int,device=device))
                counts=wp.zeros(3,dtype=int,device=device)
                wp.launch(scene_ellipsoids,3,inputs=[q,wp.array([[0,1,2]],dtype=wp.vec3i,device=device),mesh.id,
                    wp.array([[2,0,0,0,0,0,1]],dtype=wp.transform,device=device),
                    wp.array([-1,-1,0],dtype=int,device=device),
                    wp.array([[0,0,0,0,0,0,1],[0,0,.021,0,0,0,1],[0,0,0,0,0,0,1]],dtype=wp.transform,device=device),
                    wp.array([[.2,.1,.02]]*3,dtype=wp.vec3,device=device),
                    wp.array([0,1,2],dtype=int,device=device),counts],device=device)
                np.testing.assert_array_equal(counts.numpy(),[1,0,0])

    def test_scene_query_and_owner_transforms(self):
        for device in ("cpu",*(["cuda:0"] if wp.is_cuda_available() else [])):
            with self.subTest(device=device):
                q=wp.array([[-1,-1,0],[1,-1,0],[0,1,0]],dtype=wp.vec3,device=device)
                indices=wp.array([0,1,2],dtype=int,device=device)
                mesh=wp.Mesh(points=q,indices=indices)
                counts=wp.zeros(4,dtype=int,device=device)
                wp.launch(scene_round,4,inputs=[q,wp.array([[0,1,2]],dtype=wp.vec3i,device=device),mesh.id,
                    wp.array([[2,0,0,0,0,0,1]],dtype=wp.transform,device=device),
                    wp.array([-1,0,-1,-1],dtype=int,device=device),
                    wp.array([[0,0,.05,0,0,0,1],[0,0,.05,0,0,0,1],
                              [0,0,0,0,0,0,1],[0,-2,0,0,0,0,1]],dtype=wp.transform,device=device),
                    wp.array([[.1,0,0],[.1,0,0],[.1,.2,0],[.1,.2,0]],dtype=wp.vec3,device=device),
                    wp.array([3,3,4,4],dtype=int,device=device),
                    wp.array([0,1,2,3],dtype=int,device=device),counts],device=device)
                np.testing.assert_array_equal(counts.numpy(),[1,0,1,0])

    def test_axis_and_sphere_distances_on_cpu_and_cuda(self):
        segments=np.array([
            [[0,0,-1],[0,0,1]], # crosses face interior
            [[0,0,.2],[0,0,.2]], # sphere axis, inside projection
            [[0,0,.2],[.1,0,.2]], # parallel interior
            [[0,-2,-1],[0,-2,1]], # nearest triangle edge
            [[1,-1,0],[2,-1,0]], # exact triangle vertex
            [[0,0,-2],[0,0,-1]], # beyond segment endpoint
        ],np.float32).reshape(-1,3)
        for device in ("cpu",*(["cuda:0"] if wp.is_cuda_available() else [])):
            with self.subTest(device=device):
                out=wp.zeros(6,dtype=float,device=device)
                wp.launch(distances,6,inputs=[wp.array(segments,dtype=wp.vec3,device=device),out],device=device)
                np.testing.assert_allclose(out.numpy(),[0,.04,.04,1,0,1],atol=1e-7)


if __name__=="__main__":unittest.main()
