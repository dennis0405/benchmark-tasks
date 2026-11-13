"""Force routing tests, not a physical stability certificate."""
import unittest
import numpy as np
import warp as wp
from tool_feedback import tool_reaction


class ReactionRouting(unittest.TestCase):
    def test_full_wrench_offset_mask_and_active_count(self):
        forces = np.array([[99]*6, [88]*6, [1,2,3,4,5,6],
                           [-1,2,3,4,5,1], [9]*6, [77]*6], np.float32)
        body = wp.array(np.full((2,6), .5, np.float32), dtype=wp.spatial_vector, device="cpu")
        wp.launch(tool_reaction, dim=4,
                  inputs=[wp.array([3],dtype=int,device="cpu"),
                          wp.array([1,1,0,2],dtype=int,device="cpu"), 2,
                          wp.array(forces,dtype=wp.spatial_vector,device="cpu"),
                          wp.array([0,1,-1],dtype=int,device="cpu"),
                          wp.array([0,1],dtype=int,device="cpu")],
                  outputs=[body], device="cpu")
        expected = np.full((2,6), .5, np.float32)
        expected[1] += forces[2]+forces[3]
        np.testing.assert_array_equal(body.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
