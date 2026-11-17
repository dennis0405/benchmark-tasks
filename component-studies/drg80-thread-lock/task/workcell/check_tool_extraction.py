from itertools import product
import numpy as np
from tool_extraction import certify_extraction

handle = np.array(list(product((-0.012, 0.012), (-0.012, 0.012), (-0.025, 0.025))))
h = np.array([0.0, 0.0, 1.0])
# Two separated walls enclose the centroid in their overall box, but leave
# the complete axial extraction cylinder clear.
lo = np.array([[-0.1, -0.1, -0.1], [0.04, -0.1, -0.1]])
hi = np.array([[-0.04, 0.1, 0.1], [0.1, 0.1, 0.1]])
assert (
    certify_extraction(np.array([0.11, 0, 0]), handle, h, lo, hi)
    == "centroid_outside_fixture_envelope"
)
assert certify_extraction(np.zeros(3), handle, h, lo, hi) == "straight_handle_corridor"
lo = np.vstack([lo, [-0.04, -0.04, 0.03], [-0.04, -0.04, -0.08]])
hi = np.vstack([hi, [0.04, 0.04, 0.08], [0.04, 0.04, -0.03]])
assert certify_extraction(np.zeros(3), handle, h, lo, hi) is None
print("verified outside, clear-corridor and unresolved-enclosure controls")
