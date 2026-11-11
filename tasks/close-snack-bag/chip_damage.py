"""Experimental volumetric strain indicator, not a real fracture model."""
import numpy as np


def reference_inverse(points, tetrahedra):
    t = np.asarray(points)[np.asarray(tetrahedra)]
    edges = np.stack((t[:, 1]-t[:, 0], t[:, 2]-t[:, 0], t[:, 3]-t[:, 0]), axis=-1)
    if np.any(np.abs(np.linalg.det(edges)) < 1e-15):
        raise ValueError('degenerate reference tetrahedron')
    return np.linalg.inv(edges)


def max_strain(points, tetrahedra, reference_inv):
    t = np.asarray(points)[np.asarray(tetrahedra)]
    edges = np.stack((t[:, 1]-t[:, 0], t[:, 2]-t[:, 0], t[:, 3]-t[:, 0]), axis=-1)
    f = edges @ reference_inv
    singular_values = np.linalg.svd(f, compute_uv=False)
    return float(np.max(np.abs(singular_values - 1)))
