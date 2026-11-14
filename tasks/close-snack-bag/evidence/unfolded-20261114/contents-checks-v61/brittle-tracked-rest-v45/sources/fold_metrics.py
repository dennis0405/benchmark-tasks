"""Measure actual pouch folds from material landmarks; no prescribed poses."""
import numpy as np


def fold_metrics(points, grid, fold_rows):
    points, grid = np.asarray(points), np.asarray(grid)
    if not np.isfinite(points).all():
        return {"valid": False, "folds": []}
    nx, ny = grid.shape[1]-1, grid.shape[2]-1
    band = slice(max(1, ny//8), min(ny, ny-ny//8)+1)
    reports = []
    for pivot, body, flap in fold_rows:
        if not 0 <= body < pivot < flap <= nx:
            raise ValueError("landmarks must run from bag body to mouth")
        a = points[grid[:, pivot, band]]-points[grid[:, body, band]]
        b = points[grid[:, flap, band]]-points[grid[:, pivot, band]]
        norm = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
        valid = norm > 1e-10
        angles = np.arccos(np.clip(np.sum(a*b, axis=-1)/np.maximum(norm, 1e-10), -1, 1))
        reports.append(dict(median_angle_rad=float(np.median(angles)),
                            folded_fraction=float(np.mean(valid & (angles >= np.deg2rad(120)))),
                            valid=bool(valid.all())))
    gap = np.linalg.norm(points[grid[0, -1, band]]-points[grid[1, -1, band]], axis=-1)
    return dict(valid=all(r["valid"] for r in reports), folds=reports,
                mouth_layer_gap_p90_m=float(np.quantile(gap, .9)))
