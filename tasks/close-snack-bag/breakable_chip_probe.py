"""Newton VBD tensile test for a pre-fragmented, breakable chip surrogate.

Two tetrahedral halves are connected by springs across a pre-defined seam.
The spring break threshold is experimental, not measured snack fracture data.
This is a tensile fixture, not a robot, pouch, or benchmark scoring run.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
import newton


def run(force, duration, out, direction="x"):
    wp.init()
    builder = newton.ModelBuilder(gravity=wp.vec3(0, 0, 0))
    # Grid origins are their lower corners. Six bridge springs connect
    # matching nodes across the 0.5 mm artificial fracture seam.
    for x in (0.0, 0.0145):
        builder.add_soft_grid(pos=wp.vec3(x, 0, 0), rot=wp.quat_identity(), vel=wp.vec3(0, 0, 0),
            dim_x=2, dim_y=2, dim_z=1, cell_x=.007, cell_y=.012, cell_z=.0025,
            density=300, k_mu=1.e6, k_lambda=1.e6, k_damp=10,
            fix_left=x == 0, particle_radius=.0001)
    points = np.asarray(builder.particle_q)
    left = np.flatnonzero(np.isclose(points[:, 0], .014, atol=1e-7))
    right = np.flatnonzero(np.isclose(points[:, 0], .0145, atol=1e-7))
    for i in left:
        distances = np.linalg.norm(points[right, 1:] - points[i, 1:], axis=1)
        j = int(right[np.argmin(distances)])
        if distances.min() > 1e-6:
            raise RuntimeError('no paired seam vertex')
        builder.add_spring(int(i), j, ke=20.0, kd=.002, control=0)
    pull_nodes = np.flatnonzero(np.isclose(points[:, 0], .0285, atol=1e-7)).astype(np.int32)
    if len(left) != 6 or len(right) != 6 or len(pull_nodes) != 6:
        raise RuntimeError('unexpected grid topology')
    builder.color()
    model = builder.finalize(device='cuda:0')
    solver = newton.solvers.SolverVBD(model=model, iterations=30, particle_enable_self_contact=False,
                                      particle_enable_tile_solve=False)
    state, next_state, control = model.state(), model.state(), model.control()
    indices = model.spring_indices.numpy().reshape(-1, 2)
    rest = model.spring_rest_length.numpy()
    baseline_stiffness = model.spring_stiffness.numpy()
    baseline_damping = model.spring_damping.numpy()
    dt = 1/1920
    break_extension = .00015  # 0.15 mm, provisional surrogate calibration
    broken, peak, trace = False, 0.0, []
    for step in range(round(duration / dt)):
        state.clear_forces()
        applied = np.zeros_like(points, dtype=np.float32)
        applied[pull_nodes, {"x": 0, "z": 2}[direction]] = (0.0 if broken else force) / len(pull_nodes)
        wp.copy(state.particle_f, wp.array(applied, dtype=wp.vec3, device=model.device))
        solver.step(state, next_state, control, None, dt)
        state, next_state = next_state, state
        p = state.particle_q.numpy()
        extension = np.linalg.norm(p[indices[:, 1]] - p[indices[:, 0]], axis=1) - rest
        peak = max(peak, float(np.max(extension)))
        if not broken and peak >= break_extension:
            broken = True
            wp.copy(model.spring_stiffness, wp.array(np.zeros_like(baseline_stiffness), dtype=float, device=model.device))
            wp.copy(model.spring_damping, wp.array(np.zeros_like(baseline_damping), dtype=float, device=model.device))
        if step in (0, round(duration / dt)-1) or (broken and not any(r.get('break') for r in trace)):
            trace.append({'time_s': (step+1)*dt, 'max_extension_m': float(np.max(extension)),
                          'mean_gap_m': float(np.mean(np.linalg.norm(p[indices[:, 1]] - p[indices[:, 0]], axis=1))),
                          'broken': broken, 'break': broken})
    result = {'force_n':force, 'direction':direction, 'duration_s':duration, 'seam_springs':len(indices),
              'break_extension_m':break_extension, 'broken':broken, 'peak_extension_m':peak,
              'final_gap_m': trace[-1]['mean_gap_m'], 'finite': bool(np.isfinite(p).all()), 'trace': trace}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--force', type=float, default=0)
    parser.add_argument('--duration', type=float, default=.2)
    parser.add_argument('--direction', choices=('x', 'z'), default='x')
    args = parser.parse_args()
    if args.force < 0 or args.duration <= 0:
        parser.error('force must be nonnegative and duration must be positive')
    run(args.force, args.duration, args.output, args.direction)
