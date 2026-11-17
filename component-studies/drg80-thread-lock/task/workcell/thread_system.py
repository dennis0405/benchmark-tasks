"""Implicit prediction of internal forces; physical contacts are solved by PhysX."""

import numpy as np


def unilateral(hessian, rhs, rows, lower, locks=()):
    # Solve the small convex velocity problem with unilateral contacts and optional
    # sticking axes. Drop any separating reaction; enforce only a closing contact.
    free = np.linalg.solve(hessian, rhs)
    active = []
    for _ in range(24):
        axes = [*locks, *(rows[i] for i in active)]
        if axes:
            a = np.array(axes)
            inv_at = np.linalg.solve(hessian, a.T)
            b = np.r_[np.zeros(len(locks)), lower[active]]
            reaction = np.linalg.lstsq(a @ inv_at, b - a @ free, rcond=1e-10)[0]
            v = free + inv_at @ reaction
            neg = reaction[len(locks) :]
            if len(neg) and min(neg) < -1e-7:
                active.pop(int(np.argmin(neg)))
                continue
        else:
            v = free
            reaction = np.array([])
        violation = rows @ v - lower
        if len(active):
            violation[active] = 0.0
        if min(violation) >= -1e-8:
            return v, reaction[: len(locks)]
        active.append(int(np.argmin(violation)))
    raise RuntimeError("fixture contact active set did not converge")


def solve_internal(
    mass,
    velocity,
    external,
    gradients,
    extension,
    stiffness,
    damping,
    rows,
    gaps,
    h,
    angular_axes,
):
    b = damping + h * stiffness
    hessian = mass / h + gradients.T @ (b[:, None] * gradients)
    rhs = mass @ velocity / h + external - gradients.T @ (stiffness * extension)
    # Native unilateral contacts own penetration recovery. Predicting a full
    # overlap correction again here generates a fictitious velocity and can push
    # a damped thread deeper into the seat on the following physics step.
    lower = -np.maximum(gaps, 0.0) / h
    # The friction capacity comes from measured extension/rate at the start
    # of the interval. A predicted next-step damping force can create a
    # sticking torque while the physical nut still has zero stored preload.
    measured = stiffness * extension + damping * (gradients @ velocity)
    static = np.abs(measured[:2]) * 0.0035 * np.array([0.08, 0.35])
    sliding = np.abs(measured[:2]) * 0.0035 * np.array([0.064, 0.28])
    from itertools import product

    for modes in product((0, -1, 1), repeat=len(angular_axes)):
        locks = [i for i, s in enumerate(modes) if s == 0]
        torque = -np.array(modes) * sliding
        try:
            v, r = unilateral(
                hessian,
                rhs + angular_axes.T @ torque,
                rows,
                lower,
                [angular_axes[i] for i in locks],
            )
        except RuntimeError:
            continue
        if any(abs(x) > static[i] + 1e-9 for i, x in zip(locks, r)):
            continue
        if any(m * (a @ v) < -1e-7 for m, a in zip(modes, angular_axes)):
            continue
        torque[locks] = r
        load = stiffness * extension + b * (gradients @ v)
        return -gradients.T @ load + angular_axes.T @ torque, load, torque
    raise RuntimeError("no admissible passive thread friction mode")
