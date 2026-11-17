"""One backward-Euler step of a passive screw thread between two movable bodies.

The caller passes a native relative state, generalized free velocities and effective
masses. Bounds, when available, are relative translations reachable at the end of
the step from actual colliders. They only predict contact for the force solve.
The simulator receives forces and remains responsible for positions and contacts.
"""
from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class Thread:
    lead: float
    stiffness: float
    damping: float

    def effort(self, extension, free_linear, free_angular, inv_mass, inv_inertia, dt,
               min_velocity=-inf, max_velocity=inf):
        p = self.lead
        impedance = self.damping + dt*self.stiffness
        numerator = self.stiffness*extension + impedance*(free_linear-p*free_angular)
        q = numerator/(1 + impedance*dt*(inv_mass+p*p*inv_inertia))
        linear = free_linear - dt*inv_mass*q
        if min_velocity <= linear <= max_velocity:
            return q
        linear = min(max_velocity, max(min_velocity, linear))
        return (self.stiffness*extension + impedance*(linear-p*free_angular)) / (
            1+impedance*dt*p*p*inv_inertia)
