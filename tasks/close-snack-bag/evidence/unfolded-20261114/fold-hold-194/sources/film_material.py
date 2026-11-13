"""Uncalibrated isotropic equivalent for a laminated snack film.

Thickness controls mass, membrane stiffness and bending separately. These
relations are a starting hypothesis, not measured package properties.
"""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class FilmMaterial:
    thickness: float = 80e-6
    density: float = 1000.0
    young: float = 1e8
    poisson: float = 0.35
    yield_curvature: float = 100.0
    recovery_per_second: float = 0.1

    def coefficients(self, dual_width):
        if not all(math.isfinite(v) for v in (*asdict(self).values(), dual_width)):
            raise ValueError("finite material parameters required")
        if min(self.thickness, self.density, self.young, dual_width) <= 0:
            raise ValueError("positive geometry, density and modulus required")
        if not 0 <= self.poisson < 0.5 or min(self.yield_curvature, self.recovery_per_second) < 0:
            raise ValueError("invalid Poisson ratio, yield or recovery")
        h, e, nu = self.thickness, self.young, self.poisson
        # Plane-stress membrane and thin isotropic plate theory.
        mu = e * h / (2 * (1 + nu))
        lam = e * h * nu / (1 - nu * nu)
        bending = e * h**3 / (12 * (1 - nu * nu))
        # Newton 1.7 bending energy is 0.5 * edge_ke * edge_length * delta**2.
        # Matching a strip of dual width w gives edge_ke ~= D/w. A uniform
        # width is an approximation; seams and irregular triangles need fitting.
        return dict(areal_density=self.density*h, tri_ke=mu, tri_ka=lam,
                    edge_ke=bending/dual_width, bending_rigidity=bending,
                    yield_angle=min(math.pi*0.9, self.yield_curvature*dual_width),
                    recovery=self.recovery_per_second)
