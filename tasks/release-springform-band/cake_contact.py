"""Contact metrics for a rigid proxy of the cake, in SI units unless labeled."""
from math import fabs

MAX_NORMAL_IMPULSE_NS = 0.025
MAX_PENETRATION_MM = 2.0


def touches_exposed_cake(paths):
    for cake, other in (paths, paths[::-1]):
        if '/pan_fixture/cake/' not in cake:
            continue
        # The plate supports the cake for the entire trial. Grippers, moving
        # metal, mats and all other colliders can damage exposed cake.
        return '/pan_fixture/base/' not in other and '/pan_fixture/cake/' not in other
    return False


def manifold_peaks(points, offset, count):
    normal_impulse = 0.0
    penetration_mm = 0.0
    for index in range(offset, offset + count):
        point = points[index]
        normal_impulse += fabs(sum(point.impulse[i] * point.normal[i] for i in range(3)))
        penetration_mm = max(penetration_mm, -1000.0 * point.separation)
    return normal_impulse, penetration_mm
