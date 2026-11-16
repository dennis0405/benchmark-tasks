"""Analytic helical-coupling reference; NO native adapter or solver validation.

SI units throughout. This deliberately does not advance states, set poses,
detect a grasp or decide whether a task is locked. A native implementation may
use another conservative coupling; it must verify its measured response.
Explicit integration of this spring can be unstable for small tool inertias.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadParameters:
    lead_m_per_turn: float
    stiffness_n_per_m: float
    damping_ns_per_m: float
    friction_nm: float
    friction_velocity_rad_s: float
    rest_offset_m: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in self.__dict__.values()):
            raise ValueError("Thread parameters must be finite")
        if self.lead_m_per_turn == 0:
            raise ValueError("Thread lead must be nonzero; its sign sets handedness")
        if self.stiffness_n_per_m <= 0 or self.friction_velocity_rad_s <= 0:
            raise ValueError("Stiffness and friction velocity scale must be positive")
        if self.damping_ns_per_m < 0 or self.friction_nm < 0:
            raise ValueError("Dissipative coefficients must be nonnegative")


@dataclass(frozen=True)
class Response:
    extension_m: float
    energy_j: float
    elastic_energy_rate_w: float
    dissipation_w: float
    nut_axial_force_n: float
    nut_axial_torque_nm: float
    screw_axial_force_n: float
    screw_axial_torque_nm: float


def response(parameters, relative_z, relative_angle, relative_z_rate, relative_angle_rate):
    """Return an internal wrench pair in a shared measured coaxial frame.

    Angles are UNWRAPPED radians. Positive rotation advances the moving nut
    along negative Z for a positive lead. With Z pointing upward and positive
    angles counterclockwise from above, an ordinary right-handed thread thus
    uses a NEGATIVE lead here (clockwise rotation advances downward). This is
    a signed coordinate convention, not a claim about a manufacturer's thread.
    Translation is nut minus screw;
    velocities are also relative. In a native adapter, transform both wrenches
    correctly, apply them at the common thread axis and account for moments
    about each body's COM. Bearing/cover friction is native contact, not here.
    """
    if not all(
        math.isfinite(value)
        for value in (relative_z, relative_angle, relative_z_rate, relative_angle_rate)
    ):
        raise ValueError("Measured thread state must be finite")
    lead = parameters.lead_m_per_turn / (2 * math.pi)
    extension = relative_z + lead * relative_angle - parameters.rest_offset_m
    extension_rate = relative_z_rate + lead * relative_angle_rate
    elastic_force = -parameters.stiffness_n_per_m * extension
    axial_force = elastic_force - parameters.damping_ns_per_m * extension_rate
    friction_torque = -parameters.friction_nm * math.tanh(
        relative_angle_rate / parameters.friction_velocity_rad_s
    )
    torque = lead * axial_force + friction_torque
    return Response(
        extension_m=extension,
        energy_j=0.5 * parameters.stiffness_n_per_m * extension**2,
        elastic_energy_rate_w=-elastic_force * extension_rate,
        dissipation_w=(
            parameters.damping_ns_per_m * extension_rate**2
            - friction_torque * relative_angle_rate
        ),
        nut_axial_force_n=axial_force,
        nut_axial_torque_nm=torque,
        screw_axial_force_n=-axial_force,
        screw_axial_torque_nm=-torque,
    )
