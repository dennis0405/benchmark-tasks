"""Independent energy/reciprocity checks; not a native-physics result."""

import math

import pytest
from thread_reference import ThreadParameters, response

PARAMETERS = ThreadParameters(
    lead_m_per_turn=0.00125,
    stiffness_n_per_m=1200.0,
    damping_ns_per_m=0.35,
    friction_nm=0.003,
    friction_velocity_rad_s=0.05,
    rest_offset_m=0.0,
)


def test_elastic_wrench_is_the_negative_energy_gradient():
    z, angle = 0.0003, 1.3
    result = response(PARAMETERS, z, angle, 0, 0)
    dz, da = 1e-8, 1e-6
    gradient_z = (
        response(PARAMETERS, z + dz, angle, 0, 0).energy_j
        - response(PARAMETERS, z - dz, angle, 0, 0).energy_j
    ) / (2 * dz)
    gradient_angle = (
        response(PARAMETERS, z, angle + da, 0, 0).energy_j
        - response(PARAMETERS, z, angle - da, 0, 0).energy_j
    ) / (2 * da)
    assert result.nut_axial_force_n == pytest.approx(-gradient_z, rel=1e-8)
    assert result.nut_axial_torque_nm == pytest.approx(-gradient_angle, rel=1e-8)


@pytest.mark.parametrize("speed", [-12, -0.2, 0.0, 0.2, 12])
def test_internal_power_equals_elastic_energy_loss_plus_dissipation(speed):
    # Exercise simultaneous translation/rotation; neither sign may generate
    # energy. These are relative velocities of actual bodies, not a goal rate.
    axial_speed = 0.0008 - 0.0002 * speed
    result = response(PARAMETERS, -0.0003, 2.7, axial_speed, speed)
    power = result.nut_axial_force_n * axial_speed + result.nut_axial_torque_nm * speed
    assert result.dissipation_w >= 0
    assert power + result.elastic_energy_rate_w == pytest.approx(
        -result.dissipation_w, abs=1e-12
    )
    assert result.screw_axial_force_n == -result.nut_axial_force_n
    assert result.screw_axial_torque_nm == -result.nut_axial_torque_nm


def test_one_turn_advances_one_lead_and_retains_no_elastic_preload():
    # An angle that wraps at 2*pi would lose a whole thread lead. The native
    # adapter therefore has to supply a measured, unwrapped relative angle.
    frictionless = ThreadParameters(0.00125, 1200, 0, 0, 0.05, 0)
    for turns in (-3, -1, 0, 1, 3):
        result = response(frictionless, -turns * 0.00125, turns * 2 * math.pi, 0, 0)
        assert result.energy_j == pytest.approx(0, abs=1e-24)
        assert result.nut_axial_force_n == pytest.approx(0, abs=1e-12)


def test_nonfinite_input_cannot_become_an_unobserved_native_force():
    with pytest.raises(ValueError, match="finite"):
        response(PARAMETERS, math.nan, 0, 0, 0)
    with pytest.raises(ValueError, match="positive"):
        ThreadParameters(0.00125, 1200, 0, 0.003, 0, 0)
