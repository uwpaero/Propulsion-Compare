import math

import pytest

from propselect.core.atmosphere import G0
from propselect.core.battery import InternalResistanceBattery, OCVTable
from propselect.core.motor import MotorSpec
from propselect.core.propeller import PropellerDataTable
from propselect.core.takeoff import (
    AircraftConfig,
    KinematicsResult,
    aerodynamic_drag_n,
    aerodynamic_lift_n,
    closed_form_thrust_estimate,
    effective_cd_ground,
    induced_drag_ground_effect_factor,
    integrate_ground_roll,
    integrate_kinematics,
    net_accelerating_force_n,
)


def test_aspect_ratio_is_derived_from_span_and_area_not_stored():
    # AR = b^2/S is the exact definition -- it must never be an independently
    # settable field that could drift out of sync with span/area.
    aircraft = AircraftConfig(
        mass_kg=8.0, wing_area_m2=0.5, span_m=2.0, cd0_ground=0.09, cl_ground=0.60, mu=0.10,
    )
    assert aircraft.aspect_ratio == pytest.approx(2.0**2 / 0.5)
    with pytest.raises(TypeError):
        AircraftConfig(
            mass_kg=8.0, wing_area_m2=0.5, span_m=2.0, aspect_ratio=6.0,
            cd0_ground=0.09, cl_ground=0.60, mu=0.10,
        )


def test_aspect_ratio_updates_if_span_or_area_differ():
    narrow = AircraftConfig(
        mass_kg=8.0, wing_area_m2=0.5, span_m=1.0, cd0_ground=0.09, cl_ground=0.60, mu=0.10,
    )
    wide = AircraftConfig(
        mass_kg=8.0, wing_area_m2=0.5, span_m=3.0, cd0_ground=0.09, cl_ground=0.60, mu=0.10,
    )
    assert wide.aspect_ratio > narrow.aspect_ratio


def test_hand_calc_closed_form_case():
    # From the build spec's hand-calc check case.
    estimate = closed_form_thrust_estimate(
        mass_kg=8.0,
        wing_area_m2=0.65,
        v_t_m_s=15.0,
        distance_allowed_m=30.5,
        cd=0.09,
        cl=0.60,
        mu=0.10,
        rho_kg_m3=1.16,
    )
    assert estimate.inertia_n == pytest.approx(29.5, abs=0.1)
    assert estimate.q_bar_pa == pytest.approx(65.3, abs=0.1)
    assert estimate.drag_n == pytest.approx(3.8, abs=0.1)
    assert estimate.lift_n == pytest.approx(25.5, abs=0.1)
    assert estimate.friction_n == pytest.approx(5.3, abs=0.1)
    assert estimate.thrust_required_n == pytest.approx(38.6, abs=0.2)
    assert estimate.inertia_pct == pytest.approx(76.0, abs=1.0)
    assert estimate.drag_pct == pytest.approx(10.0, abs=1.0)
    assert estimate.friction_pct == pytest.approx(14.0, abs=1.0)


def test_aerodynamic_drag_and_lift_formulas():
    assert aerodynamic_drag_n(1.16, 15.0, 0.65, 0.09) == pytest.approx(
        0.5 * 1.16 * 15.0**2 * 0.65 * 0.09
    )
    assert aerodynamic_lift_n(1.16, 15.0, 0.65, 0.60) == pytest.approx(
        0.5 * 1.16 * 15.0**2 * 0.65 * 0.60
    )


def test_net_accelerating_force_formula():
    f = net_accelerating_force_n(thrust_n=40.0, drag_n=4.0, lift_n=20.0, mu=0.1, mass_kg=8.0)
    expected = 40.0 - 4.0 - 0.1 * (8.0 * G0 - 20.0)
    assert f == pytest.approx(expected)


def test_ground_effect_factor_zero_at_high_wheel_and_bounded():
    phi_low = induced_drag_ground_effect_factor(wheel_height_m=0.05, span_m=2.0)
    phi_high = induced_drag_ground_effect_factor(wheel_height_m=1.0, span_m=2.0)
    assert 0.0 <= phi_low < phi_high < 1.0


def test_effective_cd_ground_off_by_default_matches_cd0():
    aircraft = AircraftConfig(
        mass_kg=8.0,
        wing_area_m2=0.65,
        span_m=2.0,
        cd0_ground=0.09,
        cl_ground=0.60,
        mu=0.10,
    )
    assert aircraft.ground_effect is False
    assert effective_cd_ground(aircraft) == pytest.approx(0.09)


def test_effective_cd_ground_with_ground_effect_adds_induced_term():
    aircraft = AircraftConfig(
        mass_kg=8.0,
        wing_area_m2=0.65,
        span_m=2.0,
        cd0_ground=0.09,
        cl_ground=0.60,
        mu=0.10,
        wheel_height_m=0.1,
        oswald_efficiency=0.8,
        ground_effect=True,
    )
    cd = effective_cd_ground(aircraft)
    assert cd > 0.09


# --- Analytical check: T constant, D=0, L=0, mu=0 -> s = m*Vt^2/(2T) ---


def test_analytical_takeoff_check_constant_thrust_no_drag_no_friction():
    mass_kg = 8.0
    thrust_n_const = 40.0
    v_t = 15.0
    dv = 0.01
    n_steps = round(v_t / dv)
    v_grid = [i * dv for i in range(n_steps + 1)]
    v_grid[-1] = v_t
    thrust_arr = [thrust_n_const] * len(v_grid)
    zeros = [0.0] * len(v_grid)

    result = integrate_kinematics(v_grid, thrust_arr, zeros, zeros, mass_kg=mass_kg, mu=0.0)
    assert isinstance(result, KinematicsResult)
    assert result.success

    expected_distance = mass_kg * v_t**2 / (2.0 * thrust_n_const)
    assert result.distance_m == pytest.approx(expected_distance, rel=1e-3)


def test_kinematics_reports_failure_when_thrust_insufficient():
    mass_kg = 8.0
    v_grid = [0.0, 1.0, 2.0, 3.0]
    # Thrust exactly equal to a constant drag-like deficit -> immediately non-positive.
    thrust_arr = [1.0, 1.0, 1.0, 1.0]
    drag_arr = [5.0, 5.0, 5.0, 5.0]
    zeros = [0.0] * len(v_grid)
    result = integrate_kinematics(v_grid, thrust_arr, drag_arr, zeros, mass_kg=mass_kg, mu=0.0)
    assert not result.success
    assert result.distance_m == math.inf
    assert result.stall_v_m_s == 0.0
    assert result.deficit_n == pytest.approx(4.0)


def test_kinematics_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError):
        integrate_kinematics([0.0, 1.0], [1.0], [0.0, 0.0], [0.0, 0.0], mass_kg=8.0, mu=0.0)


# --- Full coupled integration ---


def make_full_setup():
    soc = [0.0, 0.5, 1.0]
    voltage = [3.3, 3.7, 4.2]
    battery = InternalResistanceBattery(
        ocv_table=OCVTable(soc, voltage),
        series=6,
        parallel=2,
        r_internal_per_cell_ohm=0.006,
        capacity_ah=5.0,
    )
    motor = MotorSpec(name="test-motor", kv_rpm_per_v=900.0, r_motor_ohm=0.05, i0_a=0.8)
    j = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    ct = [0.11, 0.105, 0.09, 0.07, 0.045, 0.015, -0.02]
    cp = [0.045, 0.044, 0.040, 0.033, 0.024, 0.015, 0.008]
    prop = PropellerDataTable(j, ct, cp, diameter_m=0.28, pitch_m=0.16, name="11x6")
    aircraft = AircraftConfig(
        mass_kg=8.0,
        wing_area_m2=0.65,
        span_m=2.0,
        cd0_ground=0.09,
        cl_ground=0.60,
        mu=0.10,
    )
    return aircraft, prop, motor, battery


def test_full_ground_roll_integration_succeeds_and_tracks_soc():
    aircraft, prop, motor, battery = make_full_setup()
    result = integrate_ground_roll(
        v_t_m_s=15.0, rho_kg_m3=1.16, aircraft=aircraft, prop=prop, motor=motor, battery=battery
    )
    assert result.success
    assert result.distance_m > 0.0
    assert result.time_s > 0.0
    assert result.capacity_used_ah > 0.0
    # SoC should have decreased (not increased) over the roll.
    assert result.soc_profile[-1] < result.soc_profile[0]
    assert result.soc_profile[0] == pytest.approx(1.0)


def test_ground_roll_reports_structured_failure_for_underpowered_setup():
    aircraft, prop, motor, battery = make_full_setup()
    # An unreasonably heavy aircraft that this powertrain cannot accelerate.
    heavy_aircraft = AircraftConfig(
        mass_kg=800.0,
        wing_area_m2=aircraft.wing_area_m2,
        span_m=aircraft.span_m,
        cd0_ground=aircraft.cd0_ground,
        cl_ground=aircraft.cl_ground,
        mu=aircraft.mu,
    )
    result = integrate_ground_roll(
        v_t_m_s=15.0,
        rho_kg_m3=1.16,
        aircraft=heavy_aircraft,
        prop=prop,
        motor=motor,
        battery=battery,
    )
    assert not result.success
    assert result.distance_m == math.inf
    assert result.reason is not None
    assert result.stall_v_m_s is not None


def test_motor_count_scales_thrust_and_pack_current_draw():
    aircraft, prop, motor, battery = make_full_setup()

    single = integrate_ground_roll(
        v_t_m_s=15.0, rho_kg_m3=1.16, aircraft=aircraft, prop=prop, motor=motor,
        battery=battery, motor_count=1,
    )
    twin = integrate_ground_roll(
        v_t_m_s=15.0, rho_kg_m3=1.16, aircraft=aircraft, prop=prop, motor=motor,
        battery=battery, motor_count=2,
    )
    assert single.success
    assert twin.success

    # Two motors accelerate the same airframe faster -> shorter, quicker roll.
    assert twin.distance_m < single.distance_m
    assert twin.time_s < single.time_s

    # Capacity used is positive and finite either way -- but note it is NOT
    # guaranteed to be larger for the twin-motor case: doubling thrust
    # shortens the roll so much that the time-integral of (higher) pack
    # current can still come out lower than the single-motor roll's.
    assert 0.0 < twin.capacity_used_ah < math.inf
    assert 0.0 < single.capacity_used_ah < math.inf

    single_at_vt = single.operating_points[-1]
    twin_at_vt = twin.operating_points[-1]
    # Pack current at any shared operating point must be motor_count x the
    # per-motor current, exactly.
    assert twin_at_vt.current_pack_a == pytest.approx(2.0 * twin_at_vt.current_a, rel=1e-9)
    assert single_at_vt.current_pack_a == pytest.approx(single_at_vt.current_a, rel=1e-9)
