import pytest

from propselect.core.battery import InternalResistanceBattery, OCVTable
from propselect.core.motor import MotorSpec
from propselect.core.operating_point import (
    OperatingPoint,
    OperatingPointFailure,
    solve_operating_point,
    sweep_operating_points,
)
from propselect.core.propeller import PropellerDataTable


def make_battery() -> InternalResistanceBattery:
    soc = [0.0, 0.5, 1.0]
    voltage = [3.3, 3.7, 4.2]
    table = OCVTable(soc, voltage)
    # 6S1P, modest internal resistance, generous capacity.
    return InternalResistanceBattery(
        ocv_table=table, series=6, parallel=1, r_internal_per_cell_ohm=0.006, capacity_ah=5.0
    )


def make_motor() -> MotorSpec:
    return MotorSpec(name="test-motor", kv_rpm_per_v=900.0, r_motor_ohm=0.05, i0_a=0.8)


def make_prop() -> PropellerDataTable:
    j = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    ct = [0.11, 0.105, 0.09, 0.07, 0.045, 0.015, -0.02]
    cp = [0.045, 0.044, 0.040, 0.033, 0.024, 0.015, 0.008]
    return PropellerDataTable(j, ct, cp, diameter_m=0.254, pitch_m=0.1524, name="11x6")


def test_static_operating_point_converges_with_positive_thrust():
    result = solve_operating_point(
        v_m_s=0.0,
        rho_kg_m3=1.225,
        prop=make_prop(),
        motor=make_motor(),
        battery=make_battery(),
    )
    assert isinstance(result, OperatingPoint)
    assert result.success
    assert result.thrust_n > 0.0
    assert result.current_a > 0.0
    assert result.n_rev_s > 0.0
    # Static: J should be ~0.
    assert result.j == pytest.approx(0.0, abs=1e-6)
    # No airspeed -> zero propulsive efficiency by definition.
    assert result.eta_prop == 0.0


def test_operating_point_thrust_decreases_with_airspeed():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    static = solve_operating_point(0.0, 1.225, prop, motor, battery)
    cruise = solve_operating_point(15.0, 1.225, prop, motor, battery)
    assert isinstance(static, OperatingPoint)
    assert isinstance(cruise, OperatingPoint)
    assert cruise.thrust_n < static.thrust_n
    # RPM should increase with airspeed as the prop unloads.
    assert cruise.n_rev_s > static.n_rev_s


def test_motor_and_propulsive_efficiency_are_bounded():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    result = solve_operating_point(12.0, 1.225, prop, motor, battery)
    assert isinstance(result, OperatingPoint)
    assert 0.0 <= result.eta_motor <= 1.0
    assert 0.0 <= result.eta_prop <= 1.0


def test_electrical_power_balance_consistency():
    # P_elec = V*I must equal P_shaft / eta_motor within numerical tolerance,
    # since eta_motor = Q*omega/(V*I) = P_shaft/P_elec.
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    result = solve_operating_point(10.0, 1.225, prop, motor, battery)
    assert isinstance(result, OperatingPoint)
    if result.eta_motor > 0:
        implied_p_shaft = result.power_elec_w * result.eta_motor
        assert implied_p_shaft == pytest.approx(result.power_shaft_w, rel=1e-6)


def test_infeasible_combination_returns_structured_failure_not_exception():
    # Absurdly large ESC resistance makes any nonzero current produce a huge
    # voltage drop -- no feasible operating point exists.
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    result = solve_operating_point(10.0, 1.225, prop, motor, battery, r_esc_ohm=1.0e6)
    assert isinstance(result, OperatingPointFailure)
    assert not result.success
    assert isinstance(result.reason, str) and len(result.reason) > 0


def test_motor_count_scales_thrust_and_shaft_power_but_not_per_motor_current():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    single = solve_operating_point(10.0, 1.225, prop, motor, battery, motor_count=1)
    twin = solve_operating_point(10.0, 1.225, prop, motor, battery, motor_count=2)
    assert isinstance(single, OperatingPoint)
    assert isinstance(twin, OperatingPoint)

    # Each motor is identical and symmetric, so both solve for the same bus
    # voltage. Two motors produce strictly more total thrust than one, but
    # LESS than a naive 2x, because the doubled pack current sags the
    # battery voltage further and derates every motor's own operating point.
    assert twin.thrust_n > single.thrust_n
    assert twin.thrust_n < 2.0 * single.thrust_n
    assert twin.power_shaft_w > single.power_shaft_w
    assert twin.power_shaft_w < 2.0 * single.power_shaft_w
    assert twin.current_pack_a == pytest.approx(2.0 * twin.current_a, rel=1e-9)
    # Per-motor current drops relative to the single-motor case (more sag,
    # lower bus voltage, lower per-motor RPM/torque) -- it does not double.
    assert twin.current_a < single.current_a
    assert twin.voltage_v < single.voltage_v


def test_motor_count_pack_current_drives_more_voltage_sag():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    single = solve_operating_point(10.0, 1.225, prop, motor, battery, motor_count=1)
    quad = solve_operating_point(10.0, 1.225, prop, motor, battery, motor_count=4)
    assert isinstance(single, OperatingPoint)
    assert isinstance(quad, OperatingPoint)
    # A pack asked for ~4x the current sags to a lower bus voltage.
    assert quad.voltage_v < single.voltage_v
    assert quad.current_pack_a == pytest.approx(4.0 * quad.current_a, rel=1e-9)


def test_motor_count_one_is_identical_to_default_behavior():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    default = solve_operating_point(10.0, 1.225, prop, motor, battery)
    explicit = solve_operating_point(10.0, 1.225, prop, motor, battery, motor_count=1)
    assert isinstance(default, OperatingPoint)
    assert isinstance(explicit, OperatingPoint)
    assert default.thrust_n == pytest.approx(explicit.thrust_n)
    assert default.current_a == pytest.approx(explicit.current_a)
    assert explicit.current_pack_a == pytest.approx(explicit.current_a)


def test_motor_count_zero_or_negative_is_structured_failure():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    result = solve_operating_point(10.0, 1.225, prop, motor, battery, motor_count=0)
    assert isinstance(result, OperatingPointFailure)
    assert not result.success


def test_sweep_operating_points_length_and_soc_validation():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    v_values = [0.0, 5.0, 10.0, 15.0]
    results = sweep_operating_points(v_values, 1.225, prop, motor, battery)
    assert len(results) == len(v_values)
    assert all(isinstance(r, (OperatingPoint, OperatingPointFailure)) for r in results)

    with pytest.raises(ValueError):
        sweep_operating_points(v_values, 1.225, prop, motor, battery, soc_values=[1.0, 0.9])
