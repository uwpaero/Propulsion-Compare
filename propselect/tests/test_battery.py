import pytest

from propselect.core.battery import (
    InternalResistanceBattery,
    MeasuredCurveBattery,
    OCVTable,
    soc_from_capacity_used,
)


def make_ocv_table() -> OCVTable:
    # Typical Li-ion per-cell OCV curve, SoC 0..1 -> ~3.3..4.2 V.
    soc = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    voltage = [3.30, 3.55, 3.70, 3.80, 3.95, 4.20]
    return OCVTable(soc, voltage)


def test_ocv_table_monotone_and_endpoints():
    table = make_ocv_table()
    assert table.voltage(0.0).voltage_v == pytest.approx(3.30, abs=1e-6)
    assert table.voltage(1.0).voltage_v == pytest.approx(4.20, abs=1e-6)
    # Monotonically increasing with SoC (PCHIP must not overshoot/dip).
    vals = [table.voltage(s).voltage_v for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_ocv_table_clamps_outside_range_and_warns():
    table = make_ocv_table()
    result = table.voltage(1.2)
    assert result.clamped
    assert result.warning is not None
    assert result.voltage_v == pytest.approx(4.20, abs=1e-6)

    result_low = table.voltage(-0.5)
    assert result_low.clamped
    assert result_low.voltage_v == pytest.approx(3.30, abs=1e-6)


def test_internal_resistance_scales_with_series_parallel():
    table = make_ocv_table()
    battery_6s2p = InternalResistanceBattery(
        ocv_table=table, series=6, parallel=2, r_internal_per_cell_ohm=0.01, capacity_ah=10.0
    )
    # R_pack = R_cell * S / P
    assert battery_6s2p.r_internal_ohm() == pytest.approx(0.01 * 6 / 2)

    result = battery_6s2p.terminal_voltage(current_a=20.0, soc=1.0)
    v_ocv_pack = 4.20 * 6
    expected = v_ocv_pack - 20.0 * battery_6s2p.r_internal_ohm()
    assert result.voltage_v == pytest.approx(expected)
    assert not result.clamped


def test_internal_resistance_voltage_drops_with_current():
    table = make_ocv_table()
    battery = InternalResistanceBattery(
        ocv_table=table, series=4, parallel=1, r_internal_per_cell_ohm=0.02, capacity_ah=5.0
    )
    v_light = battery.terminal_voltage(current_a=1.0, soc=0.8).voltage_v
    v_heavy = battery.terminal_voltage(current_a=30.0, soc=0.8).voltage_v
    assert v_heavy < v_light


def test_measured_curve_interpolates_and_clamps():
    battery = MeasuredCurveBattery(
        current_table_a=[0, 10, 20, 30, 40],
        voltage_table_v=[16.8, 16.2, 15.6, 14.8, 13.5],
        capacity_ah=5.0,
    )
    at_zero = battery.terminal_voltage(0.0)
    assert at_zero.voltage_v == pytest.approx(16.8)
    assert not at_zero.clamped

    beyond = battery.terminal_voltage(100.0)
    assert beyond.clamped
    assert beyond.warning is not None
    assert beyond.voltage_v == pytest.approx(13.5)

    # Interpolated midpoint should lie between the bracketing table values.
    mid = battery.terminal_voltage(15.0)
    assert 15.6 < mid.voltage_v < 16.2


def test_measured_curve_monotone_decreasing_with_current():
    battery = MeasuredCurveBattery(
        current_table_a=[0, 10, 20, 30, 40],
        voltage_table_v=[16.8, 16.2, 15.6, 14.8, 13.5],
        capacity_ah=5.0,
    )
    currents = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    voltages = [battery.terminal_voltage(i).voltage_v for i in currents]
    assert all(b <= a for a, b in zip(voltages, voltages[1:]))


def test_soc_from_capacity_used():
    assert soc_from_capacity_used(initial_soc=1.0, capacity_ah=5.0, ah_used=0.0) == pytest.approx(
        1.0
    )
    assert soc_from_capacity_used(
        initial_soc=1.0, capacity_ah=5.0, ah_used=2.5
    ) == pytest.approx(0.5)
