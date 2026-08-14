"""Known-good regression fixture: a real APC propeller + a real motor spec,
run through the full coupled solver and ground-roll integration. These
numbers were recorded from one known-good run of the current implementation;
if the physics core changes and this test breaks, that's the point -- verify
the new numbers by hand before updating the fixture.
"""

from pathlib import Path

import pytest

from propselect.core.atmosphere import isa_offset_atmosphere
from propselect.core.battery import InternalResistanceBattery, OCVTable
from propselect.core.takeoff import AircraftConfig, integrate_ground_roll
from propselect.data.loaders import load_apc_dat_file, load_motor_library

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _build_regression_scenario():
    motors = load_motor_library(DATA_DIR / "motors.json")
    motor = next(m for m in motors if m.name == "AXI 2820/10")
    prop = load_apc_dat_file(DATA_DIR / "props" / "PER3_10x7.dat", expected_rpm=8000).prop

    soc = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    voltage = [3.30, 3.55, 3.70, 3.80, 3.95, 4.20]
    battery = InternalResistanceBattery(
        ocv_table=OCVTable(soc, voltage),
        series=4,
        parallel=3,
        r_internal_per_cell_ohm=0.006,
        capacity_ah=3.3,
    )

    aircraft = AircraftConfig(
        mass_kg=1.6,
        wing_area_m2=0.30,
        span_m=1.4,
        cd0_ground=0.05,
        cl_ground=0.45,
        mu=0.06,
    )

    atm = isa_offset_atmosphere(0.0)
    return motor, prop, battery, aircraft, atm


def test_regression_apc_10x7_axi_2820_10_ground_roll():
    motor, prop, battery, aircraft, atm = _build_regression_scenario()

    result = integrate_ground_roll(
        v_t_m_s=9.0,
        rho_kg_m3=atm.density,
        aircraft=aircraft,
        prop=prop,
        motor=motor,
        battery=battery,
        r_esc_ohm=0.01,
        dv_m_s=0.1,
        initial_soc=1.0,
    )

    assert result.success

    # Recorded from a known-good run of this implementation.
    assert result.distance_m == pytest.approx(2.2513, rel=1e-3)
    assert result.time_s == pytest.approx(0.4893, rel=1e-3)
    assert result.capacity_used_ah == pytest.approx(0.008944, rel=2e-2)
    assert result.soc_profile[-1] == pytest.approx(0.99729, rel=1e-3)

    static = result.operating_points[0]
    assert static.thrust_n == pytest.approx(32.254, rel=1e-3)
    assert static.current_a == pytest.approx(64.146, rel=1e-3)
    assert static.power_shaft_w == pytest.approx(778.55, rel=1e-3)

    at_vt = result.operating_points[-1]
    assert at_vt.n_rev_s == pytest.approx(221.418, rel=1e-3)
    assert at_vt.thrust_n == pytest.approx(28.660, rel=1e-3)
    assert at_vt.current_a == pytest.approx(67.235, rel=1e-3)
    assert at_vt.voltage_v == pytest.approx(16.246, rel=1e-3)
    assert at_vt.eta_motor == pytest.approx(0.73346, rel=1e-3)
    assert at_vt.eta_prop == pytest.approx(0.32197, rel=1e-3)


def test_regression_scenario_is_deterministic_across_runs():
    # Running the exact same scenario twice must reproduce bit-for-bit-close
    # results -- there is no randomness anywhere in the physics core.
    motor, prop, battery, aircraft, atm = _build_regression_scenario()

    def run():
        return integrate_ground_roll(
            v_t_m_s=9.0,
            rho_kg_m3=atm.density,
            aircraft=aircraft,
            prop=prop,
            motor=motor,
            battery=battery,
            r_esc_ohm=0.01,
            dv_m_s=0.1,
            initial_soc=1.0,
        )

    first = run()
    second = run()
    assert first.distance_m == pytest.approx(second.distance_m, rel=1e-9)
    assert first.time_s == pytest.approx(second.time_s, rel=1e-9)
