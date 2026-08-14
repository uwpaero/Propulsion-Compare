import math

import pytest

from propselect.core import atmosphere as atm


def test_sea_level_isa_reference_values():
    state = atm.isa_offset_atmosphere(0.0)
    assert state.temperature == pytest.approx(288.15, abs=1e-9)
    assert state.pressure == pytest.approx(101325.0, abs=1e-6)
    # Standard sea-level density is 1.225 kg/m^3.
    assert state.density == pytest.approx(1.225, abs=2e-4)
    assert state.speed_of_sound == pytest.approx(340.29, abs=0.05)


def test_density_decreases_with_altitude():
    sea_level = atm.isa_offset_atmosphere(0.0)
    high = atm.isa_offset_atmosphere(2000.0)
    assert high.density < sea_level.density
    assert high.pressure < sea_level.pressure
    assert high.temperature < sea_level.temperature


def test_dISA_offset_raises_temperature_and_lowers_density():
    baseline = atm.isa_offset_atmosphere(500.0)
    hot_day = atm.isa_offset_atmosphere(500.0, dISA_k=20.0)
    assert hot_day.temperature == pytest.approx(baseline.temperature + 20.0)
    assert hot_day.pressure == pytest.approx(baseline.pressure)
    assert hot_day.density < baseline.density


def test_actual_temperature_overrides_isa_profile():
    state = atm.isa_offset_atmosphere(500.0, actual_temperature_k=310.0)
    assert state.temperature == 310.0
    expected_rho = state.pressure / (atm.R_AIR * 310.0)
    assert state.density == pytest.approx(expected_rho)


def test_direct_atmosphere_ideal_gas_consistency():
    state = atm.direct_atmosphere(density_kg_m3=1.16, temperature_k=300.0)
    assert state.density == 1.16
    assert state.temperature == 300.0
    # rho = P / (R*T)  =>  P = rho*R*T
    assert state.pressure == pytest.approx(1.16 * atm.R_AIR * 300.0)
    assert state.density == pytest.approx(state.pressure / (atm.R_AIR * state.temperature))


def test_speed_of_sound_unit_consistency():
    # a = sqrt(gamma*R*T); at 288.15 K this must be ~340.3 m/s.
    a = atm.speed_of_sound(288.15)
    assert a == pytest.approx(math.sqrt(1.4 * 287.05 * 288.15))
    assert 330 < a < 350


def test_celsius_to_kelvin():
    assert atm.celsius_to_kelvin(0.0) == pytest.approx(273.15)
    assert atm.celsius_to_kelvin(15.0) == pytest.approx(288.15)
