import os

# Headless-safe: this test creates real QWidget instances, which requires a
# Qt platform plugin. Default to offscreen so this file runs without a
# display server, without forcing that choice on any other test.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from propselect.gui.units import (
    AREA,
    LENGTH_LARGE,
    LENGTH_SMALL,
    MASS,
    TEMPERATURE,
    VELOCITY,
    UnitAwareSpinBox,
    UnitSystem,
)

_app = QApplication.instance() or QApplication([])


def test_mass_conversion_reference_points():
    assert MASS.to_us(1.0) == pytest.approx(2.20462, rel=1e-4)
    assert MASS.to_si(1.0) == pytest.approx(0.45359237, rel=1e-6)
    assert MASS.to_si(MASS.to_us(8.0)) == pytest.approx(8.0, rel=1e-9)


def test_length_large_conversion_reference_points():
    assert LENGTH_LARGE.to_us(1.0) == pytest.approx(3.28084, rel=1e-4)
    assert LENGTH_LARGE.to_si(1.0) == pytest.approx(0.3048, rel=1e-6)


def test_length_small_conversion_reference_points():
    assert LENGTH_SMALL.to_si(1.0) == pytest.approx(0.0254, rel=1e-6)
    assert LENGTH_SMALL.to_us(0.0254) == pytest.approx(1.0, rel=1e-6)


def test_area_conversion_reference_points():
    assert AREA.to_si(1.0) == pytest.approx(0.3048**2, rel=1e-6)
    assert AREA.to_us(AREA.to_si(10.0)) == pytest.approx(10.0, rel=1e-9)


def test_velocity_conversion_reference_points():
    assert VELOCITY.to_si(1.0) == pytest.approx(0.44704, rel=1e-6)
    # A common general-aviation reference: 100 mph ~= 44.7 m/s.
    assert VELOCITY.to_si(100.0) == pytest.approx(44.704, rel=1e-4)


def test_temperature_conversion_reference_points():
    assert TEMPERATURE.to_us(0.0) == pytest.approx(32.0)
    assert TEMPERATURE.to_us(100.0) == pytest.approx(212.0)
    assert TEMPERATURE.to_si(32.0) == pytest.approx(0.0)
    assert TEMPERATURE.to_si(212.0) == pytest.approx(100.0)
    assert TEMPERATURE.to_si(TEMPERATURE.to_us(15.0)) == pytest.approx(15.0, rel=1e-9)


def test_unit_aware_spinbox_defaults_to_si():
    box = UnitAwareSpinBox(MASS, si_minimum=0.0, si_maximum=100.0, si_decimals=3, si_step=0.1)
    box.set_si_value(8.0)
    assert box.value() == pytest.approx(8.0)
    assert box.suffix().strip() == "kg"


def test_toggle_to_us_converts_displayed_value_not_overwrites():
    box = UnitAwareSpinBox(MASS, si_minimum=0.0, si_maximum=100.0, si_decimals=3, si_step=0.1)
    box.set_si_value(8.0)
    box.set_unit_system(UnitSystem.US)
    # 8 kg -> ~17.6 lb, not reset to 0 or some default.
    assert box.value() == pytest.approx(17.6370, rel=1e-3)
    assert box.suffix().strip() == "lb"
    # The canonical SI value is preserved through the round trip, up to the
    # display widget's own decimal-place rounding (a spinbox can only ever
    # store what it displays -- this is not the "overwrite" being guarded
    # against, just normal UI rounding to si_decimals=3 significant digits).
    assert box.si_value() == pytest.approx(8.0, abs=1e-4)


def test_toggle_round_trip_preserves_si_value_and_user_edit():
    box = UnitAwareSpinBox(LENGTH_LARGE, si_minimum=0.0, si_maximum=1000.0, si_decimals=2, si_step=0.5)
    box.set_si_value(30.5)  # e.g. a takeoff distance in meters
    box.set_unit_system(UnitSystem.US)
    assert box.value() == pytest.approx(30.5 / 0.3048, abs=0.01)

    # Simulate the user editing the value while displayed in US units.
    box.setValue(120.0)  # 120 ft
    assert box.si_value() == pytest.approx(120.0 * 0.3048, rel=1e-6)

    # Toggling back to SI must show the user's edited value converted, not
    # the original 30.5 m -- the edit is not lost or overwritten.
    box.set_unit_system(UnitSystem.SI)
    assert box.value() == pytest.approx(120.0 * 0.3048, abs=0.01)
    assert box.suffix().strip() == "m"


def test_toggle_is_a_no_op_when_system_unchanged():
    box = UnitAwareSpinBox(VELOCITY, si_minimum=0.0, si_maximum=100.0, si_decimals=2, si_step=0.5)
    box.set_si_value(15.0)
    box.set_unit_system(UnitSystem.SI)
    assert box.value() == pytest.approx(15.0)


def test_temperature_toggle_handles_affine_offset_correctly():
    box = UnitAwareSpinBox(TEMPERATURE, si_minimum=-40.0, si_maximum=60.0, si_decimals=1, si_step=1.0)
    box.set_si_value(20.0)  # 20 degC
    box.set_unit_system(UnitSystem.US)
    assert box.value() == pytest.approx(68.0, rel=1e-3)  # 20 degC = 68 degF
    assert box.si_value() == pytest.approx(20.0, rel=1e-6)
    # Step size (a delta) must not carry the +32 offset.
    assert box.singleStep() == pytest.approx(1.8, rel=1e-3)


def test_range_flips_correctly_after_conversion():
    # SI range 0..100 m maps to a US range that must still be increasing.
    box = UnitAwareSpinBox(LENGTH_LARGE, si_minimum=0.0, si_maximum=100.0, si_decimals=2, si_step=1.0)
    box.set_unit_system(UnitSystem.US)
    assert box.minimum() < box.maximum()
    assert box.minimum() == pytest.approx(0.0, abs=1e-6)
    assert box.maximum() == pytest.approx(100.0 / 0.3048, abs=0.01)
