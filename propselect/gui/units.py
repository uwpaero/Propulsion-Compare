"""SI/US unit display conversion for the GUI.

The physics core and the ``Project`` model are always SI internally --
nothing here ever touches them. This module only converts what a spinbox
*displays*: toggling unit systems must convert the number already on screen
in place, never reset or overwrite it. ``UnitAwareSpinBox`` is the widget
that makes that possible: it stores nothing of its own beyond what Qt's
QDoubleSpinBox already stores, but interprets that stored value as "the
current unit system's number" and converts through ``QuantityUnit`` on toggle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from PySide6.QtWidgets import QDoubleSpinBox, QWidget


class UnitSystem(Enum):
    SI = "SI"
    US = "US"


@dataclass(frozen=True)
class QuantityUnit:
    """A physical quantity's SI and US customary display units.

    ``to_us``/``to_si`` are general affine maps (they handle temperature's
    offset as well as pure-scale conversions like mass or length).
    """

    si_suffix: str
    us_suffix: str
    to_us: Callable[[float], float]
    to_si: Callable[[float], float]


def _scale(si_per_us: float) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """Build (to_us, to_si) for a pure scale conversion: 1 US unit = si_per_us SI units."""
    return (lambda si: si / si_per_us), (lambda us: us * si_per_us)


_mass_to_us, _mass_to_si = _scale(0.45359237)  # 1 lb = 0.45359237 kg
MASS = QuantityUnit("kg", "lb", _mass_to_us, _mass_to_si)

_len_large_to_us, _len_large_to_si = _scale(0.3048)  # 1 ft = 0.3048 m
LENGTH_LARGE = QuantityUnit("m", "ft", _len_large_to_us, _len_large_to_si)

_len_small_to_us, _len_small_to_si = _scale(0.0254)  # 1 in = 0.0254 m
LENGTH_SMALL = QuantityUnit("m", "in", _len_small_to_us, _len_small_to_si)

_area_to_us, _area_to_si = _scale(0.3048**2)  # 1 ft^2 = 0.3048^2 m^2
AREA = QuantityUnit("m^2", "ft^2", _area_to_us, _area_to_si)

_vel_to_us, _vel_to_si = _scale(0.44704)  # 1 mph = 0.44704 m/s
VELOCITY = QuantityUnit("m/s", "mph", _vel_to_us, _vel_to_si)

TEMPERATURE = QuantityUnit(
    "degC", "degF",
    to_us=lambda c: c * 9.0 / 5.0 + 32.0,
    to_si=lambda f: (f - 32.0) * 5.0 / 9.0,
)


class UnitAwareSpinBox(QDoubleSpinBox):
    """A QDoubleSpinBox whose displayed number is always in the active unit
    system, but which can always be read/written in SI via ``si_value()`` /
    ``set_si_value()``.

    Calling ``set_unit_system`` converts the currently displayed value into
    the new unit system and updates the suffix/range to match -- it never
    clears or resets the field.
    """

    def __init__(
        self,
        quantity: QuantityUnit,
        si_minimum: float,
        si_maximum: float,
        si_decimals: int = 3,
        si_step: float = 0.1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.quantity = quantity
        self._unit_system = UnitSystem.SI
        self._si_minimum = si_minimum
        self._si_maximum = si_maximum
        self._si_decimals = si_decimals
        self._si_step = si_step
        self._apply_range_and_suffix()

    def _apply_range_and_suffix(self) -> None:
        if self._unit_system is UnitSystem.SI:
            lo, hi = self._si_minimum, self._si_maximum
            step = self._si_step
            suffix = self.quantity.si_suffix
        else:
            us_lo = self.quantity.to_us(self._si_minimum)
            us_hi = self.quantity.to_us(self._si_maximum)
            lo, hi = (us_lo, us_hi) if us_lo <= us_hi else (us_hi, us_lo)
            # Convert a step (a delta), not an absolute value: strip any
            # affine offset (matters for temperature) by differencing against 0.
            step = self.quantity.to_us(self._si_step) - self.quantity.to_us(0.0)
            suffix = self.quantity.us_suffix
        self.setRange(lo, hi)
        self.setDecimals(self._si_decimals)
        self.setSingleStep(abs(step))
        self.setSuffix(f" {suffix}")

    def si_value(self) -> float:
        """The current value converted to SI, regardless of display unit."""
        value = self.value()
        return value if self._unit_system is UnitSystem.SI else self.quantity.to_si(value)

    def set_si_value(self, si_value: float) -> None:
        """Set the value from SI; displays it converted to the active unit system."""
        display_value = (
            si_value if self._unit_system is UnitSystem.SI else self.quantity.to_us(si_value)
        )
        self.setValue(display_value)

    def set_unit_system(self, system: UnitSystem) -> None:
        """Switch display units, converting the currently shown value in place."""
        if system is self._unit_system:
            return
        current_si = self.si_value()
        self._unit_system = system
        self.blockSignals(True)
        self._apply_range_and_suffix()
        self.set_si_value(current_si)
        self.blockSignals(False)
