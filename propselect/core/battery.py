"""Battery pack terminal-voltage models.

Two selectable models:

A. ``InternalResistanceBattery`` -- V_terminal(I) = V_ocv(SoC) - I*R_internal,
   built from a user-supplied per-cell OCV(SoC) table scaled by series count.
B. ``MeasuredCurveBattery`` -- a directly measured loaded V-vs-I table,
   interpolated with a monotone cubic (PCHIP).

Both clamp and warn rather than silently extrapolating outside their data
range. Cumulative amp-hours (tracked externally, e.g. by the takeoff
integrator) combine with ``capacity_ah`` via ``soc_from_capacity_used`` to
produce a state of charge that drives model A's voltage sag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class BatteryResult:
    """Result of a terminal-voltage lookup.

    Attributes:
        voltage_v: terminal (loaded) voltage [V].
        clamped: True if the query fell outside the source data range.
        warning: human-readable message if clamped, else None.
    """

    voltage_v: float
    clamped: bool
    warning: str | None = None


class Battery(Protocol):
    """Common interface implemented by both battery models."""

    capacity_ah: float

    def terminal_voltage(self, current_a: float, soc: float) -> BatteryResult: ...


def soc_from_capacity_used(initial_soc: float, capacity_ah: float, ah_used: float) -> float:
    """State of charge after drawing ``ah_used`` amp-hours from a pack.

    Args:
        initial_soc: starting state of charge, in [0, 1].
        capacity_ah: total pack capacity [Ah].
        ah_used: cumulative amp-hours drawn so far [Ah].
    """
    return initial_soc - ah_used / capacity_ah


class OCVTable:
    """Per-cell open-circuit-voltage vs state-of-charge table.

    SoC is dimensionless in [0, 1]; voltage is per-cell volts. Interpolated
    with a monotone cubic (PCHIP) so the curve cannot overshoot between
    sparse table entries.
    """

    def __init__(self, soc: list[float] | np.ndarray, voltage_v: list[float] | np.ndarray):
        soc_arr = np.asarray(soc, dtype=float)
        voltage_arr = np.asarray(voltage_v, dtype=float)
        if soc_arr.shape != voltage_arr.shape or soc_arr.size < 2:
            raise ValueError("OCV table requires matching soc/voltage arrays with >= 2 points")
        order = np.argsort(soc_arr)
        self._soc = soc_arr[order]
        self._voltage = voltage_arr[order]
        self._interp = PchipInterpolator(self._soc, self._voltage, extrapolate=False)
        self.soc_min = float(self._soc.min())
        self.soc_max = float(self._soc.max())

    def voltage(self, soc: float) -> BatteryResult:
        """Per-cell OCV at a state of charge [V], clamped to the table range."""
        clamped_soc = min(max(soc, self.soc_min), self.soc_max)
        clamped = clamped_soc != soc
        v = float(self._interp(clamped_soc))
        warning = None
        if clamped:
            warning = (
                f"SoC {soc:.3f} outside OCV table range "
                f"[{self.soc_min:.3f}, {self.soc_max:.3f}]; clamped to {clamped_soc:.3f}"
            )
        return BatteryResult(voltage_v=v, clamped=clamped, warning=warning)


@dataclass
class InternalResistanceBattery:
    """Model A: OCV table plus a lumped internal resistance.

    Attributes:
        ocv_table: per-cell OCV(SoC) table.
        series: number of cells in series (S).
        parallel: number of cells in parallel (P).
        r_internal_per_cell_ohm: single-cell internal resistance [ohm].
        capacity_ah: total pack capacity [Ah] (per-cell capacity * parallel).
    """

    ocv_table: OCVTable
    series: int
    parallel: int
    r_internal_per_cell_ohm: float
    capacity_ah: float

    def r_internal_ohm(self) -> float:
        """Pack internal resistance [ohm]: scales as series/parallel."""
        return self.r_internal_per_cell_ohm * self.series / self.parallel

    def terminal_voltage(self, current_a: float, soc: float) -> BatteryResult:
        """Loaded pack terminal voltage [V] at a given total pack current and SoC."""
        cell = self.ocv_table.voltage(soc)
        v_ocv_pack = cell.voltage_v * self.series
        v_terminal = v_ocv_pack - current_a * self.r_internal_ohm()
        return BatteryResult(voltage_v=v_terminal, clamped=cell.clamped, warning=cell.warning)


@dataclass
class MeasuredCurveBattery:
    """Model B: a directly measured loaded voltage-vs-current table.

    Attributes:
        current_table_a: measured pack current points [A].
        voltage_table_v: corresponding measured loaded voltage points [V].
        capacity_ah: total pack capacity [Ah], used for SoC/energy bookkeeping
            (not used by the voltage lookup itself).
    """

    current_table_a: list[float]
    voltage_table_v: list[float]
    capacity_ah: float = field(default=float("nan"))

    def __post_init__(self) -> None:
        current_arr = np.asarray(self.current_table_a, dtype=float)
        voltage_arr = np.asarray(self.voltage_table_v, dtype=float)
        if current_arr.shape != voltage_arr.shape or current_arr.size < 2:
            raise ValueError(
                "Measured curve requires matching current/voltage arrays with >= 2 points"
            )
        order = np.argsort(current_arr)
        self._current = current_arr[order]
        self._voltage = voltage_arr[order]
        self._interp = PchipInterpolator(self._current, self._voltage, extrapolate=False)
        self.current_min = float(self._current.min())
        self.current_max = float(self._current.max())

    def terminal_voltage(self, current_a: float, soc: float = 1.0) -> BatteryResult:
        """Loaded pack terminal voltage [V] at a given total pack current.

        ``soc`` is accepted for interface parity with ``InternalResistanceBattery``
        but is not used: the measured curve already reflects whatever SoC it was
        recorded at.
        """
        clamped_i = min(max(current_a, self.current_min), self.current_max)
        clamped = clamped_i != current_a
        v = float(self._interp(clamped_i))
        warning = None
        if clamped:
            warning = (
                f"Current {current_a:.2f} A outside measured range "
                f"[{self.current_min:.2f}, {self.current_max:.2f}] A; clamped to {clamped_i:.2f} A"
            )
        return BatteryResult(voltage_v=v, clamped=clamped, warning=warning)
