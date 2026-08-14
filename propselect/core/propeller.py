"""Propeller thrust/power coefficient model.

    J = V / (n*D)                    [n in rev/s, D in m]
    T = C_T(J) * rho * n^2 * D^4     [N]
    P = C_P(J) * rho * n^3 * D^5     [W]
    Q = C_P(J) * rho * n^2 * D^5 / (2*pi)   [N*m]

C_T(J) and C_P(J) come from tabulated data, interpolated with a monotone
cubic (PCHIP). Below the table's J range the lowest table value is held
(flagged). At and beyond the zero-thrust point, C_T is clamped to zero and
C_P is held at its value at that point -- interpolation is never allowed to
produce negative power.

A parametric fallback model is provided for propellers with no tabulated
data; any candidate built from it must be reported as low confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

# Flags describing why a coefficient lookup deviated from plain interpolation.
FLAG_BELOW_TABLE_RANGE = "below_table_range"
FLAG_BEYOND_ZERO_THRUST = "beyond_zero_thrust"
FLAG_LOW_CONFIDENCE_PARAMETRIC = "low_confidence_parametric"


@dataclass(frozen=True)
class PropCoeffs:
    """A single (C_T, C_P) lookup result.

    Attributes:
        ct: thrust coefficient, dimensionless.
        cp: power coefficient, dimensionless.
        flag: reason the value deviates from direct table interpolation, if any.
    """

    ct: float
    cp: float
    flag: str | None = None


class PropellerModel(Protocol):
    """Common interface for tabulated and parametric propeller models."""

    diameter_m: float
    pitch_m: float
    name: str
    is_low_confidence: bool

    def evaluate(self, advance_ratio: float) -> PropCoeffs: ...


class PropellerDataTable:
    """Tabulated C_T(J), C_P(J) curve built from real propeller test data."""

    is_low_confidence = False

    def __init__(
        self,
        advance_ratio: list[float] | np.ndarray,
        ct: list[float] | np.ndarray,
        cp: list[float] | np.ndarray,
        diameter_m: float,
        pitch_m: float,
        name: str = "",
        source: str = "",
    ):
        j_arr = np.asarray(advance_ratio, dtype=float)
        ct_arr = np.asarray(ct, dtype=float)
        cp_arr = np.asarray(cp, dtype=float)
        if not (j_arr.shape == ct_arr.shape == cp_arr.shape) or j_arr.size < 2:
            raise ValueError("J, CT, CP must be equal-length arrays with >= 2 points")
        order = np.argsort(j_arr)
        self._j = j_arr[order]
        self._ct = ct_arr[order]
        self._cp = cp_arr[order]
        self._ct_interp = PchipInterpolator(self._j, self._ct, extrapolate=False)
        self._cp_interp = PchipInterpolator(self._j, self._cp, extrapolate=False)

        self.diameter_m = diameter_m
        self.pitch_m = pitch_m
        self.name = name
        self.source = source
        self.j_min = float(self._j.min())
        self.j_max = float(self._j.max())
        self.j_zero_thrust = self._find_zero_thrust_j()
        cutover_j = self.j_zero_thrust if self.j_zero_thrust is not None else self.j_max
        self._cp_at_cutover = max(float(self._cp_interp(cutover_j)), 0.0)

    def _find_zero_thrust_j(self) -> float | None:
        ct_at_min = float(self._ct_interp(self.j_min))
        ct_at_max = float(self._ct_interp(self.j_max))
        if ct_at_min <= 0.0:
            return self.j_min
        if ct_at_max > 0.0:
            return None
        js = np.linspace(self.j_min, self.j_max, 400)
        cts = self._ct_interp(js)
        for a, b, ca, cb in zip(js[:-1], js[1:], cts[:-1], cts[1:]):
            if ca > 0.0 and cb <= 0.0:
                return float(brentq(lambda x: float(self._ct_interp(x)), a, b))
        return None

    def evaluate(self, advance_ratio: float) -> PropCoeffs:
        j = advance_ratio
        if j < self.j_min:
            return PropCoeffs(
                ct=max(float(self._ct_interp(self.j_min)), 0.0),
                cp=max(float(self._cp_interp(self.j_min)), 0.0),
                flag=FLAG_BELOW_TABLE_RANGE,
            )
        cutover_j = self.j_zero_thrust if self.j_zero_thrust is not None else self.j_max
        if j <= cutover_j:
            ct = max(float(self._ct_interp(j)), 0.0)
            cp = max(float(self._cp_interp(j)), 0.0)
            return PropCoeffs(ct=ct, cp=cp, flag=None)
        return PropCoeffs(ct=0.0, cp=self._cp_at_cutover, flag=FLAG_BEYOND_ZERO_THRUST)


@dataclass
class ParametricPropellerModel:
    """Fallback model for a propeller with no tabulated test data.

    Linear C_T decay from a static value to zero at J = 0.9*(pitch/diameter),
    with constant C_P. Any candidate built from this model is LOW CONFIDENCE.
    """

    diameter_m: float
    pitch_m: float
    ct_static: float
    cp_constant: float
    name: str = ""
    is_low_confidence: bool = True

    def evaluate(self, advance_ratio: float) -> PropCoeffs:
        j_zero_thrust = 0.9 * (self.pitch_m / self.diameter_m)
        if advance_ratio >= j_zero_thrust:
            ct = 0.0
        else:
            ct = self.ct_static * (1.0 - advance_ratio / j_zero_thrust)
        return PropCoeffs(
            ct=max(ct, 0.0), cp=self.cp_constant, flag=FLAG_LOW_CONFIDENCE_PARAMETRIC
        )


def advance_ratio(v_m_s: float, n_rev_s: float, diameter_m: float) -> float:
    """J = V / (n*D), dimensionless. Returns +inf for n<=0 and V>0 (locked-rotor limit)."""
    if n_rev_s <= 0.0:
        return math.inf if v_m_s > 0.0 else 0.0
    return v_m_s / (n_rev_s * diameter_m)


def thrust_n(ct: float, rho_kg_m3: float, n_rev_s: float, diameter_m: float) -> float:
    """T = C_T(J) * rho * n^2 * D^4  [N]."""
    return ct * rho_kg_m3 * n_rev_s**2 * diameter_m**4


def power_w(cp: float, rho_kg_m3: float, n_rev_s: float, diameter_m: float) -> float:
    """P = C_P(J) * rho * n^3 * D^5  [W]."""
    return cp * rho_kg_m3 * n_rev_s**3 * diameter_m**5


def torque_n_m(cp: float, rho_kg_m3: float, n_rev_s: float, diameter_m: float) -> float:
    """Q = C_P(J) * rho * n^2 * D^5 / (2*pi)  [N*m]."""
    return cp * rho_kg_m3 * n_rev_s**2 * diameter_m**5 / (2.0 * math.pi)


def momentum_theory_static_thrust_n(
    power_shaft_w: float, rho_kg_m3: float, diameter_m: float, figure_of_merit: float
) -> float:
    """Ideal (actuator disk) static thrust: T = FM*(2*rho*A)^(1/3) * P_shaft^(2/3)  [N]."""
    disk_area_m2 = math.pi * (diameter_m / 2.0) ** 2
    return figure_of_merit * (2.0 * rho_kg_m3 * disk_area_m2) ** (1.0 / 3.0) * power_shaft_w ** (
        2.0 / 3.0
    )


def momentum_theory_thrust_band_n(
    power_shaft_w: float,
    rho_kg_m3: float,
    diameter_m: float,
    fm_min: float = 0.4,
    fm_max: float = 0.75,
) -> tuple[float, float]:
    """Plausible static-thrust band [N] from actuator disk theory, FM in [fm_min, fm_max].

    Used as a sanity cross-check on the solved static operating point: a
    solved static thrust well outside this band usually indicates a data or
    unit problem, not a real propeller.
    """
    t_min = momentum_theory_static_thrust_n(power_shaft_w, rho_kg_m3, diameter_m, fm_min)
    t_max = momentum_theory_static_thrust_n(power_shaft_w, rho_kg_m3, diameter_m, fm_max)
    return t_min, t_max
