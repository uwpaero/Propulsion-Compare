"""Cruise (steady, level flight) operating point and candidate evaluation.

Unlike takeoff, cruise is a single equilibrium condition, not an
integration: thrust must equal drag at one airspeed, with the aircraft in
steady, unaccelerated, level flight (L=W).

    C_L     = 2*m*g / (rho*V^2*S)
    C_D     = C_D0_cruise + C_L^2/(pi*AR*e)
    T_req   = D = 0.5*rho*V^2*S*C_D

``solve_cruise_point`` finds the propeller speed n that produces exactly
T_req at V (thrust depends only on n and V, not on throttle setting), then
works backward to current, required motor voltage, and battery draw -- the
motor/prop torque-current relation (Q = Kt*(I-I0)) holds at any throttle, so
the current calc needs no throttle model at all. The full-throttle operating
point (reusing ``solve_operating_point``) supplies both the n-search upper
bound and the feasibility ceiling: if the aircraft can't produce T_req even
at WOT, cruise at that speed is not achievable with this combination.

The existing motor/ESC circuit model (operating_point.py) has no throttle
knob -- WOT is the only condition it solves directly. Partial throttle is
modeled here as the ESC linearly dropping the excess voltage
(V_battery_available - V_required) resistively, so pack current equals
motor current exactly as at WOT (no buck step-down).
ponytail: linear/resistive-ESC throttle model, not a switching buck
converter -- conservative (pessimistic) for partial-throttle battery draw
and endurance. Upgrade path: model duty = V_required/V_battery_available and
I_pack = duty*I_motor if endurance estimates need to be tighter than this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from scipy.optimize import brentq

from propselect.core.atmosphere import G0
from propselect.core.battery import Battery
from propselect.core.candidate import CandidateSpec, FilterResult
from propselect.core.motor import MotorSpec, back_emf_voltage, motor_efficiency, omega_rad_s
from propselect.core.operating_point import (
    DEFAULT_N_MIN_REV_S,
    OperatingPoint,
    OperatingPointFailure,
    solve_operating_point,
)
from propselect.core.propeller import PropellerModel, advance_ratio, power_w, thrust_n, torque_n_m
from propselect.core.takeoff import AircraftConfig, aerodynamic_drag_n


def _tip_mach(n_rev_s: float, diameter_m: float, v_m_s: float, speed_of_sound_m_s: float) -> float:
    tip_speed_m_s = math.pi * n_rev_s * diameter_m
    return math.sqrt(tip_speed_m_s**2 + v_m_s**2) / speed_of_sound_m_s


@dataclass(frozen=True)
class CruiseRequirement:
    """Aircraft, requirement, and environment inputs for a cruise evaluation.

    Reuses the aircraft's mass/wing area/span/oswald efficiency from
    ``AircraftConfig`` (its ground-only fields -- mu, cl_ground, wheel_height,
    ground_effect -- are simply unused here); ``cd0_cruise`` is the one
    cruise-specific input, since the ground-roll C_D0 (gear down, ground
    proximity) does not describe a clean-configuration cruise condition.
    """

    aircraft: AircraftConfig
    v_cruise_m_s: float
    cd0_cruise: float
    rho_kg_m3: float
    speed_of_sound_m_s: float
    tip_mach_limit: float = 0.75
    motor_eta_threshold: float = 0.75
    power_limit_w: float | None = None
    c_rate_limit: float | None = None
    endurance_required_s: float | None = None
    initial_soc: float = 1.0


@dataclass(frozen=True)
class CruisePoint:
    """A converged cruise operating point: thrust equals required drag at V.

    Attributes mirror ``OperatingPoint`` where the meaning is the same; see
    that class for the motor_count aggregation convention. ``voltage_v`` here
    is the voltage the ESC must deliver to the motor (V_required), not the
    battery's own terminal voltage -- ``throttle_fraction`` is the ratio of
    the two.
    """

    v_m_s: float
    n_rev_s: float
    j: float
    ct: float
    cp: float
    prop_flag: str | None
    motor_count: int
    thrust_n: float
    torque_prop_n_m: float
    current_a: float
    current_pack_a: float
    voltage_v: float
    throttle_fraction: float
    power_shaft_w: float
    power_elec_w: float
    eta_motor: float
    eta_prop: float
    battery_warning: str | None

    success: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class CruisePointFailure:
    """A structured, non-exceptional cruise solve failure."""

    v_m_s: float
    reason: str

    success: bool = False


CruisePointResult = Union[CruisePoint, CruisePointFailure]


def solve_cruise_point(
    v_m_s: float,
    thrust_required_n: float,
    rho_kg_m3: float,
    prop: PropellerModel,
    motor: MotorSpec,
    battery: Battery,
    r_esc_ohm: float = 0.0,
    soc: float = 1.0,
    motor_count: int = 1,
    n_min_rev_s: float = DEFAULT_N_MIN_REV_S,
) -> CruisePointResult:
    """Solve for the propeller speed that produces exactly ``thrust_required_n`` at V.

    Returns a ``CruisePoint`` on success or a ``CruisePointFailure`` with a
    diagnostic reason -- never raises for an infeasible combination.
    """
    if thrust_required_n <= 0.0:
        return CruisePointFailure(
            v_m_s=v_m_s, reason=f"Required cruise thrust must be positive, got {thrust_required_n:.3g} N"
        )

    wot = solve_operating_point(v_m_s, rho_kg_m3, prop, motor, battery, r_esc_ohm, soc, motor_count=motor_count)
    if isinstance(wot, OperatingPointFailure):
        return CruisePointFailure(v_m_s=v_m_s, reason=f"Full-throttle solve failed: {wot.reason}")
    if wot.thrust_n < thrust_required_n:
        return CruisePointFailure(
            v_m_s=v_m_s,
            reason=(
                f"Required thrust {thrust_required_n:.2f} N exceeds max available thrust "
                f"{wot.thrust_n:.2f} N at V={v_m_s:.1f} m/s (full throttle) -- cannot sustain cruise "
                "with this combination."
            ),
        )

    def thrust_deficit(n: float) -> float:
        j = advance_ratio(v_m_s, n, prop.diameter_m)
        ct = prop.evaluate(j).ct
        return motor_count * thrust_n(ct, rho_kg_m3, n, prop.diameter_m) - thrust_required_n

    try:
        f_lo = thrust_deficit(n_min_rev_s)
        f_hi = thrust_deficit(wot.n_rev_s)
    except Exception as exc:
        return CruisePointFailure(v_m_s=v_m_s, reason=f"Thrust evaluation failed: {exc}")

    if f_lo * f_hi > 0.0:
        return CruisePointFailure(
            v_m_s=v_m_s,
            reason=(
                f"No sign change bracketed between n={n_min_rev_s:.3g} rev/s and "
                f"n={wot.n_rev_s:.4g} rev/s (full throttle) while searching for the thrust-matched "
                "propeller speed."
            ),
        )

    try:
        n_solution = brentq(thrust_deficit, n_min_rev_s, wot.n_rev_s, xtol=1e-9, rtol=1e-12)
    except (ValueError, RuntimeError) as exc:
        return CruisePointFailure(v_m_s=v_m_s, reason=f"brentq failed to converge: {exc}")

    j = advance_ratio(v_m_s, n_solution, prop.diameter_m)
    coeffs = prop.evaluate(j)
    thrust_per_motor = thrust_n(coeffs.ct, rho_kg_m3, n_solution, prop.diameter_m)
    q_prop = torque_n_m(coeffs.cp, rho_kg_m3, n_solution, prop.diameter_m)
    current = motor.i0_a + q_prop / motor.kt_n_m_per_a()
    current_pack = motor_count * current
    back_emf = back_emf_voltage(n_solution, motor.kv_rpm_per_v)
    v_required = current * (motor.r_motor_ohm + r_esc_ohm) + back_emf

    battery_result = battery.terminal_voltage(current_pack, soc)
    v_batt_avail = battery_result.voltage_v
    throttle_fraction = v_required / v_batt_avail if v_batt_avail > 0.0 else math.inf

    omega_motor = omega_rad_s(n_solution)
    power_shaft_per_motor = power_w(coeffs.cp, rho_kg_m3, n_solution, prop.diameter_m)
    power_elec_battery_w = v_batt_avail * current_pack
    eta_mot = motor_efficiency(q_prop, omega_motor, v_required, current)
    eta_prop = (
        0.0
        if v_m_s <= 0.0 or power_shaft_per_motor <= 0.0
        else (thrust_per_motor * v_m_s) / power_shaft_per_motor
    )

    return CruisePoint(
        v_m_s=v_m_s,
        n_rev_s=n_solution,
        j=j,
        ct=coeffs.ct,
        cp=coeffs.cp,
        prop_flag=coeffs.flag,
        motor_count=motor_count,
        thrust_n=motor_count * thrust_per_motor,
        torque_prop_n_m=q_prop,
        current_a=current,
        current_pack_a=current_pack,
        voltage_v=v_required,
        throttle_fraction=throttle_fraction,
        power_shaft_w=motor_count * power_shaft_per_motor,
        power_elec_w=power_elec_battery_w,
        eta_motor=eta_mot,
        eta_prop=eta_prop,
        battery_warning=battery_result.warning,
    )


@dataclass(frozen=True)
class CruiseCandidateResult:
    """Full evaluation of one motor + propeller combination at cruise."""

    spec: CandidateSpec
    cruise_point: CruisePointResult
    filters: list[FilterResult]
    all_pass: bool
    eligible: bool
    is_low_confidence: bool
    thrust_required_n: float
    cl_cruise: float
    cd_cruise: float
    tip_mach: float | None
    throttle_fraction: float | None
    current_per_motor_a: float | None
    current_pack_a: float | None
    power_elec_w: float | None
    eta_motor: float | None
    eta_prop: float | None
    endurance_s: float | None
    range_m: float | None


def evaluate_cruise_candidate(
    spec: CandidateSpec, requirement: CruiseRequirement, battery: Battery
) -> CruiseCandidateResult:
    """Solve the cruise equilibrium and run every filter for one candidate combination."""
    aircraft = requirement.aircraft
    v = requirement.v_cruise_m_s
    cl_cruise = 2.0 * aircraft.mass_kg * G0 / (requirement.rho_kg_m3 * v**2 * aircraft.wing_area_m2)
    cd_induced = cl_cruise**2 / (math.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    cd_cruise = requirement.cd0_cruise + cd_induced
    thrust_required_n = aerodynamic_drag_n(requirement.rho_kg_m3, v, aircraft.wing_area_m2, cd_cruise)

    point = solve_cruise_point(
        v_m_s=v,
        thrust_required_n=thrust_required_n,
        rho_kg_m3=requirement.rho_kg_m3,
        prop=spec.prop,
        motor=spec.motor,
        battery=battery,
        r_esc_ohm=spec.r_esc_ohm,
        soc=requirement.initial_soc,
        motor_count=spec.motor_count,
    )

    ok = isinstance(point, CruisePoint)
    tip_mach = _tip_mach(point.n_rev_s, spec.prop.diameter_m, v, requirement.speed_of_sound_m_s) if ok else None
    current_per_motor = point.current_a if ok else None
    current_pack = point.current_pack_a if ok else None
    power_elec = point.power_elec_w if ok else None
    eta_motor = point.eta_motor if ok else None
    eta_prop = point.eta_prop if ok else None
    throttle_fraction = point.throttle_fraction if ok else None

    capacity_ah = getattr(battery, "capacity_ah", None)
    endurance_s: float | None = None
    if ok and capacity_ah is not None and math.isfinite(capacity_ah) and current_pack:
        endurance_s = requirement.initial_soc * capacity_ah / current_pack * 3600.0
    range_m = endurance_s * v if endurance_s is not None else None

    motor_current_limit = spec.motor.i_max_cont_a
    esc_current_limit = spec.esc_current_cont_a
    pack_current_limit = (
        requirement.c_rate_limit * capacity_ah
        if requirement.c_rate_limit is not None and capacity_ah is not None and math.isfinite(capacity_ah)
        else None
    )

    filters: list[FilterResult] = []

    filters.append(
        FilterResult(
            name="cruise_solve",
            passed=ok,
            value=None,
            threshold=None,
            detail=("converged" if ok else point.reason),  # type: ignore[union-attr]
        )
    )

    filters.append(
        FilterResult(
            name="tip_mach",
            passed=(tip_mach is not None) and (tip_mach <= requirement.tip_mach_limit),
            value=tip_mach,
            threshold=requirement.tip_mach_limit,
            evaluated=ok,
            detail=(
                f"M_tip {tip_mach:.3f} vs limit {requirement.tip_mach_limit:.3f}"
                if tip_mach is not None
                else "cruise point unavailable"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="current_per_motor",
            passed=(not ok) or (motor_current_limit is None) or (current_per_motor <= motor_current_limit),
            value=current_per_motor,
            threshold=motor_current_limit,
            evaluated=ok and motor_current_limit is not None,
            detail=(
                f"I {current_per_motor:.2f} A/motor vs motor rating {motor_current_limit:.2f} A"
                if ok and motor_current_limit is not None
                else "cruise point unavailable" if not ok else f"I {current_per_motor:.2f} A/motor (no motor current rating configured)"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="current_esc",
            passed=(not ok) or (esc_current_limit is None) or (current_per_motor <= esc_current_limit),
            value=current_per_motor,
            threshold=esc_current_limit,
            evaluated=ok and esc_current_limit is not None,
            detail=(
                f"I {current_per_motor:.2f} A/motor vs ESC rating {esc_current_limit:.2f} A"
                if ok and esc_current_limit is not None
                else "cruise point unavailable" if not ok else f"I {current_per_motor:.2f} A/motor (no ESC current rating configured)"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="current_pack",
            passed=(not ok) or (pack_current_limit is None) or (current_pack <= pack_current_limit),
            value=current_pack,
            threshold=pack_current_limit,
            evaluated=ok and pack_current_limit is not None,
            detail=(
                f"I_pack {current_pack:.2f} A ({spec.motor_count}x motor) vs C-rate limit {pack_current_limit:.2f} A"
                if ok and pack_current_limit is not None
                else "cruise point unavailable" if not ok else f"I_pack {current_pack:.2f} A (no C-rate limit configured)"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="power",
            passed=(not ok) or (requirement.power_limit_w is None) or (power_elec <= requirement.power_limit_w),
            value=power_elec,
            threshold=requirement.power_limit_w,
            evaluated=ok and requirement.power_limit_w is not None,
            detail=(
                f"P {power_elec:.1f} W vs limit {requirement.power_limit_w:.1f} W"
                if ok and requirement.power_limit_w is not None
                else "cruise point unavailable" if not ok else f"P {power_elec:.1f} W (no power limit configured)"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="endurance",
            passed=(not ok) or (requirement.endurance_required_s is None) or (
                endurance_s is not None and endurance_s >= requirement.endurance_required_s
            ),
            value=endurance_s,
            threshold=requirement.endurance_required_s,
            evaluated=ok and requirement.endurance_required_s is not None,
            detail=(
                f"endurance {endurance_s:.0f} s vs required {requirement.endurance_required_s:.0f} s"
                if ok and endurance_s is not None and requirement.endurance_required_s is not None
                else "cruise point unavailable" if not ok else "no endurance requirement configured"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="motor_efficiency",
            passed=(not ok) or (eta_motor is None) or (eta_motor >= requirement.motor_eta_threshold),
            value=eta_motor,
            threshold=requirement.motor_eta_threshold,
            evaluated=ok,
            hard=False,
            detail=(
                f"eta_motor {eta_motor:.3f} vs threshold {requirement.motor_eta_threshold:.3f}"
                if ok and eta_motor is not None
                else "cruise point unavailable"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="prop_efficiency",
            passed=True,
            value=eta_prop,
            threshold=None,
            evaluated=ok,
            hard=False,
            detail=f"eta_prop {eta_prop:.3f}" if ok and eta_prop is not None else "cruise point unavailable",
        )
    )

    all_pass = all(f.passed for f in filters)
    eligible = all(f.passed for f in filters if f.hard)

    return CruiseCandidateResult(
        spec=spec,
        cruise_point=point,
        filters=filters,
        all_pass=all_pass,
        eligible=eligible,
        is_low_confidence=spec.prop.is_low_confidence,
        thrust_required_n=thrust_required_n,
        cl_cruise=cl_cruise,
        cd_cruise=cd_cruise,
        tip_mach=tip_mach,
        throttle_fraction=throttle_fraction,
        current_per_motor_a=current_per_motor,
        current_pack_a=current_pack,
        power_elec_w=power_elec,
        eta_motor=eta_motor,
        eta_prop=eta_prop,
        endurance_s=endurance_s,
        range_m=range_m,
    )


def evaluate_cruise_candidates(
    specs: list[CandidateSpec], requirement: CruiseRequirement, battery: Battery
) -> list[CruiseCandidateResult]:
    """Evaluate a list of candidates (the cross product of motors x props x motor counts)."""
    return [evaluate_cruise_candidate(spec, requirement, battery) for spec in specs]
