"""Candidate combination evaluation: run the full physics stack for one
motor + propeller combination (direct drive) and score it against every filter.

Every candidate is evaluated for every filter and reported with a per-filter
pass/fail status -- failures are never hard-dropped from the results, so a
combination that misses one filter by a hair is still visible next to one
that misses badly.

    Tip Mach:       M_tip = sqrt((pi*n*D)^2 + V^2) / a        <= tip_mach_limit
    Current/motor:  max I(V) (per motor) <= min(ESC_cont, motor_cont)
    Current/pack:   max I_pack(V) <= C_rate*capacity (if set)
    Power:          max P_elec (total, all motors) <= power_limit_w (if set)
    Pitch speed:    V_pitch = n*pitch at V_t, report ratio V_pitch/V_t
    Motor eta:      eta_motor at V_t >= motor_eta_threshold
    Prop eta:       report peak eta_prop and eta_prop at V_t
    Distance:       s <= distance_allowed_m, with margin % reported

With ``motor_count`` identical motors sharing one pack (see
``CandidateSpec.motor_count``), the current filter necessarily splits in
two: ESC/motor continuous ratings are per-motor limits, while a C-rate limit
is a pack-level limit on the summed current of every motor. Thrust, shaft
power, and electrical power are aggregates across all motors; tip Mach,
pitch speed, and both efficiencies are per-motor (symmetric, so identical
across motors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from propselect.core.battery import Battery
from propselect.core.motor import MotorSpec
from propselect.core.operating_point import (
    OperatingPoint,
    OperatingPointFailure,
    OperatingPointResult,
    sweep_operating_points,
)
from propselect.core.propeller import PropellerModel, momentum_theory_thrust_band_n
from propselect.core.takeoff import AircraftConfig, GroundRollResult, integrate_ground_roll

# Number of points used for the diagnostic 0 -> 1.2*V_t sweep (max current,
# max power, max tip Mach, peak prop efficiency). This is independent of the
# ground-roll integration grid.
DIAGNOSTIC_SWEEP_POINTS: int = 61


@dataclass(frozen=True)
class RequirementSpec:
    """Aircraft, requirement, and environment inputs shared by every candidate.

    Attributes:
        aircraft: airframe and ground-roll parameters.
        v_t_m_s: target (rotation) velocity [m/s].
        distance_allowed_m: allowed takeoff distance [m].
        rho_kg_m3: air density [kg/m^3].
        speed_of_sound_m_s: speed of sound [m/s], for tip Mach.
        tip_mach_limit: maximum allowed tip Mach number.
        motor_eta_threshold: minimum required motor efficiency at V_t.
        power_limit_w: maximum allowed electrical power [W], if set.
        c_rate_limit: maximum allowed battery discharge C-rate, if set.
        dv_m_s: velocity integration step for the ground roll [m/s].
        initial_soc: starting battery state of charge, in [0, 1].
    """

    aircraft: AircraftConfig
    v_t_m_s: float
    distance_allowed_m: float
    rho_kg_m3: float
    speed_of_sound_m_s: float
    tip_mach_limit: float = 0.75
    motor_eta_threshold: float = 0.75
    power_limit_w: float | None = None
    c_rate_limit: float | None = None
    dv_m_s: float = 0.1
    initial_soc: float = 1.0


@dataclass(frozen=True)
class CandidateSpec:
    """One motor + propeller combination to evaluate (direct drive only).

    Attributes:
        motor: motor spec.
        prop: propeller model (tabulated or parametric).
        r_esc_ohm: ESC resistance [ohm] (per motor -- each motor has its own ESC).
        esc_current_cont_a: ESC continuous current rating [A] (per motor), if known.
        motor_count: number of identical motor+prop+ESC units sharing one
            pack. Each solves to the same per-motor operating point; the
            pack sees motor_count times the per-motor current.
    """

    motor: MotorSpec
    prop: PropellerModel
    r_esc_ohm: float = 0.0
    esc_current_cont_a: float | None = None
    motor_count: int = 1


@dataclass(frozen=True)
class FilterResult:
    """Pass/fail status for one filter, with the computed value and threshold.

    Attributes:
        name: filter name.
        passed: whether the candidate passes this filter.
        value: computed value being checked, if applicable.
        threshold: threshold checked against, if configured.
        evaluated: False if the filter had no threshold configured (e.g. no
            power limit set) and was therefore not actually constraining.
        detail: human-readable summary.
    """

    name: str
    passed: bool
    value: float | None
    threshold: float | None
    evaluated: bool = True
    detail: str = ""


@dataclass(frozen=True)
class CandidateResult:
    """Full evaluation of one motor + propeller combination."""

    spec: CandidateSpec
    ground_roll: GroundRollResult
    diagnostic_operating_points: list[OperatingPointResult]
    filters: list[FilterResult]
    all_pass: bool
    is_low_confidence: bool
    distance_m: float
    distance_margin_pct: float | None
    time_s: float
    capacity_used_mah: float
    thrust_at_vt_n: float | None
    power_max_w: float
    current_max_per_motor_a: float
    current_max_pack_a: float
    tip_mach_max: float
    v_pitch_ratio: float | None
    eta_motor_at_vt: float | None
    eta_prop_peak: float
    eta_prop_at_vt: float | None
    momentum_theory_warning: str | None


def _tip_mach(n_rev_s: float, diameter_m: float, v_m_s: float, speed_of_sound_m_s: float) -> float:
    """M_tip = sqrt((pi*n*D)^2 + V^2) / a."""
    tip_speed_m_s = math.pi * n_rev_s * diameter_m
    return math.sqrt(tip_speed_m_s**2 + v_m_s**2) / speed_of_sound_m_s


def _per_motor_current_limit_a(motor: MotorSpec, esc_current_cont_a: float | None) -> float | None:
    """min(ESC_cont, motor_cont), the per-motor current limit."""
    candidates = []
    if motor.i_max_cont_a is not None:
        candidates.append(motor.i_max_cont_a)
    if esc_current_cont_a is not None:
        candidates.append(esc_current_cont_a)
    return min(candidates) if candidates else None


def _pack_current_limit_a(c_rate_limit: float | None, capacity_ah: float | None) -> float | None:
    """C_rate * capacity, the pack-level current limit."""
    if c_rate_limit is not None and capacity_ah is not None and math.isfinite(capacity_ah):
        return c_rate_limit * capacity_ah
    return None


def evaluate_candidate(
    spec: CandidateSpec, requirement: RequirementSpec, battery: Battery
) -> CandidateResult:
    """Run the ground roll and every filter for one candidate combination."""
    ground_roll = integrate_ground_roll(
        v_t_m_s=requirement.v_t_m_s,
        rho_kg_m3=requirement.rho_kg_m3,
        aircraft=requirement.aircraft,
        prop=spec.prop,
        motor=spec.motor,
        battery=battery,
        r_esc_ohm=spec.r_esc_ohm,
        dv_m_s=requirement.dv_m_s,
        initial_soc=requirement.initial_soc,
        motor_count=spec.motor_count,
    )

    # Diagnostic sweep 0 -> 1.2*V_t for max current/power/tip-Mach and peak
    # prop efficiency, at a constant initial SoC (independent of the
    # ground-roll's SoC-coupled 0->V_t integration).
    v_diag = np.linspace(0.0, 1.2 * requirement.v_t_m_s, DIAGNOSTIC_SWEEP_POINTS)
    diag_points = sweep_operating_points(
        v_diag,
        requirement.rho_kg_m3,
        spec.prop,
        spec.motor,
        battery,
        spec.r_esc_ohm,
        soc_values=[requirement.initial_soc] * len(v_diag),
        motor_count=spec.motor_count,
    )
    valid_diag = [p for p in diag_points if isinstance(p, OperatingPoint)]

    current_max_per_motor_a = max((p.current_a for p in valid_diag), default=0.0)
    current_max_pack_a = max((p.current_pack_a for p in valid_diag), default=0.0)
    power_max_w = max((p.power_elec_w for p in valid_diag), default=0.0)
    tip_mach_max = max(
        (
            _tip_mach(p.n_rev_s, spec.prop.diameter_m, p.v_m_s, requirement.speed_of_sound_m_s)
            for p in valid_diag
        ),
        default=0.0,
    )
    eta_prop_peak = max((p.eta_prop for p in valid_diag), default=0.0)

    # Operating point at exactly V_t: prefer the ground roll's own last point
    # (which carries the actual SoC sag accumulated over the roll).
    op_at_vt: OperatingPointResult | None = None
    if ground_roll.operating_points:
        last = ground_roll.operating_points[-1]
        if isinstance(last, OperatingPoint) and math.isclose(
            last.v_m_s, requirement.v_t_m_s, rel_tol=1e-6, abs_tol=1e-6
        ):
            op_at_vt = last
    if op_at_vt is None:
        op_at_vt = next(
            (p for p in diag_points if isinstance(p, OperatingPoint) and math.isclose(
                p.v_m_s, requirement.v_t_m_s, rel_tol=1e-6, abs_tol=1e-3
            )),
            None,
        )

    thrust_at_vt_n = op_at_vt.thrust_n if isinstance(op_at_vt, OperatingPoint) else None
    eta_motor_at_vt = op_at_vt.eta_motor if isinstance(op_at_vt, OperatingPoint) else None
    eta_prop_at_vt = op_at_vt.eta_prop if isinstance(op_at_vt, OperatingPoint) else None

    v_pitch_ratio: float | None = None
    if isinstance(op_at_vt, OperatingPoint) and requirement.v_t_m_s > 0:
        v_pitch_m_s = op_at_vt.n_rev_s * spec.prop.pitch_m
        v_pitch_ratio = v_pitch_m_s / requirement.v_t_m_s

    capacity_ah = getattr(battery, "capacity_ah", None)
    per_motor_current_limit = _per_motor_current_limit_a(spec.motor, spec.esc_current_cont_a)
    pack_current_limit = _pack_current_limit_a(requirement.c_rate_limit, capacity_ah)

    filters: list[FilterResult] = []

    filters.append(
        FilterResult(
            name="tip_mach",
            passed=tip_mach_max <= requirement.tip_mach_limit,
            value=tip_mach_max,
            threshold=requirement.tip_mach_limit,
            detail=f"M_tip max {tip_mach_max:.3f} vs limit {requirement.tip_mach_limit:.3f}",
        )
    )

    filters.append(
        FilterResult(
            name="current_per_motor",
            passed=(per_motor_current_limit is None) or (current_max_per_motor_a <= per_motor_current_limit),
            value=current_max_per_motor_a,
            threshold=per_motor_current_limit,
            evaluated=per_motor_current_limit is not None,
            detail=(
                f"I max {current_max_per_motor_a:.2f} A/motor vs limit {per_motor_current_limit:.2f} A"
                if per_motor_current_limit is not None
                else f"I max {current_max_per_motor_a:.2f} A/motor (no ESC/motor current limit configured)"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="current_pack",
            passed=(pack_current_limit is None) or (current_max_pack_a <= pack_current_limit),
            value=current_max_pack_a,
            threshold=pack_current_limit,
            evaluated=pack_current_limit is not None,
            detail=(
                f"I_pack max {current_max_pack_a:.2f} A ({spec.motor_count}x motor) vs "
                f"C-rate limit {pack_current_limit:.2f} A"
                if pack_current_limit is not None
                else f"I_pack max {current_max_pack_a:.2f} A ({spec.motor_count}x motor) (no C-rate limit configured)"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="power",
            passed=(requirement.power_limit_w is None) or (power_max_w <= requirement.power_limit_w),
            value=power_max_w,
            threshold=requirement.power_limit_w,
            evaluated=requirement.power_limit_w is not None,
            detail=(
                f"P max {power_max_w:.1f} W vs limit {requirement.power_limit_w:.1f} W"
                if requirement.power_limit_w is not None
                else f"P max {power_max_w:.1f} W (no power limit configured)"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="pitch_speed",
            passed=True,
            value=v_pitch_ratio,
            threshold=None,
            evaluated=v_pitch_ratio is not None,
            detail=(
                f"V_pitch/V_t = {v_pitch_ratio:.2f}" if v_pitch_ratio is not None else "not available"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="motor_efficiency",
            passed=(eta_motor_at_vt is None) or (eta_motor_at_vt >= requirement.motor_eta_threshold),
            value=eta_motor_at_vt,
            threshold=requirement.motor_eta_threshold,
            evaluated=eta_motor_at_vt is not None,
            detail=(
                f"eta_motor@Vt {eta_motor_at_vt:.3f} vs threshold {requirement.motor_eta_threshold:.3f}"
                if eta_motor_at_vt is not None
                else "operating point at V_t unavailable"
            ),
        )
    )

    filters.append(
        FilterResult(
            name="prop_efficiency",
            passed=True,
            value=eta_prop_peak,
            threshold=None,
            detail=(
                f"eta_prop peak {eta_prop_peak:.3f}, at V_t "
                f"{eta_prop_at_vt:.3f}" if eta_prop_at_vt is not None else f"eta_prop peak {eta_prop_peak:.3f}"
            ),
        )
    )

    distance_m = ground_roll.distance_m
    distance_margin_pct: float | None
    if math.isfinite(distance_m):
        distance_margin_pct = (
            (requirement.distance_allowed_m - distance_m) / requirement.distance_allowed_m * 100.0
        )
    else:
        distance_margin_pct = None
    filters.append(
        FilterResult(
            name="distance",
            passed=math.isfinite(distance_m) and distance_m <= requirement.distance_allowed_m,
            value=distance_m if math.isfinite(distance_m) else None,
            threshold=requirement.distance_allowed_m,
            detail=(
                f"s={distance_m:.1f} m vs allowed {requirement.distance_allowed_m:.1f} m "
                f"(margin {distance_margin_pct:.1f}%)"
                if distance_margin_pct is not None
                else (
                    f"did not reach V_t"
                    + (f" (stalled at {ground_roll.stall_v_m_s:.1f} m/s)" if ground_roll.stall_v_m_s else "")
                    + (f", {ground_roll.reason}" if ground_roll.reason else "")
                )
            ),
        )
    )

    all_pass = all(f.passed for f in filters)

    momentum_theory_warning: str | None = None
    static_op = ground_roll.operating_points[0] if ground_roll.operating_points else None
    if isinstance(static_op, OperatingPoint) and static_op.power_shaft_w > 0:
        # Momentum theory is a per-disk relationship (T ~ P^(2/3), not
        # linear), so aggregate thrust/power across motor_count motors must
        # be divided back down to per-motor values before checking the band
        # -- checking totals directly would bias the band by motor_count^(1/3).
        per_motor_power_w = static_op.power_shaft_w / spec.motor_count
        per_motor_thrust_n = static_op.thrust_n / spec.motor_count
        t_min, t_max = momentum_theory_thrust_band_n(
            per_motor_power_w, requirement.rho_kg_m3, spec.prop.diameter_m
        )
        if not (t_min <= per_motor_thrust_n <= t_max):
            momentum_theory_warning = (
                f"Per-motor static thrust {per_motor_thrust_n:.1f} N is outside the "
                f"momentum-theory plausible band [{t_min:.1f}, {t_max:.1f}] N for "
                f"P_shaft={per_motor_power_w:.0f} W/motor -- check propeller data and units."
            )

    capacity_used_mah = ground_roll.capacity_used_ah * 1000.0

    return CandidateResult(
        spec=spec,
        ground_roll=ground_roll,
        diagnostic_operating_points=diag_points,
        filters=filters,
        all_pass=all_pass,
        is_low_confidence=spec.prop.is_low_confidence,
        distance_m=distance_m,
        distance_margin_pct=distance_margin_pct,
        time_s=ground_roll.time_s,
        capacity_used_mah=capacity_used_mah,
        thrust_at_vt_n=thrust_at_vt_n,
        power_max_w=power_max_w,
        current_max_per_motor_a=current_max_per_motor_a,
        current_max_pack_a=current_max_pack_a,
        tip_mach_max=tip_mach_max,
        v_pitch_ratio=v_pitch_ratio,
        eta_motor_at_vt=eta_motor_at_vt,
        eta_prop_peak=eta_prop_peak,
        eta_prop_at_vt=eta_prop_at_vt,
        momentum_theory_warning=momentum_theory_warning,
    )


def evaluate_candidates(
    specs: list[CandidateSpec], requirement: RequirementSpec, battery: Battery
) -> list[CandidateResult]:
    """Evaluate a list of candidates (the cross product of motors x props x gears)."""
    return [evaluate_candidate(spec, requirement, battery) for spec in specs]
