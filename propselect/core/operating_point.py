"""Coupled motor/propeller operating point solver (direct drive only).

Solved as a 1-D root find rather than a fragile 2-D fsolve. For a trial
propeller speed ``n`` at a given airspeed ``V``:

    1. J        = V / (n*D)
    2. Q_prop   = C_P(J)*rho*n^2*D^5 / (2*pi)
    3. I        = I0 + Q_prop / Kt                    (per-motor current)
    4. residual = V_batt(motor_count*I) - I*(R_motor + R_esc) - 60*n/Kv

Note: ``V_batt(I_pack)`` here is the battery model's own loaded terminal
voltage, which for the internal-resistance model already nets out
I_pack*R_battery_internal internally. Adding only R_motor + R_esc on top
(rather than re-adding R_battery_internal separately) is what keeps this
equivalent to "V_ocv - I_pack*R_battery_internal - I*(R_motor+R_esc) =
back-emf" without double counting the battery's own resistive drop -- and it
generalizes to the measured-curve battery model, which has no separate R to
extract.

With ``motor_count`` identical motors sharing one pack (all seeing the same
bus voltage, all symmetric so they solve to the same per-motor operating
point), the pack current is motor_count times the per-motor current -- that
total is what drives battery voltage sag and SoC depletion. R_motor and
R_esc are per-motor components, so their drop still uses the per-motor
current I, not the pack current.

Root-find residual(n) = 0 with scipy.optimize.brentq over
n in [n_min, n_noload], where n_noload = Kv*V_batt(0)/60 (independent of
motor_count, since it's evaluated at zero current). The bracket is checked
before calling brentq; if the residual has the same sign at both ends, a
structured failure is returned with the reason -- never a bare exception,
never a silently wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

from scipy.optimize import brentq

from propselect.core.battery import Battery
from propselect.core.motor import (
    MotorSpec,
    back_emf_voltage,
    motor_efficiency,
    no_load_speed_rev_s,
    omega_rad_s,
)
from propselect.core.propeller import PropellerModel, advance_ratio, power_w, thrust_n, torque_n_m

# Minimum bracketed propeller speed for the root find [rev/s]. Not zero, to
# keep J = V/(n*D) finite; small enough that torque/current at this bound are
# negligible for any realistic propeller.
DEFAULT_N_MIN_REV_S: float = 1.0e-6


@dataclass(frozen=True)
class OperatingPoint:
    """A converged motor/propeller operating point at one airspeed.

    With ``motor_count`` identical motors sharing one pack, per-motor and
    aggregate quantities diverge -- each is named explicitly below.

    Attributes:
        v_m_s: airspeed [m/s].
        n_rev_s: propeller (and motor) shaft speed [rev/s] (same for every motor).
        j: advance ratio, dimensionless.
        ct: thrust coefficient at this J, dimensionless.
        cp: power coefficient at this J, dimensionless.
        prop_flag: propeller-model flag (e.g. below/above data range), if any.
        motor_count: number of identical motors sharing this pack.
        thrust_n: TOTAL thrust across all motors [N].
        torque_prop_n_m: PER-MOTOR propeller shaft torque [N*m].
        current_a: PER-MOTOR current [A].
        current_pack_a: PACK current = motor_count * current_a [A].
        voltage_v: battery terminal voltage at current_pack_a [V] (shared bus).
        power_shaft_w: TOTAL propeller shaft power across all motors [W].
        power_elec_w: TOTAL electrical input power = voltage_v * current_pack_a [W].
        eta_motor: per-motor motor efficiency, dimensionless.
        eta_prop: per-motor propulsive efficiency, dimensionless (0 at V=0).
        battery_warning: battery clamp warning at this operating point, if any.
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
    power_shaft_w: float
    power_elec_w: float
    eta_motor: float
    eta_prop: float
    battery_warning: str | None

    success: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class OperatingPointFailure:
    """A structured, non-exceptional solver failure. Never a silently wrong number."""

    v_m_s: float
    reason: str

    success: bool = False


OperatingPointResult = Union[OperatingPoint, OperatingPointFailure]


def _residual(
    n_rev_s: float,
    v_m_s: float,
    rho_kg_m3: float,
    prop: PropellerModel,
    motor: MotorSpec,
    battery: Battery,
    r_esc_ohm: float,
    soc: float,
    motor_count: int,
) -> float:
    j = advance_ratio(v_m_s, n_rev_s, prop.diameter_m)
    coeffs = prop.evaluate(j)
    q_prop = torque_n_m(coeffs.cp, rho_kg_m3, n_rev_s, prop.diameter_m)
    current = motor.i0_a + q_prop / motor.kt_n_m_per_a()
    v_batt = battery.terminal_voltage(motor_count * current, soc).voltage_v
    back_emf = back_emf_voltage(n_rev_s, motor.kv_rpm_per_v)
    return v_batt - current * (motor.r_motor_ohm + r_esc_ohm) - back_emf


def solve_operating_point(
    v_m_s: float,
    rho_kg_m3: float,
    prop: PropellerModel,
    motor: MotorSpec,
    battery: Battery,
    r_esc_ohm: float = 0.0,
    soc: float = 1.0,
    n_min_rev_s: float = DEFAULT_N_MIN_REV_S,
    motor_count: int = 1,
) -> OperatingPointResult:
    """Solve the coupled motor/propeller operating point at one airspeed.

    ``motor_count`` identical motors are assumed to share one pack and split
    the load symmetrically: each solves to the same per-motor operating
    point, and the pack sees motor_count times the per-motor current.

    Returns an ``OperatingPoint`` on success or an ``OperatingPointFailure``
    with a diagnostic reason -- never raises for an infeasible combination.
    """
    if motor_count < 1:
        return OperatingPointFailure(v_m_s=v_m_s, reason=f"motor_count must be >= 1, got {motor_count}")

    try:
        v_batt_open = battery.terminal_voltage(0.0, soc).voltage_v
        n_noload = no_load_speed_rev_s(motor.kv_rpm_per_v, v_batt_open)
    except Exception as exc:  # battery/motor data pathology, not a bare crash
        return OperatingPointFailure(
            v_m_s=v_m_s, reason=f"Failed to compute no-load bracket: {exc}"
        )

    if n_noload <= n_min_rev_s:
        return OperatingPointFailure(
            v_m_s=v_m_s,
            reason=(
                f"No-load speed ({n_noload:.6g} rev/s) is not above the solver's "
                f"minimum bracket ({n_min_rev_s:.3g} rev/s); check battery voltage and Kv."
            ),
        )

    def residual(n: float) -> float:
        return _residual(n, v_m_s, rho_kg_m3, prop, motor, battery, r_esc_ohm, soc, motor_count)

    try:
        r_lo = residual(n_min_rev_s)
        r_hi = residual(n_noload)
    except Exception as exc:
        return OperatingPointFailure(v_m_s=v_m_s, reason=f"Residual evaluation failed: {exc}")

    if r_lo * r_hi > 0.0:
        return OperatingPointFailure(
            v_m_s=v_m_s,
            reason=(
                f"No sign change bracketed between n={n_min_rev_s:.3g} rev/s "
                f"(residual={r_lo:.4g} V) and n={n_noload:.4g} rev/s "
                f"(residual={r_hi:.4g} V); combination is likely infeasible "
                "(insufficient battery voltage, mismatched Kv/prop, or excess "
                "resistance)."
            ),
        )

    try:
        n_solution = brentq(residual, n_min_rev_s, n_noload, xtol=1e-9, rtol=1e-12)
    except (ValueError, RuntimeError) as exc:
        return OperatingPointFailure(v_m_s=v_m_s, reason=f"brentq failed to converge: {exc}")

    j = advance_ratio(v_m_s, n_solution, prop.diameter_m)
    coeffs = prop.evaluate(j)
    thrust_per_motor = thrust_n(coeffs.ct, rho_kg_m3, n_solution, prop.diameter_m)
    q_prop = torque_n_m(coeffs.cp, rho_kg_m3, n_solution, prop.diameter_m)
    current = motor.i0_a + q_prop / motor.kt_n_m_per_a()
    current_pack = motor_count * current
    battery_result = battery.terminal_voltage(current_pack, soc)
    voltage = battery_result.voltage_v
    omega_motor = omega_rad_s(n_solution)
    power_shaft_per_motor = power_w(coeffs.cp, rho_kg_m3, n_solution, prop.diameter_m)
    power_elec_total = voltage * current_pack
    eta_mot = motor_efficiency(q_prop, omega_motor, voltage, current)
    eta_prop = (
        0.0
        if v_m_s <= 0.0 or power_shaft_per_motor <= 0.0
        else (thrust_per_motor * v_m_s) / power_shaft_per_motor
    )

    return OperatingPoint(
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
        voltage_v=voltage,
        power_shaft_w=motor_count * power_shaft_per_motor,
        power_elec_w=power_elec_total,
        eta_motor=eta_mot,
        eta_prop=eta_prop,
        battery_warning=battery_result.warning,
    )


def sweep_operating_points(
    v_values_m_s: Iterable[float],
    rho_kg_m3: float,
    prop: PropellerModel,
    motor: MotorSpec,
    battery: Battery,
    r_esc_ohm: float = 0.0,
    soc_values: Iterable[float] | None = None,
    motor_count: int = 1,
) -> list[OperatingPointResult]:
    """Solve the operating point at each airspeed in ``v_values_m_s``.

    ``soc_values``, if given, must have the same length as ``v_values_m_s`` and
    lets a caller (e.g. the takeoff integrator) drive SoC-based voltage sag
    point by point. Defaults to a constant SoC of 1.0.
    """
    v_list = list(v_values_m_s)
    if soc_values is None:
        soc_list = [1.0] * len(v_list)
    else:
        soc_list = list(soc_values)
        if len(soc_list) != len(v_list):
            raise ValueError("soc_values must be the same length as v_values_m_s")
    return [
        solve_operating_point(v, rho_kg_m3, prop, motor, battery, r_esc_ohm, soc, motor_count=motor_count)
        for v, soc in zip(v_list, soc_list)
    ]
