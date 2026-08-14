"""BLDC motor model (direct drive only -- no gearbox).

    Kt [N*m/A]  = 9.5493 / Kv [RPM/V]
    Q_motor     = Kt * (I - I0)
    V_batt(I) - I*R_total = 60*n / Kv      [n in rev/s]
    R_total     = R_motor + R_esc + R_battery_internal
    eta_motor   = Q*omega / (V_batt*I),    omega = 2*pi*n
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Converts Kv [RPM/V] to Kt [N*m/A]: 60 / (2*pi), i.e. rad/s per RPM inverted
# into the torque constant via P = Q*omega = V*I at unity efficiency.
KT_FROM_KV_CONSTANT: float = 60.0 / (2.0 * math.pi)


@dataclass(frozen=True)
class MotorSpec:
    """A motor's electrical and physical parameters.

    Attributes:
        name: identifying label.
        kv_rpm_per_v: velocity constant [RPM/V].
        r_motor_ohm: motor winding resistance [ohm].
        i0_a: no-load current [A].
        i_max_cont_a: continuous current rating [A], if known.
        i_max_burst_a: burst/peak current rating [A], if known.
        mass_kg: motor mass [kg], if known.
        shaft_dia_mm: shaft diameter [mm], if known.
        source_url: where this data came from.
        notes: free-text notes.
    """

    name: str
    kv_rpm_per_v: float
    r_motor_ohm: float
    i0_a: float
    i_max_cont_a: float | None = None
    i_max_burst_a: float | None = None
    mass_kg: float | None = None
    shaft_dia_mm: float | None = None
    source_url: str | None = None
    notes: str | None = None

    def kt_n_m_per_a(self) -> float:
        """Torque constant [N*m/A]: Kt = 9.5493 / Kv."""
        return KT_FROM_KV_CONSTANT / self.kv_rpm_per_v


def motor_torque_n_m(kt_n_m_per_a: float, current_a: float, i0_a: float) -> float:
    """Q_motor = Kt * (I - I0)  [N*m]."""
    return kt_n_m_per_a * (current_a - i0_a)


def back_emf_voltage(motor_speed_rev_s: float, kv_rpm_per_v: float) -> float:
    """60*n/Kv  [V], with n the motor shaft speed in rev/s."""
    return 60.0 * motor_speed_rev_s / kv_rpm_per_v


def total_resistance_ohm(
    r_motor_ohm: float, r_esc_ohm: float = 0.0, r_battery_internal_ohm: float = 0.0
) -> float:
    """R_total = R_motor + R_esc + R_battery_internal  [ohm]."""
    return r_motor_ohm + r_esc_ohm + r_battery_internal_ohm


def omega_rad_s(n_rev_s: float) -> float:
    """omega = 2*pi*n  [rad/s]."""
    return 2.0 * math.pi * n_rev_s


def motor_efficiency(
    torque_n_m: float, omega_rad_s_: float, v_batt_v: float, current_a: float
) -> float:
    """eta_motor = Q*omega / (V_batt*I), dimensionless. Zero if input power <= 0."""
    input_power_w = v_batt_v * current_a
    if input_power_w <= 0.0:
        return 0.0
    return (torque_n_m * omega_rad_s_) / input_power_w


def no_load_speed_rev_s(kv_rpm_per_v: float, v_batt_no_load_v: float) -> float:
    """No-load propeller speed n_noload = Kv*V_batt(0) / 60  [rev/s]."""
    return kv_rpm_per_v * v_batt_no_load_v / 60.0
