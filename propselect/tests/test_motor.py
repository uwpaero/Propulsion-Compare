import math

import pytest

from propselect.core.motor import (
    MotorSpec,
    back_emf_voltage,
    motor_efficiency,
    motor_torque_n_m,
    no_load_speed_rev_s,
    omega_rad_s,
    total_resistance_ohm,
)


def test_kt_from_kv():
    spec = MotorSpec(name="test", kv_rpm_per_v=1000.0, r_motor_ohm=0.05, i0_a=1.0)
    # Kt = 9.5493 / Kv
    assert spec.kt_n_m_per_a() == pytest.approx(9.5493 / 1000.0, rel=1e-4)


def test_motor_torque():
    # Q = Kt*(I - I0)
    assert motor_torque_n_m(kt_n_m_per_a=0.01, current_a=15.0, i0_a=1.0) == pytest.approx(
        0.01 * 14.0
    )
    # At I == I0, torque is zero (no-load point).
    assert motor_torque_n_m(kt_n_m_per_a=0.01, current_a=1.0, i0_a=1.0) == pytest.approx(0.0)


def test_back_emf_voltage():
    # 60*n/Kv
    assert back_emf_voltage(motor_speed_rev_s=100.0, kv_rpm_per_v=1000.0) == pytest.approx(6.0)


def test_total_resistance_sums_components():
    assert total_resistance_ohm(0.05, 0.01, 0.02) == pytest.approx(0.08)
    assert total_resistance_ohm(0.05) == pytest.approx(0.05)


def test_omega_rad_s():
    assert omega_rad_s(1.0) == pytest.approx(2 * math.pi)


def test_motor_efficiency_formula_and_bounds():
    # eta = Q*omega / (V*I)
    torque, omega, v, i = 0.5, 100.0, 20.0, 10.0
    eta = motor_efficiency(torque, omega, v, i)
    assert eta == pytest.approx((torque * omega) / (v * i))
    assert 0.0 < eta

    # Non-physical / zero input power must not divide by zero.
    assert motor_efficiency(0.0, 0.0, 0.0, 0.0) == 0.0
    assert motor_efficiency(0.0, 0.0, 20.0, 0.0) == 0.0


def test_no_load_speed():
    # n_noload = Kv*V/60
    n = no_load_speed_rev_s(kv_rpm_per_v=1000.0, v_batt_no_load_v=22.2)
    assert n == pytest.approx(1000.0 * 22.2 / 60.0)
