import math

import numpy as np
import pytest

from propselect.core.propeller import (
    FLAG_BELOW_TABLE_RANGE,
    FLAG_BEYOND_ZERO_THRUST,
    FLAG_LOW_CONFIDENCE_PARAMETRIC,
    ParametricPropellerModel,
    PropellerDataTable,
    advance_ratio,
    momentum_theory_static_thrust_n,
    power_w,
    thrust_n,
    torque_n_m,
)


def make_table_crossing_zero() -> PropellerDataTable:
    # Roughly APC-shaped CT(J) curve that crosses zero within the table,
    # and a CP(J) curve that stays positive throughout.
    j = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ct = [0.11, 0.10, 0.085, 0.06, 0.03, -0.01]
    cp = [0.045, 0.048, 0.047, 0.040, 0.028, 0.010]
    return PropellerDataTable(j, ct, cp, diameter_m=0.254, pitch_m=0.254, name="test-prop")


def test_advance_ratio_formula():
    assert advance_ratio(v_m_s=10.0, n_rev_s=100.0, diameter_m=0.25) == pytest.approx(
        10.0 / (100.0 * 0.25)
    )


def test_advance_ratio_locked_rotor_edge_cases():
    assert advance_ratio(v_m_s=0.0, n_rev_s=0.0, diameter_m=0.25) == 0.0
    assert advance_ratio(v_m_s=5.0, n_rev_s=0.0, diameter_m=0.25) == math.inf


def test_table_interpolates_within_range():
    table = make_table_crossing_zero()
    result = table.evaluate(0.3)
    assert result.flag is None
    # Should lie between the bracketing table values.
    assert 0.085 < result.ct < 0.10
    assert 0.040 < result.cp < 0.048


def test_table_holds_lowest_value_below_range_and_flags():
    table = make_table_crossing_zero()
    result = table.evaluate(-0.5)
    assert result.flag == FLAG_BELOW_TABLE_RANGE
    assert result.ct == pytest.approx(0.11, abs=1e-6)
    assert result.cp == pytest.approx(0.045, abs=1e-6)


def test_table_zero_thrust_point_found_and_clamped():
    table = make_table_crossing_zero()
    assert table.j_zero_thrust is not None
    # CT must be ~0 right at the crossing.
    at_crossing = table.evaluate(table.j_zero_thrust)
    assert at_crossing.ct == pytest.approx(0.0, abs=1e-6)


def test_table_beyond_zero_thrust_ct_zero_cp_held_no_negative_power():
    table = make_table_crossing_zero()
    far_beyond = table.evaluate(5.0)
    assert far_beyond.flag == FLAG_BEYOND_ZERO_THRUST
    assert far_beyond.ct == 0.0
    assert far_beyond.cp >= 0.0
    # CP must be held at the cutover value, not the raw (possibly negative)
    # table endpoint extrapolation.
    assert far_beyond.cp == pytest.approx(table._cp_at_cutover)


def test_table_never_produces_negative_ct_or_cp():
    table = make_table_crossing_zero()
    for j in np.linspace(-1.0, 6.0, 50):
        result = table.evaluate(float(j))
        assert result.ct >= 0.0
        assert result.cp >= 0.0


def test_table_without_zero_crossing_holds_table_max():
    # CT stays positive across the whole table -- no crossing found.
    j = [0.0, 0.3, 0.6]
    ct = [0.12, 0.09, 0.05]
    cp = [0.05, 0.045, 0.035]
    table = PropellerDataTable(j, ct, cp, diameter_m=0.25, pitch_m=0.25)
    assert table.j_zero_thrust is None
    beyond = table.evaluate(1.0)
    assert beyond.ct == 0.0
    assert beyond.cp == pytest.approx(0.035, abs=1e-6)


def test_parametric_model_linear_decay_and_low_confidence_flag():
    model = ParametricPropellerModel(
        diameter_m=0.254, pitch_m=0.1524, ct_static=0.10, cp_constant=0.04
    )
    j_zero = 0.9 * (0.1524 / 0.254)
    at_zero_advance = model.evaluate(0.0)
    assert at_zero_advance.ct == pytest.approx(0.10)
    assert at_zero_advance.flag == FLAG_LOW_CONFIDENCE_PARAMETRIC

    at_cutoff = model.evaluate(j_zero)
    assert at_cutoff.ct == pytest.approx(0.0, abs=1e-9)

    beyond = model.evaluate(j_zero + 1.0)
    assert beyond.ct == 0.0
    assert beyond.cp == pytest.approx(0.04)
    assert model.is_low_confidence is True


def test_thrust_power_torque_formulas():
    ct, cp, rho, n, d = 0.10, 0.04, 1.225, 150.0, 0.254
    assert thrust_n(ct, rho, n, d) == pytest.approx(ct * rho * n**2 * d**4)
    assert power_w(cp, rho, n, d) == pytest.approx(cp * rho * n**3 * d**5)
    assert torque_n_m(cp, rho, n, d) == pytest.approx(cp * rho * n**2 * d**5 / (2 * math.pi))


def test_momentum_theory_hand_calc_case():
    # From the build spec's hand-calc check case: P_shaft=750W, FM=0.55, rho=1.16.
    t_16in = momentum_theory_static_thrust_n(750.0, 1.16, 0.406, 0.55)
    t_22in = momentum_theory_static_thrust_n(750.0, 1.16, 0.559, 0.55)
    assert t_16in == pytest.approx(30.4, rel=0.02)
    assert t_22in == pytest.approx(37.6, rel=0.02)
    # ~24% gain from diameter alone at constant power (T ~ D^(2/3) scaling).
    gain = (t_22in - t_16in) / t_16in
    assert gain == pytest.approx(0.24, abs=0.02)


def test_power_from_torque_and_omega_consistent():
    # P = Q*omega should match the direct power formula (P = C_P*rho*n^3*D^5).
    cp, rho, n, d = 0.04, 1.225, 150.0, 0.254
    q = torque_n_m(cp, rho, n, d)
    omega = 2 * math.pi * n
    p_from_q = q * omega
    p_direct = power_w(cp, rho, n, d)
    assert p_from_q == pytest.approx(p_direct, rel=1e-9)
