import math

import pytest

from propselect.core.atmosphere import speed_of_sound
from propselect.core.battery import InternalResistanceBattery, OCVTable
from propselect.core.candidate import (
    CandidateSpec,
    RequirementSpec,
    evaluate_candidate,
    evaluate_candidates,
)
from propselect.core.motor import MotorSpec
from propselect.core.propeller import ParametricPropellerModel, PropellerDataTable
from propselect.core.takeoff import AircraftConfig


def make_battery(capacity_ah=5.0) -> InternalResistanceBattery:
    soc = [0.0, 0.5, 1.0]
    voltage = [3.3, 3.7, 4.2]
    return InternalResistanceBattery(
        ocv_table=OCVTable(soc, voltage),
        series=6,
        parallel=2,
        r_internal_per_cell_ohm=0.006,
        capacity_ah=capacity_ah,
    )


def make_motor(**overrides) -> MotorSpec:
    defaults = dict(
        name="test-motor",
        kv_rpm_per_v=900.0,
        r_motor_ohm=0.05,
        i0_a=0.8,
        i_max_cont_a=60.0,
    )
    defaults.update(overrides)
    return MotorSpec(**defaults)


def make_prop() -> PropellerDataTable:
    j = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    ct = [0.11, 0.105, 0.09, 0.07, 0.045, 0.015, -0.02]
    cp = [0.045, 0.044, 0.040, 0.033, 0.024, 0.015, 0.008]
    return PropellerDataTable(j, ct, cp, diameter_m=0.28, pitch_m=0.16, name="11x6")


def make_requirement(**overrides) -> RequirementSpec:
    aircraft = AircraftConfig(
        mass_kg=8.0,
        wing_area_m2=0.65,
        span_m=2.0,
        cd0_ground=0.09,
        cl_ground=0.60,
        mu=0.10,
    )
    defaults = dict(
        aircraft=aircraft,
        v_t_m_s=15.0,
        distance_allowed_m=60.0,
        rho_kg_m3=1.16,
        speed_of_sound_m_s=speed_of_sound(288.15),
    )
    defaults.update(overrides)
    return RequirementSpec(**defaults)


def test_evaluate_candidate_reports_all_filters_never_drops():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    requirement = make_requirement()
    result = evaluate_candidate(spec, requirement, make_battery())

    filter_names = {f.name for f in result.filters}
    assert filter_names == {
        "tip_mach",
        "current_per_motor",
        "current_pack",
        "power",
        "pitch_speed",
        "motor_efficiency",
        "prop_efficiency",
        "distance",
    }
    # All 8 filters must be present regardless of pass/fail.
    assert len(result.filters) == 8


def test_candidate_result_not_low_confidence_for_tabulated_prop():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    result = evaluate_candidate(spec, make_requirement(), make_battery())
    assert result.is_low_confidence is False


def test_candidate_result_low_confidence_for_parametric_prop():
    parametric_prop = ParametricPropellerModel(
        diameter_m=0.28, pitch_m=0.16, ct_static=0.10, cp_constant=0.035
    )
    spec = CandidateSpec(motor=make_motor(), prop=parametric_prop)
    result = evaluate_candidate(spec, make_requirement(), make_battery())
    assert result.is_low_confidence is True


def test_current_per_motor_filter_uses_min_of_esc_and_motor_limits():
    # ESC continuous limit is the binding (lowest) constraint here.
    spec = CandidateSpec(
        motor=make_motor(i_max_cont_a=60.0), prop=make_prop(), esc_current_cont_a=5.0
    )
    result = evaluate_candidate(spec, make_requirement(), make_battery())
    current_filter = next(f for f in result.filters if f.name == "current_per_motor")
    assert current_filter.threshold == pytest.approx(5.0)
    # With such a low ESC limit the candidate should fail the per-motor current filter.
    assert current_filter.passed is False
    assert result.all_pass is False


def test_current_pack_filter_uses_c_rate_times_capacity():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    requirement = make_requirement(c_rate_limit=1.0)  # 1C * 5Ah = 5A pack limit -- very low
    result = evaluate_candidate(spec, requirement, make_battery(capacity_ah=5.0))
    pack_filter = next(f for f in result.filters if f.name == "current_pack")
    assert pack_filter.threshold == pytest.approx(5.0)
    assert pack_filter.passed is False
    assert result.all_pass is False


def test_current_filters_not_evaluated_when_no_limits_configured():
    spec = CandidateSpec(motor=make_motor(i_max_cont_a=None), prop=make_prop())
    requirement = make_requirement()
    result = evaluate_candidate(spec, requirement, make_battery())
    per_motor_filter = next(f for f in result.filters if f.name == "current_per_motor")
    pack_filter = next(f for f in result.filters if f.name == "current_pack")
    assert per_motor_filter.evaluated is False
    assert per_motor_filter.passed is True
    assert pack_filter.evaluated is False
    assert pack_filter.passed is True


def test_motor_count_current_filters_split_correctly():
    # A 2-motor candidate: per-motor current stays modest, but pack current
    # (2x) can trip a C-rate limit that a single motor wouldn't.
    spec = CandidateSpec(motor=make_motor(i_max_cont_a=None), prop=make_prop(), motor_count=2)
    requirement = make_requirement(c_rate_limit=3.0)  # 3C * 5Ah = 15A pack limit
    result = evaluate_candidate(spec, requirement, make_battery(capacity_ah=5.0))
    assert result.current_max_pack_a == pytest.approx(2.0 * result.current_max_per_motor_a, rel=0.05)
    pack_filter = next(f for f in result.filters if f.name == "current_pack")
    per_motor_filter = next(f for f in result.filters if f.name == "current_per_motor")
    assert pack_filter.evaluated is True
    assert per_motor_filter.evaluated is False  # no ESC/motor limit configured


def test_motor_count_scales_thrust_at_vt():
    single = evaluate_candidate(
        CandidateSpec(motor=make_motor(), prop=make_prop(), motor_count=1),
        make_requirement(),
        make_battery(),
    )
    twin = evaluate_candidate(
        CandidateSpec(motor=make_motor(), prop=make_prop(), motor_count=2),
        make_requirement(),
        make_battery(),
    )
    if single.thrust_at_vt_n is not None and twin.thrust_at_vt_n is not None:
        assert twin.thrust_at_vt_n > single.thrust_at_vt_n
        assert twin.thrust_at_vt_n < 2.0 * single.thrust_at_vt_n


def test_distance_filter_and_margin_reported_on_success():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    requirement = make_requirement(distance_allowed_m=200.0)  # generous allowance
    result = evaluate_candidate(spec, requirement, make_battery())
    distance_filter = next(f for f in result.filters if f.name == "distance")
    if math.isfinite(result.distance_m):
        assert result.distance_margin_pct is not None
        assert distance_filter.passed == (result.distance_m <= 200.0)


def test_distance_filter_fails_gracefully_when_infeasible():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    heavy_aircraft = AircraftConfig(
        mass_kg=500.0, wing_area_m2=0.65, span_m=2.0,
        cd0_ground=0.09, cl_ground=0.60, mu=0.10,
    )
    requirement = make_requirement(aircraft=heavy_aircraft, distance_allowed_m=60.0)
    result = evaluate_candidate(spec, requirement, make_battery())
    distance_filter = next(f for f in result.filters if f.name == "distance")
    assert not distance_filter.passed
    assert result.distance_m == math.inf
    assert result.distance_margin_pct is None
    assert not result.all_pass
    # Result must still carry every other filter, not just bail out.
    assert len(result.filters) == 8


def test_tip_mach_and_efficiency_values_are_sane():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    result = evaluate_candidate(spec, make_requirement(), make_battery())
    assert 0.0 < result.tip_mach_max < 1.0
    assert result.eta_prop_peak >= 0.0
    if result.eta_motor_at_vt is not None:
        assert 0.0 <= result.eta_motor_at_vt <= 1.0


def test_evaluate_candidates_runs_full_cross_product():
    motors = [make_motor(name="m1"), make_motor(name="m2", kv_rpm_per_v=1100.0)]
    props = [make_prop()]
    specs = [CandidateSpec(motor=m, prop=p) for m in motors for p in props]
    results = evaluate_candidates(specs, make_requirement(), make_battery())
    assert len(results) == len(specs)
    assert all(len(r.filters) == 8 for r in results)
