import math

import pytest

from propselect.core.atmosphere import G0, speed_of_sound
from propselect.core.battery import InternalResistanceBattery, OCVTable
from propselect.core.candidate import CandidateSpec
from propselect.core.cruise import (
    CruisePoint,
    CruisePointFailure,
    CruiseRequirement,
    cruise_envelope,
    evaluate_cruise_candidate,
    evaluate_cruise_candidates,
    solve_cruise_point,
)
from propselect.core.motor import MotorSpec
from propselect.core.propeller import PropellerDataTable
from propselect.core.takeoff import AircraftConfig, aerodynamic_drag_n


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
    defaults = dict(name="test-motor", kv_rpm_per_v=900.0, r_motor_ohm=0.05, i0_a=0.8, i_max_cont_a=60.0)
    defaults.update(overrides)
    return MotorSpec(**defaults)


def make_prop() -> PropellerDataTable:
    j = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    ct = [0.11, 0.105, 0.09, 0.07, 0.045, 0.015, -0.02]
    cp = [0.045, 0.044, 0.040, 0.033, 0.024, 0.015, 0.008]
    return PropellerDataTable(j, ct, cp, diameter_m=0.28, pitch_m=0.16, name="11x6")


def make_aircraft(**overrides) -> AircraftConfig:
    defaults = dict(mass_kg=8.0, wing_area_m2=0.65, span_m=2.0, cd0_ground=0.09, cl_ground=0.60, mu=0.10)
    defaults.update(overrides)
    return AircraftConfig(**defaults)


def make_requirement(**overrides) -> CruiseRequirement:
    defaults = dict(
        aircraft=make_aircraft(),
        v_cruise_m_s=15.0,
        cd0_cruise=0.035,
        rho_kg_m3=1.16,
        speed_of_sound_m_s=speed_of_sound(288.15),
    )
    defaults.update(overrides)
    return CruiseRequirement(**defaults)


def test_solve_cruise_point_matches_required_thrust():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    aircraft = make_aircraft()
    v = 12.0
    rho = 1.16
    cl = 2.0 * aircraft.mass_kg * G0 / (rho * v**2 * aircraft.wing_area_m2)
    cd = 0.035 + cl**2 / (math.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    t_req = aerodynamic_drag_n(rho, v, aircraft.wing_area_m2, cd)

    result = solve_cruise_point(v, t_req, rho, prop, motor, battery)
    assert isinstance(result, CruisePoint)
    assert result.thrust_n == pytest.approx(t_req, rel=1e-6)
    assert 0.0 < result.throttle_fraction <= 1.0


def test_solve_cruise_point_throttles_down_from_wot():
    # A modest thrust requirement should not need full throttle.
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    from propselect.core.operating_point import solve_operating_point

    v = 12.0
    wot = solve_operating_point(v, 1.16, prop, motor, battery)
    assert wot.success
    result = solve_cruise_point(v, 0.3 * wot.thrust_n, 1.16, prop, motor, battery)
    assert isinstance(result, CruisePoint)
    assert result.n_rev_s < wot.n_rev_s
    assert result.current_a < wot.current_a
    assert result.throttle_fraction < 1.0


def test_solve_cruise_point_infeasible_when_thrust_exceeds_wot():
    prop, motor, battery = make_prop(), make_motor(), make_battery()
    from propselect.core.operating_point import solve_operating_point

    v = 12.0
    wot = solve_operating_point(v, 1.16, prop, motor, battery)
    assert wot.success
    result = solve_cruise_point(v, 5.0 * wot.thrust_n, 1.16, prop, motor, battery)
    assert isinstance(result, CruisePointFailure)
    assert "exceeds max available thrust" in result.reason


def test_solve_cruise_point_rejects_nonpositive_thrust():
    result = solve_cruise_point(12.0, 0.0, 1.16, make_prop(), make_motor(), make_battery())
    assert isinstance(result, CruisePointFailure)


def test_evaluate_cruise_candidate_reports_all_filters_never_drops():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    result = evaluate_cruise_candidate(spec, make_requirement(), make_battery())
    filter_names = {f.name for f in result.filters}
    assert filter_names == {
        "cruise_solve",
        "tip_mach",
        "current_per_motor",
        "current_esc",
        "current_pack",
        "power",
        "endurance",
        "motor_efficiency",
        "prop_efficiency",
    }
    assert len(result.filters) == 9


def test_evaluate_cruise_candidate_endurance_and_range_are_consistent():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    result = evaluate_cruise_candidate(spec, make_requirement(), make_battery(capacity_ah=5.0))
    if result.endurance_s is not None:
        assert result.endurance_s > 0.0
        assert result.range_m == pytest.approx(result.endurance_s * 15.0)


def test_evaluate_cruise_candidate_endurance_filter_is_hard_and_gates_eligibility():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    result = evaluate_cruise_candidate(spec, make_requirement(), make_battery(capacity_ah=5.0))
    baseline_endurance = result.endurance_s
    assert baseline_endurance is not None

    requirement = make_requirement(endurance_required_s=baseline_endurance * 10.0)
    strict_result = evaluate_cruise_candidate(spec, requirement, make_battery(capacity_ah=5.0))
    endurance_filter = next(f for f in strict_result.filters if f.name == "endurance")
    assert endurance_filter.hard is True
    assert endurance_filter.passed is False
    assert strict_result.eligible is False


def test_evaluate_cruise_candidate_infeasible_solve_still_reports_every_filter():
    # A tiny, low-Kv motor can't produce enough thrust for this aircraft at
    # this speed -- the candidate must still come back with every filter
    # marked unevaluated/failed rather than raising or dropping fields.
    spec = CandidateSpec(motor=make_motor(kv_rpm_per_v=50.0), prop=make_prop())
    result = evaluate_cruise_candidate(spec, make_requirement(v_cruise_m_s=40.0), make_battery())
    assert isinstance(result.cruise_point, CruisePointFailure)
    assert len(result.filters) == 9
    assert result.eligible is False
    solve_filter = next(f for f in result.filters if f.name == "cruise_solve")
    assert solve_filter.passed is False


def test_cruise_envelope_sweeps_v_and_leaves_base_requirement_unchanged():
    spec = CandidateSpec(motor=make_motor(), prop=make_prop())
    requirement = make_requirement(v_cruise_m_s=15.0)
    v_values = [8.0, 12.0, 16.0, 20.0]
    envelope = cruise_envelope(spec, requirement, make_battery(), v_values)
    assert len(envelope) == len(v_values)
    assert requirement.v_cruise_m_s == 15.0  # frozen dataclass, sweep must not mutate it
    ok_points = [r for r in envelope if isinstance(r.cruise_point, CruisePoint)]
    assert any(ok_points)
    for r, v in zip(envelope, v_values):
        assert r.cruise_point.v_m_s == pytest.approx(v)


def test_evaluate_cruise_candidates_runs_full_cross_product():
    motors = [make_motor(name="m1"), make_motor(name="m2", kv_rpm_per_v=1100.0)]
    props = [make_prop()]
    specs = [CandidateSpec(motor=m, prop=p) for m in motors for p in props]
    results = evaluate_cruise_candidates(specs, make_requirement(), make_battery())
    assert len(results) == len(specs)
    assert all(len(r.filters) == 9 for r in results)
