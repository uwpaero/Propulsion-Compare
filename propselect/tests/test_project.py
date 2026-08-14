import pytest

from propselect.core.battery import InternalResistanceBattery, MeasuredCurveBattery
from propselect.core.candidate import RequirementSpec
from propselect.project import AircraftConfigModel, BatteryConfigModel, Project


def test_default_project_builds_valid_requirement_and_battery():
    project = Project()
    requirement = project.to_requirement()
    assert isinstance(requirement, RequirementSpec)
    assert requirement.v_t_m_s == pytest.approx(15.0)
    assert requirement.rho_kg_m3 > 0.0

    battery = project.battery.to_battery()
    assert isinstance(battery, InternalResistanceBattery)


def test_aircraft_config_model_aspect_ratio_is_derived():
    model = AircraftConfigModel(span_m=2.0, wing_area_m2=0.5)
    assert model.aspect_ratio == pytest.approx(2.0**2 / 0.5)

    # Changing span/area changes the derived AR without any separate field to update.
    model.span_m = 3.0
    assert model.aspect_ratio == pytest.approx(3.0**2 / 0.5)


def test_aircraft_config_model_to_aircraft_config_carries_derived_ar():
    model = AircraftConfigModel(span_m=2.0, wing_area_m2=0.5)
    aircraft = model.to_aircraft_config()
    assert aircraft.aspect_ratio == pytest.approx(model.aspect_ratio)


def test_loading_old_project_json_with_stale_aspect_ratio_key_does_not_error(tmp_path):
    # A project file saved by a previous schema version might still have an
    # explicit "aspect_ratio" key in its aircraft dict; loading must not choke
    # on an unexpected keyword argument now that it's a derived property.
    import json

    path = tmp_path / "old_project.json"
    path.write_text(json.dumps({"aircraft": {"span_m": 2.0, "wing_area_m2": 0.5, "aspect_ratio": 999.0}}))
    loaded = Project.load(path)
    assert loaded.aircraft.aspect_ratio == pytest.approx(2.0**2 / 0.5)


def test_measured_curve_battery_model_selection():
    battery_config = BatteryConfigModel(
        model="measured_curve",
        measured_current_a=[0, 10, 20, 30],
        measured_voltage_v=[16.8, 16.0, 15.2, 14.0],
        capacity_ah=3.0,
    )
    battery = battery_config.to_battery()
    assert isinstance(battery, MeasuredCurveBattery)
    result = battery.terminal_voltage(10.0)
    assert result.voltage_v == pytest.approx(16.0)


def test_field_temperature_overrides_isa_profile():
    aircraft = AircraftConfigModel(field_elevation_m=500.0, field_temperature_c=35.0)
    atm = aircraft.atmosphere()
    assert atm.temperature == pytest.approx(35.0 + 273.15)


def test_project_save_and_load_round_trip(tmp_path):
    project = Project(
        name="test-aircraft",
        aircraft=AircraftConfigModel(mass_kg=9.5, v_t_m_s=17.0, ground_effect=True),
        battery=BatteryConfigModel(series=6, parallel=2, capacity_ah=5.0),
        selected_motor_names=["AXI 2820/10"],
        selected_prop_names=["10x7"],
        tip_mach_limit=0.70,
    )
    path = tmp_path / "project.json"
    project.save(path)

    loaded = Project.load(path)
    assert loaded.name == "test-aircraft"
    assert loaded.aircraft.mass_kg == pytest.approx(9.5)
    assert loaded.aircraft.v_t_m_s == pytest.approx(17.0)
    assert loaded.aircraft.ground_effect is True
    assert loaded.battery.series == 6
    assert loaded.battery.parallel == 2
    assert loaded.battery.capacity_ah == pytest.approx(5.0)
    assert loaded.selected_motor_names == ["AXI 2820/10"]
    assert loaded.selected_prop_names == ["10x7"]
    assert loaded.tip_mach_limit == pytest.approx(0.70)


def test_loaded_project_requirement_matches_original():
    project = Project(aircraft=AircraftConfigModel(mass_kg=10.0, v_t_m_s=20.0))
    req_original = project.to_requirement()

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as d:
        path = _Path(d) / "p.json"
        project.save(path)
        loaded = Project.load(path)
    req_loaded = loaded.to_requirement()

    assert req_loaded.v_t_m_s == pytest.approx(req_original.v_t_m_s)
    assert req_loaded.aircraft.mass_kg == pytest.approx(req_original.aircraft.mass_kg)
    assert req_loaded.rho_kg_m3 == pytest.approx(req_original.rho_kg_m3)
