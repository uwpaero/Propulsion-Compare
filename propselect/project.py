"""Project persistence: a JSON-serializable description of every GUI input,
shared by the CLI and the GUI's File > Save/Open Project.

This module has no GUI dependency; it only converts between plain,
JSON-friendly config dataclasses and the runtime physics-core objects
(AircraftConfig, Battery, RequirementSpec).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from propselect.core.atmosphere import celsius_to_kelvin, isa_offset_atmosphere
from propselect.core.battery import Battery, InternalResistanceBattery, MeasuredCurveBattery, OCVTable
from propselect.core.candidate import RequirementSpec
from propselect.core.cruise import CruiseRequirement
from propselect.core.takeoff import AircraftConfig

DEFAULT_OCV_SOC = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_OCV_VOLTAGE_V = [3.30, 3.55, 3.70, 3.80, 3.95, 4.20]


@dataclass
class BatteryConfigModel:
    """Serializable battery configuration (Tab 2)."""

    model: str = "internal_resistance"  # or "measured_curve"
    series: int = 4
    parallel: int = 1
    r_internal_per_cell_ohm: float = 0.006
    capacity_ah: float = 3.0
    ocv_soc: list[float] = field(default_factory=lambda: list(DEFAULT_OCV_SOC))
    ocv_voltage_v: list[float] = field(default_factory=lambda: list(DEFAULT_OCV_VOLTAGE_V))
    measured_current_a: list[float] = field(default_factory=list)
    measured_voltage_v: list[float] = field(default_factory=list)
    esc_r_ohm: float = 0.0
    esc_current_cont_a: float | None = None
    esc_current_burst_a: float | None = None
    c_rate_limit: float | None = None
    initial_soc: float = 1.0

    def to_battery(self) -> Battery:
        if self.model == "measured_curve":
            return MeasuredCurveBattery(
                current_table_a=self.measured_current_a,
                voltage_table_v=self.measured_voltage_v,
                capacity_ah=self.capacity_ah,
            )
        table = OCVTable(self.ocv_soc, self.ocv_voltage_v)
        return InternalResistanceBattery(
            ocv_table=table,
            series=self.series,
            parallel=self.parallel,
            r_internal_per_cell_ohm=self.r_internal_per_cell_ohm,
            capacity_ah=self.capacity_ah,
        )


@dataclass
class AircraftConfigModel:
    """Serializable aircraft + requirement + environment configuration (Tab 1)."""

    mass_kg: float = 8.0
    wing_area_m2: float = 0.65
    span_m: float = 2.0
    cd0_ground: float = 0.09
    cl_ground: float = 0.60
    mu: float = 0.10
    wheel_height_m: float = 0.05
    oswald_efficiency: float = 0.8
    ground_effect: bool = False
    v_t_m_s: float = 15.0
    distance_allowed_m: float = 30.5
    field_elevation_m: float = 0.0
    field_temperature_c: float | None = None
    v_cruise_m_s: float = 20.0
    cd0_cruise: float = 0.035
    endurance_required_s: float | None = None

    @property
    def aspect_ratio(self) -> float:
        """AR = b^2 / S -- the exact definition, always derived from span and area."""
        return self.span_m**2 / self.wing_area_m2

    def to_aircraft_config(self) -> AircraftConfig:
        return AircraftConfig(
            mass_kg=self.mass_kg,
            wing_area_m2=self.wing_area_m2,
            span_m=self.span_m,
            cd0_ground=self.cd0_ground,
            cl_ground=self.cl_ground,
            mu=self.mu,
            wheel_height_m=self.wheel_height_m,
            oswald_efficiency=self.oswald_efficiency,
            ground_effect=self.ground_effect,
        )

    def atmosphere(self):
        actual_t_k = (
            celsius_to_kelvin(self.field_temperature_c)
            if self.field_temperature_c is not None
            else None
        )
        return isa_offset_atmosphere(self.field_elevation_m, actual_temperature_k=actual_t_k)


@dataclass
class Project:
    """The full set of GUI inputs, persisted to a JSON project file."""

    name: str = "untitled"
    aircraft: AircraftConfigModel = field(default_factory=AircraftConfigModel)
    battery: BatteryConfigModel = field(default_factory=BatteryConfigModel)
    motor_counts: list[int] = field(default_factory=lambda: [1])
    motor_library_path: str | None = None
    prop_library_paths: list[str] = field(default_factory=list)
    selected_motor_names: list[str] = field(default_factory=list)
    selected_prop_names: list[str] = field(default_factory=list)
    tip_mach_limit: float = 0.75
    motor_eta_threshold: float = 0.75
    power_limit_w: float | None = None
    dv_m_s: float = 0.1
    # Project-scoped snapshot of the motor/prop library (add/edit/delete in
    # the GUI persists here). ``None`` means "not yet saved with a library
    # snapshot" -- new/legacy projects fall back to the bundled defaults.
    motors: list[dict] | None = None
    props: list[dict] | None = None

    def to_requirement(self) -> RequirementSpec:
        atm = self.aircraft.atmosphere()
        return RequirementSpec(
            aircraft=self.aircraft.to_aircraft_config(),
            v_t_m_s=self.aircraft.v_t_m_s,
            distance_allowed_m=self.aircraft.distance_allowed_m,
            rho_kg_m3=atm.density,
            speed_of_sound_m_s=atm.speed_of_sound,
            tip_mach_limit=self.tip_mach_limit,
            motor_eta_threshold=self.motor_eta_threshold,
            power_limit_w=self.power_limit_w,
            c_rate_limit=self.battery.c_rate_limit,
            dv_m_s=self.dv_m_s,
            initial_soc=self.battery.initial_soc,
        )

    def to_cruise_requirement(self) -> CruiseRequirement:
        atm = self.aircraft.atmosphere()
        return CruiseRequirement(
            aircraft=self.aircraft.to_aircraft_config(),
            v_cruise_m_s=self.aircraft.v_cruise_m_s,
            cd0_cruise=self.aircraft.cd0_cruise,
            rho_kg_m3=atm.density,
            speed_of_sound_m_s=atm.speed_of_sound,
            tip_mach_limit=self.tip_mach_limit,
            motor_eta_threshold=self.motor_eta_threshold,
            power_limit_w=self.power_limit_w,
            c_rate_limit=self.battery.c_rate_limit,
            endurance_required_s=self.aircraft.endurance_required_s,
            initial_soc=self.battery.initial_soc,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        data = json.loads(Path(path).read_text())
        return _project_from_dict(data)


def _filter_known_fields(cls, data: dict) -> dict:
    """Drop keys that aren't constructor fields of ``cls``.

    Keeps old project files loadable across schema changes (e.g. a field
    that became a derived property, like AircraftConfigModel.aspect_ratio)
    instead of raising on an unexpected keyword argument.
    """
    valid_names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid_names}


def _project_from_dict(data: dict) -> Project:
    aircraft_data = _filter_known_fields(AircraftConfigModel, data.get("aircraft", {}))
    battery_data = _filter_known_fields(BatteryConfigModel, data.get("battery", {}))
    return Project(
        name=data.get("name", "untitled"),
        aircraft=AircraftConfigModel(**aircraft_data),
        battery=BatteryConfigModel(**battery_data),
        motor_counts=data.get("motor_counts", [1]),
        motor_library_path=data.get("motor_library_path"),
        prop_library_paths=data.get("prop_library_paths", []),
        selected_motor_names=data.get("selected_motor_names", []),
        selected_prop_names=data.get("selected_prop_names", []),
        tip_mach_limit=data.get("tip_mach_limit", 0.75),
        motor_eta_threshold=data.get("motor_eta_threshold", 0.75),
        power_limit_w=data.get("power_limit_w"),
        dv_m_s=data.get("dv_m_s", 0.1),
        motors=data.get("motors"),
        props=data.get("props"),
    )
