"""Tab 1 -- Aircraft & Requirement: inputs plus a live closed-form estimate."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from propselect.core.takeoff import closed_form_thrust_estimate
from propselect.gui.units import AREA, LENGTH_LARGE, LENGTH_SMALL, MASS, TEMPERATURE, VELOCITY, UnitAwareSpinBox, UnitSystem

# Rolling-friction coefficient presets, by surface.
SURFACE_PRESETS: dict[str, float] = {
    "Pavement": 0.02,
    "Hard turf": 0.05,
    "Short grass": 0.08,
    "Mown grass": 0.10,
    "Tall grass": 0.15,
    "Custom": None,  # type: ignore[dict-item]
}


def _spin(minimum: float, maximum: float, decimals: int, step: float, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    if suffix:
        box.setSuffix(f" {suffix}")
    return box


class AircraftTab(QWidget):
    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state

        # Unit-aware fields: displayed in whichever unit system is active,
        # always readable/writable in SI via si_value()/set_si_value().
        self.mass = UnitAwareSpinBox(MASS, 0.01, 2000.0, si_decimals=3, si_step=0.1)
        self.wing_area = UnitAwareSpinBox(AREA, 0.001, 500.0, si_decimals=4, si_step=0.01)
        self.span = UnitAwareSpinBox(LENGTH_LARGE, 0.01, 100.0, si_decimals=3, si_step=0.05)
        self.v_t = UnitAwareSpinBox(VELOCITY, 0.1, 200.0, si_decimals=2, si_step=0.5)
        self.distance_allowed = UnitAwareSpinBox(LENGTH_LARGE, 0.1, 10000.0, si_decimals=2, si_step=1.0)
        self.wheel_height = UnitAwareSpinBox(LENGTH_SMALL, 0.0, 10.0, si_decimals=4, si_step=0.01)
        self.elevation = UnitAwareSpinBox(LENGTH_LARGE, -500.0, 6000.0, si_decimals=1, si_step=10.0)
        self.temp_c = UnitAwareSpinBox(TEMPERATURE, -60.0, 60.0, si_decimals=1, si_step=1.0)
        self._unit_aware_widgets = [
            self.mass,
            self.wing_area,
            self.span,
            self.v_t,
            self.distance_allowed,
            self.wheel_height,
            self.elevation,
            self.temp_c,
        ]

        # Dimensionless or angle fields: no SI/US distinction.
        # Aspect ratio is AR = b^2/S -- the exact definition, not an
        # independent input -- so it's a live read-only display, not an
        # editable field, and can never drift out of sync with span/area.
        self.aspect_ratio_label = QLabel()
        self.cd0 = _spin(0.0, 5.0, 4, 0.001, "(C_D0)")
        self.cl = _spin(-2.0, 5.0, 4, 0.001, "(C_L)")
        self.mu = _spin(0.0, 2.0, 4, 0.005)
        self.oswald_efficiency = _spin(0.1, 1.0, 3, 0.01, "(e)")

        self.surface_combo = QComboBox()
        self.surface_combo.addItems(list(SURFACE_PRESETS.keys()))

        self.ground_effect = QCheckBox("apply ground-effect correction (off by default)")

        self.temp_override_enabled = QCheckBox("override ISA temperature")
        self.temp_c.setEnabled(False)

        # Filter thresholds (RequirementSpec / Project top-level fields, not
        # part of AircraftConfig -- these gate the Tab 5 pass/fail filters,
        # not the physics itself).
        self.tip_mach_limit = _spin(0.1, 1.0, 3, 0.01, "(M_tip)")
        self.motor_eta_threshold = _spin(0.0, 1.0, 3, 0.01, "(eta_motor)")
        self.power_limit_enabled = QCheckBox("limit total electrical power")
        self.power_limit_w = _spin(0.0, 1_000_000.0, 1, 10.0, "W")
        self.power_limit_w.setEnabled(False)

        self._build_layout()
        self._wire_signals()
        self.load_from_project()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Mass", self.mass)
        form.addRow("Wing area (S)", self.wing_area)
        form.addRow("Span (b)", self.span)
        form.addRow("Aspect ratio (AR = b^2/S)", self.aspect_ratio_label)
        form.addRow("Ground-roll C_D", self.cd0)
        form.addRow("Ground-roll C_L", self.cl)
        form.addRow("Surface preset", self.surface_combo)
        form.addRow("Friction coeff. (mu)", self.mu)
        form.addRow("Target velocity V_t", self.v_t)
        form.addRow("Allowed distance", self.distance_allowed)
        form.addRow("Wheel height", self.wheel_height)

        ge_row = QHBoxLayout()
        ge_row.addWidget(self.ground_effect)
        ge_row.addWidget(self.oswald_efficiency)
        form.addRow("Ground effect", ge_row)

        form.addRow("Field elevation", self.elevation)
        temp_row = QHBoxLayout()
        temp_row.addWidget(self.temp_override_enabled)
        temp_row.addWidget(self.temp_c)
        form.addRow("Field temperature", temp_row)

        filters_box = QGroupBox("Filter thresholds (Tab 5 pass/fail gates)")
        filters_form = QFormLayout()
        filters_form.addRow("Tip Mach limit", self.tip_mach_limit)
        filters_form.addRow("Min motor efficiency @ V_t", self.motor_eta_threshold)
        power_row = QHBoxLayout()
        power_row.addWidget(self.power_limit_enabled)
        power_row.addWidget(self.power_limit_w)
        filters_form.addRow("Power limit", power_row)
        filters_box.setLayout(filters_form)

        left_layout = QVBoxLayout()
        left_layout.addLayout(form)
        left_layout.addWidget(filters_box)
        left = QWidget()
        left.setLayout(left_layout)

        self.summary_box = QGroupBox("Live thrust-required estimate (closed form)")
        summary_layout = QVBoxLayout()
        self.v_eff_label = QLabel()
        self.q_bar_label = QLabel()
        self.t_req_label = QLabel()
        self.t_req_label.setStyleSheet("font-weight: bold; font-size: 14pt;")

        self.inertia_bar, self.inertia_label = self._make_term_row(summary_layout, "Inertia (m*Vt^2/2s)")
        self.drag_bar, self.drag_label = self._make_term_row(summary_layout, "Drag (q*S*Cd)")
        self.friction_bar, self.friction_label = self._make_term_row(summary_layout, "Friction (mu*(mg-L))")

        summary_layout.addWidget(self.v_eff_label)
        summary_layout.addWidget(self.q_bar_label)
        summary_layout.addWidget(self.t_req_label)
        self.summary_box.setLayout(summary_layout)

        outer = QHBoxLayout(self)
        outer.addWidget(left, 2)
        outer.addWidget(self.summary_box, 1)

    def _make_term_row(self, layout: QVBoxLayout, title: str) -> tuple[QProgressBar, QLabel]:
        label = QLabel(title)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(True)
        layout.addWidget(label)
        layout.addWidget(bar)
        return bar, label

    def _wire_signals(self) -> None:
        for widget in [
            *self._unit_aware_widgets,
            self.cd0,
            self.cl,
            self.mu,
            self.oswald_efficiency,
            self.tip_mach_limit,
            self.motor_eta_threshold,
            self.power_limit_w,
        ]:
            widget.valueChanged.connect(self._on_change)
        self.surface_combo.currentTextChanged.connect(self._on_surface_preset)
        self.ground_effect.toggled.connect(self._on_change)
        self.temp_override_enabled.toggled.connect(self._on_temp_toggle)
        self.power_limit_enabled.toggled.connect(self._on_power_limit_toggle)

    def set_unit_system(self, system: UnitSystem) -> None:
        """Switch every unit-aware field's display unit, converting values in place."""
        for widget in self._unit_aware_widgets:
            widget.blockSignals(True)
            widget.set_unit_system(system)
            widget.blockSignals(False)

    def _on_surface_preset(self, text: str) -> None:
        value = SURFACE_PRESETS.get(text)
        if value is not None:
            self.mu.blockSignals(True)
            self.mu.setValue(value)
            self.mu.blockSignals(False)
        self._on_change()

    def _on_temp_toggle(self, checked: bool) -> None:
        self.temp_c.setEnabled(checked)
        self._on_change()

    def _on_power_limit_toggle(self, checked: bool) -> None:
        self.power_limit_w.setEnabled(checked)
        self._on_change()

    def _on_change(self) -> None:
        self.save_to_project()
        self._update_summary()

    def _update_summary(self) -> None:
        model = self.state.project.aircraft
        self.aspect_ratio_label.setText(f"{model.aspect_ratio:.2f}")
        try:
            atm = model.atmosphere()
        except Exception:
            return
        estimate = closed_form_thrust_estimate(
            mass_kg=model.mass_kg,
            wing_area_m2=model.wing_area_m2,
            v_t_m_s=model.v_t_m_s,
            distance_allowed_m=model.distance_allowed_m,
            cd=model.cd0_ground,
            cl=model.cl_ground,
            mu=model.mu,
            rho_kg_m3=atm.density,
        )
        self.v_eff_label.setText(f"V_eff = {estimate.v_eff_m_s:.2f} m/s")
        self.q_bar_label.setText(f"q_bar = {estimate.q_bar_pa:.1f} Pa   (rho = {atm.density:.4f} kg/m^3)")
        self.t_req_label.setText(f"T_req ~= {estimate.thrust_required_n:.2f} N")

        self.inertia_bar.setValue(max(0, min(100, round(estimate.inertia_pct))))
        self.inertia_label.setText(f"Inertia: {estimate.inertia_n:.2f} N ({estimate.inertia_pct:.1f}%)")
        self.drag_bar.setValue(max(0, min(100, round(estimate.drag_pct))))
        self.drag_label.setText(f"Drag: {estimate.drag_n:.2f} N ({estimate.drag_pct:.1f}%)")
        self.friction_bar.setValue(max(0, min(100, round(estimate.friction_pct))))
        self.friction_label.setText(f"Friction: {estimate.friction_n:.2f} N ({estimate.friction_pct:.1f}%)")

    def save_to_project(self) -> None:
        model = self.state.project.aircraft
        model.mass_kg = self.mass.si_value()
        model.wing_area_m2 = self.wing_area.si_value()
        model.span_m = self.span.si_value()
        model.cd0_ground = self.cd0.value()
        model.cl_ground = self.cl.value()
        model.mu = self.mu.value()
        model.v_t_m_s = self.v_t.si_value()
        model.distance_allowed_m = self.distance_allowed.si_value()
        model.wheel_height_m = self.wheel_height.si_value()
        model.ground_effect = self.ground_effect.isChecked()
        model.oswald_efficiency = self.oswald_efficiency.value()
        model.field_elevation_m = self.elevation.si_value()
        model.field_temperature_c = (
            self.temp_c.si_value() if self.temp_override_enabled.isChecked() else None
        )

        project = self.state.project
        project.tip_mach_limit = self.tip_mach_limit.value()
        project.motor_eta_threshold = self.motor_eta_threshold.value()
        project.power_limit_w = (
            self.power_limit_w.value() if self.power_limit_enabled.isChecked() else None
        )

    def load_from_project(self) -> None:
        model = self.state.project.aircraft
        widgets = [
            *self._unit_aware_widgets,
            self.cd0,
            self.cl,
            self.mu,
            self.oswald_efficiency,
            self.tip_mach_limit,
            self.motor_eta_threshold,
            self.power_limit_w,
        ]
        for w in widgets:
            w.blockSignals(True)
        self.mass.set_si_value(model.mass_kg)
        self.wing_area.set_si_value(model.wing_area_m2)
        self.span.set_si_value(model.span_m)
        self.cd0.setValue(model.cd0_ground)
        self.cl.setValue(model.cl_ground)
        self.mu.setValue(model.mu)
        self.v_t.set_si_value(model.v_t_m_s)
        self.distance_allowed.set_si_value(model.distance_allowed_m)
        self.wheel_height.set_si_value(model.wheel_height_m)
        self.ground_effect.setChecked(model.ground_effect)
        self.oswald_efficiency.setValue(model.oswald_efficiency)
        self.elevation.set_si_value(model.field_elevation_m)
        self.temp_override_enabled.setChecked(model.field_temperature_c is not None)
        self.temp_c.set_si_value(model.field_temperature_c or 15.0)
        self.temp_c.setEnabled(model.field_temperature_c is not None)

        project = self.state.project
        self.tip_mach_limit.setValue(project.tip_mach_limit)
        self.motor_eta_threshold.setValue(project.motor_eta_threshold)
        self.power_limit_enabled.setChecked(project.power_limit_w is not None)
        self.power_limit_w.setValue(project.power_limit_w or 0.0)
        self.power_limit_w.setEnabled(project.power_limit_w is not None)
        for w in widgets:
            w.blockSignals(False)
        self._update_summary()
