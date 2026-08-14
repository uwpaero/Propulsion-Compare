"""Tab 2 -- Battery & ESC: model selection, OCV/measured-curve editor, V(I) plot."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from propselect.gui.plots import MplCanvas
from propselect.project import DEFAULT_OCV_SOC, DEFAULT_OCV_VOLTAGE_V

# Cell chemistry presets: per-cell OCV(SoC) table.
CHEMISTRY_PRESETS: dict[str, tuple[list[float], list[float]]] = {
    "LiPo (generic)": (DEFAULT_OCV_SOC, DEFAULT_OCV_VOLTAGE_V),
    "Li-ion (generic)": ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [3.00, 3.40, 3.55, 3.70, 3.90, 4.20]),
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


class BatteryTab(QWidget):
    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Internal resistance", "Measured curve"])

        self.chemistry_combo = QComboBox()
        self.chemistry_combo.addItems(list(CHEMISTRY_PRESETS.keys()))

        self.series = QSpinBox()
        self.series.setRange(1, 24)
        self.parallel = QSpinBox()
        self.parallel.setRange(1, 20)
        self.capacity = _spin(0.05, 100.0, 3, 0.1, "Ah")
        self.c_rate = _spin(0.0, 200.0, 1, 1.0, "C")
        self.r_cell = _spin(0.0, 1.0, 5, 0.001, "ohm/cell")
        self.initial_soc = _spin(0.0, 1.0, 3, 0.01)

        self.esc_r = _spin(0.0, 1.0, 5, 0.001, "ohm")
        self.esc_cont = _spin(0.0, 500.0, 1, 1.0, "A")
        self.esc_burst = _spin(0.0, 500.0, 1, 1.0, "A")

        self.ocv_table = QTableWidget(0, 2)
        self.ocv_table.setHorizontalHeaderLabels(["SoC", "Voltage (V/cell)"])
        self.ocv_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.measured_table = QTableWidget(0, 2)
        self.measured_table.setHorizontalHeaderLabels(["Current (A)", "Voltage (V, pack)"])
        self.measured_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.add_row_btn = QPushButton("Add row")
        self.remove_row_btn = QPushButton("Remove row")

        self.plot = MplCanvas(self)

        self._build_layout()
        self._wire_signals()
        self.load_from_project()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Model", self.model_combo)
        form.addRow("Chemistry preset", self.chemistry_combo)
        form.addRow("Series (S)", self.series)
        form.addRow("Parallel (P)", self.parallel)
        form.addRow("Capacity", self.capacity)
        form.addRow("C-rating limit (optional)", self.c_rate)
        form.addRow("Internal resistance / cell", self.r_cell)
        form.addRow("Initial SoC", self.initial_soc)
        form.addRow("ESC resistance", self.esc_r)
        form.addRow("ESC continuous current", self.esc_cont)
        form.addRow("ESC burst current", self.esc_burst)

        table_row = QHBoxLayout()
        table_row.addWidget(self.add_row_btn)
        table_row.addWidget(self.remove_row_btn)

        tables_box = QGroupBox("OCV table / measured curve")
        tables_layout = QVBoxLayout()
        tables_layout.addLayout(table_row)
        tables_layout.addWidget(self.ocv_table)
        tables_layout.addWidget(self.measured_table)
        tables_box.setLayout(tables_layout)

        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(tables_box)
        left_widget = QWidget()
        left_widget.setLayout(left)

        outer = QHBoxLayout(self)
        outer.addWidget(left_widget, 1)
        outer.addWidget(self.plot, 1)

    def _wire_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.chemistry_combo.currentTextChanged.connect(self._on_chemistry_preset)
        for widget in [
            self.series,
            self.parallel,
            self.capacity,
            self.c_rate,
            self.r_cell,
            self.initial_soc,
            self.esc_r,
            self.esc_cont,
            self.esc_burst,
        ]:
            widget.valueChanged.connect(self._on_change)
        self.add_row_btn.clicked.connect(self._add_row)
        self.remove_row_btn.clicked.connect(self._remove_row)
        self.ocv_table.itemChanged.connect(self._on_change)
        self.measured_table.itemChanged.connect(self._on_change)

    def _active_table(self) -> QTableWidget:
        return self.measured_table if self.model_combo.currentIndex() == 1 else self.ocv_table

    def _on_model_changed(self) -> None:
        is_measured = self.model_combo.currentIndex() == 1
        self.ocv_table.setVisible(not is_measured)
        self.measured_table.setVisible(is_measured)
        self.chemistry_combo.setEnabled(not is_measured)
        self.r_cell.setEnabled(not is_measured)
        self._on_change()

    def _on_chemistry_preset(self, text: str) -> None:
        preset = CHEMISTRY_PRESETS.get(text)
        if preset is not None:
            soc, voltage = preset
            self._fill_table(self.ocv_table, soc, voltage)
        self._on_change()

    def _fill_table(self, table: QTableWidget, xs: list[float], ys: list[float]) -> None:
        table.blockSignals(True)
        table.setRowCount(len(xs))
        for row, (x, y) in enumerate(zip(xs, ys)):
            table.setItem(row, 0, QTableWidgetItem(f"{x:g}"))
            table.setItem(row, 1, QTableWidgetItem(f"{y:g}"))
        table.blockSignals(False)

    def _add_row(self) -> None:
        table = self._active_table()
        table.blockSignals(True)
        table.insertRow(table.rowCount())
        table.blockSignals(False)

    def _remove_row(self) -> None:
        table = self._active_table()
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)
        self._on_change()

    def _read_table(self, table: QTableWidget) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in range(table.rowCount()):
            x_item = table.item(row, 0)
            y_item = table.item(row, 1)
            if x_item is None or y_item is None:
                continue
            try:
                xs.append(float(x_item.text()))
                ys.append(float(y_item.text()))
            except ValueError:
                continue
        return xs, ys

    def _on_change(self) -> None:
        self.save_to_project()
        self._update_plot()

    def _update_plot(self) -> None:
        battery_model = self.state.project.battery
        self.plot.clear()
        ax = self.plot.axes
        try:
            battery = battery_model.to_battery()
        except Exception as exc:
            ax.text(0.5, 0.5, f"Cannot build battery:\n{exc}", ha="center", va="center", wrap=True)
            self.plot.redraw()
            return

        i_max = self.esc_cont.value() if self.esc_cont.value() > 0 else max(battery_model.capacity_ah * 20, 10.0)
        currents = np.linspace(0.0, i_max, 100)
        voltages = [battery.terminal_voltage(i, self.initial_soc.value()).voltage_v for i in currents]
        ax.plot(currents, voltages, color="#3b6ea5")
        ax.set_xlabel("Current [A]")
        ax.set_ylabel("Terminal voltage [V]")
        ax.set_title("Battery V(I)")
        ax.grid(True, alpha=0.3)
        self.plot.redraw()

    def save_to_project(self) -> None:
        model = self.state.project.battery
        model.model = "measured_curve" if self.model_combo.currentIndex() == 1 else "internal_resistance"
        model.series = self.series.value()
        model.parallel = self.parallel.value()
        model.capacity_ah = self.capacity.value()
        model.c_rate_limit = self.c_rate.value() if self.c_rate.value() > 0 else None
        model.r_internal_per_cell_ohm = self.r_cell.value()
        model.initial_soc = self.initial_soc.value()
        model.esc_r_ohm = self.esc_r.value()
        model.esc_current_cont_a = self.esc_cont.value() if self.esc_cont.value() > 0 else None
        model.esc_current_burst_a = self.esc_burst.value() if self.esc_burst.value() > 0 else None

        soc, voltage = self._read_table(self.ocv_table)
        if len(soc) >= 2:
            model.ocv_soc = soc
            model.ocv_voltage_v = voltage
        current, voltage_m = self._read_table(self.measured_table)
        if len(current) >= 2:
            model.measured_current_a = current
            model.measured_voltage_v = voltage_m

    def load_from_project(self) -> None:
        model = self.state.project.battery
        widgets = [
            self.series,
            self.parallel,
            self.capacity,
            self.c_rate,
            self.r_cell,
            self.initial_soc,
            self.esc_r,
            self.esc_cont,
            self.esc_burst,
        ]
        for w in widgets:
            w.blockSignals(True)
        self.model_combo.blockSignals(True)

        self.model_combo.setCurrentIndex(1 if model.model == "measured_curve" else 0)
        self.series.setValue(model.series)
        self.parallel.setValue(model.parallel)
        self.capacity.setValue(model.capacity_ah)
        self.c_rate.setValue(model.c_rate_limit or 0.0)
        self.r_cell.setValue(model.r_internal_per_cell_ohm)
        self.initial_soc.setValue(model.initial_soc)
        self.esc_r.setValue(model.esc_r_ohm)
        self.esc_cont.setValue(model.esc_current_cont_a or 0.0)
        self.esc_burst.setValue(model.esc_current_burst_a or 0.0)

        self._fill_table(self.ocv_table, model.ocv_soc, model.ocv_voltage_v)
        if model.measured_current_a:
            self._fill_table(self.measured_table, model.measured_current_a, model.measured_voltage_v)

        for w in widgets:
            w.blockSignals(False)
        self.model_combo.blockSignals(False)
        self._on_model_changed()
