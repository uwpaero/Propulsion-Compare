"""Tab 5 -- Sweep & Results: cross-product sweep on a QThread, sortable table."""

from __future__ import annotations

import csv
import math

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from propselect.core.candidate import CandidateResult, CandidateSpec, evaluate_candidate

RESULT_COLUMNS = [
    "Motor",
    "Prop",
    "Motors",
    "Distance (m)",
    "Margin (%)",
    "Time (s)",
    "T@Vt (N)",
    "I/motor (A)",
    "I_pack (A)",
    "mAh",
    "M_tip",
    "V_pitch/V_t",
    "eta_mot@Vt",
    "eta_prop_pk",
    "Filters",
]

GREEN = QColor(200, 235, 200)
AMBER = QColor(250, 230, 170)
RED = QColor(245, 200, 200)


class _NumericItem(QTableWidgetItem):
    def __init__(self, value: float, text: str) -> None:
        super().__init__(text)
        self._value = value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class SweepWorker(QThread):
    progress = Signal(int, int)
    finished_with_results = Signal(list)
    failed = Signal(str)

    def __init__(self, specs: list[CandidateSpec], requirement, battery, parent=None) -> None:
        super().__init__(parent)
        self.specs = specs
        self.requirement = requirement
        self.battery = battery

    def run(self) -> None:
        try:
            results: list[CandidateResult] = []
            total = len(self.specs)
            for i, spec in enumerate(self.specs):
                results.append(evaluate_candidate(spec, self.requirement, self.battery))
                self.progress.emit(i + 1, total)
            self.finished_with_results.emit(results)
        except Exception as exc:  # keep the UI thread from ever seeing a bare crash
            self.failed.emit(str(exc))


class SweepResultsTab(QWidget):
    candidate_selected = Signal(object)  # CandidateResult
    sweep_finished = Signal(list)  # list[CandidateResult]

    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.results: list[CandidateResult] = []
        self._worker: SweepWorker | None = None

        self.run_btn = QPushButton("Run Sweep")
        self.export_btn = QPushButton("Export CSV...")
        self.export_btn.setEnabled(False)
        self.status_label = QLabel("No sweep run yet.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.table = QTableWidget(0, len(RESULT_COLUMNS))
        self.table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        # Per-cell pass/fail background colors (set below) would otherwise
        # wash out Qt's default selection highlight -- force a color that
        # stays visibly distinct from red/amber/green regardless of theme.
        self.table.setStyleSheet(
            "QTableWidget::item:selected { background-color: #1a73e8; color: white; }"
        )

        self._build_layout()
        self._wire_signals()

    def _build_layout(self) -> None:
        top_row = QHBoxLayout()
        top_row.addWidget(self.run_btn)
        top_row.addWidget(self.export_btn)
        top_row.addWidget(self.progress_bar, 1)

        hint_label = QLabel(
            "Click a row to view it in Tab 6 (Detail) -- or use the picker there. "
            "See Tab 7 (Comparison) for cross-candidate plots."
        )
        hint_label.setStyleSheet("color: gray; font-style: italic;")

        outer = QVBoxLayout(self)
        outer.addLayout(top_row)
        outer.addWidget(self.status_label)
        outer.addWidget(hint_label)
        outer.addWidget(self.table)

    def _wire_signals(self) -> None:
        self.run_btn.clicked.connect(self._run_sweep)
        self.export_btn.clicked.connect(self._export_csv)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

    def _build_specs(self) -> list[CandidateSpec]:
        motors = [m for m in self.state.motors if m.name in self.state.project.selected_motor_names]
        props = [p for name, p in self.state.props.items() if name in self.state.project.selected_prop_names]
        motor_counts = self.state.project.motor_counts or [1]
        specs = []
        for motor in motors:
            for prop in props:
                for count in motor_counts:
                    specs.append(
                        CandidateSpec(
                            motor=motor,
                            prop=prop,
                            r_esc_ohm=self.state.project.battery.esc_r_ohm,
                            esc_current_cont_a=self.state.project.battery.esc_current_cont_a,
                            motor_count=count,
                        )
                    )
        return specs

    def _run_sweep(self) -> None:
        specs = self._build_specs()
        if not specs:
            QMessageBox.warning(
                self,
                "Nothing to sweep",
                "Select at least one motor (Tab 3) and one propeller (Tab 4) first.",
            )
            return
        try:
            requirement = self.state.project.to_requirement()
            battery = self.state.project.battery.to_battery()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot start sweep", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Running {len(specs)} combinations...")

        self._worker = SweepWorker(specs, requirement, battery, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_with_results.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, completed: int, total: int) -> None:
        self.progress_bar.setValue(round(100 * completed / total) if total else 0)
        self.status_label.setText(f"Running... {completed}/{total}")

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Sweep failed.")
        QMessageBox.critical(self, "Sweep failed", message)

    def _on_finished(self, results: list[CandidateResult]) -> None:
        self.results = results
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(bool(results))
        n_eligible = sum(1 for r in results if r.eligible)
        n_optimal = sum(1 for r in results if r.all_pass)
        self.status_label.setText(
            f"Done: {len(results)} combinations, {n_eligible} eligible (hard constraints met), "
            f"{n_optimal} optimal (all filters met)."
        )
        self._populate_table(results)
        self.sweep_finished.emit(results)

    def _populate_table(self, results: list[CandidateResult]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(results))
        for row, result in enumerate(results):
            distance = result.distance_m
            distance_text = f"{distance:.2f}" if math.isfinite(distance) else "DNF"
            margin = result.distance_margin_pct
            margin_text = f"{margin:+.1f}" if margin is not None else "-"
            thrust_text = f"{result.thrust_at_vt_n:.2f}" if result.thrust_at_vt_n is not None else "-"
            v_pitch_text = f"{result.v_pitch_ratio:.2f}" if result.v_pitch_ratio is not None else "-"
            eta_mot_text = f"{result.eta_motor_at_vt:.3f}" if result.eta_motor_at_vt is not None else "-"
            hard_filters = [f for f in result.filters if f.hard]
            soft_filters = [f for f in result.filters if not f.hard]
            n_hard_pass = sum(1 for f in hard_filters if f.passed)
            n_soft_pass = sum(1 for f in soft_filters if f.passed)
            failing_hard = [f.name for f in hard_filters if not f.passed]
            failing_soft = [f.name for f in soft_filters if not f.passed]
            filters_text = f"{n_hard_pass}/{len(hard_filters)} hard, {n_soft_pass}/{len(soft_filters)} soft"
            if failing_hard:
                filters_text += f" (hard fail: {', '.join(failing_hard)})"
            if failing_soft:
                filters_text += f" (soft fail: {', '.join(failing_soft)})"
            if result.is_low_confidence:
                filters_text += "  [LOW CONFIDENCE]"

            values = [
                (result.spec.motor.name, result.spec.motor.name),
                (result.spec.prop.name, result.spec.prop.name),
                (f"{result.spec.motor_count}", result.spec.motor_count),
                (distance_text, distance if math.isfinite(distance) else float("inf")),
                (margin_text, margin if margin is not None else float("-inf")),
                (f"{result.time_s:.2f}" if math.isfinite(result.time_s) else "DNF", result.time_s),
                (thrust_text, result.thrust_at_vt_n or 0.0),
                (f"{result.current_max_per_motor_a:.2f}", result.current_max_per_motor_a),
                (f"{result.current_max_pack_a:.2f}", result.current_max_pack_a),
                (f"{result.capacity_used_mah:.1f}", result.capacity_used_mah),
                (f"{result.tip_mach_max:.3f}", result.tip_mach_max),
                (v_pitch_text, result.v_pitch_ratio or 0.0),
                (eta_mot_text, result.eta_motor_at_vt or 0.0),
                (f"{result.eta_prop_peak:.3f}", result.eta_prop_peak),
                (filters_text, n_hard_pass),
            ]

            if not result.eligible:
                color = RED  # a hard/disqualifying constraint failed
            elif result.all_pass:
                color = GREEN  # eligible and every soft objective met too
            else:
                color = AMBER  # eligible, but a soft objective (e.g. efficiency) missed

            for col, (text, sort_value) in enumerate(values):
                item = _NumericItem(sort_value, text) if isinstance(sort_value, (int, float)) else QTableWidgetItem(text)
                item.setBackground(color)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)

    def _on_selection_changed(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.results):
            # table rows are re-sorted; map back to the originating result via
            # the row's stored candidate rather than assuming index order.
            self.candidate_selected.emit(self._result_for_row(row))

    def _result_for_row(self, row: int) -> CandidateResult | None:
        motor_item = self.table.item(row, 0)
        prop_item = self.table.item(row, 1)
        motors_item = self.table.item(row, 2)
        if not (motor_item and prop_item and motors_item):
            return None
        for result in self.results:
            if (
                result.spec.motor.name == motor_item.text()
                and result.spec.prop.name == prop_item.text()
                and str(result.spec.motor_count) == motors_item.text()
            ):
                return result
        return None

    def select_candidate(self, result: CandidateResult) -> None:
        """Select the row matching ``result`` (e.g. from a Tab 7 plot click)."""
        for row in range(self.table.rowCount()):
            motor_item = self.table.item(row, 0)
            prop_item = self.table.item(row, 1)
            motors_item = self.table.item(row, 2)
            if not (motor_item and prop_item and motors_item):
                continue
            if (
                motor_item.text() == result.spec.motor.name
                and prop_item.text() == result.spec.prop.name
                and motors_item.text() == str(result.spec.motor_count)
            ):
                self.table.selectRow(row)
                self.table.scrollToItem(motor_item)
                return

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export results CSV", "", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(RESULT_COLUMNS)
            for row in range(self.table.rowCount()):
                writer.writerow(
                    [self.table.item(row, col).text() if self.table.item(row, col) else "" for col in range(len(RESULT_COLUMNS))]
                )
