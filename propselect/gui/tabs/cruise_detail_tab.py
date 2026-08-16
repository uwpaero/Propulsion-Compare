"""Tab 9 -- Cruise Detail: envelope sweep for one candidate.

Cruise is a single equilibrium point at one airspeed, not a time
integration -- so unlike Tab 6 (which plots one takeoff roll's own
0->V_t history), this plots how the equilibrium itself shifts across a
speed range around the project's cruise velocity (using
``cruise_envelope``), with the actual configured cruise point marked.
"""

from __future__ import annotations

import csv
import math

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from propselect.core.cruise import CruiseCandidateResult, CruisePoint, cruise_envelope
from propselect.core.operating_point import solve_operating_point
from propselect.gui.plots import MplCanvas

# Envelope sweep span around the configured cruise velocity, and point count.
ENVELOPE_V_MIN_FACTOR = 0.5
ENVELOPE_V_MAX_FACTOR = 1.5
ENVELOPE_POINTS = 25


class CruiseDetailTab(QWidget):
    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.current_result: CruiseCandidateResult | None = None
        self.results: list[CruiseCandidateResult] = []
        self._envelope: list[CruiseCandidateResult] = []

        self.candidate_combo = QComboBox()
        self.candidate_combo.setMinimumWidth(400)
        self.candidate_combo.addItem("(run a sweep in Tab 8, or click a row there)")
        self.candidate_combo.setEnabled(False)

        self.title_label = QLabel("Select a row in Tab 8 or use the picker above to see detail.")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")

        self.plot = MplCanvas(self, nrows=2, ncols=2, figsize=(11, 7))
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(160)

        self.export_png_btn = QPushButton("Export figure PNG...")
        self.export_csv_btn = QPushButton("Export envelope CSV...")
        self.export_png_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)

        self._build_layout()
        self._wire_signals()

    def _build_layout(self) -> None:
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Candidate:"))
        picker_row.addWidget(self.candidate_combo, 1)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.export_png_btn)
        btn_row.addWidget(self.export_csv_btn)
        btn_row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.addLayout(picker_row)
        outer.addWidget(self.title_label)
        outer.addWidget(self.plot, 3)
        outer.addLayout(btn_row)
        outer.addWidget(self.summary_text, 1)

    def _wire_signals(self) -> None:
        self.export_png_btn.clicked.connect(self._export_png)
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.candidate_combo.currentIndexChanged.connect(self._on_combo_changed)

    def _candidate_label(self, result: CruiseCandidateResult) -> str:
        motor_count_desc = f" (x{result.spec.motor_count} motors)" if result.spec.motor_count > 1 else ""
        status = "OK" if result.all_pass else ("eligible" if result.eligible else "NOT ELIGIBLE")
        throttle_desc = (
            f"{result.throttle_fraction * 100:.0f}% throttle"
            if result.throttle_fraction is not None
            else "infeasible"
        )
        return (
            f"{result.spec.motor.name} + {result.spec.prop.name}"
            f"{motor_count_desc} -- {throttle_desc}, {status}"
        )

    def set_candidates(self, results: list[CruiseCandidateResult]) -> None:
        """Populate the picker with every candidate from the latest sweep."""
        self.results = results
        self.candidate_combo.blockSignals(True)
        self.candidate_combo.clear()
        if not results:
            self.candidate_combo.addItem("(run a sweep in Tab 8, or click a row there)")
            self.candidate_combo.setEnabled(False)
        else:
            for result in results:
                self.candidate_combo.addItem(self._candidate_label(result))
            self.candidate_combo.setEnabled(True)
        self.candidate_combo.blockSignals(False)

    def _on_combo_changed(self, index: int) -> None:
        if 0 <= index < len(self.results):
            self.show_candidate(self.results[index], _from_combo=True)

    def show_candidate(self, result: CruiseCandidateResult | None, _from_combo: bool = False) -> None:
        self.current_result = result
        if result is None:
            return
        self.export_png_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        motor_count_desc = f"  (x{result.spec.motor_count} motors)" if result.spec.motor_count > 1 else ""
        self.title_label.setText(
            f"{result.spec.motor.name}  +  {result.spec.prop.name}{motor_count_desc}"
        )
        if not _from_combo:
            index = next((i for i, r in enumerate(self.results) if r is result), None)
            if index is not None:
                self.candidate_combo.blockSignals(True)
                self.candidate_combo.setCurrentIndex(index)
                self.candidate_combo.blockSignals(False)
        self._compute_envelope(result)
        self._draw_plots(result)
        self._write_summary(result)

    def _compute_envelope(self, result: CruiseCandidateResult) -> None:
        v_cruise = result.cruise_point.v_m_s
        v_lo = max(0.1, ENVELOPE_V_MIN_FACTOR * v_cruise)
        v_hi = ENVELOPE_V_MAX_FACTOR * v_cruise
        v_values = list(np.linspace(v_lo, v_hi, ENVELOPE_POINTS))
        requirement = self.state.project.to_cruise_requirement()
        battery = self.state.project.battery.to_battery()
        self._envelope = cruise_envelope(result.spec, requirement, battery, v_values)

    def _draw_plots(self, result: CruiseCandidateResult) -> None:
        self.plot.clear()
        axes = self.plot.axes  # 2x2

        envelope = self._envelope
        v_all = [r.cruise_point.v_m_s for r in envelope]
        ok_mask = [isinstance(r.cruise_point, CruisePoint) for r in envelope]

        # Full-throttle thrust available at each V, for reference against
        # the required-thrust curve every candidate already computes.
        spec, battery = result.spec, self.state.project.battery.to_battery()
        requirement = self.state.project.to_cruise_requirement()
        wot_thrust = [
            (
                wt.thrust_n
                if (wt := solve_operating_point(
                    v, requirement.rho_kg_m3, spec.prop, spec.motor, battery,
                    r_esc_ohm=spec.r_esc_ohm, motor_count=spec.motor_count,
                )).success
                else float("nan")
            )
            for v in v_all
        ]
        t_req = [r.thrust_required_n for r in envelope]

        # 1. Thrust available (WOT) vs required, with the cruise point marked.
        ax = axes[0][0]
        ax.plot(v_all, wot_thrust, color="#3b6ea5", label="T available (WOT)")
        ax.plot(v_all, t_req, "--", color="#c0622d", label="T required (drag)")
        if isinstance(result.cruise_point, CruisePoint):
            ax.plot(result.cruise_point.v_m_s, result.thrust_required_n, "o", color="#2d8a4f", markersize=8, label="cruise point")
        ax.set_title("Thrust available vs required")
        ax.set_xlabel("V [m/s]")
        ax.set_ylabel("N")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 2. Throttle fraction vs V.
        ax = axes[0][1]
        throttle = [r.throttle_fraction if ok else float("nan") for r, ok in zip(envelope, ok_mask)]
        ax.plot(v_all, [t * 100 if t is not None else float("nan") for t in throttle], color="#3b6ea5")
        ax.axhline(100.0, color="gray", linestyle=":", linewidth=1)
        if isinstance(result.cruise_point, CruisePoint):
            ax.axvline(result.cruise_point.v_m_s, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Throttle")
        ax.set_xlabel("V [m/s]")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3)

        # 3. Current vs V, with motor/ESC limits.
        ax = axes[1][0]
        current = [r.current_per_motor_a if ok else float("nan") for r, ok in zip(envelope, ok_mask)]
        ax.plot(v_all, current, color="#3b6ea5", label="I/motor")
        if spec.motor_count > 1:
            pack = [r.current_pack_a if ok else float("nan") for r, ok in zip(envelope, ok_mask)]
            ax.plot(v_all, pack, color="#2d8a4f", label="I_pack")
        if spec.motor.i_max_cont_a:
            ax.axhline(spec.motor.i_max_cont_a, color="#c0622d", linestyle="--", linewidth=1, label="motor cont")
        if spec.esc_current_cont_a:
            ax.axhline(spec.esc_current_cont_a, color="#8a4fae", linestyle="--", linewidth=1, label="ESC cont")
        ax.set_title("Current draw")
        ax.set_xlabel("V [m/s]")
        ax.set_ylabel("I [A]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. Efficiency vs V.
        ax = axes[1][1]
        eta_motor = [r.eta_motor if ok else float("nan") for r, ok in zip(envelope, ok_mask)]
        eta_prop = [r.eta_prop if ok else float("nan") for r, ok in zip(envelope, ok_mask)]
        ax.plot(v_all, eta_motor, color="#3b6ea5", label="eta_motor")
        ax.plot(v_all, eta_prop, color="#c0622d", label="eta_prop")
        ax.set_title("Efficiency")
        ax.set_xlabel("V [m/s]")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        self.plot.redraw()

    def _write_summary(self, result: CruiseCandidateResult) -> None:
        lines = []
        point = result.cruise_point
        if isinstance(point, CruisePoint):
            lines.append(f"Cruise V: {point.v_m_s:.2f} m/s   Throttle: {point.throttle_fraction * 100:.0f}%")
            lines.append(f"T_required: {result.thrust_required_n:.2f} N   C_L: {result.cl_cruise:.3f}   C_D: {result.cd_cruise:.4f}")
            if result.endurance_s is not None:
                lines.append(f"Endurance: {result.endurance_s / 60.0:.1f} min   Range: {result.range_m / 1000.0:.2f} km")
        else:
            lines.append(f"CRUISE INFEASIBLE at this velocity: {point.reason}")
        lines.append("")
        lines.append(f"Eligible (hard constraints met): {'YES' if result.eligible else 'NO'}")
        lines.append("Filters:")
        for f in result.filters:
            status = "PASS" if f.passed else "FAIL"
            kind = "hard" if f.hard else "soft"
            lines.append(f"  [{status}] ({kind}) {f.name}: {f.detail}")
        if result.is_low_confidence:
            lines.append("")
            lines.append("NOTE: this propeller uses the LOW CONFIDENCE parametric fallback model.")
        self.summary_text.setPlainText("\n".join(lines))

    def _export_png(self) -> None:
        if self.current_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", "", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.plot.save_png(path)

    def _export_csv(self) -> None:
        if self.current_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export envelope", "", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["V_m_s", "feasible", "thrust_required_N", "throttle_fraction", "current_A_per_motor", "current_A_pack", "eta_motor", "eta_prop", "endurance_s"]
            )
            for r in self._envelope:
                ok = isinstance(r.cruise_point, CruisePoint)
                writer.writerow(
                    [
                        r.cruise_point.v_m_s,
                        ok,
                        r.thrust_required_n,
                        r.throttle_fraction if ok else "",
                        r.current_per_motor_a if ok else "",
                        r.current_pack_a if ok else "",
                        r.eta_motor if ok else "",
                        r.eta_prop if ok else "",
                        r.endurance_s if ok else "",
                    ]
                )
