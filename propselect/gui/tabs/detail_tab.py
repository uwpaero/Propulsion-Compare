"""Tab 6 -- Detail: 2x3 plot grid, filter summary, PNG/CSV export."""

from __future__ import annotations

import csv
import math

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

from propselect.core.candidate import CandidateResult
from propselect.core.operating_point import OperatingPoint
from propselect.core.takeoff import closed_form_thrust_estimate
from propselect.gui.plots import MplCanvas


class DetailTab(QWidget):
    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.current_result: CandidateResult | None = None
        self.results: list[CandidateResult] = []

        self.candidate_combo = QComboBox()
        self.candidate_combo.setMinimumWidth(400)
        self.candidate_combo.addItem("(run a sweep in Tab 5, or click a row there)")
        self.candidate_combo.setEnabled(False)

        self.title_label = QLabel("Select a row in Tab 5 or use the picker above to see detail.")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")

        self.plot = MplCanvas(self, nrows=2, ncols=3, figsize=(13, 7))
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(160)

        self.export_png_btn = QPushButton("Export figure PNG...")
        self.export_csv_btn = QPushButton("Export swept data CSV...")
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

    def _candidate_label(self, result: CandidateResult) -> str:
        motor_count_desc = f" (x{result.spec.motor_count} motors)" if result.spec.motor_count > 1 else ""
        distance_desc = f"{result.distance_m:.1f} m" if math.isfinite(result.distance_m) else "DNF"
        status = "OK" if result.all_pass else ("eligible" if result.eligible else "NOT ELIGIBLE")
        return (
            f"{result.spec.motor.name} + {result.spec.prop.name}"
            f"{motor_count_desc} -- {distance_desc}, {status}"
        )

    def set_candidates(self, results: list[CandidateResult]) -> None:
        """Populate the picker with every candidate from the latest sweep."""
        self.results = results
        self.candidate_combo.blockSignals(True)
        self.candidate_combo.clear()
        if not results:
            self.candidate_combo.addItem("(run a sweep in Tab 5, or click a row there)")
            self.candidate_combo.setEnabled(False)
        else:
            for result in results:
                self.candidate_combo.addItem(self._candidate_label(result))
            self.candidate_combo.setEnabled(True)
        self.candidate_combo.blockSignals(False)

    def _on_combo_changed(self, index: int) -> None:
        if 0 <= index < len(self.results):
            self.show_candidate(self.results[index], _from_combo=True)

    def show_candidate(self, result: CandidateResult | None, _from_combo: bool = False) -> None:
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
            # Identity match (not ==): CandidateResult carries nested
            # dataclasses/lists, so equality would do an expensive and
            # fragile deep compare instead of just checking "is this the
            # same object Tab 5 handed us".
            index = next((i for i, r in enumerate(self.results) if r is result), None)
            if index is not None:
                self.candidate_combo.blockSignals(True)
                self.candidate_combo.setCurrentIndex(index)
                self.candidate_combo.blockSignals(False)
        self._draw_plots(result)
        self._write_summary(result)

    def _draw_plots(self, result: CandidateResult) -> None:
        self.plot.clear()
        axes = self.plot.axes  # 2x3 array

        diag = [p for p in result.diagnostic_operating_points if isinstance(p, OperatingPoint)]
        v = [p.v_m_s for p in diag]

        # 1. T(V) and T_req(V) overlaid -- the money plot.
        ax = axes[0][0]
        if diag:
            thrust = [p.thrust_n for p in diag]
            ax.plot(v, thrust, label="T(V)", color="#3b6ea5")
            requirement = self.state.project.to_requirement()
            aircraft = requirement.aircraft
            t_req = []
            for vi in v:
                q = 0.5 * requirement.rho_kg_m3 * vi**2
                drag = q * aircraft.wing_area_m2 * aircraft.cd0_ground
                lift = q * aircraft.wing_area_m2 * aircraft.cl_ground
                t_req.append(drag + aircraft.mu * (aircraft.mass_kg * 9.80665 - lift))
            ax.plot(v, t_req, "--", label="T_req(V) (steady)", color="#c0622d")
            ax.axvline(requirement.v_t_m_s, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Thrust")
        ax.set_xlabel("V [m/s]")
        ax.set_ylabel("N")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 2. F_net(V) with a zero line.
        ax = axes[0][1]
        gr = result.ground_roll
        if gr.v_grid_m_s and gr.drag_n:
            v_roll = gr.v_grid_m_s
            thrust_roll = [
                p.thrust_n if isinstance(p, OperatingPoint) else float("nan") for p in gr.operating_points
            ]
            fnet = [
                (t - d - self._mu_term(l, result)) if not math.isnan(t) else float("nan")
                for t, d, l in zip(thrust_roll, gr.drag_n, gr.lift_n)
            ]
            ax.plot(v_roll, fnet, color="#3b6ea5")
        ax.axhline(0.0, color="gray", linewidth=1)
        ax.set_title("Net accelerating force")
        ax.set_xlabel("V [m/s]")
        ax.set_ylabel("F_net [N]")
        ax.grid(True, alpha=0.3)

        # 3. I(V) with ESC/motor/pack limit lines.
        ax = axes[0][2]
        if diag:
            current = [p.current_a for p in diag]
            ax.plot(v, current, color="#3b6ea5", label="I/motor")
            if result.spec.motor_count > 1:
                pack_current = [p.current_pack_a for p in diag]
                ax.plot(v, pack_current, color="#2d8a4f", label="I_pack")
        motor = result.spec.motor
        if motor.i_max_cont_a:
            ax.axhline(motor.i_max_cont_a, color="#c0622d", linestyle="--", linewidth=1, label="motor cont")
        if result.spec.esc_current_cont_a:
            ax.axhline(
                result.spec.esc_current_cont_a, color="#8a4fae", linestyle="--", linewidth=1, label="ESC cont"
            )
        ax.set_title("Current draw")
        ax.set_xlabel("V [m/s]")
        ax.set_ylabel("I [A]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. n(V) and J(V) on twin axes.
        ax = axes[1][0]
        if diag:
            n_rpm = [p.n_rev_s * 60.0 for p in diag]
            j_vals = [p.j for p in diag]
            ax.plot(v, n_rpm, color="#3b6ea5", label="n [RPM]")
            ax2 = ax.twinx()
            ax2.plot(v, j_vals, color="#c0622d", label="J")
            ax2.set_ylabel("J", color="#c0622d")
            ax.set_ylabel("RPM", color="#3b6ea5")
        ax.set_title("Speed & advance ratio")
        ax.set_xlabel("V [m/s]")
        ax.grid(True, alpha=0.3)

        # 5. eta_motor(V) and eta_prop(V).
        ax = axes[1][1]
        if diag:
            eta_motor = [p.eta_motor for p in diag]
            eta_prop = [p.eta_prop for p in diag]
            ax.plot(v, eta_motor, label="eta_motor", color="#3b6ea5")
            ax.plot(v, eta_prop, label="eta_prop", color="#c0622d")
        ax.set_title("Efficiency")
        ax.set_xlabel("V [m/s]")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 6. Distance vs velocity, showing the integration.
        ax = axes[1][2]
        if gr.distance_profile_m and gr.v_grid_m_s:
            n_points = len(gr.distance_profile_m)
            ax.plot(gr.v_grid_m_s[:n_points], gr.distance_profile_m, color="#3b6ea5")
        ax.set_title("Ground-roll distance")
        ax.set_xlabel("V [m/s]")
        ax.set_ylabel("s [m]")
        ax.grid(True, alpha=0.3)

        self.plot.redraw()

    def _mu_term(self, lift_n: float, result: CandidateResult) -> float:
        aircraft = self.state.project.aircraft
        return aircraft.mu * (aircraft.mass_kg * 9.80665 - lift_n)

    def _write_summary(self, result: CandidateResult) -> None:
        lines = []
        lines.append(f"Distance: {result.distance_m:.2f} m" if math.isfinite(result.distance_m) else "Distance: DID NOT FINISH")
        if result.distance_margin_pct is not None:
            lines.append(f"Margin: {result.distance_margin_pct:+.1f}%")
        if not result.ground_roll.success and result.ground_roll.reason:
            lines.append(f"Failure reason: {result.ground_roll.reason}")
        lines.append("")
        lines.append(f"Eligible (hard constraints met): {'YES' if result.eligible else 'NO'}")
        lines.append("Filters:")
        for f in result.filters:
            status = "PASS" if f.passed else "FAIL"
            kind = "hard" if f.hard else "soft"
            lines.append(f"  [{status}] ({kind}) {f.name}: {f.detail}")
        if result.momentum_theory_warning:
            lines.append("")
            lines.append(f"WARNING: {result.momentum_theory_warning}")
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
        path, _ = QFileDialog.getSaveFileName(self, "Export swept data", "", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        diag = [p for p in self.current_result.diagnostic_operating_points if isinstance(p, OperatingPoint)]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["V_m_s", "n_rev_s", "J", "thrust_N", "current_A_per_motor", "current_A_pack", "voltage_V", "eta_motor", "eta_prop"]
            )
            for p in diag:
                writer.writerow(
                    [p.v_m_s, p.n_rev_s, p.j, p.thrust_n, p.current_a, p.current_pack_a, p.voltage_v, p.eta_motor, p.eta_prop]
                )
