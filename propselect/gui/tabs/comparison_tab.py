"""Tab 7 -- Comparison: cross-candidate plots for the passing results of the
latest sweep. Distinct from Tab 6 (Detail, one candidate's own V-sweep) --
this tab compares already-computed candidates against each other on the
metrics that actually decide which one to build.

Bars/points are not text-labeled (with more than a handful of candidates,
per-item labels overlap into an unreadable mess) -- instead, every bar and
point is clickable: it emits ``candidate_clicked``, which the main window
wires to jump to that combination's row in Tab 5.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from propselect.core.candidate import CandidateResult
from propselect.gui.plots import MplCanvas


class ComparisonTab(QWidget):
    candidate_clicked = Signal(object)  # CandidateResult

    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.results: list[CandidateResult] = []
        self._axes_candidates: dict[Axes, list[CandidateResult]] = {}

        self.status_label = QLabel("Run a sweep in Tab 5 to populate these plots.")
        self.hint_label = QLabel(
            "Click a bar or point to jump to that combination in Tab 5 (Sweep & Results)."
        )
        self.hint_label.setStyleSheet("color: gray; font-style: italic;")
        self.plot = MplCanvas(self, nrows=2, ncols=3, figsize=(15, 8))
        self.plot.canvas.mpl_connect("pick_event", self._on_pick)

        outer = QVBoxLayout(self)
        outer.addWidget(self.status_label)
        outer.addWidget(self.hint_label)
        outer.addWidget(self.plot)

    def set_candidates(self, results: list[CandidateResult]) -> None:
        self.results = results
        n_eligible = sum(1 for r in results if r.eligible)
        self.status_label.setText(
            f"{len(results)} candidates evaluated, {n_eligible} eligible (hard constraints met) "
            "-- only eligible candidates are plotted below, so soft-objective tradeoffs "
            "(e.g. motor efficiency) among legal builds are visible."
        )
        self._update_plots(results)

    def _on_pick(self, event) -> None:
        ax = event.artist.axes
        candidates = self._axes_candidates.get(ax)
        if not candidates:
            return
        if hasattr(event, "ind") and len(event.ind) > 0:
            idx = int(event.ind[0])
        else:
            try:
                idx = list(ax.patches).index(event.artist)
            except ValueError:
                return
        if 0 <= idx < len(candidates):
            self.candidate_clicked.emit(candidates[idx])

    def _bar_rank(self, ax: Axes, ranked: list[CandidateResult], values: list[float], color: str) -> None:
        """A horizontal bar chart ranked best-first, with #1..#N tick labels
        instead of full names (which overlap badly past a handful of bars)
        -- click a bar to identify it instead.
        """
        y_pos = list(range(len(ranked)))
        bars = ax.barh(y_pos, values, color=color, picker=True)
        for patch in bars.patches:
            patch.set_picker(True)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"#{i + 1}" for i in y_pos], fontsize=7)
        ax.invert_yaxis()
        self._axes_candidates[ax] = ranked

    def _update_plots(self, results: list[CandidateResult]) -> None:
        self.plot.clear()
        self._axes_candidates.clear()
        axes = self.plot.axes  # 2x3
        passing = [r for r in results if r.eligible]

        if not passing:
            for row in axes:
                for ax in row:
                    ax.text(
                        0.5, 0.5, "No candidates meet the hard constraints yet.",
                        ha="center", va="center", wrap=True,
                    )
            self.plot.redraw()
            return

        allowed = self.state.project.aircraft.distance_allowed_m

        # 1. Distance comparison, shortest (best) first.
        ax = axes[0][0]
        ranked = sorted(passing, key=lambda r: r.distance_m)
        self._bar_rank(ax, ranked, [r.distance_m for r in ranked], "#3b9e5f")
        ax.axvline(allowed, color="black", linestyle="--", linewidth=1, label=f"allowed ({allowed:.1f} m)")
        ax.set_xlabel("Ground-roll distance [m]")
        ax.set_title("Distance (#1 = shortest)")
        ax.legend(fontsize=6)
        ax.grid(True, axis="x", alpha=0.3)

        # 2. Power vs thrust "efficiency frontier".
        ax = axes[0][1]
        thrusts = [r.thrust_at_vt_n or 0.0 for r in passing]
        powers = [r.power_max_w for r in passing]
        ax.scatter(thrusts, powers, color="#3b6ea5", s=32, picker=True)
        self._axes_candidates[ax] = passing
        ax.set_xlabel("Thrust @ V_t [N]")
        ax.set_ylabel("Max electrical power [W]")
        ax.set_title("Power vs thrust (bottom-right best)")
        ax.grid(True, alpha=0.3)

        # 3. Pack current draw comparison, lowest (gentlest on the battery) first.
        ax = axes[0][2]
        ranked_i = sorted(passing, key=lambda r: r.current_max_pack_a)
        self._bar_rank(ax, ranked_i, [r.current_max_pack_a for r in ranked_i], "#c0622d")
        ax.set_xlabel("Max pack current [A]")
        ax.set_title("Pack current draw (#1 = lowest)")
        ax.grid(True, axis="x", alpha=0.3)

        # 4. Motor efficiency at V_t, highest (best) first.
        ax = axes[1][0]
        with_eta = [r for r in passing if r.eta_motor_at_vt is not None]
        ranked_eta = sorted(with_eta, key=lambda r: r.eta_motor_at_vt, reverse=True)
        self._bar_rank(ax, ranked_eta, [r.eta_motor_at_vt for r in ranked_eta], "#8a4fae")
        ax.set_xlim(0, 1)
        ax.set_xlabel("eta_motor @ V_t")
        ax.set_title("Motor efficiency (#1 = highest)")
        ax.grid(True, axis="x", alpha=0.3)

        # 5. Battery capacity used over the roll, lowest (best for endurance) first.
        ax = axes[1][1]
        ranked_mah = sorted(passing, key=lambda r: r.capacity_used_mah)
        self._bar_rank(ax, ranked_mah, [r.capacity_used_mah for r in ranked_mah], "#2d8ac0")
        ax.set_xlabel("Capacity used over roll [mAh]")
        ax.set_title("Energy used (#1 = lowest)")
        ax.grid(True, axis="x", alpha=0.3)

        # 6. Distance margin vs pack current -- top-left is best (safe margin,
        # gentle on the battery).
        ax = axes[1][2]
        margins = [r.distance_margin_pct if r.distance_margin_pct is not None else 0.0 for r in passing]
        currents_for_margin = [r.current_max_pack_a for r in passing]
        ax.scatter(currents_for_margin, margins, color="#c02d5a", s=32, picker=True)
        self._axes_candidates[ax] = passing
        ax.set_xlabel("Max pack current [A]")
        ax.set_ylabel("Distance margin [%]")
        ax.set_title("Margin vs current draw (top-left best)")
        ax.grid(True, alpha=0.3)

        self.plot.redraw()
