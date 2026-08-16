"""Tab 10 -- Cruise Comparison: cross-candidate plots for the latest cruise sweep.

Mirrors Tab 7 (Comparison, for takeoff): ranked/scatter plots over every
eligible candidate, each point/bar clickable to jump to its row in Tab 8.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from propselect.core.cruise import CruiseCandidateResult
from propselect.gui.plots import MplCanvas


class CruiseComparisonTab(QWidget):
    candidate_clicked = Signal(object)  # CruiseCandidateResult

    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.results: list[CruiseCandidateResult] = []
        self._axes_candidates: dict[Axes, list[CruiseCandidateResult]] = {}

        self.status_label = QLabel("Run a sweep in Tab 8 to populate these plots.")
        self.hint_label = QLabel(
            "Click a bar or point to jump to that combination in Tab 8 (Cruise)."
        )
        self.hint_label.setStyleSheet("color: gray; font-style: italic;")
        self.plot = MplCanvas(self, nrows=2, ncols=2, figsize=(12, 8))
        self.plot.canvas.mpl_connect("pick_event", self._on_pick)

        outer = QVBoxLayout(self)
        outer.addWidget(self.status_label)
        outer.addWidget(self.hint_label)
        outer.addWidget(self.plot)

    def set_candidates(self, results: list[CruiseCandidateResult]) -> None:
        self.results = results
        n_eligible = sum(1 for r in results if r.eligible)
        self.status_label.setText(
            f"{len(results)} candidates evaluated, {n_eligible} eligible (hard constraints met) "
            "-- only eligible candidates are plotted below."
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

    def _bar_rank(self, ax: Axes, ranked: list[CruiseCandidateResult], values: list[float], color: str) -> None:
        y_pos = list(range(len(ranked)))
        bars = ax.barh(y_pos, values, color=color, picker=True)
        for patch in bars.patches:
            patch.set_picker(True)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"#{i + 1}" for i in y_pos], fontsize=7)
        ax.invert_yaxis()
        self._axes_candidates[ax] = ranked

    def _update_plots(self, results: list[CruiseCandidateResult]) -> None:
        self.plot.clear()
        self._axes_candidates.clear()
        axes = self.plot.axes  # 2x2
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

        # 1. Endurance comparison, longest (best) first.
        ax = axes[0][0]
        with_endurance = [r for r in passing if r.endurance_s is not None]
        ranked = sorted(with_endurance, key=lambda r: r.endurance_s, reverse=True)
        self._bar_rank(ax, ranked, [r.endurance_s / 60.0 for r in ranked], "#3b9e5f")
        ax.set_xlabel("Endurance [min]")
        ax.set_title("Endurance (#1 = longest)")
        ax.grid(True, axis="x", alpha=0.3)

        # 2. Pack current draw comparison, lowest (gentlest on the battery) first.
        ax = axes[0][1]
        with_current = [r for r in passing if r.current_pack_a is not None]
        ranked_i = sorted(with_current, key=lambda r: r.current_pack_a)
        self._bar_rank(ax, ranked_i, [r.current_pack_a for r in ranked_i], "#c0622d")
        ax.set_xlabel("Cruise pack current [A]")
        ax.set_title("Pack current draw (#1 = lowest)")
        ax.grid(True, axis="x", alpha=0.3)

        # 3. Motor efficiency at cruise, highest (best) first.
        ax = axes[1][0]
        with_eta = [r for r in passing if r.eta_motor is not None]
        ranked_eta = sorted(with_eta, key=lambda r: r.eta_motor, reverse=True)
        self._bar_rank(ax, ranked_eta, [r.eta_motor for r in ranked_eta], "#8a4fae")
        ax.set_xlim(0, 1)
        ax.set_xlabel("eta_motor @ cruise")
        ax.set_title("Motor efficiency (#1 = highest)")
        ax.grid(True, axis="x", alpha=0.3)

        # 4. Range vs throttle -- top-left is best (long range, headroom left).
        ax = axes[1][1]
        with_range = [r for r in passing if r.range_m is not None and r.throttle_fraction is not None]
        throttles = [r.throttle_fraction * 100.0 for r in with_range]
        ranges_km = [r.range_m / 1000.0 for r in with_range]
        ax.scatter(throttles, ranges_km, color="#2d8ac0", s=32, picker=True)
        self._axes_candidates[ax] = with_range
        ax.set_xlabel("Throttle [%]")
        ax.set_ylabel("Range [km]")
        ax.set_title("Range vs throttle (top-left best)")
        ax.grid(True, alpha=0.3)

        self.plot.redraw()
