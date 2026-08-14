"""Matplotlib canvas widgets embedded in the PySide6 GUI."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MplCanvas(QWidget):
    """A single matplotlib Figure embedded as a Qt widget, with a nav toolbar."""

    def __init__(self, parent: QWidget | None = None, nrows: int = 1, ncols: int = 1, figsize=(6, 4)):
        super().__init__(parent)
        self._nrows = nrows
        self._ncols = ncols
        self.figure = Figure(figsize=figsize, layout="constrained")
        self._build_grid()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def _build_grid(self) -> None:
        if self._nrows == 1 and self._ncols == 1:
            self.axes = self.figure.add_subplot(111)
        else:
            self.axes = self.figure.subplots(self._nrows, self._ncols)

    def clear(self) -> None:
        # Clearing the whole figure (not just each axes' content) is
        # required to remove any twinx()/twiny() axes a caller may have
        # added on a previous draw -- those live outside self.axes and
        # would otherwise silently accumulate, showing stale data from
        # earlier selections underneath the current plot.
        self.figure.clear()
        self._build_grid()

    def redraw(self) -> None:
        self.canvas.draw_idle()

    def save_png(self, path: str) -> None:
        self.figure.savefig(path, dpi=150)
