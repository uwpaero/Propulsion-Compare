"""Tab 4 -- Propeller Library: table, import APC/UIUC/CSV, CT(J)/CP(J) plot."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from propselect.core.propeller import PropellerModel
from propselect.data.loaders import IN_TO_M, PropellerImportError, load_apc_dat_file, load_csv_file, load_uiuc_file
from propselect.gui.plots import MplCanvas

PROP_COLUMNS = ["Select", "Name", "Diameter (m)", "Pitch (m)", "p/D", "Source", "Quality"]


def _diameter_bucket_in(prop: PropellerModel) -> int:
    """Nominal diameter in whole inches, for grouping props by size class."""
    return round(prop.diameter_m / IN_TO_M)


class PropellerLibraryTab(QWidget):
    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self._row_names: list[str] = []

        self.table = QTableWidget(0, len(PROP_COLUMNS))
        self.table.setHorizontalHeaderLabels(PROP_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self.import_apc_btn = QPushButton("Import APC .dat...")
        self.import_uiuc_btn = QPushButton("Import UIUC .txt...")
        self.import_csv_btn = QPushButton("Import CSV...")
        self.remove_btn = QPushButton("Remove selected")
        self.select_all_btn = QPushButton("Select all")
        self.select_none_btn = QPushButton("Select none")

        self.diameter_filter_combo = QComboBox()
        self.select_size_btn = QPushButton("Select this size")
        self.deselect_size_btn = QPushButton("Deselect this size")

        self.plot = MplCanvas(self, nrows=2, ncols=1, figsize=(5, 6))

        self._build_layout()
        self._wire_signals()
        self._load_bundled_props()
        self._refresh_table()

    def _build_layout(self) -> None:
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.import_apc_btn)
        btn_row.addWidget(self.import_uiuc_btn)
        btn_row.addWidget(self.import_csv_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.select_none_btn)
        btn_row.addStretch(1)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        size_row.addWidget(self.diameter_filter_combo)
        size_row.addWidget(self.select_size_btn)
        size_row.addWidget(self.deselect_size_btn)
        size_row.addStretch(1)

        table_box = QGroupBox("Propeller library (check to include in sweep)")
        table_layout = QVBoxLayout()
        table_layout.addLayout(btn_row)
        table_layout.addLayout(size_row)
        table_layout.addWidget(self.table)
        table_box.setLayout(table_layout)

        outer = QHBoxLayout(self)
        outer.addWidget(table_box, 1)
        outer.addWidget(self.plot, 1)

    def _wire_signals(self) -> None:
        self.import_apc_btn.clicked.connect(self._import_apc)
        self.import_uiuc_btn.clicked.connect(self._import_uiuc)
        self.import_csv_btn.clicked.connect(self._import_csv)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.select_all_btn.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Checked))
        self.select_none_btn.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Unchecked))
        self.select_size_btn.clicked.connect(lambda: self._set_size_checks(Qt.CheckState.Checked))
        self.deselect_size_btn.clicked.connect(lambda: self._set_size_checks(Qt.CheckState.Unchecked))
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.currentCellChanged.connect(lambda *_: self._update_plot())

    def _load_bundled_props(self) -> None:
        if self.state.props:
            return
        self._scan_bundled_props()

    def _scan_bundled_props(self) -> None:
        from propselect.gui.main_window import DEFAULT_PROP_DIR

        if not DEFAULT_PROP_DIR.exists():
            return
        for path in sorted(DEFAULT_PROP_DIR.glob("*.dat")):
            try:
                result = load_apc_dat_file(path)
            except PropellerImportError:
                continue
            self.state.props[result.prop.name] = result.prop

    def reset_to_bundled(self) -> None:
        """Discard the current prop list and re-scan the bundled defaults (New Project)."""
        self.state.props.clear()
        self.state.project.selected_prop_names = []
        self._scan_bundled_props()
        self._refresh_table()

    def rescan_bundled_into_state(self) -> None:
        """Re-scan the bundled defaults into ``state.props`` without touching selection.

        Used when opening a project saved before per-project prop snapshots
        existed: the project's own ``selected_prop_names`` should still apply
        to the freshly (re)loaded bundled library.
        """
        self.state.props.clear()
        self._scan_bundled_props()

    def load_from_project(self) -> None:
        """Resync the table with ``state.props``/``state.project`` (e.g. after Open Project)."""
        self._refresh_table()

    def _refresh_table(self) -> None:
        selected = set(self.state.project.selected_prop_names)
        names = list(self.state.props.keys())
        self._row_names = names
        self.table.blockSignals(True)
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            prop = self.state.props[name]
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check_item.setCheckState(Qt.CheckState.Checked if name in selected else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, check_item)
            self.table.setItem(row, 1, QTableWidgetItem(prop.name))
            self.table.setItem(row, 2, QTableWidgetItem(f"{prop.diameter_m:.4f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{prop.pitch_m:.4f}"))
            pd_ratio = prop.pitch_m / prop.diameter_m if prop.diameter_m else 0.0
            self.table.setItem(row, 4, QTableWidgetItem(f"{pd_ratio:.3f}"))
            source = getattr(prop, "source", "") or ""
            self.table.setItem(row, 5, QTableWidgetItem(source))
            quality = "LOW CONFIDENCE (parametric)" if prop.is_low_confidence else "tabulated data"
            quality_item = QTableWidgetItem(quality)
            if prop.is_low_confidence:
                quality_item.setForeground(Qt.GlobalColor.darkYellow)
            self.table.setItem(row, 6, quality_item)
            for col in range(1, len(PROP_COLUMNS)):
                item = self.table.item(row, col)
                if item is not None:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.blockSignals(False)
        self._refresh_diameter_filter()
        self._update_plot()

    def _refresh_diameter_filter(self) -> None:
        sizes = sorted({_diameter_bucket_in(self.state.props[name]) for name in self._row_names})
        current = self.diameter_filter_combo.currentData()
        self.diameter_filter_combo.blockSignals(True)
        self.diameter_filter_combo.clear()
        for size_in in sizes:
            self.diameter_filter_combo.addItem(f"{size_in} in", userData=size_in)
        if current in sizes:
            self.diameter_filter_combo.setCurrentIndex(sizes.index(current))
        self.diameter_filter_combo.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self.state.project.selected_prop_names = self._selected_names()

    def _selected_names(self) -> list[str]:
        names = []
        for row, name in enumerate(self._row_names):
            check_item = self.table.item(row, 0)
            if check_item and check_item.checkState() == Qt.CheckState.Checked:
                names.append(name)
        return names

    def _set_all_checks(self, check_state: Qt.CheckState) -> None:
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            check_item = self.table.item(row, 0)
            if check_item is not None:
                check_item.setCheckState(check_state)
        self.table.blockSignals(False)
        self.state.project.selected_prop_names = self._selected_names()

    def _set_size_checks(self, check_state: Qt.CheckState) -> None:
        target_size = self.diameter_filter_combo.currentData()
        if target_size is None:
            return
        self.table.blockSignals(True)
        for row, name in enumerate(self._row_names):
            prop = self.state.props.get(name)
            if prop is None or _diameter_bucket_in(prop) != target_size:
                continue
            check_item = self.table.item(row, 0)
            if check_item is not None:
                check_item.setCheckState(check_state)
        self.table.blockSignals(False)
        self.state.project.selected_prop_names = self._selected_names()

    def _current_prop(self) -> PropellerModel | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._row_names):
            return None
        return self.state.props.get(self._row_names[row])

    def _update_plot(self) -> None:
        self.plot.clear()
        ax_ct, ax_cp = self.plot.axes
        prop = self._current_prop()
        if prop is None:
            self.plot.redraw()
            return
        j_max = getattr(prop, "j_max", None) or (0.9 * prop.pitch_m / prop.diameter_m)
        j_values = np.linspace(0.0, max(j_max * 1.2, 0.1), 150)
        ct_values = [prop.evaluate(j).ct for j in j_values]
        cp_values = [prop.evaluate(j).cp for j in j_values]
        ax_ct.plot(j_values, ct_values, color="#3b6ea5")
        ax_ct.set_ylabel("C_T")
        ax_ct.set_title(f"{prop.name}: C_T(J), C_P(J)")
        ax_ct.grid(True, alpha=0.3)
        ax_cp.plot(j_values, cp_values, color="#c0622d")
        ax_cp.set_xlabel("J (advance ratio)")
        ax_cp.set_ylabel("C_P")
        ax_cp.grid(True, alpha=0.3)
        self.plot.redraw()

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._row_names):
            return
        name = self._row_names[row]
        self.state.props.pop(name, None)
        self._refresh_table()
        self.state.project.selected_prop_names = self._selected_names()

    def _import_apc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import APC .dat", "", "APC data (*.dat)")
        if not path:
            return
        try:
            result = load_apc_dat_file(path)
        except PropellerImportError as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.state.props[result.prop.name] = result.prop
        if result.warnings:
            QMessageBox.warning(self, "Import warnings", "\n".join(result.warnings))
        self._refresh_table()

    def _import_uiuc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import UIUC file", "", "Text (*.txt)")
        if not path:
            return
        try:
            result = load_uiuc_file(path)
        except (PropellerImportError, ValueError):
            diameter_in, ok1 = QInputDialog.getDouble(self, "Diameter", "Diameter (in):", 10.0, 0.1, 100.0, 2)
            if not ok1:
                return
            pitch_in, ok2 = QInputDialog.getDouble(self, "Pitch", "Pitch (in):", 6.0, 0.1, 100.0, 2)
            if not ok2:
                return
            try:
                result = load_uiuc_file(path, diameter_in=diameter_in, pitch_in=pitch_in)
            except PropellerImportError as exc:
                QMessageBox.critical(self, "Import failed", str(exc))
                return
        self.state.props[result.prop.name] = result.prop
        self._refresh_table()

    def _import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV (*.csv)")
        if not path:
            return
        diameter_in, ok1 = QInputDialog.getDouble(self, "Diameter", "Diameter (in):", 10.0, 0.1, 100.0, 2)
        if not ok1:
            return
        pitch_in, ok2 = QInputDialog.getDouble(self, "Pitch", "Pitch (in):", 6.0, 0.1, 100.0, 2)
        if not ok2:
            return
        try:
            result = load_csv_file(path, diameter_in=diameter_in, pitch_in=pitch_in)
        except PropellerImportError as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.state.props[result.prop.name] = result.prop
        self._refresh_table()
