"""Tab 3 -- Motor Library: table view, add/edit/delete, import/export, motor counts."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from propselect.core.motor import MotorSpec
from propselect.data.loaders import load_motor_library, save_motor_library

MOTOR_COLUMNS = [
    "Select",
    "Name",
    "Kv (RPM/V)",
    "R_motor (ohm)",
    "I0 (A)",
    "I_max_cont (A)",
    "I_max_burst (A)",
    "Mass (kg)",
    "Shaft dia (mm)",
    "Source URL",
    "Notes",
]


class MotorLibraryTab(QWidget):
    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state

        self.motor_table = QTableWidget(0, len(MOTOR_COLUMNS))
        self.motor_table.setHorizontalHeaderLabels(MOTOR_COLUMNS)
        self.motor_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.motor_table.horizontalHeader().setStretchLastSection(True)

        self.add_btn = QPushButton("Add motor")
        self.delete_btn = QPushButton("Delete selected row")
        self.import_btn = QPushButton("Import JSON...")
        self.export_btn = QPushButton("Export JSON...")
        self.select_all_btn = QPushButton("Select all")
        self.select_none_btn = QPushButton("Select none")

        self.motor_counts_edit = QLineEdit()
        self.motor_counts_edit.setPlaceholderText("e.g. 1, 2, 4")
        self.motor_counts_edit.setToolTip(
            "Comma-separated motor counts to sweep (identical motors sharing one pack). "
            "Each value adds motor_count x thrust/power at motor_count x pack current draw."
        )

        self._build_layout()
        self._wire_signals()
        self._populate_from_state_motors()
        self._populate_motor_counts()

    def _build_layout(self) -> None:
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.import_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.select_none_btn)
        btn_row.addStretch(1)

        motor_box = QGroupBox("Motor library (check to include in sweep)")
        motor_layout = QVBoxLayout()
        motor_layout.addLayout(btn_row)
        motor_layout.addWidget(self.motor_table)
        motor_box.setLayout(motor_layout)

        motor_count_box = QGroupBox(
            "Motor counts (identical motors sharing one pack; every value swept)"
        )
        motor_count_layout = QHBoxLayout()
        motor_count_layout.addWidget(QLabel("Counts:"))
        motor_count_layout.addWidget(self.motor_counts_edit)
        motor_count_box.setLayout(motor_count_layout)

        outer = QVBoxLayout(self)
        outer.addWidget(motor_box, 3)
        outer.addWidget(motor_count_box)

    def _wire_signals(self) -> None:
        self.add_btn.clicked.connect(self._add_motor_row)
        self.delete_btn.clicked.connect(self._delete_motor_row)
        self.import_btn.clicked.connect(self._import_json)
        self.export_btn.clicked.connect(self._export_json)
        self.select_all_btn.clicked.connect(lambda: self._set_all_motor_checks(Qt.CheckState.Checked))
        self.select_none_btn.clicked.connect(lambda: self._set_all_motor_checks(Qt.CheckState.Unchecked))
        self.motor_table.itemChanged.connect(self._on_table_changed)

        self.motor_counts_edit.editingFinished.connect(self._sync_motor_counts_to_project)

    def _populate_from_state_motors(self) -> None:
        self._set_motor_rows(self.state.motors, selected_names=set(self.state.project.selected_motor_names))

    def load_from_project(self) -> None:
        """Resync the table with ``state.motors``/``state.project`` (e.g. after Open Project)."""
        self._populate_from_state_motors()

    def _set_motor_rows(self, motors: list[MotorSpec], selected_names: set[str] | None = None) -> None:
        selected_names = selected_names or set()
        self.motor_table.blockSignals(True)
        self.motor_table.setRowCount(len(motors))
        for row, m in enumerate(motors):
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check_item.setCheckState(
                Qt.CheckState.Checked if m.name in selected_names else Qt.CheckState.Unchecked
            )
            self.motor_table.setItem(row, 0, check_item)
            values = [
                m.name,
                m.kv_rpm_per_v,
                m.r_motor_ohm,
                m.i0_a,
                m.i_max_cont_a,
                m.i_max_burst_a,
                m.mass_kg,
                m.shaft_dia_mm,
                m.source_url,
                m.notes,
            ]
            for col, value in enumerate(values, start=1):
                text = "" if value is None else str(value)
                self.motor_table.setItem(row, col, QTableWidgetItem(text))
        self.motor_table.blockSignals(False)

    def _read_motor_rows(self) -> list[MotorSpec]:
        motors = []
        for row in range(self.motor_table.rowCount()):
            name_item = self.motor_table.item(row, 1)
            if name_item is None or not name_item.text().strip():
                continue
            try:
                motors.append(
                    MotorSpec(
                        name=name_item.text().strip(),
                        kv_rpm_per_v=float(self._cell(row, 2, "0")),
                        r_motor_ohm=float(self._cell(row, 3, "0")),
                        i0_a=float(self._cell(row, 4, "0")),
                        i_max_cont_a=self._optional_float(row, 5),
                        i_max_burst_a=self._optional_float(row, 6),
                        mass_kg=self._optional_float(row, 7),
                        shaft_dia_mm=self._optional_float(row, 8),
                        source_url=self._cell(row, 9, "") or None,
                        notes=self._cell(row, 10, "") or None,
                    )
                )
            except ValueError:
                continue
        return motors

    def _cell(self, row: int, col: int, default: str) -> str:
        item = self.motor_table.item(row, col)
        return item.text() if item is not None else default

    def _optional_float(self, row: int, col: int) -> float | None:
        text = self._cell(row, col, "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _selected_motor_names(self) -> list[str]:
        names = []
        for row in range(self.motor_table.rowCount()):
            check_item = self.motor_table.item(row, 0)
            name_item = self.motor_table.item(row, 1)
            if check_item and name_item and check_item.checkState() == Qt.CheckState.Checked:
                names.append(name_item.text().strip())
        return names

    def _on_table_changed(self) -> None:
        self.state.motors = self._read_motor_rows()
        self.state.project.selected_motor_names = self._selected_motor_names()

    def _set_all_motor_checks(self, check_state: Qt.CheckState) -> None:
        self.motor_table.blockSignals(True)
        for row in range(self.motor_table.rowCount()):
            check_item = self.motor_table.item(row, 0)
            if check_item is not None:
                check_item.setCheckState(check_state)
        self.motor_table.blockSignals(False)
        self._on_table_changed()

    def _add_motor_row(self) -> None:
        self.motor_table.blockSignals(True)
        row = self.motor_table.rowCount()
        self.motor_table.insertRow(row)
        check_item = QTableWidgetItem()
        check_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        check_item.setCheckState(Qt.CheckState.Unchecked)
        self.motor_table.setItem(row, 0, check_item)
        self.motor_table.setItem(row, 1, QTableWidgetItem("new-motor"))
        for col in range(2, len(MOTOR_COLUMNS)):
            self.motor_table.setItem(row, col, QTableWidgetItem(""))
        self.motor_table.blockSignals(False)
        self._on_table_changed()

    def _delete_motor_row(self) -> None:
        row = self.motor_table.currentRow()
        if row >= 0:
            self.motor_table.removeRow(row)
        self._on_table_changed()

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import motor library", "", "JSON (*.json)")
        if not path:
            return
        try:
            motors = load_motor_library(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.state.motors = motors
        self._set_motor_rows(motors)
        self.state.project.selected_motor_names = self._selected_motor_names()

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export motor library", "", "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_motor_library(path, self._read_motor_rows())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # --- motor counts ---

    def _populate_motor_counts(self) -> None:
        self.motor_counts_edit.blockSignals(True)
        self.motor_counts_edit.setText(", ".join(str(c) for c in self.state.project.motor_counts))
        self.motor_counts_edit.blockSignals(False)

    def _sync_motor_counts_to_project(self) -> None:
        counts = []
        for token in self.motor_counts_edit.text().split(","):
            token = token.strip()
            if not token:
                continue
            try:
                count = int(token)
            except ValueError:
                continue
            if count >= 1:
                counts.append(count)
        if not counts:
            counts = [1]
        self.state.project.motor_counts = counts
        self._populate_motor_counts()
