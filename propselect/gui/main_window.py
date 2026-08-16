"""Main application window: tabbed layout, project save/load, shared state."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QTabWidget

from propselect.core.motor import MotorSpec
from propselect.core.propeller import PropellerModel
from propselect.data.loaders import (
    load_motor_library,
    motor_from_dict,
    motor_to_dict,
    prop_from_dict,
    prop_to_dict,
    save_motor_library,
)
from propselect.gui.units import UnitSystem
from propselect.project import Project

DEFAULT_MOTOR_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "data" / "motors.json"
DEFAULT_PROP_DIR = Path(__file__).resolve().parents[1] / "data" / "props"


class AppState:
    """Shared, mutable application state: the current project, motor/prop libraries."""

    def __init__(self) -> None:
        self.project = Project()
        self.project_path: Path | None = None
        self.motors: list[MotorSpec] = []
        self.props: dict[str, PropellerModel] = {}
        self.last_results = []  # list[CandidateResult], populated by Tab 5

        self.load_bundled_motors()

    def load_bundled_motors(self) -> None:
        """(Re)load the shared bundled motor library, discarding any in-memory edits."""
        try:
            self.motors = load_motor_library(DEFAULT_MOTOR_LIBRARY_PATH)
        except Exception:
            self.motors = []


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("propselect -- electric propulsion selection")
        self.resize(1280, 860)

        self.state = AppState()

        self._build_menu()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Imported here (not at module scope) so tab construction order is
        # explicit and each tab module only needs to exist once used.
        from propselect.gui.tabs.aircraft_tab import AircraftTab
        from propselect.gui.tabs.battery_tab import BatteryTab
        from propselect.gui.tabs.comparison_tab import ComparisonTab
        from propselect.gui.tabs.cruise_results_tab import CruiseResultsTab
        from propselect.gui.tabs.detail_tab import DetailTab
        from propselect.gui.tabs.motor_library_tab import MotorLibraryTab
        from propselect.gui.tabs.propeller_library_tab import PropellerLibraryTab
        from propselect.gui.tabs.sweep_results_tab import SweepResultsTab

        self.aircraft_tab = AircraftTab(self.state)
        self.battery_tab = BatteryTab(self.state)
        self.motor_tab = MotorLibraryTab(self.state)
        self.prop_tab = PropellerLibraryTab(self.state)
        self.sweep_tab = SweepResultsTab(self.state)
        self.detail_tab = DetailTab(self.state)
        self.comparison_tab = ComparisonTab(self.state)
        self.cruise_tab = CruiseResultsTab(self.state)

        self.tabs.addTab(self.aircraft_tab, "1. Aircraft && Requirement")
        self.tabs.addTab(self.battery_tab, "2. Battery && ESC")
        self.tabs.addTab(self.motor_tab, "3. Motor Library")
        self.tabs.addTab(self.prop_tab, "4. Propeller Library")
        self.tabs.addTab(self.sweep_tab, "5. Sweep && Results")
        self.tabs.addTab(self.detail_tab, "6. Detail")
        self.tabs.addTab(self.comparison_tab, "7. Comparison")
        self.tabs.addTab(self.cruise_tab, "8. Cruise")

        # Cross-tab wiring: a row selected in Tab 5 populates Tab 6.
        self.sweep_tab.candidate_selected.connect(self.detail_tab.show_candidate)
        self.sweep_tab.sweep_finished.connect(self.detail_tab.set_candidates)
        self.sweep_tab.sweep_finished.connect(self.comparison_tab.set_candidates)
        self.comparison_tab.candidate_clicked.connect(self._on_comparison_candidate_clicked)

    def _on_comparison_candidate_clicked(self, result) -> None:
        self.sweep_tab.select_candidate(result)
        self.tabs.setCurrentWidget(self.sweep_tab)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        menu.addAction("&New Project", self._new_project)
        menu.addAction("&Open Project...", self._open_project)
        menu.addAction("&Save Project", self._save_project)
        menu.addAction("Save Project &As...", self._save_project_as)

        units_menu = self.menuBar().addMenu("&Units")
        self.unit_system = UnitSystem.SI
        group = QActionGroup(self)
        group.setExclusive(True)
        si_action = units_menu.addAction("&SI (kg, m, m/s, degC)")
        si_action.setCheckable(True)
        si_action.setChecked(True)
        us_action = units_menu.addAction("&US customary (lb, ft, mph, degF)")
        us_action.setCheckable(True)
        group.addAction(si_action)
        group.addAction(us_action)
        si_action.triggered.connect(lambda: self._set_unit_system(UnitSystem.SI))
        us_action.triggered.connect(lambda: self._set_unit_system(UnitSystem.US))

    def _set_unit_system(self, system: UnitSystem) -> None:
        """Switch every tab's displayed unit system, converting values in place.

        This never touches the underlying (always-SI) Project model -- each
        tab's unit-aware fields convert their own displayed numbers.
        """
        if system is self.unit_system:
            return
        self.unit_system = system
        self.aircraft_tab.set_unit_system(system)

    def _new_project(self) -> None:
        self.state.project = Project()
        self.state.project_path = None
        self.state.load_bundled_motors()
        self.prop_tab.reset_to_bundled()
        self._refresh_all_tabs()

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "propselect project (*.json)")
        if not path:
            return
        try:
            self.state.project = Project.load(path)
            self.state.project_path = Path(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Project failed", str(exc))
            return

        # A project saved with its own library snapshot (add/edit/delete
        # persisted) overrides the shared bundled defaults; older/new
        # projects with no snapshot yet fall back to the bundled library.
        if self.state.project.motors is not None:
            self.state.motors = [motor_from_dict(e) for e in self.state.project.motors]
        else:
            self.state.load_bundled_motors()

        if self.state.project.props is not None:
            self.state.props = {}
            for entry in self.state.project.props:
                prop = prop_from_dict(entry)
                self.state.props[prop.name] = prop
        else:
            self.prop_tab.rescan_bundled_into_state()

        self._refresh_all_tabs()

    def _save_project(self) -> None:
        if self.state.project_path is None:
            self._save_project_as()
            return
        self._collect_all_tabs()
        self.state.project.save(self.state.project_path)

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "propselect project (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self._collect_all_tabs()
        self.state.project.save(path)
        self.state.project_path = Path(path)

    def _refresh_all_tabs(self) -> None:
        self.aircraft_tab.load_from_project()
        self.battery_tab.load_from_project()
        self.motor_tab.load_from_project()
        self.prop_tab.load_from_project()

    def _collect_all_tabs(self) -> None:
        self.aircraft_tab.save_to_project()
        self.battery_tab.save_to_project()
        # The motor/prop *lists themselves* (not just which names are
        # selected) live in state.motors/state.props, kept in sync by the
        # library tabs on every add/edit/delete -- snapshot them into the
        # project here so deletions actually persist across a reload.
        self.state.project.motors = [motor_to_dict(m) for m in self.state.motors]
        self.state.project.props = [prop_to_dict(p) for p in self.state.props.values()]


def run_app() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_app())
