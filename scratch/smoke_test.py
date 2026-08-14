import math
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from propselect.gui.main_window import MainWindow

win = MainWindow()
print("motors loaded:", [m.name for m in win.state.motors])
print("props loaded:", list(win.state.props.keys()))

# --- Tab 1: set aircraft params to the hand-calc case ---
t1 = win.aircraft_tab
t1.mass.setValue(8.0)
t1.wing_area.setValue(0.65)
t1.span.setValue(2.0)
t1.aspect_ratio.setValue(6.0)
t1.cd0.setValue(0.09)
t1.cl.setValue(0.60)
t1.mu.setValue(0.10)
t1.v_t.setValue(15.0)
t1.distance_allowed.setValue(30.5)
print("T_req label:", t1.t_req_label.text())
assert "38." in t1.t_req_label.text() or "39." in t1.t_req_label.text(), t1.t_req_label.text()

# --- Tab 2: battery defaults ---
t2 = win.battery_tab
t2.series.setValue(6)
t2.parallel.setValue(2)
t2.capacity.setValue(5.0)
print("battery model built OK:", type(win.state.project.battery.to_battery()).__name__)

# --- Tab 3: select two motors ---
t3 = win.motor_tab
for row in range(min(2, t3.motor_table.rowCount())):
    item = t3.motor_table.item(row, 0)
    item.setCheckState(Qt.CheckState.Checked)
t3._on_table_changed()
print("selected motors:", win.state.project.selected_motor_names)
assert len(win.state.project.selected_motor_names) == 2

# --- Tab 4: select two props ---
t4 = win.prop_tab
for row in range(min(2, t4.table.rowCount())):
    item = t4.table.item(row, 0)
    item.setCheckState(Qt.CheckState.Checked)
win.state.project.selected_prop_names = t4._selected_names()
print("selected props:", win.state.project.selected_prop_names)
assert len(win.state.project.selected_prop_names) == 2

# --- Tab 5: run sweep synchronously (bypass QThread for the smoke test) ---
t5 = win.sweep_tab
specs = t5._build_specs()
print("num specs:", len(specs))
assert len(specs) == 4
requirement = win.state.project.to_requirement()
battery = win.state.project.battery.to_battery()
from propselect.core.candidate import evaluate_candidate
results = [evaluate_candidate(s, requirement, battery) for s in specs]
t5._on_finished(results)
print("table rows:", t5.table.rowCount())
assert t5.table.rowCount() == 4

# select first row -> should populate detail tab
t5.table.selectRow(0)
t5._on_selection_changed()
detail_result = win.detail_tab.current_result
print("detail tab candidate:", detail_result.spec.motor.name if detail_result else None)
assert detail_result is not None

# --- Tab 7: sensitivity ---
t7 = win.sensitivity_tab
t7.set_candidates(results)
t7.steps_spin.setValue(4)
t7._run_sensitivity()
print("sensitivity plot lines:", len(t7.plot.axes.lines))

# --- save/load project round trip ---
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "proj.json")
    win._collect_all_tabs()
    win.state.project.save(path)
    from propselect.project import Project
    loaded = Project.load(path)
    assert loaded.aircraft.mass_kg == 8.0
    print("project save/load OK")

print("SMOKE TEST PASSED")
