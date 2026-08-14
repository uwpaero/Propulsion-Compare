import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from propselect.gui.main_window import MainWindow
from propselect.core.candidate import evaluate_candidate

win = MainWindow()

t3 = win.motor_tab
for row in range(min(1, t3.motor_table.rowCount())):
    t3.motor_table.item(row, 0).setCheckState(Qt.CheckState.Checked)
t3._on_table_changed()

t4 = win.prop_tab
for row in range(min(1, t4.table.rowCount())):
    t4.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
win.state.project.selected_prop_names = t4._selected_names()

# set motor counts to 1, 2, 4
t3.motor_counts_edit.setText("1, 2, 4")
t3._sync_motor_counts_to_project()
print("project.motor_counts:", win.state.project.motor_counts)
assert win.state.project.motor_counts == [1, 2, 4]

t5 = win.sweep_tab
specs = t5._build_specs()
print("num specs (1 motor x 1 prop x 1 gearbox x 3 motor_counts):", len(specs))
assert len(specs) == 3
assert sorted(s.motor_count for s in specs) == [1, 2, 4]

requirement = win.state.project.to_requirement()
battery = win.state.project.battery.to_battery()
results = [evaluate_candidate(s, requirement, battery) for s in specs]
t5._on_finished(results)
print("table rows:", t5.table.rowCount())
assert t5.table.rowCount() == 3

# check the Motors column (index 3) shows 1, 2, 4 somewhere
motors_col_values = sorted(int(t5.table.item(r, 3).text()) for r in range(3))
print("motors column values:", motors_col_values)
assert motors_col_values == [1, 2, 4]

# select the motor_count=2 row and check detail tab shows it
for r in range(3):
    if t5.table.item(r, 3).text() == "2":
        t5.table.selectRow(r)
        t5._on_selection_changed()
        break
detail_result = win.detail_tab.current_result
print("detail candidate motor_count:", detail_result.spec.motor_count)
assert detail_result.spec.motor_count == 2
print("detail title:", win.detail_tab.title_label.text())
assert "x2 motors" in win.detail_tab.title_label.text()

# verify pack current > per-motor current for this candidate
assert detail_result.current_max_pack_a > detail_result.current_max_per_motor_a

# project save/load round trip preserves motor_counts
import tempfile, os
from propselect.project import Project
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "proj.json")
    win.state.project.save(path)
    loaded = Project.load(path)
    assert loaded.motor_counts == [1, 2, 4]
    print("motor_counts round-trip OK")

print("MOTOR COUNT SMOKE TEST PASSED")
