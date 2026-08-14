import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from propselect.gui.main_window import MainWindow
from propselect.core.candidate import evaluate_candidate

win = MainWindow()
win.resize(1280, 860)

t1 = win.aircraft_tab
t1.mass.setValue(8.0)
t1.wing_area.setValue(0.65)
t1.v_t.setValue(15.0)
t1.distance_allowed.setValue(30.5)

t3 = win.motor_tab
for row in range(min(2, t3.motor_table.rowCount())):
    t3.motor_table.item(row, 0).setCheckState(Qt.CheckState.Checked)
t3._on_table_changed()

t4 = win.prop_tab
for row in range(min(2, t4.table.rowCount())):
    t4.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
win.state.project.selected_prop_names = t4._selected_names()

t5 = win.sweep_tab
specs = t5._build_specs()
requirement = win.state.project.to_requirement()
battery = win.state.project.battery.to_battery()
results = [evaluate_candidate(s, requirement, battery) for s in specs]
t5._on_finished(results)
t5.table.selectRow(0)
t5._on_selection_changed()

t7 = win.sensitivity_tab
t7.set_candidates(results)
t7._run_sensitivity()

win.show()
app.processEvents()

for idx, tabname in enumerate(["tab1", "tab2", "tab3", "tab4", "tab5", "tab6", "tab7"]):
    win.tabs.setCurrentIndex(idx)
    app.processEvents()
    pixmap = win.grab()
    pixmap.save(f"scratch/{tabname}.png")
    print("saved", tabname)

print("DONE")
