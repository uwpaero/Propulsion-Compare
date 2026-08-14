import sys

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from propselect.gui.main_window import MainWindow
from propselect.gui.units import UnitSystem

win = MainWindow()
t1 = win.aircraft_tab

# Set values in SI.
t1.mass.set_si_value(8.0)
t1.distance_allowed.set_si_value(30.5)
t1.v_t.set_si_value(15.0)
t1.elevation.set_si_value(500.0)
t1.temp_c.set_si_value(20.0)
t1._on_change()  # triggers save_to_project()

model = win.state.project.aircraft
print("Before toggle (SI): mass_kg=", model.mass_kg, "distance_allowed_m=", model.distance_allowed_m)
assert model.mass_kg == 8.0
assert model.distance_allowed_m == 30.5

# Toggle to US -- displayed values must convert, SI project model must NOT change.
win._set_unit_system(UnitSystem.US)
print("mass display:", t1.mass.value(), t1.mass.suffix())
print("distance display:", t1.distance_allowed.value(), t1.distance_allowed.suffix())
print("v_t display:", t1.v_t.value(), t1.v_t.suffix())
print("elevation display:", t1.elevation.value(), t1.elevation.suffix())
print("temp display:", t1.temp_c.value(), t1.temp_c.suffix())

assert abs(t1.mass.value() - 17.637) < 0.01
assert abs(t1.distance_allowed.value() - 100.07) < 0.1
assert abs(t1.v_t.value() - 33.55) < 0.1
assert abs(t1.elevation.value() - 1640.4) < 1.0
assert abs(t1.temp_c.value() - 68.0) < 0.1

# Project model (SI, canonical) must be completely unchanged by the toggle alone.
print("After toggle, project still SI: mass_kg=", model.mass_kg, "distance_allowed_m=", model.distance_allowed_m)
assert model.mass_kg == 8.0
assert model.distance_allowed_m == 30.5

# Now simulate a user edit WHILE in US mode -- e.g. change distance to 150 ft.
t1.distance_allowed.setValue(150.0)
t1._on_change()
print("edited distance (US)=150 ft -> project distance_allowed_m=", model.distance_allowed_m)
assert abs(model.distance_allowed_m - 150.0 * 0.3048) < 0.01

# Toggle back to SI: must show the EDITED value converted, not the original 30.5.
win._set_unit_system(UnitSystem.SI)
print("distance display after toggling back to SI:", t1.distance_allowed.value(), t1.distance_allowed.suffix())
assert abs(t1.distance_allowed.value() - 150.0 * 0.3048) < 0.01
assert t1.distance_allowed.value() != 30.5  # must NOT have reverted to the pre-edit value

print("UNITS SMOKE TEST PASSED")
