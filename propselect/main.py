"""Entry point: ``propselect cli ...`` runs one combination headlessly;
``propselect gui`` launches the PySide6 application.
"""

from __future__ import annotations

import argparse
import sys

from propselect.core.atmosphere import speed_of_sound
from propselect.core.candidate import CandidateSpec, evaluate_candidate
from propselect.core.motor import MotorSpec
from propselect.core.operating_point import OperatingPoint
from propselect.core.propeller import PropellerModel
from propselect.core.takeoff import closed_form_thrust_estimate
from propselect.data.loaders import load_apc_dat_file, load_csv_file, load_motor_library, load_uiuc_file
from propselect.project import AircraftConfigModel, BatteryConfigModel, Project


def _load_propeller(args: argparse.Namespace) -> PropellerModel:
    path = args.prop
    lower = path.lower()
    if lower.endswith(".csv"):
        if args.prop_diameter_in is None or args.prop_pitch_in is None:
            raise SystemExit("--prop-diameter-in and --prop-pitch-in are required for CSV props")
        result = load_csv_file(path, diameter_in=args.prop_diameter_in, pitch_in=args.prop_pitch_in)
    elif lower.endswith(".dat"):
        result = load_apc_dat_file(path, expected_rpm=args.expected_rpm)
    else:
        result = load_uiuc_file(path, diameter_in=args.prop_diameter_in, pitch_in=args.prop_pitch_in)
    for warning in result.warnings:
        print(f"[prop import warning] {warning}", file=sys.stderr)
    return result.prop


def _load_motor(args: argparse.Namespace) -> MotorSpec:
    motors = load_motor_library(args.motors_json)
    for m in motors:
        if m.name == args.motor:
            return m
    available = ", ".join(m.name for m in motors)
    raise SystemExit(f"Motor {args.motor!r} not found in {args.motors_json}. Available: {available}")


def _build_project(args: argparse.Namespace) -> Project:
    aircraft = AircraftConfigModel(
        mass_kg=args.mass,
        wing_area_m2=args.wing_area,
        span_m=args.span,
        cd0_ground=args.cd,
        cl_ground=args.cl,
        mu=args.mu,
        wheel_height_m=args.wheel_height,
        ground_effect=args.ground_effect,
        v_t_m_s=args.vt,
        distance_allowed_m=args.distance,
        field_elevation_m=args.elevation,
        field_temperature_c=args.temp_c,
    )
    battery = BatteryConfigModel(
        series=args.series,
        parallel=args.parallel,
        r_internal_per_cell_ohm=args.r_cell,
        capacity_ah=args.capacity,
        esc_r_ohm=args.esc_r,
        esc_current_cont_a=args.esc_current_cont,
        c_rate_limit=args.c_rate_limit,
    )
    return Project(
        aircraft=aircraft,
        battery=battery,
        tip_mach_limit=args.tip_mach_limit,
        motor_eta_threshold=args.motor_eta_threshold,
        power_limit_w=args.power_limit,
        dv_m_s=args.dv,
    )


def _print_summary(
    motor: MotorSpec, prop: PropellerModel, project: Project, motor_count: int
) -> None:
    requirement = project.to_requirement()
    battery = project.battery.to_battery()

    print("=== propselect CLI ===")
    print(
        f"Motor: {motor.name} (Kv={motor.kv_rpm_per_v:g} RPM/V, "
        f"R={motor.r_motor_ohm:g} ohm, I0={motor.i0_a:g} A)"
    )
    print(
        f"Propeller: {prop.name} (D={prop.diameter_m:.4f} m, P={prop.pitch_m:.4f} m)"
        f"{' [LOW CONFIDENCE - parametric fallback]' if prop.is_low_confidence else ''}"
    )
    print(
        f"Battery: {project.battery.series}S{project.battery.parallel}P, "
        f"capacity {project.battery.capacity_ah:.2f} Ah"
    )
    print(
        f"Environment: field elev {project.aircraft.field_elevation_m:g} m, "
        f"rho={requirement.rho_kg_m3:.4f} kg/m^3, a={requirement.speed_of_sound_m_s:.1f} m/s"
    )

    estimate = closed_form_thrust_estimate(
        mass_kg=project.aircraft.mass_kg,
        wing_area_m2=project.aircraft.wing_area_m2,
        v_t_m_s=project.aircraft.v_t_m_s,
        distance_allowed_m=project.aircraft.distance_allowed_m,
        cd=project.aircraft.cd0_ground,
        cl=project.aircraft.cl_ground,
        mu=project.aircraft.mu,
        rho_kg_m3=requirement.rho_kg_m3,
    )
    print("\n--- Closed-form thrust-required estimate ---")
    print(
        f"V_eff={estimate.v_eff_m_s:.2f} m/s  q_bar={estimate.q_bar_pa:.1f} Pa  "
        f"T_req~={estimate.thrust_required_n:.1f} N"
    )
    print(
        f"  inertia {estimate.inertia_n:5.1f} N ({estimate.inertia_pct:4.1f}%)   "
        f"drag {estimate.drag_n:5.1f} N ({estimate.drag_pct:4.1f}%)   "
        f"friction {estimate.friction_n:5.1f} N ({estimate.friction_pct:4.1f}%)"
    )

    spec = CandidateSpec(
        motor=motor,
        prop=prop,
        r_esc_ohm=project.battery.esc_r_ohm,
        esc_current_cont_a=project.battery.esc_current_cont_a,
        motor_count=motor_count,
    )
    result = evaluate_candidate(spec, requirement, battery)
    if motor_count > 1:
        print(f"Motor count: {motor_count} (identical motors sharing one pack)")

    print("\n--- Ground roll ---")
    if result.ground_roll.success:
        print(
            f"STATUS: PASS   distance={result.distance_m:.2f} m  "
            f"(allowed {requirement.distance_allowed_m:.1f} m, margin {result.distance_margin_pct:+.1f}%)"
        )
        print(f"  time={result.time_s:.2f} s   capacity used={result.capacity_used_mah:.1f} mAh")
    else:
        print("STATUS: FAIL (distance = inf)")
        print(f"  reason: {result.ground_roll.reason}")

    op_at_vt = (
        result.ground_roll.operating_points[-1] if result.ground_roll.operating_points else None
    )
    if isinstance(op_at_vt, OperatingPoint):
        print("\n--- Operating point at V_t ---")
        current_desc = (
            f"I={op_at_vt.current_a:.2f} A/motor  I_pack={op_at_vt.current_pack_a:.2f} A"
            if motor_count > 1
            else f"I={op_at_vt.current_a:.2f} A"
        )
        print(
            f"n={op_at_vt.n_rev_s * 60:.0f} rpm  J={op_at_vt.j:.3f}  T={op_at_vt.thrust_n:.2f} N  "
            f"{current_desc}  V={op_at_vt.voltage_v:.2f} V"
        )
        print(f"eta_motor={op_at_vt.eta_motor:.3f}  eta_prop={op_at_vt.eta_prop:.3f}")

    print("\n--- Filters ---")
    for f in result.filters:
        status = "PASS" if f.passed else "FAIL"
        marker = "  " if f.evaluated else "(-)"
        print(f"[{status}]{marker} {f.name:18s} {f.detail}")

    if result.momentum_theory_warning:
        print(f"\n[WARNING] {result.momentum_theory_warning}")

    if not result.eligible:
        overall = "NOT ELIGIBLE (a hard/disqualifying constraint failed)"
    elif result.all_pass:
        overall = "ELIGIBLE, ALL FILTERS PASS"
    else:
        overall = "ELIGIBLE, but a soft objective was missed (see FAIL lines above)"
    print(f"\nOVERALL: {overall}")


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motor", required=True, help="Motor name in the motor library JSON")
    parser.add_argument(
        "--motors-json", default="propselect/data/motors.json", help="Path to motor library JSON"
    )
    parser.add_argument("--prop", required=True, help="Path to a .dat (APC), .csv, or UIUC prop file")
    parser.add_argument("--prop-diameter-in", type=float, default=None)
    parser.add_argument("--prop-pitch-in", type=float, default=None)
    parser.add_argument("--expected-rpm", type=float, default=None)
    parser.add_argument(
        "--motor-count", type=int, default=1,
        help="Number of identical motors sharing one pack (default 1)",
    )

    parser.add_argument("--mass", type=float, default=8.0, help="Aircraft mass [kg]")
    parser.add_argument("--wing-area", type=float, default=0.65, help="Wing area [m^2]")
    parser.add_argument("--span", type=float, default=2.0, help="Wingspan [m]")
    parser.add_argument("--cd", type=float, default=0.09, help="Ground-roll C_D")
    parser.add_argument("--cl", type=float, default=0.60, help="Ground-roll C_L")
    parser.add_argument("--mu", type=float, default=0.10, help="Rolling friction coefficient")
    parser.add_argument("--wheel-height", type=float, default=0.05)
    parser.add_argument("--ground-effect", action="store_true")
    parser.add_argument("--vt", type=float, default=15.0, help="Target/rotation velocity [m/s]")
    parser.add_argument("--distance", type=float, default=30.5, help="Allowed takeoff distance [m]")
    parser.add_argument("--elevation", type=float, default=0.0, help="Field elevation [m]")
    parser.add_argument("--temp-c", type=float, default=None, help="Field OAT [degC]")

    parser.add_argument("--series", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--r-cell", type=float, default=0.006, help="Per-cell internal resistance [ohm]")
    parser.add_argument("--capacity", type=float, default=3.0, help="Pack capacity [Ah]")
    parser.add_argument("--esc-r", type=float, default=0.0)
    parser.add_argument("--esc-current-cont", type=float, default=None)
    parser.add_argument("--c-rate-limit", type=float, default=None)

    parser.add_argument("--tip-mach-limit", type=float, default=0.75)
    parser.add_argument("--motor-eta-threshold", type=float, default=0.75)
    parser.add_argument("--power-limit", type=float, default=None)
    parser.add_argument("--dv", type=float, default=0.1, help="Velocity integration step [m/s]")


def run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="propselect cli")
    _add_cli_args(parser)
    args = parser.parse_args(argv)

    motor = _load_motor(args)
    prop = _load_propeller(args)
    project = _build_project(args)

    _print_summary(motor, prop, project, motor_count=args.motor_count)
    return 0


def run_gui() -> int:
    from propselect.gui.main_window import run_app

    return run_app()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        return run_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        return run_gui()

    print("Usage: propselect {cli|gui} ...", file=sys.stderr)
    print("  propselect cli --motor NAME --prop PATH [options]", file=sys.stderr)
    print("  propselect gui", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
