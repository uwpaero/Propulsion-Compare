"""Ground roll integration.

    F_net(V) = T(V) - D(V) - mu*(m*g - L(V))
    D(V) = 0.5*rho*V^2*S*C_D_ground
    L(V) = 0.5*rho*V^2*S*C_L_ground
    ds = m*V*dV / F_net(V)
    dt = m*dV   / F_net(V)

Integrated with the trapezoidal rule on a fixed dV grid. If F_net(V) <= 0
anywhere, the result reports distance = infinity, the airspeed at which it
went non-positive, and the force deficit at that point -- never a plausible
looking wrong number.

The design is split in two layers:

* ``integrate_kinematics`` -- pure trapezoidal ds/dt integration from
  precomputed thrust/drag/lift arrays. No motor, propeller, or battery
  coupling; this is what the analytical check (T=const, D=L=mu=0) exercises
  directly.
* ``integrate_ground_roll`` -- the full coupled version. Battery SoC sag
  depends on cumulative current draw, which depends on the (still unknown)
  time profile, which depends on thrust, which for the internal-resistance
  battery model depends on SoC. This is solved with two fixed-point passes:
  the first assumes constant initial SoC to get an approximate current/time
  profile, the second re-solves operating points against the SoC sag implied
  by that profile. SoC sag over a single takeoff run is small enough that
  this converges to a good approximation without an inner iterative solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from propselect.core.atmosphere import G0
from propselect.core.battery import Battery, soc_from_capacity_used
from propselect.core.motor import MotorSpec
from propselect.core.operating_point import (
    OperatingPointFailure,
    OperatingPointResult,
    sweep_operating_points,
)
from propselect.core.propeller import PropellerModel

# Number of fixed-point passes used to converge SoC-driven voltage sag against
# the current/time profile. SoC sag over one takeoff run is small (typically
# well under 1% of pack capacity), so this converges quickly; two passes is
# enough for the sag to be captured without an inner iterative solve per step.
SOC_REFINEMENT_PASSES: int = 2

# Default velocity integration step, per the build spec.
DEFAULT_DV_M_S: float = 0.1


@dataclass(frozen=True)
class AircraftConfig:
    """Aircraft and ground-roll requirement parameters.

    Attributes:
        mass_kg: aircraft mass [kg].
        wing_area_m2: reference wing area S [m^2].
        span_m: wingspan [m].
        cd0_ground: baseline (parasite) ground-roll drag coefficient. Used
            directly as C_D_ground when ``ground_effect`` is False; used as
            C_D0 in the ground-effect correction otherwise.
        cl_ground: ground-roll lift coefficient.
        mu: rolling friction coefficient.
        wheel_height_m: main-wheel height above ground [m], for ground effect.
        oswald_efficiency: span efficiency factor e, dimensionless.
        ground_effect: whether to apply the induced-drag ground-effect
            correction. Off by default.
    """

    mass_kg: float
    wing_area_m2: float
    span_m: float
    cd0_ground: float
    cl_ground: float
    mu: float
    wheel_height_m: float = 0.0
    oswald_efficiency: float = 0.8
    ground_effect: bool = False

    @property
    def aspect_ratio(self) -> float:
        """AR = b^2 / S -- the exact definition, always derived from span and area."""
        return self.span_m**2 / self.wing_area_m2


def induced_drag_ground_effect_factor(wheel_height_m: float, span_m: float) -> float:
    """Phi = (16h/b)^2 / (1 + (16h/b)^2), dimensionless in [0, 1)."""
    ratio = 16.0 * wheel_height_m / span_m
    return ratio**2 / (1.0 + ratio**2)


def effective_cd_ground(aircraft: AircraftConfig) -> float:
    """C_D_ground, with the optional ground-effect induced-drag correction.

    C_D_ground = C_D0 + Phi * C_L^2 / (pi*AR*e), when ground_effect is enabled.
    """
    if not aircraft.ground_effect:
        return aircraft.cd0_ground
    phi = induced_drag_ground_effect_factor(aircraft.wheel_height_m, aircraft.span_m)
    induced_term = phi * aircraft.cl_ground**2 / (math.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    return aircraft.cd0_ground + induced_term


def aerodynamic_drag_n(rho_kg_m3: float, v_m_s: float, wing_area_m2: float, cd: float) -> float:
    """D = 0.5*rho*V^2*S*C_D  [N]."""
    return 0.5 * rho_kg_m3 * v_m_s**2 * wing_area_m2 * cd


def aerodynamic_lift_n(rho_kg_m3: float, v_m_s: float, wing_area_m2: float, cl: float) -> float:
    """L = 0.5*rho*V^2*S*C_L  [N]."""
    return 0.5 * rho_kg_m3 * v_m_s**2 * wing_area_m2 * cl


@dataclass(frozen=True)
class ClosedFormEstimate:
    """Closed-form thrust-required estimate for the live Tab 1 display.

        V_eff = V_t / sqrt(2)
        q_bar = 0.5*rho*V_eff^2
        T_req ~= m*V_t^2/(2*s) + q_bar*S*C_D + mu*(m*g - q_bar*S*C_L)

    This is a rough single-point estimate (constant-acceleration, mean
    dynamic pressure), not a substitute for the full integration -- expect
    the two to differ, but not by a large factor.
    """

    inertia_n: float
    drag_n: float
    friction_n: float
    lift_n: float
    q_bar_pa: float
    v_eff_m_s: float
    thrust_required_n: float
    inertia_pct: float
    drag_pct: float
    friction_pct: float


def closed_form_thrust_estimate(
    mass_kg: float,
    wing_area_m2: float,
    v_t_m_s: float,
    distance_allowed_m: float,
    cd: float,
    cl: float,
    mu: float,
    rho_kg_m3: float,
    g: float = G0,
) -> ClosedFormEstimate:
    """Closed-form thrust-required estimate with the inertia/drag/friction breakdown."""
    v_eff = v_t_m_s / math.sqrt(2.0)
    q_bar = 0.5 * rho_kg_m3 * v_eff**2
    inertia = mass_kg * v_t_m_s**2 / (2.0 * distance_allowed_m)
    drag = q_bar * wing_area_m2 * cd
    lift = q_bar * wing_area_m2 * cl
    friction = mu * (mass_kg * g - lift)
    t_req = inertia + drag + friction
    return ClosedFormEstimate(
        inertia_n=inertia,
        drag_n=drag,
        friction_n=friction,
        lift_n=lift,
        q_bar_pa=q_bar,
        v_eff_m_s=v_eff,
        thrust_required_n=t_req,
        inertia_pct=(inertia / t_req * 100.0) if t_req else 0.0,
        drag_pct=(drag / t_req * 100.0) if t_req else 0.0,
        friction_pct=(friction / t_req * 100.0) if t_req else 0.0,
    )


def net_accelerating_force_n(
    thrust_n: float, drag_n: float, lift_n: float, mu: float, mass_kg: float, g: float = G0
) -> float:
    """F_net = T - D - mu*(m*g - L)  [N]."""
    return thrust_n - drag_n - mu * (mass_kg * g - lift_n)


@dataclass(frozen=True)
class KinematicsResult:
    """Result of the pure trapezoidal ground-roll kinematics integration.

    Attributes:
        success: False if F_net went non-positive before the end of the grid.
        distance_m: total ground-roll distance [m], or +inf on failure.
        time_s: total ground-roll time [s], or +inf on failure.
        distance_profile_m: cumulative distance at each grid point [m].
        time_profile_s: cumulative time at each grid point [s].
        net_force_n: F_net at each grid point reached [N].
        stall_v_m_s: airspeed where F_net first went non-positive, if any.
        deficit_n: magnitude of the force shortfall at that airspeed, if any.
        reason: human-readable failure diagnostic, if any.
    """

    success: bool
    distance_m: float
    time_s: float
    distance_profile_m: list[float]
    time_profile_s: list[float]
    net_force_n: list[float]
    stall_v_m_s: float | None = None
    deficit_n: float | None = None
    reason: str | None = None


def integrate_kinematics(
    v_grid_m_s: list[float] | np.ndarray,
    thrust_n: list[float] | np.ndarray,
    drag_n: list[float] | np.ndarray,
    lift_n: list[float] | np.ndarray,
    mass_kg: float,
    mu: float,
) -> KinematicsResult:
    """Integrate ds = m*V*dV/F_net and dt = m*dV/F_net with the trapezoidal rule."""
    v = np.asarray(v_grid_m_s, dtype=float)
    thrust = np.asarray(thrust_n, dtype=float)
    drag = np.asarray(drag_n, dtype=float)
    lift = np.asarray(lift_n, dtype=float)
    n_points = len(v)
    if not (len(thrust) == len(drag) == len(lift) == n_points):
        raise ValueError("v_grid_m_s, thrust_n, drag_n, lift_n must be equal length")
    if n_points < 2:
        raise ValueError("Need at least two grid points to integrate")

    fnet = np.array(
        [net_accelerating_force_n(thrust[i], drag[i], lift[i], mu, mass_kg) for i in range(n_points)]
    )

    if fnet[0] <= 0.0:
        return KinematicsResult(
            success=False,
            distance_m=math.inf,
            time_s=math.inf,
            distance_profile_m=[0.0],
            time_profile_s=[0.0],
            net_force_n=[float(fnet[0])],
            stall_v_m_s=float(v[0]),
            deficit_n=float(-fnet[0]),
            reason=(
                f"Net accelerating force non-positive at V={v[0]:.2f} m/s "
                f"(deficit {-fnet[0]:.2f} N); cannot begin takeoff roll."
            ),
        )

    distance_profile = [0.0]
    time_profile = [0.0]
    net_force_list = [float(fnet[0])]
    distance_m = 0.0
    time_s = 0.0

    for i in range(1, n_points):
        if fnet[i] <= 0.0:
            return KinematicsResult(
                success=False,
                distance_m=math.inf,
                time_s=time_s,
                distance_profile_m=distance_profile,
                time_profile_s=time_profile,
                net_force_n=net_force_list,
                stall_v_m_s=float(v[i]),
                deficit_n=float(-fnet[i]),
                reason=(
                    f"Net accelerating force went non-positive at V={v[i]:.2f} m/s "
                    f"(deficit {-fnet[i]:.2f} N); thrust-limited before reaching target speed."
                ),
            )
        dv = float(v[i] - v[i - 1])
        ds_prev = mass_kg * v[i - 1] / fnet[i - 1]
        ds_curr = mass_kg * v[i] / fnet[i]
        dt_prev = mass_kg / fnet[i - 1]
        dt_curr = mass_kg / fnet[i]
        distance_m += 0.5 * (ds_prev + ds_curr) * dv
        time_s += 0.5 * (dt_prev + dt_curr) * dv
        distance_profile.append(distance_m)
        time_profile.append(time_s)
        net_force_list.append(float(fnet[i]))

    return KinematicsResult(
        success=True,
        distance_m=distance_m,
        time_s=time_s,
        distance_profile_m=distance_profile,
        time_profile_s=time_profile,
        net_force_n=net_force_list,
        stall_v_m_s=None,
        deficit_n=None,
        reason=None,
    )


def _cumulative_amp_hours(current_a: list[float], time_s: list[float]) -> list[float]:
    """Trapezoidal cumulative amp-hours of current_a[i] against time_s[i]."""
    ah = [0.0]
    for i in range(1, len(current_a)):
        dt = time_s[i] - time_s[i - 1]
        d_ah = 0.5 * (current_a[i - 1] + current_a[i]) * dt / 3600.0
        ah.append(ah[-1] + d_ah)
    return ah


@dataclass(frozen=True)
class GroundRollResult:
    """Full coupled ground-roll result.

    Attributes:
        success: False if the operating-point solver or the kinematics
            integration failed anywhere on the grid.
        distance_m: ground-roll distance [m], or +inf on failure.
        time_s: ground-roll time [s], or +inf on failure.
        capacity_used_ah: cumulative amp-hours drawn over the roll [Ah].
        v_grid_m_s: the velocity grid used [m/s].
        operating_points: solved (or failed) operating point at each grid point.
        drag_n: aerodynamic drag at each grid point [N].
        lift_n: aerodynamic lift at each grid point [N].
        distance_profile_m: cumulative distance at each grid point [m].
        time_profile_s: cumulative time at each grid point [s].
        soc_profile: state of charge at each grid point.
        stall_v_m_s: airspeed of first failure, if any.
        deficit_n: force deficit at that airspeed, if any.
        reason: human-readable failure diagnostic, if any.
    """

    success: bool
    distance_m: float
    time_s: float
    capacity_used_ah: float
    v_grid_m_s: list[float]
    operating_points: list[OperatingPointResult]
    drag_n: list[float]
    lift_n: list[float]
    distance_profile_m: list[float]
    time_profile_s: list[float]
    soc_profile: list[float]
    stall_v_m_s: float | None = None
    deficit_n: float | None = None
    reason: str | None = None


def integrate_ground_roll(
    v_t_m_s: float,
    rho_kg_m3: float,
    aircraft: AircraftConfig,
    prop: PropellerModel,
    motor: MotorSpec,
    battery: Battery,
    r_esc_ohm: float = 0.0,
    dv_m_s: float = DEFAULT_DV_M_S,
    initial_soc: float = 1.0,
    motor_count: int = 1,
) -> GroundRollResult:
    """Integrate the takeoff ground roll from 0 to ``v_t_m_s``.

    With ``motor_count`` identical motors sharing one pack, thrust (and hence
    the ground-roll kinematics) is already the aggregate across all motors
    (see ``OperatingPoint.thrust_n``), while SoC/amp-hour depletion is driven
    by the pack current (``OperatingPoint.current_pack_a``), not any single
    motor's current.
    """
    if v_t_m_s <= 0.0:
        raise ValueError("v_t_m_s must be positive")
    if dv_m_s <= 0.0:
        raise ValueError("dv_m_s must be positive")

    n_steps = max(1, round(v_t_m_s / dv_m_s))
    v_grid = np.linspace(0.0, v_t_m_s, n_steps + 1)

    cd = effective_cd_ground(aircraft)
    drag_arr = [aerodynamic_drag_n(rho_kg_m3, v, aircraft.wing_area_m2, cd) for v in v_grid]
    lift_arr = [
        aerodynamic_lift_n(rho_kg_m3, v, aircraft.wing_area_m2, aircraft.cl_ground) for v in v_grid
    ]

    capacity_ah = getattr(battery, "capacity_ah", float("nan"))
    soc_profile = [initial_soc] * len(v_grid)
    ops: list[OperatingPointResult] = []
    kin: KinematicsResult | None = None

    for _ in range(SOC_REFINEMENT_PASSES):
        ops = sweep_operating_points(
            v_grid, rho_kg_m3, prop, motor, battery, r_esc_ohm, soc_profile,
            motor_count=motor_count,
        )
        first_failure_idx = next(
            (i for i, op in enumerate(ops) if isinstance(op, OperatingPointFailure)), None
        )
        if first_failure_idx is not None:
            failure = ops[first_failure_idx]
            assert isinstance(failure, OperatingPointFailure)
            return GroundRollResult(
                success=False,
                distance_m=math.inf,
                time_s=math.inf,
                capacity_used_ah=0.0,
                v_grid_m_s=list(v_grid),
                operating_points=ops,
                drag_n=drag_arr,
                lift_n=lift_arr,
                distance_profile_m=[],
                time_profile_s=[],
                soc_profile=soc_profile,
                stall_v_m_s=float(v_grid[first_failure_idx]),
                deficit_n=None,
                reason=(
                    f"Operating point solve failed at V={v_grid[first_failure_idx]:.2f} m/s: "
                    f"{failure.reason}"
                ),
            )

        thrust_arr = [op.thrust_n for op in ops]  # type: ignore[union-attr]
        pack_current_arr = [op.current_pack_a for op in ops]  # type: ignore[union-attr]
        kin = integrate_kinematics(v_grid, thrust_arr, drag_arr, lift_arr, aircraft.mass_kg, aircraft.mu)

        if not kin.success:
            break

        if math.isfinite(capacity_ah) and capacity_ah > 0:
            ah_cum = _cumulative_amp_hours(pack_current_arr, kin.time_profile_s)
            soc_profile = [soc_from_capacity_used(initial_soc, capacity_ah, ah) for ah in ah_cum]
        else:
            break  # no capacity known -- SoC sag cannot be tracked, no point iterating

    assert kin is not None
    pack_current_arr = [
        op.current_pack_a for op in ops if not isinstance(op, OperatingPointFailure)
    ]  # type: ignore[union-attr]

    if not kin.success:
        # kin.time_profile_s only covers the grid points reached before the
        # kinematics failure; truncate pack_current_arr to match before integrating.
        matched_current = pack_current_arr[: len(kin.time_profile_s)]
        capacity_used_ah = (
            _cumulative_amp_hours(matched_current, kin.time_profile_s)[-1]
            if len(kin.time_profile_s) > 1
            else 0.0
        )
        return GroundRollResult(
            success=False,
            distance_m=math.inf,
            time_s=kin.time_s,
            capacity_used_ah=capacity_used_ah,
            v_grid_m_s=list(v_grid),
            operating_points=ops,
            drag_n=drag_arr,
            lift_n=lift_arr,
            distance_profile_m=kin.distance_profile_m,
            time_profile_s=kin.time_profile_s,
            soc_profile=soc_profile,
            stall_v_m_s=kin.stall_v_m_s,
            deficit_n=kin.deficit_n,
            reason=kin.reason,
        )

    ah_cum = _cumulative_amp_hours(pack_current_arr, kin.time_profile_s)

    return GroundRollResult(
        success=True,
        distance_m=kin.distance_m,
        time_s=kin.time_s,
        capacity_used_ah=ah_cum[-1],
        v_grid_m_s=list(v_grid),
        operating_points=ops,
        drag_n=drag_arr,
        lift_n=lift_arr,
        distance_profile_m=kin.distance_profile_m,
        time_profile_s=kin.time_profile_s,
        soc_profile=soc_profile,
        stall_v_m_s=None,
        deficit_n=None,
        reason=None,
    )
