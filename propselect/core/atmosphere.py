"""Atmospheric density and speed of sound.

ISA (International Standard Atmosphere), troposphere only (<11 km), with an
optional temperature offset, or direct entry of density/temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

# Ratio of specific heats for air (dimensionless).
GAMMA_AIR: float = 1.4

# Specific gas constant for dry air [J/(kg*K)].
R_AIR: float = 287.05

# Standard gravity, used in the ISA pressure lapse formula [m/s^2].
G0: float = 9.80665

# ISA sea-level reference temperature [K] (15 degC).
T0_ISA: float = 288.15

# ISA sea-level reference pressure [Pa].
P0_ISA: float = 101325.0

# ISA tropospheric temperature lapse rate, valid below 11 km [K/m].
LAPSE_RATE: float = 0.0065


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric conditions at a point.

    Attributes:
        density: kg/m^3
        pressure: Pa
        temperature: K
        speed_of_sound: m/s
    """

    density: float
    pressure: float
    temperature: float
    speed_of_sound: float


def celsius_to_kelvin(temperature_c: float) -> float:
    """Convert Celsius to Kelvin [K]."""
    return temperature_c + 273.15


def isa_temperature(pressure_altitude_m: float, dISA_k: float = 0.0) -> float:
    """ISA temperature at a pressure altitude, plus an offset [K].

    Args:
        pressure_altitude_m: pressure altitude [m], valid below 11000 m.
        dISA_k: temperature offset from the standard ISA profile [K].
    """
    return T0_ISA + dISA_k - LAPSE_RATE * pressure_altitude_m


def isa_pressure(pressure_altitude_m: float) -> float:
    """ISA static pressure at a pressure altitude [Pa], valid below 11000 m."""
    exponent = G0 / (R_AIR * LAPSE_RATE)
    return P0_ISA * (1.0 - LAPSE_RATE * pressure_altitude_m / T0_ISA) ** exponent


def speed_of_sound(temperature_k: float) -> float:
    """Speed of sound at a given temperature [m/s]."""
    return sqrt(GAMMA_AIR * R_AIR * temperature_k)


def density_from_pressure_temp(pressure_pa: float, temperature_k: float) -> float:
    """Ideal-gas density from pressure and temperature [kg/m^3]."""
    return pressure_pa / (R_AIR * temperature_k)


def isa_offset_atmosphere(
    pressure_altitude_m: float,
    actual_temperature_k: float | None = None,
    dISA_k: float = 0.0,
) -> AtmosphereState:
    """Atmosphere from pressure altitude, using either an actual OAT or an ISA offset.

    This is the standard "field elevation + outside air temperature" convention:
    pressure comes from the standard ISA profile at the given pressure altitude,
    while density uses whichever temperature is more specific. If
    ``actual_temperature_k`` is supplied it is used directly (this is the normal
    case: user enters field elevation and measured OAT). Otherwise the ISA
    standard temperature profile plus ``dISA_k`` is used.

    Args:
        pressure_altitude_m: pressure altitude [m].
        actual_temperature_k: measured outside air temperature [K], if known.
        dISA_k: temperature offset from ISA standard [K], used only when
            ``actual_temperature_k`` is not given.
    """
    pressure_pa = isa_pressure(pressure_altitude_m)
    if actual_temperature_k is not None:
        temperature_k = actual_temperature_k
    else:
        temperature_k = isa_temperature(pressure_altitude_m, dISA_k)
    density = density_from_pressure_temp(pressure_pa, temperature_k)
    a = speed_of_sound(temperature_k)
    return AtmosphereState(
        density=density, pressure=pressure_pa, temperature=temperature_k, speed_of_sound=a
    )


def direct_atmosphere(density_kg_m3: float, temperature_k: float) -> AtmosphereState:
    """Atmosphere from directly entered density and temperature.

    Pressure is back-computed from the ideal gas law for consistency in the
    returned state; it is not otherwise used.
    """
    pressure_pa = density_kg_m3 * R_AIR * temperature_k
    a = speed_of_sound(temperature_k)
    return AtmosphereState(
        density=density_kg_m3, pressure=pressure_pa, temperature=temperature_k, speed_of_sound=a
    )
