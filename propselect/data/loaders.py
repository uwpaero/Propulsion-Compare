"""Propeller data loaders: APC PERFILES_WEB ``.dat``, UIUC static/dynamic
files, and generic CSV. All produce a ``propselect.core.propeller.PropellerDataTable``.

Silent unit errors here poison everything downstream, so every loader either
produces data it has cross-checked, or raises ``PropellerImportError`` with a
specific reason -- it never returns a silently-wrong curve.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from propselect.core.motor import MotorSpec
from propselect.core.propeller import PropellerDataTable

# --- unit conversion constants (all exact by definition) ---
IN_TO_M: float = 0.0254  # international inch -> meter
MPH_TO_MPS: float = 0.44704  # mile/hour -> meter/second
LBF_TO_N: float = 4.4482216152605  # pound-force -> newton
HP_TO_W: float = 745.6998715822702  # mechanical horsepower -> watt (550 ft*lbf/s)
IN_LBF_TO_NM: float = 0.1129848290276167  # inch-pound-force -> newton-meter

# Standard sea-level air density [kg/m^3], per ISA. APC's own SI power/thrust
# columns in PERFILES_WEB .dat files are computed assuming this density.
STANDARD_SEA_LEVEL_RHO_KG_M3: float = 1.225

# Recomputed-thrust-vs-file-thrust tolerance for import validation (spec: 2%).
THRUST_VALIDATION_TOLERANCE: float = 0.02

# CT(J) disagreement tolerance between RPM blocks before it's flagged as a
# Reynolds-number effect rather than measurement noise (spec: ~5%).
RPM_BLOCK_COLLAPSE_TOLERANCE: float = 0.05

# CT magnitude floor for the RPM-collapse comparison: near the zero-thrust
# crossing, relative differences blow up (dividing by a near-zero CT) without
# reflecting a real physical disagreement, so such points are excluded.
COLLAPSE_CHECK_CT_FLOOR: float = 0.02

_RPM_HEADER_RE = re.compile(r"PROP\s+RPM\s*=\s*([\d.]+)")
_NAME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)")


class PropellerImportError(Exception):
    """Raised when an import fails validation and is refused."""


@dataclass(frozen=True)
class LoadResult:
    """A successfully loaded propeller, plus any non-fatal warnings."""

    prop: PropellerDataTable
    warnings: list[str]


def parse_diameter_pitch_from_name(name: str) -> tuple[float, float]:
    """Parse a 'DxP' propeller name (e.g. '11x7E', '9x4.7') into (diameter_in, pitch_in)."""
    match = _NAME_RE.search(name)
    if not match:
        raise ValueError(f"Could not parse diameter x pitch from propeller name: {name!r}")
    return float(match.group(1)), float(match.group(2))


# --- APC PERFILES_WEB .dat ---


@dataclass(frozen=True)
class _RawApcData:
    name: str
    diameter_in: float
    pitch_in: float
    blocks: dict[float, dict[str, list[float]]]


def _parse_apc_dat_text(text: str) -> _RawApcData:
    lines = text.splitlines()
    first_nonblank = next((line for line in lines if line.strip()), "")
    name = first_nonblank.strip().split()[0] if first_nonblank.strip() else "unknown"
    diameter_in, pitch_in = parse_diameter_pitch_from_name(name)

    blocks: dict[float, dict[str, list[float]]] = {}
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        match = _RPM_HEADER_RE.search(lines[i])
        if not match:
            i += 1
            continue
        rpm = float(match.group(1))
        i += 1
        while i < n_lines and lines[i].strip() == "":
            i += 1
        if i < n_lines and lines[i].strip().startswith("V"):
            i += 1  # column header row
        if i < n_lines and lines[i].strip().startswith("("):
            i += 1  # units row

        j_vals: list[float] = []
        ct_vals: list[float] = []
        cp_vals: list[float] = []
        thrust_n_vals: list[float] = []
        power_w_vals: list[float] = []
        while i < n_lines and lines[i].strip() != "" and not _RPM_HEADER_RE.search(lines[i]):
            parts = lines[i].split()
            if len(parts) >= 11:
                try:
                    j = float(parts[1])
                    ct = float(parts[3])
                    cp = float(parts[4])
                    power_w = float(parts[8])
                    thrust_n = float(parts[10])
                except ValueError:
                    i += 1
                    continue
                j_vals.append(j)
                ct_vals.append(ct)
                cp_vals.append(cp)
                power_w_vals.append(power_w)
                thrust_n_vals.append(thrust_n)
            i += 1

        if j_vals:
            blocks[rpm] = {
                "j": j_vals,
                "ct": ct_vals,
                "cp": cp_vals,
                "power_w": power_w_vals,
                "thrust_n": thrust_n_vals,
            }

    return _RawApcData(name=name, diameter_in=diameter_in, pitch_in=pitch_in, blocks=blocks)


def _validate_apc_thrust_recomputation(
    diameter_m: float, blocks: dict[float, dict[str, list[float]]]
) -> tuple[bool, str | None]:
    """Recompute thrust from CT and compare against the file's own thrust column."""
    rel_errors: list[float] = []
    for rpm, data in blocks.items():
        n_rev_s = rpm / 60.0
        for ct, thrust_file_n in zip(data["ct"], data["thrust_n"]):
            if abs(thrust_file_n) < 0.05:  # near-zero thrust: relative error is meaningless
                continue
            thrust_calc_n = ct * STANDARD_SEA_LEVEL_RHO_KG_M3 * n_rev_s**2 * diameter_m**4
            rel_errors.append(abs(thrust_calc_n - thrust_file_n) / abs(thrust_file_n))

    if not rel_errors:
        return True, None
    mean_err = sum(rel_errors) / len(rel_errors)
    if mean_err > THRUST_VALIDATION_TOLERANCE:
        return False, (
            f"Recomputed thrust from Ct disagrees with the file's own thrust column by "
            f"{mean_err * 100:.1f}% on average (tolerance {THRUST_VALIDATION_TOLERANCE * 100:.0f}%). "
            "This usually means a different Ct/Cp normalization convention or unit system than "
            f"assumed here (rho={STANDARD_SEA_LEVEL_RHO_KG_M3} kg/m^3 standard sea level, "
            "Ct = T / (rho * n^2 * D^4))."
        )
    return True, None


def _check_rpm_block_collapse(blocks: dict[float, dict[str, list[float]]]) -> list[str]:
    """Compare CT(J) across RPM blocks at common J values; flag disagreement as a Reynolds effect."""
    rpms = sorted(blocks.keys())
    if len(rpms) < 2:
        return []
    warnings: list[str] = []
    base_rpm = rpms[0]
    base = blocks[base_rpm]
    for rpm in rpms[1:]:
        blk = blocks[rpm]
        j_lo = max(min(base["j"]), min(blk["j"]))
        j_hi = min(max(base["j"]), max(blk["j"]))
        if j_hi <= j_lo:
            continue
        j_samples = np.linspace(j_lo, j_hi, 10)
        ct_base = np.interp(j_samples, base["j"], base["ct"])
        ct_blk = np.interp(j_samples, blk["j"], blk["ct"])
        keep = np.abs(ct_base) >= COLLAPSE_CHECK_CT_FLOOR
        if not np.any(keep):
            continue
        rel_diff = np.abs(ct_base[keep] - ct_blk[keep]) / np.abs(ct_base[keep])
        max_rel_diff = float(np.max(rel_diff))
        if max_rel_diff > RPM_BLOCK_COLLAPSE_TOLERANCE:
            warnings.append(
                f"RPM {base_rpm:.0f} and {rpm:.0f} Ct(J) curves disagree by up to "
                f"{max_rel_diff * 100:.1f}% (> {RPM_BLOCK_COLLAPSE_TOLERANCE * 100:.0f}%) over "
                f"J in [{j_lo:.3f}, {j_hi:.3f}] -- likely a Reynolds-number effect."
            )
    return warnings


def _select_rpm_block(blocks: dict[float, dict[str, list[float]]], expected_rpm: float | None) -> float:
    rpms = sorted(blocks.keys())
    if expected_rpm is None:
        return rpms[len(rpms) // 2]
    return min(rpms, key=lambda r: abs(r - expected_rpm))


def _merge_duplicate_j(
    j_arr: np.ndarray, ct_arr: np.ndarray, cp_arr: np.ndarray, decimals: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average CT/CP for near-identical J values (from overlapping RPM blocks)."""
    groups: dict[float, list[tuple[float, float]]] = {}
    for j, ct, cp in zip(j_arr, ct_arr, cp_arr):
        key = round(float(j), decimals)
        groups.setdefault(key, []).append((float(ct), float(cp)))
    keys = sorted(groups)
    j_out = np.array(keys)
    ct_out = np.array([sum(p[0] for p in groups[k]) / len(groups[k]) for k in keys])
    cp_out = np.array([sum(p[1] for p in groups[k]) / len(groups[k]) for k in keys])
    return j_out, ct_out, cp_out


def _consolidate_rpm_blocks(
    blocks: dict[float, dict[str, list[float]]], expected_rpm: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Consolidate RPM blocks into one CT(J), CP(J) curve, per the collapse check."""
    collapse_warnings = _check_rpm_block_collapse(blocks)
    if collapse_warnings:
        chosen_rpm = _select_rpm_block(blocks, expected_rpm)
        chosen = blocks[chosen_rpm]
        j_arr = np.array(chosen["j"])
        ct_arr = np.array(chosen["ct"])
        cp_arr = np.array(chosen["cp"])
        note = (
            f"RPM blocks disagree by more than {RPM_BLOCK_COLLAPSE_TOLERANCE * 100:.0f}%; "
            f"using the RPM={chosen_rpm:.0f} block only (nearest to the expected operating RPM) "
            "rather than averaging blindly."
        )
        return j_arr, ct_arr, cp_arr, collapse_warnings + [note]

    all_j: list[float] = []
    all_ct: list[float] = []
    all_cp: list[float] = []
    for rpm in sorted(blocks):
        all_j.extend(blocks[rpm]["j"])
        all_ct.extend(blocks[rpm]["ct"])
        all_cp.extend(blocks[rpm]["cp"])
    j_arr, ct_arr, cp_arr = _merge_duplicate_j(np.array(all_j), np.array(all_ct), np.array(all_cp))
    return j_arr, ct_arr, cp_arr, []


def load_apc_dat_file(path: str | Path, expected_rpm: float | None = None) -> LoadResult:
    """Load an APC PERFILES_WEB ``.dat`` file (English units, RPM blocks).

    Args:
        path: path to the ``.dat`` file.
        expected_rpm: expected operating RPM, used to pick the nearest block
            if the RPM blocks fail the collapse check (Reynolds effect).
    """
    path = Path(path)
    text = path.read_text()
    raw = _parse_apc_dat_text(text)
    if not raw.blocks:
        raise PropellerImportError(f"No RPM data blocks found in {path}")

    diameter_m = raw.diameter_in * IN_TO_M
    pitch_m = raw.pitch_in * IN_TO_M

    ok, err_msg = _validate_apc_thrust_recomputation(diameter_m, raw.blocks)
    if not ok:
        raise PropellerImportError(f"{path}: {err_msg}")

    j_arr, ct_arr, cp_arr, warnings = _consolidate_rpm_blocks(raw.blocks, expected_rpm)
    prop = PropellerDataTable(
        j_arr, ct_arr, cp_arr, diameter_m=diameter_m, pitch_m=pitch_m, name=raw.name, source=str(path)
    )
    return LoadResult(prop=prop, warnings=warnings)


# --- UIUC propeller data site (static/dynamic) ---


def load_uiuc_file(
    path: str | Path, diameter_in: float | None = None, pitch_in: float | None = None
) -> LoadResult:
    """Load a UIUC propeller data site file (columns include J, CT, CP[, eta])."""
    path = Path(path)
    lines = [line for line in path.read_text().splitlines() if line.strip()]

    header_tokens: list[str] | None = None
    data_start = 0
    for idx, line in enumerate(lines):
        tokens = [t.upper() for t in line.split()]
        if "J" in tokens and "CT" in tokens and "CP" in tokens:
            header_tokens = tokens
            data_start = idx + 1
            break

    j_vals: list[float] = []
    ct_vals: list[float] = []
    cp_vals: list[float] = []

    if header_tokens is not None:
        j_idx = header_tokens.index("J")
        ct_idx = header_tokens.index("CT")
        cp_idx = header_tokens.index("CP")
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) <= max(j_idx, ct_idx, cp_idx):
                continue
            try:
                j_vals.append(float(parts[j_idx]))
                ct_vals.append(float(parts[ct_idx]))
                cp_vals.append(float(parts[cp_idx]))
            except ValueError:
                continue
    else:
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                j_vals.append(float(parts[0]))
                ct_vals.append(float(parts[1]))
                cp_vals.append(float(parts[2]))
            except ValueError:
                continue

    if not j_vals:
        raise PropellerImportError(f"Could not find J/CT/CP data in {path}")

    name = path.stem
    if diameter_in is None or pitch_in is None:
        parsed_d, parsed_p = parse_diameter_pitch_from_name(name)
        diameter_in = diameter_in if diameter_in is not None else parsed_d
        pitch_in = pitch_in if pitch_in is not None else parsed_p

    prop = PropellerDataTable(
        j_vals,
        ct_vals,
        cp_vals,
        diameter_m=diameter_in * IN_TO_M,
        pitch_m=pitch_in * IN_TO_M,
        name=name,
        source=str(path),
    )
    return LoadResult(prop=prop, warnings=[])


# --- generic CSV ---


def load_csv_file(
    path: str | Path, diameter_in: float, pitch_in: float, name: str | None = None
) -> LoadResult:
    """Load a generic CSV with columns J, CT, CP (case-insensitive)."""
    path = Path(path)
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    required = ["j", "ct", "cp"]
    missing = [c for c in required if c not in cols_lower]
    if missing:
        raise PropellerImportError(f"CSV {path} missing required columns: {missing}")

    j_vals = df[cols_lower["j"]].to_numpy(dtype=float)
    ct_vals = df[cols_lower["ct"]].to_numpy(dtype=float)
    cp_vals = df[cols_lower["cp"]].to_numpy(dtype=float)

    prop = PropellerDataTable(
        j_vals,
        ct_vals,
        cp_vals,
        diameter_m=diameter_in * IN_TO_M,
        pitch_m=pitch_in * IN_TO_M,
        name=name or path.stem,
        source=str(path),
    )
    return LoadResult(prop=prop, warnings=[])


# --- motor library JSON ---


def motor_from_dict(entry: dict) -> MotorSpec:
    """Build a ``MotorSpec`` from one motor-library JSON entry."""
    return MotorSpec(
        name=entry["name"],
        kv_rpm_per_v=entry["Kv"],
        r_motor_ohm=entry["R_motor"],
        i0_a=entry["I0"],
        i_max_cont_a=entry.get("I_max_cont"),
        i_max_burst_a=entry.get("I_max_burst"),
        mass_kg=entry.get("mass"),
        shaft_dia_mm=entry.get("shaft_dia"),
        source_url=entry.get("source_url"),
        notes=entry.get("notes"),
    )


def motor_to_dict(m: MotorSpec) -> dict:
    """Serialize a ``MotorSpec`` to one motor-library JSON entry."""
    return {
        "name": m.name,
        "Kv": m.kv_rpm_per_v,
        "R_motor": m.r_motor_ohm,
        "I0": m.i0_a,
        "I_max_cont": m.i_max_cont_a,
        "I_max_burst": m.i_max_burst_a,
        "mass": m.mass_kg,
        "shaft_dia": m.shaft_dia_mm,
        "source_url": m.source_url,
        "notes": m.notes,
    }


def load_motor_library(path: str | Path) -> list[MotorSpec]:
    """Load a motor library JSON file into a list of ``MotorSpec``."""
    entries = json.loads(Path(path).read_text())
    return [motor_from_dict(entry) for entry in entries]


def save_motor_library(path: str | Path, motors: list[MotorSpec]) -> None:
    """Save a list of ``MotorSpec`` to a motor library JSON file."""
    entries = [motor_to_dict(m) for m in motors]
    Path(path).write_text(json.dumps(entries, indent=2))


# --- propeller model <-> plain dict (for embedding in a project file) ---


def prop_to_dict(prop: PropellerDataTable) -> dict:
    """Serialize a ``PropellerDataTable``'s full tabulated curve to a plain dict.

    Only ``PropellerDataTable`` (the tabulated model) is supported -- the GUI's
    propeller library never holds ``ParametricPropellerModel`` instances.
    """
    return {
        "name": prop.name,
        "diameter_m": prop.diameter_m,
        "pitch_m": prop.pitch_m,
        "source": prop.source,
        "j": prop._j.tolist(),
        "ct": prop._ct.tolist(),
        "cp": prop._cp.tolist(),
    }


def prop_from_dict(entry: dict) -> PropellerDataTable:
    """Rebuild a ``PropellerDataTable`` from a dict produced by ``prop_to_dict``."""
    return PropellerDataTable(
        entry["j"],
        entry["ct"],
        entry["cp"],
        diameter_m=entry["diameter_m"],
        pitch_m=entry["pitch_m"],
        name=entry.get("name", ""),
        source=entry.get("source", ""),
    )
