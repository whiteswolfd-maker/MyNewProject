"""Read-only 50 mm real-d3plot diagnostic.

This script does not modify the extractor, the contact-air data contract, the
network, the losses, the optimizer or the configuration.  It reads the frames
already extracted by ``cli.extract_profiles``, runs the current per-frame
boundary/separation diagnostics, and writes an auditable table.

The ambient-tolerance sensitivity below is diagnostic only.  It is never used
to build a cache and the configuration is not changed.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import _precursor_config
from spherical_blast_pinn.contact_air.data import (
    classify_separation_frames,
    contact_interval_from_band,
    contact_interval_from_material,
    detect_right_boundary_report,
    select_air_native_resolution,
)
from spherical_blast_pinn.contact_air.geometry import (
    ContactIntervalTable,
    RightBoundaryTrajectory,
)
from spherical_blast_pinn.contact_air.precursor import compute_precursor
from spherical_blast_pinn.data.extracted_case import load_extracted_case
from spherical_blast_pinn.data.kfile_parser import parse_deck


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "outputs" / "diagnostic_50mm_20261105" / "profile_v2"
OUT = ROOT / "outputs" / "diagnostic_50mm_20261105"
CONFIG_PATH = ROOT / "configs" / "contact_air" / "base.yaml"
DECK = Path("F:/tnt/tnt_files/dp0/SYS/MECH/input.k")
D3PLOT = Path("F:/tnt/tnt_files/dp0/SYS/MECH")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def build_contact(case, material: dict[str, np.ndarray]):
    """Reproduce the contact-bracket construction in build_case_from_profile."""
    phase = np.asarray(case.material_phase)
    band_inner = np.asarray(material["contact_mixing_band_inner_m"], dtype=np.float64)
    band_outer = np.asarray(material["contact_mixing_band_outer_m"], dtype=np.float64)
    alpha = np.asarray(material["alpha_product"], dtype=np.float64)
    mixed_frames = (alpha > 0.05) & (alpha < 0.95)
    eligible = np.asarray(material["contact_has_supervision"], dtype=bool)
    n = case.times.size
    low = band_inner.copy()
    up = band_outer.copy()
    raw = 0.5 * (band_inner + band_outer)
    ok = np.zeros(n, dtype=bool)
    reasons: list[str] = []
    for frame in range(n):
        lo, hi, rr, good, reason = contact_interval_from_band(
            case.radius, phase[frame], float(band_inner[frame]), float(band_outer[frame])
        )
        if not good:
            lo, hi, rr, good, reason = contact_interval_from_material(
                case.radius, phase[frame], mixed=mixed_frames[frame]
            )
        if good:
            low[frame], up[frame], raw[frame] = lo, hi, rr
            ok[frame] = True
        reasons.append(reason)
    valid = eligible & np.isfinite(low) & np.isfinite(up) & (low < up) & ok
    table = ContactIntervalTable(
        case_id=case.case_id,
        times=np.asarray(case.times, dtype=np.float64),
        low=low,
        up=up,
        raw=raw,
        valid=valid,
    )
    return table, low, up, valid, reasons


def strict_boundary_reports(case, contact_up, valid, tolerances):
    reports = []
    for frame in range(case.times.size):
        report = detect_right_boundary_report(
            case.radius,
            case.pressure[frame],
            case.velocity[frame],
            case.density[frame],
            case.material_phase[frame],
            ambient_pressure=float(tolerances["ambient_pressure_pa"]),
            ambient_density=float(tolerances["ambient_density_kg_m3"]),
            tol_pressure=float(tolerances["pressure_pa"]),
            tol_density=float(tolerances["density_kg_m3"]),
            tol_velocity=float(tolerances["velocity_m_s"]),
            min_ambient_cells=int(tolerances.get("min_ambient_cells", 2)),
            valid_density=case.field_validity["density"][frame],
            valid_velocity=case.field_validity["velocity"][frame],
            valid_pressure=case.field_validity["pressure"][frame],
            contact_radius=float(contact_up[frame]) if bool(valid[frame]) else None,
            material_is_phase=True,
        )
        reports.append(report)
    return reports


def sensitivity_boundary_reports(case, contact_up, valid):
    """Diagnostic-only ambient tolerance; never written into a cache.

    The measured initial air density is 1.216-1.225 kg/m3 (0.7% radial
    variation), so the 1e-9 kg/m3 configured tolerance cannot identify an
    ambient run.  This sensitivity uses a 0.005 kg/m3 absolute density
    tolerance (0.408% of 1.225) and 1e-4 m/s velocity to ask whether the
    boundary evidence itself is otherwise present.  It is reported as a
    diagnostic, not as an accepted setting.
    """
    reports = []
    for frame in range(case.times.size):
        report = detect_right_boundary_report(
            case.radius,
            case.pressure[frame],
            case.velocity[frame],
            case.density[frame],
            case.material_phase[frame],
            ambient_pressure=101320.0,
            ambient_density=1.225,
            tol_pressure=0.10132,
            tol_density=0.005,
            tol_velocity=1.0e-4,
            min_ambient_cells=2,
            valid_density=case.field_validity["density"][frame],
            valid_velocity=case.field_validity["velocity"][frame],
            valid_pressure=case.field_validity["pressure"][frame],
            contact_radius=float(contact_up[frame]) if bool(valid[frame]) else None,
            material_is_phase=True,
        )
        reports.append(report)
    return reports


def separation_for(case, contact, reports, tolerances, native, sources, reasons, coverage):
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id=case.case_id,
        times=np.asarray(case.times, dtype=np.float64),
        radii=np.asarray([report.radius for report in reports], dtype=np.float64),
        valid=np.asarray([report.identified for report in reports], dtype=bool),
    )
    return classify_separation_frames(
        case_id=case.case_id,
        times=np.asarray(case.times, dtype=np.float64),
        radius=case.radius,
        material=case.material_phase,
        density=case.density,
        velocity=case.velocity,
        pressure=case.pressure,
        contact=contact,
        boundary=boundary,
        boundary_reports=reports,
        valid_density=case.field_validity["density"],
        valid_velocity=case.field_validity["velocity"],
        valid_pressure=case.field_validity["pressure"],
        ambient_pressure=float(tolerances["ambient_pressure_pa"])
        if tolerances is not None else 101320.0,
        ambient_density=float(tolerances["ambient_density_kg_m3"])
        if tolerances is not None else 1.225,
        tol_pressure=float(tolerances["pressure_pa"]) if tolerances is not None else 0.10132,
        tol_density=float(tolerances["density_kg_m3"]) if tolerances is not None else 0.005,
        tol_velocity=float(tolerances["velocity_m_s"]) if tolerances is not None else 1.0e-4,
        min_shocked_air_samples=int(tolerances.get("min_shocked_air_samples", 2))
        if tolerances is not None else 2,
        air_native_resolution_m=native,
        air_native_resolution_source=sources,
        air_native_resolution_reason=reasons,
        air_coverage=coverage,
        min_air_coverage=float(tolerances.get("min_air_coverage", 0.5))
        if tolerances is not None else 0.5,
        min_air_layer_native_cells=float(tolerances.get("min_air_layer_native_cells", 1.0))
        if tolerances is not None else 1.0,
    )


def deck_material_summary() -> dict[str, Any]:
    summary = parse_deck(DECK, geometry=False)
    explosive = summary.require_explosive()
    jwl = summary.require_jwl()
    # The deck is mm-s-t.  These factors are the same ones extract_profiles uses.
    pressure = 1.0e6
    density = 1.0e12
    velocity = 1.0e-3
    air_cards = [card for card in summary.materials.values() if card.kind == "null"]
    air_eos = [card for card in summary.eos.values() if card.kind == "linear_polynomial"]
    air_card = air_cards[0]
    air_eos_card = air_eos[0]
    return {
        "deck_path": str(DECK),
        "unit_system": "mm_s_t",
        "initial_detonation": list(summary.initial_detonation or ()),
        "initial_detonation_time": summary.initial_detonation_time,
        "termination_time": summary.termination_time,
        "d3plot_interval_s": summary.d3plot_interval,
        "detonation_velocity_m_s": float(explosive.detonation_velocity) * velocity,
        "cj_pressure_pa": float(explosive.p_cj) * pressure,
        "explosive_density_kg_m3": float(explosive.density) * density,
        "jwl": {
            "A_pa": float(jwl.parameters["A"]) * pressure,
            "B_pa": float(jwl.parameters["B"]) * pressure,
            "C_pa": None,
            "R1": float(jwl.parameters["R1"]),
            "R2": float(jwl.parameters["R2"]),
            "omega": float(jwl.parameters["omeg"]),
            "E0_deck": float(jwl.parameters["E0"]),
            "V0": float(jwl.parameters["V0"]),
        },
        "air": {
            "density_kg_m3": float(air_card.density) * density,
            "eos_kind": air_eos_card.kind,
            "C0": float(air_eos_card.parameters["C0"]),
            "C1": float(air_eos_card.parameters["C1"]),
            "C2": float(air_eos_card.parameters["C2"]),
            "C3": float(air_eos_card.parameters["C3"]),
            "C4": float(air_eos_card.parameters["C4"]),
            "C5": float(air_eos_card.parameters["C5"]),
            "C6": float(air_eos_card.parameters["C6"]),
            "E0_deck": float(air_eos_card.parameters["E0"]),
            "V0": float(air_eos_card.parameters["V0"]),
        },
        "parts": {
            str(part.part_id): {
                "name": part.name,
                "material_id": part.material_id,
                "eos_id": part.eos_id,
            }
            for part in summary.parts.values()
        },
    }


def frame_counts(case, material, frame: int) -> dict[str, Any]:
    phase = np.asarray(case.material_phase[frame])
    r = np.asarray(case.radius)
    phase_air = phase == 1
    fv_d = np.asarray(case.field_validity["density"][frame], dtype=bool)
    fv_u = np.asarray(case.field_validity["velocity"][frame], dtype=bool)
    fv_p = np.asarray(case.field_validity["pressure"][frame], dtype=bool)
    state_ok = (
        fv_d & fv_u & fv_p
        & np.isfinite(case.density[frame])
        & np.isfinite(case.velocity[frame])
        & np.isfinite(case.pressure[frame])
    )
    # Training observations use the legacy material label and the legacy
    # ``valid`` mask.  ``phase_air`` is a diagnostic field, not a training
    # eligibility mask.
    material1 = np.asarray(case.material[frame]) == 1
    location = (
        np.asarray(case.training_location_valid[frame], dtype=bool)
        if getattr(case, "training_location_valid", None) is not None
        else np.asarray(case.valid[frame], dtype=bool)
    )
    train_air = material1 & location
    old_valid_air = material1 & np.asarray(case.valid[frame], dtype=bool)
    return {
        "n_phase_air": int(np.count_nonzero(phase_air)),
        "n_material1": int(np.count_nonzero(material1)),
        "n_material1_field_valid_all": int(np.count_nonzero(material1 & state_ok)),
        "n_field_valid_air_all": int(np.count_nonzero(phase_air & state_ok)),
        "n_field_valid_air_rho": int(np.count_nonzero(phase_air & fv_d)),
        "n_field_valid_air_u": int(np.count_nonzero(phase_air & fv_u)),
        "n_field_valid_air_p": int(np.count_nonzero(phase_air & fv_p)),
        "n_training_location_mat1": int(np.count_nonzero(train_air)),
        "n_old_valid_air": int(np.count_nonzero(old_valid_air)),
        "n_training_valid_air": int(np.count_nonzero(train_air)),
        "n_training_valid_air_rho": int(np.count_nonzero(train_air & fv_d)),
        "n_training_valid_air_u": int(np.count_nonzero(train_air & fv_u)),
        "n_training_valid_air_p": int(np.count_nonzero(train_air & fv_p)),
        "n_air_withdrawn": int(np.count_nonzero(phase_air & ~np.asarray(case.valid[frame], dtype=bool))),
        "state_ok": state_ok,
        "phase_air": phase_air,
        "r": r,
    }


def ambient_metrics(case, material, frame: int, counts: dict[str, Any], tol) -> dict[str, Any]:
    r = counts["r"]
    phase_air = counts["phase_air"]
    state_ok = counts["state_ok"]
    p = np.asarray(case.pressure[frame], dtype=np.float64)
    rho = np.asarray(case.density[frame], dtype=np.float64)
    u = np.asarray(case.velocity[frame], dtype=np.float64)
    p_ok = state_ok & (np.abs(p - float(tol["ambient_pressure_pa"])) <= float(tol["pressure_pa"]))
    rho_ok = state_ok & (np.abs(rho - float(tol["ambient_density_kg_m3"])) <= float(tol["density_kg_m3"]))
    u_ok = state_ok & (np.abs(u) <= float(tol["velocity_m_s"]))
    ambient = state_ok & p_ok & rho_ok & u_ok
    idx = np.flatnonzero(phase_air & state_ok)
    outer = idx[-100:] if idx.size else idx
    return {
        "n_phase_air_p_tol_ok": int(np.count_nonzero(phase_air & p_ok)),
        "n_phase_air_rho_tol_ok": int(np.count_nonzero(phase_air & rho_ok)),
        "n_phase_air_u_tol_ok": int(np.count_nonzero(phase_air & u_ok)),
        "n_phase_air_all_tol_ok": int(np.count_nonzero(phase_air & ambient)),
        "outer100_r_m": None if not outer.size else float(np.median(r[outer])),
        "outer100_rho_median": None if not outer.size else float(np.median(rho[outer])),
        "outer100_p_median": None if not outer.size else float(np.median(p[outer])),
        "outer100_u_median": None if not outer.size else float(np.median(u[outer])),
        "outer100_rho_abs_offset": (
            None if not outer.size else float(np.median(rho[outer]) - float(tol["ambient_density_kg_m3"]))
        ),
        "outer100_p_abs_offset": (
            None if not outer.size else float(np.median(p[outer]) - float(tol["ambient_pressure_pa"]))
        ),
        "outer100_u_abs": None if not outer.size else float(abs(np.median(u[outer]))),
    }


def first_valid_air(case, material, frame: int, contact_up: float) -> dict[str, Any]:
    phase = np.asarray(case.material_phase[frame])
    r = np.asarray(case.radius)
    fv = (
        np.asarray(case.field_validity["density"][frame], dtype=bool)
        & np.asarray(case.field_validity["velocity"][frame], dtype=bool)
        & np.asarray(case.field_validity["pressure"][frame], dtype=bool)
        & np.isfinite(case.density[frame])
        & np.isfinite(case.velocity[frame])
        & np.isfinite(case.pressure[frame])
    )
    candidates = np.flatnonzero((phase == 1) & (r >= contact_up))
    valid_candidates = candidates[fv[candidates]]
    first = int(valid_candidates[0]) if valid_candidates.size else None
    gap = None if first is None else float(r[first] - contact_up)
    invalid_between = (
        int(np.count_nonzero((phase[candidates] == 1) & ~fv[candidates]
                             & (r[candidates] < r[first])))
        if first is not None else int(candidates.size)
    )
    contam = np.asarray(material["contaminated_shell_fraction"][frame], dtype=np.float64)
    return {
        "first_valid_air_radius_m": None if first is None else float(r[first]),
        "first_valid_air_gap_m": gap,
        "n_invalid_air_shells_before_first_valid": invalid_between,
        "contaminated_fraction_at_contact_up": (
            float(contam[candidates[0]]) if candidates.size else None
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG_PATH)
    section = config.section("contact_air")
    tolerances = dict(section["tolerances"])
    precursor = compute_precursor(_precursor_config(section, 0.05))
    case = load_extracted_case(PROFILE)
    with np.load(PROFILE / "material.npz", allow_pickle=False) as archive:
        material = {key: np.array(archive[key]) for key in archive.files}
    with np.load(PROFILE / "raw_line.npz", allow_pickle=False) as archive:
        raw = {key: np.array(archive[key]) for key in archive.files}

    contact, contact_low, contact_up, contact_valid, contact_reasons = build_contact(case, material)
    native, native_sources, native_reasons = select_air_native_resolution(
        case.times.size,
        primary=material["shock_cell_thickness_m"],
        primary_source="shock_cell_thickness_m",
        fallback=material["air_native_resolution_fallback_m"],
        fallback_applicable=material["air_native_resolution_fallback_applicable"],
        fallback_source="air_native_resolution_fallback_m",
    )
    strict_reports = strict_boundary_reports(case, contact_up, contact_valid, tolerances)
    sensitivity_reports = sensitivity_boundary_reports(case, contact_up, contact_valid)
    strict_sep = separation_for(
        case, contact, strict_reports, tolerances, native, native_sources, native_reasons,
        material["coverage_air"],
    )
    sens_tol = {
        "ambient_pressure_pa": 101320.0,
        "ambient_density_kg_m3": 1.225,
        "pressure_pa": 0.10132,
        "density_kg_m3": 0.005,
        "velocity_m_s": 1.0e-4,
        "min_shocked_air_samples": 2,
        "min_air_coverage": 0.5,
        "min_air_layer_native_cells": 1.0,
    }
    sensitivity_sep = separation_for(
        case, contact, sensitivity_reports, sens_tol, native, native_sources, native_reasons,
        material["coverage_air"],
    )

    frames: list[dict[str, Any]] = []
    for i, t in enumerate(case.times):
        counts = frame_counts(case, material, i)
        ambient = ambient_metrics(case, material, i, counts, tolerances)
        first = first_valid_air(case, material, i, float(contact_up[i]))
        strict = strict_reports[i]
        sensitive = sensitivity_reports[i]
        strict_frame = strict_sep["frames"][i]
        sens_frame = sensitivity_sep["frames"][i]
        shock_r = float(material["shock_radius"][i])
        shock_ok = bool(np.isfinite(shock_r))
        layer_to_shock = shock_r - float(contact_up[i]) if shock_ok else None
        air_to_shock = (
            counts["phase_air"] & (np.asarray(case.radius) >= float(contact_up[i]))
            & (np.asarray(case.radius) < shock_r)
        ) if shock_ok and shock_r > float(contact_up[i]) else None
        if air_to_shock is not None:
            n_air_to_shock = int(np.count_nonzero(air_to_shock))
            n_air_to_shock_valid = int(np.count_nonzero(air_to_shock & counts["state_ok"]))
            n_air_to_shock_rho = int(np.count_nonzero(
                air_to_shock & np.asarray(case.field_validity["density"][i], dtype=bool)
            ))
            n_air_to_shock_u = int(np.count_nonzero(
                air_to_shock & np.asarray(case.field_validity["velocity"][i], dtype=bool)
            ))
            n_air_to_shock_p = int(np.count_nonzero(
                air_to_shock & np.asarray(case.field_validity["pressure"][i], dtype=bool)
            ))
        else:
            n_air_to_shock = 0
            n_air_to_shock_valid = 0
            n_air_to_shock_rho = 0
            n_air_to_shock_u = 0
            n_air_to_shock_p = 0
        frames.append({
            "selected_index": i,
            "state_index": int(material["state_index"][i]),
            "time_s": float(t),
            "time_us": float(t) * 1.0e6,
            "contact_low_m": float(contact_low[i]),
            "contact_up_m": float(contact_up[i]),
            "contact_raw_m": float(contact.raw[i]),
            "contact_valid": bool(contact_valid[i]),
            "contact_reconstruction_radius_m": float(material["contact_radius"][i]),
            "contact_band_width_m": float(contact_up[i] - contact_low[i]),
            "contact_interval_half_width_m": 0.5 * float(contact_up[i] - contact_low[i]),
            "contact_mixing_cell_count": int(material["contact_mixing_cell_count"][i]),
            "contact_reconstruction_status": str(material["contact_reconstruction_status"][i]),
            "extractor_contact_position_uncertainty_m": float(
                material["contact_position_uncertainty_m"][i]
            ),
            "contact_reason": contact_reasons[i],
            "shock_radius_m": (
                float(material["shock_radius"][i])
                if np.isfinite(float(material["shock_radius"][i])) else None
            ),
            "shock_status": str(material["shock_status"][i]),
            "shock_has_supervision": bool(material["shock_has_supervision"][i]),
            "contact_up_ge_shock_radius": bool(shock_ok and float(contact_up[i]) >= shock_r),
            "shock_minus_contact_up_m": layer_to_shock,
            "n_air_between_contact_up_and_shock": n_air_to_shock,
            "n_field_valid_air_between_contact_up_and_shock": n_air_to_shock_valid,
            "n_field_valid_air_rho_between_contact_up_and_shock": n_air_to_shock_rho,
            "n_field_valid_air_u_between_contact_up_and_shock": n_air_to_shock_u,
            "n_field_valid_air_p_between_contact_up_and_shock": n_air_to_shock_p,
            "shock_cell_thickness_m": (
                float(material["shock_cell_thickness_m"][i])
                if np.isfinite(float(material["shock_cell_thickness_m"][i])) else None
            ),
            "air_native_fallback_m": (
                float(material["air_native_resolution_fallback_m"][i])
                if np.isfinite(float(material["air_native_resolution_fallback_m"][i])) else None
            ),
            "air_native_fallback_applicable": bool(material["air_native_resolution_fallback_applicable"][i]),
            "air_native_resolution_m": (
                float(native[i]) if np.isfinite(float(native[i])) else None
            ),
            "air_native_resolution_source": str(native_sources[i]),
            "air_native_resolution_reason": str(native_reasons[i]),
            "rb_candidate_strict_m": (
                float(strict.candidate_radius) if np.isfinite(strict.candidate_radius) else None
            ),
            "rb_verified_strict": bool(strict.identified),
            "rb_strict_radius_m": float(strict.radius) if np.isfinite(strict.radius) else None,
            "rb_strict_verification": strict.verification,
            "rb_strict_reasons": list(strict.reasons),
            "separation_strict_status": str(strict_frame["status"]),
            "separation_strict_reasons": list(strict_frame["reasons"]),
            "rb_candidate_sensitivity_m": (
                float(sensitive.candidate_radius) if np.isfinite(sensitive.candidate_radius) else None
            ),
            "rb_verified_sensitivity": bool(sensitive.identified),
            "rb_sensitivity_radius_m": (
                float(sensitive.radius) if np.isfinite(sensitive.radius) else None
            ),
            "rb_sensitivity_verification": sensitive.verification,
            "separation_sensitivity_status": str(sens_frame["status"]),
            "separation_sensitivity_reasons": list(sens_frame["reasons"]),
            "n_phase_air": counts["n_phase_air"],
            "n_material1": counts["n_material1"],
            "n_material1_field_valid_all": counts["n_material1_field_valid_all"],
            "n_training_location_mat1": counts["n_training_location_mat1"],
            "n_old_valid_air": counts["n_old_valid_air"],
            "n_field_valid_air_all": counts["n_field_valid_air_all"],
            "n_field_valid_air_rho": counts["n_field_valid_air_rho"],
            "n_field_valid_air_u": counts["n_field_valid_air_u"],
            "n_field_valid_air_p": counts["n_field_valid_air_p"],
            "n_training_valid_air": counts["n_training_valid_air"],
            "n_training_valid_air_rho": counts["n_training_valid_air_rho"],
            "n_training_valid_air_u": counts["n_training_valid_air_u"],
            "n_training_valid_air_p": counts["n_training_valid_air_p"],
            "n_air_withdrawn": counts["n_air_withdrawn"],
            "withdrawal_cause": (
                "no_confirmed_shock (all air labels withdrawn)"
                if not bool(material["shock_has_supervision"][i])
                else "outside confirmed shock (outer air labels withdrawn)"
            ),
            **ambient,
            **first,
        })

    ambient_t0 = {}
    idx0 = 0
    phase0 = np.asarray(case.material_phase[idx0])
    ok0 = (
        (phase0 == 1)
        & np.asarray(case.field_validity["density"][idx0], dtype=bool)
        & np.asarray(case.field_validity["velocity"][idx0], dtype=bool)
        & np.asarray(case.field_validity["pressure"][idx0], dtype=bool)
        & np.isfinite(case.pressure[idx0])
    )
    r0 = np.asarray(case.radius)[ok0]
    rho0 = np.asarray(case.density[idx0])[ok0]
    p0 = np.asarray(case.pressure[idx0])[ok0]
    u0 = np.asarray(case.velocity[idx0])[ok0]
    ambient_t0 = {
        "time_s": 0.0,
        "n_phase_air_field_valid": int(r0.size),
        "radius_min_m": float(r0.min()) if r0.size else None,
        "radius_max_m": float(r0.max()) if r0.size else None,
        "rho_min_kg_m3": float(rho0.min()) if r0.size else None,
        "rho_max_kg_m3": float(rho0.max()) if r0.size else None,
        "rho_relative_span": (
            float((rho0.max() - rho0.min()) / float(tolerances["ambient_density_kg_m3"]))
            if r0.size else None
        ),
        "p_min_pa": float(p0.min()) if p0.size else None,
        "p_max_pa": float(p0.max()) if p0.size else None,
        "u_max_abs_m_s": float(np.max(np.abs(u0))) if u0.size else None,
        "config_ambient_density_kg_m3": float(tolerances["ambient_density_kg_m3"]),
        "config_ambient_pressure_pa": float(tolerances["ambient_pressure_pa"]),
    }
    # Radial samples of the t=0 ambient field for the report/figure.
    sample_radii_mm = [50.2, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0, 300.0,
                       400.0, 500.0, 800.0, 1200.0, 1500.0, 1900.0]
    ambient_samples = []
    for r_mm in sample_radii_mm:
        j = int(np.searchsorted(case.radius, r_mm * 1.0e-3))
        if j >= case.radius.size:
            j = case.radius.size - 1
        ambient_samples.append({
            "radius_m": float(case.radius[j]),
            "rho_kg_m3": float(case.density[idx0, j]),
            "u_m_s": float(case.velocity[idx0, j]),
            "p_pa": float(case.pressure[idx0, j]),
            "phase": int(case.material_phase[idx0, j]),
            "field_valid": bool(
                case.field_validity["density"][idx0, j]
                and case.field_validity["velocity"][idx0, j]
                and case.field_validity["pressure"][idx0, j]
            ),
        })

    deck = deck_material_summary()
    config_material = {
        "jwl": dict(section["precursor"]["jwl"]),
        "detonation": dict(section["precursor"]["detonation"]),
        "air": dict(section["precursor"]["air"]),
        "material_source": section["precursor"].get("material_source"),
        "material_source_verified": section["precursor"].get("material_source_verified"),
    }
    material_comparison = {
        "detonation_velocity_m_s": {
            "deck": deck["detonation_velocity_m_s"],
            "config": float(config_material["detonation"]["velocity"]),
        },
        "cj_pressure_pa": {
            "deck": deck["cj_pressure_pa"],
            "config": float(config_material["detonation"]["cj_pressure"]),
        },
        "rho_tnt_kg_m3": {
            "deck": deck["explosive_density_kg_m3"],
            "config": float(config_material["jwl"]["rho_tnt"]),
        },
        "jwl_A_pa": {"deck": deck["jwl"]["A_pa"], "config": float(config_material["jwl"]["A"])},
        "jwl_B_pa": {"deck": deck["jwl"]["B_pa"], "config": float(config_material["jwl"]["B"])},
        "jwl_C_pa": {"deck": None, "config": float(precursor.C), "source": "closed from p(V_cj)=p_CJ"},
        "jwl_R1": {"deck": deck["jwl"]["R1"], "config": float(config_material["jwl"]["R1"])},
        "jwl_R2": {"deck": deck["jwl"]["R2"], "config": float(config_material["jwl"]["R2"])},
        "jwl_omega": {"deck": deck["jwl"]["omega"], "config": float(config_material["jwl"]["omega"])},
        "air_density_kg_m3": {
            "deck": deck["air"]["density_kg_m3"],
            "config": float(config_material["air"]["density"]),
        },
    }

    p_inf_config = float(tolerances["ambient_pressure_pa"])
    p_inf_data = float(frames[0]["outer100_p_median"])
    pressure_comparison = {
        "p_c_t0_pa": float(precursor.p_c_t0),
        "p_air_IC_shocked_pa": float(precursor.p_sep),
        "p_air_IC_ambient_pa": p_inf_config,
        "p_air_IC_ambient_data_outer_median_pa": p_inf_data,
        "p_c_t0_minus_p_air_IC_shocked_pa": float(precursor.p_c_t0 - precursor.p_sep),
        "p_c_t0_vs_p_air_IC_shocked_relative": float(
            abs(precursor.p_c_t0 - precursor.p_sep) / precursor.p_sep
        ),
        "p_c_t0_over_p_air_IC_shocked": float(precursor.p_c_t0 / precursor.p_sep),
        "p_c_t0_minus_p_air_IC_ambient_pa": float(precursor.p_c_t0 - p_inf_config),
        "p_c_t0_vs_p_air_IC_ambient_relative": float(
            abs(precursor.p_c_t0 - p_inf_config) / p_inf_config
        ),
        "p_c_t0_over_p_air_IC_ambient": float(precursor.p_c_t0 / p_inf_config),
        "p_c_t0_minus_p_air_IC_ambient_data_pa": float(precursor.p_c_t0 - p_inf_data),
        "p_c_t0_vs_p_air_IC_ambient_data_relative": float(
            abs(precursor.p_c_t0 - p_inf_data) / p_inf_data
        ),
        "forced_equal": False,
        "note": (
            "p_air_IC_shocked is the matched air state at separation (p_a_sep); "
            "p_air_IC_ambient is the undisturbed ambient pressure p_inf.  "
            "The 22.22% compatibility difference is against p_a_sep.  The huge "
            "ratio against p_inf is expected for a shocked contact state and is "
            "not a closure target."
        ),
    }

    # Linear interpolations of the contact bracket at the theoretical times.
    t = np.asarray(case.times)
    interp = {}
    for name, values in (
        ("low", np.asarray(contact_low)), ("up", np.asarray(contact_up)),
        ("reconstruction", np.asarray(material["contact_radius"])),
    ):
        interp[name] = {
            "at_t_sep_m": float(PchipInterpolator(t, values)(precursor.t_sep)),
            "at_t0_m": float(PchipInterpolator(t, values)(precursor.t0)),
        }
    frame_before = int(np.searchsorted(t, precursor.t0, side="left") - 1)
    frame_after = int(np.searchsorted(t, precursor.t0, side="left"))

    def crossing(values: np.ndarray, target: float) -> float | None:
        delta = np.asarray(values, dtype=np.float64) - float(target)
        hits = np.flatnonzero(np.sign(delta[:-1]) != np.sign(delta[1:]))
        for hit in hits:
            if t[hit + 1] > t[hit]:
                fraction = -delta[hit] / (delta[hit + 1] - delta[hit])
                return float(t[hit] + fraction * (t[hit + 1] - t[hit]))
        return None

    timing = {
        "initial_data_frame_t_s": 0.0,
        "network_t0_s": float(precursor.t0),
        "theoretical_t_sep_s": float(precursor.t_sep),
        "theoretical_t0_s": float(precursor.t0),
        "t0_definition": "t0 = t_sep + epsilon_s",
        "epsilon_s": float(precursor.epsilon),
        "t_cj_s": float(precursor.t_cj),
        "surface_arrival_ref_s": float(precursor.t_surface_ref),
        "frame_before_t0": {
            "selected_index": frame_before,
            "state_index": int(material["state_index"][frame_before]),
            "time_s": float(t[frame_before]),
            "time_us": float(t[frame_before]) * 1.0e6,
        },
        "frame_after_t0": {
            "selected_index": frame_after,
            "state_index": int(material["state_index"][frame_after]),
            "time_s": float(t[frame_after]),
            "time_us": float(t[frame_after]) * 1.0e6,
        },
        "contact_at_theory": interp,
        "contact_up_crosses_R_sep_s": crossing(contact_up, precursor.R_sep),
        "contact_up_crosses_Rc0_s": crossing(contact_up, precursor.Rc0),
        "contact_up_crosses_Rs0_s": crossing(contact_up, precursor.Rs0),
        "reconstruction_crosses_R_sep_s": crossing(np.asarray(material["contact_radius"]), precursor.R_sep),
        "reconstruction_crosses_Rc0_s": crossing(np.asarray(material["contact_radius"]), precursor.Rc0),
        "reconstruction_crosses_Rs0_s": crossing(np.asarray(material["contact_radius"]), precursor.Rs0),
        "theory_R_sep_m": float(precursor.R_sep),
        "theory_Rc0_m": float(precursor.Rc0),
        "theory_Rs0_m": float(precursor.Rs0),
        "contact_low_at_t0_minus_Rc0_m": float(interp["low"]["at_t0_m"] - precursor.Rc0),
        "contact_up_at_t0_minus_Rc0_m": float(interp["up"]["at_t0_m"] - precursor.Rc0),
        "contact_up_at_t0_minus_Rs0_m": float(interp["up"]["at_t0_m"] - precursor.Rs0),
        "reconstruction_at_t0_minus_Rc0_m": float(
            interp["reconstruction"]["at_t0_m"] - precursor.Rc0
        ),
        "reconstruction_at_t0_minus_Rs0_m": float(
            interp["reconstruction"]["at_t0_m"] - precursor.Rs0
        ),
        "rb_strict_continuous_coverage_at_t0": False,
        "rb_strict_reason": "no frame has a verified strict-tolerance ambient run; the first R_b cache build fails in build_initial_geometry",
        "no_time_shift_applied": True,
    }

    summary = {
        "n_selected_frames": len(frames),
        "n_d3plot_states_available": int(case.source["metadata_raw"]["n_states_available"]),
        "n_d3plot_states_available_source": case.source["metadata_raw"]["n_states_source"],
        "n_strict_rb_verified": int(sum(1 for item in frames if item["rb_verified_strict"])),
        "n_strict_rb_candidates": int(sum(1 for item in frames if item["rb_candidate_strict_m"] is not None)),
        "n_strict_separated": int(sum(1 for item in frames if item["separation_strict_status"] == "separated")),
        "n_strict_unresolvable": int(sum(1 for item in frames if item["separation_strict_status"] == "unresolvable")),
        "n_sensitivity_rb_verified": int(sum(1 for item in frames if item["rb_verified_sensitivity"])),
        "n_sensitivity_separated": int(sum(1 for item in frames if item["separation_sensitivity_status"] == "separated")),
        "n_shock_supervised_frames": int(sum(1 for item in frames if item["shock_has_supervision"])),
        "n_frames_all_air_withdrawn": int(
            sum(1 for item in frames if item["withdrawal_cause"].startswith("no_confirmed_shock"))
        ),
        "n_frames_with_valid_training_air": int(
            sum(1 for item in frames if item["n_training_valid_air"] > 0)
        ),
        "n_frames_contact_up_ge_shock": int(
            sum(1 for item in frames if item["contact_up_ge_shock_radius"])
        ),
        "blocking": [
            "contact/interface resolution: at t0 the contact bracket upper edge is 111.66 mm while state 19's confirmed shock is 109.24 mm (reliable range 103.0-109.24 mm); the current shell representation has no resolved pure-air layer. Raw cells span 20-50 mm; this is a resolution statement, not proof that no physical layer exists",
            "strict ambient verification: 0 ambient cells in every selected frame",
            "air-side field validity: the first pure-air shells immediately outside the contact are contaminated/invalid; first valid air is 126.4 mm at t0 while contact_up is 111.6 mm; no selected frame has a material1 point passing training_location_valid",
            "old shock supervision: only 4 of 16 selected frames have confirmed shock supervision; state 20 (19.987 us, immediately after t0) is doubtful and has all air labels withdrawn",
            "R_b: no strict verified candidate and no continuous R_b segment at t0; actual AirObservationStore.size=0 and final field-valid counts are 0/0/0 because the geometric range is empty, not because a material count was substituted",
        ],
    }

    payload = {
        "diagnostic": "50 mm real d3plot contact_air data-chain diagnosis",
        "date": "2026-11-05",
        "real_data": True,
        "synthetic": False,
        "config_path": str(CONFIG_PATH),
        "profile_dir": str(PROFILE),
        "deck": deck,
        "config_material": config_material,
        "material_comparison": material_comparison,
        "precursor": precursor.as_dict(),
        "timing": timing,
        "pressure_comparison": pressure_comparison,
        "ambient_t0": ambient_t0,
        "ambient_samples_t0": ambient_samples,
        "summary": summary,
        "frames": frames,
        "strict_separation": {
            "status": strict_sep["status"],
            "status_counts": strict_sep["status_counts"],
        },
        "sensitivity_boundary": {
            "note": (
                "diagnostic only: ambient tolerance widened to 0.005 kg/m3 "
                "absolute density (0.408% of 1.225) and 1e-4 m/s velocity because "
                "the measured initial air density spans 1.216-1.225 kg/m3. This is "
                "NOT an accepted configuration and no cache was built from it. "
                "The t=0 false positive is analyzed in raw_review/report.md."
            ),
            "tolerances": sens_tol,
            "separation_status": sensitivity_sep["status"],
            "status_counts": sensitivity_sep["status_counts"],
        },
    }
    (OUT / "diagnostic.json").write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = [
        "selected_index", "state_index", "time_us",
        "contact_low_m", "contact_up_m", "contact_valid",
        "contact_reconstruction_radius_m", "contact_band_width_m",
        "contact_interval_half_width_m", "contact_mixing_cell_count",
        "extractor_contact_position_uncertainty_m",
        "rb_candidate_strict_m", "rb_verified_strict", "rb_strict_radius_m",
        "rb_strict_verification", "separation_strict_status",
        "shock_radius_m", "shock_status", "shock_has_supervision",
        "contact_up_ge_shock_radius", "shock_minus_contact_up_m",
        "n_air_between_contact_up_and_shock",
        "n_field_valid_air_between_contact_up_and_shock",
        "n_field_valid_air_rho_between_contact_up_and_shock",
        "n_field_valid_air_u_between_contact_up_and_shock",
        "n_field_valid_air_p_between_contact_up_and_shock",
        "air_native_resolution_source", "air_native_resolution_m",
        "air_native_fallback_m", "air_native_fallback_applicable",
        "n_phase_air", "n_material1", "n_material1_field_valid_all",
        "n_training_location_mat1", "n_old_valid_air",
        "n_field_valid_air_all", "n_field_valid_air_rho",
        "n_field_valid_air_u", "n_field_valid_air_p",
        "n_training_valid_air", "n_training_valid_air_rho",
        "n_training_valid_air_u", "n_training_valid_air_p",
        "n_air_withdrawn", "withdrawal_cause",
        "n_phase_air_p_tol_ok", "n_phase_air_rho_tol_ok",
        "n_phase_air_u_tol_ok", "n_phase_air_all_tol_ok",
        "outer100_rho_median", "outer100_p_median", "outer100_u_median",
        "first_valid_air_radius_m", "first_valid_air_gap_m",
        "n_invalid_air_shells_before_first_valid",
        "contaminated_fraction_at_contact_up",
        "rb_candidate_sensitivity_m", "rb_verified_sensitivity",
        "rb_sensitivity_radius_m", "separation_sensitivity_status",
    ]
    with (OUT / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in frames:
            writer.writerow({key: item.get(key) for key in columns})

    lines = []
    lines.append("# 50 mm real d3plot contact_air diagnostic")
    lines.append("")
    lines.append(f"- Deck: `{DECK}`")
    lines.append(f"- d3plot: `{D3PLOT}` (779 states available)")
    lines.append(f"- Extracted profile: `{PROFILE}` (16 selected states)")
    lines.append("- Real frames; no synthetic substitution; no time shift.")
    lines.append("")
    lines.append("## Nomenclature")
    lines.append("")
    lines.append("- `t=0` is the first **data** frame; it is not the network start.")
    lines.append(
        "- `t0=19.531333 us` is the network/precursor start, `t0=t_sep+epsilon`."
    )
    lines.append(
        "- `p_air_IC_shocked` is the matched air state at separation (`p_a_sep`); "
        "`p_air_IC_ambient` is the undisturbed ambient pressure `p_inf`."
    )
    lines.append(
        "- `contact_band_width` is `R_c,up-R_c,low`; `contact_interval_half_width` "
        "is half of it.  Neither is an ALE cell thickness and neither is a "
        "statistical uncertainty.  The cell thickness is "
        "`contact_cell_thickness_m`/`shock_cell_thickness_m`; the extractor's "
        "field named `contact_position_uncertainty_m` is reported as an interval "
        "half-width, not as a statistical error bar."
    )
    lines.append(
        "- The diagnostic sensitivity uses 0.005 kg/m3 **absolute** density "
        "tolerance, which is 0.408% of 1.225, not 0.5%."
    )
    lines.append("")
    lines.append("## Theory and clock")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key in (
        "initial_data_frame_t_s", "network_t0_s",
        "theoretical_t_sep_s", "theoretical_t0_s", "epsilon_s", "t_cj_s",
        "surface_arrival_ref_s", "theory_R_sep_m", "theory_Rc0_m", "theory_Rs0_m",
        "contact_up_crosses_R_sep_s", "contact_up_crosses_Rc0_s",
        "contact_up_crosses_Rs0_s", "reconstruction_crosses_R_sep_s",
        "reconstruction_crosses_Rc0_s", "reconstruction_crosses_Rs0_s",
        "contact_low_at_t0_minus_Rc0_m", "contact_up_at_t0_minus_Rc0_m",
        "contact_up_at_t0_minus_Rs0_m", "reconstruction_at_t0_minus_Rc0_m",
        "reconstruction_at_t0_minus_Rs0_m",
    ):
        value = timing[key]
        lines.append(f"| {key} | {'—' if value is None else f'{value:.9g}'} |")
    lines.append("")
    lines.append(
        f"Frame before t0: state {timing['frame_before_t0']['state_index']} at "
        f"{timing['frame_before_t0']['time_us']:.6f} us; frame after t0: state "
        f"{timing['frame_after_t0']['state_index']} at "
        f"{timing['frame_after_t0']['time_us']:.6f} us."
    )
    lines.append("")
    lines.append("## Pressure comparison")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in pressure_comparison.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Per-frame table (strict current configuration)")
    lines.append("")
    lines.append(
        "| # | state | t (us) | contact low/up (mm) | R_s (mm) | R_s - R_cup (mm) | "
        "R_b cand/verified | R_b failure | sep | native source/value (mm) | "
        "training-location rho/u/p | diagnostic field-valid rho/u/p | air withdrawn | first valid air (mm) |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|---|---|---|---:|---:|---:|---:|")
    for item in frames:
        rb_cand = "—" if item["rb_candidate_strict_m"] is None else f"{item['rb_candidate_strict_m']*1e3:.3f}"
        rb_reason = (
            "no finite ambient run under configured density/velocity tolerances"
            if not item["rb_verified_strict"] else "verified"
        )
        native_value = "—" if item["air_native_resolution_m"] is None else f"{item['air_native_resolution_m']*1e3:.3f}"
        first_valid = (
            "—" if item["first_valid_air_radius_m"] is None
            else f"{item['first_valid_air_radius_m']*1e3:.2f}"
        )
        shock = "—" if item["shock_radius_m"] is None else f"{item['shock_radius_m']*1e3:.2f}"
        shock_gap = (
            "—" if item["shock_minus_contact_up_m"] is None
            else f"{item['shock_minus_contact_up_m']*1e3:+.2f}"
        )
        lines.append(
            f"| {item['selected_index']} | {item['state_index']} | {item['time_us']:.3f} | "
            f"{item['contact_low_m']*1e3:.2f}/{item['contact_up_m']*1e3:.2f} | "
            f"{shock} | {shock_gap} | "
            f"{rb_cand}/{item['rb_verified_strict']} | {rb_reason} | "
            f"{item['separation_strict_status']} | "
            f"{item['air_native_resolution_source']}/{native_value} | "
            f"{item['n_training_valid_air_rho']}/{item['n_training_valid_air_u']}/{item['n_training_valid_air_p']} | "
            f"{item['n_field_valid_air_rho']}/{item['n_field_valid_air_u']}/{item['n_field_valid_air_p']} | "
            f"{item['n_air_withdrawn']} | {first_valid} |"
        )
    lines.append("")
    lines.append(
        "The `training-location` columns are `legacy material==1 & "
        "training_location_valid & field mask`.  They are **not** final training "
        "counts.  Because the strict R_b is absent, "
        "`build_air_observations` has no [R_c,raw, R_b] geometric interval and the "
        "actual observation store is empty on every frame.  See `raw_review/` for "
        "the staged material/field/protection/geometry counts."
    )
    lines.append("")
    lines.append("## Ambient verification")
    lines.append("")
    lines.append(f"- t=0 ambient field-valid shells: {ambient_t0['n_phase_air_field_valid']}")
    lines.append(f"- radial range: {ambient_t0['radius_min_m']} .. {ambient_t0['radius_max_m']} m")
    lines.append(
        f"- rho: {ambient_t0['rho_min_kg_m3']} .. {ambient_t0['rho_max_kg_m3']} kg/m3 "
        f"(relative span {ambient_t0['rho_relative_span']:.6g})"
    )
    lines.append(f"- p: {ambient_t0['p_min_pa']} .. {ambient_t0['p_max_pa']} Pa")
    lines.append(f"- max |u|: {ambient_t0['u_max_abs_m_s']} m/s")
    lines.append(
        f"- strict per-frame ambient counts: p/rho/u/all = "
        f"{frames[0]['n_phase_air_p_tol_ok']}/"
        f"{frames[0]['n_phase_air_rho_tol_ok']}/"
        f"{frames[0]['n_phase_air_u_tol_ok']}/"
        f"{frames[0]['n_phase_air_all_tol_ok']} at t=0"
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "The blocking item is **not a d3plot read failure**: all 16 selected real "
        "states read, and the air fields are finite where the extractor marks "
        "them valid. The first-order blockers are data-side: (a) the contact "
        "mixing band is about 40 mm wide and its upper edge can equal or exceed "
        "the detected shock (t0: 111.66 mm vs 109.24 mm), so there is no "
        "resolved shocked-air layer.  This is a bracket width, not an ALE cell "
        "thickness and not a statistical uncertainty: at state 19 the bracket is "
        "40.20 mm, the contact cell thickness is 21.57 mm, the shock cell "
        "thickness is 32.90 mm, and the contact interval half-width is 20.10 mm. "
        "The extractor mixing-band half-width is 20.04 mm. Neither is a "
        "statistical uncertainty. "
        "(b) the first pure-air shells outside the "
        "contact are contaminated/invalid (first valid air 126.4 mm), and no "
        "selected frame has a material1 point passing training_location_valid; the "
        "actual final training count is 0 because the geometric R_b interval is "
        "empty, not because a material count was substituted for it.  The raw-cell "
        "review shows a transition spread over several coarse cells; this is a "
        "resolution statement, not proof that no physical layer exists. "
        "(c) the strict ambient "
        "tolerances are incompatible with the measured radial air-density "
        "profile (1.216-1.225 kg/m3), so no ambient run or R_b is verified. The "
        "old shock supervision leaves only 4/16 frames 'confirmed', and the "
        "frame immediately after t0 has all air labels withdrawn. The theory "
        "clock is not proven incompatible: R_cup reaches R_sep at 18.18 us vs "
        "t_sep=18.53 us, and reaches Rc0/Rs0 at 20.22/20.73 us vs t0=19.53 us, "
        "but this agreement is with the broad upper bracket edge, not with the "
        "reconstruction centre. No version-5 cache was written."
    )
    lines.append("")
    lines.append("A diagnostic tolerance relaxation is included in `diagnostic.json`; "
                 "it is not an accepted configuration and no cache was built from it.")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    make_figures(case, material, frames, precursor, timing, contact_low, contact_up)
    print(json.dumps(jsonable({
        "out": str(OUT),
        "n_frames": len(frames),
        "strict": summary,
        "timing": timing,
        "pressure_comparison": pressure_comparison,
        "ambient_t0": ambient_t0,
    }), indent=2, ensure_ascii=False))
    return 0


def make_figures(case, material, frames, precursor, timing, contact_low, contact_up):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [5, 6, 12]  # 17.964, 19.987, 27.963 us
    labels = ["17.964 us", "19.987 us", "27.963 us"]
    fields = [
        ("density", r"$\rho$ (kg/m$^3$)", True),
        ("velocity", r"$u$ (m/s)", False),
        ("pressure", r"$p$ (Pa)", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5), sharex=True)
    for col, (frame, label) in enumerate(zip(selected, labels)):
        r = np.asarray(case.radius)
        mask = (r >= 0.035) & (r <= 0.30)
        for row, (key, ylabel, logy) in enumerate(fields):
            ax = axes[row, col]
            values = getattr(case, key)[frame]
            ax.plot(r[mask] * 1e3, values[mask], color="black", lw=1.2)
            ax.axvspan(contact_low[frame] * 1e3, contact_up[frame] * 1e3,
                       color="tab:orange", alpha=0.25, label="contact bracket")
            ax.axvline(precursor.Rc0 * 1e3, color="tab:blue", ls="--", lw=1.0,
                       label="$R_{c0}$")
            ax.axvline(precursor.Rs0 * 1e3, color="tab:green", ls="--", lw=1.0,
                       label="$R_{s0}$")
            shock = float(material["shock_radius"][frame])
            if np.isfinite(shock):
                ax.axvline(shock * 1e3, color="tab:red", ls=":", lw=1.0,
                           label="$R_s$ data")
            ax.set_xlim(35, 300)
            if logy:
                positive = values[mask][values[mask] > 0]
                if positive.size:
                    ax.set_yscale("log")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("r (mm)")
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("50 mm real d3plot: radial fields near t_sep/t0 and later")
    fig.tight_layout()
    fig.savefig(OUT / "radial_compare.png", dpi=170)
    plt.close(fig)

    # Ambient radial density at t=0, compared with the configured constant.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    r0 = np.asarray(case.radius)
    phase0 = np.asarray(case.material_phase[0])
    fv0 = (
        np.asarray(case.field_validity["density"][0], dtype=bool)
        & np.asarray(case.field_validity["pressure"][0], dtype=bool)
        & np.isfinite(case.density[0])
    )
    mask0 = (phase0 == 1) & fv0
    ax.plot(r0[mask0] * 1e3, case.density[0][mask0], lw=1.1, color="black",
            label="extracted air density, t=0")
    ax.axhline(1.225, color="tab:red", ls="--", lw=1.2, label="config ambient 1.225")
    ax.set_xlabel("r (mm)")
    ax.set_ylabel(r"$\rho$ (kg/m$^3$)")
    ax.set_title("Measured initial air density is radially non-uniform")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ambient_profile_t0.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
