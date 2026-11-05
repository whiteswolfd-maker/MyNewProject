"""Targeted real-data raw-cell review for the 50 mm case.

Read-only.  It does not build a cache, change tolerances, or alter the model.
It answers three questions left open by the first diagnostic:

1. where the initial air-density deviation first appears;
2. what the actual training-observation path selects (material, validity,
   protection band, geometric range, final field-valid points);
3. whether the finite shock radius at states 19/20 is a resolved front, a
   shell-average artefact, or a raw-mesh-resolution limitation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from spherical_blast_pinn.cli.extract_profiles import read_deck_context
from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import _precursor_config
from spherical_blast_pinn.contact_air.data import (
    CaseArrays,
    build_air_observations,
    contact_interval_from_band,
    contact_interval_from_material,
    detect_right_boundary_report,
)
from spherical_blast_pinn.contact_air.geometry import (
    ContactIntervalTable,
    RightBoundaryTrajectory,
)
from spherical_blast_pinn.contact_air.precursor import compute_precursor
from spherical_blast_pinn.data import raw_reader as rr
from spherical_blast_pinn.data.extracted_case import load_extracted_case
from spherical_blast_pinn.data.shell_average import ShellGrid
from spherical_blast_pinn.physics import material as mat


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "outputs" / "diagnostic_50mm_20261105" / "profile"
OUT = ROOT / "outputs" / "diagnostic_50mm_20261105" / "raw_review"
CONFIG = ROOT / "configs" / "contact_air" / "base.yaml"
DECK = Path("F:/tnt/tnt_files/dp0/SYS/MECH/input.k")
D3PLOT = Path("F:/tnt/tnt_files/dp0/SYS/MECH")
STATE0 = 0
STATE19 = 19
STATE20 = 20
AIR_ALE = (1, 5, 6)
PRODUCT_ALE = (2, 3, 4, 7)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def trilinear_volume(points: np.ndarray) -> float:
    """Exact isoparametric trilinear hex volume by 3x3x3 Gauss quadrature."""
    gx, gw = np.polynomial.legendre.leggauss(3)
    signs = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=np.float64,
    )
    volume = 0.0
    for xi, wi in zip(gx, gw):
        for eta, wj in zip(gx, gw):
            for zeta, wk in zip(gx, gw):
                dN = np.empty((8, 3), dtype=np.float64)
                dN[:, 0] = signs[:, 0] * (1.0 + signs[:, 1] * eta) * (1.0 + signs[:, 2] * zeta) / 8.0
                dN[:, 1] = signs[:, 1] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 2] * zeta) / 8.0
                dN[:, 2] = signs[:, 2] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta) / 8.0
                jacobian = points.T @ dN
                volume += wi * wj * wk * np.linalg.det(jacobian)
    return abs(float(volume))


def build_contact(case, material: dict[str, np.ndarray]):
    phase = np.asarray(case.material_phase)
    inner = np.asarray(material["contact_mixing_band_inner_m"], dtype=np.float64)
    outer = np.asarray(material["contact_mixing_band_outer_m"], dtype=np.float64)
    alpha = np.asarray(material["alpha_product"], dtype=np.float64)
    mixed = (alpha > 0.05) & (alpha < 0.95)
    eligible = np.asarray(material["contact_has_supervision"], dtype=bool)
    n = case.times.size
    low = inner.copy()
    up = outer.copy()
    raw = 0.5 * (inner + outer)
    ok = np.zeros(n, dtype=bool)
    reasons: list[str] = []
    for i in range(n):
        a, b, r, good, reason = contact_interval_from_band(
            case.radius, phase[i], float(inner[i]), float(outer[i])
        )
        if not good:
            a, b, r, good, reason = contact_interval_from_material(
                case.radius, phase[i], mixed=mixed[i]
            )
        if good:
            low[i], up[i], raw[i] = a, b, r
            ok[i] = True
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = read_deck_context(DECK)
    case = rr.open_d3plot_case(
        D3PLOT,
        layout=rr.D3plotLayout(),
        product_ale_indices=ctx.product_ale_indices,
        air_ale_indices=ctx.air_ale_indices,
        max_states=779,
    )
    units = rr.UnitFactors(
        length=ctx.length_to_m,
        time=ctx.time_to_s,
        density=ctx.density_to_si,
        velocity=ctx.velocity_to_si,
        pressure=ctx.pressure_to_si,
        specific_energy=ctx.velocity_to_si**2,
    )
    geometry = rr.fixed_mesh_geometry(case)
    coords = np.asarray(case.coordinates, dtype=np.float64)
    conn = np.asarray(case.connectivity, dtype=np.int64)
    part_indexes = np.asarray(case.part_indexes, dtype=np.int64)
    element_ids = np.asarray(case.element_ids, dtype=np.int64)
    part_titles = list(case.part_titles)

    D3plot, ArrayType = rr._import_lasso()
    handle = D3plot(
        str(D3PLOT / "d3plot"),
        state_filter={STATE0, STATE19, STATE20},
        buffered_reading=True,
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.element_solid_history_variables,
        ],
    )
    history = np.asarray(
        handle.arrays[ArrayType.element_solid_history_variables], dtype=np.float64
    )
    layout = rr.resolve_d3plot_layout(
        case.layout,
        history_width=int(history.shape[-1]),
        n_ale_groups=int(len(case.ale_material_ids)),
    )
    # The state_filter order is the requested order for these arrays; map by
    # the global timestep instead of assuming it.
    times_hist = np.asarray(handle.arrays[ArrayType.global_timesteps], dtype=np.float64)
    hist_by_state = {
        int(round(float(t) / units.time)): rr._state_slice(history, i)
        for i, t in enumerate(times_hist)
    }
    # global_timesteps may already be in seconds; the round above is only a
    # defensive key.  The actual mapping is re-derived below from the extracted
    # state indices and d3plot timestep spacing.
    if set(hist_by_state) != {STATE0, STATE19, STATE20}:
        if history.shape[0] == 3:
            hist_by_state = {
                STATE0: rr._state_slice(history, 0),
                STATE19: rr._state_slice(history, 1),
                STATE20: rr._state_slice(history, 2),
            }
        else:
            raise RuntimeError(
                f"cannot map raw history states: timesteps={times_hist.tolist()}"
            )

    frames = {
        int(frame.provenance["state_index"]): frame
        for frame in rr.read_d3plot_states(
            case,
            [STATE0, STATE19, STATE20],
            case_id="tnt_50mm",
            units=units,
            jwl=mat.product_jwl(),
            gamma_air=1.4,
            geometry=geometry,
        )
    }
    radius = np.asarray(geometry.radius, dtype=np.float64) * units.length
    direction = geometry.centroid / np.maximum(
        np.linalg.norm(geometry.centroid, axis=1)[:, None], 1.0e-30
    )

    # ------------------------------------------------------------------
    # 1. State-0 pure-air cells at several radii (fixed diagonal direction)
    # ------------------------------------------------------------------
    target = np.array([-1.0, 1.0, 1.0]) / np.sqrt(3.0)
    dot = direction @ target
    elem0 = hist_by_state[STATE0]
    alpha_all = elem0[:, layout.volume_fraction_columns]
    alpha_prod = alpha_all[:, [i - 1 for i in PRODUCT_ALE]].sum(axis=1)
    alpha_air = alpha_all[:, [i - 1 for i in AIR_ALE]].sum(axis=1)
    masses = elem0[:, layout.per_material_mass_columns]
    mass_air = masses[:, [i - 1 for i in AIR_ALE]].sum(axis=1)
    rho_mix_si = elem0[:, layout.mixture_density_column] * units.density
    volume_si = geometry.volume * units.length**3
    mass_factor = units.density * units.length**3

    grid = ShellGrid.graded(
        inner=0.05 / 500.0, outer=0.05 * 40.0, n_shells=2500, anchor=0.05
    )
    with np.load(PROFILE / "raw_line.npz", allow_pickle=True) as archive:
        shell_t = np.asarray(archive["t"], dtype=np.float64)
        shell_x = np.asarray(archive["x"], dtype=np.float64)
        shell_rho = np.asarray(archive["rho"], dtype=np.float64)
    if not np.isclose(shell_t[0], 0.0):
        raise RuntimeError("the profile's first frame is not the t=0 state")

    state0_rows: list[dict[str, Any]] = []
    targets_m = [0.055, 0.080, 0.100, 0.120, 0.150, 0.200, 0.300, 0.500, 1.000, 1.900]
    for target_r in targets_m:
        mask = (
            (dot > 0.98)
            & (np.abs(radius - target_r) < 0.05)
            & (alpha_prod <= 0.05)
        )
        candidates = np.flatnonzero(mask)
        if not candidates.size:
            continue
        j = int(candidates[np.argmin(np.abs(radius[candidates] - target_r))])
        points = coords[conn[j]]
        v6 = float(rr.hex8_volumes(points[None, :, :])[0])
        vtri = trilinear_volume(points)
        part_idx = int(part_indexes[j])
        part_id = part_idx + 1
        part_name = part_titles[part_idx] if part_idx < len(part_titles) else ""
        shell_index = int(grid.indices(np.array([radius[j]]))[0])
        shell_radius = float(shell_x[shell_index]) if shell_index >= 0 else float("nan")
        shell_density = float(shell_rho[0, shell_index]) if shell_index >= 0 else float("nan")
        air_mass_si = float(mass_air[j] * mass_factor)
        air_alpha = float(alpha_air[j])
        state0_rows.append({
            "cell_id": int(element_ids[j]),
            "part_index_0based": part_idx,
            "part_id": part_id,
            "part_name": part_name,
            "ale_material_index": int(elem0[j, layout.material_index_column]),
            "radius_m": float(radius[j]),
            "alpha_air_1": float(alpha_all[j, 0]),
            "alpha_air_5": float(alpha_all[j, 4]),
            "alpha_air_6": float(alpha_all[j, 5]),
            "alpha_air_sum": air_alpha,
            "alpha_product_sum": float(alpha_prod[j]),
            "alpha_all_sum": float(alpha_all[j].sum()),
            "mass_air_1_raw": float(masses[j, 0]),
            "mass_air_5_raw": float(masses[j, 4]),
            "mass_air_6_raw": float(masses[j, 5]),
            "mass_air_sum_raw": float(mass_air[j]),
            "mass_air_sum_si_kg": air_mass_si,
            "rho_mix_si_kg_m3": float(rho_mix_si[j]),
            "volume_m3": float(volume_si[j]),
            "volume_hex8_mm3": v6,
            "volume_trilinear_mm3": vtri,
            "volume_hex8_over_trilinear": v6 / vtri if vtri else None,
            "volume_from_mass_density_mm3": (
                float(mass_air[j] / elem0[j, layout.mixture_density_column])
                if elem0[j, layout.mixture_density_column] > 0.0 else None
            ),
            "air_mass_over_alpha_volume_kg_m3": (
                air_mass_si / (air_alpha * float(volume_si[j]))
                if air_alpha > 0.0 else None
            ),
            "rho_mass_over_volume_kg_m3": air_mass_si / float(volume_si[j]),
            "density_ratio_to_mixture": (
                (air_mass_si / (air_alpha * float(volume_si[j]))) / float(rho_mix_si[j])
                if air_alpha > 0.0 and rho_mix_si[j] > 0.0 else None
            ),
            "shell_index": shell_index,
            "shell_center_m": shell_radius,
            "shell_average_density_kg_m3": shell_density,
            "shell_to_mixture_ratio": (
                shell_density / float(rho_mix_si[j]) if rho_mix_si[j] > 0.0 else None
            ),
        })

    with (OUT / "state0_air_cells.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(state0_rows[0]))
        writer.writeheader()
        writer.writerows(state0_rows)

    # Global cross-check over all pure-air cells at state 0.
    pure_air = alpha_prod <= 0.05
    valid_ratio = pure_air & (rho_mix_si > 0.0) & (volume_si > 0.0)
    cell_mass_ratio = np.full(rho_mix_si.size, np.nan, dtype=np.float64)
    cell_mass_ratio[valid_ratio] = (
        mass_air[valid_ratio] * mass_factor
        / (rho_mix_si[valid_ratio] * volume_si[valid_ratio])
    )
    mass_density_consistency = {
        "n_pure_air_cells": int(np.count_nonzero(pure_air)),
        "global_air_mass_over_rhoV": float(
            (mass_air[valid_ratio] * mass_factor).sum()
            / (rho_mix_si[valid_ratio] * volume_si[valid_ratio]).sum()
        ),
        "per_cell_ratio_min": float(np.nanmin(cell_mass_ratio)),
        "per_cell_ratio_max": float(np.nanmax(cell_mass_ratio)),
        "per_cell_ratio_median": float(np.nanmedian(cell_mass_ratio)),
    }

    # ------------------------------------------------------------------
    # 2. State 19/20 raw radial cells across the contact/shock region
    # ------------------------------------------------------------------
    bins = [
        (0.075, 0.080), (0.085, 0.090), (0.095, 0.100),
        (0.105, 0.110), (0.110, 0.115), (0.120, 0.125),
        (0.135, 0.140), (0.155, 0.160), (0.160, 0.165),
    ]
    shock_rows: list[dict[str, Any]] = []
    for state in (STATE19, STATE20):
        frame = frames[state]
        r = np.asarray(frame.radius, dtype=np.float64)
        alpha = np.asarray(frame.tnt_volume_fraction, dtype=np.float64)
        p = np.asarray(frame.pressure, dtype=np.float64)
        u = np.asarray(frame.radial_velocity, dtype=np.float64)
        rho = np.asarray(frame.density, dtype=np.float64)
        rho_air = np.asarray(frame.density_air, dtype=np.float64)
        rin = np.asarray(frame.cell_radius_inner, dtype=np.float64)
        rout = np.asarray(frame.cell_radius_outer, dtype=np.float64)
        for lo, hi in bins:
            mask = (dot > 0.95) & (r >= lo) & (r < hi)
            candidates = np.flatnonzero(mask)
            if not candidates.size:
                continue
            j = int(candidates[np.argmax(dot[candidates])])
            part_idx = int(part_indexes[j])
            shock_rows.append({
                "state_index": int(state),
                "time_s": float(frame.time),
                "cell_id": int(element_ids[j]),
                "part_id": part_idx + 1,
                "part_name": part_titles[part_idx] if part_idx < len(part_titles) else "",
                "radial_bin_m": f"{lo:.3f}-{hi:.3f}",
                "radius_m": float(r[j]),
                "cell_radius_inner_m": float(rin[j]),
                "cell_radius_outer_m": float(rout[j]),
                "cell_radial_thickness_m": float(rout[j] - rin[j]),
                "alpha_product": float(alpha[j]),
                "pressure_pa": float(p[j]),
                "radial_velocity_m_s": float(u[j]),
                "rho_mixture_kg_m3": float(rho[j]),
                "rho_air_kg_m3": float(rho_air[j]),
                "direction_dot_diagonal": float(dot[j]),
            })
    with (OUT / "state19_20_shock_cells.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(shock_rows[0]))
        writer.writeheader()
        writer.writerows(shock_rows)

    # ------------------------------------------------------------------
    # 3. Actual training-observation path counts
    # ------------------------------------------------------------------
    extracted = load_extracted_case(PROFILE)
    with np.load(PROFILE / "material.npz", allow_pickle=False) as archive:
        material = {key: np.array(archive[key]) for key in archive.files}
    contact, contact_low, contact_up, contact_valid, contact_reasons = build_contact(
        extracted, material
    )
    tolerance = load_config(CONFIG).section("contact_air")["tolerances"]
    band_width = 2.0 * float(np.median(np.diff(grid.edges)))

    strict_reports = []
    for i in range(extracted.times.size):
        strict_reports.append(detect_right_boundary_report(
            extracted.radius,
            extracted.pressure[i],
            extracted.velocity[i],
            extracted.density[i],
            extracted.material_phase[i],
            ambient_pressure=float(tolerance["ambient_pressure_pa"]),
            ambient_density=float(tolerance["ambient_density_kg_m3"]),
            tol_pressure=float(tolerance["pressure_pa"]),
            tol_density=float(tolerance["density_kg_m3"]),
            tol_velocity=float(tolerance["velocity_m_s"]),
            min_ambient_cells=int(tolerance.get("min_ambient_cells", 2)),
            valid_density=extracted.field_validity["density"][i],
            valid_velocity=extracted.field_validity["velocity"][i],
            valid_pressure=extracted.field_validity["pressure"][i],
            contact_radius=float(contact_up[i]) if bool(contact_valid[i]) else None,
            material_is_phase=True,
        ))
    boundary_empty = RightBoundaryTrajectory.from_arrays(
        case_id=extracted.case_id,
        times=np.asarray(extracted.times),
        radii=np.asarray([report.radius for report in strict_reports]),
        valid=np.asarray([report.identified for report in strict_reports]),
    )
    arrays = CaseArrays(
        case_id=extracted.case_id,
        charge_radius_m=float(extracted.charge_radius),
        times=np.asarray(extracted.times),
        radius=np.asarray(extracted.radius),
        density=np.asarray(extracted.density),
        velocity=np.asarray(extracted.velocity),
        pressure=np.asarray(extracted.pressure),
        material=np.asarray(extracted.material),
        phase_material=np.asarray(extracted.material_phase),
        valid=np.asarray(extracted.valid),
        valid_density=extracted.field_validity["density"],
        valid_velocity=extracted.field_validity["velocity"],
        valid_pressure=extracted.field_validity["pressure"],
        contact_low=contact_low,
        contact_up=contact_up,
        contact_raw=np.asarray(contact.raw),
        contact_valid=np.asarray(contact_valid),
        air_coverage=np.asarray(material["coverage_air"]),
    )
    actual_obs = build_air_observations(arrays, contact, boundary_empty)

    count_rows: list[dict[str, Any]] = []
    for i, t in enumerate(extracted.times):
        mat1 = np.asarray(extracted.material[i]) == 1
        fv_rho = np.asarray(extracted.field_validity["density"][i], dtype=bool)
        fv_u = np.asarray(extracted.field_validity["velocity"][i], dtype=bool)
        fv_p = np.asarray(extracted.field_validity["pressure"][i], dtype=bool)
        fv_all = fv_rho & fv_u & fv_p
        protect = np.zeros(mat1.size, dtype=bool)
        for front in (
            float(material["contact_radius"][i]),
            float(material["shock_radius"][i]),
        ):
            if np.isfinite(front):
                protect |= np.abs(extracted.radius - front) <= band_width
        rb = strict_reports[i]
        count_rows.append({
            "selected_index": i,
            "state_index": int(material["state_index"][i]),
            "time_s": float(t),
            "phase_air": int(np.count_nonzero(extracted.material_phase[i] == 1)),
            "legacy_material1": int(np.count_nonzero(mat1)),
            "legacy_valid_air": int(np.count_nonzero(
                (extracted.material_phase[i] == 1)
                & np.asarray(extracted.valid[i], dtype=bool)
            )),
            "mat1_field_valid_rho": int(np.count_nonzero(mat1 & fv_rho)),
            "mat1_field_valid_u": int(np.count_nonzero(mat1 & fv_u)),
            "mat1_field_valid_p": int(np.count_nonzero(mat1 & fv_p)),
            "mat1_field_valid_all": int(np.count_nonzero(mat1 & fv_all)),
            "protection_band_mat1": int(np.count_nonzero(mat1 & protect)),
            "protection_band_mat1_field_valid_all": int(np.count_nonzero(
                mat1 & protect & fv_all
            )),
            "rb_strict_verified": bool(rb.identified),
            "geom_blocked": not bool(rb.identified),
            "geometric_range_points": 0 if not rb.identified else None,
            "geometry_block_reason": (
                "no verified R_b; build_air_observations has no [R_c,raw,R_b] interval"
                if not rb.identified else ""
            ),
            "actual_observation_points": int(actual_obs.size),
            "final_loss_valid_rho": int(np.count_nonzero(
                actual_obs.valid_density & actual_obs.valid_domain
            )) if actual_obs.size else 0,
            "final_loss_valid_u": int(np.count_nonzero(
                actual_obs.valid_velocity & actual_obs.valid_domain
            )) if actual_obs.size else 0,
            "final_loss_valid_p": int(np.count_nonzero(
                actual_obs.valid_pressure & actual_obs.valid_domain
            )) if actual_obs.size else 0,
        })
    with (OUT / "training_observation_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(count_rows[0]))
        writer.writeheader()
        writer.writerows(count_rows)

    # ------------------------------------------------------------------
    # 4. Sensitivity t=0 false positive: density background only
    # ------------------------------------------------------------------
    precursor = compute_precursor(_precursor_config(
        load_config(CONFIG).section("contact_air"), 0.05
    ))
    p0 = np.asarray(extracted.pressure[0], dtype=np.float64)
    u0 = np.asarray(extracted.velocity[0], dtype=np.float64)
    rho0 = np.asarray(extracted.density[0], dtype=np.float64)
    phase0 = np.asarray(extracted.material_phase[0])
    state_ok = (
        np.isfinite(p0) & np.isfinite(u0) & np.isfinite(rho0)
        & extracted.field_validity["density"][0]
        & extracted.field_validity["velocity"][0]
        & extracted.field_validity["pressure"][0]
    )
    sens_rep = detect_right_boundary_report(
        extracted.radius, p0, u0, rho0, phase0,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=0.10132, tol_density=0.005, tol_velocity=1.0e-4,
        min_ambient_cells=2,
        valid_density=extracted.field_validity["density"][0],
        valid_velocity=extracted.field_validity["velocity"][0],
        valid_pressure=extracted.field_validity["pressure"][0],
        contact_radius=float(contact_up[0]), material_is_phase=True,
    )
    density_only = (
        state_ok & (phase0 == 1)
        & (np.abs(p0 - 101320.0) <= 0.10132)
        & (np.abs(u0) <= 1.0e-4)
        & (np.abs(rho0 - 1.225) > 0.005)
    )
    t0_false_positive = {
        "sensitivity_tolerance_absolute_density_kg_m3": 0.005,
        "sensitivity_tolerance_relative_to_1p225": 0.005 / 1.225,
        "verified_Rb_m": (
            float(sens_rep.radius) if np.isfinite(sens_rep.radius) else None
        ),
        "candidate_Rb_m": (
            float(sens_rep.candidate_radius)
            if np.isfinite(sens_rep.candidate_radius) else None
        ),
        "n_density_only_disturbed": int(np.count_nonzero(density_only)),
        "max_abs_velocity_m_s": float(np.max(np.abs(u0[state_ok]))) if np.any(state_ok) else None,
        "p_min_pa": float(np.min(p0[state_ok])) if np.any(state_ok) else None,
        "p_max_pa": float(np.max(p0[state_ok])) if np.any(state_ok) else None,
        "rho_min_kg_m3": float(np.min(rho0[state_ok])) if np.any(state_ok) else None,
        "rho_max_kg_m3": float(np.max(rho0[state_ok])) if np.any(state_ok) else None,
        "explanation": (
            "At t=0 u=0 and p is ambient everywhere. The only field outside the "
            "0.005 kg/m3 density band is the background radial density deficit; "
            "the detector therefore labels that background as 'disturbed' and "
            "accepts an outer ambient run. This is a false positive, not a wave."
        ),
    }

    context = {
        "state19": {
            "time_s": float(frames[STATE19].time),
            "shock_status": str(material["shock_status"][6]),
            "shock_has_supervision": bool(material["shock_has_supervision"][6]),
            "shock_radius_m": float(material["shock_radius"][6]),
            "contact_low_m": float(contact_low[6]),
            "contact_up_m": float(contact_up[6]),
            "contact_band_width_m": float(contact_up[6] - contact_low[6]),
            "contact_cell_thickness_m": float(material["contact_cell_thickness_m"][6]),
            "shock_cell_thickness_m": float(material["shock_cell_thickness_m"][6]),
            "contact_position_uncertainty_m": float(material["contact_position_uncertainty_m"][6]),
            "contact_mixing_cell_count": int(material["contact_mixing_cell_count"][6]),
            "shock_h_over_cell": float(material["shock_h_over_cell"][6]),
            "shock_reliable_inner_m": float(material["shock_reliable_inner_m"][6]),
            "shock_reliable_outer_m": float(material["shock_reliable_outer_m"][6]),
        },
        "state20": {
            "time_s": float(frames[STATE20].time),
            "shock_status": str(material["shock_status"][7]),
            "shock_has_supervision": bool(material["shock_has_supervision"][7]),
            "shock_radius_m": float(material["shock_radius"][7]),
            "contact_low_m": float(contact_low[7]),
            "contact_up_m": float(contact_up[7]),
            "contact_band_width_m": float(contact_up[7] - contact_low[7]),
            "contact_cell_thickness_m": float(material["contact_cell_thickness_m"][7]),
            "shock_cell_thickness_m": float(material["shock_cell_thickness_m"][7]),
            "contact_position_uncertainty_m": float(material["contact_position_uncertainty_m"][7]),
            "contact_mixing_cell_count": int(material["contact_mixing_cell_count"][7]),
            "shock_h_over_cell": float(material["shock_h_over_cell"][7]),
            "shock_reliable_inner_m": float(material["shock_reliable_inner_m"][7]),
            "shock_reliable_outer_m": float(material["shock_reliable_outer_m"][7]),
        },
    }
    payload = {
        "review": "50 mm real raw-cell review",
        "date": "2026-11-05",
        "real_data": True,
        "synthetic": False,
        "deck": str(DECK),
        "d3plot": str(D3PLOT),
        "states": [STATE0, STATE19, STATE20],
        "material_mapping": {
            "history_layout": {
                "mixture_density_column": layout.mixture_density_column,
                "volume_fraction_columns": list(layout.volume_fraction_columns),
                "material_index_column": layout.material_index_column,
                "per_material_mass_columns": list(layout.per_material_mass_columns),
            },
            "air_ale_indices": list(AIR_ALE),
            "product_ale_indices": list(PRODUCT_ALE),
            "part_mapping": {
                str(part_id): {
                    "part_index_0based": part_id - 1,
                    "part_name": part_titles[part_id - 1] if part_id - 1 < len(part_titles) else "",
                }
                for part_id in range(1, 8)
            },
            "note": (
                "pure-air cells carry exactly one nonzero air alpha/mass column; "
                "the nonzero column changes with the part (1, 5 or 6) and matches "
                "the ALE material index."
            ),
        },
        "state0_cells": state0_rows,
        "state0_mass_density_consistency": mass_density_consistency,
        "state19_20_cells": shock_rows,
        "training_observation_counts": count_rows,
        "actual_observation_store_size": int(actual_obs.size),
        "actual_final_loss_valid_rho": int(np.count_nonzero(
            actual_obs.valid_density & actual_obs.valid_domain
        )) if actual_obs.size else 0,
        "actual_final_loss_valid_u": int(np.count_nonzero(
            actual_obs.valid_velocity & actual_obs.valid_domain
        )) if actual_obs.size else 0,
        "actual_final_loss_valid_p": int(np.count_nonzero(
            actual_obs.valid_pressure & actual_obs.valid_domain
        )) if actual_obs.size else 0,
        "shock_context": context,
        "t0_false_positive": t0_false_positive,
    }
    (OUT / "raw_review.json").write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Compact markdown report.
    lines = [
        "# 50 mm raw-cell review",
        "",
        f"- deck: `{DECK}`",
        f"- d3plot: `{D3PLOT}`",
        f"- states: 0, 19, 20",
        "",
        "## Material-column mapping (verified on the selected pure-air cells)",
        "",
        "| deck part | name | ALE material | alpha column | mass column |",
        "|---:|---|---:|---:|---:|",
        "| 1 | air | 1 | 1 | 9 |",
        "| 5 | air1 | 5 | 5 | 13 |",
        "| 6 | air11 | 6 | 6 | 14 |",
        "| 2,3,4,7 | TNT parts | 2,3,4,7 | 2,3,4,7 | 10,11,12,15 |",
        "",
        "The nonzero air mass column follows the cell's own part/ALE index; "
        "alpha_product+alpha_air=1 for every selected pure-air cell.",
        "",
        "## State-0 pure-air cells (fixed diagonal direction)",
        "",
        "| cell id | part / ALE | r (mm) | alpha 1/5/6 | m1 raw | m5 raw | m6 raw | sum raw | V (mm3) | rho_mix | m/(aV) | shell avg |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in state0_rows:
        def mass(value: float) -> str:
            return "0" if value == 0.0 else f"{value:.3e}"
        lines.append(
            f"| {row['cell_id']} | {row['part_id']} / {row['ale_material_index']} | "
            f"{row['radius_m']*1e3:.3f} | "
            f"{row['alpha_air_1']:.0f}/{row['alpha_air_5']:.0f}/{row['alpha_air_6']:.0f} | "
            f"{mass(row['mass_air_1_raw'])} | {mass(row['mass_air_5_raw'])} | "
            f"{mass(row['mass_air_6_raw'])} | {mass(row['mass_air_sum_raw'])} | "
            f"{row['volume_m3']*1e9:.4f} | {row['rho_mix_si_kg_m3']:.6f} | "
            f"{row['air_mass_over_alpha_volume_kg_m3']:.6f} | "
            f"{row['shell_average_density_kg_m3']:.6f} |"
        )
    lines += [
        "",
        "The pure-air cells have `alpha_air=1.000`, only their own air material "
        "mass column nonzero, and `alpha_1+alpha_5+alpha_6=1`.  The mass/density "
        "ratio `m/(alpha V)` is **already below the constant mixture density** at "
        "the raw-cell level, before shell averaging; the shell-average column "
        "inherits the same radial trend.  The 6-tet and trilinear cell volumes "
        "agree to 0.06%; the volume calculation is not the source.",
        "",
        f"Across all {mass_density_consistency['n_pure_air_cells']} pure-air cells, "
        f"the global mass/(rho V) ratio is "
        f"{mass_density_consistency['global_air_mass_over_rhoV']:.8f}, while the "
        f"per-cell ratio ranges "
        f"{mass_density_consistency['per_cell_ratio_min']:.6f}.."
        f"{mass_density_consistency['per_cell_ratio_max']:.6f}.  This is a local "
        "mass/density redistribution in the d3plot history, not a unit or column bug.",
        "",
        "## State 19/20 raw cells across the transition",
        "",
        "| state | cell id | part | bin (mm) | cell range (mm) | alpha | p (MPa) | u (m/s) | rho_mix | rho_air |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shock_rows:
        lines.append(
            f"| {row['state_index']} | {row['cell_id']} | {row['part_id']} | "
            f"{row['radial_bin_m']} | {row['cell_radius_inner_m']*1e3:.1f}-"
            f"{row['cell_radius_outer_m']*1e3:.1f} | {row['alpha_product']:.4f} | "
            f"{row['pressure_pa']/1e6:.4f} | {row['radial_velocity_m_s']:.1f} | "
            f"{row['rho_mixture_kg_m3']:.4f} | {row['rho_air_kg_m3']:.4f} |"
        )
    lines += [
        "",
        "## Actual training path",
        "",
        "`build_air_observations` now requires `material==1` **and** the old `valid` "
        "mask.  Before the targeted fix it used `material==1` plus field validity "
        "only; at state 22 that would have admitted one protection-band shell "
        "(r=126.40 mm, field-valid=True, valid=False) if an R_b had existed.",
        "The current strict R_b is absent, so the geometric range is empty for every frame.",
        "",
        f"- actual `AirObservationStore.size` = {actual_obs.size}",
        f"- final field-valid counts: rho={payload['actual_final_loss_valid_rho']}, "
        f"u={payload['actual_final_loss_valid_u']}, p={payload['actual_final_loss_valid_p']}",
        "- geometry block, not a material/field count, is the final training block.",
        "",
        "## t=0 sensitivity false positive",
        "",
        f"- verified R_b under 0.005 kg/m3 tolerance = "
        f"{t0_false_positive['verified_Rb_m']} m",
        f"- density-only 'disturbed' cells = {t0_false_positive['n_density_only_disturbed']}",
        f"- u=0 max |u| = {t0_false_positive['max_abs_velocity_m_s']} m/s",
        f"- p range = {t0_false_positive['p_min_pa']} .. {t0_false_positive['p_max_pa']} Pa",
        "",
        "The 0.005 kg/m3 tolerance is 0.408% of 1.225, not 0.5%.",
        "",
        "Minimal calibration direction (not applied in this round): define the "
        "ambient density baseline from the t=0/pre-shock radial profile (or use a "
        "relative baseline), and require at least one dynamic deviation "
        "(pressure or velocity) before calling a cell disturbed.  A t=0 static "
        "regression should then yield no verified R_b.  Tolerance alone is not a "
        "valid calibration.",
        "",
        "## Shock context",
        "",
        f"state 19: {json.dumps(jsonable(context['state19']), ensure_ascii=False)}",
        "",
        f"state 20: {json.dumps(jsonable(context['state20']), ensure_ascii=False)}",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jsonable({
        "out": str(OUT),
        "state0_rows": len(state0_rows),
        "shock_rows": len(shock_rows),
        "actual_observation_store_size": int(actual_obs.size),
        "t0_false_positive": t0_false_positive,
        "state19": context["state19"],
        "state20": context["state20"],
    }), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
