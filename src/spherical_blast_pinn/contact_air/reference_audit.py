"""Read-only audit of the LS-DYNA source data and the import chain.

The audit is deliberately independent of the importer for its cross-checks:
it parses the ``.xy`` files with its own small reader and applies the unit
factors stated in the audit config.  It never writes to a cache, checkpoint or
training output, and it never reruns the solver.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .blast_reference import (
    HENRYCH_SOURCE, HENRYCH_STATUS, KB_REASON, KB_STATUS,
    ReferenceRangeError, charge_mass_kg, henrych_reference, scaled_distance,
)


class AuditError(RuntimeError):
    """An audit input is missing or inconsistent."""


# ---------------------------------------------------------------------------
# independent raw reader


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Curveplot ``.xy`` file with an independent minimal parser."""
    path = Path(path)
    if not path.is_file():
        raise AuditError(f"missing profile file: {path}")
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().startswith("#pts"):
            start = index + 1
            break
    if start is None:
        raise AuditError(f"{path}: no '#pts' header")
    data: list[str] = []
    ended = False
    for line in lines[start:]:
        text = line.strip()
        if text.lower() == "endcurve":
            ended = True
            break
        if text:
            data.append(text)
    if not ended:
        raise AuditError(f"{path}: incomplete profile (no 'endcurve')")
    values = np.fromstring(" ".join(data), sep=" ", dtype=np.float64)
    if values.size % 2:
        raise AuditError(f"{path}: odd number of numeric values")
    table = values.reshape(-1, 2)
    return table[:, 0], table[:, 1]


def _split_pattern(pattern: str) -> tuple[str, str]:
    if "*" not in pattern:
        raise AuditError(f"profile pattern has no wildcard: {pattern!r}")
    prefix, suffix = pattern.split("*", 1)
    return prefix, suffix


def frame_keys(raw_dir: Path, pattern: str) -> list[str]:
    prefix, suffix = _split_pattern(pattern)
    keys = []
    for path in Path(raw_dir).glob(pattern):
        name = path.name
        if suffix:
            if not name.endswith(suffix):
                continue
            key = name[len(prefix):-len(suffix)]
        else:
            key = name[len(prefix):]
        float(key)
        keys.append(key)
    return sorted(set(keys), key=float)


def _profile_path(raw_dir: Path, pattern: str, key: str) -> Path:
    prefix, suffix = _split_pattern(pattern)
    return Path(raw_dir) / (prefix + key + suffix)


# ---------------------------------------------------------------------------
# layer A/B: identity and import consistency


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def case_manifest(case: dict, root: Path, units: dict, ambient: dict,
                  charge: dict) -> dict:
    raw_dir = Path(case["raw_dir"])
    deck = raw_dir / case["deck"]
    standard = root / case["standard"]
    cache = root / case["cache"]
    pressure_pattern = "profile2_pressure_x_time*.xy"
    keys = frame_keys(raw_dir, pressure_pattern)
    if not keys:
        raise AuditError(f"{case['case_id']}: no raw pressure frames in {raw_dir}")
    d3hsp = raw_dir / "d3hsp"
    messag = raw_dir / "messag"
    header = ""
    if d3hsp.is_file():
        with d3hsp.open("r", encoding="ascii", errors="replace") as handle:
            header = " ".join(handle.readline().strip() for _ in range(2))
    normal = messag.is_file() and "N o r m a l" in messag.read_text(
        encoding="ascii", errors="replace"
    )
    time_values = np.array([float(key) for key in keys])
    intervals = np.diff(time_values)
    standard_meta = {}
    selected = []
    if standard.is_file():
        with np.load(standard, allow_pickle=True) as archive:
            standard_meta = json.loads(str(archive["metadata"].item()))
        source = standard_meta.get("source", {})
        selected = list(source.get("selected_raw_frame_keys", []))
    mass = charge_mass_kg(float(case["radius_m"]), float(charge["density_kg_m3"]))
    sample_files = [
        _profile_path(raw_dir, pressure_pattern, keys[0]),
        _profile_path(raw_dir, pressure_pattern, keys[-1]),
    ]
    return {
        "case_id": case["case_id"],
        "charge_radius_m": float(case["radius_m"]),
        "charge_mass_kg": mass,
        "charge_mass_source": "uniform sphere from material-card density",
        "raw_dir": str(raw_dir),
        "deck": {"path": str(deck), "sha256": sha256_file(deck) if deck.is_file() else None},
        "solver_header": header,
        "normal_termination": bool(normal),
        "units": units,
        "ambient": ambient,
        "raw_frame_count": len(keys),
        "raw_time_start_s": float(time_values[0]),
        "raw_time_end_s": float(time_values[-1]),
        "raw_time_intervals": {
            "min_s": float(intervals.min()) if intervals.size else None,
            "median_s": float(np.median(intervals)) if intervals.size else None,
            "max_s": float(intervals.max()) if intervals.size else None,
        },
        "profile_fingerprint": {
            "count": len(keys),
            "bytes": int(sum(_profile_path(raw_dir, pressure_pattern, k).stat().st_size
                             for k in keys)),
            "first_frame_sha256": sha256_file(sample_files[0]) if sample_files[0].is_file() else None,
            "last_frame_sha256": sha256_file(sample_files[-1]) if sample_files[-1].is_file() else None,
        },
        "standard": {
            "path": str(standard),
            "sha256": sha256_file(standard) if standard.is_file() else None,
            "frames": len(selected),
            "dense_until_s": standard_meta.get("source", {}).get("dense_until_s"),
            "frame_stride": standard_meta.get("source", {}).get("frame_stride_after_dense"),
            "time_source": standard_meta.get("source", {}).get("time_source"),
            "selected_keys": selected,
        },
        "cache": {
            "path": str(cache),
            "sha256": sha256_file(cache) if cache.is_file() else None,
        },
    }


def _standard_frames(path: Path):
    with np.load(path, allow_pickle=True) as archive:
        offsets = np.asarray(archive["frame_offsets"])
        arrays = {
            name: np.asarray(archive[name])
            for name in ("r_left", "r_right", "radius", "pressure", "density",
                         "velocity", "alpha_product", "alpha_air",
                         "valid_density", "valid_velocity", "valid_pressure")
            if name in archive.files
        }
        metadata = json.loads(str(archive["metadata"].item()))
    frames = []
    for index in range(offsets.size - 1):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        frames.append({name: values[start:stop] for name, values in arrays.items()})
    return frames, metadata


def import_comparison(case: dict, root: Path, units: dict,
                      sample_indices: int = 4) -> list[dict]:
    """Compare raw file values with the imported standard arrays."""
    raw_dir = Path(case["raw_dir"])
    frames, metadata = _standard_frames(root / case["standard"])
    selection = list(metadata["source"]["selected_raw_frame_keys"])
    if len(selection) != len(frames):
        raise AuditError(f"{case['case_id']}: selected keys and frames differ")
    if sample_indices >= len(frames):
        indices = list(range(len(frames)))
    else:
        event = 0
        with np.load(root / case["cache"], allow_pickle=True) as archive:
            metadata_cache = json.loads(str(archive["metadata"].item()))
        event = int(metadata_cache["case"]["event_alignment"]["event_frame"])
        positions = {0, min(max(event, 0), len(frames) - 1), len(frames) // 2,
                     len(frames) - 1}
        extra = np.linspace(0, len(frames) - 1, sample_indices).astype(int)
        positions.update(int(value) for value in extra)
        indices = sorted(positions)
    factors = units["to_si"]
    rows: list[dict] = []
    for index in indices:
        key = selection[index]
        frame = frames[index]
        _, pressure = read_xy(_profile_path(raw_dir, "profile2_pressure_x_time*.xy", key))
        _, density = read_xy(_profile_path(raw_dir, "profile3_density_x_time*.xy", key))
        _, alpha_p = read_xy(_profile_path(raw_dir, "profile4_volume_fraction_x_time*.xy", key))
        node_x, node_v = read_xy(_profile_path(raw_dir, "profile1_x-velocity_x_time*.xy", key))
        velocity_cell = 0.5 * (node_v[:-1] + node_v[1:])
        comparisons = {
            "pressure": (pressure * float(factors["pressure"]), frame["pressure"]),
            "density": (density * float(factors["density"]), frame["density"]),
            "alpha_product": (alpha_p, frame["alpha_product"]),
            "velocity": (velocity_cell * float(factors["velocity"]), frame["velocity"]),
            "radius": (0.5 * (node_x[:-1] + node_x[1:]) * float(factors["length"]),
                       frame["radius"]),
        }
        classes = {
            "product": alpha_p > 0.95,
            "air": alpha_p < 0.05,
            "mixing": (alpha_p >= 0.05) & (alpha_p <= 0.95),
        }
        cell_x = 0.5 * (node_x[:-1] + node_x[1:])
        gradient = (
            np.abs(np.gradient(pressure, cell_x))
            if cell_x.size == pressure.size else np.zeros(pressure.size)
        )
        shock_index = int(np.argmax(gradient)) if gradient.size else 0
        classes["near_shock"] = np.zeros(pressure.size, dtype=bool)
        lo = max(0, shock_index - 5)
        hi = min(pressure.size, shock_index + 6)
        classes["near_shock"][lo:hi] = True
        for name, (raw_value, imported) in comparisons.items():
            if raw_value.size != imported.size:
                rows.append({
                    "case_id": case["case_id"], "frame": key, "field": name,
                    "region": "all", "n": 0, "max_abs_diff": None,
                    "max_rel_diff": None, "status": "size_mismatch",
                })
                continue
            for region, mask in classes.items():
                use = mask if name == "radius" else mask[:imported.size]
                if name == "radius":
                    use = np.ones(imported.size, dtype=bool)
                if not np.any(use):
                    continue
                difference = raw_value[use] - imported[use]
                scale = max(float(np.max(np.abs(imported[use]))), 1.0e-30)
                rows.append({
                    "case_id": case["case_id"], "frame": key, "field": name,
                    "region": region, "n": int(np.count_nonzero(use)),
                    "max_abs_diff": float(np.max(np.abs(difference))),
                    "max_rel_diff": float(np.max(np.abs(difference)) / scale),
                    "status": "ok",
                })
    return rows


# ---------------------------------------------------------------------------
# layer A: fixed-radius pressure histories and metrics


def fixed_radius_histories(case: dict, units: dict, ambient: dict,
                           charge: dict, scaled_distances: list[float]) -> dict:
    raw_dir = Path(case["raw_dir"])
    keys = frame_keys(raw_dir, "profile2_pressure_x_time*.xy")
    mass = charge_mass_kg(
        float(case["radius_m"]), float(charge["density_kg_m3"])
    )
    cube_root = mass ** (1.0 / 3.0)
    radii = [float(z) * cube_root for z in scaled_distances]
    histories = {f"{z:g}": {"z": float(z), "r_m": r, "t": [], "dp_mpa": [],
                            "offset_m": []}
                 for z, r in zip(scaled_distances, radii)}
    for key in keys:
        x_mm, pressure = read_xy(_profile_path(raw_dir, "profile2_pressure_x_time*.xy", key))
        r_m = x_mm * units["to_si"]["length"]
        for label, entry in histories.items():
            index = int(np.argmin(np.abs(r_m - entry["r_m"])))
            entry["t"].append(float(key))
            entry["dp_mpa"].append(float(pressure[index]) - float(ambient["pressure_mpa"]))
            entry["offset_m"].append(float(r_m[index] - entry["r_m"]))
    return histories


def pulse_metrics(t: np.ndarray, overpressure_mpa: np.ndarray,
                  *, record_ends: float | None = None) -> dict:
    """Peak, arrival, positive duration and impulse with completeness flags."""
    t = np.asarray(t, float)
    dp = np.asarray(overpressure_mpa, float)
    if t.size != dp.size or t.size < 3:
        return {"status": "insufficient_samples"}
    order = np.argsort(t)
    t, dp = t[order], dp[order]
    peak_index = int(np.argmax(dp))
    peak = float(dp[peak_index])
    threshold = max(1.0e-3, 0.01 * peak) if peak > 0 else 1.0e-3
    above = dp > threshold
    if not np.any(above):
        return {"status": "no_positive_phase", "peak_mpa": peak,
                "peak_time_s": float(t[peak_index])}
    arrival_index = int(np.flatnonzero(above)[0])
    after = np.flatnonzero(~above[arrival_index:])
    flags = []
    if arrival_index == 0:
        flags.append("arrival_at_first_sample")
    if peak_index == 0:
        flags.append("peak_at_first_sample")
    if peak_index == dp.size - 1:
        flags.append("peak_at_last_sample")
    if after.size:
        end_index = arrival_index + int(after[0])
        duration = float(t[end_index] - t[arrival_index])
        impulse = float(np.trapz(dp[arrival_index:end_index + 1],
                                     t[arrival_index:end_index + 1]))
        complete = True
    else:
        end_index = dp.size - 1
        duration = None
        impulse = float(np.trapz(dp[arrival_index:end_index + 1],
                                     t[arrival_index:end_index + 1]))
        complete = False
        flags.append("positive_phase_not_finished")
    dt = float(np.median(np.diff(t))) if t.size > 1 else None
    if duration is not None and dt is not None and dt > duration / 5.0:
        flags.append("time_resolution_coarse")
    # A second local maximum inside the recorded positive phase can be an
    # outer-boundary reflection or a secondary shock; the first-phase impulse is
    # then contaminated and must not be reported as a clean KB comparison.
    local_maxima = np.flatnonzero(
        (dp[1:-1] > dp[:-2]) & (dp[1:-1] > dp[2:]) & (dp[1:-1] > 0.15 * peak)
    ) + 1
    secondary = [int(index) for index in local_maxima if index > peak_index]
    contaminated = False
    if secondary:
        if secondary[0] <= end_index:
            contaminated = True
            flags.append("secondary_pulse_inside_positive_phase")
        else:
            flags.append("secondary_pulse_after_positive_phase")
    status = "complete" if complete else "partial"
    if contaminated:
        status = "partial_contaminated"
    return {
        "status": status,
        "peak_mpa": peak,
        "peak_time_s": float(t[peak_index]),
        "arrival_s": float(t[arrival_index]),
        "duration_s": duration,
        "impulse_mpa_s": impulse,
        "threshold_mpa": float(threshold),
        "samples": int(t.size),
        "median_dt_s": dt,
        "secondary_peak_times_s": [float(t[index]) for index in secondary],
        "flags": flags,
    }


def blast_metrics(case: dict, units: dict, ambient: dict, charge: dict,
                  scaled_distances: list[float], histories: dict | None = None) -> list[dict]:
    if histories is None:
        histories = fixed_radius_histories(
            case, units, ambient, charge, scaled_distances
        )
    mass = charge_mass_kg(float(case["radius_m"]), float(charge["density_kg_m3"]))
    rows = []
    for label, history in histories.items():
        t = np.asarray(history["t"], float)
        dp = np.asarray(history["dp_mpa"], float)
        metrics = pulse_metrics(t, dp, record_ends=float(t[-1]) if t.size else None)
        z = history["z"]
        try:
            reference = henrych_reference(z)
        except ReferenceRangeError as exc:  # pragma: no cover
            reference = None
            metrics["henrych_reason"] = str(exc)
        row = {
            "case_id": case["case_id"],
            "z_m_kg13": z,
            "r_m": history["r_m"],
            "cell_offset_m": float(np.max(np.abs(history["offset_m"]))),
            "charge_mass_kg": mass,
            **metrics,
            "henrych_mpa": None if reference is None else reference.value,
            "henrych_status": None if reference is None else reference.status,
            "henrych_reason": None if reference is None else reference.reason,
            "kb_status": KB_STATUS,
            "kb_reason": KB_REASON,
        }
        if row.get("peak_mpa") is not None and row["henrych_mpa"]:
            row["peak_minus_henrych_mpa"] = row["peak_mpa"] - row["henrych_mpa"]
            row["peak_relative_difference"] = row["peak_minus_henrych_mpa"] / row["henrych_mpa"]
        rows.append(row)
    return rows


def peak_preservation(case: dict, root: Path, units: dict, ambient: dict,
                      charge: dict, scaled_distances: list[float],
                      histories: dict | None = None) -> list[dict]:
    """Full raw series peak vs the peak visible in the imported frames."""
    if histories is None:
        histories = fixed_radius_histories(
            case, units, ambient, charge, scaled_distances
        )
    frames, _ = _standard_frames(root / case["standard"])
    out = []
    for label, history in histories.items():
        r_target = history["r_m"]
        raw_peak = float(np.max(history["dp_mpa"]))
        imported_peak = -np.inf
        for frame in frames:
            radius = frame["radius"]
            value = frame["pressure"][int(np.argmin(np.abs(radius - r_target)))]
            imported_peak = max(imported_peak, float(value) / 1.0e6 - ambient["pressure_mpa"])
        out.append({
            "case_id": case["case_id"], "z_m_kg13": history["z"],
            "r_m": r_target,
            "raw_peak_mpa": raw_peak,
            "imported_peak_mpa": imported_peak,
            "peak_difference_mpa": imported_peak - raw_peak,
            "raw_frames": len(history["t"]),
            "imported_frames": len(frames),
        })
    return out


# ---------------------------------------------------------------------------
# layer C: consistency of measured states and the program formulas


def rh_audit(case: dict, root: Path, ambient: dict) -> dict:
    frames, metadata = _standard_frames(root / case["standard"])
    with np.load(root / case["cache"], allow_pickle=True) as archive:
        shock_times = np.asarray(archive["shock_times"], float)
        shock_radius = np.asarray(archive["shock_radius"], float)
        raw_times = np.asarray(archive["raw_times"], float)
    if shock_times.size != len(frames):
        raise AuditError(
            f"{case['case_id']}: cache shock frames {shock_times.size} do not "
            f"match standard frames {len(frames)}"
        )
    gamma = float(ambient["gamma"])
    results = []
    for index in (len(frames) // 3, len(frames) // 2, (2 * len(frames)) // 3):
        if index <= 0 or index >= len(frames) - 1:
            continue
        frame = frames[index]
        radius = frame["radius"]
        dr = float(np.median(np.diff(radius)))
        rs = float(shock_radius[index])
        ds = float((shock_radius[index + 1] - shock_radius[index - 1])
                   / (shock_times[index + 1] - shock_times[index - 1]))
        outside = np.flatnonzero(radius > rs)
        inside = np.flatnonzero(radius < rs)
        if outside.size < 4 or inside.size < 10:
            continue
        pre_index = outside[:5]
        pre = {
            "rho": float(np.mean(frame["density"][pre_index])),
            "u": float(np.mean(frame["velocity"][pre_index])),
            "p": float(np.mean(frame["pressure"][pre_index])),
        }
        for cells_inside in (3, 5, 8):
            target = rs - cells_inside * dr
            post_index = int(np.argmin(np.abs(radius - target)))
            post = {
                "rho": float(frame["density"][post_index]),
                "u": float(frame["velocity"][post_index]),
                "p": float(frame["pressure"][post_index]),
            }
            # Shock-frame fluxes: mass rho(u-D); momentum p + rho u (u-D);
            # energy (E+p)(u-D) with E the total energy density.
            mass = post["rho"] * (post["u"] - ds) - pre["rho"] * (pre["u"] - ds)
            momentum = (
                post["p"] + post["rho"] * post["u"] * (post["u"] - ds)
                - pre["p"] - pre["rho"] * pre["u"] * (pre["u"] - ds)
            )
            e_post = post["p"] / (gamma - 1.0) + 0.5 * post["rho"] * post["u"] ** 2
            e_pre = pre["p"] / (gamma - 1.0) + 0.5 * pre["rho"] * pre["u"] ** 2
            energy = (
                (e_post + post["p"]) * (post["u"] - ds)
                - (e_pre + pre["p"]) * (pre["u"] - ds)
            )
            results.append({
                "frame_index": index,
                "raw_time_s": float(raw_times[index]),
                "Ds_m_s": ds,
                "cells_inside": cells_inside,
                "dr_m": dr,
                "pre": pre, "post": post,
                "mass_residual_normalised": mass / (pre["rho"] * ds),
                "momentum_residual_normalised": momentum / (pre["rho"] * ds ** 2),
                "energy_residual_normalised": energy / (pre["rho"] * ds ** 3),
            })
    return {
        "case_id": case["case_id"],
        "note": "Ds from central difference of measured shock positions; states measured, "
                "not synthesized from the Rankine-Hugoniot relations",
        "samples": results,
    }


def constraint_compatibility(case: dict, root: Path, ambient: dict) -> dict:
    with np.load(root / case["cache"], allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        contact_times = np.asarray(archive["contact_times"], float)
        low = np.asarray(archive["contact_low"], float)
        up = np.asarray(archive["contact_up"], float)
        valid = np.asarray(archive["contact_valid"], bool)
    frames, _ = _standard_frames(root / case["standard"])
    precursor = metadata["case"]["precursor"]
    jwl = precursor["jwl"]
    v_sep = float(precursor["V_sep"])
    r_sep = float(precursor["R_sep"])
    index = len(frames) // 2
    frame = frames[index]
    radius = frame["radius"]

    def jwl_pressure(r_m: float) -> float:
        v = v_sep * (r_m / r_sep) ** 3
        return (jwl["A"] * math.exp(-jwl["R1"] * v)
                + jwl["B"] * math.exp(-jwl["R2"] * v)
                + jwl["C"] / v ** (1.0 + jwl["omega"]))
    if valid.any() and index < low.size:
        inside = np.flatnonzero(radius > up[index])
        air_p = float(frame["pressure"][inside[0]]) if inside.size else None
        air_u = float(frame["velocity"][inside[0]]) if inside.size else None
        p_low = jwl_pressure(float(low[index]))
        p_up = jwl_pressure(float(up[index]))
        if index + 1 < low.size and index > 0:
            v_low = (low[index + 1] - low[index - 1]) / (contact_times[index + 1] - contact_times[index - 1])
            v_up = (up[index + 1] - up[index - 1]) / (contact_times[index + 1] - contact_times[index - 1])
        else:
            v_low = v_up = None
    else:
        air_p = air_u = p_low = p_up = v_low = v_up = None
    return {
        "case_id": case["case_id"],
        "initial_compatibility": metadata["case"].get("initial_compatibility"),
        "contact_jwl_vs_air": {
            "frame_index": index,
            "p_jwl_low_pa": p_low,
            "p_jwl_up_pa": p_up,
            "p_air_just_outside_pa": air_p,
            "ratio_up_over_air": (p_up / air_p) if (p_up and air_p) else None,
        },
        "contact_speed_vs_air": {
            "contact_speed_range_m_s": (
                [min(v_low, v_up), max(v_low, v_up)]
                if v_low is not None else None
            ),
            "air_velocity_just_outside_m_s": air_u,
            "note": "contact speed reported as the range from the interval bounds; "
                    "no midpoint target is used",
        },
        "event_frame_rh_diagnostic": metadata["case"].get("event_frame_rh_diagnostic"),
    }


def implementation_checks(case_data) -> dict:
    """Check the program's own derivative and alignment formulas."""
    shock = case_data.shock
    times = np.asarray(shock.times, dtype=np.float64)
    differences = []
    h = 1.0e-9
    for index in range(2, len(times) - 2, max(1, len(times) // 8)):
        value = float(times[index])
        try:
            analytic = float(np.asarray(shock.derivative(np.array([value])))[0])
            plus = float(np.asarray(shock.position(np.array([value + h])))[0])
            minus = float(np.asarray(shock.position(np.array([value - h])))[0])
        except Exception:  # outside a segment
            continue
        finite = (plus - minus) / (2.0 * h)
        differences.append(abs(analytic - finite) / max(abs(finite), 1.0e-30))
    alignment = case_data.event_alignment
    offset = float(alignment["offset_s"])
    round_trip = abs(
        offset - (float(alignment["t0_model_s"]) - float(alignment["event_time_raw_s"]))
    ) <= 1.0e-18
    return {
        "case_id": case_data.case_id,
        "event_alignment_offset_consistent": bool(round_trip),
        "offset_s": offset,
        "shock_derivative_note": "analytic shock derivative vs central difference "
                                 "of the program's own position function",
        "shock_derivative_max_rel_difference": (
            float(max(differences)) if differences else None
        ),
        "shock_derivative_samples": len(differences),
        "time_scaling_chain_rule_test": "tests/test_contact_air_time_scaling.py",
        "mapping_vs_physical_residual_test": "not duplicated here; covered by the "
                                             "existing mapping and physical residual tests",
    }


__all__ = [
    "AuditError", "blast_metrics", "case_manifest", "constraint_compatibility",
    "fixed_radius_histories", "frame_keys", "implementation_checks",
    "import_comparison", "peak_preservation", "pulse_metrics", "read_xy",
    "rh_audit", "sha256_file",
]
