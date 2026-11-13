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


#: Relative tolerances for the raw-vs-import comparison.  The importer is
#: expected to reproduce the raw values exactly; these are float64 round-off
#: bounds, not a physical tolerance.
IMPORT_TOLERANCES = {
    "pressure": 1.0e-9,
    "density": 1.0e-9,
    "alpha_product": 1.0e-9,
    "velocity": 1.0e-9,
    "radius": 1.0e-12,
}


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
    source = {}
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
            "frame_selection_policy": source.get("frame_selection_policy"),
            "raw_frame_count": source.get("raw_frame_count"),
            "imported_frame_count": source.get("imported_frame_count"),
            "dropped_frame_count": source.get("dropped_frame_count"),
            "imported_equals_raw": (
                source.get("imported_frame_count") == len(selected)
                if source.get("imported_frame_count") is not None else None
            ),
            "input_fingerprint": source.get("raw_input_fingerprint"),
            "time_source": source.get("time_source"),
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
                      sample_indices: int | None = 4) -> list[dict]:
    """Compare raw file values with the imported standard arrays.

    ``sample_indices=None`` compares every imported frame; the default samples
    four evenly spaced frames for the multi-case pass.  The R50 acceptance
    check uses every frame explicitly.
    """
    raw_dir = Path(case["raw_dir"])
    frames, metadata = _standard_frames(root / case["standard"])
    selection = list(metadata["source"]["selected_raw_frame_keys"])
    if len(selection) != len(frames):
        raise AuditError(f"{case['case_id']}: selected keys and frames differ")
    if sample_indices is None or sample_indices >= len(frames):
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
            tolerance = IMPORT_TOLERANCES[name]
            if raw_value.size != imported.size:
                rows.append({
                    "case_id": case["case_id"], "frame": key, "field": name,
                    "region": "all", "n": 0, "max_abs_diff": None,
                    "max_rel_diff": None, "tolerance": tolerance,
                    "status": "size_mismatch",
                })
                continue
            for region, mask in classes.items():
                use = mask if name == "radius" else mask[:imported.size]
                if name == "radius":
                    use = np.ones(imported.size, dtype=bool)
                if not np.any(use):
                    continue
                raw_use = raw_value[use]
                imported_use = imported[use]
                finite = np.isfinite(raw_use) & np.isfinite(imported_use)
                if not np.all(finite):
                    rows.append({
                        "case_id": case["case_id"], "frame": key, "field": name,
                        "region": region, "n": int(np.count_nonzero(use)),
                        "max_abs_diff": None, "max_rel_diff": None,
                        "tolerance": tolerance,
                        "non_finite_raw": int(np.count_nonzero(
                            ~np.isfinite(raw_use))),
                        "non_finite_imported": int(np.count_nonzero(
                            ~np.isfinite(imported_use))),
                        "status": "non_finite",
                    })
                    continue
                difference = raw_use - imported_use
                scale = max(float(np.max(np.abs(imported_use))), 1.0e-30)
                relative = float(np.max(np.abs(difference)) / scale)
                rows.append({
                    "case_id": case["case_id"], "frame": key, "field": name,
                    "region": region, "n": int(np.count_nonzero(use)),
                    "max_abs_diff": float(np.max(np.abs(difference))),
                    "max_rel_diff": relative,
                    "tolerance": tolerance,
                    "status": "ok" if relative <= tolerance else "mismatch",
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


DETECTION_FLOOR_MPA = 1.0e-3
PULSE_METHOD = (
    "first incident pulse: arrival is the first sample above the detection "
    "threshold; the positive phase ends at the first strict return of the "
    "overpressure to zero; the peak is the maximum inside that first pulse. "
    "Local maxima inside the pulse and later positive excursions are reported "
    "as candidates only; their physical origin is not classified here."
)


def _detection_threshold_mpa(dp: np.ndarray) -> float:
    """Noise-aware threshold from the leading (pre-arrival) samples.

    The leading tenth of the record is used only when it looks like ambient
    data.  If it already contains the pulse, the threshold falls back to the
    absolute floor instead of being inflated by the pulse itself.
    """
    if dp.size == 0:
        return DETECTION_FLOOR_MPA
    pre = dp[: max(3, dp.size // 10)]
    median = float(np.median(pre))
    mad = float(np.median(np.abs(pre - median)))
    noise = 1.4826 * mad
    peak = float(np.max(np.abs(dp)))
    if peak > 0.0 and noise > 0.05 * peak:
        noise = 0.0
    return max(DETECTION_FLOOR_MPA, abs(median) + 5.0 * noise)


def _interpolated_crossing(t: np.ndarray, values: np.ndarray,
                           index: int, level: float) -> float:
    """Time at which ``values`` crosses ``level`` between ``index-1`` and index."""
    if index <= 0:
        return float(t[0])
    before, after = float(values[index - 1]), float(values[index])
    if after == before:
        return float(t[index])
    fraction = (level - before) / (after - before)
    fraction = min(max(fraction, 0.0), 1.0)
    return float(t[index - 1] + fraction * (t[index] - t[index - 1]))


def _local_maxima(t: np.ndarray, dp: np.ndarray, low: int, high: int,
                  floor: float) -> list[dict]:
    """Strict local maxima in the inclusive index range ``[low, high]``."""
    out = []
    last = min(high, dp.size - 2)
    for index in range(max(low + 1, 1), last + 1):
        if dp[index] > dp[index - 1] and dp[index] >= dp[index + 1]:
            if dp[index] > floor:
                out.append({"time_s": float(t[index]),
                            "overpressure_mpa": float(dp[index])})
    return out


def pulse_metrics(t: np.ndarray, overpressure_mpa: np.ndarray,
                  *, record_ends: float | None = None,
                  detection_threshold_mpa: float | None = None) -> dict:
    """Metrics of the first incident pulse, not of the whole record.

    Arrival detection, positive-phase end and peak selection are separate:
    the peak is taken inside the first pulse (arrival to the first strict zero
    crossing), not as the maximum over the entire record.  Oscillations inside
    that pulse and later excursions are candidates, not proved secondary
    shocks; the status is never changed to ``contaminated`` on their account.
    """
    t = np.asarray(t, float)
    dp = np.asarray(overpressure_mpa, float)
    if t.size != dp.size or t.size < 3:
        return {"status": "insufficient_samples"}
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(dp))):
        return {"status": "non_finite_samples",
                "non_finite_entries": int(np.count_nonzero(~np.isfinite(dp)))}
    order = np.argsort(t)
    t, dp = t[order], dp[order]
    threshold = (
        float(detection_threshold_mpa)
        if detection_threshold_mpa is not None
        else _detection_threshold_mpa(dp)
    )
    above = dp > threshold
    if not np.any(above):
        return {
            "status": "no_positive_phase",
            "threshold_mpa": threshold,
            "peak_mpa": float(np.max(dp)),
            "method": PULSE_METHOD,
        }
    arrival_index = int(np.flatnonzero(above)[0])
    arrival_time = _interpolated_crossing(t, dp, arrival_index, threshold)

    nonpositive = np.flatnonzero(dp[arrival_index:] <= 0.0)
    if nonpositive.size:
        end_index = arrival_index + int(nonpositive[0])
        end_time = _interpolated_crossing(t, dp, end_index, 0.0)
        complete = True
    else:
        end_index = dp.size - 1
        end_time = None
        complete = False
    stop = end_index + 1 if complete else dp.size
    peak_index = arrival_index + int(np.argmax(dp[arrival_index:stop]))
    peak = float(dp[peak_index])

    flags = []
    if arrival_index == 0:
        flags.append("arrival_at_first_sample")
    if peak_index == arrival_index:
        flags.append("peak_at_arrival_sample")
    if peak_index == dp.size - 1 and not complete:
        flags.append("peak_at_last_sample")

    intra_all = _local_maxima(t, dp, arrival_index, end_index,
                              max(threshold, 0.1 * peak))
    intra = [item for item in intra_all
             if abs(item["time_s"] - float(t[peak_index])) > 0.0]
    first_local = intra_all[0] if intra_all else {
        "time_s": float(t[peak_index]), "overpressure_mpa": peak}
    if intra:
        flags.append("intra_pulse_maxima_candidates")

    later: list[dict] = []
    cursor = end_index + 1
    while cursor < dp.size and len(later) < 5:
        rising = np.flatnonzero(dp[cursor:] > threshold)
        if not rising.size:
            break
        start = cursor + int(rising[0])
        zero = np.flatnonzero(dp[start:] <= 0.0)
        stop_at = start + int(zero[0]) if zero.size else dp.size
        segment = dp[start:stop_at]
        if segment.size:
            index = start + int(np.argmax(segment))
            later.append({"time_s": float(t[index]),
                          "overpressure_mpa": float(dp[index])})
        cursor = stop_at + 1
    if later:
        flags.append("later_pulse_candidates_present")

    dt = float(np.median(np.diff(t))) if t.size > 1 else None
    duration = None if not complete else float(end_time - arrival_time)
    if complete and dt is not None and duration is not None and dt > duration / 5.0:
        flags.append("time_resolution_coarse")

    if complete:
        seg_t = t[arrival_index:end_index + 1].copy()
        seg_dp = dp[arrival_index:end_index + 1].copy()
        seg_t[-1] = end_time
        seg_dp[-1] = 0.0
    else:
        seg_t = t[arrival_index:].copy()
        seg_dp = dp[arrival_index:].copy()
        flags.append("positive_phase_not_finished")
    impulse = float(np.trapz(seg_dp, seg_t))
    if complete and duration is None:  # pragma: no cover - defensive
        duration = float(end_time - arrival_time)

    return {
        "status": "complete" if complete else "partial",
        "method": PULSE_METHOD,
        "peak_mpa": peak,
        "peak_time_s": float(t[peak_index]),
        "peak_source": "first_pulse_maximum",
        "first_local_maximum_mpa": float(first_local["overpressure_mpa"]),
        "first_local_maximum_time_s": float(first_local["time_s"]),
        "arrival_s": arrival_time,
        "arrival_sample_s": float(t[arrival_index]),
        "duration_s": duration,
        "impulse_mpa_s": impulse,
        "threshold_mpa": threshold,
        "phase_end_s": end_time,
        "samples": int(t.size),
        "median_dt_s": dt,
        "intra_pulse_maxima_s": [item["time_s"] for item in intra],
        "later_pulse_candidates_s": [item["time_s"] for item in later],
        "later_pulse_candidate_peaks_mpa": [item["overpressure_mpa"] for item in later],
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


def _imported_history(frames: list[dict], times: np.ndarray,
                      radius_m: float, ambient: dict):
    """Fixed-radius history sampled from the imported frames."""
    dp = []
    offsets = []
    for frame in frames:
        radius = np.asarray(frame["radius"], float)
        index = int(np.argmin(np.abs(radius - radius_m)))
        dp.append(float(frame["pressure"][index]) / 1.0e6
                  - float(ambient["pressure_mpa"]))
        offsets.append(float(radius[index] - radius_m))
    return np.asarray(times, float), np.asarray(dp, float), np.asarray(offsets)


def peak_preservation(case: dict, root: Path, units: dict, ambient: dict,
                      charge: dict, scaled_distances: list[float],
                      histories: dict | None = None) -> list[dict]:
    """First-pulse peak on the full raw series vs the imported frames.

    Both sides use the same extraction (arrival threshold, first strict zero
    crossing, maximum inside the first pulse) at the same fixed physical
    radius.  The selected-frame times are also checked, so a lost peak is
    separated from a change of extraction method.
    """
    if histories is None:
        histories = fixed_radius_histories(
            case, units, ambient, charge, scaled_distances
        )
    frames, metadata = _standard_frames(root / case["standard"])
    selected_times = np.asarray(
        [float(key) for key in metadata["source"]["selected_raw_frame_keys"]],
        float,
    )
    frame_times = np.asarray(metadata["frame_times"], float)
    out = []
    for label, history in histories.items():
        r_target = history["r_m"]
        raw_t = np.asarray(history["t"], float)
        raw_dp = np.asarray(history["dp_mpa"], float)
        raw_metrics = pulse_metrics(raw_t, raw_dp)
        imported_t, imported_dp, imported_offsets = _imported_history(
            frames, frame_times, r_target, ambient
        )
        imported_metrics = pulse_metrics(imported_t, imported_dp)
        raw_peak = raw_metrics.get("peak_mpa")
        raw_peak_time = raw_metrics.get("peak_time_s")
        # Use the first-pulse window determined by the FULL raw series on both
        # sides.  Re-detecting the pulse on the decimated frames would let a
        # missing peak frame move the window and hide the loss.
        window_start = raw_metrics.get("arrival_s")
        window_end = raw_metrics.get("phase_end_s")
        if window_start is None:
            window_start = float(raw_t[0]) if raw_t.size else None
        if window_end is None:
            window_end = float(raw_t[-1]) if raw_t.size else None
        if (window_start is not None and window_end is not None
                and imported_t.size):
            common = (
                (imported_t >= float(window_start) - 1.0e-15)
                & (imported_t <= float(window_end) + 1.0e-15)
            )
            if np.any(common):
                candidates = np.flatnonzero(common)
                local = candidates[int(np.argmax(imported_dp[candidates]))]
                imported_peak = float(imported_dp[local])
                imported_peak_time = float(imported_t[local])
                imported_window_n = int(candidates.size)
            else:
                imported_peak = None
                imported_peak_time = None
                imported_window_n = 0
        else:
            imported_peak = None
            imported_peak_time = None
            imported_window_n = 0
        selected = (
            abs(float(np.min(np.abs(selected_times - raw_peak_time))))
            <= 1.0e-12
            if raw_peak_time is not None and selected_times.size else False
        )
        nearest = (
            float(selected_times[np.argmin(np.abs(selected_times - raw_peak_time))])
            if raw_peak_time is not None and selected_times.size else None
        )
        difference = (
            float(imported_peak - raw_peak)
            if raw_peak is not None and imported_peak is not None else None
        )
        relative = (
            difference / raw_peak
            if difference is not None and raw_peak not in (None, 0.0) else None
        )
        out.append({
            "case_id": case["case_id"], "z_m_kg13": history["z"],
            "r_m": r_target,
            "method": PULSE_METHOD,
            "raw_peak_mpa": raw_peak,
            "raw_peak_time_s": raw_peak_time,
            "raw_metrics_status": raw_metrics.get("status"),
            "raw_phase_end_s": raw_metrics.get("phase_end_s"),
            "raw_frames": int(raw_t.size),
            "common_window_start_s": window_start,
            "common_window_end_s": window_end,
            "imported_peak_mpa": imported_peak,
            "imported_peak_time_s": imported_peak_time,
            "imported_metrics_status": imported_metrics.get("status"),
            "imported_phase_end_s": imported_metrics.get("phase_end_s"),
            "imported_samples_in_window": imported_window_n,
            "imported_frames": len(frames),
            "imported_cell_offset_max_m": float(
                np.max(np.abs(imported_offsets))) if imported_offsets.size else None,
            "peak_difference_mpa": difference,
            "peak_relative_loss": relative,
            "raw_peak_frame_selected": bool(selected),
            "nearest_selected_time_s": nearest,
            "nearest_selected_gap_s": (
                float(abs(nearest - raw_peak_time))
                if nearest is not None and raw_peak_time is not None else None
            ),
        })
    return out


# ---------------------------------------------------------------------------
# layer C: consistency of measured states and the program formulas


def shock_jump_residuals(pre: dict, post: dict, ds: float,
                         gamma: float) -> dict:
    """Moving-shock jump residuals for measured states.

    The conserved quantities are ``(rho, rho u, E)`` with the *laboratory*
    total energy density ``E = p/(gamma-1) + rho u^2/2``.  For a discontinuity
    moving at ``D_s`` the jump condition is ``[F] - D_s [U] = 0``, so the
    energy flux is ``u(E+p) - D_s E``.  Writing ``(E+p)(u-D_s)`` would subtract
    ``D_s p`` once too often on each side and manufacture an energy residual
    even for an exact Rankine-Hugoniot state.

    Residuals are normalised by ``rho_pre |D_s|``, ``rho_pre D_s^2`` and
    ``rho_pre |D_s|^3`` respectively.
    """
    rho1, u1, p1 = float(pre["rho"]), float(pre["u"]), float(pre["p"])
    rho2, u2, p2 = float(post["rho"]), float(post["u"]), float(post["p"])
    dt = float(ds)
    values = (rho1, u1, p1, rho2, u2, p2, dt)
    if not all(math.isfinite(value) for value in values):
        return {"status": "non_finite_state"}
    if rho1 <= 0.0 or rho2 <= 0.0 or abs(dt) <= 0.0:
        return {"status": "non_positive_scale"}
    mass = rho2 * (u2 - dt) - rho1 * (u1 - dt)
    momentum = (
        p2 + rho2 * u2 * (u2 - dt) - p1 - rho1 * u1 * (u1 - dt)
    )
    e_pre = p1 / (gamma - 1.0) + 0.5 * rho1 * u1 ** 2
    e_post = p2 / (gamma - 1.0) + 0.5 * rho2 * u2 ** 2
    energy = (
        u2 * (e_post + p2) - dt * e_post
        - u1 * (e_pre + p1) + dt * e_pre
    )
    speed = abs(dt)
    return {
        "status": "ok",
        "mass_residual_normalised": mass / (rho1 * speed),
        "momentum_residual_normalised": momentum / (rho1 * speed ** 2),
        "energy_residual_normalised": energy / (rho1 * speed ** 3),
        "energy_flux_form": "u(E+p) - D_s E",
    }


def rh_audit(case: dict, root: Path, ambient: dict) -> dict:
    from .data import ShockTrajectory

    frames, metadata = _standard_frames(root / case["standard"])
    with np.load(root / case["cache"], allow_pickle=True) as archive:
        shock_times = np.asarray(archive["shock_times"], float)
        shock_radius = np.asarray(archive["shock_radius"], float)
        shock_valid = np.asarray(archive["shock_valid"], bool)
        raw_times = np.asarray(archive["raw_times"], float)
    if shock_times.size != len(frames):
        raise AuditError(
            f"{case['case_id']}: cache shock frames {shock_times.size} do not "
            f"match standard frames {len(frames)}"
        )
    if raw_times.size != len(frames):
        raise AuditError(
            f"{case['case_id']}: cache raw_times {raw_times.size} do not match "
            f"standard frames {len(frames)}"
        )
    trajectory = ShockTrajectory.from_arrays(
        case_id=str(case["case_id"]), times=shock_times,
        radii=shock_radius, valid=shock_valid,
    )
    gamma = float(ambient["gamma"])
    results: list[dict] = []
    not_evaluable: list[dict] = []

    def reject(index: int, reason: str) -> None:
        not_evaluable.append({
            "frame_index": int(index),
            "raw_time_s": (
                float(raw_times[index]) if 0 <= index < raw_times.size else None
            ),
            "reason": reason,
        })

    for index in (len(frames) // 3, len(frames) // 2, (2 * len(frames)) // 3):
        if index <= 0 or index >= len(frames) - 1:
            reject(index, "central-difference stencil needs both neighbours")
            continue
        if not all(bool(shock_valid[position])
                   for position in (index - 1, index, index + 1)):
            reject(index, "a neighbouring shock frame is not valid")
            continue
        t_before, t_mid, t_after = (
            float(shock_times[index - 1]), float(shock_times[index]),
            float(shock_times[index + 1]),
        )
        if not (t_before < t_mid < t_after):
            reject(index, "shock time is not strictly increasing across the stencil")
            continue
        try:
            segment = trajectory.continuous_interval_containing(t_mid)
        except Exception as exc:  # GeometryError or no segment
            reject(index, f"no continuous shock segment at the stencil: {exc}")
            continue
        if not (segment[0] <= t_before and t_after <= segment[1]):
            reject(index, "the derivative stencil crosses a shock trajectory break")
            continue
        frame = frames[index]
        radius = frame["radius"]
        dr = float(np.median(np.diff(radius)))
        rs = float(shock_radius[index])
        ds = float((shock_radius[index + 1] - shock_radius[index - 1])
                   / (t_after - t_before))
        valid = (
            np.asarray(frame["valid_pressure"], bool)
            & np.asarray(frame["valid_density"], bool)
            & np.asarray(frame["valid_velocity"], bool)
            & np.isfinite(frame["pressure"])
            & np.isfinite(frame["density"])
            & np.isfinite(frame["velocity"])
        )
        air = np.asarray(frame["alpha_product"], float) < 0.05
        outside = np.flatnonzero((radius > rs) & valid & air)
        inside = np.flatnonzero((radius < rs) & valid & air)
        if outside.size < 4:
            reject(index, "fewer than four valid pure-air cells outside the shock")
            continue
        if inside.size < 10:
            reject(index, "fewer than ten valid pure-air cells inside the shock")
            continue
        pre_index = outside[:5]
        pre = {
            "rho": float(np.mean(frame["density"][pre_index])),
            "u": float(np.mean(frame["velocity"][pre_index])),
            "p": float(np.mean(frame["pressure"][pre_index])),
        }
        for cells_inside in (3, 5, 8):
            target = rs - cells_inside * dr
            inside_local = inside[np.argsort(np.abs(radius[inside] - target))]
            if inside_local.size == 0:
                reject(index, f"no valid cell {cells_inside} cells inside the shock")
                continue
            post_index = int(inside_local[0])
            post = {
                "rho": float(frame["density"][post_index]),
                "u": float(frame["velocity"][post_index]),
                "p": float(frame["pressure"][post_index]),
            }
            residuals = shock_jump_residuals(pre, post, ds, gamma)
            if residuals.get("status") != "ok":
                reject(index, f"non-finite or non-positive state: "
                              f"{residuals.get('status')}")
                continue
            results.append({
                "frame_index": index,
                "raw_time_s": float(raw_times[index]),
                "Ds_m_s": ds,
                "cells_inside": cells_inside,
                "dr_m": dr,
                "pre": pre, "post": post,
                "normalisation": {
                    "mass": "rho_pre * |D_s|",
                    "momentum": "rho_pre * D_s^2",
                    "energy": "rho_pre * |D_s|^3",
                },
                "status": residuals["status"],
                **{key: value for key, value in residuals.items()
                   if key != "status"},
            })
    return {
        "case_id": case["case_id"],
        "note": "Ds from central difference of measured shock positions; states measured, "
                "not synthesized from the Rankine-Hugoniot relations; energy flux is "
                "u(E+p) - D_s E with E = p/(gamma-1) + rho u^2/2.  The normalised "
                "energy residual is not a percentage of energy lost.",
        "normalisation": {
            "mass": "rho_pre * |D_s|",
            "momentum": "rho_pre * D_s^2",
            "energy": "rho_pre * |D_s|^3",
        },
        "samples": results,
        "not_evaluable": not_evaluable,
    }


def _trajectory_slope(values: np.ndarray, times: np.ndarray,
                      index: int) -> float | None:
    """Finite-difference slope of one interval edge at ``index``."""
    if values.size < 2 or times.size < 2:
        return None
    if 0 < index < values.size - 1:
        span = float(times[index + 1] - times[index - 1])
        if span == 0.0:
            return None
        return float((values[index + 1] - values[index - 1]) / span)
    if index == 0:
        span = float(times[1] - times[0])
        return None if span == 0.0 else float((values[1] - values[0]) / span)
    span = float(times[-1] - times[-2])
    return None if span == 0.0 else float((values[-1] - values[-2]) / span)


def _sample_air_outside(frame: dict, contact_up: float,
                        max_pure_cells: int = 5) -> list[dict]:
    """Nearest valid cells outside the contact interval.

    Up to ``max_pure_cells`` pure-air cells are kept; non-pure cells encountered
    before them are still listed so an excluded mixing cell is visible.
    """
    radius = np.asarray(frame["radius"], float)
    pressure = np.asarray(frame["pressure"], float)
    velocity = np.asarray(frame["velocity"], float)
    alpha = np.asarray(frame["alpha_product"], float)
    valid = np.asarray(frame["valid_pressure"], bool) & np.isfinite(pressure)
    outside = np.flatnonzero((radius > contact_up) & valid)
    out = []
    pure_count = 0
    for index in outside:
        pure = bool(alpha[index] < 0.05)
        if pure and pure_count >= max_pure_cells:
            break
        entry = {
            "cell_index": int(index),
            "radius_m": float(radius[index]),
            "offset_from_contact_up_m": float(radius[index] - contact_up),
            "pressure_pa": float(pressure[index]),
            "velocity_m_s": float(velocity[index]),
            "alpha_product": float(alpha[index]),
            "pure_air": pure,
        }
        if not pure:
            entry["excluded_reason"] = "not_pure_air"
        else:
            pure_count += 1
        out.append(entry)
    return out


def constraint_compatibility(case: dict, root: Path, ambient: dict) -> dict:
    with np.load(root / case["cache"], allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        contact_times = np.asarray(archive["contact_times"], float)
        low = np.asarray(archive["contact_low"], float)
        up = np.asarray(archive["contact_up"], float)
        valid = np.asarray(archive["contact_valid"], bool)
        raw_times = (
            np.asarray(archive["raw_times"], float)
            if "raw_times" in archive.files else contact_times
        )
    frames, _ = _standard_frames(root / case["standard"])
    precursor = metadata["case"]["precursor"]
    jwl = precursor["jwl"]
    v_sep = float(precursor["V_sep"])
    r_sep = float(precursor["R_sep"])

    def jwl_pressure(r_m: float) -> float:
        v = v_sep * (r_m / r_sep) ** 3
        return (jwl["A"] * math.exp(-jwl["R1"] * v)
                + jwl["B"] * math.exp(-jwl["R2"] * v)
                + jwl["C"] / v ** (1.0 + jwl["omega"]))

    t0 = metadata["case"].get("t0", precursor.get("t0"))
    t0 = float(t0) if t0 is not None else float(contact_times[0])
    valid_indices = [int(index) for index in np.flatnonzero(valid)
                     if index < len(frames) and index < low.size
                     and contact_times[index] >= t0]
    if valid_indices:
        picks = sorted({valid_indices[0],
                        valid_indices[len(valid_indices) // 2],
                        valid_indices[-1]})
    else:
        picks = []
    samples = []
    for index in picks:
        frame = frames[index]
        radius = np.asarray(frame["radius"], float)
        cells = _sample_air_outside(frame, float(up[index]))
        pure = [cell for cell in cells if cell["pure_air"]]
        p_low = jwl_pressure(float(low[index]))
        p_up = jwl_pressure(float(up[index]))
        ratios = [p_up / cell["pressure_pa"] for cell in pure
                  if cell["pressure_pa"] > 0.0]
        samples.append({
            "frame_index": int(index),
            "raw_time_s": float(raw_times[index]) if index < raw_times.size else None,
            "contact_model_time_s": float(contact_times[index]),
            "contact_low_m": float(low[index]),
            "contact_up_m": float(up[index]),
            "p_jwl_low_pa": p_low,
            "p_jwl_up_pa": p_up,
            "air_cells_outside": cells,
            "pure_air_cell_count": len(pure),
            "p_air_first_pure_pa": pure[0]["pressure_pa"] if pure else None,
            "ratio_jwl_up_over_air_by_distance": ratios,
            "ratio_jwl_up_over_air_first_pure": ratios[0] if ratios else None,
            "interval_edge_trajectory_derivative_m_s": {
                "low_edge": _trajectory_slope(low, contact_times, index),
                "up_edge": _trajectory_slope(up, contact_times, index),
                "note": "finite-difference slopes of the two interval edges; these are "
                        "not a physical speed bound, and no interval midpoint is used",
            },
            "air_velocity_first_pure_m_s": (
                pure[0]["velocity_m_s"] if pure else None
            ),
        })
    initial = metadata["case"].get("initial_compatibility", {})
    cached_rh = initial.get("event_frame_rh_diagnostic")
    states = initial.get("event_frame_shock_states", {})
    corrected_rh = None
    if states:
        from .one_d import _rh_diagnostic
        corrected_rh = _rh_diagnostic(
            float(states["rho_inner_kg_m3"]),
            float(states["u_inner_m_s"]),
            float(states["p_inner_pa"]),
            float(states["rho_outer_kg_m3"]),
            float(states["u_outer_m_s"]),
            float(states["p_outer_pa"]),
        )
    return {
        "case_id": case["case_id"],
        "initial_compatibility": initial,
        "sampling_note": "JWL pressure is compared with several pure-air cells outside "
                         "the contact interval at early, middle and late valid contact "
                         "frames after t0; pressure ratios are reported against distance "
                         "so the sensitivity to the sampling cell is visible.",
        "contact_samples": samples,
        "event_frame_rh_diagnostic": corrected_rh,
        "event_frame_rh_diagnostic_cache_recorded": cached_rh,
        "event_frame_rh_diagnostic_note": (
            "recomputed from the cached event-frame states with the corrected "
            "shock-frame total-enthalpy formula; the recorded value was produced "
            "by the earlier internal-energy formula and is kept only for traceability"
        ),
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
    "AuditError", "IMPORT_TOLERANCES", "PULSE_METHOD", "blast_metrics",
    "case_manifest", "constraint_compatibility", "fixed_radius_histories",
    "frame_keys", "implementation_checks", "import_comparison",
    "peak_preservation", "pulse_metrics", "read_xy", "rh_audit",
    "sha256_file", "shock_jump_residuals",
]
