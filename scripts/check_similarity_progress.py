#!/usr/bin/env python3
"""Read-only check that the real cases share one physical progress coordinate.

The coordinate is the existing theta = (t - t0) / T_star, with
T_star = R_sep / u_scale.  No network prediction is used and no time scale or
curve is shifted.  The script compares, at fixed theta:

* R_s / R_sep;
* contact lower/upper bounds / R_sep, keeping the interval;
* normalized air density, velocity and pressure at the same r / R_sep inside
  the conservative common layer;
* numerical resolution and frame/label coverage;
* the t0 event alignment.

Training cases are the decision group; validation cases are a check, not a
tuning set.  The script writes only new diagnostic JSON/CSV files.

This diagnostic deliberately does **not** decide whether the data are similar.
It reports statistics and completeness; a similarity call requires a
pre-registered quantitative criterion outside this script.  A missing frame,
an invalid field, an out-of-range query or an unchecked large-gap interpolation
makes the diagnostic incomplete and exits nonzero.

Air observations are treated field by field using valid_density,
valid_velocity and valid_pressure.  Query values are never extrapolated from a
frame endpoint and are not interpolated across a gap larger than
GAP_FACTOR * the case's median frame interval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import contact_fullcontact_pair as cf  # noqa: E402
import diagnose_contact_gradients as diag  # noqa: E402

C_CONFIG = (
    "configs/contact_air/"
    "split_train_r50_500_fullframes_similarity_none_pair.yaml"
)
THETA_MIN = 0.5
THETA_MAX = 60.0
THETA_STEP = 0.5
Q_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
WINDOWS = ((0.5, 10.0), (10.0, 30.0), (30.0, 60.0))
GAP_FACTOR = 2.0
FIELDS = ("rho", "u", "p")
AUDIT_CONFIG = "configs/contact_air/reference_audit_fullframes.yaml"


def _sha256(path: Path) -> str:
    return diag._sha256(Path(path))


def _median(values):
    values = np.asarray(
        [v for v in values if v is not None and math.isfinite(v)],
        dtype=np.float64,
    )
    return float(np.median(values)) if values.size else None


def _percentile(values, q):
    values = np.asarray(
        [v for v in values if v is not None and math.isfinite(v)],
        dtype=np.float64,
    )
    return float(np.percentile(values, q)) if values.size else None


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _grid_spacing(radius, scale):
    """Median normalized spacing from raw radii; divide by scale once."""
    radius = np.asarray(radius, dtype=np.float64)
    if radius.size < 2:
        return None
    return float(np.median(np.diff(np.sort(radius)) / float(scale)))


def _event_progress(raw_event_time, t0, time_scale, offset):
    """Event progress in model time, using t_model = t_raw + offset."""
    if raw_event_time is None or offset is None:
        return None
    return float((raw_event_time + offset - t0) / time_scale)


def _repeat_run_length(diffs, gap):
    """Longest consecutive run of intervals close to ``gap``."""
    if gap is None or not np.isfinite(gap) or np.asarray(diffs).size == 0:
        return 0
    hits = np.isclose(
        np.asarray(diffs, dtype=np.float64), gap, rtol=1.0e-3, atol=0.0
    )
    best = current = 0
    for hit in hits:
        current = current + 1 if bool(hit) else 0
        best = max(best, current)
    return int(best)


def _gap_context(times, lower, upper, gap, gap_limit):
    """Describe a gap without deciding why it exists.

    A repeated large interval is only a *candidate* explanation.  Whether it
    is the solver's sparse output cadence, a real missing frame, or an artefact
    of three-significant-figure output times is decided separately by
    ``_audit_frame_plan`` from the raw frame manifest and filter records.
    """
    diffs = np.diff(times)
    prev_gap = (
        float(times[lower] - times[lower - 1]) if lower > 0 else None
    )
    next_gap = (
        float(times[upper + 1] - times[upper])
        if upper + 1 < times.size else None
    )
    repeat_count = (
        int(np.count_nonzero(
            np.isclose(diffs, gap, rtol=1.0e-3, atol=0.0)
        ))
        if gap is not None and np.isfinite(gap) else None
    )
    repeat_run = _repeat_run_length(diffs, gap)
    if gap <= gap_limit:
        kind = "regular"
    elif prev_gap is None or next_gap is None:
        kind = "terminal_gap_candidate"
    else:
        neighbor = float(np.median([prev_gap, next_gap]))
        if repeat_run >= 3 or (repeat_count or 0) >= 3:
            kind = "repeated_large_gap_candidate"
        elif neighbor > 0.0 and gap > 3.0 * neighbor:
            kind = "isolated_gap_candidate"
        else:
            kind = "large_gap_candidate"
    return {
        "gap_kind": kind,
        "gap_kind_is_candidate": kind not in ("regular",),
        "prev_gap": prev_gap,
        "next_gap": next_gap,
        "gap_repeat_count": repeat_count,
        "gap_repeat_run_max": repeat_run,
    }


def _bracket(theta, times):
    """Bracket a query without endpoint extrapolation or unchecked gaps."""
    times = np.asarray(times, dtype=np.float64)
    empty_context = {
        "gap_kind": None,
        "gap_kind_is_candidate": None,
        "prev_gap": None,
        "next_gap": None,
        "gap_repeat_count": None,
        "gap_repeat_run_max": None,
    }
    if times.size < 2:
        return {
            "supported": False, "reason": "insufficient_points",
            "lower_index": None, "upper_index": None,
            "lower_theta": None, "upper_theta": None,
            "gap": None, "gap_limit": None,
            **empty_context,
        }
    if theta < times[0] or theta > times[-1]:
        return {
            "supported": False, "reason": "outside_range",
            "lower_index": None, "upper_index": None,
            "lower_theta": float(times[0]), "upper_theta": float(times[-1]),
            "gap": None, "gap_limit": None,
            **empty_context,
        }
    index = int(np.searchsorted(times, theta))
    if index < times.size and times[index] == theta:
        lower = upper = index
        gap = 0.0
    else:
        lower = max(0, index - 1)
        upper = min(index, times.size - 1)
        gap = float(times[upper] - times[lower])
    gap_limit = GAP_FACTOR * float(np.median(np.diff(times)))
    context = _gap_context(times, lower, upper, gap, gap_limit)
    if gap > gap_limit:
        return {
            "supported": False, "reason": "large_gap",
            "lower_index": int(lower), "upper_index": int(upper),
            "lower_theta": float(times[lower]),
            "upper_theta": float(times[upper]),
            "gap": gap, "gap_limit": gap_limit,
            **context,
        }
    return {
        "supported": True, "reason": "ok",
        "lower_index": int(lower), "upper_index": int(upper),
        "lower_theta": float(times[lower]),
        "upper_theta": float(times[upper]),
        "gap": gap, "gap_limit": gap_limit,
        **context,
    }

def _interp_series(theta, times, values, valid=None):
    """Interpolate one series only inside its valid support."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if int(valid.sum()) < 2:
        return {
            "supported": False, "reason": "insufficient_valid_points",
            "value": None,
        }
    bracket = _bracket(theta, times[valid])
    if not bracket["supported"]:
        return {"value": None, **bracket}
    lower = bracket["lower_index"]
    upper = bracket["upper_index"]
    selected_times = times[valid]
    selected_values = values[valid]
    if upper == lower:
        value = float(selected_values[lower])
    else:
        weight = (
            (theta - selected_times[lower])
            / (selected_times[upper] - selected_times[lower])
        )
        value = float(
            (1.0 - weight) * selected_values[lower]
            + weight * selected_values[upper]
        )
    return {"value": value, **bracket}


def _interp_field_at_radius(radius, values, valid, r_norm):
    """One field at one frame, preserving original point adjacency.

    A query may interpolate only between two adjacent valid points in the
    original radial ordering.  It never jumps over an invalid point or
    extrapolates beyond the stored radius range.
    """
    radius = np.asarray(radius, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if radius.size != values.size or radius.size != valid.size:
        return {
            "supported": False, "reason": "array_size_mismatch",
            "value": None, "radius_min": None, "radius_max": None,
        }
    order = np.argsort(radius)
    radius = radius[order]
    values = values[order]
    valid = valid[order]
    usable = valid & np.isfinite(radius) & np.isfinite(values)
    if int(usable.sum()) < 2:
        return {
            "supported": False, "reason": "insufficient_valid_points",
            "value": None,
            "radius_min": (
                float(radius[usable].min()) if usable.any() else None
            ),
            "radius_max": (
                float(radius[usable].max()) if usable.any() else None
            ),
        }
    if r_norm < radius[0] or r_norm > radius[-1]:
        return {
            "supported": False, "reason": "outside_field_radius",
            "value": None,
            "radius_min": float(radius[usable].min()),
            "radius_max": float(radius[usable].max()),
        }
    index = int(np.searchsorted(radius, r_norm))
    if index < radius.size and radius[index] == r_norm:
        if not usable[index]:
            return {
                "supported": False, "reason": "query_on_invalid_point",
                "value": None,
                "radius_min": float(radius[usable].min()),
                "radius_max": float(radius[usable].max()),
                "bracket_lower_index": int(index),
                "bracket_upper_index": int(index),
                "bracket_lower_radius": float(radius[index]),
                "bracket_upper_radius": float(radius[index]),
            }
        return {
            "supported": True, "reason": "ok",
            "value": float(values[index]),
            "radius_min": float(radius[usable].min()),
            "radius_max": float(radius[usable].max()),
            "bracket_lower_index": int(index),
            "bracket_upper_index": int(index),
            "bracket_lower_radius": float(radius[index]),
            "bracket_upper_radius": float(radius[index]),
            "n_valid": int(usable.sum()),
        }
    lower = index - 1
    upper = index
    if lower < 0 or upper >= radius.size:
        return {
            "supported": False, "reason": "outside_field_radius",
            "value": None,
            "radius_min": float(radius[usable].min()),
            "radius_max": float(radius[usable].max()),
        }
    if not usable[lower] or not usable[upper]:
        return {
            "supported": False, "reason": "spans_invalid_point",
            "value": None,
            "radius_min": float(radius[usable].min()),
            "radius_max": float(radius[usable].max()),
            "bracket_lower_index": int(lower),
            "bracket_upper_index": int(upper),
            "bracket_lower_radius": float(radius[lower]),
            "bracket_upper_radius": float(radius[upper]),
            "lower_valid": bool(usable[lower]),
            "upper_valid": bool(usable[upper]),
            "n_valid": int(usable.sum()),
        }
    weight = (r_norm - radius[lower]) / (radius[upper] - radius[lower])
    return {
        "supported": True, "reason": "ok",
        "value": float((1.0 - weight) * values[lower] + weight * values[upper]),
        "radius_min": float(radius[usable].min()),
        "radius_max": float(radius[usable].max()),
        "bracket_lower_index": int(lower),
        "bracket_upper_index": int(upper),
        "bracket_lower_radius": float(radius[lower]),
        "bracket_upper_radius": float(radius[upper]),
        "lower_valid": True,
        "upper_valid": True,
        "n_valid": int(usable.sum()),
    }


def _standard_entry(case_id, config_path=None):
    """Return the audit-config entry for one case, or ``None``."""
    path = Path(config_path) if config_path is not None else (
        _REPO_ROOT / AUDIT_CONFIG
    )
    if not path.is_file():
        return None
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - config is normally present
        return None
    for item in (data.get("audit", {}) or {}).get("cases", []) or []:
        if str(item.get("case_id")) == str(case_id):
            return item
    return None


def _load_standard_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"]))


def _plan_gap_interpretation(plan, large_gaps):
    """Classify repeated raw-output gaps only after the plan was checked."""
    if not plan.get("standard_available"):
        return "unverified_candidate"
    if not plan.get("all_frames_imported"):
        return "filter_or_import_gap"
    if not large_gaps:
        return "no_large_gaps"
    if any(
        gap.get("n_raw_between", 0) > gap.get("n_imported_between", 0)
        for gap in large_gaps
    ):
        return "raw_frame_not_imported"
    if any(
        gap.get("n_imported_between", 0) > gap.get("n_observed_between", 0)
        for gap in large_gaps
    ):
        return "imported_frame_not_observed"
    max_run = max(
        (gap.get("gap_repeat_run_max") or 0 for gap in large_gaps),
        default=0,
    )
    if max_run >= 3:
        return "sustained_raw_output_cadence"
    if str(plan.get("time_source", "")) == "profile_filename_3sigfig":
        return "isolated_repeated_gap_time_key_quantization_candidate"
    return "repeated_raw_output_gap_candidate"


def _frame_support(frame_id, theta, result):
    """Spatial support of one field at one time frame.

    Both bracketing frames are recorded separately so a two-frame
    space-time interpolation can be reconstructed even when one frame
    fails.
    """
    return {
        "frame_id": None if frame_id is None else int(frame_id),
        "theta": None if theta is None else float(theta),
        "supported": bool(result.get("supported", False)),
        "reason": result.get("reason"),
        "radius_min": result.get("radius_min"),
        "radius_max": result.get("radius_max"),
        "bracket_lower_radius": result.get("bracket_lower_radius"),
        "bracket_upper_radius": result.get("bracket_upper_radius"),
        "n_valid": result.get("n_valid"),
        "lower_valid": result.get("lower_valid"),
        "upper_valid": result.get("upper_valid"),
    }


class CaseData1D:
    """Native real-data fields and labels for one case."""

    def __init__(self, case):
        self.case = case
        obs = case.observations
        self.frame_ids = np.unique(obs.frame_index)
        self.frame_theta = []
        self.frame_radius = []
        self.frame_values = []
        self.frame_valid = []
        for frame in self.frame_ids:
            mask = obs.frame_index == frame
            self.frame_theta.append(
                (float(obs.time[mask][0]) - case.t0) / case.time_scale
            )
            raw_radius = np.asarray(obs.radius[mask], dtype=np.float64)
            self.frame_radius.append(raw_radius / case.length_scale)
            self.frame_values.append({
                "rho": np.asarray(obs.density[mask], dtype=np.float64)
                / case.rho_scale,
                "u": np.asarray(obs.velocity[mask], dtype=np.float64)
                / case.u_scale,
                "p": np.asarray(obs.pressure[mask], dtype=np.float64)
                / case.p_scale,
            })
            self.frame_valid.append({
                "rho": np.asarray(obs.valid_density[mask], dtype=bool),
                "u": np.asarray(obs.valid_velocity[mask], dtype=bool),
                "p": np.asarray(obs.valid_pressure[mask], dtype=bool),
            })
        self.frame_theta = np.asarray(self.frame_theta, dtype=np.float64)
        order = np.argsort(self.frame_theta)
        self.frame_theta = self.frame_theta[order]
        self.frame_ids = self.frame_ids[order]
        self.frame_radius = [self.frame_radius[index] for index in order]
        self.frame_values = [self.frame_values[index] for index in order]
        self.frame_valid = [self.frame_valid[index] for index in order]
        self.frame_dtheta = (
            np.diff(np.sort(self.frame_theta))
            if self.frame_theta.size > 1 else np.zeros(0)
        )

        labels = cf._full_labels(case)
        self.label_exclusions = {
            "excluded_outside_time": int(labels["excluded_outside_time"]),
            "excluded_no_slope": int(labels["excluded_no_slope"]),
            "total_frame_candidates": int(labels["total_frame_candidates"]),
        }
        self.label_theta = (
            np.asarray(labels["times"], dtype=np.float64) - case.t0
        ) / case.time_scale
        self.label_low = np.asarray(labels["low"], dtype=np.float64) / case.length_scale
        self.label_up = np.asarray(labels["up"], dtype=np.float64) / case.length_scale
        self.label_width = self.label_up - self.label_low
        order = np.argsort(self.label_theta)
        self.label_theta = self.label_theta[order]
        self.label_low = self.label_low[order]
        self.label_up = self.label_up[order]
        self.label_width = self.label_width[order]
        self.label_count = int(labels["n"])
        self.label_theta_min = float(self.label_theta.min())
        self.label_theta_max = float(self.label_theta.max())
        self.label_dtheta = (
            np.diff(np.sort(self.label_theta))
            if self.label_theta.size > 1 else np.zeros(0)
        )

        metadata = case.source.get("data_entry", {}).get("metadata", {})
        self.source_dir = metadata.get("source_dir")
        self.cell_count = metadata.get("cell_count")
        self.grid_dr_m = self._grid_dr_m()
        self.grid_dr_over_Rsep = (
            self.grid_dr_m / case.length_scale
            if self.grid_dr_m is not None else None
        )
        self.event = dict(case.event_alignment)
        self.event_raw_minus_t0_over_Tstar = (
            (self.event.get("event_time_raw_s") - case.t0) / case.time_scale
            if self.event.get("event_time_raw_s") is not None else None
        )
        self.event_model_over_Tstar = _event_progress(
            self.event.get("event_time_raw_s"), case.t0, case.time_scale,
            self.event.get("offset_s"),
        )
        self.frame_plan = self._audit_frame_plan()

    def _audit_frame_plan(self):
        """Compare the raw frame manifest, imported frames and the cache.

        The raw profile file names carry only three significant figures, so an
        apparent large interval in the imported time axis is not by itself
        evidence of a missing frame.  This audit reads the standard metadata
        written by the importer (selection policy, raw/imported/dropped counts,
        selected and raw keys) and the cache filter records, then reports what
        the plan actually contains.  It never asserts sparsity or a missing
        frame beyond what those records show.
        """
        case = self.case
        result = {
            "standard_available": False,
            "standard_path": None,
            "frame_selection_policy": None,
            "raw_frame_count": None,
            "imported_frame_count": None,
            "dropped_frame_count": None,
            "standard_frame_count": None,
            "selected_key_count": None,
            "raw_key_count": None,
            "cache_air_frame_count": int(self.frame_ids.size),
            "contact_sample_count": int(np.asarray(
                getattr(case.contact, "times", [])
            ).size),
            "contact_valid_count": int(np.count_nonzero(np.asarray(
                getattr(case.contact, "valid", []), dtype=bool
            ))),
            "shock_sample_count": int(np.asarray(
                getattr(case.shock, "times", [])
            ).size),
            "shock_valid_count": int(np.count_nonzero(np.asarray(
                getattr(case.shock, "valid", []), dtype=bool
            ))),
            "label_exclusions": dict(self.label_exclusions),
            "observation_diagnostics_available": False,
            "frames_without_air_observations": [],
            "large_gap_count": 0,
            "large_gaps": [],
            "max_repeat_run": 0,
            "interpretation": "unverified_candidate",
            "note": (
                "Repeated large gaps are candidates until the raw output plan "
                "and filter records are checked."
            ),
        }
        entry = _standard_entry(case.case_id)
        if entry is None:
            result["reason"] = "case_not_in_audit_config"
            return result
        standard = _REPO_ROOT / str(entry.get("standard", ""))
        result["standard_path"] = str(standard)
        if not standard.is_file():
            result["reason"] = "standard_file_missing"
            return result
        try:
            metadata = _load_standard_metadata(standard)
        except Exception as exc:  # corrupted or not a standard archive
            result["reason"] = f"standard_unreadable: {exc}"
            return result
        source = metadata.get("source", {}) or {}
        frame_times = np.asarray(metadata.get("frame_times", []), dtype=np.float64)
        raw_keys = list(source.get("raw_frame_keys", []) or [])
        selected_keys = list(source.get("selected_raw_frame_keys", []) or [])
        raw_key_times = np.asarray([float(key) for key in raw_keys], dtype=np.float64)
        result.update({
            "standard_available": True,
            "frame_selection_policy": source.get("frame_selection_policy"),
            "frame_selection_note": source.get("frame_selection_note"),
            "time_source": source.get("time_source"),
            "time_note": source.get("time_note"),
            "raw_frame_count": source.get("raw_frame_count"),
            "imported_frame_count": source.get("imported_frame_count"),
            "dropped_frame_count": source.get("dropped_frame_count"),
            "standard_frame_count": int(frame_times.size),
            "selected_key_count": len(selected_keys),
            "raw_key_count": len(raw_keys),
            "first_raw_time_s": source.get("first_raw_time_s"),
            "last_raw_time_s": source.get("last_raw_time_s"),
            "raw_keys_equal_selected": raw_keys == selected_keys,
            "all_frames_imported": bool(
                source.get("frame_selection_policy") == "all_frames"
                and source.get("raw_frame_count") == source.get("imported_frame_count")
                and source.get("dropped_frame_count") == 0
                and int(frame_times.size) == source.get("raw_frame_count")
                == len(selected_keys) == len(raw_keys)
                and raw_keys == selected_keys
            ),
        })

        # Filter records: which frames never produced air observations, and
        # why.  The separation report and observation diagnostics were written
        # by the same cache builder, so this is the actual filter trail.
        contact_times = np.asarray(
            getattr(case.contact, "times", []), dtype=np.float64
        )
        observed_model = np.unique(np.asarray(case.observations.time, dtype=np.float64))
        separation_frames = list(
            (getattr(case, "separation", {}) or {}).get("frames", []) or []
        )
        obs_diagnostics = dict(
            getattr(case.observations, "diagnostics", {}) or {}
        )
        result["observation_diagnostics_available"] = bool(obs_diagnostics)
        obs_frame_reports = {
            int(item.get("frame")): item
            for item in obs_diagnostics.get("frames", []) or []
            if item.get("frame") is not None
        }
        if observed_model.size:
            observed_set = set(np.round(observed_model, 15).tolist())
            for index, time_value in enumerate(contact_times):
                if round(float(time_value), 15) in observed_set:
                    continue
                separation = (
                    separation_frames[index]
                    if index < len(separation_frames) else {}
                )
                report = obs_frame_reports.get(index, {})
                result["frames_without_air_observations"].append({
                    "frame_index": int(index),
                    "time_model_s": float(time_value),
                    "theta": float(
                        (time_value - case.t0) / case.time_scale
                    ),
                    "separation_status": separation.get("status"),
                    "contact_locatable": separation.get("contact_locatable"),
                    "shock_identified": separation.get("shock_identified"),
                    "pure_air_between": separation.get("pure_air_between"),
                    "separation_reasons": separation.get("reasons"),
                    "observation_selected": report.get("selected", 0),
                    "observation_rejected": report.get("rejected", {}),
                })

        # Audit every observed large interval against the raw key manifest.
        times = self.frame_theta
        diffs = np.diff(times)
        if times.size > 1:
            median = float(np.median(diffs))
            limit = GAP_FACTOR * median
            offset = self.event.get("offset_s")
            for index in np.flatnonzero(diffs > limit):
                lower, upper = int(index), int(index + 1)
                gap = float(diffs[lower])
                context = _gap_context(times, lower, upper, gap, limit)
                raw_lower = (
                    float(times[lower]) * case.time_scale + case.t0 - offset
                    if offset is not None else None
                )
                raw_upper = (
                    float(times[upper]) * case.time_scale + case.t0 - offset
                    if offset is not None else None
                )
                n_raw_between = n_imported_between = n_observed_between = None
                if raw_lower is not None and raw_key_times.size:
                    tol = 1.0e-12 * max(1.0, abs(raw_upper - raw_lower))
                    n_raw_between = int(np.count_nonzero(
                        (raw_key_times > raw_lower + tol)
                        & (raw_key_times < raw_upper - tol)
                    ))
                    n_imported_between = int(np.count_nonzero(
                        (frame_times > raw_lower + tol)
                        & (frame_times < raw_upper - tol)
                    ))
                    n_observed_between = int(np.count_nonzero(
                        (times > times[lower] + 1.0e-12)
                        & (times < times[upper] - 1.0e-12)
                    ))
                result["large_gaps"].append({
                    "theta_lower": float(times[lower]),
                    "theta_upper": float(times[upper]),
                    "gap": gap,
                    "gap_kind": context["gap_kind"],
                    "gap_repeat_count": context["gap_repeat_count"],
                    "gap_repeat_run_max": context["gap_repeat_run_max"],
                    "prev_gap": context["prev_gap"],
                    "next_gap": context["next_gap"],
                    "raw_time_lower_s": raw_lower,
                    "raw_time_upper_s": raw_upper,
                    "n_raw_between": n_raw_between,
                    "n_imported_between": n_imported_between,
                    "n_observed_between": n_observed_between,
                })
        result["large_gap_count"] = len(result["large_gaps"])
        result["max_repeat_run"] = max(
            (gap.get("gap_repeat_run_max") or 0
             for gap in result["large_gaps"]),
            default=0,
        )
        result["interpretation"] = _plan_gap_interpretation(
            result, result["large_gaps"]
        )
        return result

    def _grid_dr_m(self):
        values = []
        obs = self.case.observations
        for frame in self.frame_ids:
            mask = obs.frame_index == frame
            radius = np.asarray(obs.radius[mask], dtype=np.float64)
            radius = radius[np.isfinite(radius)]
            if radius.size > 1:
                values.append(float(np.median(np.diff(np.sort(radius)))))
        return _median(values)

    def contact(self, theta):
        low = _interp_series(theta, self.label_theta, self.label_low)
        up = _interp_series(theta, self.label_theta, self.label_up)
        return {"low": low, "up": up}

    def shock(self, theta):
        t = self.case.t0 + float(theta) * self.case.time_scale
        if t < self.case.t0 or t > self.case.t_end:
            return {
                "supported": False, "reason": "outside_t_range",
                "value": None,
            }
        radius = self.case.shock_tensors(np.asarray([t], dtype=np.float64))[0][0]
        value = float(radius) / self.case.length_scale
        if not math.isfinite(value):
            return {
                "supported": False, "reason": "non_finite",
                "value": None,
            }
        return {"supported": True, "reason": "ok", "value": value}

    def fields(self, theta, r_norm):
        bracket = _bracket(theta, self.frame_theta)
        if not bracket["supported"]:
            return {
                field: {
                    "supported": False, "value": None,
                    "reason": bracket["reason"],
                    "bracket": bracket,
                    "lower_frame_support": None,
                    "upper_frame_support": None,
                }
                for field in FIELDS
            }
        lower = bracket["lower_index"]
        upper = bracket["upper_index"]
        result = {}
        for field in FIELDS:
            first = _interp_field_at_radius(
                self.frame_radius[lower], self.frame_values[lower][field],
                self.frame_valid[lower][field], r_norm,
            )
            if upper == lower:
                second = first
            else:
                second = _interp_field_at_radius(
                    self.frame_radius[upper], self.frame_values[upper][field],
                    self.frame_valid[upper][field], r_norm,
                )
            lower_support = _frame_support(
                self.frame_ids[lower], self.frame_theta[lower], first
            )
            upper_support = _frame_support(
                self.frame_ids[upper], self.frame_theta[upper], second
            )
            if not first["supported"] or not second["supported"]:
                failures = []
                if not first["supported"]:
                    failures.append(f"lower_frame:{first['reason']}")
                if not second["supported"]:
                    failures.append(f"upper_frame:{second['reason']}")
                result[field] = {
                    "supported": False, "value": None,
                    "reason": first["reason"] if not first["supported"]
                              else second["reason"],
                    "failure_reasons": failures,
                    "bracket": bracket,
                    "frame_id_lower": int(self.frame_ids[lower]),
                    "frame_id_upper": int(self.frame_ids[upper]),
                    "lower_frame_support": lower_support,
                    "upper_frame_support": upper_support,
                    "n_valid_lower": first.get("n_valid"),
                    "n_valid_upper": second.get("n_valid"),
                    "n_valid": min(
                        value for value in (
                            first.get("n_valid"), second.get("n_valid")
                        ) if value is not None
                    ) if any(
                        value is not None for value in (
                            first.get("n_valid"), second.get("n_valid")
                        )
                    ) else None,
                    "lower_valid": first.get("lower_valid"),
                    "upper_valid": second.get("upper_valid"),
                    "radius_min": min(
                        (value for value in (
                            lower_support.get("radius_min"),
                            upper_support.get("radius_min"),
                        ) if value is not None),
                        default=None,
                    ),
                    "radius_max": max(
                        (value for value in (
                            lower_support.get("radius_max"),
                            upper_support.get("radius_max"),
                        ) if value is not None),
                        default=None,
                    ),
                }
                continue
            if upper == lower:
                value = first["value"]
            else:
                weight = (
                    (theta - self.frame_theta[lower])
                    / (self.frame_theta[upper] - self.frame_theta[lower])
                )
                value = float(
                    (1.0 - weight) * first["value"]
                    + weight * second["value"]
                )
            result[field] = {
                "supported": True, "value": value, "reason": "ok",
                "bracket": bracket,
                "lower_frame": int(lower),
                "upper_frame": int(upper),
                "frame_id_lower": int(self.frame_ids[lower]),
                "frame_id_upper": int(self.frame_ids[upper]),
                "lower_frame_support": lower_support,
                "upper_frame_support": upper_support,
                "n_valid_lower": first.get("n_valid"),
                "n_valid_upper": second.get("n_valid"),
                "n_valid": min(
                    value for value in (
                        first.get("n_valid"), second.get("n_valid")
                    ) if value is not None
                ),
                "lower_valid": True,
                "upper_valid": True,
                "radius_min": min(
                    value for value in (
                        first.get("radius_min"), second.get("radius_min")
                    ) if value is not None
                ),
                "radius_max": max(
                    value for value in (
                        first.get("radius_max"), second.get("radius_max")
                    ) if value is not None
                ),
                "bracket_lower_radius": first.get("bracket_lower_radius"),
                "bracket_upper_radius": second.get("bracket_upper_radius"),
            }
        return result


def _common_layer(low, up, shock):
    """Conservative common layer: outside every contact interval."""
    lower = float(np.max(up))
    upper = float(np.min(shock))
    return lower, upper, upper - lower


def _theta_grid():
    n = int(round((THETA_MAX - THETA_MIN) / THETA_STEP))
    return THETA_MIN + np.arange(n + 1, dtype=np.float64) * THETA_STEP


def _curve_metrics(curves):
    theta = np.asarray(curves[0]["theta"], dtype=np.float64)
    names = ("shock_over_Rsep", "contact_low_over_Rsep", "contact_up_over_Rsep")
    arrays = {
        name: np.stack([
            np.asarray([
                np.nan if value is None else value
                for value in row[name]
            ], dtype=np.float64)
            for row in curves
        ])
        for name in names
    }
    result = {
        "n_theta_total": int(theta.size),
        "theta_counts": {},
    }
    for name, values in arrays.items():
        complete = np.all(np.isfinite(values), axis=0)
        result["theta_counts"][name] = {
            "n_complete": int(complete.sum()),
            "fraction": (
                float(complete.sum() / theta.size) if theta.size else None
            ),
        }
        if not complete.any():
            result[name] = None
            continue
        subset = values[:, complete]
        median = np.median(subset, axis=0)
        spread = subset.max(axis=0) - subset.min(axis=0)
        per_case = {}
        for index, row in enumerate(curves):
            deviation = subset[index] - median
            per_case[row["case_id"]] = {
                "rms": float(np.sqrt(np.mean(deviation * deviation))),
                "max_abs": float(np.max(np.abs(deviation))),
                "mean": float(np.mean(deviation)),
            }
        result[name] = {
            "n_complete": int(complete.sum()),
            "spread_median": _median(spread),
            "spread_p90": _percentile(spread, 90),
            "spread_max": float(np.max(spread)),
            "per_case": per_case,
        }
    contact_complete = (
        np.all(np.isfinite(arrays["contact_low_over_Rsep"]), axis=0)
        & np.all(np.isfinite(arrays["contact_up_over_Rsep"]), axis=0)
    )
    result["theta_counts"]["contact_envelope"] = {
        "n_complete": int(contact_complete.sum()),
        "fraction": (
            float(contact_complete.sum() / theta.size) if theta.size else None
        ),
    }
    if contact_complete.any():
        low = arrays["contact_low_over_Rsep"][:, contact_complete]
        up = arrays["contact_up_over_Rsep"][:, contact_complete]
        envelope = up.max(axis=0) - low.min(axis=0)
        result["contact_envelope"] = {
            "n_complete": int(contact_complete.sum()),
            "median": _median(envelope),
            "p90": _percentile(envelope, 90),
            "max": float(np.max(envelope)),
        }
    else:
        result["contact_envelope"] = None
    return result

def _window_aggregate(rows, key):
    values = [
        row[key] for row in rows
        if row.get(key) is not None and math.isfinite(row[key])
    ]
    return {
        "n": len(values),
        "median": _median(values),
        "p90": _percentile(values, 90),
        "max": max(values) if values else None,
    }


def _aggregate_numeric(values):
    values = [
        float(value) for value in values
        if value is not None and math.isfinite(value)
    ]
    return {
        "n": len(values),
        "median": _median(values),
        "p90": _percentile(values, 90),
        "max": max(values) if values else None,
    }


def _curve_windows(curves):
    """Window metrics from the curves alone, one row per theta.

    Contact and shock statistics never pass through the air common layer and
    never repeat a theta once per q fraction.  ``n_theta`` is the number of
    theta slots in the window; ``n_theta_complete`` is how many of those have
    a finite value for every case.
    """
    theta = np.asarray(curves[0]["theta"], dtype=np.float64)
    names = ("shock_over_Rsep", "contact_low_over_Rsep", "contact_up_over_Rsep")
    arrays = {
        name: np.stack([
            np.asarray([
                np.nan if value is None else value
                for value in row[name]
            ], dtype=np.float64)
            for row in curves
        ])
        for name in names
    }
    out = {}
    for low, high in WINDOWS:
        mask = (theta >= low) & (theta < high)
        n_theta = int(np.count_nonzero(mask))
        entry = {
            "n_theta": n_theta,
            "n_cases": len(curves),
            "source": "curves; each theta counted once, independent of q",
        }
        for array_name, output_name in (
            ("shock_over_Rsep", "shock_spread"),
            ("contact_low_over_Rsep", "contact_low_spread"),
            ("contact_up_over_Rsep", "contact_up_spread"),
        ):
            subset = arrays[array_name][:, mask]
            complete = np.all(np.isfinite(subset), axis=0)
            spread = (
                subset.max(axis=0) - subset.min(axis=0)
                if complete.any() else np.zeros(0)
            )
            entry[output_name] = {
                "n_theta": n_theta,
                "n_theta_complete": int(complete.sum()),
                **_aggregate_numeric(spread),
            }
        contact_complete = (
            np.all(np.isfinite(arrays["contact_low_over_Rsep"][:, mask]), axis=0)
            & np.all(np.isfinite(arrays["contact_up_over_Rsep"][:, mask]), axis=0)
        )
        if contact_complete.any():
            contact_low = arrays["contact_low_over_Rsep"][:, mask][
                :, contact_complete
            ]
            contact_up = arrays["contact_up_over_Rsep"][:, mask][
                :, contact_complete
            ]
            envelope = contact_up.max(axis=0) - contact_low.min(axis=0)
        else:
            envelope = np.zeros(0)
        entry["contact_envelope"] = {
            "n_theta": n_theta,
            "n_theta_complete": int(contact_complete.sum()),
            **_aggregate_numeric(envelope),
        }
        out[f"{low:g}-{high:g}"] = entry
    return out


def _air_windows(rows):
    """Air-field query windows; these do depend on q and the common layer."""
    out = {}
    for low, high in WINDOWS:
        subset = [row for row in rows if low <= row["theta"] < high]
        thetas = sorted({row["theta"] for row in subset})
        entry = {
            "n_theta": len(thetas),
            "n_theta_q_queries": len(subset),
            "q_fractions": len(Q_FRACTIONS),
            "source": "air queries; one row per theta and q fraction",
            "common_width": _window_aggregate(subset, "common_width"),
        }
        for field in FIELDS:
            complete_rows = [
                row for row in subset
                if row["fields"][field].get("complete", False)
            ]
            complete_thetas = sorted({row["theta"] for row in complete_rows})
            entry[field] = {
                "n_theta": len(thetas),
                "n_theta_q_queries": len(subset),
                "n_complete_queries": len(complete_rows),
                "n_theta_with_complete_query": len(complete_thetas),
                "std": _window_aggregate(complete_rows, f"{field}_std"),
                "relstd": _window_aggregate(complete_rows, f"{field}_relstd"),
            }
        out[f"{low:g}-{high:g}"] = entry
    return out

def _record_missing(missing, group, theta, q, case_id, field, reason,
                    evidence=None, **details):
    row = {
        "group": group,
        "theta": float(theta),
        "q": None if q is None else float(q),
        "case_id": case_id,
        "field": field,
        "reason": reason,
        "gap_kind": details.get("gap_kind"),
    }
    missing.append(row)
    if evidence is not None:
        evidence.append({
            "group": group,
            "theta": float(theta),
            "q": None if q is None else float(q),
            "kind": "contact" if field in ("contact_low", "contact_up")
                    else ("shock" if field == "shock" else "air"),
            "case_id": case_id,
            "field": field,
            "supported": False,
            "reason": reason,
            **details,
        })


def _record_support(evidence, group, theta, q, kind, case_id, field,
                    supported, reason, **details):
    evidence.append({
        "group": group,
        "theta": float(theta),
        "q": None if q is None else float(q),
        "kind": kind,
        "case_id": case_id,
        "field": field,
        "supported": bool(supported),
        "reason": reason,
        **details,
    })

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    trainer, _ = diag._build_trainer(C_CONFIG)
    groups = {
        "train": [CaseData1D(case) for case in trainer.cases],
        "validation": [
            CaseData1D(case) for case in trainer.validation_cases
        ],
    }
    thetas = _theta_grid()
    missing = []
    evidence = []

    payload = {
        "format": "spherical_blast_pinn.similarity_progress_check",
        "version": 4,
        "config": C_CONFIG,
        "config_sha256": _sha256(Path(C_CONFIG)),
        "coordinate": {
            "theta": "(t - t0) / T_star",
            "T_star": "R_sep / u_scale",
            "network_input": "theta / 64 retained",
            "theta_grid": [THETA_MIN, THETA_MAX, THETA_STEP],
            "q_fractions": list(Q_FRACTIONS),
            "gap_factor": GAP_FACTOR,
            "gap_classification": "candidate_only",
            "gap_classification_note": (
                "Repeated large intervals are labelled as candidates. A "
                "sparse-output or missing-frame verdict is only formed after "
                "the raw frame manifest and filter records are read."
            ),
            "no_time_shift": True,
            "no_extrapolation": True,
            "no_network_prediction": True,
        },
        "assessment": {
            "status": "not_assessed",
            "note": (
                "Statistics only. No similarity threshold was pre-registered "
                "in this script, so it does not declare similar/not similar."
            ),
        },
        "groups": {},
    }

    for group_name, cases in groups.items():
        expected_ids = [data.case.case_id for data in cases]
        curves = []
        for data in cases:
            shock = []
            low = []
            up = []
            for theta in thetas:
                s = data.shock(theta)
                c = data.contact(theta)
                shock.append(s["value"] if s["supported"] else None)
                low.append(c["low"]["value"] if c["low"]["supported"] else None)
                up.append(c["up"]["value"] if c["up"]["supported"] else None)
                shock_details = {
                    "lower_theta": s.get("lower_theta"),
                    "upper_theta": s.get("upper_theta"),
                    "gap": s.get("gap"),
                    "gap_limit": s.get("gap_limit"),
                    "gap_kind": s.get("gap_kind"),
                    "gap_kind_is_candidate": s.get("gap_kind_is_candidate"),
                    "gap_repeat_count": s.get("gap_repeat_count"),
                    "gap_repeat_run_max": s.get("gap_repeat_run_max"),
                }
                if s["supported"]:
                    _record_support(
                        evidence, group_name, theta, None, "shock",
                        data.case.case_id, "shock", True, "ok",
                        **shock_details,
                    )
                else:
                    _record_missing(
                        missing, group_name, theta, None,
                        data.case.case_id, "shock", s["reason"],
                        evidence=evidence, **shock_details,
                    )
                for field, result in (("contact_low", c["low"]),
                                      ("contact_up", c["up"])):
                    details = {
                        "lower_theta": result.get("lower_theta"),
                        "upper_theta": result.get("upper_theta"),
                        "gap": result.get("gap"),
                        "gap_limit": result.get("gap_limit"),
                        "gap_kind": result.get("gap_kind"),
                        "gap_kind_is_candidate":
                            result.get("gap_kind_is_candidate"),
                        "gap_repeat_count": result.get("gap_repeat_count"),
                        "gap_repeat_run_max":
                            result.get("gap_repeat_run_max"),
                    }
                    if result["supported"]:
                        _record_support(
                            evidence, group_name, theta, None, "contact",
                            data.case.case_id, field, True, "ok",
                            **details,
                        )
                    else:
                        _record_missing(
                            missing, group_name, theta, None,
                            data.case.case_id, field, result["reason"],
                            evidence=evidence, **details,
                        )
            curves.append({
                "case_id": data.case.case_id,
                "role": data.case.role,
                "theta": [float(v) for v in thetas],
                "shock_over_Rsep": shock,
                "contact_low_over_Rsep": low,
                "contact_up_over_Rsep": up,
            })
        curve_metrics = _curve_metrics(curves)

        rows = []
        for theta in thetas:
            contact_values = []
            shock_values = []
            for data in cases:
                contact_values.append(data.contact(theta))
                shock_values.append(data.shock(theta))
            if not all(
                item["low"]["supported"] and item["up"]["supported"]
                for item in contact_values
            ) or not all(item["supported"] for item in shock_values):
                continue
            low = np.asarray([item["low"]["value"] for item in contact_values])
            up = np.asarray([item["up"]["value"] for item in contact_values])
            shock = np.asarray([item["value"] for item in shock_values])
            # Conservative common layer: outside every contact interval.
            common_low, common_high, width = _common_layer(low, up, shock)
            if width <= 0.0:
                _record_missing(
                    missing, group_name, theta, None, None,
                    "common_domain", "empty_common_layer",
                )
                continue
            for q in Q_FRACTIONS:
                r_norm = common_low + float(q) * width
                field_values = {field: [] for field in FIELDS}
                for data in cases:
                    values = data.fields(theta, r_norm)
                    for field in FIELDS:
                        result = values[field]
                        bracket = result.get("bracket", {})
                        lower_support = result.get("lower_frame_support") or {}
                        upper_support = result.get("upper_frame_support") or {}
                        details = {
                            "r_over_Rsep": float(r_norm),
                            "frame_id_lower": result.get("frame_id_lower"),
                            "frame_id_upper": result.get("frame_id_upper"),
                            "gap_kind_is_candidate":
                                bracket.get("gap_kind_is_candidate"),
                            "gap_repeat_count": bracket.get("gap_repeat_count"),
                            "gap_repeat_run_max":
                                bracket.get("gap_repeat_run_max"),
                            "lower_theta": bracket.get("lower_theta"),
                            "upper_theta": bracket.get("upper_theta"),
                            "gap": bracket.get("gap"),
                            "gap_limit": bracket.get("gap_limit"),
                            "gap_kind": bracket.get("gap_kind"),
                            "radius_min": result.get("radius_min"),
                            "radius_max": result.get("radius_max"),
                            "bracket_lower_radius": result.get(
                                "bracket_lower_radius"
                            ),
                            "bracket_upper_radius": result.get(
                                "bracket_upper_radius"
                            ),
                            "n_valid": result.get("n_valid"),
                            "lower_valid": result.get("lower_valid"),
                            "upper_valid": result.get("upper_valid"),
                            "failure_reasons": result.get("failure_reasons"),
                        }
                        for prefix, support in (
                            ("lower", lower_support),
                            ("upper", upper_support),
                        ):
                            for key in (
                                "frame_id", "theta", "supported", "reason",
                                "radius_min", "radius_max",
                                "bracket_lower_radius",
                                "bracket_upper_radius", "n_valid",
                                "lower_valid", "upper_valid",
                            ):
                                details[f"{prefix}_frame_{key}"] = support.get(key)
                        if result["supported"]:
                            field_values[field].append(result["value"])
                            _record_support(
                                evidence, group_name, theta, q, "air",
                                data.case.case_id, field, True, "ok",
                                **details,
                            )
                        else:
                            _record_missing(
                                missing, group_name, theta, q,
                                data.case.case_id, field, result["reason"],
                                evidence=evidence, **details,
                            )
                row = {
                    "theta": float(theta),
                    "q": float(q),
                    "r_over_Rsep": float(r_norm),
                    "common_width": float(width),
                    "contact_low_spread": float(low.max() - low.min()),
                    "contact_up_spread": float(up.max() - up.min()),
                    "contact_envelope": float(up.max() - low.min()),
                    "shock_spread": float(shock.max() - shock.min()),
                    "fields": {},
                }
                for field in FIELDS:
                    values = np.asarray(field_values[field], dtype=np.float64)
                    if values.size != len(cases):
                        row["fields"][field] = {
                            "complete": False,
                            "n_valid": int(values.size),
                        }
                        continue
                    mean = float(values.mean())
                    std = float(values.std())
                    relative = std / abs(mean) if abs(mean) > 1e-12 else None
                    row[f"{field}_mean"] = mean
                    row[f"{field}_std"] = std
                    row[f"{field}_relstd"] = relative
                    row["fields"][field] = {
                        "complete": True,
                        "n_valid": int(values.size),
                        "mean": mean,
                        "std": std,
                        "relstd": relative,
                    }
                rows.append(row)

        theta_q_total = len(thetas) * len(Q_FRACTIONS)
        field_query_counts = {}
        for field in FIELDS:
            complete = sum(
                1 for row in rows if row["fields"][field]["complete"]
            )
            field_query_counts[field] = {
                "n_complete": int(complete),
                "n_total": int(theta_q_total),
                "fraction": (
                    float(complete / theta_q_total)
                    if theta_q_total else None
                ),
            }
        query_counts = {
            "theta_count": int(len(thetas)),
            "theta_q_count": int(theta_q_total),
            "q_fraction_count": len(Q_FRACTIONS),
            "note": (
                "theta_count counts independent time positions; theta_q_count "
                "counts air queries (each theta repeated once per q fraction)."
            ),
            "field_complete": field_query_counts,
        }
        curve_windows = _curve_windows(curves)
        air_windows = _air_windows(rows)

        grid = [
            data.grid_dr_over_Rsep for data in cases
            if data.grid_dr_over_Rsep is not None
        ]
        contact_width = [
            float(np.median(data.label_width)) for data in cases
        ]
        group = {
            "role": group_name,
            "expected_case_ids": expected_ids,
            "cases": [],
            "curves": curves,
            "curve_metrics": curve_metrics,
            "query_counts": query_counts,
            "theta_counts": curve_metrics["theta_counts"],
            "curve_windows": curve_windows,
            "air_windows": air_windows,
            "field_rows": rows,
            "resolution": {
                "cells_per_case": [data.cell_count for data in cases],
                "grid_dr_over_Rsep_median": _median(grid),
                "grid_dr_over_Rsep_min": min(grid) if grid else None,
                "grid_dr_over_Rsep_max": max(grid) if grid else None,
                "contact_width_over_Rsep_median": _median(contact_width),
                "contact_width_over_Rsep_min": min(contact_width) if contact_width else None,
                "contact_width_over_Rsep_max": max(contact_width) if contact_width else None,
            },
        }
        for data in cases:
            group["cases"].append({
                "case_id": data.case.case_id,
                "role": data.case.role,
                "t0_s": float(data.case.t0),
                "t_end_s": float(data.case.t_end),
                "T_star_s": float(data.case.time_scale),
                "R_sep_m": float(data.case.length_scale),
                "R0_over_Rsep": float(
                    data.case.charge_radius_m / data.case.length_scale
                ),
                "Rs0_over_Rsep": float(
                    data.case.initial.Rs0 / data.case.length_scale
                ),
                "Rc0_over_Rsep": float(
                    data.case.initial.Rc0 / data.case.length_scale
                ),
                "Rs_data_t0_over_Rsep": float(
                    data.case.initial.Rs_data_at_t0 / data.case.length_scale
                ),
                "event_frame": data.event.get("event_frame"),
                "event_raw_minus_t0_over_Tstar":
                    data.event_raw_minus_t0_over_Tstar,
                "event_model_over_Tstar": data.event_model_over_Tstar,
                "event_offset_over_Tstar": (
                    data.event.get("offset_s") / data.case.time_scale
                    if data.event.get("offset_s") is not None else None
                ),
                "event_in_first_frame": data.event.get("event_in_first_frame"),
                "preceding_frame_confirmed":
                    data.event.get("preceding_frame_confirmed"),
                "contact_labels": data.label_count,
                "contact_theta_min": data.label_theta_min,
                "contact_theta_max": data.label_theta_max,
                "contact_dtheta_median": _median(data.label_dtheta),
                "contact_dtheta_max": (
                    float(np.max(data.label_dtheta))
                    if data.label_dtheta.size else None
                ),
                "frame_count": int(len(data.frame_ids)),
                "frame_theta_min": float(np.min(data.frame_theta)),
                "frame_theta_max": float(np.max(data.frame_theta)),
                "frame_dtheta_median": _median(data.frame_dtheta),
                "frame_dtheta_max": (
                    float(np.max(data.frame_dtheta))
                    if data.frame_dtheta.size else None
                ),
                "grid_dr_m": data.grid_dr_m,
                "grid_dr_over_Rsep": data.grid_dr_over_Rsep,
                "source_dir": data.source_dir,
                "frame_plan": data.frame_plan,
            })
        interpretations: dict[str, int] = {}
        for data in cases:
            key = str(data.frame_plan.get("interpretation"))
            interpretations[key] = interpretations.get(key, 0) + 1
        group["frame_plan_summary"] = {
            "n_cases": len(cases),
            "n_standard_available": sum(
                1 for data in cases
                if data.frame_plan.get("standard_available")
            ),
            "n_all_frames_imported": sum(
                1 for data in cases
                if data.frame_plan.get("all_frames_imported")
            ),
            "n_cases_with_large_gaps": sum(
                1 for data in cases
                if data.frame_plan.get("large_gap_count", 0)
            ),
            "interpretation_counts": interpretations,
            "note": (
                "A repeated gap is a candidate explanation. The interpretation "
                "is limited to what the raw key manifest and filter records "
                "show; it does not prove the solver cadence beyond the recorded "
                "output files."
            ),
        }
        payload["groups"][group_name] = group

    for group_name, group in payload["groups"].items():
        res = group["resolution"]
        contact_width = res["contact_width_over_Rsep_median"]
        curve_window = group["curve_windows"]["30-60"]
        air_window = group["air_windows"]["30-60"]
        group["indicators"] = {
            "late_lower_spread_median":
                curve_window["contact_low_spread"]["median"],
            "late_upper_spread_median":
                curve_window["contact_up_spread"]["median"],
            "late_envelope_width_median":
                curve_window["contact_envelope"]["median"],
            "late_lower_spread_in_contact_widths": (
                curve_window["contact_low_spread"]["median"] / contact_width
                if curve_window["contact_low_spread"]["median"] is not None
                and contact_width else None
            ),
            "late_upper_spread_in_contact_widths": (
                curve_window["contact_up_spread"]["median"] / contact_width
                if curve_window["contact_up_spread"]["median"] is not None
                and contact_width else None
            ),
            "late_envelope_in_contact_widths": (
                curve_window["contact_envelope"]["median"] / contact_width
                if curve_window["contact_envelope"]["median"] is not None
                and contact_width else None
            ),
            "late_shock_spread_median":
                curve_window["shock_spread"]["median"],
            "late_shock_spread_in_contact_widths": (
                curve_window["shock_spread"]["median"] / contact_width
                if curve_window["shock_spread"]["median"] is not None
                and contact_width else None
            ),
            "late_rho_relative_std_median":
                air_window["rho"]["relstd"]["median"],
            "late_u_relative_std_median":
                air_window["u"]["relstd"]["median"],
            "late_p_relative_std_median":
                air_window["p"]["relstd"]["median"],
            "late_contact_n_theta":
                curve_window["contact_low_spread"]["n_theta"],
            "late_contact_n_theta_complete":
                curve_window["contact_low_spread"]["n_theta_complete"],
            "late_shock_n_theta":
                curve_window["shock_spread"]["n_theta"],
            "late_shock_n_theta_complete":
                curve_window["shock_spread"]["n_theta_complete"],
            "late_air_n_theta": air_window["n_theta"],
            "late_air_n_theta_q_queries": air_window["n_theta_q_queries"],
            "late_air_rho_n_complete_queries":
                air_window["rho"]["n_complete_queries"],
            "contact_width": contact_width,
            "source_note": (
                "Contact/shock spreads come from curve_windows (one value per "
                "theta, independent of q and the air common layer). Air "
                "relative standard deviations come from air_windows and are "
                "reported separately as theta-q queries."
            ),
        }

    from collections import Counter
    payload["missing"] = missing
    payload["query_evidence_count"] = len(evidence)
    payload["completeness"] = {
        "expected_cases_per_group": {
            name: len(group["expected_case_ids"])
            for name, group in payload["groups"].items()
        },
        "failed_query_field_count": len(missing),
        "failed_query_field_by_group": {
            name: sum(1 for item in missing if item["group"] == name)
            for name in payload["groups"]
        },
        "failed_query_field_by_field": dict(Counter(
            item["field"] for item in missing
        )),
        "failed_query_field_by_reason": dict(Counter(
            item["reason"] for item in missing
        )),
        "failed_query_field_by_gap_kind": dict(Counter(
            item.get("gap_kind") for item in missing
            if item.get("gap_kind") is not None
        )),
        "frame_plan": {
            name: group["frame_plan_summary"]
            for name, group in payload["groups"].items()
        },
        "frame_plan_all_cases_verified": all(
            group["frame_plan_summary"]["n_all_frames_imported"]
            == group["frame_plan_summary"]["n_cases"]
            for group in payload["groups"].values()
        ),
        "complete": len(missing) == 0,
    }
    if missing:
        payload["assessment"]["status"] = "diagnostic_incomplete"
        payload["assessment"]["note"] = (
            "Missing/unsupported queries are listed; do not read the available "
            "subset as a complete similarity result. A repeated large interval "
            "is only a candidate explanation until frame_plan records confirm "
            "or exclude filtering."
        )

    json_path = output / "similarity_check.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8"
    )

    curve_path = output / "theta_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "role", "case_id", "theta", "shock_over_Rsep",
            "contact_low_over_Rsep", "contact_up_over_Rsep",
        ])
        for group_name, group in payload["groups"].items():
            for row in group["curves"]:
                for theta, shock, low, up in zip(
                    row["theta"], row["shock_over_Rsep"],
                    row["contact_low_over_Rsep"], row["contact_up_over_Rsep"],
                ):
                    writer.writerow([
                        group_name, row["case_id"], theta,
                        "" if shock is None else shock,
                        "" if low is None else low,
                        "" if up is None else up,
                    ])

    curve_window_path = output / "curve_windows.csv"
    curve_metrics = (
        "contact_low_spread", "contact_up_spread",
        "contact_envelope", "shock_spread",
    )
    with curve_window_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "role", "window", "metric", "n_theta", "n_theta_complete",
            "median", "p90", "max",
        ])
        for group_name, group in payload["groups"].items():
            for window, entry in group["curve_windows"].items():
                for metric in curve_metrics:
                    values = entry[metric]
                    writer.writerow([
                        group_name, window, metric, values["n_theta"],
                        values["n_theta_complete"], values["median"],
                        values["p90"], values["max"],
                    ])

    field_path = output / "field_spread.csv"
    with field_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "role", "theta", "q", "r_over_Rsep", "common_width",
            "contact_low_spread", "contact_up_spread", "contact_envelope",
            "shock_spread", "field", "complete", "n_valid",
            "mean", "std", "relstd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, group in payload["groups"].items():
            for row in group["field_rows"]:
                for field in FIELDS:
                    detail = row["fields"][field]
                    writer.writerow({
                        "role": group_name,
                        "theta": row["theta"],
                        "q": row["q"],
                        "r_over_Rsep": row["r_over_Rsep"],
                        "common_width": row["common_width"],
                        "contact_low_spread": row["contact_low_spread"],
                        "contact_up_spread": row["contact_up_spread"],
                        "contact_envelope": row["contact_envelope"],
                        "shock_spread": row["shock_spread"],
                        "field": field,
                        "complete": detail["complete"],
                        "n_valid": detail.get("n_valid"),
                        "mean": detail.get("mean"),
                        "std": detail.get("std"),
                        "relstd": detail.get("relstd"),
                    })

    missing_path = output / "missing.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group", "theta", "q", "case_id", "field", "reason",
                "gap_kind",
            ],
        )
        writer.writeheader()
        for row in missing:
            writer.writerow(row)

    evidence_path = output / "query_evidence.csv"
    evidence_fields = [
        "group", "theta", "q", "kind", "case_id", "field", "supported",
        "reason", "r_over_Rsep", "frame_id_lower", "frame_id_upper",
        "lower_theta", "upper_theta", "gap", "gap_limit", "gap_kind",
        "gap_kind_is_candidate", "gap_repeat_count", "gap_repeat_run_max",
        "radius_min", "radius_max", "bracket_lower_radius",
        "bracket_upper_radius", "n_valid", "lower_valid", "upper_valid",
        "failure_reasons",
        "lower_frame_frame_id", "lower_frame_theta",
        "lower_frame_supported", "lower_frame_reason",
        "lower_frame_radius_min", "lower_frame_radius_max",
        "lower_frame_bracket_lower_radius",
        "lower_frame_bracket_upper_radius", "lower_frame_n_valid",
        "lower_frame_lower_valid", "lower_frame_upper_valid",
        "upper_frame_frame_id", "upper_frame_theta",
        "upper_frame_supported", "upper_frame_reason",
        "upper_frame_radius_min", "upper_frame_radius_max",
        "upper_frame_bracket_lower_radius",
        "upper_frame_bracket_upper_radius", "upper_frame_n_valid",
        "upper_frame_lower_valid", "upper_frame_upper_valid",
    ]
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=evidence_fields)
        writer.writeheader()
        for row in evidence:
            record = {}
            for key in evidence_fields:
                value = row.get(key)
                if isinstance(value, (list, tuple)):
                    value = "|".join(str(item) for item in value)
                record[key] = value
            writer.writerow(record)

    print(json.dumps({
        "wrote": str(json_path),
        "diagnostic_status": payload["assessment"]["status"],
        "complete": payload["completeness"]["complete"],
        "failed_query_field_count": len(missing),
        "failed_query_field_by_group": payload["completeness"]["failed_query_field_by_group"],
        "failed_query_field_by_field": payload["completeness"]["failed_query_field_by_field"],
        "failed_query_field_by_reason": payload["completeness"]["failed_query_field_by_reason"],
        "failed_query_field_by_gap_kind": payload["completeness"]["failed_query_field_by_gap_kind"],
        "frame_plan_all_cases_verified":
            payload["completeness"]["frame_plan_all_cases_verified"],
        "frame_plan": payload["completeness"]["frame_plan"],
        "train_indicators": payload["groups"]["train"]["indicators"],
        "validation_indicators": payload["groups"]["validation"]["indicators"],
    }, indent=2, default=_jsonable))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
