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


def _bracket(theta, times):
    """Bracket a query without endpoint extrapolation or unchecked gaps."""
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        return {
            "supported": False, "reason": "insufficient_points",
            "lower_index": None, "upper_index": None,
            "lower_theta": None, "upper_theta": None,
            "gap": None, "gap_limit": None,
        }
    if theta < times[0] or theta > times[-1]:
        return {
            "supported": False, "reason": "outside_range",
            "lower_index": None, "upper_index": None,
            "lower_theta": float(times[0]), "upper_theta": float(times[-1]),
            "gap": None, "gap_limit": None,
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
    if gap > gap_limit:
        return {
            "supported": False, "reason": "large_gap",
            "lower_index": int(lower), "upper_index": int(upper),
            "lower_theta": float(times[lower]),
            "upper_theta": float(times[upper]),
            "gap": gap, "gap_limit": gap_limit,
        }
    return {
        "supported": True, "reason": "ok",
        "lower_index": int(lower), "upper_index": int(upper),
        "lower_theta": float(times[lower]),
        "upper_theta": float(times[upper]),
        "gap": gap, "gap_limit": gap_limit,
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
    """One field at one frame; invalid points are removed per field."""
    radius = np.asarray(radius, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(radius) & np.isfinite(values)
    if int(keep.sum()) < 2:
        return {
            "supported": False, "reason": "insufficient_valid_points",
            "value": None,
        }
    use_radius = radius[keep]
    use_values = values[keep]
    order = np.argsort(use_radius)
    use_radius = use_radius[order]
    use_values = use_values[order]
    if r_norm < use_radius[0] or r_norm > use_radius[-1]:
        return {
            "supported": False, "reason": "outside_field_radius",
            "value": None,
        }
    return {
        "supported": True, "reason": "ok",
        "value": float(np.interp(r_norm, use_radius, use_values)),
        "n_valid": int(keep.sum()),
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
        self.frame_radius = [self.frame_radius[index] for index in order]
        self.frame_values = [self.frame_values[index] for index in order]
        self.frame_valid = [self.frame_valid[index] for index in order]
        self.frame_dtheta = (
            np.diff(np.sort(self.frame_theta))
            if self.frame_theta.size > 1 else np.zeros(0)
        )

        labels = cf._full_labels(case)
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
            if not first["supported"] or not second["supported"]:
                result[field] = {
                    "supported": False, "value": None,
                    "reason": (
                        first["reason"] if not first["supported"]
                        else second["reason"]
                    ),
                    "bracket": bracket,
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
                "n_valid_lower": first.get("n_valid"),
                "n_valid_upper": second.get("n_valid"),
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
    complete = np.all([
        np.all(np.isfinite(arrays[name]), axis=0) for name in names
    ], axis=0)
    result = {
        "n_theta_total": int(theta.size),
        "n_theta_complete": int(complete.sum()),
        "complete_theta_fraction": (
            float(complete.sum() / theta.size) if theta.size else None
        ),
    }
    if not complete.any():
        return result
    metrics = {}
    for name, values in arrays.items():
        values = values[:, complete]
        median = np.median(values, axis=0)
        spread = values.max(axis=0) - values.min(axis=0)
        per_case = {}
        for index, row in enumerate(curves):
            deviation = values[index] - median
            per_case[row["case_id"]] = {
                "rms": float(np.sqrt(np.mean(deviation * deviation))),
                "max_abs": float(np.max(np.abs(deviation))),
                "mean": float(np.mean(deviation)),
            }
        metrics[name] = {
            "spread_median": _median(spread),
            "spread_p90": _percentile(spread, 90),
            "spread_max": float(np.max(spread)),
            "per_case": per_case,
        }
    low = arrays["contact_low_over_Rsep"][:, complete]
    up = arrays["contact_up_over_Rsep"][:, complete]
    envelope = up.max(axis=0) - low.min(axis=0)
    result.update(metrics)
    result["contact_envelope"] = {
        "median": _median(envelope),
        "p90": _percentile(envelope, 90),
        "max": float(np.max(envelope)),
    }
    return result


def _window_aggregate(rows, key):
    values = [
        row[key] for row in rows
        if row[key] is not None and math.isfinite(row[key])
    ]
    return {
        "n": len(values),
        "median": _median(values),
        "p90": _percentile(values, 90),
        "max": max(values) if values else None,
    }


def _aggregate_windows(rows):
    out = {}
    for low, high in WINDOWS:
        subset = [
            row for row in rows
            if low <= row["theta"] < high and row["complete"]
        ]
        entry = {"n_complete": len(subset)}
        for key in (
            "contact_low_spread", "contact_up_spread", "contact_envelope",
            "shock_spread", "common_width",
            "rho_relstd", "u_relstd", "p_relstd",
            "rho_std", "u_std", "p_std",
        ):
            entry[key] = _window_aggregate(subset, key)
        out[f"{low:g}-{high:g}"] = entry
    return out


def _record_missing(missing, group, theta, q, case_id, field, reason):
    missing.append({
        "group": group,
        "theta": float(theta),
        "q": None if q is None else float(q),
        "case_id": case_id,
        "field": field,
        "reason": reason,
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

    payload = {
        "format": "spherical_blast_pinn.similarity_progress_check",
        "version": 2,
        "config": C_CONFIG,
        "config_sha256": _sha256(Path(C_CONFIG)),
        "coordinate": {
            "theta": "(t - t0) / T_star",
            "T_star": "R_sep / u_scale",
            "network_input": "theta / 64 retained",
            "theta_grid": [THETA_MIN, THETA_MAX, THETA_STEP],
            "q_fractions": list(Q_FRACTIONS),
            "gap_factor": GAP_FACTOR,
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
                if not s["supported"]:
                    _record_missing(
                        missing, group_name, theta, None,
                        data.case.case_id, "shock", s["reason"],
                    )
                for field, result in (("contact_low", c["low"]),
                                      ("contact_up", c["up"])):
                    if not result["supported"]:
                        _record_missing(
                            missing, group_name, theta, None,
                            data.case.case_id, field, result["reason"],
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
                missing_here = []
                for data in cases:
                    values = data.fields(theta, r_norm)
                    for field in FIELDS:
                        result = values[field]
                        if not result["supported"]:
                            missing_here.append((data.case.case_id, field))
                            _record_missing(
                                missing, group_name, theta, q,
                                data.case.case_id, field, result["reason"],
                            )
                        else:
                            field_values[field].append(result["value"])
                complete = (
                    not missing_here
                    and all(len(field_values[field]) == len(cases)
                            for field in FIELDS)
                )
                if not complete:
                    continue
                arrays = {
                    field: np.asarray(field_values[field], dtype=np.float64)
                    for field in FIELDS
                }
                means = {
                    field: float(arrays[field].mean()) for field in FIELDS
                }
                stds = {
                    field: float(arrays[field].std()) for field in FIELDS
                }
                relative = {
                    field: (
                        stds[field] / abs(means[field])
                        if abs(means[field]) > 1e-12 else None
                    )
                    for field in FIELDS
                }
                rows.append({
                    "theta": float(theta),
                    "q": float(q),
                    "r_over_Rsep": float(r_norm),
                    "common_width": float(width),
                    "contact_low_spread": float(low.max() - low.min()),
                    "contact_up_spread": float(up.max() - up.min()),
                    "contact_envelope": float(up.max() - low.min()),
                    "shock_spread": float(shock.max() - shock.min()),
                    "complete": True,
                    "rho_mean": means["rho"],
                    "u_mean": means["u"],
                    "p_mean": means["p"],
                    "rho_std": stds["rho"],
                    "u_std": stds["u"],
                    "p_std": stds["p"],
                    "rho_relstd": relative["rho"],
                    "u_relstd": relative["u"],
                    "p_relstd": relative["p"],
                })

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
            "field_windows": _aggregate_windows(rows),
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
            })
        payload["groups"][group_name] = group

    for group_name, group in payload["groups"].items():
        res = group["resolution"]
        contact_width = res["contact_width_over_Rsep_median"]
        window = group["field_windows"]["30-60"]
        group["indicators"] = {
            "late_lower_spread_median": window["contact_low_spread"]["median"],
            "late_upper_spread_median": window["contact_up_spread"]["median"],
            "late_envelope_width_median": window["contact_envelope"]["median"],
            "late_lower_spread_in_contact_widths": (
                window["contact_low_spread"]["median"] / contact_width
                if window["contact_low_spread"]["median"] is not None
                and contact_width else None
            ),
            "late_upper_spread_in_contact_widths": (
                window["contact_up_spread"]["median"] / contact_width
                if window["contact_up_spread"]["median"] is not None
                and contact_width else None
            ),
            "late_envelope_in_contact_widths": (
                window["contact_envelope"]["median"] / contact_width
                if window["contact_envelope"]["median"] is not None
                and contact_width else None
            ),
            "late_shock_spread_median": window["shock_spread"]["median"],
            "late_shock_spread_in_contact_widths": (
                window["shock_spread"]["median"] / contact_width
                if window["shock_spread"]["median"] is not None
                and contact_width else None
            ),
            "late_rho_relative_std_median": window["rho_relstd"]["median"],
            "late_u_relative_std_median": window["u_relstd"]["median"],
            "late_p_relative_std_median": window["p_relstd"]["median"],
            "contact_width": contact_width,
        }

    payload["missing"] = missing
    payload["completeness"] = {
        "expected_cases_per_group": {
            name: len(group["expected_case_ids"])
            for name, group in payload["groups"].items()
        },
        "missing_count": len(missing),
        "missing_by_group": {
            name: sum(1 for item in missing if item["group"] == name)
            for name in payload["groups"]
        },
        "complete": len(missing) == 0,
    }
    if missing:
        payload["assessment"]["status"] = "diagnostic_incomplete"
        payload["assessment"]["note"] = (
            "Missing/unsupported queries are listed; do not read the available "
            "subset as a complete similarity result."
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

    field_path = output / "field_spread.csv"
    with field_path.open("w", newline="", encoding="utf-8") as handle:
        sample = next(
            (group["field_rows"][0] for group in payload["groups"].values()
             if group["field_rows"]),
            None,
        )
        fieldnames = ["role"] + (list(sample) if sample else [])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, group in payload["groups"].items():
            for row in group["field_rows"]:
                writer.writerow({"role": group_name, **row})

    missing_path = output / "missing.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "theta", "q", "case_id", "field", "reason"],
        )
        writer.writeheader()
        for row in missing:
            writer.writerow(row)

    print(json.dumps({
        "wrote": str(json_path),
        "diagnostic_status": payload["assessment"]["status"],
        "complete": payload["completeness"]["complete"],
        "missing_count": len(missing),
        "missing_by_group": payload["completeness"]["missing_by_group"],
        "train_indicators": payload["groups"]["train"]["indicators"],
        "validation_indicators": payload["groups"]["validation"]["indicators"],
    }, indent=2, default=_jsonable))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
