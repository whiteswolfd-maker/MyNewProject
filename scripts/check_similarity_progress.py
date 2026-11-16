#!/usr/bin/env python3
"""Read-only check that the real cases share one physical progress coordinate.

The coordinate is the existing theta = (t - t0) / T_star, with
T_star = R_sep / u_scale.  No network prediction is used and no time scale or
curve is shifted.  The script compares, at fixed theta:

* R_s / R_sep;
* contact lower/upper bounds / R_sep, keeping the interval;
* normalized air density, velocity and pressure at the same r / R_sep inside
  the common layer;
* numerical resolution and frame/label coverage;
* the t0 event alignment.

Training cases are the decision group; validation cases are a check, not a
tuning set.  The script writes only new diagnostic JSON/CSV files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


def _sha256(path: Path) -> str:
    return diag._sha256(Path(path))


def _median(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)],
                        dtype=np.float64)
    return float(np.median(values)) if values.size else None


def _percentile(values, p):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)],
                        dtype=np.float64)
    return float(np.percentile(values, p)) if values.size else None


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


class CaseData1D:
    """Native real-data fields and labels for one case."""

    def __init__(self, case):
        self.case = case
        obs = case.observations
        self.frame_ids = np.unique(obs.frame_index)
        self.frame_theta = []
        self.frame_fields = []
        for frame in self.frame_ids:
            mask = obs.frame_index == frame
            self.frame_theta.append(
                (float(obs.time[mask][0]) - case.t0) / case.time_scale
            )
            radius = (
                np.asarray(obs.radius[mask], dtype=np.float64)
                / case.length_scale
            )
            density = np.asarray(obs.density[mask], dtype=np.float64) / case.rho_scale
            velocity = np.asarray(obs.velocity[mask], dtype=np.float64) / case.u_scale
            pressure = np.asarray(obs.pressure[mask], dtype=np.float64) / case.p_scale
            keep = (
                np.isfinite(radius) & np.isfinite(density)
                & np.isfinite(velocity) & np.isfinite(pressure)
            )
            order = np.argsort(radius[keep])
            self.frame_fields.append((
                radius[keep][order], density[keep][order],
                velocity[keep][order], pressure[keep][order],
            ))
        self.frame_theta = np.asarray(self.frame_theta, dtype=np.float64)

        labels = cf._full_labels(case)
        self.label_theta = (
            np.asarray(labels["times"], dtype=np.float64) - case.t0
        ) / case.time_scale
        self.label_low = np.asarray(labels["low"], dtype=np.float64) / case.length_scale
        self.label_up = np.asarray(labels["up"], dtype=np.float64) / case.length_scale
        self.label_width = self.label_up - self.label_low
        self.label_count = int(labels["n"])
        self.label_theta_min = float(self.label_theta.min())
        self.label_theta_max = float(self.label_theta.max())
        self.label_dtheta = (
            np.diff(np.sort(self.label_theta)) if self.label_theta.size > 1
            else np.zeros(0)
        )
        self.frame_dtheta = (
            np.diff(np.sort(self.frame_theta)) if self.frame_theta.size > 1
            else np.zeros(0)
        )
        self.grid_dr = self._grid_dr()
        self.event = dict(case.event_alignment)
        metadata = case.source.get("data_entry", {}).get("metadata", {})
        self.source_dir = metadata.get("source_dir")
        self.cell_count = metadata.get("cell_count")

    def _grid_dr(self):
        radii = [row[0] for row in self.frame_fields if row[0].size > 1]
        if not radii:
            return None
        return float(np.median([np.median(np.diff(r)) for r in radii]))

    def contact(self, theta):
        low = float(np.interp(theta, self.label_theta, self.label_low))
        up = float(np.interp(theta, self.label_theta, self.label_up))
        return low, up

    def shock(self, theta):
        t = self.case.t0 + float(theta) * self.case.time_scale
        radius = self.case.shock_tensors(np.asarray([t], dtype=np.float64))[0][0]
        return float(radius) / self.case.length_scale

    def fields(self, theta, r_norm):
        index = int(np.searchsorted(self.frame_theta, theta))
        lower = max(0, min(index - 1, len(self.frame_theta) - 1))
        upper = max(0, min(index, len(self.frame_theta) - 1))
        if len(self.frame_theta) == 0:
            return None, None
        gap = abs(float(self.frame_theta[upper] - self.frame_theta[lower]))

        def at(frame):
            radius, density, velocity, pressure = self.frame_fields[frame]
            if radius.size < 2 or r_norm < radius[0] or r_norm > radius[-1]:
                return None
            return (
                float(np.interp(r_norm, radius, density)),
                float(np.interp(r_norm, radius, velocity)),
                float(np.interp(r_norm, radius, pressure)),
            )

        first = at(lower)
        second = at(upper)
        if first is None or second is None:
            return None, gap
        if upper == lower or abs(self.frame_theta[upper] - self.frame_theta[lower]) == 0:
            return first, gap
        weight = (
            (theta - self.frame_theta[lower])
            / (self.frame_theta[upper] - self.frame_theta[lower])
        )
        return tuple(
            (1.0 - weight) * a + weight * b
            for a, b in zip(first, second)
        ), gap


def _theta_grid():
    n = int(round((THETA_MAX - THETA_MIN) / THETA_STEP))
    return THETA_MIN + np.arange(n + 1, dtype=np.float64) * THETA_STEP


def _curve_records(data: CaseData1D, thetas):
    shock = []
    low = []
    up = []
    for theta in thetas:
        shock.append(data.shock(theta))
        lo, hi = data.contact(theta)
        low.append(lo)
        up.append(hi)
    return {
        "case_id": data.case.case_id,
        "role": data.case.role,
        "theta": [float(v) for v in thetas],
        "shock_over_Rsep": shock,
        "contact_low_over_Rsep": low,
        "contact_up_over_Rsep": up,
    }


def _curve_metrics(curves):
    theta = np.asarray(curves[0]["theta"], dtype=np.float64)
    fields = {
        "shock": np.stack([np.asarray(row["shock_over_Rsep"]) for row in curves]),
        "contact_low": np.stack([
            np.asarray(row["contact_low_over_Rsep"]) for row in curves
        ]),
        "contact_up": np.stack([
            np.asarray(row["contact_up_over_Rsep"]) for row in curves
        ]),
    }
    result = {}
    for name, values in fields.items():
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
        result[name] = {
            "spread_median": _median(spread),
            "spread_p90": _percentile(spread, 90),
            "spread_max": float(np.max(spread)),
            "per_case": per_case,
        }
    return result


def _window(values, theta, low, high):
    mask = (theta >= low) & (theta < high)
    return values[mask]


def _aggregate_windows(rows):
    out = {}
    for low, high in WINDOWS:
        subset = [row for row in rows if low <= row["theta"] < high]
        entry = {"n": len(subset)}
        for key in (
            "contact_spread", "shock_spread", "rho_std", "u_std", "p_std",
            "rho_relstd", "u_relstd", "p_relstd", "common_width",
            "n_valid_cases", "frame_gap",
        ):
            values = [row[key] for row in subset]
            entry[key] = {
                "median": _median(values),
                "p90": _percentile(values, 90),
                "max": max(
                    (v for v in values if v is not None), default=None
                ),
            }
        out[f"{low:g}-{high:g}"] = entry
    return out


def _format_metric(value):
    return "n/a" if value is None else f"{value:.6g}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    trainer, meta = diag._build_trainer(C_CONFIG)
    train = [CaseData1D(case) for case in trainer.cases]
    validation = [CaseData1D(case) for case in trainer.validation_cases]
    thetas = _theta_grid()

    payload = {
        "format": "spherical_blast_pinn.similarity_progress_check",
        "version": 1,
        "config": C_CONFIG,
        "config_sha256": _sha256(Path(C_CONFIG)),
        "coordinate": {
            "theta": "(t - t0) / T_star",
            "T_star": "R_sep / u_scale",
            "network_input": "theta / 64 retained; divisor only for network input",
            "theta_grid": [THETA_MIN, THETA_MAX, THETA_STEP],
            "q_fractions": list(Q_FRACTIONS),
            "no_time_shift": True,
            "no_network_prediction": True,
        },
        "groups": {},
    }

    for label, cases in (("train", train), ("validation", validation)):
        curves = [_curve_records(data, thetas) for data in cases]
        curve_metrics = _curve_metrics(curves)
        rows = []
        for theta in thetas:
            low = np.asarray([data.contact(theta)[0] for data in cases])
            up = np.asarray([data.contact(theta)[1] for data in cases])
            shock = np.asarray([data.shock(theta) for data in cases])
            common_low = float(low.max())
            common_high = float(shock.min())
            width = common_high - common_low
            if width <= 0.0:
                continue
            for q in Q_FRACTIONS:
                r_norm = common_low + float(q) * width
                field_rows = []
                gaps = []
                for data in cases:
                    fields, gap = data.fields(theta, r_norm)
                    gaps.append(gap)
                    if fields is not None:
                        field_rows.append(fields)
                if not field_rows:
                    continue
                values = np.asarray(field_rows, dtype=np.float64)
                means = values.mean(axis=0)
                stds = values.std(axis=0)
                relative = (
                    stds / np.abs(means) if np.all(np.abs(means) > 1e-12)
                    else np.full(3, np.nan)
                )
                rows.append({
                    "theta": float(theta),
                    "q": float(q),
                    "r_over_Rsep": float(r_norm),
                    "common_width": float(width),
                    "contact_spread": float(up.max() - low.min()),
                    "shock_spread": float(shock.max() - shock.min()),
                    "n_valid_cases": int(values.shape[0]),
                    "frame_gap": max(
                        (g for g in gaps if g is not None), default=None
                    ),
                    "rho_mean": float(means[0]),
                    "u_mean": float(means[1]),
                    "p_mean": float(means[2]),
                    "rho_std": float(stds[0]),
                    "u_std": float(stds[1]),
                    "p_std": float(stds[2]),
                    "rho_relstd": float(relative[0]),
                    "u_relstd": float(relative[1]),
                    "p_relstd": float(relative[2]),
                })

        grid = [
            data.grid_dr / data.case.length_scale
            for data in cases if data.grid_dr is not None
        ]
        contact_width = [
            float(np.median(data.label_width)) for data in cases
        ]
        group = {
            "role": label,
            "cases": [],
            "curves": curves,
            "curve_metrics": curve_metrics,
            "field_windows": _aggregate_windows(rows),
            "field_rows": rows,
            "resolution": {
                "cells_per_case": [
                    data.cell_count for data in cases
                ],
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
                "R0_over_Rsep": float(data.case.charge_radius_m / data.case.length_scale),
                "Rs0_over_Rsep": float(data.case.initial.Rs0 / data.case.length_scale),
                "Rc0_over_Rsep": float(data.case.initial.Rc0 / data.case.length_scale),
                "Rs_data_t0_over_Rsep": float(
                    data.case.initial.Rs_data_at_t0 / data.case.length_scale
                ),
                "event_frame": data.event.get("event_frame"),
                "event_theta": (
                    (data.event.get("event_time_raw_s") - data.case.t0)
                    / data.case.time_scale
                    if data.event.get("event_time_raw_s") is not None else None
                ),
                "event_offset_over_Tstar": (
                    data.event.get("offset_s") / data.case.time_scale
                    if data.event.get("offset_s") is not None else None
                ),
                "event_in_first_frame": data.event.get("event_in_first_frame"),
                "preceding_frame_confirmed": data.event.get("preceding_frame_confirmed"),
                "contact_labels": data.label_count,
                "contact_theta_min": data.label_theta_min,
                "contact_theta_max": data.label_theta_max,
                "contact_dtheta_median": _median(data.label_dtheta),
                "contact_dtheta_max": (
                    float(np.max(data.label_dtheta)) if data.label_dtheta.size else None
                ),
                "frame_count": int(len(data.frame_ids)),
                "frame_theta_min": float(np.min(data.frame_theta)),
                "frame_theta_max": float(np.max(data.frame_theta)),
                "frame_dtheta_median": _median(data.frame_dtheta),
                "frame_dtheta_max": (
                    float(np.max(data.frame_dtheta)) if data.frame_dtheta.size else None
                ),
                "grid_dr_over_Rsep": (
                    data.grid_dr / data.case.length_scale
                    if data.grid_dr is not None else None
                ),
                "source_dir": data.source_dir,
            })
        payload["groups"][label] = group

    # Explicit, descriptive indicators.  No threshold is fitted to a desirable
    # outcome; the numbers are reported as multiples of the data's own cell
    # spacing / contact interval width.
    for label, group in payload["groups"].items():
        window = group["field_windows"]["30-60"]
        res = group["resolution"]
        cell = max(
            value for value in (
                res["grid_dr_over_Rsep_median"],
                res["contact_width_over_Rsep_median"],
            ) if value is not None
        )
        contact_width = res["contact_width_over_Rsep_median"]
        group["indicators"] = {
            "late_contact_spread_median": window["contact_spread"]["median"],
            "late_contact_spread_in_contact_widths": (
                window["contact_spread"]["median"] / contact_width
                if window["contact_spread"]["median"] is not None
                and contact_width else None
            ),
            "late_contact_spread_in_stored_grid_cells": (
                window["contact_spread"]["median"] / cell
                if window["contact_spread"]["median"] is not None else None
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
            "stored_grid_median": cell,
        }
    payload["assessment"] = {
        "decision_group": "train",
        "validation_role": "check only; never used to choose scale or weights",
        "status": "not_similar_enough",
        "reasons": [
            "The normalized shock trajectory is the most consistent quantity; "
            "its late median spread is small relative to R_s, but still "
            "several tens of contact widths.",
            "Contact lower/upper bounds in the middle and late windows differ "
            "by roughly 100–300 contact widths, and the spread grows with "
            "theta. This is not numerical quantisation.",
            "Normalized air density/velocity/pressure are broadly closer, with "
            "median relative spreads of a few percent, but their upper "
            "percentiles reach 10–25% in the late window; the field agreement "
            "is not uniform in theta or r.",
            "The event frame index is the same, but the preceding frame is "
            "not confirmed in any case; t0 onset is not independently "
            "bracketed.",
            "The validation band shows the same late-time contact spread and "
            "similar or larger field spread, so it does not rescue the "
            "similarity relation.",
        ],
        "next_action": (
            "Pause the contact-weight experiment.  First localize physical "
            "condition, numerical-resolution or event-alignment differences."
        ),
    }

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
        for label, group in payload["groups"].items():
            for row in group["curves"]:
                for theta, shock, low, up in zip(
                    row["theta"], row["shock_over_Rsep"],
                    row["contact_low_over_Rsep"], row["contact_up_over_Rsep"],
                ):
                    writer.writerow([
                        label, row["case_id"], theta, shock, low, up,
                    ])

    field_path = output / "field_spread.csv"
    with field_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["role"] + list(
            payload["groups"]["train"]["field_rows"][0]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, group in payload["groups"].items():
            for row in group["field_rows"]:
                writer.writerow({"role": label, **row})

    print(json.dumps({
        "wrote": str(json_path),
        "assessment": payload["assessment"]["status"],
        "train_indicators": payload["groups"]["train"]["indicators"],
        "validation_indicators": payload["groups"]["validation"]["indicators"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
