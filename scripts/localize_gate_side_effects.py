#!/usr/bin/env python3
"""Localise the C/D gate side effects on the existing last checkpoints.

Read-only analysis.  Energy residuals use one fixed, data-derived (t, r)
reference set and one common [Rc, Rs] interior mask for both models.  Interface
pressure is evaluated at each model's own predicted contact surface, so those
points are not shared physical points.

Stages are fractions of each case's aligned interval [t0, t_end]:
[0,1%), [1%,5%), [5%,10%), [10%,100%].
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
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402
import evaluate_contact_lr_resume as base  # noqa: E402
import evaluate_gate_pair as pair  # noqa: E402
import physical_residuals as phys  # noqa: E402


STAGES = (
    ("0_1pct", 0.0, 0.01),
    ("1_5pct", 0.01, 0.05),
    ("5_10pct", 0.05, 0.10),
    ("10_100pct", 0.10, 1.0),
)
DEFAULT_OUTPUT = (
    "outputs/fullframes/time_input_experiment/gate_pair/side_effects_last"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_arrays(items) -> str:
    digest = hashlib.sha256()
    for case_id, arrays in items:
        digest.update(case_id.encode("utf-8"))
        for name, array in arrays:
            value = np.ascontiguousarray(np.asarray(array))
            digest.update(name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def _stage_mask(fraction, stage):
    name, lo, hi = stage
    if name == "10_100pct":
        return fraction >= lo
    return (fraction >= lo) & (fraction < hi)


def _ss(values):
    if values.size == 0:
        return 0.0
    return float(np.sum(np.square(values, dtype=np.float64)))


def _mean_or_none(total, n):
    return None if n == 0 else total / n


def _rms_or_none(total, n):
    return None if n == 0 else math.sqrt(total / n)


def _delta_stats(delta):
    """Statistics of the *increment* delta_i = r_D_i^2 - r_C_i^2.

    ``top_*_all`` uses the top fraction of all points in the group, including
    non-positive deltas; the number of positive points inside that top set and
    the all-point count are reported so the denominator is unambiguous.
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = int(delta.size)
    positive = delta[delta > 0.0]
    negative = delta[delta < 0.0]
    positive_sum = float(positive.sum())
    negative_sum = float(negative.sum())
    net_sum = positive_sum + negative_sum
    out = {
        "n_points": n,
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "n_zero": int(np.count_nonzero(delta == 0.0)),
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
        "net_sum": net_sum,
        "gross_absolute_sum": positive_sum + abs(negative_sum),
        "positive_share_of_gross": (
            None if positive_sum + abs(negative_sum) == 0.0
            else positive_sum / (positive_sum + abs(negative_sum))
        ),
        "fraction_points_D_worse": (
            None if n == 0 else float(np.mean(delta > 0.0))
        ),
        "fraction_points_D_better": (
            None if n == 0 else float(np.mean(delta < 0.0))
        ),
    }
    for label, fraction in (
        ("0_1pct", 0.001), ("1pct", 0.01), ("5pct", 0.05),
    ):
        if n == 0:
            out[f"top_{label}_all_n"] = 0
            out[f"top_{label}_all_positive_points"] = 0
            out[f"top_{label}_all_positive_sum"] = 0.0
            out[f"top_{label}_all_share_of_positive"] = None
            out[f"top_{label}_positive_n"] = 0
            out[f"top_{label}_positive_share"] = None
            continue
        k_all = max(1, int(math.ceil(fraction * n)))
        ordered = np.sort(np.maximum(delta, 0.0))[::-1]
        top = ordered[:k_all]
        out[f"top_{label}_all_n"] = k_all
        out[f"top_{label}_all_positive_points"] = int(
            np.count_nonzero(top > 0.0)
        )
        out[f"top_{label}_all_positive_sum"] = float(top.sum())
        out[f"top_{label}_all_share_of_positive"] = (
            None if positive_sum == 0.0
            else float(top.sum() / positive_sum)
        )
        n_positive = int(positive.size)
        if n_positive:
            k_positive = max(1, int(math.ceil(fraction * n_positive)))
            top_positive = np.sort(positive)[::-1][:k_positive]
            out[f"top_{label}_positive_n"] = k_positive
            out[f"top_{label}_positive_share"] = (
                None if positive_sum == 0.0
                else float(top_positive.sum() / positive_sum)
            )
        else:
            out[f"top_{label}_positive_n"] = 0
            out[f"top_{label}_positive_share"] = None
    return out


def _top_stage_composition(delta, stage_index, stage_names):
    """Which stages supply the largest positive increments."""
    delta = np.asarray(delta, dtype=np.float64)
    stage_index = np.asarray(stage_index, dtype=np.int64)
    out = []
    for label, fraction in (
        ("0_1pct", 0.001), ("1pct", 0.01), ("5pct", 0.05),
    ):
        n = int(delta.size)
        k = max(1, int(math.ceil(fraction * n))) if n else 0
        if k == 0:
            out.append({"label": label, "n_all": 0, "k": 0})
            continue
        index = np.argsort(delta)[-k:]
        counts = np.bincount(
            stage_index[index], minlength=len(stage_names)
        )
        out.append({
            "label": label,
            "n_all": n,
            "k": k,
            "positive_points": int(np.count_nonzero(delta[index] > 0.0)),
            "stage_counts": {
                str(name): int(counts[i])
                for i, name in enumerate(stage_names)
            },
            "positive_sum": float(
                np.maximum(delta[index], 0.0).sum()
            ),
        })
    return out


def _time_structure(case, times, radii, delta, metric, *,
                    contact_radius_c=None, contact_radius_d=None):
    """Unique-time structure of the [0,1%) points for one case."""
    times = np.asarray(times, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if times.size == 0:
        return {
            "case_id": case.case_id,
            "role": case.role,
            "charge_radius_m": float(case.charge_radius_m),
            "metric": metric,
            "stage": "0_1pct",
            "n_points": 0,
            "n_unique_times": 0,
            "points_per_time_min": None,
            "points_per_time_median": None,
            "points_per_time_max": None,
            "time_s_min": None,
            "time_s_max": None,
            "fraction_min": None,
            "fraction_max": None,
            "per_time": [],
        }
    unique, inverse, counts = np.unique(
        times, return_inverse=True, return_counts=True
    )
    fraction = (
        unique - float(case.t0)
    ) / (float(case.t_end) - float(case.t0))
    per_time = []
    for index, time in enumerate(unique):
        select = inverse == index
        row = {
            "case_id": case.case_id,
            "role": case.role,
            "metric": metric,
            "stage": "0_1pct",
            "time_s": float(time),
            "fraction": float(fraction[index]),
            "n_spatial": int(counts[index]),
            "delta_sum": float(delta[select].sum()),
            "delta_positive_sum": float(
                np.maximum(delta[select], 0.0).sum()
            ),
        }
        if radii is not None:
            radius = np.asarray(radii, dtype=np.float64)
            row["r_min"] = float(np.min(radius[select]))
            row["r_max"] = float(np.max(radius[select]))
        if contact_radius_c is not None:
            row["contact_radius_C_mean"] = float(
                np.mean(np.asarray(contact_radius_c)[select])
            )
            row["contact_radius_D_mean"] = float(
                np.mean(np.asarray(contact_radius_d)[select])
            )
        per_time.append(row)
    return {
        "case_id": case.case_id,
        "role": case.role,
        "charge_radius_m": float(case.charge_radius_m),
        "metric": metric,
        "stage": "0_1pct",
        "n_points": int(times.size),
        "n_unique_times": int(unique.size),
        "points_per_time_min": int(counts.min()),
        "points_per_time_median": float(np.median(counts)),
        "points_per_time_max": int(counts.max()),
        "time_s_min": float(unique.min()),
        "time_s_max": float(unique.max()),
        "fraction_min": float(fraction.min()),
        "fraction_max": float(fraction.max()),
        "per_time": per_time,
    }


def _domain(case, trainer, reference):
    t = torch.as_tensor(reference["arrays"][f"t_{case.case_id}"],
                        dtype=torch.float64)
    r = torch.as_tensor(reference["arrays"][f"r_{case.case_id}"],
                        dtype=torch.float64)
    slope = reference["arrays"][f"slope_{case.case_id}"]
    inside, rc, rc_dot, rs, ds, _, scales = phys._domain(
        case, trainer.contact_net, t, r
    )
    finite = (
        torch.isfinite(rc) & torch.isfinite(rc_dot)
        & torch.isfinite(rs) & torch.isfinite(ds)
    ).numpy()
    length = rs - rc
    positive = torch.isfinite(length) & (length > 0.0)
    zeta = torch.where(
        positive, (r - rc) / torch.where(
            positive, length, torch.ones_like(length)
        ), torch.zeros_like(length),
    )
    return {
        "t": t, "r": r, "slope": slope,
        "inside": inside.numpy(), "finite": finite,
        "zeta": zeta.numpy(), "rc": rc, "rc_dot": rc_dot,
        "rs": rs, "ds": ds, "scales": scales,
    }


def _stage_records(case, times, values_c, values_d, radius_c, radius_d,
                   extra_radius_key):
    fraction = (
        np.asarray(times, dtype=np.float64) - float(case.t0)
    ) / (float(case.t_end) - float(case.t0))
    rows = []
    for stage in STAGES:
        mask = _stage_mask(fraction, stage)
        n = int(mask.sum())
        c = np.asarray(values_c, dtype=np.float64)[mask]
        d = np.asarray(values_d, dtype=np.float64)[mask]
        ss_c = _ss(c)
        ss_d = _ss(d)
        row = {
            "case_id": case.case_id,
            "role": case.role,
            "charge_radius_m": float(case.charge_radius_m),
            "stage": stage[0],
            "n": n,
            "ss_C": ss_c,
            "ss_D": ss_d,
            "mse_C": _mean_or_none(ss_c, n),
            "mse_D": _mean_or_none(ss_d, n),
            "rmse_C": _rms_or_none(ss_c, n),
            "rmse_D": _rms_or_none(ss_d, n),
            "diff_ss": ss_d - ss_c if n else None,
            "diff_mse": (
                (ss_d - ss_c) / n if n else None
            ),
            "mean_radius": (
                float(np.mean(extra_radius_key[mask])) if n else None
            ),
            "min_radius": (
                float(np.min(extra_radius_key[mask])) if n else None
            ),
            "max_radius": (
                float(np.max(extra_radius_key[mask])) if n else None
            ),
        }
        if radius_c is not None:
            row["mean_contact_radius_C"] = (
                float(np.mean(radius_c[mask])) if n else None
            )
            row["mean_contact_radius_D"] = (
                float(np.mean(radius_d[mask])) if n else None
            )
        rows.append(row)
    return rows


def _aggregate(rows, key):
    n = int(sum(row["n"] for row in rows if row.get("n")))
    ss_c = sum(row["ss_C"] for row in rows if row.get("n"))
    ss_d = sum(row["ss_D"] for row in rows if row.get("n"))
    return {
        "n": n,
        "ss_C": ss_c,
        "ss_D": ss_d,
        "mse_C": _mean_or_none(ss_c, n),
        "mse_D": _mean_or_none(ss_d, n),
        "rmse_C": _rms_or_none(ss_c, n),
        "rmse_D": _rms_or_none(ss_d, n),
        "diff_ss": ss_d - ss_c if n else None,
        "diff_mse": (ss_d - ss_c) / n if n else None,
        "n_cases": len({row["case_id"] for row in rows if row.get("n")}),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--points-per-case", type=int, default=20000)
    parser.add_argument("--interface-points", type=int, default=128)
    args = parser.parse_args(argv)
    output = _REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    c_trainer, _ = diag._build_trainer(pair.C_CONFIG)
    d_trainer, _ = diag._build_trainer(pair.D_CONFIG)
    c_trainer.load(pair.C_LAST)
    d_trainer.load(pair.D_LAST)
    cases = list(c_trainer.cases) + list(c_trainer.validation_cases)
    reference = base._reference_physical_points(
        c_trainer, cases, int(args.points_per_case)
    )

    energy_rows = []
    interface_rows = []
    case_records = {}
    early_time = {"energy": [], "interface": []}
    early_time_rows = []
    point_energy = {
        "case_index": [], "t": [], "r": [], "residual_C": [],
        "residual_D": [], "stage": [],
    }
    point_interface = {
        "case_index": [], "t": [], "contact_radius_C": [],
        "contact_radius_D": [], "residual_C": [], "residual_D": [],
        "stage": [],
    }
    case_energy_arrays = {}
    case_interface_arrays = {}
    all_energy_c = []
    all_energy_d = []
    all_interface_c = []
    all_interface_d = []
    stage_energy = {
        stage[0]: {
            split: {"C": [], "D": []} for split in ("train", "validation")
        }
        for stage in STAGES
    }
    stage_interface = {
        stage[0]: {
            split: {"C": [], "D": []} for split in ("train", "validation")
        }
        for stage in STAGES
    }
    identity_items = []
    stage_index = {stage[0]: i for i, stage in enumerate(STAGES)}
    for case_index, case in enumerate(cases):
        dc = _domain(case, c_trainer, reference)
        dd = _domain(case, d_trainer, reference)
        if not np.array_equal(dc["t"].numpy(), dd["t"].numpy()):
            raise RuntimeError(f"{case.case_id}: reference t differs")
        if not np.array_equal(dc["r"].numpy(), dd["r"].numpy()):
            raise RuntimeError(f"{case.case_id}: reference r differs")
        common = dc["slope"].copy()
        common &= dc["inside"]
        common &= dd["inside"]
        interior = common.copy()
        interior &= (dc["zeta"] > 1.0e-12) & (dc["zeta"] < 1.0 - 1.0e-12)
        interior &= (dd["zeta"] > 1.0e-12) & (dd["zeta"] < 1.0 - 1.0e-12)
        if not np.array_equal(
            common, dc["slope"] & dc["inside"] & dd["inside"]
        ):
            raise RuntimeError(f"{case.case_id}: common mask mismatch")
        keep = torch.as_tensor(interior, dtype=torch.bool)
        vc = phys._residual_values(
            case, c_trainer.air_net, dc["t"], dc["r"], dc["rc"], dc["rc_dot"],
            dc["rs"], dc["ds"], dc["scales"], keep,
        )
        vd = phys._residual_values(
            case, d_trainer.air_net, dd["t"], dd["r"], dd["rc"], dd["rc_dot"],
            dd["rs"], dd["ds"], dd["scales"], keep,
        )
        keep_np = interior
        both = (
            vc["finite"].numpy() & vd["finite"].numpy()
        )
        if not np.array_equal(both, vc["finite"].numpy()) or not np.array_equal(
            both, vd["finite"].numpy()
        ):
            raise RuntimeError(f"{case.case_id}: non-finite residual mismatch")
        t_keep = dc["t"].numpy()[keep_np]
        r_keep = dc["r"].numpy()[keep_np]
        energy_c = vc["energy"].detach().numpy()[both]
        energy_d = vd["energy"].detach().numpy()[both]
        case_records[case.case_id] = {
            "case_id": case.case_id,
            "role": case.role,
            "charge_radius_m": float(case.charge_radius_m),
            "n_reference": int(dc["t"].numel()),
            "n_common": int(common.sum()),
            "n_interior": int(interior.sum()),
            "n_energy_compared": int(both.sum()),
            "n_nonfinite_C": int(vc["n_nonfinite"]),
            "n_nonfinite_D": int(vd["n_nonfinite"]),
            "n_below_t0": int((t_keep < float(case.t0)).sum()),
        }
        identity_items.append((case.case_id, (
            ("t", dc["t"].numpy()), ("r", dc["r"].numpy()),
            ("common", common), ("interior", interior),
        )))
        e_rows = _stage_records(
            case, t_keep[both], energy_c, energy_d,
            None, None, r_keep[both],
        )
        energy_rows.extend(e_rows)
        energy_fraction = (
            t_keep[both] - float(case.t0)
        ) / (float(case.t_end) - float(case.t0))
        delta_energy = energy_d ** 2 - energy_c ** 2
        case_energy_arrays[case.case_id] = {
            "C": energy_c, "D": energy_d,
        }
        point_energy["case_index"].extend([case_index] * energy_c.size)
        point_energy["t"].extend(t_keep[both])
        point_energy["r"].extend(r_keep[both])
        point_energy["residual_C"].extend(energy_c)
        point_energy["residual_D"].extend(energy_d)
        stage_values = np.zeros(energy_c.size, dtype=np.int64)
        for stage in STAGES:
            mask = _stage_mask(energy_fraction, stage)
            stage_energy[stage[0]][case.role]["C"].append(energy_c[mask])
            stage_energy[stage[0]][case.role]["D"].append(energy_d[mask])
            stage_values[mask] = stage_index[stage[0]]
        point_energy["stage"].extend(stage_values)
        mask_early = _stage_mask(energy_fraction, STAGES[0])
        early_summary = _time_structure(
            case, t_keep[both][mask_early], r_keep[both][mask_early],
            delta_energy[mask_early], "energy",
        )
        early_time["energy"].append(early_summary)
        early_time_rows.extend(early_summary["per_time"])

        ivc = phys._interface_values(
            case, c_trainer.contact_net, c_trainer.air_net,
            points=int(args.interface_points),
        )
        ivd = phys._interface_values(
            case, d_trainer.contact_net, d_trainer.air_net,
            points=int(args.interface_points),
        )
        if not np.array_equal(ivc["t"].numpy(), ivd["t"].numpy()):
            raise RuntimeError(f"{case.case_id}: interface time grid differs")
        iface_finite = ivc["finite"].numpy() & ivd["finite"].numpy()
        if not np.array_equal(iface_finite, ivc["finite"].numpy()) or not \
                np.array_equal(iface_finite, ivd["finite"].numpy()):
            raise RuntimeError(f"{case.case_id}: non-finite interface mismatch")
        t_iface = ivc["t"].numpy()[iface_finite]
        iface_c = ivc["pressure"].numpy()[iface_finite]
        iface_d = ivd["pressure"].numpy()[iface_finite]
        i_rows = _stage_records(
            case, t_iface, iface_c, iface_d,
            ivc["contact_radius"].numpy()[iface_finite],
            ivd["contact_radius"].numpy()[iface_finite],
            np.zeros_like(t_iface),
        )
        # Radius columns for interface are contact radii, not physical sample r.
        for row in i_rows:
            row.pop("mean_radius", None)
            row.pop("min_radius", None)
            row.pop("max_radius", None)
        interface_rows.extend(i_rows)
        iface_fraction = (
            t_iface - float(case.t0)
        ) / (float(case.t_end) - float(case.t0))
        delta_interface = iface_d ** 2 - iface_c ** 2
        case_interface_arrays[case.case_id] = {
            "C": iface_c, "D": iface_d,
        }
        point_interface["case_index"].extend(
            [case_index] * iface_c.size
        )
        point_interface["t"].extend(t_iface)
        point_interface["contact_radius_C"].extend(
            ivc["contact_radius"].numpy()[iface_finite]
        )
        point_interface["contact_radius_D"].extend(
            ivd["contact_radius"].numpy()[iface_finite]
        )
        point_interface["residual_C"].extend(iface_c)
        point_interface["residual_D"].extend(iface_d)
        iface_stage_values = np.zeros(iface_c.size, dtype=np.int64)
        for stage in STAGES:
            mask = _stage_mask(iface_fraction, stage)
            stage_interface[stage[0]][case.role]["C"].append(iface_c[mask])
            stage_interface[stage[0]][case.role]["D"].append(iface_d[mask])
            iface_stage_values[mask] = stage_index[stage[0]]
        point_interface["stage"].extend(iface_stage_values)
        iface_mask_early = _stage_mask(iface_fraction, STAGES[0])
        iface_early_summary = _time_structure(
            case, t_iface[iface_mask_early], None,
            delta_interface[iface_mask_early], "interface_pressure",
            contact_radius_c=ivc["contact_radius"].numpy()[iface_finite][
                iface_mask_early
            ],
            contact_radius_d=ivd["contact_radius"].numpy()[iface_finite][
                iface_mask_early
            ],
        )
        early_time["interface"].append(iface_early_summary)
        early_time_rows.extend(iface_early_summary["per_time"])

        all_energy_c.append(energy_c)
        all_energy_d.append(energy_d)
        all_interface_c.append(iface_c)
        all_interface_d.append(iface_d)

    ec = np.concatenate(all_energy_c) if all_energy_c else np.zeros(0)
    ed = np.concatenate(all_energy_d) if all_energy_d else np.zeros(0)
    ic = np.concatenate(all_interface_c) if all_interface_c else np.zeros(0)
    id_ = np.concatenate(all_interface_d) if all_interface_d else np.zeros(0)

    def stage_aggregates(rows):
        out = {}
        for stage in STAGES:
            subset = [row for row in rows if row["stage"] == stage[0]]
            out[stage[0]] = _aggregate(subset, stage[0])
        return out

    def split_stage_aggregates(rows):
        out = {}
        for split in ("train", "validation"):
            out[split] = stage_aggregates(
                [row for row in rows if row["role"] == split]
            )
        return out

    def case_aggregates(rows):
        out = {}
        for case in cases:
            subset = [row for row in rows if row["case_id"] == case.case_id]
            out[case.case_id] = {
                "role": case.role,
                "charge_radius_m": float(case.charge_radius_m),
                **_aggregate(subset, case.case_id),
            }
        return out

    def bucket_delta(bucket):
        c = np.concatenate(bucket["C"]) if bucket["C"] else np.zeros(0)
        d = np.concatenate(bucket["D"]) if bucket["D"] else np.zeros(0)
        return _delta_stats(d * d - c * c)

    def delta_by_stage(stage_values, split=None):
        out = {}
        for stage in STAGES:
            name = stage[0]
            bucket = (
                stage_values[name][split] if split is not None
                else {
                    key: [
                        item for s in ("train", "validation")
                        for item in stage_values[name][s][key]
                    ]
                    for key in ("C", "D")
                }
            )
            out[name] = bucket_delta(bucket)
        return out

    def delta_by_split(stage_values):
        out = {}
        for split in ("train", "validation"):
            out[split] = bucket_delta({
                "C": [
                    item for stage in STAGES
                    for item in stage_values[stage[0]][split]["C"]
                ],
                "D": [
                    item for stage in STAGES
                    for item in stage_values[stage[0]][split]["D"]
                ],
            })
        return out

    def delta_by_case(case_arrays):
        out = {}
        for case in cases:
            c = case_arrays[case.case_id]["C"]
            d = case_arrays[case.case_id]["D"]
            out[case.case_id] = {
                "role": case.role,
                "charge_radius_m": float(case.charge_radius_m),
                **_delta_stats(d * d - c * c),
            }
        return out

    energy_delta = {
        "overall": _delta_stats(ed * ed - ec * ec),
        "by_stage": delta_by_stage(stage_energy),
        "by_split": delta_by_split(stage_energy),
        "by_split_stage": {
            split: delta_by_stage(stage_energy, split)
            for split in ("train", "validation")
        },
        "by_case": delta_by_case(case_energy_arrays),
        "top_stage_composition": _top_stage_composition(
            ed * ed - ec * ec,
            np.asarray(point_energy["stage"], dtype=np.int64),
            [stage[0] for stage in STAGES],
        ),
    }
    interface_delta = {
        "overall": _delta_stats(id_ * id_ - ic * ic),
        "by_stage": delta_by_stage(stage_interface),
        "by_split": delta_by_split(stage_interface),
        "by_split_stage": {
            split: delta_by_stage(stage_interface, split)
            for split in ("train", "validation")
        },
        "by_case": delta_by_case(case_interface_arrays),
        "top_stage_composition": _top_stage_composition(
            id_ * id_ - ic * ic,
            np.asarray(point_interface["stage"], dtype=np.int64),
            [stage[0] for stage in STAGES],
        ),
    }

    energy = {
        "method": (
            "normalised mapped Euler energy residual at one fixed data-derived "
            "(t,r) reference per case; one common [Rc,Rs] interior mask for C "
            "and D.  Pooled by sums of squared residuals; RMS = sqrt(ss/n), "
            "never a sum of RMS values."
        ),
        "stages": stage_aggregates(energy_rows),
        "by_split": split_stage_aggregates(energy_rows),
        "by_case": case_aggregates(energy_rows),
        "rows": energy_rows,
        "delta": energy_delta,
        "early_0_1pct_time_structure": early_time["energy"],
    }
    interface = {
        "method": (
            "interface pressure residual at each model's own predicted contact "
            "surface R_c(t), zeta=0, on a common time grid.  These are not the "
            "same physical radius for C and D.  Pooled by sums of squares."
        ),
        "stages": stage_aggregates(interface_rows),
        "by_split": split_stage_aggregates(interface_rows),
        "by_case": case_aggregates(interface_rows),
        "rows": interface_rows,
        "delta": interface_delta,
        "early_0_1pct_time_structure": early_time["interface"],
    }
    metrics_path = (
        _REPO_ROOT
        / "outputs/fullframes/time_input_experiment/gate_pair/evaluation/metrics.json"
    )
    consistency = {"metrics_path": str(metrics_path), "checks": []}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            energy_agg = _aggregate(
                [row for row in energy_rows if row["role"] == split], split
            )
            interface_agg = _aggregate(
                [row for row in interface_rows if row["role"] == split], split
            )
            for model in ("C", "D"):
                expected_energy = existing["sets"]["last"]["pde"][split][
                    "aggregate"
                ][model]["energy"]
                expected_pressure = existing["sets"]["last"]["interface"][
                    model
                ][split]["pressure_rms"]
                for kind, ours, expected in (
                    ("energy", energy_agg[f"rmse_{model}"], expected_energy),
                    (
                        "interface_pressure",
                        interface_agg[f"rmse_{model}"],
                        expected_pressure,
                    ),
                ):
                    if ours is None or expected is None:
                        match = ours is None and expected is None
                        rel = None
                    else:
                        rel = abs(ours - expected) / max(abs(expected), 1e-30)
                        match = rel <= 1.0e-9
                    consistency["checks"].append({
                        "split": split,
                        "model": model,
                        "quantity": kind,
                        "ours": ours,
                        "existing_metrics": expected,
                        "relative_difference": rel,
                        "match": match,
                    })
        consistency["all_match"] = all(
            item["match"] for item in consistency["checks"]
        )
        if not consistency["all_match"]:
            raise RuntimeError(
                "localisation does not reproduce existing last metrics: "
                f"{[item for item in consistency['checks'] if not item['match']]}"
            )
    else:
        consistency["all_match"] = None

    case_ids = np.asarray([case.case_id for case in cases])
    stage_names = np.asarray([stage[0] for stage in STAGES])
    energy_points_path = output / "energy_points.npz"
    np.savez_compressed(
        energy_points_path,
        case_ids=case_ids,
        case_index=np.asarray(point_energy["case_index"], dtype=np.int64),
        t_s=np.asarray(point_energy["t"], dtype=np.float64),
        r_m=np.asarray(point_energy["r"], dtype=np.float64),
        residual_C=np.asarray(point_energy["residual_C"], dtype=np.float64),
        residual_D=np.asarray(point_energy["residual_D"], dtype=np.float64),
        stage_index=np.asarray(point_energy["stage"], dtype=np.int64),
        stage_names=stage_names,
    )
    interface_points_path = output / "interface_points.npz"
    np.savez_compressed(
        interface_points_path,
        case_ids=case_ids,
        case_index=np.asarray(point_interface["case_index"], dtype=np.int64),
        t_s=np.asarray(point_interface["t"], dtype=np.float64),
        contact_radius_C=np.asarray(
            point_interface["contact_radius_C"], dtype=np.float64
        ),
        contact_radius_D=np.asarray(
            point_interface["contact_radius_D"], dtype=np.float64
        ),
        residual_C=np.asarray(
            point_interface["residual_C"], dtype=np.float64
        ),
        residual_D=np.asarray(
            point_interface["residual_D"], dtype=np.float64
        ),
        stage_index=np.asarray(point_interface["stage"], dtype=np.int64),
        stage_names=stage_names,
    )
    early_time_path = output / "early_0_1pct_time_structure.csv"
    if early_time_rows:
        keys = []
        for row in early_time_rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with early_time_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=keys, restval=""
            )
            writer.writeheader()
            writer.writerows(early_time_rows)

    payload = {
        "experiment": "C_none_vs_D_similarity_gate_last_4000_side_effects",
        "checkpoints": {
            "C_last": {"path": str(pair.C_LAST), "sha256": _sha256(pair.C_LAST),
                       "loop": int(c_trainer.loop)},
            "D_last": {"path": str(pair.D_LAST), "sha256": _sha256(pair.D_LAST),
                       "loop": int(d_trainer.loop)},
        },
        "reference": {
            "points_per_case_limit": int(args.points_per_case),
            "identity_sha256": _hash_arrays(identity_items),
            "arrays_identical_for_C_and_D": True,
            "common_interior_identical_for_C_and_D": True,
            "case_records": case_records,
        },
        "energy": energy,
        "interface_pressure": interface,
        "point_files": {
            "energy": {
                "path": str(energy_points_path),
                "sha256": _sha256(energy_points_path),
                "n_points": int(len(point_energy["t"])),
            },
            "interface": {
                "path": str(interface_points_path),
                "sha256": _sha256(interface_points_path),
                "n_points": int(len(point_interface["t"])),
            },
        },
        "early_0_1pct_time_structure_csv": {
            "path": str(early_time_path),
            "sha256": (
                _sha256(early_time_path) if early_time_path.exists() else None
            ),
            "n_rows": len(early_time_rows),
        },
        "consistency_with_existing_metrics": consistency,
        "notes": [
            "Energy uses one fixed physical reference and a common domain; "
            "interface pressure uses each model's own contact surface.",
            "The common interior excludes zeta <= 1e-12 or >= 1-1e-12 for "
            "either model, matching the existing PDE evaluation.",
            "Stage boundaries are [0,1%), [1%,5%), [5%,10%), [10%,100%] of "
            "each case's [t0,t_end].",
        ],
    }
    (output / "localization.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    for name, rows in (("energy", energy_rows), ("interface", interface_rows)):
        path = output / f"{name}_by_case_stage.csv"
        keys = list(rows[0].keys()) if rows else ["case_id"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    _write_report(output, payload)
    print(json.dumps({
        "output": str(output),
        "localization_sha256": _sha256(output / "localization.json"),
        "energy_delta": energy_delta["overall"],
        "interface_delta": interface_delta["overall"],
        "point_files": {
            "energy": payload["point_files"]["energy"],
            "interface": payload["point_files"]["interface"],
        },
    }, indent=2))
    return 0


def _fmt(value, digits=5):
    return "n/a" if value is None else f"{value:.{digits}g}"


def _write_report(output: Path, payload: dict):
    lines = ["# C/D 门控副作用定位（last 4000 对 last 4000）", ""]
    lines.append(
        "只用已有 C_last / D_last，不训练、不调参。能量残差固定在同一个"
        "数据导出的物理 (t,r) 参考集和共同域上；界面压力在各自模型的"
        "预测接触面上，二者不是同一物理点。"
    )
    lines.append(
        "阶段按每个算例的 [t0,t_end] 归一化时间划分："
        "[0,1%)、[1%,5%)、[5%,10%)、[10%,100%]。"
    )
    lines.append(
        "门控在该归一化时间上的典型值约为：1% 处 0.473、5% 处 0.959、"
        "10% 处 0.998；时间重合只用于定位，不证明因果。"
    )
    lines.append("")

    for kind, title in (
        ("energy", "能量方程"),
        ("interface_pressure", "界面压力相容误差"),
    ):
        block = payload[kind]
        lines.append(f"## {title}")
        lines.append("### 按 stage（train+validation）")
        lines.append("| stage | n | C ss | D ss | C RMS | D RMS | diff ss |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for stage, row in block["stages"].items():
            lines.append(
                f"| {stage} | {row['n']} | {_fmt(row['ss_C'])} | "
                f"{_fmt(row['ss_D'])} | {_fmt(row['rmse_C'])} | "
                f"{_fmt(row['rmse_D'])} | {_fmt(row['diff_ss'])} |"
            )
        lines.append("")
        lines.append("### 按 split × stage")
        lines.append("| split | stage | n | C ss | D ss | C RMS | D RMS | diff ss |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for split in ("train", "validation"):
            for stage, row in block["by_split"][split].items():
                lines.append(
                    f"| {split} | {stage} | {row['n']} | "
                    f"{_fmt(row['ss_C'])} | {_fmt(row['ss_D'])} | "
                    f"{_fmt(row['rmse_C'])} | {_fmt(row['rmse_D'])} | "
                    f"{_fmt(row['diff_ss'])} |"
                )
        lines.append("")
        lines.append("### 按算例（全部阶段合计，按 net delta 降序）")
        lines.append("| case | role | R0 | n | C ss | D ss | net delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        case_rows = sorted(
            block["by_case"].items(),
            key=lambda item: (
                item[1]["diff_ss"] if item[1]["diff_ss"] is not None
                else float("-inf")
            ),
            reverse=True,
        )
        for case_id, row in case_rows:
            lines.append(
                f"| {case_id} | {row['role']} | "
                f"{_fmt(row['charge_radius_m'])} | {row['n']} | "
                f"{_fmt(row['ss_C'])} | {_fmt(row['ss_D'])} | "
                f"{_fmt(row['diff_ss'])} |"
            )
        lines.append("")
        net_diff = sum(
            row["diff_ss"] for row in block["rows"]
            if row["diff_ss"] is not None
        )
        lines.append("### 最大算例 × stage（按 net delta 降序）")
        lines.append("| case | role | R0 | stage | n | net delta | 占净值 |")
        lines.append("|---|---|---:|---|---:|---:|---:|")
        top_rows = sorted(
            [row for row in block["rows"] if row["diff_ss"] is not None],
            key=lambda row: row["diff_ss"], reverse=True,
        )[:10]
        for row in top_rows:
            share = (
                None if net_diff == 0.0 else row["diff_ss"] / net_diff
            )
            lines.append(
                f"| {row['case_id']} | {row['role']} | "
                f"{_fmt(row['charge_radius_m'])} | {row['stage']} | "
                f"{row['n']} | {_fmt(row['diff_ss'])} | {_fmt(share)} |"
            )
        lines.append("")
        lines.append("### 恶化增量 delta_i 的正/负分解")
        lines.append(
            "delta_i = residual_D_i^2 - residual_C_i^2。"
            "`top 1%/5%` 的分母是该组全部点数；表中同时给出实际 k 和"
            "其中正增量点数。"
        )
        lines.append(
            "| group | n | positive n | positive sum | negative sum | net | "
            "top1% k | top1% share | top5% k | top5% share |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        groups = [("overall", block["delta"]["overall"])]
        groups.extend(
            (f"split={split}", block["delta"]["by_split"][split])
            for split in ("train", "validation")
        )
        groups.extend(
            (f"stage={stage}", block["delta"]["by_stage"][stage])
            for stage in block["delta"]["by_stage"]
        )
        for label, row in groups:
            lines.append(
                f"| {label} | {row['n_points']} | {row['n_positive']} | "
                f"{_fmt(row['positive_sum'])} | {_fmt(row['negative_sum'])} | "
                f"{_fmt(row['net_sum'])} | {row['top_1pct_all_n']} | "
                f"{_fmt(row['top_1pct_all_share_of_positive'])} | "
                f"{row['top_5pct_all_n']} | "
                f"{_fmt(row['top_5pct_all_share_of_positive'])} |"
            )
        lines.append("")
        lines.append("top 点来自哪些 stage：")
        lines.append("| top | k | positive points | stage counts |")
        lines.append("|---|---:|---:|---|")
        for item in block["delta"]["top_stage_composition"]:
            counts = ", ".join(
                f"{name}:{count}"
                for name, count in item.get("stage_counts", {}).items()
            )
            lines.append(
                f"| {item['label']} | {item['k']} | "
                f"{item.get('positive_points', 0)} | {counts} |"
            )
        lines.append("")
        lines.append("### [0,1%) 的时间结构")
        lines.append(
            "`n_unique_times` 是独立时间位置数；`points/time` 是同一时间上的"
            "空间点数。完整时间位置和逐时间空间点数在 "
            "`early_0_1pct_time_structure.csv`。"
        )
        lines.append("| case | role | R0 | n points | unique times | points/time min/median/max | t min | t max |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|")
        for item in block["early_0_1pct_time_structure"]:
            lines.append(
                f"| {item['case_id']} | {item['role']} | "
                f"{_fmt(item['charge_radius_m'])} | {item['n_points']} | "
                f"{item['n_unique_times']} | "
                f"{item['points_per_time_min']}/"
                f"{_fmt(item['points_per_time_median'])}/"
                f"{item['points_per_time_max']} | "
                f"{_fmt(item['time_s_min'])} | {_fmt(item['time_s_max'])} |"
            )
        lines.append("")

    lines.append("## 与旧评估的一致性")
    checks = payload["consistency_with_existing_metrics"].get("checks", [])
    if checks:
        lines.append(
            f"- 用同一 last 权重复算的 split 级能量/界面压力 RMS 与旧 "
            f"`metrics.json` 共 {len(checks)} 项全部匹配："
            f"`{payload['consistency_with_existing_metrics']['all_match']}`。"
        )
    lines.append(
        "- 因此本轮副作用定位没有依赖新的训练或新的评估口径；"
        "旧报告和检查点未被覆盖。"
    )
    lines.append("")
    lines.append("## 结论边界")
    lines.append(
        "- 定位到早期不等于证明门控变化速度是原因；门控值与时间重合只是线索。"
    )
    lines.append(
        "- 多数点变差不等于排除了少数点主导增量；上表的 delta 正/负分解和"
        "top 1%/5% 正增量占比就是用来区分这两件事。"
    )
    lines.append(
        "- 不使用 D 自身平方和的集中度代替恶化增量集中度。"
    )
    lines.append("")
    lines.append("## 聚合口径")
    lines.append(
        "- 每个 stage 的 ss 是该 stage 所有点归一化残差平方和；"
        "MSE=ss/n，RMS=sqrt(ss/n)。"
    )
    lines.append(
        "- 跨算例/跨 stage 的合计只加平方和，不加 RMS；"
        "CSV 中保留每个算例每个 stage 的原始 ss。"
    )
    lines.append(
        "- 能量参考点和共同域对 C/D 完全相同；界面点各自在自身 R_c(t)，"
        "不能当作同一物理点。"
    )
    lines.append(
        "- 固定参考点是观测行的确定性 20000 点/算例子采样，"
        "因此各 stage 的点数反映数据采样密度，跨 stage 的平方和不能"
        "直接当作时间区间长度权重。"
    )
    lines.append("")
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

if __name__ == "__main__":
    raise SystemExit(main())
