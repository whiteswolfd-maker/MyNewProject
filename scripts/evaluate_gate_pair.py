#!/usr/bin/env python3
"""Evaluate the paired C (none) / D (similarity_time) gate experiment.

C and D have the same training budget but different AirNet input semantics, so
each checkpoint is loaded into its own trainer.  Last (both loop 4000) is the
primary comparison; best is reported separately.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_initial_error as ana  # noqa: E402
import diagnose_contact_gradients as diag  # noqa: E402
import evaluate_contact_lr_resume as base  # noqa: E402
import physical_residuals as phys  # noqa: E402
from spherical_blast_pinn.contact_air.losses import finite_mach_rh_state  # noqa: E402


C_CONFIG = (
    "configs/contact_air/"
    "split_train_r50_500_fullframes_similarity_none_pair.yaml"
)
D_CONFIG = (
    "configs/contact_air/"
    "split_train_r50_500_fullframes_similarity_gate_pair.yaml"
)
C_LAST = Path("outputs/contact_air_time_input_pair_c_none_4000/last.pt")
C_BEST = Path("outputs/contact_air_time_input_pair_c_none_4000/best.pt")
D_LAST = Path("outputs/contact_air_time_input_pair_d_gate_4000/last.pt")
D_BEST = Path("outputs/contact_air_time_input_pair_d_gate_4000/best.pt")
WINDOWS = {
    "0_1pct": (0.0, 0.01),
    "0_5pct": (0.0, 0.05),
    "0_10pct": (0.0, 0.10),
}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _aggregate_air_rows(rows, field, label, mask_key, scale_key):
    """Pool finite per-case MSEs without letting ``None`` into the sum.

    ``ana._metrics`` returns ``mse=None`` for a mask with no finite points.
    A case can also be partially non-finite: those points are dropped from that
    case's metric and counted, but the pooled value stays finite.  Completeness
    is reported separately and is false whenever a candidate point was dropped.
    """
    values = []
    n_candidates = 0
    n_nonfinite = 0
    skipped = []
    for row in rows:
        mask = row.get(mask_key)
        if mask is None:
            skipped.append({"case_id": row.get("case_id"), "reason": "no_mask"})
            continue
        metric = ana._metrics(
            row["pred"][field], row["target"][field], mask,
            row["scales"][scale_key],
        )
        n_candidates += int(metric.get("n_candidates", 0))
        n_nonfinite += int(metric.get("n_nonfinite", 0))
        if (
            metric.get("status") != "ok"
            or metric.get("n", 0) == 0
            or metric.get("mse") is None
            or not math.isfinite(float(metric["mse"]))
        ):
            skipped.append({
                "case_id": row.get("case_id"),
                "reason": metric.get("status", "no_metric"),
                "n_candidates": int(metric.get("n_candidates", 0)),
            })
            continue
        values.append(metric)
    n = sum(v["n"] for v in values)
    mse = sum(v["mse"] * v["n"] for v in values) / n if n else None
    return {
        "n": n,
        "n_candidates": n_candidates,
        "n_nonfinite": n_nonfinite,
        "n_cases": len(rows),
        "n_cases_used": len(values),
        "n_cases_skipped": len(skipped),
        "skipped_cases": skipped,
        "mse": mse,
        "rmse": math.sqrt(mse) if mse is not None else None,
        "status": "ok" if n else "not_evaluable",
        "complete": bool(
            n > 0 and n == n_candidates and not skipped
        ),
    }


def _failure_record(case, c, d):
    reasons = []
    if c.get("status") != "ok":
        reasons.append(f"C:{c.get('status', 'unknown')}")
    if d.get("status") != "ok":
        reasons.append(f"D:{d.get('status', 'unknown')}")
    return {
        "case_id": case.case_id,
        "role": case.role,
        "C_status": c.get("status", "unknown"),
        "D_status": d.get("status", "unknown"),
        "C_selected": int(c.get("selected", 0)),
        "D_selected": int(d.get("selected", 0)),
        "C_used": int(c.get("used", 0)),
        "D_used": int(d.get("used", 0)),
        "C_inside": 0,
        "D_inside": 0,
        "common": 0,
        "C_excluded": 0,
        "D_excluded": 0,
        "common_excluded": 0,
        "evaluated": False,
        "status": "unevaluable_case",
        "failure_reasons": reasons,
    }


def _early_pair(c_trainer, d_trainer, cases, window):
    lo, hi = WINDOWS[window]
    rows_c = {"train": [], "validation": []}
    rows_d = {"train": [], "validation": []}
    records = {"train": [], "validation": []}
    expected = {"train": [], "validation": []}
    for case in cases:
        split = case.role
        expected[split].append(case.case_id)
        c = ana._window_case(c_trainer, case, lo, hi)
        d = ana._window_case(d_trainer, case, lo, hi)
        if c.get("status") != "ok" or d.get("status") != "ok":
            records[split].append(_failure_record(case, c, d))
            continue
        assert np.array_equal(c["time"], d["time"])
        assert np.array_equal(c["radius"], d["radius"])
        common = np.asarray(c["inside"], bool) & np.asarray(d["inside"], bool)
        record = {
            "case_id": case.case_id,
            "role": split,
            "C_status": c["status"],
            "D_status": d["status"],
            "C_selected": int(c["selected"]),
            "D_selected": int(d["selected"]),
            "C_used": int(c["used"]),
            "D_used": int(d["used"]),
            "C_inside": int(np.asarray(c["inside"], bool).sum()),
            "D_inside": int(np.asarray(d["inside"], bool).sum()),
            "common": int(common.sum()),
            "C_excluded": int(c["used"] - np.asarray(c["inside"], bool).sum()),
            "D_excluded": int(d["used"] - np.asarray(d["inside"], bool).sum()),
            "common_excluded": int(c["used"] - common.sum()),
            "evaluated": False,
            "status": "ok",
            "failure_reasons": [],
        }
        if int(common.sum()) == 0:
            record["status"] = "empty_common_domain"
            record["failure_reasons"].append(
                "no physical points inside both models' [Rc,Rs] domains"
            )
            records[split].append(record)
            continue
        c["common_mask"] = common
        d["common_mask"] = common
        for row in (c, d):
            row["scales"] = {
                "rho": case.rho_scale, "u": case.u_scale, "p": case.p_scale,
            }
        rows_c[split].append(c)
        rows_d[split].append(d)
        record["evaluated"] = True
        records[split].append(record)

    def aggregate(rows, mask_key):
        return {
            field: _aggregate_air_rows(rows, field, "", mask_key, field)
            for field in ("rho", "u", "p")
        }

    result = {"window": window, "by_split": {}}
    for split in ("train", "validation"):
        c_own = aggregate(rows_c[split], "inside")
        d_own = aggregate(rows_d[split], "inside")
        c_common = aggregate(rows_c[split], "common_mask")
        d_common = aggregate(rows_d[split], "common_mask")
        split_records = records[split]
        evaluated_ids = [
            row["case_id"] for row in split_records if row["evaluated"]
        ]
        missing = [
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "failure_reasons": row["failure_reasons"],
            }
            for row in split_records if not row["evaluated"]
        ]
        expected_ids = expected[split]
        field_blocks = {
            "C_own": c_own, "D_own": d_own,
            "C_common": c_common, "D_common": d_common,
        }
        field_complete = all(
            block[field]["complete"]
            for block in field_blocks.values()
            for field in ("rho", "u", "p")
        )
        reasons = []
        if len(expected_ids) != 10:
            reasons.append(
                f"expected {len(expected_ids)} cases, tournament expects 10"
            )
        if len(split_records) != len(expected_ids):
            reasons.append(
                f"recorded {len(split_records)} cases, expected {len(expected_ids)}"
            )
        if set(evaluated_ids) != set(expected_ids):
            reasons.append("evaluated case ids differ from expected case ids")
        if missing:
            reasons.append(f"{len(missing)} case(s) not evaluable")
        if not field_complete:
            reasons.append(
                "at least one field has non-finite or skipped case points"
            )
        complete = not reasons
        result["by_split"][split] = {
            "C_own": c_own,
            "D_own": d_own,
            "C_common": c_common,
            "D_common": d_common,
            "expected_case_ids": expected_ids,
            "evaluated_case_ids": evaluated_ids,
            "missing_cases": missing,
            "unevaluable_cases": [
                item for item in missing
                if item["status"] == "unevaluable_case"
            ],
            "empty_common_domain_cases": [
                item for item in missing
                if item["status"] == "empty_common_domain"
            ],
            "case_records": split_records,
            "n_expected": len(expected_ids),
            "n_evaluated": len(evaluated_ids),
            "n_missing": len(missing),
            "complete": complete,
            "incomplete_reasons": reasons,
            "exclusions": [
                {
                    **row,
                    "selected_total": int(row.get("C_selected", 0)),
                    "used": int(row.get("C_used", 0)),
                }
                for row in split_records
            ],
            "aggregate_exclusions": {
                key: int(sum(row.get(key, 0) for row in split_records))
                for key in (
                    "C_selected", "D_selected", "C_used", "D_used",
                    "C_inside", "D_inside", "common", "C_excluded",
                    "D_excluded", "common_excluded",
                )
            },
        }
        result["by_split"][split]["aggregate_exclusions"].update({
            "selected_total": result["by_split"][split][
                "aggregate_exclusions"
            ]["C_selected"],
            "used": result["by_split"][split]["aggregate_exclusions"]["C_used"],
        })
    result["by_split"]["train"]["complete"] = (
        result["by_split"]["train"]["complete"]
        and len(result["by_split"]["train"]["expected_case_ids"]) == 10
    )
    result["by_split"]["validation"]["complete"] = (
        result["by_split"]["validation"]["complete"]
        and len(result["by_split"]["validation"]["expected_case_ids"]) == 10
    )
    result["complete"] = (
        result["by_split"]["train"]["complete"]
        and result["by_split"]["validation"]["complete"]
    )
    return result


def _right_boundary(trainer, case, points=128):
    t = torch.linspace(
        float(case.t0), float(case.t_end), int(points), dtype=torch.float64
    )
    r0 = torch.full_like(t, float(case.charge_radius_m))
    _, shock_speed = case.shock_tensors(t)
    c_inf = math.sqrt(
        float(case.gamma_air) * float(case.ambient_pressure)
        / float(case.ambient_density)
    )
    valid = torch.isfinite(shock_speed) & (shock_speed > c_inf)
    n_excluded = int((~valid).sum())
    t = t[valid]
    r0 = r0[valid]
    shock_speed = shock_speed[valid]
    if t.numel() == 0:
        return {
            "n": 0, "n_excluded": n_excluded, "status": "no_compression_times",
            "rho_rms": None, "u_rms": None, "p_rms": None,
        }
    state = finite_mach_rh_state(
        shock_speed.reshape(-1, 1),
        gamma=case.gamma_air,
        ambient_density=case.ambient_density,
        ambient_pressure=case.ambient_pressure,
    )
    zeta = torch.ones_like(t).reshape(-1, 1)
    with torch.no_grad():
        fields = trainer.air_net(
            zeta, t.reshape(-1, 1), r0.reshape(-1, 1),
            trainer._scales[case.case_id], time_input=case.time_input,
        )
    values = {}
    for name, pred, target, scale in (
        ("rho", fields.density, state["density"], case.rho_scale),
        ("u", fields.velocity, state["velocity"], case.u_scale),
        ("p", fields.pressure, state["pressure"], case.p_scale),
    ):
        error = (pred - target) / float(scale)
        values[f"{name}_rms"] = float(torch.sqrt(torch.mean(error * error)).detach())
    return {
        "n": int(t.numel()), "n_excluded": n_excluded,
        "status": "ok", **values,
    }


def _aggregate_right(rows):
    valid = [row for row in rows if row.get("n", 0) > 0]
    n = sum(row["n"] for row in valid)
    excluded = sum(row.get("n_excluded", 0) for row in rows)
    if not n:
        return {
            "n": 0, "n_excluded": excluded, "status": "not_evaluable",
            "rho_rms": None, "u_rms": None, "p_rms": None,
        }
    return {
        "n": n,
        "n_excluded": excluded,
        "status": "ok",
        "rho_rms": math.sqrt(sum(row["rho_rms"] ** 2 * row["n"] for row in valid) / n),
        "u_rms": math.sqrt(sum(row["u_rms"] ** 2 * row["n"] for row in valid) / n),
        "p_rms": math.sqrt(sum(row["p_rms"] ** 2 * row["n"] for row in valid) / n),
    }


def _t0_spread(trainer, cases):
    zeta = torch.as_tensor(np.linspace(0.01, 0.99, 33), dtype=torch.float64).reshape(-1, 1)
    normalized = {}
    for case in cases:
        t = torch.full_like(zeta, float(case.t0))
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        with torch.no_grad():
            fields = trainer.air_net(
                zeta, t, r0, trainer._scales[case.case_id],
                time_input=case.time_input,
            )
        normalized[case.case_id] = {
            "rho": (fields.density / case.rho_scale).reshape(-1).numpy(),
            "u": (fields.velocity / case.u_scale).reshape(-1).numpy(),
            "p": (fields.pressure / case.p_scale).reshape(-1).numpy(),
        }
    ref = normalized[cases[0].case_id]
    return {
        field: float(max(np.max(np.abs(v[field] - ref[field]))
                          for v in normalized.values()))
        for field in ("rho", "u", "p")
    }


def _aggregate_counts(rows, label):
    fields = (
        "candidates", "slope_invalid", "nonfinite", "outside",
        "layer_invalid", "common", "common_interior", "boundary_excluded",
    )
    return {
        field: int(sum(row["counts"][label][field] for row in rows))
        for field in fields
    }


def _physics_section(fixed, cases, labels):
    result = {}
    for split in ("train", "validation"):
        rows = [fixed[case.case_id] for case in cases if case.role == split]
        result[split] = {
            "aggregate": {
                label: base._aggregate_physics_rows(rows, label) for label in labels
            },
            "counts": {label: _aggregate_counts(rows, label) for label in labels},
        }
    return result


def _contact_section(trainer, cases):
    points = base._points(trainer, cases)
    train = base._reference_metrics(
        trainer, [c for c in cases if c.role == "train"],
        [p for p in points if p["case_id"] in {
            c.case_id for c in cases if c.role == "train"
        }],
    )
    validation = base._reference_metrics(
        trainer, [c for c in cases if c.role == "validation"],
        [p for p in points if p["case_id"] in {
            c.case_id for c in cases if c.role == "validation"
        }],
    )
    return {"train": train, "validation": validation}


def _t0_section(trainer, cases):
    rows = {
        case.case_id: base._initial_condition_metrics(trainer, case)
        for case in cases
    }
    return {
        "train": base._aggregate_initial(
            [rows[c.case_id] for c in cases if c.role == "train"]
        ),
        "validation": base._aggregate_initial(
            [rows[c.case_id] for c in cases if c.role == "validation"]
        ),
        "cases": rows,
    }


def _air_full(c_trainer, d_trainer, cases):
    pred_c = {case.case_id: base._air_predictions(c_trainer, case) for case in cases}
    pred_d = {case.case_id: base._air_predictions(d_trainer, case) for case in cases}
    per_case = {
        case.case_id: base._air_case_metrics(
            {"C": pred_c[case.case_id], "D": pred_d[case.case_id]},
            case, labels=("C", "D"), pair_labels=("C", "D"),
        )
        for case in cases
    }
    result = {}
    for split in ("train", "validation"):
        subset = {
            cid: item for cid, item in per_case.items()
            if next(c for c in cases if c.case_id == cid).role == split
        }
        result[split] = {
            "own": {
                field: {
                    label: base._aggregate_block(
                        subset, lambda item, f=field, l=label: item["own"][l][f]
                    )
                    for label in ("C", "D")
                }
                for field in ("rho", "u", "p")
            },
            "common": {
                field: {
                    label: base._aggregate_block(
                        subset, lambda item, f=field, l=label:
                        item["common_two"][f][l]
                    )
                    for label in ("C", "D")
                }
                for field in ("rho", "u", "p")
            },
            "counts": {
                label: {
                    key: int(sum(item["counts"][label][key] for item in subset.values()))
                    for key in ("n_total", "n_time_valid", "n_inside", "n_outside")
                }
                for label in ("C", "D")
            },
        }
    return result


def _contact_window_r350(trainer, case):
    indices = case.contact.indices()
    times = np.asarray(case.contact.times[indices], dtype=np.float64)
    low = np.asarray(case.contact.low[indices], dtype=np.float64)
    up = np.asarray(case.contact.up[indices], dtype=np.float64)
    masks = diag._third_masks(times, case.t0, case.t_end)
    return {
        str(i): {
            "status": "ok",
            **diag._contact_metrics(trainer, case, times[m], low[m], up[m]),
            "n_labels": int(m.sum()),
        }
        for i, m in enumerate(masks)
    }


def _interface(rows):
    return base._aggregate_interface(rows)


def _fmt(value, digits=5):
    return "n/a" if value is None else f"{value:.{digits}g}"


def _metrics_for_suffix(c_trainer, d_trainer, cases, ckpt_c, ckpt_d, suffix):
    c_trainer.load(ckpt_c)
    d_trainer.load(ckpt_d)
    c_t0 = _t0_section(c_trainer, cases)
    d_t0 = _t0_section(d_trainer, cases)
    c_contact = _contact_section(c_trainer, cases)
    d_contact = _contact_section(d_trainer, cases)
    c_air = _air_full(c_trainer, d_trainer, cases)
    c_interface = {
        case.case_id: phys._interface_errors(
            case, c_trainer.contact_net, c_trainer.air_net, points=128
        )
        for case in cases
    }
    d_interface = {
        case.case_id: phys._interface_errors(
            case, d_trainer.contact_net, d_trainer.air_net, points=128
        )
        for case in cases
    }
    c_right = {case.case_id: _right_boundary(c_trainer, case) for case in cases}
    d_right = {case.case_id: _right_boundary(d_trainer, case) for case in cases}

    reference = base._reference_physical_points(c_trainer, cases, 20000)
    fixed = base._fixed_physical_physics(
        c_trainer, cases, {"C": ckpt_c, "D": ckpt_d}, reference,
        labels=("C", "D"), trainers={"C": c_trainer, "D": d_trainer},
    )
    early = {}
    for window in ("0_1pct", "0_5pct", "0_10pct"):
        # Load the matching checkpoints again before each raw-window pass.
        c_trainer.load(ckpt_c)
        d_trainer.load(ckpt_d)
        early[window] = _early_pair(c_trainer, d_trainer, cases, window)

    r350 = next(c for c in cases if c.case_id == "r350_full")
    c_trainer.load(ckpt_c)
    d_trainer.load(ckpt_d)
    return {
        "loop_C": int(c_trainer.loop),
        "loop_D": int(d_trainer.loop),
        "t0": {"C": c_t0, "D": d_t0},
        "contact": {"C": c_contact, "D": d_contact},
        "air_full": c_air,
        "early": early,
        "interface": {
            "C": {
                split: _interface([
                    c_interface[c.case_id] for c in cases if c.role == split
                ]) for split in ("train", "validation")
            },
            "D": {
                split: _interface([
                    d_interface[c.case_id] for c in cases if c.role == split
                ]) for split in ("train", "validation")
            },
        },
        "right": {
            "C": {
                split: _aggregate_right([
                    c_right[c.case_id] for c in cases if c.role == split
                ]) for split in ("train", "validation")
            },
            "D": {
                split: _aggregate_right([
                    d_right[c.case_id] for c in cases if c.role == split
                ]) for split in ("train", "validation")
            },
        },
        "pde": _physics_section(fixed, cases, ("C", "D")),
        "r350": {
            "C": _contact_window_r350(c_trainer, r350),
            "D": _contact_window_r350(d_trainer, r350),
        },
        "t0_spread_construction": {
            "C": _t0_spread(c_trainer, cases),
            "D": _t0_spread(d_trainer, cases),
        },
    }


def _report(metrics, output: Path):
    lines = ["# C/D 门控配对实验", ""]
    lines.append(
        "C = similarity + `r0_input_gate=none`；D = similarity + "
        "`r0_input_gate=similarity_time`。last 为同为 4000 步的主比较；"
        "best 另表。C/D 从同初始化、同 seed、同预算、全新 Adam 训练；"
        "只改变 AirNet 门控。"
    )
    lines.append("")
    for suffix in ("last", "best"):
        block = metrics["sets"][suffix]
        lines.append(f"## {suffix}")
        lines.append(f"C loop {block['loop_C']} / D loop {block['loop_D']}")
        lines.append("")
        lines.append("### t0 理论 IC（归一化 RMSE）")
        lines.append("| 模型 | split | Rc | rho | u | p |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for model in ("C", "D"):
            for split in ("train", "validation"):
                row = block["t0"][model][split]
                lines.append(
                    f"| {model} | {split} | {_fmt(row.get('initial_rc_rmse'))} | "
                    f"{_fmt(row.get('rho_rmse'))} | {_fmt(row.get('u_rmse'))} | "
                    f"{_fmt(row.get('p_rmse'))} |"
                )
        lines.append("")
        lines.append("### 早期窗口完整性")
        lines.append("| 窗口 | split | expected | evaluated | missing | empty common | complete |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for window in ("0_1pct", "0_5pct", "0_10pct"):
            for split in ("train", "validation"):
                row = block["early"][window]["by_split"][split]
                lines.append(
                    f"| {window} | {split} | {row['n_expected']} | "
                    f"{row['n_evaluated']} | {row['n_missing']} | "
                    f"{len(row['empty_common_domain_cases'])} | "
                    f"{'yes' if row['complete'] else 'NO'} |"
                )
        incomplete = [
            (window, split, row["incomplete_reasons"])
            for window in ("0_1pct", "0_5pct", "0_10pct")
            for split in ("train", "validation")
            for row in (block["early"][window]["by_split"][split],)
            if not row["complete"]
        ]
        if incomplete:
            for window, split, reasons in incomplete:
                lines.append(
                    f"- {window}/{split} incomplete: {'; '.join(reasons)}"
                )
        else:
            lines.append(
                "- 所有窗口均为 10 个训练算例 + 10 个验证算例，"
                "无整例缺失或空共同域。"
            )
        lines.append("")
        lines.append("### 早期实际窗口：自身有效点 RMSE")
        lines.append("| 窗口 | split | C rho/u/p | D rho/u/p |")
        lines.append("|---|---|---|---|")
        for window in ("0_1pct", "0_5pct", "0_10pct"):
            for split in ("train", "validation"):
                row = block["early"][window]["by_split"][split]
                def joined(side):
                    return "/".join(
                        _fmt(row[f"{side}_own"][field]["rmse"])
                        for field in ("rho", "u", "p")
                    )
                lines.append(
                    f"| {window} | {split} | {joined('C')} | {joined('D')} |"
                )
        lines.append("")
        lines.append("### 早期实际窗口：共同物理点 RMSE")
        lines.append("| 窗口 | split | C rho/u/p | D rho/u/p |")
        lines.append("|---|---|---|---|")
        for window in ("0_1pct", "0_5pct", "0_10pct"):
            for split in ("train", "validation"):
                row = block["early"][window]["by_split"][split]
                def joined_common(side):
                    return "/".join(
                        _fmt(row[f"{side}_common"][field]["rmse"])
                        for field in ("rho", "u", "p")
                    )
                lines.append(
                    f"| {window} | {split} | {joined_common('C')} | "
                    f"{joined_common('D')} |"
                )
        lines.append("")
        lines.append("### 早期窗口排除数（汇总）")
        lines.append("| 窗口 | split | selected | used | C inside | D inside | common |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for window in ("0_1pct", "0_5pct", "0_10pct"):
            for split in ("train", "validation"):
                row = block["early"][window]["by_split"][split]["aggregate_exclusions"]
                lines.append(
                    f"| {window} | {split} | {row['selected_total']} | "
                    f"{row['used']} | {row['C_inside']} | {row['D_inside']} | "
                    f"{row['common']} |"
                )
        lines.append("")
        lines.append("### 全程空气场（自身/共同物理点）")
        lines.append("| split | 场 | C own | C common | D own | D common |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for split in ("train", "validation"):
            for field in ("rho", "u", "p"):
                block_row = block["air_full"][split]
                lines.append(
                    f"| {split} | {field} | "
                    f"{_fmt(block_row['own'][field]['C'].get('pooled_rmse'))} | "
                    f"{_fmt(block_row['common'][field]['C'].get('pooled_rmse'))} | "
                    f"{_fmt(block_row['own'][field]['D'].get('pooled_rmse'))} | "
                    f"{_fmt(block_row['common'][field]['D'].get('pooled_rmse'))} |"
                )
        lines.append("")
        lines.append("### 全程空气域计数")
        lines.append("| split | 模型 | n_total | n_time_valid | n_inside | n_outside |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for split in ("train", "validation"):
            for model in ("C", "D"):
                counts = block["air_full"][split]["counts"][model]
                lines.append(
                    f"| {split} | {model} | {counts['n_total']} | "
                    f"{counts['n_time_valid']} | {counts['n_inside']} | "
                    f"{counts['n_outside']} |"
                )
        lines.append("")
        lines.append("### 接触（macro/pooled MSE, mean violation, coverage）")
        lines.append("| 模型 | split | macro MSE | macro RMSE | pooled MSE | pooled RMSE | mean viol | covered/n |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for model in ("C", "D"):
            for split in ("train", "validation"):
                row = block["contact"][model][split]
                lines.append(
                    f"| {model} | {split} | {_fmt(row.get('macro_mse'))} | "
                    f"{_fmt(row.get('macro_rmse'))} | {_fmt(row.get('pooled_mse'))} | "
                    f"{_fmt(row.get('pooled_rmse'))} | "
                    f"{_fmt(row.get('mean_violation_over_Rsep_macro'))} | "
                    f"{row.get('covered_count')}/{row.get('n')} |"
                )
        lines.append("")
        lines.append("### 固定物理共同点 PDE（RMS）")
        lines.append("| 模型 | split | mass | momentum | energy | n |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for model in ("C", "D"):
            for split in ("train", "validation"):
                row = block["pde"][split]["aggregate"][model]
                lines.append(
                    f"| {model} | {split} | {_fmt(row.get('mass'))} | "
                    f"{_fmt(row.get('momentum'))} | {_fmt(row.get('energy'))} | "
                    f"{row.get('n')} |"
                )
        lines.append("")
        lines.append("### 界面与右边界")
        lines.append("| 模型 | split | interface u RMS | interface p RMS | right rho RMS | right u RMS | right p RMS |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for model in ("C", "D"):
            for split in ("train", "validation"):
                irow = block["interface"][model][split]
                rrow = block["right"][model][split]
                lines.append(
                    f"| {model} | {split} | {_fmt(irow.get('velocity_rms'))} | "
                    f"{_fmt(irow.get('pressure_rms'))} | {_fmt(rrow.get('rho_rms'))} | "
                    f"{_fmt(rrow.get('u_rms'))} | {_fmt(rrow.get('p_rms'))} |"
                )
        lines.append("")
        lines.append("### R350 接触三分段（MSE/RMSE）")
        lines.append("| 模型 | third | MSE | RMSE | covered/n |")
        lines.append("|---|---:|---:|---:|---:|")
        for model in ("C", "D"):
            for third in ("0", "1", "2"):
                row = block["r350"][model][third]
                lines.append(
                    f"| {model} | {third} | {_fmt(row.get('mse'))} | "
                    f"{_fmt(row.get('rmse'))} | {row.get('covered_count')}/{row.get('n')} |"
                )
        lines.append("")
        lines.append("### t0 spread 构造检查（不是精度）")
        for model in ("C", "D"):
            spread = block["t0_spread_construction"][model]
            lines.append(
                f"- {model}: rho/u/p = {_fmt(spread['rho'])} / "
                f"{_fmt(spread['u'])} / {_fmt(spread['p'])}"
            )
        lines.append("")
    lines.append("## 说明")
    lines.append(
        "- 早期窗口的 own/common 逐项和排除数在 `metrics.json`；"
        "报告不把 t0 spread=0 当成精度成功。"
    )
    lines.append(
        "- 归一化误差除以全局 rho/u/p/length scale，不是相对局部真值的百分比。"
    )
    lines.append(
        "- R350 中晚段单独报告，不用本实验宣称全部误差已解决。"
    )
    lines.append(
        "- best 的 loop 可能不同，不能当同一步比较；last 都是 4000。"
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    c_trainer, _ = diag._build_trainer(C_CONFIG)
    d_trainer, _ = diag._build_trainer(D_CONFIG)
    cases = list(c_trainer.cases) + list(c_trainer.validation_cases)

    metrics = {
        "configs": {"C": C_CONFIG, "D": D_CONFIG},
        "checkpoints": {
            "C_last": {"path": str(C_LAST), "sha256": _sha256(C_LAST)},
            "C_best": {"path": str(C_BEST), "sha256": _sha256(C_BEST)},
            "D_last": {"path": str(D_LAST), "sha256": _sha256(D_LAST)},
            "D_best": {"path": str(D_BEST), "sha256": _sha256(D_BEST)},
        },
        "sets": {},
        "notes": [
            "C/D are fresh paired runs from the same initial state, seed and budget.",
            "last is the primary comparison at loop 4000; best is separate.",
            "The existing B is only an external reference, not this pair.",
        ],
    }
    metrics["sets"]["last"] = _metrics_for_suffix(
        c_trainer, d_trainer, cases, C_LAST, D_LAST, "last"
    )
    metrics["sets"]["best"] = _metrics_for_suffix(
        c_trainer, d_trainer, cases, C_BEST, D_BEST, "best"
    )
    (output / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2), encoding="utf-8"
    )
    _report(metrics, output)
    print(json.dumps({
        "wrote": str(output / "metrics.json"),
        "sha256": _sha256(output / "metrics.json"),
        "last_loops": {
            "C": metrics["sets"]["last"]["loop_C"],
            "D": metrics["sets"]["last"]["loop_D"],
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
