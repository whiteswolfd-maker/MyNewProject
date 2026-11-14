#!/usr/bin/env python3
"""Summarise the read-only contact-training diagnostics (metric schema v3).

Reads JSON produced by ``diagnose_contact_gradients.py`` and writes
``summary.json``, ``summary.md`` and ``report.md``.  Macro (case-equal) and
pooled (frame-weighted) reference metrics are kept separate; coverage uses
exact integer hit counts; empty directions and failed runs never enter
medians or improvement rates.

If any source diagnostic failed or is incomplete, the summary is still written
but the command returns a non-zero exit code and the report carries a banner.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _median_valid(values) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return statistics.median(finite) if finite else None


def _change_stats(values, *, greater_is_better: bool) -> dict:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return {
            "n_evaluable": 0, "n_missing": len(values),
            "improved": None, "unchanged": None, "worsened": None,
            "median": None, "improved_fraction": None,
        }
    if greater_is_better:
        improved = sum(1 for value in finite if value > 0)
        worsened = sum(1 for value in finite if value < 0)
    else:
        improved = sum(1 for value in finite if value < 0)
        worsened = sum(1 for value in finite if value > 0)
    unchanged = len(finite) - improved - worsened
    return {
        "n_evaluable": len(finite),
        "n_missing": len(values) - len(finite),
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
        "median": statistics.median(finite),
        "improved_fraction": improved / len(finite),
    }


def _cosine_stats(batches: list[dict], key: str) -> dict:
    entries = [
        batch.get("aggregate_gradients", {}).get(key, {})
        for batch in batches
    ]
    statuses = [entry.get("status") for entry in entries]
    values = [
        float(entry["cosine"]) for entry in entries
        if entry.get("status") == "ok" and _finite(entry.get("cosine"))
    ]
    return {
        "requested": len(entries),
        "valid_directions": len(values),
        "undefined": statuses.count("direction_undefined"),
        "non_finite": statuses.count("non_finite"),
        "median": statistics.median(values) if values else None,
        "negative": sum(1 for value in values if value < 0),
        "negative_fraction": (
            sum(1 for value in values if value < 0) / len(values)
            if values else None
        ),
    }


def _term_stats(batches: list[dict], name: str) -> dict:
    losses = [
        batch["terms"][name]["raw_loss"] for batch in batches
        if _finite(batch["terms"][name].get("raw_loss"))
    ]
    norms = [
        batch["terms"][name]["grad_norm_weighted"] for batch in batches
        if _finite(batch["terms"][name].get("grad_norm_weighted"))
    ]
    return {
        "loss_median": _median_valid(losses),
        "grad_norm_median": _median_valid(norms),
        "n_loss": len(losses),
        "n_grad": len(norms),
    }


def _source_issues(payload: dict) -> list[str]:
    issues = []
    if payload.get("version") != 3 or payload.get("metric_version") != 3:
        issues.append("source schema is not metric version 3; rerun required")
    if payload.get("ok") is False:
        issues.extend(payload.get("failure_reasons") or ["source reported ok=false"])
    if payload.get("failed_batches", 0):
        issues.append(
            f"static failures {payload['failed_batches']}/"
            f"{payload.get('requested_batches')}"
        )
    if payload.get("completed_batches", 0) < payload.get("requested_batches", 0):
        issues.append(
            f"static incomplete {payload.get('completed_batches')}/"
            f"{payload.get('requested_batches')}"
        )
    if payload.get("non_finite"):
        issues.append("non-finite loss or gradient")
    if not payload.get("checkpoint_sha256", {}).get("unchanged", True):
        issues.append("checkpoint changed during the diagnostic")
    if payload.get("adam_requested", 0):
        if not payload.get("adam_available", False):
            issues.append("Adam requested but optimizer state is missing")
        if payload.get("adam_failed", 0):
            issues.append(
                f"Adam failures {payload['adam_failed']}/"
                f"{payload.get('adam_requested')}"
            )
        if payload.get("adam_completed", 0) < payload.get("adam_requested", 0):
            issues.append(
                f"Adam incomplete {payload.get('adam_completed')}/"
                f"{payload.get('adam_requested')}"
            )
    return issues


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("manifest.experiments must be a non-empty list")
    return payload


def _require_type(value, name: str, kind, issues: list[str]) -> bool:
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        issues.append(f"{name} has wrong type: {type(value).__name__}")
        return False
    return True


def _validate_static(payload: dict, expected: dict, issues: list[str]) -> dict:
    local: list[str] = []
    for key in ("requested_batches", "completed_batches", "failed_batches"):
        _require_type(payload.get(key), key, int, local)
    batches = payload.get("batches")
    if not isinstance(batches, list):
        local.append("batches is not a list")
        issues.extend(local)
        return {"complete": False, "n_records": 0}
    indices = [batch.get("batch") for batch in batches if isinstance(batch, dict)]
    if len(indices) != len(batches):
        local.append("batches contains non-object entries")
    if len(set(indices)) != len(indices):
        local.append("duplicate static batch indices")
    requested = payload.get("requested_batches")
    if isinstance(requested, int) and len(batches) != requested:
        local.append(
            f"static record count {len(batches)} != requested {requested}"
        )
    if isinstance(requested, int) and set(indices) != set(range(requested)):
        local.append("static batch indices do not cover 0..requested-1 exactly once")
    for batch in batches:
        if batch.get("status") == "failed":
            continue
        if not isinstance(batch.get("terms"), dict) or not isinstance(
            batch.get("aggregate_gradients"), dict
        ):
            local.append(f"batch {batch.get('batch')} missing terms/gradients")
    failed = sum(1 for batch in batches if batch.get("status") == "failed")
    completed = len(batches) - failed
    if payload.get("failed_batches") != failed:
        local.append(
            f"failed_batches metadata {payload.get('failed_batches')} != records {failed}"
        )
    if payload.get("completed_batches") != completed:
        local.append(
            f"completed_batches metadata {payload.get('completed_batches')} "
            f"!= records {completed}"
        )
    if expected.get("mode") is not None and payload.get("mode") != expected["mode"]:
        local.append(
            f"mode {payload.get('mode')!r} != manifest {expected['mode']!r}"
        )
    if expected.get("seed") is not None and payload.get("batch_seed") != expected["seed"]:
        local.append(
            f"batch_seed {payload.get('batch_seed')!r} != manifest {expected['seed']!r}"
        )
    issues.extend(local)
    return {"complete": not local and completed == requested, "n_records": len(batches)}


def _validate_adam(payload: dict, expected: dict, issues: list[str]) -> dict:
    local: list[str] = []
    for key in ("adam_requested", "adam_completed", "adam_failed"):
        _require_type(payload.get(key), key, int, local)
    trials = payload.get("adam_trials")
    if not isinstance(trials, list):
        local.append("adam_trials is not a list")
        issues.extend(local)
        return {"complete": False, "n_records": 0, "reference_complete": False}
    ids = [trial.get("trial") for trial in trials if isinstance(trial, dict)]
    if len(ids) != len(trials):
        local.append("adam_trials contains non-object entries")
    if len(set(ids)) != len(ids):
        local.append("duplicate Adam trial indices")
    requested = payload.get("adam_requested")
    if isinstance(requested, int) and len(trials) != requested:
        local.append(f"Adam record count {len(trials)} != requested {requested}")
    if isinstance(requested, int) and set(ids) != set(range(requested)):
        local.append("Adam trial indices do not cover 0..requested-1 exactly once")
    failed = sum(1 for trial in trials if trial.get("status") == "failed")
    completed = len(trials) - failed
    if payload.get("adam_failed") != failed:
        local.append(
            f"adam_failed metadata {payload.get('adam_failed')} != records {failed}"
        )
    if payload.get("adam_completed") != completed:
        local.append(
            f"adam_completed metadata {payload.get('adam_completed')} "
            f"!= records {completed}"
        )
    if expected.get("seed") is not None and payload.get("adam_seed") != expected["seed"]:
        # Older v3 JSONs do not record adam_seed; only flag when present.
        if payload.get("adam_seed") is not None:
            local.append(
                f"adam_seed {payload.get('adam_seed')!r} != manifest {expected['seed']!r}"
            )
    reference_complete = True
    expected_cases = set(expected.get("training_cases") or [])
    for trial in trials:
        if trial.get("status") != "ok":
            continue
        for key in (
            "contact_data_loss_change", "reference_loss_change",
            "reference_contact_before", "reference_contact_after",
            "parameter_delta",
        ):
            if key not in trial:
                local.append(f"trial {trial.get('trial')} missing {key}")
        for side in ("reference_contact_before", "reference_contact_after"):
            reference = trial.get(side, {})
            if not reference.get("complete_training_target", False):
                reference_complete = False
                local.append(f"trial {trial.get('trial')} {side} is incomplete")
            case_ids = {
                item.get("case_id") for item in reference.get("cases", [])
            }
            if expected_cases and case_ids != expected_cases:
                reference_complete = False
                local.append(
                    f"trial {trial.get('trial')} {side} case set differs from manifest"
                )
    issues.extend(local)
    return {
        "complete": not local and completed == requested,
        "n_records": len(trials),
        "reference_complete": reference_complete,
    }


def _static(paths: list[Path]) -> dict:
    batches = []
    meta = []
    issues = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        issues.extend(f"{path.name}: {item}" for item in _source_issues(payload))
        meta.append({
            "file": path.name,
            "requested": payload.get("requested_batches"),
            "completed": payload.get("completed_batches"),
            "failed": payload.get("failed_batches"),
            "non_finite": payload.get("non_finite"),
            "ok": payload.get("ok"),
        })
        for batch in payload.get("batches", []):
            if batch.get("status") == "failed":
                continue
            if not isinstance(batch, dict) or not isinstance(
                batch.get("terms"), dict
            ) or not all(name in batch["terms"] for name in
                         ("pde", "contact_ic", "interface", "contact_data")):
                issues.append(f"{path.name}: batch missing required terms")
                continue
            batches.append(batch)
    terms = {
        name: _term_stats(batches, name)
        for name in ("pde", "contact_ic", "interface", "contact_data")
    }
    ratios = [
        batch["terms"]["contact_data"]["grad_norm_weighted"]
        / batch["terms"]["pde"]["grad_norm_weighted"]
        for batch in batches
        if _finite(batch["terms"]["pde"].get("grad_norm_weighted"))
        and batch["terms"]["pde"]["grad_norm_weighted"] > 0
        and _finite(batch["terms"]["contact_data"].get("grad_norm_weighted"))
    ]
    cosines = {
        key: _cosine_stats(batches, key)
        for key in ("cos_data_vs_pde", "cos_data_vs_ic",
                    "cos_data_vs_interface", "cos_data_vs_other",
                    "cos_data_vs_total")
    }
    by_slice = {}
    slices = sorted({
        batch.get("slice") for batch in batches
        if batch.get("slice") is not None
    })
    for which in slices:
        subset = [batch for batch in batches if batch.get("slice") == which]
        by_slice[str(which)] = {
            "n_batches": len(subset),
            "terms": {
                name: _term_stats(subset, name)
                for name in ("pde", "contact_ic", "interface", "contact_data")
            },
            "cosines": {
                key: _cosine_stats(subset, key)
                for key in ("cos_data_vs_pde", "cos_data_vs_ic",
                            "cos_data_vs_interface", "cos_data_vs_other",
                            "cos_data_vs_total")
            },
        }
    return {
        "files": meta,
        "source_issues": issues,
        "n_batches": len(batches),
        "terms": terms,
        "data_to_pde_ratio_median": _median_valid(ratios),
        "cosines": cosines,
        "by_slice": by_slice,
    }


def _difference_value(before, after):
    if not (_finite(before) and _finite(after)):
        return None
    return float(after) - float(before)


def _adam(paths: list[Path]) -> dict:
    trials = []
    meta = []
    issues = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        issues.extend(f"{path.name}: {item}" for item in _source_issues(payload))
        meta.append({
            "file": path.name,
            "requested": payload.get("adam_requested"),
            "completed": payload.get("adam_completed"),
            "failed": payload.get("adam_failed"),
            "non_finite": payload.get("non_finite"),
            "checkpoint_unchanged": payload.get("checkpoint_sha256", {}).get(
                "unchanged"
            ),
            "checkpoint_loop": payload.get("checkpoint_loop"),
            "ok": payload.get("ok"),
        })
        for trial in payload.get("adam_trials", []):
            if trial.get("status") != "ok":
                continue
            required = (
                "contact_data_loss_change", "reference_loss_change",
                "reference_contact_before", "reference_contact_after",
                "parameter_delta",
            )
            if not isinstance(trial, dict) or any(
                key not in trial for key in required
            ):
                issues.append(f"{path.name}: Adam trial missing required fields")
                continue
            trials.append(trial)
    batch_change = [
        trial.get("contact_data_loss_change", {}).get("weighted_current_batch")
        for trial in trials
    ]
    macro_change = [
        trial.get("reference_loss_change", {}).get("macro_mse")
        for trial in trials
    ]
    pooled_change = [
        trial.get("reference_loss_change", {}).get("pooled_mse")
        for trial in trials
    ]
    coverage_change = [
        trial.get("reference_loss_change", {}).get("coverage")
        for trial in trials
    ]
    prediction = [
        trial.get("contact_data_loss_change", {}).get(
            "first_order_prediction_g_d_dot_delta"
        )
        for trial in trials
    ]
    pairs = [
        (float(a), float(b)) for a, b in zip(batch_change, macro_change)
        if _finite(a) and _finite(b)
    ]
    pairs_pooled = [
        (float(a), float(b)) for a, b in zip(batch_change, pooled_change)
        if _finite(a) and _finite(b)
    ]
    sign_pairs = [
        (float(a), float(b)) for a, b in zip(prediction, batch_change)
        if _finite(a) and _finite(b)
    ]
    cases: dict[str, list[float]] = {}
    case_reference_counts: dict[str, dict] = {}
    for trial in trials:
        after = {
            item["case_id"]: item for item in
            trial["reference_contact_after"].get("cases", [])
        }
        for before in trial["reference_contact_before"].get("cases", []):
            item = after.get(before["case_id"])
            if item is None:
                continue
            if _finite(before.get("mse")) and _finite(item.get("mse")):
                cases.setdefault(before["case_id"], []).append(
                    float(item["mse"]) - float(before["mse"])
                )
            case_reference_counts.setdefault(before["case_id"], {
                "n": 0, "covered_before": 0, "covered_after": 0,
            })
            case_reference_counts[before["case_id"]]["n"] += int(before.get("n", 0))
            case_reference_counts[before["case_id"]]["covered_before"] += int(
                before.get("covered_count", 0)
            )
            case_reference_counts[before["case_id"]]["covered_after"] += int(
                item.get("covered_count", 0)
            )
    per_case = {
        case_id: {
            **_change_stats(values, greater_is_better=False),
            **{
                key: case_reference_counts[case_id][key]
                for key in ("n", "covered_before", "covered_after")
            },
        }
        for case_id, values in cases.items()
    }
    thirds = {}
    for which in range(3):
        macro_values = []
        pooled_values = []
        for trial in trials:
            before = trial["reference_contact_before"].get("thirds", [])
            after = trial["reference_contact_after"].get("thirds", [])
            if len(before) <= which or len(after) <= which:
                continue
            macro_values.append(
                _difference_value(before[which].get("macro_mse"),
                                  after[which].get("macro_mse"))
            )
            pooled_values.append(
                _difference_value(before[which].get("pooled_mse"),
                                  after[which].get("pooled_mse"))
            )
        thirds[str(which)] = {
            "macro": _change_stats(macro_values, greater_is_better=False),
            "pooled": _change_stats(pooled_values, greater_is_better=False),
        }
    reference_n = [
        trial["reference_contact_after"].get("n") for trial in trials
        if _finite(trial["reference_contact_after"].get("n"))
    ]
    covered_before = [
        trial["reference_contact_before"].get("covered_count")
        for trial in trials
    ]
    covered_after = [
        trial["reference_contact_after"].get("covered_count")
        for trial in trials
    ]
    return {
        "files": meta,
        "source_issues": issues,
        "n_trials": len(trials),
        "batch": _change_stats(batch_change, greater_is_better=False),
        "macro_mse": _change_stats(macro_change, greater_is_better=False),
        "pooled_mse": _change_stats(pooled_change, greater_is_better=False),
        "coverage": _change_stats(coverage_change, greater_is_better=True),
        "both_improved_denominator": len(pairs),
        "both_improved": sum(1 for a, b in pairs if a < 0 and b < 0),
        "batch_good_macro_bad": sum(1 for a, b in pairs if a < 0 and b > 0),
        "both_improved_pooled": sum(
            1 for a, b in pairs_pooled if a < 0 and b < 0
        ),
        "batch_good_pooled_bad": sum(
            1 for a, b in pairs_pooled if a < 0 and b > 0
        ),
        "first_order_sign_matches": sum(
            1 for a, b in sign_pairs if (a < 0) == (b < 0)
        ),
        "first_order_sign_evaluable": len(sign_pairs),
        "parameter_delta_l2_median": _median_valid([
            trial.get("parameter_delta", {}).get("l2") for trial in trials
        ]),
        "reference_n_per_trial": (
            sorted(set(int(value) for value in reference_n))
            if reference_n else []
        ),
        "reference_n_cumulative": int(sum(reference_n)),
        "reference_n_total": int(sum(reference_n)),
        "covered_before_total": int(sum(
            value for value in covered_before if _finite(value)
        )),
        "covered_after_total": int(sum(
            value for value in covered_after if _finite(value)
        )),
        "per_case": per_case,
        "thirds": thirds,
    }


def _fmt(value, digits=4):
    if not _finite(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def _status_line(entry: dict) -> str:
    if entry["n_evaluable"] == 0:
        return "不可评估"
    return (
        f"{entry['improved']}/{entry['n_evaluable']} 改善，"
        f"{entry['unchanged']} 持平，{entry['worsened']} 变差"
    )


def _write_markdown(summary: dict, out: Path) -> None:
    lines = ["# 接触训练诊断（metric v3：macro + pooled）\n"]
    if summary["source_issues"]:
        lines.append("\n> **源诊断失败或不完整**：本摘要不能作为正常结论。\n")
        for issue in summary["source_issues"]:
            lines.append(f"> - {issue}\n")
    lines.append("\n所有结果由只读诊断产生；检查点未修改。参考指标同时报告"
                 "算例等权 macro 与帧数加权 pooled。\n")
    lines.append("\n## 静态聚合梯度（10 个训练算例的平均损失）\n")
    lines.append("| 检查点 | 批数 | data/pde | contact_data 梯度 | "
                 "data vs other 有效/总 (负值) | undefined | non_finite |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for tag, entry in summary["static"].items():
        cos = entry["cosines"]["cos_data_vs_other"]
        lines.append(
            f"| {tag} | {entry['n_batches']} | "
            f"{_fmt(entry['data_to_pde_ratio_median'], 2)} | "
            f"{_fmt(entry['terms']['contact_data']['grad_norm_median'], 4)} | "
            f"{cos['valid_directions']}/{cos['requested']} "
            f"({cos['negative']}) | {cos['undefined']} | {cos['non_finite']} |\n"
        )
    lines.append("\n## 单步 Adam（独立试验，不是连续训练）\n")
    lines.append("| 检查点 | 试验 | 批次 data 改善 | macro_mse 改善 | pooled_mse 改善 | "
                 "batch+macro 同时 | batch+pooled 同时 | 批次好/macro坏 | 覆盖率改善 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"| {tag} | {entry['n_trials']} | "
            f"{_status_line(entry['batch'])} | {_status_line(entry['macro_mse'])} | "
            f"{_status_line(entry['pooled_mse'])} | "
            f"{entry['both_improved']}/{entry['both_improved_denominator']} | "
            f"{entry['both_improved_pooled']}/{entry['both_improved_denominator']} | "
            f"{entry['batch_good_macro_bad']} | {_status_line(entry['coverage'])} |\n"
        )
    lines.append("\n变化中位数与一阶预测符号一致率：\n")
    lines.append("| 检查点 | 批次 data Δ | macro_mse Δ | pooled_mse Δ | 覆盖率 Δ | "
                 "一阶符号一致 | 参数 Δ L2 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"| {tag} | {_fmt(entry['batch']['median'], 8)} | "
            f"{_fmt(entry['macro_mse']['median'], 8)} | "
            f"{_fmt(entry['pooled_mse']['median'], 8)} | "
            f"{_fmt(entry['coverage']['median'], 8)} | "
            f"{entry['first_order_sign_matches']}/{entry['first_order_sign_evaluable']} | "
            f"{_fmt(entry['parameter_delta_l2_median'], 6)} |\n"
        )
    lines.append("\n固定参考点按半径的 macro MSE 变化中位数（负值改善）：\n")
    case_ids = sorted({
        case_id for entry in summary["adam"].values()
        for case_id in entry["per_case"]
    }, key=lambda value: int(value.split("_")[0][1:]))
    lines.append("| 检查点 | " + " | ".join(case_ids) + " |\n")
    lines.append("|---|" + "---:|" * len(case_ids) + "\n")
    for tag, entry in summary["adam"].items():
        lines.append(f"| {tag} | " + " | ".join(
            _fmt(entry["per_case"].get(case_id, {}).get("median"), 6)
            for case_id in case_ids
        ) + " |\n")
    lines.append("\n按时间三分位的 macro / pooled MSE 变化中位数：\n")
    lines.append("| 检查点 | 时段 | macro Δ | pooled Δ |\n")
    lines.append("|---|---|---:|---:|\n")
    for tag, entry in summary["adam"].items():
        for which, item in entry["thirds"].items():
            lines.append(
                f"| {tag} | {which} | {_fmt(item['macro']['median'], 7)} | "
                f"{_fmt(item['pooled']['median'], 7)} |\n"
            )
    lines.append("\n精确覆盖率计数（参考点，所有成功试验合计）：\n")
    lines.append("| 检查点 | 覆盖率改善 | covered before | covered after | n |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"| {tag} | {_status_line(entry['coverage'])} | "
            f"{entry['covered_before_total']} | {entry['covered_after_total']} | "
            f"{entry['reference_n_total']} |\n"
        )
    lines.append("\n## 分时段补充诊断\n")
    lines.append("分时段批次只用于定位，不代表正式训练分布。\n")
    lines.append("\n| 检查点 | 时段 | 批数 | contact_data 梯度 | data vs other 有效/总 (负) |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for tag, entry in summary["stratified"].items():
        for which, item in entry.get("by_slice", {}).items():
            cos = item["cosines"]["cos_data_vs_other"]
            lines.append(
                f"| {tag} | {which} | {item['n_batches']} | "
                f"{_fmt(item['terms']['contact_data']['grad_norm_median'], 4)} | "
                f"{cos['valid_directions']}/{cos['requested']} "
                f"({cos['negative']}) |\n"
            )
    out.write_text("".join(lines), encoding="utf-8")


def _majority(entry: dict) -> str:
    """Describe a change distribution without treating ties as worsening.

    ``n`` is accepted for the paired-experiment summary, ``n_evaluable`` for
    the training diagnostic summary.  The rule is shared so both reports use
    the same strict-majority definition.
    """
    n = int(entry.get("n_evaluable", entry.get("n", 0)) or 0)
    if n == 0:
        return "不可判断"
    improved = int(entry.get("improved") or 0)
    worsened = int(entry.get("worsened") or 0)
    unchanged = int(entry.get("unchanged") or 0)
    if improved == n:
        return "全部改善"
    if worsened == n:
        return "全部变差"
    if unchanged == n:
        return "全部持平"
    if improved * 2 == n:
        return "半数改善"
    if improved * 2 > n:
        return "多数改善"
    if worsened * 2 > n:
        return "多数变差"
    if unchanged * 2 > n:
        return "多数持平"
    return "无多数（方向不一致）"


def _report_text(summary: dict) -> str:
    lines = ["# 接触训练诊断报告（修复版，metric v3）\n"]
    if summary["source_issues"]:
        lines.append("\n> **输入不完整或源诊断失败，以下不能作为正常结论。**\n")
        for issue in summary["source_issues"]:
            lines.append(f"> - {issue}\n")
    lines.append("\n本轮是同一检查点反复开始的独立单步试验，不是连续训练，"
                 "不能据此证明长期收敛或长期误差的根因。\n")
    independent = sorted({
        value
        for entry in summary["adam"].values()
        for value in entry.get("reference_n_per_trial", [])
    })
    if independent:
        per_trial = "、".join(str(value) for value in independent)
        cumulative = sum(
            entry.get("reference_n_cumulative", 0)
            for entry in summary["adam"].values()
        )
        lines.append(
            "参考指标同时使用算例等权 macro 与帧数加权 pooled；"
            f"每次独立试验的参考帧集合为 {per_trial} 帧（由实际记录计算），"
            f"所有成功试验累计评估点数为 {cumulative}，二者不能混用。\n"
        )
    else:
        lines.append("参考指标同时使用算例等权 macro 与帧数加权 pooled；"
                     "没有可用的独立参考帧记录，不能判断评估规模。\n")
    if summary.get("not_requested"):
        lines.append("\n未运行的清单项：" + "；".join(
            item.get("file", "?") for item in summary["not_requested"]
        ) + "。\n")
    if summary.get("unexpected_files"):
        lines.append("\n清单外文件（未纳入结论）：" + "、".join(
            summary["unexpected_files"]
        ) + "。\n")

    lines.append("\n## 1. 正式训练分布下的静态梯度\n")
    for tag, entry in summary["static"].items():
        cos = entry["cosines"]["cos_data_vs_other"]
        contact = entry["terms"]["contact_data"]["grad_norm_median"]
        pde = entry["terms"]["pde"]["grad_norm_median"]
        ic = entry["terms"]["contact_ic"]["grad_norm_median"]
        interface = entry["terms"]["interface"]["grad_norm_median"]
        values = [value for value in (contact, pde, ic, interface)
                  if _finite(value)]
        if not values or not _finite(contact):
            dominance = "接触梯度大小不可判断"
        elif contact >= max(values):
            dominance = "接触梯度是四项中最大"
        else:
            dominance = "接触梯度不是四项中最大"
        if cos["valid_directions"] == 0:
            conflict = "方向不可判断（无有效方向）"
        elif cos["negative"] * 2 == cos["valid_directions"]:
            conflict = "有效方向中负值为半数"
        elif cos["negative"] * 2 > cos["valid_directions"]:
            conflict = "有效方向中负值为多数"
        else:
            conflict = "有效方向中负值为少数"
        lines.append(
            f"- {tag}：contact_data 中位 {_fmt(contact, 5)}，"
            f"PDE {_fmt(pde, 5)}，IC {_fmt(ic, 5)}，interface {_fmt(interface, 5)}；"
            f"data/pde {_fmt(entry['data_to_pde_ratio_median'], 3)}；{dominance}。"
            f"data-vs-other 有效方向 {cos['valid_directions']}/{cos['requested']}，"
            f"负值 {cos['negative']}，undefined {cos['undefined']}，"
            f"non_finite {cos['non_finite']}；{conflict}。\n"
        )

    lines.append("\n## 2. 单步 Adam：macro 与 pooled\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"- {tag}：batch {_status_line(entry['batch'])}（{_majority(entry['batch'])}）；"
            f"macro {_status_line(entry['macro_mse'])}（{_majority(entry['macro_mse'])}）；"
            f"pooled {_status_line(entry['pooled_mse'])}（{_majority(entry['pooled_mse'])}）；"
            f"覆盖率 {_status_line(entry['coverage'])}（{_majority(entry['coverage'])}）。\n"
        )
        lines.append(
            f"  中位变化：batch {_fmt(entry['batch']['median'], 8)}，"
            f"macro {_fmt(entry['macro_mse']['median'], 8)}，"
            f"pooled {_fmt(entry['pooled_mse']['median'], 8)}；"
            f"batch+macro 同时改善 {entry['both_improved']}/"
            f"{entry['both_improved_denominator']}，"
            f"batch+pooled 同时改善 {entry['both_improved_pooled']}/"
            f"{entry['both_improved_denominator']}，"
            f"批次好/macro 坏 {entry['batch_good_macro_bad']}。\n"
        )

    lines.append("\n## 3. 覆盖率（精确整数计数）\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"- {tag}：每次独立试验的参考帧集合 "
            f"{entry.get('reference_n_per_trial') or '不可判断'}；"
            f"所有成功试验累计评估点 "
            f"{entry.get('reference_n_cumulative', entry.get('reference_n_total', '不可判断'))}；"
            f"covered {entry['covered_before_total']} → "
            f"{entry['covered_after_total']}；覆盖率变化 "
            f"{_status_line(entry['coverage'])}（{_majority(entry['coverage'])}）。"
            "旧版的浮点恢复可能少算，本版直接累加整数。\n"
        )

    lines.append("\n## 4. 半径与时段折中\n")
    for tag, entry in summary["adam"].items():
        parts = []
        for case_id, item in sorted(
            entry["per_case"].items(),
            key=lambda pair: int(pair[0].split("_")[0][1:]),
        ):
            if _finite(item.get("median")):
                direction = "改善" if item["median"] < 0 else (
                    "变差" if item["median"] > 0 else "持平"
                )
                parts.append(f"{case_id} {_fmt(item['median'], 3)}（{direction}）")
        lines.append(f"- {tag} 各半径 macro MSE 中位变化：" + "；".join(parts) + "。\n")
        for which, item in entry["thirds"].items():
            lines.append(
                f"  时段 {which}：macro {_fmt(item['macro']['median'], 3)}"
                f"（{_majority(item['macro'])}），"
                f"pooled {_fmt(item['pooled']['median'], 3)}"
                f"（{_majority(item['pooled'])}）。\n"
            )

    lines.append("\n## 5. 由当前输入生成的结论\n")
    if summary["source_issues"]:
        lines.append("- 源诊断不完整或失败，不能生成正常结论。\n")
    else:
        for tag, entry in summary["adam"].items():
            lines.append(
                f"- {tag}：batch {_majority(entry['batch'])}；"
                f"macro {_majority(entry['macro_mse'])}；"
                f"pooled {_majority(entry['pooled_mse'])}；"
                f"覆盖率 {_majority(entry['coverage'])}。"
            )
            if (entry["macro_mse"]["improved"] is not None
                    and entry["pooled_mse"]["improved"] is not None
                    and entry["macro_mse"]["improved"] != entry["pooled_mse"]["improved"]):
                lines.append(
                    " macro 与 pooled 的改善次数不同，不能用一个口径代替另一个。"
                )
            lines.append("\n")
        if all(
            summary["adam"][tag]["macro_mse"]["n_evaluable"] == 0
            for tag in summary["adam"]
        ):
            lines.append("- 所有检查点的 macro 均不可评估，不能判断方向。\n")
    lines.append("\n“多数”仅在改善数超过有效试验数一半时使用；恰好一半写“半数”。"
                 "所有比例使用对应有效分母。\n")
    return "".join(lines)


def _manifest_expectations(manifest: dict) -> dict:
    expected = {}
    for item in manifest.get("experiments", []):
        kind = item.get("kind")
        file_name = item.get("file")
        if kind not in ("static", "adam", "stratified") or not file_name:
            raise ValueError(f"invalid manifest experiment entry: {item!r}")
        expected[file_name] = item
    return expected


def _check_group_consistency(paths: list[Path], payloads: list[dict],
                             manifest: dict, issues: list[str]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for path, payload in zip(paths, payloads):
        kind = "adam" if "adam_trials" in payload else "static"
        checkpoint = str(payload.get("checkpoint", ""))
        key = (kind, checkpoint, payload.get("mode"))
        groups.setdefault(key, []).append({
            "file": path.name, "payload": payload
        })
    for key, entries in groups.items():
        configs = {entry["payload"].get("config") for entry in entries}
        versions = {entry["payload"].get("metric_version") for entry in entries}
        hashes = {
            entry["payload"].get("checkpoint_sha256", {}).get("after")
            for entry in entries
        }
        if len(configs) > 1:
            issues.append(f"mixed config in group {key}: {sorted(map(str, configs))}")
        if len(versions) > 1:
            issues.append(f"mixed metric_version in group {key}: {versions}")
        if len(hashes) > 1:
            issues.append(f"mixed checkpoint hashes in group {key}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory",
        default="outputs/fullframes/contact_training_diagnostics_fixed",
    )
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    manifest_path = (
        Path(args.manifest) if args.manifest
        else directory / "diagnostic_manifest.json"
    )
    issues = []
    if not manifest_path.is_file():
        issues.append(f"diagnostic manifest not found: {manifest_path}")
        manifest = {"experiments": [], "checkpoints": {}}
    else:
        try:
            manifest = _load_manifest(manifest_path)
        except Exception as exc:
            issues.append(f"invalid diagnostic manifest: {exc}")
            manifest = {"experiments": [], "checkpoints": {}}
    expected = _manifest_expectations(manifest) if manifest.get("experiments") else {}

    static_paths = {tag: [] for tag in ("best", "last")}
    adam_paths = {tag: [] for tag in ("best", "last")}
    stratified_paths = {tag: [] for tag in ("best", "last")}
    not_requested = []
    payloads_for_consistency = []
    paths_for_consistency = []
    seen = set()
    for file_name, item in expected.items():
        key = (
            item.get("kind"), item.get("checkpoint"), item.get("mode"),
            item.get("seed"),
        )
        if key in seen:
            issues.append(f"duplicate manifest entry: {key}")
        seen.add(key)
        path = directory / file_name
        if not path.is_file():
            if item.get("optional", False):
                not_requested.append({"file": file_name, "reason": "optional not run"})
            else:
                issues.append(f"required result missing: {file_name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"{file_name}: invalid JSON: {exc}")
            continue
        if payload.get("version") != 3 or payload.get("metric_version") != 3:
            issues.append(f"{file_name}: schema is not metric version 3")
        if payload.get("format") != "spherical_blast_pinn.contact_gradient_diagnostic":
            issues.append(f"{file_name}: wrong format field")
        if not isinstance(payload.get("config"), str):
            issues.append(f"{file_name}: config is not a string")
        if payload.get("ok") is False:
            issues.extend(
                f"{file_name}: {reason}"
                for reason in (payload.get("failure_reasons") or ["ok=false"])
            )
        checkpoint_info = manifest.get("checkpoints", {}).get(item.get("checkpoint"), {})
        expected_hash = checkpoint_info.get("sha256")
        before = payload.get("checkpoint_sha256", {}).get("before")
        after = payload.get("checkpoint_sha256", {}).get("after")
        if expected_hash and before != expected_hash:
            issues.append(f"{file_name}: checkpoint hash != manifest {item['checkpoint']}")
        if expected_hash and after != expected_hash:
            issues.append(f"{file_name}: checkpoint after hash != manifest {item['checkpoint']}")
        if item["kind"] == "adam":
            validation = _validate_adam(
                payload, {**item, "training_cases": manifest.get("training_cases")},
                issues,
            )
            adam_paths[item["checkpoint"]].append(path)
        else:
            validation = _validate_static(payload, item, issues)
            target = (
                stratified_paths if item["kind"] == "stratified"
                else static_paths
            )
            target[item["checkpoint"]].append(path)
        payloads_for_consistency.append(payload)
        paths_for_consistency.append(path)
        if not validation.get("complete", False):
            issues.append(f"{file_name}: records are incomplete")
        if payload.get("non_finite"):
            issues.append(f"{file_name}: non-finite value reported")

    _check_group_consistency(
        paths_for_consistency, payloads_for_consistency, manifest, issues
    )
    # Unexpected files are listed, not treated as failures.
    expected_names = set(expected)
    unexpected = sorted(
        path.name for path in directory.glob("*.json")
        if path.name not in expected_names
        and path.name not in (
            "summary.json", "diagnostic_config.json", "diagnostic_manifest.json"
        )
    )

    payload = {
        "directory": str(directory),
        "manifest": str(manifest_path),
        "manifest_ok": manifest_path.is_file() and not any(
            item.startswith("invalid diagnostic manifest") for item in issues
        ),
        "metric_version": 3,
        "static": {},
        "adam": {},
        "stratified": {},
        "not_requested": not_requested,
        "unexpected_files": unexpected,
    }
    for tag in ("best", "last"):
        payload["static"][tag] = _static(static_paths[tag])
        payload["adam"][tag] = _adam(adam_paths[tag])
        payload["stratified"][tag] = _static(stratified_paths[tag])
    for group in ("static", "adam", "stratified"):
        for entry in payload[group].values():
            issues.extend(entry.get("source_issues", []))
    payload["source_issues"] = list(dict.fromkeys(issues))
    payload["ok"] = not payload["source_issues"]
    out_json = directory / "summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    _write_markdown(payload, directory / "summary.md")
    (directory / "report.md").write_text(_report_text(payload),
                                         encoding="utf-8")
    print(f"summary: {out_json}")
    print(f"markdown: {directory / 'summary.md'}")
    print(f"report: {directory / 'report.md'}")
    if payload["source_issues"]:
        print("summary: FAILED (input incomplete or failed)", file=sys.stderr)
        for issue in payload["source_issues"][:50]:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
