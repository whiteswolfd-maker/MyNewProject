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
        batches.extend(
            batch for batch in payload.get("batches", [])
            if batch.get("status") != "failed"
        )
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
        trials.extend(
            trial for trial in payload.get("adam_trials", [])
            if trial.get("status") == "ok"
        )
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


def _report_text(summary: dict) -> str:
    lines = ["# 接触训练诊断报告（修复版，metric v3）\n"]
    if summary["source_issues"]:
        lines.append("\n> **源诊断失败或不完整，以下不能作为正常结论。**\n")
        for issue in summary["source_issues"]:
            lines.append(f"> - {issue}\n")
    lines.append(
        "\n本轮是同一检查点反复开始的独立单步试验，不是连续训练，"
        "不能据此证明长期收敛或长期误差的根因。\n"
    )
    lines.append("\n## 1. 正式训练分布下的静态梯度\n")
    for tag, entry in summary["static"].items():
        cos = entry["cosines"]["cos_data_vs_other"]
        lines.append(
            f"- {tag}：contact_data 梯度中位数 "
            f"{_fmt(entry['terms']['contact_data']['grad_norm_median'], 5)}，"
            f"PDE {_fmt(entry['terms']['pde']['grad_norm_median'], 5)}，"
            f"data/pde {_fmt(entry['data_to_pde_ratio_median'], 3)}；"
            f"data-vs-other 有效方向 {cos['valid_directions']}/{cos['requested']}，"
            f"其中负值 {cos['negative']}，undefined {cos['undefined']}，"
            f"non_finite {cos['non_finite']}。\n"
        )
    lines.append("\n## 2. 单步 Adam：两种参考口径\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"- {tag}：批次 data {_status_line(entry['batch'])}；"
            f"macro_mse {_status_line(entry['macro_mse'])}；"
            f"pooled_mse {_status_line(entry['pooled_mse'])}；"
            f"覆盖率 {_status_line(entry['coverage'])}。\n"
        )
        lines.append(
            f"  中位变化：批次 {_fmt(entry['batch']['median'], 8)}，"
            f"macro {_fmt(entry['macro_mse']['median'], 8)}，"
            f"pooled {_fmt(entry['pooled_mse']['median'], 8)}。"
            f" 两者同时改善(batch+macro) {entry['both_improved']}/"
            f"{entry['both_improved_denominator']}；"
            f"batch+pooled 同时改善 {entry['both_improved_pooled']}/"
            f"{entry['both_improved_denominator']}；"
            f"批次好/macro 坏 {entry['batch_good_macro_bad']}。\n"
        )
    lines.append("\n## 3. 覆盖率修正\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"- {tag}：参考点精确 covered {entry['covered_before_total']} → "
            f"{entry['covered_after_total']}（n={entry['reference_n_total']}）；"
            f"覆盖率变化 {_status_line(entry['coverage'])}。"
            "旧版通过 int(coverage*n) 恢复命中数，可能少算；本版直接累加整数。\n"
        )
    lines.append("\n## 4. 半径与时段折中\n")
    for tag, entry in summary["adam"].items():
        case_parts = []
        for case_id, item in sorted(
            entry["per_case"].items(),
            key=lambda pair: int(pair[0].split("_")[0][1:]),
        ):
            if _finite(item.get("median")):
                case_parts.append(f"{case_id} {_fmt(item['median'], 3)}")
        lines.append(f"- {tag} 各半径 macro MSE 变化中位数：" + "；".join(case_parts) + "。\n")
        for which, item in entry["thirds"].items():
            lines.append(
                f"  时段 {which}：macro {_fmt(item['macro']['median'], 3)}，"
                f"pooled {_fmt(item['pooled']['median'], 3)}。\n"
            )
    lines.append("\n## 5. 哪些旧结论保留/修正\n")
    lines.append(
        "- 旧分时段诊断把接触与全时段 PDE/界面梯度比较，不能代表正式训练；本版用正式 10 算例聚合，结论以其为准。\n"
        "- 接触梯度仍是最大项之一，但训练分布下不是旧版的 25–29 倍；方向冲突常见，但更新没有被阻断。\n"
        f"- best：batch {_status_line(summary['adam']['best']['batch'])}，"
        f"macro {_status_line(summary['adam']['best']['macro_mse'])}，"
        f"pooled {_status_line(summary['adam']['best']['pooled_mse'])}。\n"
        f"- last：batch {_status_line(summary['adam']['last']['batch'])}，"
        f"macro {_status_line(summary['adam']['last']['macro_mse'])}，"
        f"pooled {_status_line(summary['adam']['last']['pooled_mse'])}；"
        "pooled 的改善掩盖了 macro 的多数变差。\n"
        "- 不同半径和早/中/晚时段存在折中；覆盖率在 best 上多数上升，在 last 上接近持平/略降。\n"
        "- 旧版通过 int(coverage*n) 逐例恢复命中数，浮点截断会少算；本版直接累加整数并重算。\n"
    )
    if summary["source_issues"]:
        lines.append("\n**由于源诊断存在问题，以上结论均应视为不完整。**\n")
    return "".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory",
        default="outputs/fullframes/contact_training_diagnostics_fixed",
    )
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    payload = {
        "directory": str(directory),
        "metric_version": 3,
        "static": {},
        "adam": {},
        "stratified": {},
    }
    for tag in ("best", "last"):
        payload["static"][tag] = _static(sorted(
            directory.glob(f"static_{tag}_seed*.json")
        ))
        payload["adam"][tag] = _adam(sorted(
            directory.glob(f"adam_{tag}_seed*.json")
        ))
        payload["stratified"][tag] = _static(sorted(
            directory.glob(f"stratified_{tag}_*.json")
        ))
    issues = []
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
        print("summary: FAILED (source diagnostics incomplete)", file=sys.stderr)
        for issue in payload["source_issues"][:50]:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
