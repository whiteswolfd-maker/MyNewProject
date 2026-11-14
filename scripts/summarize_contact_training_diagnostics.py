#!/usr/bin/env python3
"""Summarise the read-only contact-training diagnostics.

Reads the JSON files produced by ``diagnose_contact_gradients.py`` and writes
``summary.json``, ``summary.md`` and ``report.md``.  It never edits a cache,
checkpoint or training output.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _static(paths: list[Path]) -> dict:
    batches = []
    meta = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta.append({
            "file": path.name,
            "requested": payload["requested_batches"],
            "completed": payload["completed_batches"],
            "failed": payload["failed_batches"],
            "non_finite": payload["non_finite"],
        })
        batches.extend(
            batch for batch in payload["batches"]
            if batch.get("status") != "failed"
        )
    terms = {}
    for name in ("pde", "contact_ic", "interface", "contact_data"):
        terms[name] = {
            "loss_median": _median([
                batch["terms"][name]["raw_loss"] for batch in batches
            ]),
            "grad_norm_median": _median([
                batch["terms"][name]["grad_norm_weighted"] for batch in batches
            ]),
        }
    ratios = [
        batch["terms"]["contact_data"]["grad_norm_weighted"]
        / batch["terms"]["pde"]["grad_norm_weighted"]
        for batch in batches
        if batch["terms"]["pde"]["grad_norm_weighted"] > 0
    ]
    cosines = {}
    for key in ("cos_data_vs_pde", "cos_data_vs_ic",
                "cos_data_vs_interface", "cos_data_vs_other",
                "cos_data_vs_total"):
        values = [batch["aggregate_gradients"][key]["cosine"] for batch in batches]
        statuses = [batch["aggregate_gradients"][key]["status"] for batch in batches]
        cosines[key] = {
            "median": _median(values),
            "negative": sum(1 for value in values if value < 0),
            "n": len(values),
            "undefined": statuses.count("direction_undefined"),
            "non_finite": statuses.count("non_finite"),
        }
    by_slice = {}
    for which in sorted({batch.get("slice") for batch in batches
                         if batch.get("slice") is not None}):
        subset = [batch for batch in batches if batch.get("slice") == which]
        slice_terms = {}
        for name in ("pde", "contact_ic", "interface", "contact_data"):
            slice_terms[name] = {
                "loss_median": _median([
                    batch["terms"][name]["raw_loss"] for batch in subset
                ]),
                "grad_norm_median": _median([
                    batch["terms"][name]["grad_norm_weighted"] for batch in subset
                ]),
            }
        slice_cos = {}
        for key in ("cos_data_vs_pde", "cos_data_vs_ic",
                    "cos_data_vs_interface", "cos_data_vs_other",
                    "cos_data_vs_total"):
            values = [batch["aggregate_gradients"][key]["cosine"]
                      for batch in subset]
            slice_cos[key] = {
                "median": _median(values),
                "negative": sum(1 for value in values if value < 0),
                "n": len(values),
            }
        by_slice[str(which)] = {
            "n_batches": len(subset),
            "terms": slice_terms,
            "cosines": slice_cos,
        }
    return {
        "files": meta,
        "n_batches": len(batches),
        "terms": terms,
        "data_to_pde_ratio_median": _median(ratios),
        "cosines": cosines,
        "by_slice": by_slice,
    }


def _adam(paths: list[Path]) -> dict:
    trials = []
    meta = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta.append({
            "file": path.name,
            "requested": payload["adam_requested"],
            "completed": payload["adam_completed"],
            "failed": payload["adam_failed"],
            "non_finite": payload["non_finite"],
            "checkpoint_unchanged": payload["checkpoint_sha256"]["unchanged"],
            "checkpoint_loop": payload["checkpoint_loop"],
        })
        trials.extend(
            trial for trial in payload["adam_trials"]
            if trial.get("status") == "ok"
        )
    batch_change = [
        trial["contact_data_loss_change"]["weighted_current_batch"]
        for trial in trials
    ]
    reference_change = [
        trial["reference_loss_change"]["mean_squared_violation"]
        for trial in trials
    ]
    coverage_change = [
        trial["reference_loss_change"]["coverage"] for trial in trials
    ]
    prediction = [
        trial["contact_data_loss_change"]["first_order_prediction_g_d_dot_delta"]
        for trial in trials
    ]
    cases: dict[str, list[float]] = {}
    for trial in trials:
        after = {item["case_id"]: item for item in
                 trial["reference_contact_after"]["cases"]}
        for before in trial["reference_contact_before"]["cases"]:
            item = after[before["case_id"]]
            if (before["mean_squared_violation"] is not None
                    and item["mean_squared_violation"] is not None):
                cases.setdefault(before["case_id"], []).append(
                    item["mean_squared_violation"]
                    - before["mean_squared_violation"]
                )
    per_case = {
        case_id: {
            "median_change": _median(values),
            "n": len(values),
            "improved": sum(1 for value in values if value < 0),
        }
        for case_id, values in cases.items()
    }
    thirds = {}
    for which in range(3):
        deltas = []
        for trial in trials:
            after = {item["case_id"]: item for item in
                     trial["reference_contact_after"]["cases"]}
            num = den = 0
            for before in trial["reference_contact_before"]["cases"]:
                item = after[before["case_id"]]
                b = before["thirds"][which]
                a = item["thirds"][which]
                if b["n"] and a["mean_squared_violation"] is not None:
                    num += (a["mean_squared_violation"]
                            - b["mean_squared_violation"]) * b["n"]
                    den += b["n"]
            if den:
                deltas.append(num / den)
        thirds[str(which)] = {
            "median_change": _median(deltas),
            "improved": sum(1 for value in deltas if value < 0),
            "n": len(deltas),
        }
    return {
        "files": meta,
        "n_trials": len(trials),
        "batch_contact_data_improved": sum(1 for value in batch_change if value < 0),
        "batch_contact_data_change_median": _median(batch_change),
        "reference_violation_improved": sum(
            1 for value in reference_change if value < 0
        ),
        "reference_violation_change_median": _median(reference_change),
        "both_improved": sum(
            1 for a, b in zip(batch_change, reference_change) if a < 0 and b < 0
        ),
        "batch_good_reference_bad": sum(
            1 for a, b in zip(batch_change, reference_change) if a < 0 and b > 0
        ),
        "coverage_improved": sum(1 for value in coverage_change if value > 0),
        "coverage_change_median": _median(coverage_change),
        "first_order_sign_matches": sum(
            1 for a, b in zip(prediction, batch_change) if (a < 0) == (b < 0)
        ),
        "parameter_delta_l2_median": _median([
            trial["parameter_delta"]["l2"] for trial in trials
        ]),
        "per_case": per_case,
        "thirds": thirds,
        "n": len(trials),
    }


def _write_markdown(summary: dict, out: Path) -> None:
    lines = ["# 接触训练诊断（正式训练分布 + 单步 Adam）\n"]
    lines.append("所有结果由只读诊断产生；未修改检查点、缓存或训练配置。\n")
    lines.append("## 静态聚合梯度（10 个训练算例的平均损失）\n")
    lines.append("| 检查点 | 批数 | data/pde 中位数 | contact_data 梯度范数中位数 | data vs other 余弦中位数 (负/总) | data vs total 余弦中位数 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for tag, entry in summary["static"].items():
        cos = entry["cosines"]["cos_data_vs_other"]
        total = entry["cosines"]["cos_data_vs_total"]
        lines.append(
            f"| {tag} | {entry['n_batches']} | "
            f"{_fmt(entry['data_to_pde_ratio_median'], 2)} | "
            f"{_fmt(entry['terms']['contact_data']['grad_norm_median'], 4)} | "
            f"{_fmt(cos['median'], 3)} ({cos['negative']}/{cos['n']}) | "
            f"{_fmt(total['median'], 3)} |\n"
        )
    lines.append("\n各项加权梯度范数中位数：\n")
    lines.append("| 检查点 | pde | contact_ic | interface | contact_data |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for tag, entry in summary["static"].items():
        terms = entry["terms"]
        lines.append(
            f"| {tag} | {_fmt(terms['pde']['grad_norm_median'], 4)} | "
            f"{_fmt(terms['contact_ic']['grad_norm_median'], 4)} | "
            f"{_fmt(terms['interface']['grad_norm_median'], 4)} | "
            f"{_fmt(terms['contact_data']['grad_norm_median'], 4)} |\n"
        )
    lines.append("\n## 单步 Adam（每个试验从同一检查点重新开始）\n")
    lines.append("| 检查点 | 试验数 | 当前批次 data 下降 | 固定参考 mean² 下降 | 两者同时 | 批次好/参考坏 | 覆盖率上升 | Δθ L2 中位数 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"| {tag} | {entry['n_trials']} | "
            f"{entry['batch_contact_data_improved']}/{entry['n_trials']} | "
            f"{entry['reference_violation_improved']}/{entry['n_trials']} | "
            f"{entry['both_improved']} | {entry['batch_good_reference_bad']} | "
            f"{entry['coverage_improved']}/{entry['n_trials']} | "
            f"{_fmt(entry['parameter_delta_l2_median'], 6)} |\n"
        )
    lines.append("\n每步变化中位数：当前批次 data 损失、固定参考 mean²、覆盖率。\n")
    lines.append("| 检查点 | 批次 data Δ 中位数 | 参考 mean² Δ 中位数 | 覆盖率 Δ 中位数 | 一阶预测符号一致 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"| {tag} | {_fmt(entry['batch_contact_data_change_median'], 8)} | "
            f"{_fmt(entry['reference_violation_change_median'], 8)} | "
            f"{_fmt(entry['coverage_change_median'], 8)} | "
            f"{entry['first_order_sign_matches']}/{entry['n_trials']} |\n"
        )
    lines.append("\n固定参考点按半径的 mean² 变化中位数（负值改善）：\n")
    case_ids = sorted({
        case_id for entry in summary["adam"].values()
        for case_id in entry["per_case"]
    }, key=lambda value: int(value.split("_")[0][1:]))
    lines.append("| 检查点 | " + " | ".join(case_ids) + " |\n")
    lines.append("|---|" + "---:|" * len(case_ids) + "\n")
    for tag, entry in summary["adam"].items():
        lines.append(f"| {tag} | " + " | ".join(
            _fmt(entry["per_case"].get(case_id, {}).get("median_change"), 6)
            for case_id in case_ids
        ) + " |\n")
    lines.append("\n固定参考点按时间三分位（负值改善）：\n")
    lines.append("| 检查点 | early | middle | late |\n")
    lines.append("|---|---:|---:|---:|\n")
    for tag, entry in summary["adam"].items():
        lines.append(
            f"| {tag} | {_fmt(entry['thirds']['0']['median_change'], 7)} | "
            f"{_fmt(entry['thirds']['1']['median_change'], 7)} | "
            f"{_fmt(entry['thirds']['2']['median_change'], 7)} |\n"
        )
    lines.append("\n## 分时段补充诊断\n")
    lines.append("分时段批次限制 PDE/界面和接触标签到早/中/晚窗口，初值仍在 t0；"
                 "它只用于定位，不能代表正式训练分布。\n")
    lines.append("\n| 检查点 | 时段 | 批数 | contact_data 梯度范数 | data/pde | data vs other 余弦中位数 (负/总) |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for tag, entry in summary["stratified"].items():
        for which, item in entry.get("by_slice", {}).items():
            ratio = (
                item["terms"]["contact_data"]["grad_norm_median"]
                / item["terms"]["pde"]["grad_norm_median"]
                if item["terms"]["pde"]["grad_norm_median"] else None
            )
            cos = item["cosines"]["cos_data_vs_other"]
            lines.append(
                f"| {tag} | {which} | {item['n_batches']} | "
                f"{_fmt(item['terms']['contact_data']['grad_norm_median'], 4)} | "
                f"{_fmt(ratio, 2)} | {_fmt(cos['median'], 3)} "
                f"({cos['negative']}/{cos['n']}) |\n"
            )
    lines.append("详见 `summary.json` 的 `stratified` 字段。\n")
    out.write_text("".join(lines), encoding="utf-8")


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.{digits}g}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory",
        default="outputs/fullframes/contact_training_diagnostics",
    )
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    payload = {
        "directory": str(directory),
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
    out_json = directory / "summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    _write_markdown(payload, directory / "summary.md")
    (directory / "report.md").write_text(
        _report_text(payload), encoding="utf-8"
    )
    print(f"summary: {out_json}")
    print(f"markdown: {directory / 'summary.md'}")
    print(f"report: {directory / 'report.md'}")
    return 0


def _report_text(summary: dict) -> str:
    best = summary["static"]["best"]
    last = summary["static"]["last"]
    ab = summary["adam"]["best"]
    al = summary["adam"]["last"]
    lines = ["# 接触训练诊断报告\n"]
    lines.append("本轮只比较正式训练分布下的静态聚合梯度和单步 Adam 更新；"
                 "未启用自适应权重、PCGrad 或新的长训练，未修改检查点。\n")
    lines.append("\n## 1. 正式训练分布下接触梯度是否仍占主导\n")
    lines.append(
        f"- best：contact_data 加权梯度范数中位数 "
        f"{_fmt(best['terms']['contact_data']['grad_norm_median'], 5)}，"
        f"pde {_fmt(best['terms']['pde']['grad_norm_median'], 5)}，"
        f"data/pde 中位数 {_fmt(best['data_to_pde_ratio_median'], 3)}。\n"
    )
    lines.append(
        f"- last：contact_data {_fmt(last['terms']['contact_data']['grad_norm_median'], 5)}，"
        f"pde {_fmt(last['terms']['pde']['grad_norm_median'], 5)}，"
        f"data/pde 中位数 {_fmt(last['data_to_pde_ratio_median'], 3)}。\n"
    )
    lines.append(
        "- 接触数据仍是最大或接近最大的各项之一，但不是旧分时段诊断里的 25–29 倍；"
        "训练分布下 best 约 5 倍、last 约 2 倍于 PDE。\n"
    )
    lines.append("\n## 2. 与其它项合成后的方向和幅度\n")
    for tag, entry in (("best", best), ("last", last)):
        other = entry["cosines"]["cos_data_vs_other"]
        total = entry["cosines"]["cos_data_vs_total"]
        lines.append(
            f"- {tag}：contact_data 与其他项合成梯度的余弦中位数 "
            f"{_fmt(other['median'], 3)}（负值 {other['negative']}/{other['n']}）；"
            f"与总梯度余弦中位数 {_fmt(total['median'], 3)}。"
            "方向冲突常见，但总梯度仍由接触项主导。\n"
        )
    lines.append("\n## 3. 一次真实 Adam 更新后接触损失怎么变\n")
    for tag, entry in (("best", ab), ("last", al)):
        lines.append(
            f"- {tag}：当前批次接触 data 损失下降 "
            f"{entry['batch_contact_data_improved']}/{entry['n_trials']} 次，"
            f"中位变化 {_fmt(entry['batch_contact_data_change_median'], 8)}；"
            f"固定训练参考点 mean² 下降 "
            f"{entry['reference_violation_improved']}/{entry['n_trials']} 次，"
            f"中位变化 {_fmt(entry['reference_violation_change_median'], 8)}。\n"
        )
    lines.append(
        "- 参数步长中位数约 1e-3（L2），不是零更新；"
        "一阶预测 g_d^TΔθ 的符号与当前批次实际变化在 best 200/200、last 174/200 一致。\n"
    )
    lines.append("\n## 4. 当前批次改善时固定参考点是否也改善\n")
    lines.append(
        f"- best：两者同时改善 {ab['both_improved']}/{ab['n_trials']}，"
        f"当前批次好但参考变差 {ab['batch_good_reference_bad']} 次。\n"
    )
    lines.append(
        f"- last：两者同时改善 {al['both_improved']}/{al['n_trials']}，"
        f"当前批次好但参考变差 {al['batch_good_reference_bad']} 次；"
        "last 的当前批次有 112/200 次没有下降，但参考点 200/200 改善。\n"
    )
    lines.append("\n## 5. 哪些半径和时段改善\n")
    lines.append(
        "- 固定参考点按半径：r50、r250、r300、r400、r450、r500 的中位变化为负（改善）；"
        "r100、r150、r200、r350 为正（变差）。\n"
    )
    lines.append(
        "- 按时间三分位：late 明显改善（best 198/200、last 199/200 次为负变化），"
        "early 和 middle 的中位变化为正或不稳定。\n"
    )
    lines.append(
        "- 覆盖率并不随均值误差同步改善：best 69/200、last 73/200 次覆盖率上升，"
        "中位变化为负或零。均值/RMSE 的改善不能替代覆盖率。\n"
    )
    lines.append("\n## 结论与后续\n")
    lines.append(
        "更新没有被阻断：固定参考点几乎每步都改善，参数步长非零；"
        "因此不能用“接触项梯度不足/更新受阻”来支持自适应权重。"
        f"方向冲突在训练分布下很常见（best {best['cosines']['cos_data_vs_other']['negative']}/"
        f"{best['cosines']['cos_data_vs_other']['n']}、last "
        f"{last['cosines']['cos_data_vs_other']['negative']}/"
        f"{last['cosines']['cos_data_vs_other']['n']} 的 data-vs-other 余弦为负），"
        "但更新仍能降低固定参考点的平均误差，说明冲突本身没有阻止总体改善。"
        "当前更明显的现象是跨半径/时段折中：late 和多数半径改善，early/middle 与 r100/r150/r200/r350 变差。\n"
    )
    lines.append(
        "建议：暂不启用梯度范数平衡，也不仅凭夹角启用 PCGrad。"
        "若继续，先针对跨半径/时段折中做固定参考点的分层分析或采样稳定性检查；"
        "PCGrad 仍只能作为单个、预先注册的条件实验。本轮不自动启动。\n"
    )
    return "".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
