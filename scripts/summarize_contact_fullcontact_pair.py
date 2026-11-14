#!/usr/bin/env python3
"""Summarise paired random-contact (A) vs all-contact (B) one-step trials."""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _median(values):
    finite = [float(value) for value in values if _finite(value)]
    return statistics.median(finite) if finite else None


def _stats(values, *, greater_is_better=False):
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return {"n": 0, "median": None, "improved": None, "unchanged": None,
                "worsened": None}
    if greater_is_better:
        improved = sum(1 for value in finite if value > 0)
        worsened = sum(1 for value in finite if value < 0)
    else:
        improved = sum(1 for value in finite if value < 0)
        worsened = sum(1 for value in finite if value > 0)
    return {
        "n": len(finite),
        "median": statistics.median(finite),
        "improved": improved,
        "unchanged": len(finite) - improved - worsened,
        "worsened": worsened,
    }


def _majority(entry):
    n = int(entry.get("n", 0))
    if n == 0:
        return "不可判断"
    improved = int(entry.get("improved") or 0)
    if improved == n:
        return "全部改善"
    if improved == 0:
        return "全部变差"
    if improved * 2 == n:
        return "半数改善"
    if improved * 2 > n:
        return "多数改善"
    return "少数改善（多数变差）"


def _reference_change(trial, arm):
    return trial[arm]["reference_change"]


def _collect(paths):
    trials = []
    issues = []
    files = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files.append({
            "file": path.name,
            "ok": payload.get("ok"),
            "requested": payload.get("requested_trials"),
            "completed": payload.get("completed_trials"),
            "failed": payload.get("failed_trials"),
            "non_finite": payload.get("non_finite"),
            "checkpoint_unchanged": payload.get("checkpoint_sha256", {}).get("unchanged"),
            "checkpoint_loop": payload.get("checkpoint_loop"),
            "full_contact_labels": payload.get("full_contact_labels"),
            "labels": payload.get("labels"),
            "elapsed_s": payload.get("elapsed_s"),
            "rss_mb": payload.get("rss_mb"),
        })
        if payload.get("ok") is not True:
            issues.append(f"{path.name}: source ok is not true")
        if payload.get("failed_trials"):
            issues.append(f"{path.name}: failed trials")
        if payload.get("non_finite"):
            issues.append(f"{path.name}: non-finite value")
        if not payload.get("checkpoint_sha256", {}).get("unchanged", True):
            issues.append(f"{path.name}: checkpoint changed")
        if payload.get("full_contact_labels") != payload.get("reference_frames_per_trial"):
            issues.append(f"{path.name}: full labels != reference frames")
        for trial in payload.get("trials", []):
            if trial.get("status") == "failed":
                issues.append(f"{path.name}: trial {trial.get('trial')} failed")
                continue
            required = ("A", "B", "paired")
            if any(key not in trial for key in required):
                issues.append(f"{path.name}: trial missing arm data")
                continue
            for arm in ("A", "B"):
                for key in ("reference_change", "reference_before",
                            "reference_after", "full_labels_before",
                            "full_labels_after"):
                    if key not in trial[arm]:
                        issues.append(f"{path.name}: trial missing {arm}.{key}")
            paired = trial.get("paired", {})
            for flag in ("initial_param_equal", "initial_optimizer_equal",
                         "pre_contact_param_equal",
                         "pre_contact_optimizer_equal", "air_state_equal",
                         "air_loss_equal"):
                if not paired.get(flag, False):
                    issues.append(f"{path.name}: trial {trial.get('trial')} {flag}=false")
            if not trial.get("non_contact_sampling_equal", False):
                issues.append(f"{path.name}: trial {trial.get('trial')} non-contact sampling differs")
            trials.append(trial)
    return trials, files, issues


def _summarise(trials):
    def values(arm, path):
        result = []
        for trial in trials:
            current = trial
            for key in path:
                current = current.get(key, {}) if isinstance(current, dict) else {}
            result.append(current)
        return result

    a_macro = [t["A"]["reference_change"]["macro_mse"] for t in trials]
    b_macro = [t["B"]["reference_change"]["macro_mse"] for t in trials]
    a_pooled = [t["A"]["reference_change"]["pooled_mse"] for t in trials]
    b_pooled = [t["B"]["reference_change"]["pooled_mse"] for t in trials]
    a_cov = [t["A"]["reference_change"]["coverage"] for t in trials]
    b_cov = [t["B"]["reference_change"]["coverage"] for t in trials]
    paired_macro = [t["paired"]["reference_macro_mse_difference_B_minus_A"] for t in trials]
    paired_pooled = [t["paired"]["reference_pooled_mse_difference_B_minus_A"] for t in trials]
    paired_cov = [t["paired"]["reference_coverage_difference_B_minus_A"] for t in trials]
    paired_full = [t["paired"]["full_label_macro_mse_difference_B_minus_A"] for t in trials]
    paired_common = [t["paired"]["common_A_macro_mse_difference_B_minus_A"] for t in trials]
    cases = {}
    for trial in trials:
        after_a = {item["case_id"]: item for item in
                   trial["A"]["reference_after"]["cases"]}
        after_b = {item["case_id"]: item for item in
                   trial["B"]["reference_after"]["cases"]}
        before_a = {item["case_id"]: item for item in
                    trial["A"]["reference_before"]["cases"]}
        before_b = {item["case_id"]: item for item in
                    trial["B"]["reference_before"]["cases"]}
        for case_id in before_a:
            if case_id not in before_b or case_id not in after_a or case_id not in after_b:
                continue
            cases.setdefault(case_id, {"A": [], "B": [], "diff": []})
            if all(_finite(x) for x in (
                before_a[case_id].get("mse"), after_a[case_id].get("mse"),
                before_b[case_id].get("mse"), after_b[case_id].get("mse"),
            )):
                da = after_a[case_id]["mse"] - before_a[case_id]["mse"]
                db = after_b[case_id]["mse"] - before_b[case_id]["mse"]
                cases[case_id]["A"].append(da)
                cases[case_id]["B"].append(db)
                cases[case_id]["diff"].append(db - da)
    per_case = {
        case_id: {
            "A": _stats(values["A"], greater_is_better=False),
            "B": _stats(values["B"], greater_is_better=False),
            "B_minus_A": _stats(values["diff"], greater_is_better=False),
        }
        for case_id, values in cases.items()
    }
    thirds = {}
    for which in range(3):
        a_vals, b_vals, diffs = [], [], []
        for trial in trials:
            ba = trial["A"]["reference_before"].get("thirds", [])
            aa = trial["A"]["reference_after"].get("thirds", [])
            bb = trial["B"]["reference_before"].get("thirds", [])
            ab = trial["B"]["reference_after"].get("thirds", [])
            if min(len(ba), len(aa), len(bb), len(ab)) <= which:
                continue
            va = aa[which].get("macro_mse") - ba[which].get("macro_mse") \
                if _finite(aa[which].get("macro_mse")) and _finite(ba[which].get("macro_mse")) else None
            vb = ab[which].get("macro_mse") - bb[which].get("macro_mse") \
                if _finite(ab[which].get("macro_mse")) and _finite(bb[which].get("macro_mse")) else None
            a_vals.append(va)
            b_vals.append(vb)
            diffs.append(vb - va if va is not None and vb is not None else None)
        thirds[str(which)] = {
            "A": _stats(a_vals), "B": _stats(b_vals),
            "B_minus_A": _stats(diffs),
        }
    def terms(arm, phase):
        result = {}
        for name in ("pde", "contact_ic", "interface", "contact_data"):
            result[name] = _median([
                t[arm][phase]["terms"][name]["weighted_loss"] for t in trials
            ])
        result["total"] = _median([
            sum(t[arm][phase]["terms"][name]["weighted_loss"]
                for name in ("pde", "contact_ic", "interface", "contact_data"))
            for t in trials
        ])
        return result
    def coverage(arm):
        before = [t[arm]["reference_before"].get("covered_count") for t in trials]
        after = [t[arm]["reference_after"].get("covered_count") for t in trials]
        n = [t[arm]["reference_after"].get("n") for t in trials]
        return {
            "covered_before": int(sum(v for v in before if _finite(v))),
            "covered_after": int(sum(v for v in after if _finite(v))),
            "n": int(sum(v for v in n if _finite(v))),
        }
    return {
        "n_trials": len(trials),
        "A_terms_before": terms("A", "before"),
        "A_terms_after": terms("A", "after"),
        "B_terms_before": terms("B", "before"),
        "B_terms_after": terms("B", "after"),
        "A_coverage_counts": coverage("A"),
        "B_coverage_counts": coverage("B"),
        "A": {
            "macro": _stats(a_macro), "pooled": _stats(a_pooled),
            "coverage": _stats(a_cov, greater_is_better=True),
        },
        "B": {
            "macro": _stats(b_macro), "pooled": _stats(b_pooled),
            "coverage": _stats(b_cov, greater_is_better=True),
        },
        "paired": {
            "macro_B_minus_A": _stats(paired_macro),
            "pooled_B_minus_A": _stats(paired_pooled),
            "coverage_B_minus_A": _stats(paired_cov, greater_is_better=True),
            "full_label_macro_B_minus_A": _stats(paired_full),
            "common_A_macro_B_minus_A": _stats(paired_common),
        },
        "per_case": per_case,
        "thirds": thirds,
        "A_current_contact_data_change": _stats([
            t["A"]["after"]["terms"]["contact_data"]["weighted_loss"]
            - t["A"]["before"]["terms"]["contact_data"]["weighted_loss"]
            for t in trials
        ]),
        "B_current_contact_data_change": _stats([
            t["B"]["after"]["terms"]["contact_data"]["weighted_loss"]
            - t["B"]["before"]["terms"]["contact_data"]["weighted_loss"]
            for t in trials
        ]),
        "A_parameter_delta_l2": _stats([t["A"]["parameter_delta_l2"] for t in trials]),
        "B_parameter_delta_l2": _stats([t["B"]["parameter_delta_l2"] for t in trials]),
        "A_time_s": _stats([t["A"]["time_s"] for t in trials]),
        "B_time_s": _stats([t["B"]["time_s"] for t in trials]),
        "trial_time_s": _stats([t.get("elapsed_s") for t in trials]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory", default="outputs/fullframes/contact_fullcontact_pair"
    )
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    files = sorted(directory.glob("pair_*_seed*.json"))
    if not files:
        print("no paired experiment files found", file=sys.stderr)
        return 1
    grouped = {}
    all_issues = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tag = str(payload.get("tag", "unknown"))
        seed = int(payload.get("seed", 0))
        grouped.setdefault(tag, {}).setdefault(seed, []).append(path)
    output = {
        "format": "spherical_blast_pinn.contact_fullcontact_pair_summary",
        "version": 1,
        "directory": str(directory),
        "checkpoints": {},
        "source_issues": [],
    }
    for tag, seeds in grouped.items():
        output["checkpoints"][tag] = {"seeds": {}, "combined": None}
        combined_trials = []
        for seed, paths in sorted(seeds.items()):
            trials, files_info, issues = _collect(paths)
            all_issues.extend(issues)
            output["checkpoints"][tag]["seeds"][str(seed)] = {
                "files": files_info,
                "trials": len(trials),
                "summary": _summarise(trials) if trials else None,
                "source_issues": issues,
            }
            combined_trials.extend(trials)
        output["checkpoints"][tag]["combined"] = (
            _summarise(combined_trials) if combined_trials else None
        )
    output["source_issues"] = list(dict.fromkeys(all_issues))
    output["ok"] = not output["source_issues"]
    out = directory / "summary.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_report(output, directory / "report.md", directory / "summary.md")
    print(f"summary: {out}")
    if output["source_issues"]:
        print("summary: FAILED", file=sys.stderr)
        for issue in output["source_issues"][:50]:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    return 0


def _fmt(value, digits=4):
    if not _finite(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def _write_report(output, report_path, summary_path):
    lines = ["# 随机接触抽样 (A) 与全接触监督 (B) 配对单步实验"]
    if output["source_issues"]:
        lines.append("")
        lines.append("> **源诊断失败或不完整，以下不能作为正常结论。**")
        for issue in output["source_issues"]:
            lines.append(f"> - {issue}")
    lines.append("")
    lines.append("每一对 A/B 从同一检查点、同一非接触批次和同一 AirNet 更新开始；"
                 "只替换 contact_data 的标签集合。单步独立实验，不是连续训练。")
    lines.append("每个独立参考帧集合为 7,165 帧；跨试验累计评估点数为 "
                 "n_trials × 7,165。A 的随机批次损失与 B 的全标签损失不是同一"
                 "集合，不能直接比较数值；共同 A 标签与全标签评估分别列出。")
    lines.append("")
    lines.append("| 检查点 | 种子 | A macro 中位 | B macro 中位 | B-A macro 中位 | "
                 "A pooled 中位 | B pooled 中位 | B-A pooled 中位 | B-A 覆盖率中位 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            lines.append(
                f"| {tag} | {seed} | {_fmt(summary['A']['macro']['median'], 8)} | "
                f"{_fmt(summary['B']['macro']['median'], 8)} | "
                f"{_fmt(summary['paired']['macro_B_minus_A']['median'], 8)} | "
                f"{_fmt(summary['A']['pooled']['median'], 8)} | "
                f"{_fmt(summary['B']['pooled']['median'], 8)} | "
                f"{_fmt(summary['paired']['pooled_B_minus_A']['median'], 8)} | "
                f"{_fmt(summary['paired']['coverage_B_minus_A']['median'], 8)} |"
            )
        combined = entry["combined"]
        if combined:
            lines.append(
                f"| {tag} | combined | {_fmt(combined['A']['macro']['median'], 8)} | "
                f"{_fmt(combined['B']['macro']['median'], 8)} | "
                f"{_fmt(combined['paired']['macro_B_minus_A']['median'], 8)} | "
                f"{_fmt(combined['A']['pooled']['median'], 8)} | "
                f"{_fmt(combined['B']['pooled']['median'], 8)} | "
                f"{_fmt(combined['paired']['pooled_B_minus_A']['median'], 8)} | "
                f"{_fmt(combined['paired']['coverage_B_minus_A']['median'], 8)} |"
            )
    lines.append("")
    lines.append("改善次数（B-A 负值表示 B 更有利）：")
    lines.append("| 检查点 | 种子 | A macro | B macro | B 更优 macro | "
                 "A pooled | B pooled | B 更优 pooled |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            lines.append(
                f"| {tag} | {seed} | {_majority(summary['A']['macro'])} | "
                f"{_majority(summary['B']['macro'])} | "
                f"{_majority(summary['paired']['macro_B_minus_A'])} | "
                f"{_majority(summary['A']['pooled'])} | "
                f"{_majority(summary['B']['pooled'])} | "
                f"{_majority(summary['paired']['pooled_B_minus_A'])} |"
            )
    lines.append("")
    lines.append("## 固定参考点：按半径（combined，A/B 变化中位数及 B-A）")
    lines.append("| 检查点 | 半径 | A | B | B-A |")
    lines.append("|---|---|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        summary = entry["combined"]
        if not summary:
            continue
        for case_id, item in sorted(
            summary["per_case"].items(),
            key=lambda pair: int(pair[0].split("_")[0][1:]),
        ):
            lines.append(
                f"| {tag} | {case_id} | {_fmt(item['A']['median'], 7)} | "
                f"{_fmt(item['B']['median'], 7)} | "
                f"{_fmt(item['B_minus_A']['median'], 7)} |"
            )
    lines.append("")
    lines.append("## 固定参考点：早/中/晚（combined）")
    lines.append("| 检查点 | 时段 | A | B | B-A |")
    lines.append("|---|---|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        summary = entry["combined"]
        if not summary:
            continue
        for which, item in summary["thirds"].items():
            lines.append(
                f"| {tag} | {which} | {_fmt(item['A']['median'], 7)} | "
                f"{_fmt(item['B']['median'], 7)} | "
                f"{_fmt(item['B_minus_A']['median'], 7)} |"
            )
    lines.append("")
    lines.append("## 当前目标分项（加权损失中位数，A/B 各自标签集合）")
    lines.append("| 检查点 | 组 | 阶段 | pde | IC | interface | contact_data | total |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        summary = entry["combined"]
        if not summary:
            continue
        for arm in ("A", "B"):
            for phase in ("before", "after"):
                terms = summary[f"{arm}_terms_{phase}"]
                lines.append(
                    f"| {tag} | {arm} | {phase} | {_fmt(terms['pde'], 5)} | "
                    f"{_fmt(terms['contact_ic'], 5)} | "
                    f"{_fmt(terms['interface'], 5)} | "
                    f"{_fmt(terms['contact_data'], 5)} | "
                    f"{_fmt(terms['total'], 5)} |"
                )
    lines.append("")
    lines.append("## 精确覆盖率与配对差")
    lines.append("| 检查点 | 组 | covered before | covered after | n |")
    lines.append("|---|---|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        summary = entry["combined"]
        if not summary:
            continue
        for arm in ("A", "B"):
            counts = summary[f"{arm}_coverage_counts"]
            lines.append(
                f"| {tag} | {arm} | {counts['covered_before']} | "
                f"{counts['covered_after']} | {counts['n']} |"
            )
    lines.append("")
    lines.append("配对 B-A：macro/pooled/全标签 macro/共同 A macro 的负值表示 B "
                 "更有利；覆盖率的正值表示 B 更有利。")
    lines.append("| 检查点 | 指标 | 中位 | 改善/持平/变差 |")
    lines.append("|---|---|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        summary = entry["combined"]
        if not summary:
            continue
        for key in ("macro_B_minus_A", "pooled_B_minus_A",
                    "coverage_B_minus_A", "full_label_macro_B_minus_A",
                    "common_A_macro_B_minus_A"):
            item = summary["paired"][key]
            lines.append(
                f"| {tag} | {key} | {_fmt(item['median'], 7)} | "
                f"{item['improved']}/{item['unchanged']}/{item['worsened']} |"
            )
    lines.append("")
    lines.append("## 运行时间与内存（文件级）")
    lines.append("| 检查点 | 种子 | 用时 s | RSS MB | A/B 单步时间中位 s |")
    lines.append("|---|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            for file in item["files"]:
                summary = item["summary"]
                times = ("n/a" if not summary else
                         f"{_fmt(summary['A_time_s']['median'], 3)}/"
                         f"{_fmt(summary['B_time_s']['median'], 3)}")
                lines.append(
                    f"| {tag} | {seed} | {_fmt(file.get('elapsed_s'), 4)} | "
                    f"{_fmt(file.get('rss_mb'), 4)} | {times} |"
                )
    lines.append("")
    lines.append("")
    lines.append("## 全接触标签与排除原因（每个训练算例）")
    lines.append("| 检查点 | 算例 | n | out_of_time | no_slope | candidates |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        first = next(
            (file for item in entry["seeds"].values() for file in item["files"]
             if file.get("labels")),
            None,
        )
        if not first:
            continue
        for case_id, item in sorted(
            first["labels"].items(),
            key=lambda pair: int(pair[0].split("_")[0][1:]),
        ):
            lines.append(
                f"| {tag} | {case_id} | {item.get('n')} | "
                f"{item.get('excluded_outside_time')} | "
                f"{item.get('excluded_no_slope')} | "
                f"{item.get('total_frame_candidates')} |"
            )
    lines.append("")
    lines.append("## 判读（由当前输入生成）")
    stable = True
    for tag, entry in output["checkpoints"].items():
        seed_items = [
            item["summary"]["paired"]["macro_B_minus_A"]
            for item in entry["seeds"].values()
            if item["summary"] is not None
        ]
        if not seed_items:
            lines.append(f"- {tag}：不可判断（没有可评估的配对结果）。")
            stable = False
            continue
        better = sum(item["improved"] or 0 for item in seed_items)
        worse = sum(item["worsened"] or 0 for item in seed_items)
        medians = [item["median"] for item in seed_items if _finite(item["median"])]
        sign = "B 更有利" if medians and all(value < 0 for value in medians) else (
            "A 更有利" if medians and all(value > 0 for value in medians)
            else "方向不一致"
        )
        lines.append(
            f"- {tag}：B-A macro 各种子中位 "
            + "、".join(_fmt(value, 4) for value in medians)
            + f"；B 更优 {better} 次，A 更优 {worse} 次；{sign}。"
        )
        if not medians or not all(value < 0 for value in medians):
            stable = False
        if any(
            item["summary"]["paired"]["common_A_macro_B_minus_A"]["median"] is not None
            and item["summary"]["paired"]["common_A_macro_B_minus_A"]["median"] > 0
            for item in entry["seeds"].values()
            if item["summary"] is not None
        ):
            stable = False
    lines.append(
        "- 两个检查点/两个种子在相同方向上稳定更优："
        + ("是" if stable else "否")
        + "。这只能说明接触监督随机抽样是否为一个影响因素；"
        "不能证明长期收敛或误差根因。"
    )
    for tag, entry in output["checkpoints"].items():
        common = [
            item["summary"]["paired"]["common_A_macro_B_minus_A"]["median"]
            for item in entry["seeds"].values()
            if item["summary"] is not None
        ]
        if common:
            lines.append(
                f"- {tag} 在保留的 A 抽样标签共同集上，B-A macro 中位为 "
                + "、".join(_fmt(value, 4) for value in common)
                + "；该集合与全标签训练损失不是同一目标，不能直接比较数值。"
            )
    text = chr(10).join(lines) + chr(10)
    report_path.write_text(text, encoding="utf-8")
    summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
