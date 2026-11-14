#!/usr/bin/env python3
"""Summarise actual-Adam-direction trials on the full-contact B path.

For each checkpoint/seed the source records, per independent trial:

* the four weighted ContactNet gradients and their sum;
* the real Adam parameter change ``delta``;
* ``gC . delta`` and ``gJ . delta``, and the actual ``C``/``J`` changes;
* C/J along ``theta + a*delta`` for ``a = 1, 1/2, 1/4, 1/8``;
* an algebraic Adam numerator decomposition (stored first moment plus the four
  current weighted terms) under the actual second-moment scaling.

The report is generated from the records; no direction or root cause is
hard-coded.  The decomposition is an algebraic split at one step, not a
"remove a loss and rerun Adam" causal experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from summarize_contact_training_diagnostics import (  # noqa: E402
    _finite,
    _majority,
    _require_type,
)

_TERMS = ("pde", "contact_ic", "interface", "contact_data")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _same_path(first, second) -> bool:
    try:
        return Path(first).resolve() == Path(second).resolve()
    except (OSError, ValueError):
        return str(first) == str(second)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}"}


def _correlation(first, second):
    pairs = [
        (float(a), float(b)) for a, b in zip(first, second)
        if _finite(a) and _finite(b)
    ]
    if len(pairs) < 3:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    numerator = sum((a - mx) * (b - my) for a, b in pairs)
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)
    )
    return numerator / denominator if denominator > 0.0 else None


def _stats(values, *, greater_is_better=False):
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return {"n": 0, "median": None, "improved": None,
                "unchanged": None, "worsened": None}
    improved = (
        sum(1 for value in finite if value > 0) if greater_is_better
        else sum(1 for value in finite if value < 0)
    )
    worsened = (
        sum(1 for value in finite if value < 0) if greater_is_better
        else sum(1 for value in finite if value > 0)
    )
    return {
        "n": len(finite),
        "median": statistics.median(finite),
        "max": max(finite),
        "improved": improved,
        "unchanged": len(finite) - improved - worsened,
        "worsened": worsened,
    }


def _sign_counts(values, near_zero=1.0e-12):
    positive = negative = near = missing = 0
    for value in values:
        if not _finite(value):
            missing += 1
        elif abs(float(value)) <= near_zero:
            near += 1
        elif float(value) > 0:
            positive += 1
        else:
            negative += 1
    return {
        "n_evaluable": positive + negative + near,
        "positive": positive,
        "negative": negative,
        "near_zero": near,
        "missing": missing,
    }


def _fmt(value, digits=4):
    return f"{float(value):.{digits}g}" if _finite(value) else "n/a"


def _max_valid(values):
    finite = [float(value) for value in values if _finite(value)]
    return max(finite) if finite else None


def _counts(entry):
    return f"{entry.get('improved')}/{entry.get('unchanged')}/{entry.get('worsened')}"


def _manifest_expectations(manifest: dict):
    issues = []
    if manifest.get("format") != "spherical_blast_pinn.contact_adam_direction.manifest":
        return None, [
            f"unsupported manifest format: {manifest.get('format')!r}"
        ]
    config = manifest.get("config") or {}
    checkpoints = manifest.get("checkpoints") or {}
    seeds = manifest.get("seeds")
    trials = manifest.get("trials_per_checkpoint_seed")
    if not config.get("path"):
        issues.append("manifest config.path is missing")
    if not isinstance(checkpoints, dict) or not checkpoints:
        issues.append("manifest checkpoints are missing")
    if not isinstance(seeds, list) or not seeds:
        issues.append("manifest seeds are missing")
    if not isinstance(trials, int) or trials <= 0:
        issues.append("manifest trials_per_checkpoint_seed must be positive")
    if issues:
        return None, issues
    expected = {}
    for tag, checkpoint in checkpoints.items():
        for seed in seeds:
            name = f"direction_{tag}_seed{seed}.json"
            expected[name] = {
                "file": name,
                "tag": str(tag),
                "seed": int(seed),
                "trials": int(trials),
                "config_path": config.get("path"),
                "config_sha256": config.get("sha256"),
                "checkpoint_path": checkpoint.get("path"),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "checkpoint_loop": checkpoint.get("loop"),
            }
    return {
        "expected": expected,
        "training_cases": manifest.get("training_cases") or [],
    }, issues


def _source_issues(payload: dict, expected: dict, expected_cases: set[str]):
    issues = []
    name = expected["file"]
    if not isinstance(payload, dict) or "_load_error" in payload:
        return [f"{name}: cannot read JSON ({payload.get('_load_error')})"], []
    if payload.get("format") != "spherical_blast_pinn.contact_adam_direction":
        issues.append(f"{name}: wrong source format")
    if payload.get("version") != 1:
        issues.append(f"{name}: unsupported source version")
    for key in ("requested_trials", "completed_trials", "failed_trials"):
        _require_type(payload.get(key), f"{name}.{key}", int, issues)
    if payload.get("ok") is not True:
        issues.append(f"{name}: source ok is not true")
    if payload.get("non_finite"):
        issues.append(f"{name}: non-finite value")
    if not payload.get("checkpoint_sha256", {}).get("unchanged", False):
        issues.append(f"{name}: checkpoint changed during the run")
    if payload.get("seed") != expected["seed"]:
        issues.append(f"{name}: seed does not match the manifest")
    if not _same_path(payload.get("config"), expected["config_path"]):
        issues.append(f"{name}: config does not match the manifest")
    if not _same_path(payload.get("checkpoint"), expected["checkpoint_path"]):
        issues.append(f"{name}: checkpoint does not match the manifest")
    hashes = payload.get("checkpoint_sha256") or {}
    if expected.get("checkpoint_sha256") and (
        hashes.get("before") != expected["checkpoint_sha256"]
        or hashes.get("after") != expected["checkpoint_sha256"]
    ):
        issues.append(f"{name}: checkpoint SHA-256 differs from the manifest")
    checkpoint_path = Path(str(expected["checkpoint_path"]))
    if expected.get("checkpoint_sha256"):
        if checkpoint_path.exists():
            if _sha256(checkpoint_path) != expected["checkpoint_sha256"]:
                issues.append(f"{name}: checkpoint file SHA-256 differs from manifest")
        else:
            issues.append(f"{name}: manifest checkpoint file is missing")
    if expected.get("checkpoint_loop") is not None and (
        payload.get("checkpoint_loop") != expected["checkpoint_loop"]
    ):
        issues.append(f"{name}: checkpoint loop differs from the manifest")
    if expected.get("config_sha256"):
        config_path = Path(str(expected["config_path"]))
        if not config_path.exists():
            issues.append(f"{name}: manifest config file is missing")
        elif _sha256(config_path) != expected["config_sha256"]:
            issues.append(f"{name}: config SHA-256 differs from the manifest")
    label_map = payload.get("labels") or {}
    label_cases = set(label_map)
    if expected_cases and label_cases != expected_cases:
        issues.append(f"{name}: label case set differs from the experiment")
    empty = [
        case for case, item in label_map.items()
        if not isinstance(item, dict) or int(item.get("n", 0)) <= 0
    ]
    if empty:
        issues.append(f"{name}: empty legal label cases: {empty}")
    reference_frames = payload.get("reference_frames_per_trial")
    label_total = sum(
        int(item.get("n", 0)) for item in label_map.values()
        if isinstance(item, dict)
    )
    if not isinstance(reference_frames, int) or reference_frames <= 0:
        issues.append(f"{name}: reference_frames_per_trial is missing")
    elif reference_frames != label_total:
        issues.append(
            f"{name}: reference frames {reference_frames} != full labels "
            f"{label_total}"
        )
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return issues + [f"{name}: trials is not a list"], []
    ids = [trial.get("trial") for trial in trials if isinstance(trial, dict)]
    if len(ids) != len(trials):
        issues.append(f"{name}: trials contains non-object entries")
    if len(set(ids)) != len(ids):
        issues.append(f"{name}: duplicate trial indices")
    requested = payload.get("requested_trials")
    if isinstance(requested, int):
        if requested != expected["trials"]:
            issues.append(f"{name}: requested count differs from the manifest")
        if len(trials) != requested:
            issues.append(f"{name}: record count differs from requested")
        if set(ids) != set(range(requested)):
            issues.append(f"{name}: trial indices do not cover 0..requested-1")
    failed = sum(1 for trial in trials if trial.get("status") == "failed")
    completed = len(trials) - failed
    if payload.get("failed_trials") != failed:
        issues.append(f"{name}: failed_trials metadata differs from records")
    if payload.get("completed_trials") != completed:
        issues.append(f"{name}: completed_trials metadata differs from records")
    valid = []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        local = []
        if trial.get("status") == "failed":
            local.append(f"trial {trial.get('trial')} failed: {trial.get('error')}")
        if trial.get("seed") != expected["seed"] + int(trial.get("trial", 0)) * 104729:
            local.append(f"trial {trial.get('trial')} seed formula differs")
        if trial.get("checkpoint_loop") != payload.get("checkpoint_loop"):
            local.append(f"trial {trial.get('trial')} checkpoint loop differs")
        checks = trial.get("checks") or {}
        for key, value in checks.items():
            if value is not True:
                local.append(f"trial {trial.get('trial')} check {key}=false")
        per_case = trial.get("full_contact_labels_per_case")
        if not isinstance(per_case, dict) or set(per_case) != expected_cases:
            local.append(f"trial {trial.get('trial')} per-case labels differ")
        elif any(int(n) <= 0 for n in per_case.values()):
            local.append(f"trial {trial.get('trial')} has an empty label case")
        for side in ("A", "B"):
            reference = trial.get(side, {}).get("reference_before", {})
            if isinstance(reference_frames, int) and reference.get("n") != reference_frames:
                local.append(
                    f"trial {trial.get('trial')} {side} reference frames differ"
                )
        objectives = trial.get("objectives") or {}
        for key in (
            "contact_before", "total_before", "contact_after", "total_after",
            "contact_change", "total_change", "gC_dot_delta", "gJ_dot_delta",
            "gC_direction", "gJ_direction",
        ):
            if key not in objectives:
                local.append(f"trial {trial.get('trial')} missing objectives.{key}")
        gradients = trial.get("gradients") or {}
        for key in ("term_sum_abs_error", "total_gradient_norm",
                    "contact_gradient_norm", "finite"):
            if key not in gradients:
                local.append(f"trial {trial.get('trial')} missing gradients.{key}")
        if _finite(gradients.get("term_sum_abs_error")) and _finite(
            gradients.get("total_gradient_norm")
        ):
            tolerance = 1.0e-9 * max(1.0, float(gradients["total_gradient_norm"]))
            if float(gradients["term_sum_abs_error"]) > tolerance:
                local.append(
                    f"trial {trial.get('trial')} term gradients do not sum "
                    "to the total gradient"
                )
        if gradients.get("finite") is not True:
            local.append(f"trial {trial.get('trial')} gradients are not finite")
        decomposition = (trial.get("adam") or {}).get("decomposition") or {}
        for key in ("reconstruction_relative_error",
                    "contribution_sum_relative_error",
                    "gC_dot_history", "gC_dot_terms",
                    "term_contribution_sum", "step_values"):
            if key not in decomposition:
                local.append(
                    f"trial {trial.get('trial')} missing decomposition.{key}"
                )
        for key in ("reconstruction_relative_error",
                    "contribution_sum_relative_error"):
            if _finite(decomposition.get(key)) and float(decomposition[key]) > 1.0e-9:
                local.append(
                    f"trial {trial.get('trial')} {key} is too large"
                )
        scaling = trial.get("step_scaling")
        if not isinstance(scaling, list) or len(scaling) != 4:
            local.append(f"trial {trial.get('trial')} step_scaling is incomplete")
        fd = trial.get("finite_difference")
        if not isinstance(fd, list) or len(fd) != 2:
            local.append(f"trial {trial.get('trial')} finite_difference is incomplete")
        if local:
            issues.extend(f"{name}: {item}" for item in local)
        else:
            valid.append(trial)
    return issues, valid


def _summarise(trials):
    def values(path):
        result = []
        for trial in trials:
            current = trial
            for key in path:
                current = current.get(key) if isinstance(current, dict) else None
            result.append(current)
        return result

    dot_c = values(("objectives", "gC_dot_delta"))
    dot_j = values(("objectives", "gJ_dot_delta"))
    change_c = values(("objectives", "contact_change"))
    change_j = values(("objectives", "total_change"))

    def agreement(first, second):
        agree = total = 0
        for a, b in zip(first, second):
            if not (_finite(a) and _finite(b)):
                continue
            total += 1
            if (float(a) < 0) == (float(b) < 0):
                agree += 1
        return {"n": total, "agree": agree}

    scale_summary = {}
    for index, scale in enumerate((1.0, 0.5, 0.25, 0.125)):
        contact_changes = [
            (trial.get("step_scaling") or [])[index].get("contact_change")
            for trial in trials
            if len(trial.get("step_scaling") or []) > index
        ]
        total_changes = [
            (trial.get("step_scaling") or [])[index].get("total_change")
            for trial in trials
            if len(trial.get("step_scaling") or []) > index
        ]
        reversals = 0
        evaluable = 0
        for trial in trials:
            scaling = trial.get("step_scaling") or []
            if len(scaling) <= index:
                continue
            full = scaling[0].get("contact_change")
            current = scaling[index].get("contact_change")
            if _finite(full) and _finite(current):
                evaluable += 1
                if (float(full) < 0) != (float(current) < 0):
                    reversals += 1
        scale_summary[str(scale)] = {
            "contact": _stats(contact_changes),
            "total": _stats(total_changes),
            "contact_direction_reversals": reversals,
            "contact_direction_evaluable": evaluable,
        }

    fd_summary = {}
    for index, h in enumerate((1.0e-3, 1.0e-4)):
        contact_errors = [
            (trial.get("finite_difference") or [])[index].get("contact_relative_error")
            for trial in trials
            if len(trial.get("finite_difference") or []) > index
        ]
        total_errors = [
            (trial.get("finite_difference") or [])[index].get("total_relative_error")
            for trial in trials
            if len(trial.get("finite_difference") or []) > index
        ]
        finite_contact = [float(v) for v in contact_errors if _finite(v)]
        finite_total = [float(v) for v in total_errors if _finite(v)]
        fd_summary[str(h)] = {
            "contact_error_median": (
                statistics.median(finite_contact) if finite_contact else None
            ),
            "contact_error_max": max(finite_contact) if finite_contact else None,
            "total_error_median": (
                statistics.median(finite_total) if finite_total else None
            ),
            "total_error_max": max(finite_total) if finite_total else None,
            "contact_within_5pct": sum(1 for v in finite_contact if v <= 0.05),
            "contact_evaluable": len(finite_contact),
        }

    contribution = {
        "history": _stats(values(("adam", "decomposition", "gC_dot_history"))),
        "reconstruction": _stats(values(
            ("adam", "decomposition", "gC_dot_reconstruction")
        )),
        "term_contribution_sum": _stats(values(
            ("adam", "decomposition", "term_contribution_sum")
        )),
        "reconstruction_relative_error": _stats(values(
            ("adam", "decomposition", "reconstruction_relative_error")
        )),
        "contribution_sum_relative_error": _stats(values(
            ("adam", "decomposition", "contribution_sum_relative_error")
        )),
        "terms": {
            name: _stats([
                (trial.get("adam", {}).get("decomposition", {})
                 .get("gC_dot_terms", {}).get(name))
                for trial in trials
            ])
            for name in _TERMS
        },
        "term_signs": {
            name: _sign_counts([
                (trial.get("adam", {}).get("decomposition", {})
                 .get("gC_dot_terms", {}).get(name))
                for trial in trials
            ])
            for name in _TERMS
        },
    }
    tradeoff = {
        "contact_up_total_down": sum(
            1 for c, j in zip(change_c, change_j)
            if _finite(c) and _finite(j) and float(c) > 0 and float(j) < 0
        ),
        "contact_down_total_up": sum(
            1 for c, j in zip(change_c, change_j)
            if _finite(c) and _finite(j) and float(c) < 0 and float(j) > 0
        ),
        "both_down": sum(
            1 for c, j in zip(change_c, change_j)
            if _finite(c) and _finite(j) and float(c) < 0 and float(j) < 0
        ),
        "both_up": sum(
            1 for c, j in zip(change_c, change_j)
            if _finite(c) and _finite(j) and float(c) > 0 and float(j) > 0
        ),
        "evaluable": sum(
            1 for c, j in zip(change_c, change_j)
            if _finite(c) and _finite(j)
        ),
    }
    check_names = sorted({
        key for trial in trials for key in (trial.get("checks") or {})
    })
    checks_summary = {
        name: {
            "true": sum(
                1 for trial in trials
                if (trial.get("checks") or {}).get(name) is True
            ),
            "false": sum(
                1 for trial in trials
                if (trial.get("checks") or {}).get(name) is False
            ),
            "n": len(trials),
        }
        for name in check_names
    }
    reference_n = values(("A", "reference_before", "n"))
    decomposition_entries = [
        (trial.get("adam") or {}).get("decomposition") or {}
        for trial in trials
    ]

    def unique(key):
        found = set()
        for entry in decomposition_entries:
            value = entry.get(key)
            if isinstance(value, list):
                found.update(str(item) for item in value)
            elif _finite(value):
                found.add(float(value))
            elif value is not None:
                found.add(str(value))
        return sorted(found)

    adam_config = {
        "step_values": unique("step_values"),
        "lr": unique("lr"),
        "betas": unique("betas"),
        "eps": unique("eps"),
        "bias_correction1": unique("bias_correction1_values"),
        "bias_correction2": unique("bias_correction2_values"),
        "denominator_median": _stats([
            entry.get("denominator_median") for entry in decomposition_entries
        ]),
        "history_norm": _stats([
            entry.get("history_norm") for entry in decomposition_entries
        ]),
        "delta_norm": _stats([
            entry.get("delta_norm") for entry in decomposition_entries
        ]),
    }
    return {
        "n_trials": len(trials),
        "reference_n_per_trial": sorted({
            int(value) for value in reference_n if _finite(value)
        }),
        "reference_n_cumulative": int(sum(
            value for value in reference_n if _finite(value)
        )),
        "gC_dot_delta_signs": _sign_counts(dot_c),
        "gJ_dot_delta_signs": _sign_counts(dot_j),
        "contact_change_signs": _sign_counts(change_c),
        "total_change_signs": _sign_counts(change_j),
        "gC_first_order_strength": _stats(dot_c),
        "gJ_first_order_strength": _stats(dot_j),
        "contact_change": _stats(change_c),
        "total_change": _stats(change_j),
        "first_order_agreement_contact": agreement(dot_c, change_c),
        "first_order_agreement_total": agreement(dot_j, change_j),
        "contact_second_order_offset": _stats([
            (float(actual) - float(first))
            for actual, first in zip(change_c, dot_c)
            if _finite(actual) and _finite(first)
        ]),
        "contact_correlation_first_order": _correlation(dot_c, change_c),
        "step_scaling": scale_summary,
        "finite_difference": fd_summary,
        "contribution": contribution,
        "tradeoff": tradeoff,
        "checks": checks_summary,
        "adam_config": adam_config,
        "A_association": {
            "gC_dot_delta_A": _stats(values(("objectives", "gC_dot_delta_A"))),
            "gJ_dot_delta_A": _stats(values(("objectives", "gJ_dot_delta_A"))),
            "A_macro_change": _stats(values(
                ("A", "reference_change", "macro_mse")
            )),
            "A_delta_l2": _stats(values(("A", "parameter_delta_l2"))),
        },
        "runtime": {
            "A_step_s": _stats(values(("A", "time_s"))),
            "B_step_s": _stats(values(("B", "time_s"))),
            "trial_s": _stats(values(("elapsed_s",))),
        },
    }


def _write_report(output, report_path, summary_path):
    lines = ["# 实际 Adam 更新方向分析（全接触 B 路径）"]
    if output["source_issues"]:
        lines.append("")
        lines.append("> **源诊断失败或不完整，以下不能作为正常结论。**")
        for issue in output["source_issues"]:
            lines.append(f"> - {issue}")
    lines.append("")
    lines.append("每个试验从原检查点独立开始，先做相同 AirNet 更新，再记录 "
                 "ContactNet 四项加权梯度、真实 Adam 更新和固定全标签参考目标。"
                 "C 为全部合法训练接触标签的算例等权区间 MSE；J 为同一全标签"
                 "批次上的当前加权总目标。单步诊断，不是连续训练。")
    lines.append("沿实际方向 θ+a·δ 的求值只在内存参数副本上完成；不重新执行 "
                 "Adam、不改变学习率、不改变检查点或优化器状态。")
    lines.append("")
    lines.append("## 运行级验证")
    lines.append("| 检查点 | 种子 | 试验 | 失败条件 |")
    lines.append("|---|---:|---:|---|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                lines.append(f"| {tag} | {seed} | 0 | 不可判断 |")
                continue
            failed = [
                f"{name}({record['false']}/{record['n']})"
                for name, record in summary["checks"].items()
                if record["false"]
            ]
            lines.append(
                f"| {tag} | {seed} | {summary['n_trials']} | "
                + ("无，全部通过" if not failed else "；".join(failed))
                + " |"
            )
    lines.append("")
    lines.append("## 一阶方向与实际变化")
    lines.append("| 检查点 | 种子 | 试验 | gC·δ 正/负/近零 | gC·δ 中位 | "
                 "实际 C 变化中位 | 符号一致 | gJ·δ 正/负/近零 | gJ·δ 中位 | "
                 "实际 J 变化中位 | 符号一致 |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            lines.append(
                f"| {tag} | {seed} | {summary['n_trials']} | "
                f"{_signs(summary['gC_dot_delta_signs'])} | "
                f"{_fmt(summary['gC_first_order_strength']['median'], 7)} | "
                f"{_fmt(summary['contact_change']['median'], 7)} | "
                f"{summary['first_order_agreement_contact']['agree']}/"
                f"{summary['first_order_agreement_contact']['n']} | "
                f"{_signs(summary['gJ_dot_delta_signs'])} | "
                f"{_fmt(summary['gJ_first_order_strength']['median'], 7)} | "
                f"{_fmt(summary['total_change']['median'], 7)} | "
                f"{summary['first_order_agreement_total']['agree']}/"
                f"{summary['first_order_agreement_total']['n']} |"
            )
    lines.append("")
    lines.append("正/负/近零按当前记录计数；近零或缺失不强行判方向。"
                 "“符号一致”只比较一阶内积与实际变化的符号。")
    lines.append("")
    lines.append("| 检查点 | 种子 | 实际 C 变化减一阶预测 中位 | "
                 "一阶/实际 C 变化相关 |")
    lines.append("|---|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            lines.append(
                f"| {tag} | {seed} | "
                f"{_fmt(summary['contact_second_order_offset']['median'], 7)} | "
                f"{_fmt(summary['contact_correlation_first_order'], 5)} |"
            )
    lines.append("")
    lines.append("实际变化减一阶预测的差值主要反映完整一步的二阶/有限步长效应；"
                 "两者高度相关但存在不可忽略截距时，不能只看一阶符号。")
    lines.append("")
    lines.append("## 同方向步长缩放")
    lines.append("| 检查点 | 种子 | a | 接触改善/持平/变差 | 接触变化中位 | "
                 "总目标改善/持平/变差 | 总变化中位 | 相对 a=1 方向反转 |")
    lines.append("|---|---:|---:|---|---:|---|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            for scale, record in summary["step_scaling"].items():
                lines.append(
                    f"| {tag} | {seed} | {scale} | "
                    f"{_counts(record['contact'])} | "
                    f"{_fmt(record['contact']['median'], 7)} | "
                    f"{_counts(record['total'])} | "
                    f"{_fmt(record['total']['median'], 7)} | "
                    f"{record['contact_direction_reversals']}/"
                    f"{record['contact_direction_evaluable']} |"
                )
    lines.append("")
    lines.append("负的接触变化表示 C 下降。缩短步长后若改善/恶化符号改变，"
                 "说明完整一步的非线性或 Adam 历史影响不能由小步长直接代替。")
    lines.append("")
    lines.append("## 实际总目标与接触目标取舍")
    lines.append("| 检查点 | 种子 | C升J降 | C降J升 | 两者降 | 两者升 | 可评估 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            tradeoff = summary["tradeoff"]
            lines.append(
                f"| {tag} | {seed} | {tradeoff['contact_up_total_down']} | "
                f"{tradeoff['contact_down_total_up']} | {tradeoff['both_down']} | "
                f"{tradeoff['both_up']} | {tradeoff['evaluable']} |"
            )
    lines.append("")
    lines.append("## Adam 配置、真实 step 与二阶矩缩放")
    lines.append("| 检查点 | 种子 | step | lr | betas | eps | 偏差修正 1/2 | "
                 "分母中位 | 历史一阶矩范数中位 | δ 范数中位 |")
    lines.append("|---|---:|---|---:|---|---:|---|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            config = summary["adam_config"]
            lines.append(
                f"| {tag} | {seed} | {config['step_values']} | "
                f"{config['lr']} | {config['betas']} | {config['eps']} | "
                f"{config['bias_correction1']}/{config['bias_correction2']} | "
                f"{_fmt(config['denominator_median']['median'], 5)} | "
                f"{_fmt(config['history_norm']['median'], 6)} | "
                f"{_fmt(config['delta_norm']['median'], 6)} |"
            )
    lines.append("")
    lines.append("## Adam 代数分解（固定本次二阶矩缩放）")
    lines.append("| 检查点 | 种子 | gC·历史 | gC·pde | gC·IC | "
                 "gC·interface | gC·contact_data | 贡献和中位 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            contribution = summary["contribution"]
            terms = contribution["terms"]
            lines.append(
                f"| {tag} | {seed} | "
                f"{_fmt(contribution['history']['median'], 7)} | "
                f"{_fmt(terms['pde']['median'], 7)} | "
                f"{_fmt(terms['contact_ic']['median'], 7)} | "
                f"{_fmt(terms['interface']['median'], 7)} | "
                f"{_fmt(terms['contact_data']['median'], 7)} | "
                f"{_fmt(contribution['term_contribution_sum']['median'], 7)} |"
            )
    lines.append("")
    lines.append("| 检查点 | 种子 | 历史项正/负/近零 | pde 正/负/近零 | "
                 "IC 正/负/近零 | interface 正/负/近零 | contact_data 正/负/近零 |")
    lines.append("|---|---:|---|---|---|---|---|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            signs = summary["contribution"]["term_signs"]
            history_signs = _sign_counts_from_stats(
                summary["contribution"]["history"]
            )
            lines.append(
                f"| {tag} | {seed} | {_signs(history_signs)} | "
                f"{_signs(signs['pde'])} | {_signs(signs['contact_ic'])} | "
                f"{_signs(signs['interface'])} | "
                f"{_signs(signs['contact_data'])} |"
            )
    lines.append("")
    lines.append("这是固定本次实际二阶矩分母后的代数拆分：历史一阶矩与当前四项"
                 "加权梯度贡献之和重构真实 δ。它不能解释为“删除某项后重新运行 "
                 "Adam”的因果效应，也不能直接把某个正贡献称为长期根因。")
    lines.append("")
    lines.append("## 重构与有限差分验证误差")
    lines.append("| 检查点 | 种子 | 重构相对误差中位 | 重构最大 | "
                 "贡献和相对误差中位 | h=1e-3 C 中位/5%内 | h=1e-4 C 中位/5%内 |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            contribution = summary["contribution"]
            fd = summary["finite_difference"]
            lines.append(
                f"| {tag} | {seed} | "
                f"{_fmt(contribution['reconstruction_relative_error']['median'], 3)} | "
                f"{_fmt(contribution['reconstruction_relative_error']['max'], 3)} | "
                f"{_fmt(contribution['contribution_sum_relative_error']['median'], 3)} | "
                f"{_fmt(fd['0.001']['contact_error_median'], 3)}/"
                f"{fd['0.001']['contact_within_5pct']}/"
                f"{fd['0.001']['contact_evaluable']} | "
                f"{_fmt(fd['0.0001']['contact_error_median'], 3)}/"
                f"{fd['0.0001']['contact_within_5pct']}/"
                f"{fd['0.0001']['contact_evaluable']} |"
            )
    lines.append("")
    lines.append("有限差分是沿同一 δ 的中心差分，只用于核对一阶内积；"
                 "它不改变检查点、优化器状态或学习率。")
    lines.append("")
    lines.append("## 与正式 A 组的关联")
    lines.append("| 检查点 | 种子 | A 的 macro 变化中位 | A 参数步长中位 | "
                 "gC·δ_A 中位 | gJ·δ_A 中位 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            association = summary["A_association"]
            lines.append(
                f"| {tag} | {seed} | "
                f"{_fmt(association['A_macro_change']['median'], 7)} | "
                f"{_fmt(association['A_delta_l2']['median'], 6)} | "
                f"{_fmt(association['gC_dot_delta_A']['median'], 7)} | "
                f"{_fmt(association['gJ_dot_delta_A']['median'], 7)} |"
            )
    lines.append("")
    lines.append("## 参考帧规模与运行")
    independent = sorted({
        value for entry in output["checkpoints"].values()
        if entry.get("combined")
        for value in entry["combined"].get("reference_n_per_trial", [])
    })
    cumulative = sum(
        entry["combined"].get("reference_n_cumulative", 0)
        for entry in output["checkpoints"].values()
        if entry.get("combined")
    )
    lines.append(
        "每次独立试验的参考帧集合（由实际记录计算）："
        + ("、".join(str(value) for value in independent) if independent
           else "不可判断")
        + f" 帧；当前汇总累计评估点数为 {cumulative}。两者不能混用。"
    )
    lines.append("")
    lines.append("| 检查点 | 种子 | 用时 s | A 单步中位 s | B 单步中位 s |")
    lines.append("|---|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            runtime = summary["runtime"]
            lines.append(
                f"| {tag} | {seed} | {_fmt(runtime['trial_s']['median'], 4)} | "
                f"{_fmt(runtime['A_step_s']['median'], 4)} | "
                f"{_fmt(runtime['B_step_s']['median'], 4)} |"
            )
    lines.append("")
    lines.append("## 判读（由当前记录生成）")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                lines.append(f"- {tag}/{seed}：不可判断（没有可评估试验）。")
                continue
            signs = summary["gC_dot_delta_signs"]
            if signs["n_evaluable"] == 0:
                direction = "不可判断"
            elif signs["positive"] > signs["negative"]:
                direction = "多数一阶预测 C 上升"
            elif signs["negative"] > signs["positive"]:
                direction = "多数一阶预测 C 下降"
            else:
                direction = "正负数量相同"
            full = summary["step_scaling"]["1.0"]["contact"]
            small = summary["step_scaling"]["0.125"]["contact"]
            lines.append(
                f"- {tag}/{seed}：gC·δ 正 {signs['positive']}、负 "
                f"{signs['negative']}、近零 {signs['near_zero']}、缺失 "
                f"{signs['missing']}；{direction}。实际 C 变化中位 "
                f"{_fmt(summary['contact_change']['median'], 5)}，J 变化中位 "
                f"{_fmt(summary['total_change']['median'], 5)}；"
                f"a=1 接触改善/持平/变差 {_counts(full)}，a=1/8 为 "
                f"{_counts(small)}。"
            )
            fd = summary["finite_difference"]
            agreement = summary["first_order_agreement_contact"]
            if agreement["n"] and agreement["agree"] < agreement["n"]:
                offset = summary["contact_second_order_offset"]["median"]
                correlation_value = summary["contact_correlation_first_order"]
                lines.append(
                    f"  - 一阶内积与实际接触变化的符号只一致 "
                    f"{agreement['agree']}/{agreement['n']}；实际变化减一阶"
                    f"预测的中位为 {_fmt(offset, 5)}，两者相关 "
                    f"{_fmt(correlation_value, 3)}。实际 C 方向不能由 gC·δ "
                    "的符号直接代表，完整一步存在明显的有限步长/非线性效应。"
                )
            tradeoff = summary["tradeoff"]
            if tradeoff["contact_up_total_down"]:
                lines.append(
                    f"  - C 上升而总目标下降的试验有 "
                    f"{tradeoff['contact_up_total_down']}/"
                    f"{tradeoff['evaluable']} 次：接触目标与当前总目标在"
                    "本次实际更新上方向相反，存在取舍。"
                )
            reversals = summary["step_scaling"]["0.125"][
                "contact_direction_reversals"
            ]
            evaluable = summary["step_scaling"]["0.125"][
                "contact_direction_evaluable"
            ]
            if reversals:
                lines.append(
                    f"  - 把同方向步长缩到 1/8 后，接触改善/恶化方向相对 "
                    f"a=1 反转 {reversals}/{evaluable} 次；小步长结果与完整"
                    "一步不一致，不能把 a=1 的结果直接外推为方向性质。"
                )
            lines.append(
                f"  - 与有限差分核对：h=1e-3 的 C 相对误差中位 "
                f"{_fmt(fd['0.001']['contact_error_median'], 3)}，"
                f"5% 内 {fd['0.001']['contact_within_5pct']}/"
                f"{fd['0.001']['contact_evaluable']}；Adam 重构相对误差中位 "
                f"{_fmt(summary['contribution']['reconstruction_relative_error']['median'], 3)}。"
            )
            if full["worsened"] == full["n"] and small["improved"]:
                lines.append(
                    f"  - 完整一步接触全部变差，但 1/8 步长已有 "
                    f"{small['improved']}/{small['n']} 次改善；这是“方向"
                    "本身不利还是完整步长过大”需要继续区分的直接依据。"
                )
            elif full["improved"] == full["n"] and small["improved"] == small["n"]:
                lines.append(
                    "  - 完整一步与 1/8 步长都使接触改善，方向在这一缩放"
                    "范围内一致。"
                )
    lines.append(
        "- 以上是固定检查点的独立单步现象；不能直接推广为长期训练根因。"
        "若不同检查点或种子方向不同，应分别保留，不做合并结论。"
    )
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    summary_path.write_text(text, encoding="utf-8")


def _signs(entry):
    return (f"{entry.get('positive', 0)}/{entry.get('negative', 0)}/"
            f"{entry.get('near_zero', 0)}")


def _sign_counts_from_stats(entry):
    return {
        "positive": entry.get("worsened") or 0,
        "negative": entry.get("improved") or 0,
        "near_zero": entry.get("unchanged") or 0,
        "missing": 0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory", default="outputs/fullframes/contact_adam_direction"
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-directory", default=None)
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    out_dir = Path(args.output_directory) if args.output_directory else directory
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest) if args.manifest
        else directory / "direction_manifest.json"
    )
    issues = []
    if not manifest_path.exists():
        issues.append(f"manifest not found: {manifest_path}")
        expectations = None
    else:
        manifest = _load_json(manifest_path)
        if "_load_error" in manifest:
            issues.append(f"manifest cannot be read: {manifest['_load_error']}")
            expectations = None
        else:
            expectations, manifest_issues = _manifest_expectations(manifest)
            issues.extend(manifest_issues)
    files = sorted(directory.glob("direction_*_seed*.json"))
    payloads = {path.name: _load_json(path) for path in files}
    expected_files = expectations["expected"] if expectations else {}
    actual = set(payloads)
    for name in sorted(expected_files):
        if name not in actual:
            issues.append(f"expected file is missing: {name}")
    for name in sorted(actual - set(expected_files)):
        issues.append(f"unexpected direction file: {name}")
    expected_cases = set(expectations.get("training_cases") or []) if expectations else set()
    if not expected_cases:
        for payload in payloads.values():
            labels = payload.get("labels")
            if isinstance(labels, dict):
                expected_cases.update(labels)
    if not expected_cases:
        issues.append("no expected training cases found")

    grouped = {}
    for name, expected in expected_files.items():
        payload = payloads.get(name)
        if payload is None:
            continue
        local, valid = _source_issues(payload, expected, expected_cases)
        issues.extend(local)
        grouped.setdefault(expected["tag"], {}).setdefault(
            expected["seed"], []
        ).extend(valid)

    output = {
        "format": "spherical_blast_pinn.contact_adam_direction_summary",
        "version": 1,
        "directory": str(directory),
        "manifest": str(manifest_path),
        "checkpoints": {},
        "source_issues": list(dict.fromkeys(issues)),
    }
    for tag, seeds in sorted(grouped.items()):
        output["checkpoints"][tag] = {"seeds": {}, "combined": None}
        combined = []
        for seed, trials in sorted(seeds.items()):
            output["checkpoints"][tag]["seeds"][str(seed)] = {
                "trials": len(trials),
                "summary": _summarise(trials) if trials else None,
            }
            combined.extend(trials)
        output["checkpoints"][tag]["combined"] = (
            _summarise(combined) if combined else None
        )
    output["source_issues"] = list(dict.fromkeys(issues))
    output["ok"] = not output["source_issues"]
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_report(output, out_dir / "report.md", out_dir / "summary.md")
    print(f"summary: {summary_path}")
    if output["source_issues"]:
        print("summary: FAILED", file=sys.stderr)
        for issue in output["source_issues"][:50]:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
