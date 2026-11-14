#!/usr/bin/env python3
"""Summarise paired random-contact (A) vs all-contact (B) one-step trials.

The source ``ok`` flag is not trusted.  An experiment manifest (the existing
``experiment_config.json`` or a ``pair_manifest.json``) supplies the expected
files, checkpoints, configuration, seeds and trial counts.  Missing files,
duplicate trial indices, mixed checkpoints/configurations and failed runtime
pairing conditions are reported as source issues and make the command return
non-zero.
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

# Reuse the same finite/majority rules as the training-diagnostic summary so
# the two reports cannot drift apart.
from summarize_contact_training_diagnostics import (  # noqa: E402
    _finite,
    _majority,
    _require_type,
)


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


def _manifest_expectations(manifest: dict) -> tuple[dict | None, list[str]]:
    """Return expected files/checkpoints/config from either manifest schema."""
    issues: list[str] = []
    fmt = manifest.get("format")
    config = manifest.get("config") or {}
    checkpoints = manifest.get("checkpoints") or {}
    if fmt == "spherical_blast_pinn.contact_fullcontact_pair.config":
        design = manifest.get("design") or {}
        seeds = design.get("seeds")
        trials = design.get("trials_per_checkpoint_seed")
    elif fmt == "spherical_blast_pinn.contact_fullcontact_pair.manifest":
        seeds = manifest.get("seeds")
        trials = manifest.get("trials_per_checkpoint_seed")
    else:
        return None, [f"unsupported manifest format: {fmt!r}"]
    if not isinstance(config, dict) or not config.get("path"):
        issues.append("manifest config.path is missing")
    if not isinstance(checkpoints, dict) or not checkpoints:
        issues.append("manifest checkpoints are missing")
    if not isinstance(seeds, list) or not seeds:
        issues.append("manifest seeds are missing")
    if not isinstance(trials, int) or trials <= 0:
        issues.append("manifest trials_per_checkpoint_seed must be a positive int")
    if issues:
        return None, issues
    expected = {}
    manifest_reference_frames = manifest.get("reference_frames_per_trial")
    for tag, checkpoint in checkpoints.items():
        if not isinstance(checkpoint, dict) or not checkpoint.get("path"):
            issues.append(f"manifest checkpoint {tag!r} has no path")
            continue
        for seed in seeds:
            name = f"pair_{tag}_seed{seed}.json"
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
                "reference_frames_per_trial": manifest_reference_frames,
            }
    explicit = manifest.get("expected_files")
    if explicit is not None:
        if not isinstance(explicit, list):
            issues.append("manifest expected_files must be a list")
        else:
            expected = {}
            for item in explicit:
                if not isinstance(item, dict) or not item.get("file"):
                    issues.append("manifest expected_files entry has no file")
                    continue
                file = str(item["file"])
                if file in expected:
                    issues.append(f"manifest lists {file} more than once")
                expected[file] = {
                    **item,
                    "file": file,
                    "tag": str(item.get("tag", "")),
                    "seed": int(item.get("seed", 0)),
                    "trials": int(item.get("trials", trials)),
                    "config_path": item.get("config_path", config.get("path")),
                    "config_sha256": item.get(
                        "config_sha256", config.get("sha256")
                    ),
                    "checkpoint_path": item.get(
                        "checkpoint_path",
                        checkpoints.get(item.get("tag"), {}).get("path"),
                    ),
                    "checkpoint_sha256": item.get(
                        "checkpoint_sha256",
                        checkpoints.get(item.get("tag"), {}).get("sha256"),
                    ),
                    "checkpoint_loop": item.get(
                        "checkpoint_loop",
                        checkpoints.get(item.get("tag"), {}).get("loop"),
                    ),
                    "reference_frames_per_trial": item.get(
                        "reference_frames_per_trial", manifest_reference_frames
                    ),
                }
    return {
        "expected": expected,
        "config": config,
        "checkpoints": checkpoints,
        "seeds": [int(seed) for seed in seeds],
        "trials": int(trials),
        "training_cases": manifest.get("training_cases") or [],
    }, issues


def _record_issues(trial: dict, expected_cases: set[str]) -> list[str]:
    issues: list[str] = []
    if trial.get("status") == "failed":
        return [f"trial {trial.get('trial')} failed: {trial.get('error')}"]
    for key in ("A", "B", "paired"):
        if key not in trial:
            issues.append(f"trial {trial.get('trial')} missing {key}")
    if issues:
        return issues
    for arm in ("A", "B"):
        for key in ("reference_change", "reference_before", "reference_after",
                    "full_labels_before", "full_labels_after"):
            if key not in trial[arm]:
                issues.append(f"trial {trial.get('trial')} missing {arm}.{key}")
    paired = trial.get("paired") or {}
    for flag in ("initial_param_equal", "initial_optimizer_equal",
                 "pre_contact_param_equal", "pre_contact_optimizer_equal",
                 "air_state_equal", "air_loss_equal", "initial_state_equal"):
        if paired.get(flag) is not True:
            issues.append(
                f"trial {trial.get('trial')} pairing condition {flag}=false"
            )
    if trial.get("non_contact_sampling_equal") is not True:
        issues.append(f"trial {trial.get('trial')} non-contact sampling differs")
    per_case = trial.get("full_contact_labels_per_case")
    if not isinstance(per_case, dict):
        issues.append(f"trial {trial.get('trial')} has no per-case label counts")
    else:
        if set(per_case) != expected_cases:
            issues.append(
                f"trial {trial.get('trial')} label case set differs from expected"
            )
        empty = [case for case, n in per_case.items() if int(n) <= 0]
        if empty:
            issues.append(f"trial {trial.get('trial')} empty label cases: {empty}")
    for side in ("reference_before", "reference_after"):
        reference = trial.get("A", {}).get(side, {})
        if reference.get("complete_training_target") is not True:
            issues.append(f"trial {trial.get('trial')} A.{side} incomplete")
        case_ids = {item.get("case_id") for item in reference.get("cases", [])}
        if expected_cases and case_ids != expected_cases:
            issues.append(f"trial {trial.get('trial')} A.{side} case set differs")
    return issues


def _validate_source(payload: dict, expected: dict, expected_cases: set[str]) -> tuple[list[dict], dict, list[str]]:
    issues: list[str] = []
    name = expected["file"]
    if not isinstance(payload, dict) or "_load_error" in payload:
        return [], {"file": name}, [
            f"{name}: cannot read JSON ({payload.get('_load_error')})"
        ]
    if payload.get("format") != "spherical_blast_pinn.contact_fullcontact_pair":
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
        issues.append(
            f"{name}: seed {payload.get('seed')!r} != manifest {expected['seed']!r}"
        )
    if not _same_path(payload.get("config"), expected["config_path"]):
        issues.append(
            f"{name}: config {payload.get('config')!r} != manifest "
            f"{expected['config_path']!r}"
        )
    if not _same_path(payload.get("checkpoint"), expected["checkpoint_path"]):
        issues.append(
            f"{name}: checkpoint {payload.get('checkpoint')!r} != manifest "
            f"{expected['checkpoint_path']!r}"
        )
    if expected.get("config_sha256"):
        config_path = Path(str(expected["config_path"]))
        if config_path.exists():
            actual = _sha256(config_path)
            if actual != expected["config_sha256"]:
                issues.append(f"{name}: config SHA-256 differs from manifest")
        else:
            issues.append(f"{name}: manifest config file is missing")
    hashes = payload.get("checkpoint_sha256") or {}
    if expected.get("checkpoint_sha256") and (
        hashes.get("before") != expected["checkpoint_sha256"]
        or hashes.get("after") != expected["checkpoint_sha256"]
    ):
        issues.append(f"{name}: checkpoint SHA-256 differs from manifest")
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
        issues.append(
            f"{name}: checkpoint loop {payload.get('checkpoint_loop')!r} != "
            f"manifest {expected['checkpoint_loop']!r}"
        )
    trials = payload.get("trials")
    if not isinstance(trials, list):
        issues.append(f"{name}: trials is not a list")
        return [], {"file": name}, issues
    ids = [trial.get("trial") for trial in trials if isinstance(trial, dict)]
    if len(ids) != len(trials):
        issues.append(f"{name}: trials contains non-object entries")
    if len(set(ids)) != len(ids):
        issues.append(f"{name}: duplicate trial indices")
    requested = payload.get("requested_trials")
    if isinstance(requested, int):
        if requested != expected["trials"]:
            issues.append(
                f"{name}: requested {requested} != manifest {expected['trials']}"
            )
        if len(trials) != requested:
            issues.append(
                f"{name}: record count {len(trials)} != requested {requested}"
            )
        if set(ids) != set(range(requested)):
            issues.append(f"{name}: trial indices do not cover 0..requested-1")
    failed = sum(1 for trial in trials if trial.get("status") == "failed")
    completed = len(trials) - failed
    if payload.get("failed_trials") != failed:
        issues.append(
            f"{name}: failed_trials {payload.get('failed_trials')!r} != records {failed}"
        )
    if payload.get("completed_trials") != completed:
        issues.append(
            f"{name}: completed_trials {payload.get('completed_trials')!r} != "
            f"records {completed}"
        )
    label_map = payload.get("labels")
    per_case = payload.get("full_contact_labels_per_case")
    if not isinstance(per_case, dict) and isinstance(label_map, dict):
        per_case = {
            case_id: item.get("n")
            for case_id, item in label_map.items()
            if isinstance(item, dict)
        }
    expected_cases_from_source = set(label_map) if isinstance(label_map, dict) else set()
    if expected_cases_from_source and expected_cases and (
        expected_cases_from_source != expected_cases
    ):
        issues.append(f"{name}: label case set differs from the experiment set")
    if isinstance(per_case, dict):
        if set(per_case) != expected_cases:
            issues.append(f"{name}: per-case label case set differs")
        empty = [case for case, n in per_case.items() if int(n) <= 0]
        if empty:
            issues.append(f"{name}: empty label cases: {empty}")
        label_total = sum(int(n) for n in per_case.values())
        if payload.get("full_contact_labels") != label_total:
            issues.append(f"{name}: full_contact_labels != per-case sum")
    else:
        issues.append(f"{name}: full_contact_labels_per_case is missing")
        label_total = payload.get("full_contact_labels")
    reference_frames = payload.get("reference_frames_per_trial")
    if not isinstance(reference_frames, int) or reference_frames <= 0:
        issues.append(f"{name}: reference_frames_per_trial is missing")
    elif label_total is not None and reference_frames != label_total:
        issues.append(
            f"{name}: reference frames {reference_frames} != full labels {label_total}"
        )
    expected_frames = expected.get("reference_frames_per_trial")
    if isinstance(expected_frames, int) and reference_frames != expected_frames:
        issues.append(
            f"{name}: reference frames {reference_frames} != manifest "
            f"{expected_frames}"
        )
    valid: list[dict] = []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        local = _record_issues(trial, expected_cases)
        if isinstance(reference_frames, int) and isinstance(
            trial.get("A", {}).get("reference_before"), dict
        ):
            if trial["A"]["reference_before"].get("n") != reference_frames:
                local.append(
                    f"trial {trial.get('trial')} reference n != "
                    f"{reference_frames}"
                )
        if trial.get("checkpoint_loop") != payload.get("checkpoint_loop"):
            local.append(
                f"trial {trial.get('trial')} checkpoint loop differs from source"
            )
        if expected.get("checkpoint_loop") is not None and (
            trial.get("checkpoint_loop") != expected["checkpoint_loop"]
        ):
            local.append(f"trial {trial.get('trial')} checkpoint loop differs")
        if trial.get("seed") != expected["seed"] + int(trial.get("trial", 0)) * 104729:
            local.append(f"trial {trial.get('trial')} seed formula differs")
        if local:
            issues.extend(f"{name}: {item}" for item in local)
        else:
            valid.append(trial)
    info = {
        "file": name,
        "ok": payload.get("ok"),
        "requested": requested,
        "completed": payload.get("completed_trials"),
        "failed": payload.get("failed_trials"),
        "non_finite": payload.get("non_finite"),
        "checkpoint_unchanged": hashes.get("unchanged"),
        "checkpoint_loop": payload.get("checkpoint_loop"),
        "full_contact_labels": label_total,
        "reference_frames_per_trial": reference_frames,
        "elapsed_s": payload.get("elapsed_s"),
        "rss_mb": payload.get("rss_mb"),
    }
    return valid, info, issues


def _summarise(trials):
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
            "per_trial_n": sorted({
                int(v) for v in n if _finite(v)
            }),
        }

    reference_counts = [
        t["A"]["reference_before"].get("n") for t in trials
        if _finite(t["A"]["reference_before"].get("n"))
    ]
    return {
        "n_trials": len(trials),
        "reference_n_per_trial": sorted({int(v) for v in reference_counts}),
        "reference_n_cumulative": int(sum(reference_counts)),
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


def _fmt(value, digits=4):
    if not _finite(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def _counts(entry):
    return f"{entry.get('improved')}/{entry.get('unchanged')}/{entry.get('worsened')}"


def _case_sort_key(case_id):
    digits = "".join(character for character in str(case_id) if character.isdigit())
    return (0, int(digits)) if digits else (1, str(case_id))


def _write_report(output, report_path, summary_path):
    lines = ["# 随机接触抽样 (A) 与全接触监督 (B) 配对单步实验"]
    if output["source_issues"]:
        lines.append("")
        lines.append("> **源诊断失败或不完整，以下不能作为正常结论。**")
        for issue in output["source_issues"]:
            lines.append(f"> - {issue}")
    lines.append("")
    lines.append("每一对 A/B 从同一检查点、同一非接触批次和同一 AirNet "
                 "更新开始；只替换 contact_data 的标签集合。单步独立实验，"
                 "不是连续训练。")
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
    if independent:
        lines.append(
            "每次独立试验的参考帧集合（由实际记录计算）："
            + "、".join(str(value) for value in independent)
            + f" 帧；当前汇总的累计评估点数为 {cumulative}。"
            "独立帧数与跨试验累计点数不能混用。"
        )
    else:
        lines.append("没有可用的参考帧记录，不能判断评估规模。")
    lines.append("A 的随机批次损失与 B 的全标签损失不是同一集合，不能直接"
                 "比较数值；共同 A 标签与全标签评估分别列出。")
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
    lines.append("改善/持平/变差次数（B-A 负值表示 B 更有利）：")
    lines.append("| 检查点 | 种子 | A macro | B macro | B-A macro | "
                 "B-A pooled | B-A 覆盖率 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            lines.append(
                f"| {tag} | {seed} | {_counts(summary['A']['macro'])} | "
                f"{_counts(summary['B']['macro'])} | "
                f"{_counts(summary['paired']['macro_B_minus_A'])} | "
                f"{_counts(summary['paired']['pooled_B_minus_A'])} | "
                f"{_counts(summary['paired']['coverage_B_minus_A'])} |"
            )
    lines.append("")
    lines.append("趋势措辞：改善 > 半数才是“多数改善”；恰好一半是“半数改善”；"
                 "多数为 0 时是“多数持平/全部持平”，不把未改善一律称为变差。")
    lines.append("| 检查点 | 种子 | A macro | B macro | B-A macro | "
                 "B-A pooled | B-A 覆盖率 |")
    lines.append("|---|---:|---|---|---|---|---|")
    for tag, entry in output["checkpoints"].items():
        for seed, item in entry["seeds"].items():
            summary = item["summary"]
            if summary is None:
                continue
            lines.append(
                f"| {tag} | {seed} | {_majority(summary['A']['macro'])} | "
                f"{_majority(summary['B']['macro'])} | "
                f"{_majority(summary['paired']['macro_B_minus_A'])} | "
                f"{_majority(summary['paired']['pooled_B_minus_A'])} | "
                f"{_majority(summary['paired']['coverage_B_minus_A'])} |"
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
            key=lambda pair: _case_sort_key(pair[0]),
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
    lines.append("## 精确覆盖率与配对差（跨成功试验累计）")
    lines.append("| 检查点 | 组 | covered before | covered after | 累计 n | 单次 n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        summary = entry["combined"]
        if not summary:
            continue
        for arm in ("A", "B"):
            counts = summary[f"{arm}_coverage_counts"]
            lines.append(
                f"| {tag} | {arm} | {counts['covered_before']} | "
                f"{counts['covered_after']} | {counts['n']} | "
                f"{counts['per_trial_n']} |"
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
                f"{_counts(item)} |"
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
    lines.append("## 全接触标签与排除原因（每个训练算例）")
    lines.append("| 检查点 | 算例 | n | out_of_time | no_slope | candidates |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for tag, entry in output["checkpoints"].items():
        labels = entry.get("labels") or {}
        for case_id, item in sorted(
            labels.items(),
            key=lambda pair: _case_sort_key(pair[0]),
        ):
            lines.append(
                f"| {tag} | {case_id} | {item.get('n')} | "
                f"{item.get('excluded_outside_time')} | "
                f"{item.get('excluded_no_slope')} | "
                f"{item.get('total_frame_candidates')} |"
            )
    lines.append("")
    lines.append("## 判读（由当前输入生成）")
    directions = []
    for tag, entry in output["checkpoints"].items():
        seed_items = [
            item["summary"]["paired"]["macro_B_minus_A"]
            for item in entry["seeds"].values()
            if item["summary"] is not None
        ]
        if not seed_items:
            lines.append(f"- {tag}：不可判断（没有可评估的配对结果）。")
            continue
        better = sum(item["improved"] or 0 for item in seed_items)
        worse = sum(item["worsened"] or 0 for item in seed_items)
        tie = sum(item["unchanged"] or 0 for item in seed_items)
        medians = [item["median"] for item in seed_items if _finite(item["median"])]
        if not medians:
            sign = "不可判断"
        elif all(value < 0 for value in medians):
            sign = "B 更有利"
        elif all(value > 0 for value in medians):
            sign = "A 更有利"
        elif all(value == 0 for value in medians):
            sign = "全部持平"
        else:
            sign = "方向不一致"
        directions.append(sign)
        lines.append(
            f"- {tag}：B-A macro 各种子中位 "
            + "、".join(_fmt(value, 4) for value in medians)
            + f"；B 更优 {better} 次，A 更优 {worse} 次，持平 {tie} 次；{sign}。"
        )
    stable = bool(directions) and len(set(directions)) == 1 and directions[0] in (
        "B 更有利", "A 更有利", "全部持平"
    )
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
                + "；该集合只作补充定位，与全标签训练目标不是同一集合，"
                "不作为 B 优劣的独立否决条件。"
            )
    text = chr(10).join(lines) + chr(10)
    report_path.write_text(text, encoding="utf-8")
    summary_path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directory", default="outputs/fullframes/contact_fullcontact_pair"
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--output-directory", default=None,
        help="write summary/report here instead of overwriting the input directory",
    )
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    out_dir = Path(args.output_directory) if args.output_directory else directory
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest) if args.manifest
        else next(
            (directory / name for name in ("pair_manifest.json",
                                           "experiment_config.json")
             if (directory / name).exists()),
            None,
        )
    )
    manifest_issues: list[str] = []
    if manifest_path is None:
        manifest_issues.append("manifest not found in the input directory")
        expectations = None
    else:
        manifest = _load_json(manifest_path)
        if "_load_error" in manifest:
            manifest_issues.append(
                f"manifest cannot be read: {manifest['_load_error']}"
            )
            expectations = None
        else:
            expectations, manifest_issues = _manifest_expectations(manifest)

    files = sorted(directory.glob("pair_*_seed*.json"))
    if not files:
        manifest_issues.append("no paired experiment files found")
    payloads = {path.name: _load_json(path) for path in files}
    expected_files = expectations["expected"] if expectations else {}
    actual_names = {path.name for path in files}
    for name in sorted(expected_files):
        if name not in actual_names:
            manifest_issues.append(f"expected file is missing: {name}")
    for name in sorted(actual_names - set(expected_files)):
        manifest_issues.append(f"unexpected paired file: {name}")

    expected_cases = set()
    for payload in payloads.values():
        labels = payload.get("labels")
        if isinstance(labels, dict):
            expected_cases.update(labels)
        per_case = payload.get("full_contact_labels_per_case")
        if isinstance(per_case, dict):
            expected_cases.update(per_case)
    if not expected_cases:
        manifest_issues.append("no expected training cases found in the inputs")

    output = {
        "format": "spherical_blast_pinn.contact_fullcontact_pair_summary",
        "version": 2,
        "directory": str(directory),
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_issues": list(dict.fromkeys(manifest_issues)),
        "checkpoints": {},
        "source_issues": [],
    }
    all_issues = list(dict.fromkeys(manifest_issues))
    grouped: dict[str, dict[int, list[tuple[dict, dict]]]] = {}
    for name, expected in expected_files.items():
        payload = payloads.get(name)
        if payload is None:
            continue
        trials, info, issues = _validate_source(
            payload, expected, expected_cases
        )
        all_issues.extend(issues)
        grouped.setdefault(expected["tag"], {}).setdefault(
            expected["seed"], []
        ).append((info, payload))
        # Keep the validated trial records separately.
        info["_trials"] = trials

    for tag, seeds in sorted(grouped.items()):
        output["checkpoints"][tag] = {
            "seeds": {},
            "combined": None,
            "labels": {},
        }
        combined_trials = []
        for seed, entries in sorted(seeds.items()):
            trials = []
            files_info = []
            labels = {}
            for info, payload in entries:
                files_info.append({k: v for k, v in info.items() if k != "_trials"})
                trials.extend(info.get("_trials", []))
                labels.update(payload.get("labels") or {})
            output["checkpoints"][tag]["labels"].update(labels)
            output["checkpoints"][tag]["seeds"][str(seed)] = {
                "files": files_info,
                "trials": len(trials),
                "summary": _summarise(trials) if trials else None,
                "source_issues": [
                    issue for issue in all_issues
                    if issue.startswith(
                        tuple(name for name in (
                            info["file"] for info, _ in entries
                        ))
                    )
                ],
                "labels": labels,
            }
            combined_trials.extend(trials)
        output["checkpoints"][tag]["combined"] = (
            _summarise(combined_trials) if combined_trials else None
        )

    # Group consistency: the same tag must not mix checkpoints or configs.
    for tag, entry in output["checkpoints"].items():
        hashes = set()
        configs = set()
        loops = set()
        for item in entry["seeds"].values():
            for file in item["files"]:
                hashes.add(str(file.get("checkpoint_unchanged")))
                loops.add(str(file.get("checkpoint_loop")))
        source_paths = {
            expected["checkpoint_path"]
            for expected in expected_files.values()
            if expected["tag"] == tag
        }
        config_paths = {
            expected["config_path"]
            for expected in expected_files.values()
            if expected["tag"] == tag
        }
        if len(source_paths) > 1:
            all_issues.append(f"{tag}: mixed checkpoint paths")
        if len(config_paths) > 1:
            all_issues.append(f"{tag}: mixed configuration paths")
        if "False" in hashes:
            all_issues.append(f"{tag}: checkpoint changed in at least one file")
        if len(loops) > 1:
            all_issues.append(f"{tag}: mixed checkpoint loops {sorted(loops)}")

    output["source_issues"] = list(dict.fromkeys(all_issues))
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
