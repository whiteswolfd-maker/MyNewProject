#!/usr/bin/env python3
"""Evaluate the paired legacy/similarity time-input experiment.

The two models have different network input conventions, so they are loaded
into two trainers (legacy and similarity).  Contact, air, fixed-physical PDE,
interface and initial-time metrics reuse the corrected v2 helpers.  The
same-step endpoint (both loop 4000) and the best checkpoint selected by the
existing validation rule are reported separately and never mixed.
"""
from __future__ import annotations

import argparse
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
import physical_residuals as phys  # noqa: E402


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _best_loop(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["loop"])


def _checkpoint_info(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload.get("history", {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "loop": int(payload["loop"]),
        "best_loop_in_payload": history.get("best_loop"),
        "best_score_in_payload": history.get("best_score"),
    }


def _aggregate_counts(rows: list[dict], label: str) -> dict:
    fields = (
        "candidates", "slope_invalid", "nonfinite", "outside",
        "layer_invalid", "common", "common_interior", "boundary_excluded",
    )
    return {
        field: int(sum(row["counts"][label][field] for row in rows))
        for field in fields
    }


def _air_aggregate(case_metrics: dict, all_cases, labels: tuple[str, str]):
    result = {}
    for split in ("train", "validation"):
        subset = {
            case.case_id: case_metrics[case.case_id]
            for case in all_cases if case.role == split
        }
        own = {}
        common = {}
        for field in ("rho", "u", "p"):
            own[field] = {
                label: base._aggregate_block(
                    subset, lambda item, f=field, l=label: item["own"][l][f]
                )
                for label in labels
            }
            common[field] = {
                label: base._aggregate_block(
                    subset, lambda item, f=field, l=label:
                    item["common_two"][f][l]
                )
                for label in labels
            }
        result[split] = {"own": own, "common_pair": common}
    return result


def _endpoint_entry(path, trainer, cases, points, interface, initial, label):
    entry = {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "loop": int(trainer.loop),
        "contact": base._reference_metrics(trainer, cases, points),
        "interface": {
            case.case_id: interface[label][case.case_id] for case in cases
        },
        "initial_condition": {
            case.case_id: initial[label][case.case_id] for case in cases
        },
    }
    return entry


def _physics_section(fixed: dict, cases, labels: tuple[str, str]) -> dict:
    section = {}
    for split in ("train", "validation"):
        rows = [fixed[case.case_id] for case in cases if case.role == split]
        section[split] = {
            "aggregate": {
                label: base._aggregate_physics_rows(rows, label)
                for label in labels
            },
            "counts": {
                label: _aggregate_counts(rows, label) for label in labels
            },
        }
    return section


def _fmt(value, digits=5):
    return "n/a" if value is None else f"{value:.{digits}g}"


def _report(metrics, output: Path) -> None:
    lines = ["# 时间输入配对实验：legacy vs similarity", ""]
    lines.append(
        "A = legacy `x_t=t/time_reference_s`（默认 0.01）；B = similarity "
        "`x_t=((t-t0)/(R_sep/u_a_sep))/64`。两组同初始化/seed/数据/采样/"
        "权重/学习率，formal full 目标，4000 步，从零训练。"
    )
    lines.append("")
    lines.append("## 检查点")
    lines.append("| label | loop | best score in payload | SHA-256 |")
    lines.append("|---|---:|---:|---|")
    for label, item in metrics["checkpoints"].items():
        lines.append(
            f"| {label} | {item['loop']} | "
            f"{_fmt(item.get('best_score_in_payload'))} | `{item['sha256']}` |"
        )
    lines.append("")
    lines.append("## 接触（全帧有效标签，算例等权 macro）")
    lines.append("| 口径 | split | A / B | macro MSE | macro RMSE | pooled MSE | pooled RMSE | covered/n | 平均越界 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for set_name in ("last", "best"):
        for label in metrics["sets"][set_name]["labels"]:
            entry = metrics["models"][label]["contact"]
            for split in ("train", "validation"):
                row = entry[split]
                lines.append(
                    f"| {set_name}:{label} | {split} | {label} | "
                    f"{_fmt(row.get('macro_mse'))} | {_fmt(row.get('macro_rmse'))} | "
                    f"{_fmt(row.get('pooled_mse'))} | {_fmt(row.get('pooled_rmse'))} | "
                    f"{row.get('covered_count')}/{row.get('n')} | "
                    f"{_fmt(row.get('mean_violation_over_Rsep_macro'))} |"
                )
    lines.append("")
    for set_name in ("last", "best"):
        a_label, b_label = metrics["sets"][set_name]["labels"]
        a_contact = metrics["models"][a_label]["contact"]
        b_contact = metrics["models"][b_label]["contact"]
        lines.append(f"## 逐半径接触 macro MSE（{set_name}）")
        lines.append(f"| split | case | {a_label} | {b_label} |")
        lines.append("|---|---|---:|---:|")
        for split in ("train", "validation"):
            ad = {item["case_id"]: item for item in a_contact[split]["cases"]}
            bd = {item["case_id"]: item for item in b_contact[split]["cases"]}
            for case_id in ad:
                lines.append(
                    f"| {split} | {case_id} | {_fmt(ad[case_id]['mse'])} | "
                    f"{_fmt(bd[case_id]['mse'])} |"
                )
        lines.append("")
        lines.append(f"## 早/中/晚接触 macro MSE（{set_name}）")
        lines.append(f"| split | third | {a_label} | {b_label} |")
        lines.append("|---|---:|---:|---:|")
        for split in ("train", "validation"):
            for index in range(3):
                a = a_contact[split]["thirds"][index]
                b = b_contact[split]["thirds"][index]
                lines.append(
                    f"| {split} | {index} | {_fmt(a.get('macro_mse'))} | "
                    f"{_fmt(b.get('macro_mse'))} |"
                )
        lines.append("")
    lines.append("## 空气（pooled RMSE，own / A-B 共同点）")
    lines.append("| 口径 | split | field | A own | B own | A common | B common |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for set_name in ("last", "best"):
        a_label, b_label = metrics["sets"][set_name]["labels"]
        air = metrics["sets"][set_name]["air"]
        for split in ("train", "validation"):
            for field in ("rho", "u", "p"):
                a_own = air[split]["own"][field][a_label]
                b_own = air[split]["own"][field][b_label]
                a_com = air[split]["common_pair"][field][a_label]
                b_com = air[split]["common_pair"][field][b_label]
                lines.append(
                    f"| {set_name} | {split} | {field} | "
                    f"{_fmt(a_own.get('pooled_rmse'))} | "
                    f"{_fmt(b_own.get('pooled_rmse'))} | "
                    f"{_fmt(a_com.get('pooled_rmse'))} | "
                    f"{_fmt(b_com.get('pooled_rmse'))} |"
                )
    lines.append("")
    lines.append("## 固定物理共同点 PDE RMS")
    lines.append("| 口径 | split | field | A | B | common interior |")
    lines.append("|---|---|---|---:|---:|---:|")
    for set_name in ("last", "best"):
        a_label, b_label = metrics["sets"][set_name]["labels"]
        phys_section = metrics["sets"][set_name]["physics"]
        for split in ("train", "validation"):
            agg = phys_section[split]["aggregate"]
            counts = phys_section[split]["counts"]
            for field in ("mass", "momentum", "energy"):
                lines.append(
                    f"| {set_name} | {split} | {field} | "
                    f"{_fmt(agg[a_label].get(field))} | "
                    f"{_fmt(agg[b_label].get(field))} | "
                    f"{counts[a_label]['common_interior']}/"
                    f"{counts[b_label]['common_interior']} |"
                )
    lines.append("")
    lines.append("## 排除比例（固定物理参考点）")
    lines.append("| 口径 | split | model | candidates | slope invalid | nonfinite | outside | layer invalid | common interior |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for set_name in ("last", "best"):
        for split in ("train", "validation"):
            for label in metrics["sets"][set_name]["labels"]:
                c = metrics["sets"][set_name]["physics"][split]["counts"][label]
                lines.append(
                    f"| {set_name} | {split} | {label} | {c['candidates']} | "
                    f"{c['slope_invalid']} | {c['nonfinite']} | {c['outside']} | "
                    f"{c['layer_invalid']} | {c['common_interior']} |"
                )
    lines.append("")
    lines.append("## 界面与初始时刻（各自模型 R_c）")
    lines.append("| 口径 | split | model | interface u | interface p | IC R_c | IC rho | IC u | IC p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for set_name in ("last", "best"):
        for label in metrics["sets"][set_name]["labels"]:
            section = metrics["sets"][set_name]["interface_initial"][label]
            for split in ("train", "validation"):
                interface = section["interface"][split]
                initial = section["initial_condition"][split]
                lines.append(
                    f"| {set_name} | {split} | {label} | "
                    f"{_fmt(interface.get('velocity_rms'))} | "
                    f"{_fmt(interface.get('pressure_rms'))} | "
                    f"{_fmt(initial.get('initial_rc_rmse'))} | "
                    f"{_fmt(initial.get('rho_rmse'))} | "
                    f"{_fmt(initial.get('u_rmse'))} | "
                    f"{_fmt(initial.get('p_rmse'))} |"
                )
    lines.append("")
    lines.append("## 说明")
    lines.append(
        "- `last` 是两组相同的 loop 4000；`best` 是各组按现有验证分数规则选出的检查点，"
        "loop 可能不同，不能把 best 当作同一步比较。"
    )
    lines.append(
        "- 空气共同点、PDE 共同内部点和排除比例都按每个口径的两模型共同集合计算；"
        "未用验证误差选择尺度或 divisor。"
    )
    lines.append(
        "- 采样审计显示阶段偏早也出现在 R100/R150/R200（前 1/4 帧占比约 "
        "42.4%/47.6%/44.3%），不能只归因于大半径验证集；本轮没有修改采样。"
    )
    lines.append(
        "- 这是时间输入整体实验，不将结果归因于其中某一个单独步骤；也不宣称已找到根因。"
    )
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-legacy", required=True)
    parser.add_argument("--config-similarity", required=True)
    parser.add_argument("--a-last", required=True)
    parser.add_argument("--a-best", required=True)
    parser.add_argument("--b-last", required=True)
    parser.add_argument("--b-best", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points-per-case", type=int, default=20000)
    parser.add_argument("--interface-points", type=int, default=128)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    legacy, _ = diag._build_trainer(args.config_legacy)
    similarity, _ = diag._build_trainer(args.config_similarity)
    train_cases = list(legacy.cases)
    validation_cases = list(legacy.validation_cases)
    all_cases = train_cases + validation_cases
    points_train = base._points(legacy, train_cases)
    points_validation = base._points(legacy, validation_cases)
    points = {"train": points_train, "validation": points_validation}

    paths = {
        "A_last": Path(args.a_last), "A_best": Path(args.a_best),
        "B_last": Path(args.b_last), "B_best": Path(args.b_best),
    }
    trainers = {
        "A_last": legacy, "A_best": legacy,
        "B_last": similarity, "B_best": similarity,
    }
    checkpoints = {label: _checkpoint_info(path) for label, path in paths.items()}
    if _best_loop(paths["A_best"]) != checkpoints["A_best"]["loop"]:
        raise RuntimeError("A best loop mismatch")

    # Contact and air at each checkpoint.
    case_metrics = {}
    air_predictions = {}
    for label, path in paths.items():
        trainer = trainers[label]
        trainer.load(path)
        case_metrics[label] = {
            case.case_id: base._air_predictions(trainer, case)
            for case in all_cases
        }
        air_predictions[label] = case_metrics[label]

    sets = {
        "last": ("A_last", "B_last"),
        "best": ("A_best", "B_best"),
    }
    set_metrics = {}
    for set_name, labels in sets.items():
        air_case = {}
        for case in all_cases:
            air_case[case.case_id] = base._air_case_metrics(
                {label: air_predictions[label][case.case_id] for label in labels},
                case, labels=labels, pair_labels=labels,
            )
        set_metrics[set_name] = {
            "labels": list(labels),
            "air": _air_aggregate(air_case, all_cases, labels),
            "air_case": air_case,
        }

    # Interface and initial-time metrics per model.
    interface = {}
    initial = {}
    for label, path in paths.items():
        trainer = trainers[label]
        trainer.load(path)
        interface[label] = {
            case.case_id: phys._interface_errors(
                case, trainer.contact_net, trainer.air_net,
                points=int(args.interface_points),
            )
            for case in all_cases
        }
        initial[label] = {
            case.case_id: base._initial_condition_metrics(trainer, case)
            for case in all_cases
        }
    for set_name, labels in sets.items():
        set_metrics[set_name]["interface_initial"] = {
            label: {
                "interface": {
                    "train": base._aggregate_interface(
                        [interface[label][c.case_id] for c in train_cases]
                    ),
                    "validation": base._aggregate_interface(
                        [interface[label][c.case_id] for c in validation_cases]
                    ),
                },
                "initial_condition": {
                    "train": base._aggregate_initial(
                        [initial[label][c.case_id] for c in train_cases]
                    ),
                    "validation": base._aggregate_initial(
                        [initial[label][c.case_id] for c in validation_cases]
                    ),
                },
            }
            for label in labels
        }

    # Fixed physical reference for both comparison sets.
    reference = base._reference_physical_points(
        legacy, all_cases, int(args.points_per_case)
    )
    np.savez_compressed(
        output / "physical_reference_points.npz", **reference["arrays"]
    )
    (output / "physical_reference_metadata.json").write_text(
        json.dumps(reference["metadata"], indent=2, default=diag._jsonable),
        encoding="utf-8",
    )
    for set_name, labels in sets.items():
        fixed = base._fixed_physical_physics(
            legacy, all_cases,
            {label: paths[label] for label in labels},
            reference, labels=labels,
            trainers={label: trainers[label] for label in labels},
        )
        set_metrics[set_name]["physics"] = _physics_section(
            fixed, all_cases, labels
        )
        set_metrics[set_name]["physics_rows"] = fixed

    metrics = {
        "config_legacy": args.config_legacy,
        "config_similarity": args.config_similarity,
        "checkpoints": checkpoints,
        "sets": set_metrics,
        "models": {},
        "reference": {
            "npz": "physical_reference_points.npz",
            "metadata": "physical_reference_metadata.json",
        },
        "protocol": {
            "time_input": {"A": "legacy", "B": "similarity"},
            "similarity_input_divisor": 64.0,
            "last": "same-step endpoint, both loop 4000",
            "best": "best.pt under the existing validation-score rule; loops may differ",
            "contact": "all legal training/validation labels, per-case mean then equal-case mean",
            "air": "own valid points and pair-common physical points",
            "pde": "fixed data-derived (case_id,t,r), pair-common interior, mapped Euler residual",
            "interface": "each model on its own R_c",
            "no_scale_fit": "time_scale is R_sep/u_a_sep; divisor is fixed 64.0",
        },
    }
    for label, path in paths.items():
        trainer = trainers[label]
        trainer.load(path)
        metrics["models"][label] = {
            "checkpoint": checkpoints[label],
            "contact": {
                "train": base._reference_metrics(trainer, train_cases, points_train),
                "validation": base._reference_metrics(
                    trainer, validation_cases, points_validation
                ),
            },
        }

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=diag._jsonable), encoding="utf-8"
    )
    _report(metrics, output)
    print("wrote", output / "metrics.json")
    print("wrote", output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
