#!/usr/bin/env python3
"""Evaluate A (full objective) vs D (contact-data-only objective).

Reuses the corrected v2 evaluation helpers and definitions:
* all legal contact labels, per-case mean then case-equal;
* exact coverage, macro/pooled MSE/RMSE, early/middle/late;
* air own/A-D common/start-A-D common points;
* old t/zeta PDE kept only as a labelled comparison;
* fixed data-derived (case_id,t,r) PDE on the three-model common interior;
* interface on each model's own R_c and initial-time metrics.

Reads checkpoints only.  No training, no model selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402
import evaluate_contact_lr_resume as base  # noqa: E402
import physical_residuals as phys  # noqa: E402

LABELS = ("start_4000", "A_4500", "D_4500")
PAIR_LABELS = ("A_4500", "D_4500")
GROUPS = {
    "A_full_objective": "contact_lr_resume/contact_lr_1e-4_resume4000_to4500",
    "D_contact_data_only": "contact_objective_ablation/contact_data_only_resume4000_to4500",
}


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _endpoint_air(entry, label, split):
    return entry[f"{split}_air_own"][label]


def _make_endpoint_entry(trainer, checkpoints, label, all_cases,
                         endpoint_metrics, old_physics, fixed_physics,
                         interface, initial_condition, args):
    entry = {
        "checkpoint": str(checkpoints[label]),
        "checkpoint_sha256": _sha256(checkpoints[label]),
        "train": {},
        "validation": {},
        "air_own": {},
        "air_common_two": {},
        "air_common_three": {},
        "physics_own_mapped": {},
        "physics_fixed_common": {},
        "interface": {},
        "initial_condition": {},
    }
    for case in all_cases:
        case_metrics = endpoint_metrics[case.case_id]
        split = "train" if case.role == "train" else "validation"
        entry[split][case.case_id] = case_metrics
        entry["air_own"][case.case_id] = case_metrics["own"][label]
        entry["air_common_two"][case.case_id] = (
            case_metrics["common_two"] if label in PAIR_LABELS else None
        )
        entry["air_common_three"][case.case_id] = case_metrics["common_three"]
        entry["physics_own_mapped"][case.case_id] = old_physics[label][case.case_id]
        entry["physics_fixed_common"][case.case_id] = fixed_physics[case.case_id]["residuals"][label]
        entry["interface"][case.case_id] = interface[label][case.case_id]
        entry["initial_condition"][case.case_id] = initial_condition[label][case.case_id]
    for split in ("train", "validation"):
        subset = {
            case.case_id: entry[split][case.case_id]
            for case in all_cases if case.role == split
        }
        entry[f"{split}_air_own"] = {
            field: base._aggregate_block(
                subset, lambda item, f=field: item["own"][label][f]
            )
            for field in ("rho", "u", "p")
        }
        entry[f"{split}_air_common_two"] = {
            field: (
                base._aggregate_block(
                    subset, lambda item, f=field: item["common_two"][f][label]
                ) if label in PAIR_LABELS else None
            )
            for field in ("rho", "u", "p")
        }
        entry[f"{split}_air_common_three"] = {
            field: base._aggregate_block(
                subset, lambda item, f=field: item["common_three"][f][label]
            )
            for field in ("rho", "u", "p")
        }
        physics_rows = [
            fixed_physics[case.case_id]
            for case in all_cases if case.role == split
        ]
        entry[f"{split}_physics_fixed_common"] = base._aggregate_physics_rows(
            physics_rows, label
        )
        entry[f"{split}_physics_own_mapped"] = base._aggregate_old_physics(
            [old_physics[label][case.case_id]
             for case in all_cases if case.role == split]
        )
        entry[f"{split}_interface"] = base._aggregate_interface(
            [interface[label][case.case_id]
             for case in all_cases if case.role == split]
        )
        entry[f"{split}_initial_condition"] = base._aggregate_initial(
            [initial_condition[label][case.case_id]
             for case in all_cases if case.role == split]
        )
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points-per-case", type=int, default=20000)
    parser.add_argument("--interface-points", type=int, default=128)
    args = parser.parse_args(argv)
    experiment = Path(args.experiment_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer, _ = diag._build_trainer(args.config)
    train_cases = list(trainer.cases)
    validation_cases = list(trainer.validation_cases)
    all_cases = train_cases + validation_cases
    train_points = base._points(trainer, train_cases)
    validation_points = base._points(trainer, validation_cases)
    start = experiment / "contact_lr_resume" / "checkpoints" / "loop_4000.pt"
    checkpoints = {
        "start_4000": start,
        "A_4500": experiment / GROUPS["A_full_objective"] / "checkpoints" / "loop_4500.pt",
        "D_4500": experiment / GROUPS["D_contact_data_only"] / "checkpoints" / "loop_4500.pt",
    }
    loops = [4000]
    for group, relative in GROUPS.items():
        directory = experiment / relative / "checkpoints"
        found = sorted(
            (int(path.stem.split("_")[-1]), path)
            for path in directory.glob("loop_*.pt")
            if int(path.stem.split("_")[-1]) > 4000
        )
        loops.extend(loop for loop, _ in found)
    loops = sorted(set(loops))

    metrics = {
        "config": args.config,
        "experiment_dir": str(experiment),
        "checkpoint_start": str(start),
        "checkpoint_start_sha256": _sha256(start),
        "loops": loops,
        "groups": {},
        "endpoints": {},
        "labels": list(LABELS),
        "physics_protocol": {
            "old": "own_mapped_t_zeta: fixed t/zeta, different physical r per model",
            "new": "fixed_physical_common: data-derived fixed (t,r), per-model zeta, "
                   "three-model common interior points, mapped Euler residual",
        },
    }
    for group, relative in GROUPS.items():
        metrics["groups"][group] = {}
        for loop in loops:
            path = (
                start if loop == 4000
                else experiment / relative / "checkpoints" / f"loop_{loop}.pt"
            )
            if not path.exists():
                metrics["groups"][group][str(loop)] = {"missing": True}
                continue
            trainer.load(path)
            metrics["groups"][group][str(loop)] = {
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
                "train": base._reference_metrics(trainer, train_cases, train_points),
                "validation": base._reference_metrics(
                    trainer, validation_cases, validation_points
                ),
                "effective_learning_rates": trainer.effective_learning_rates(),
                "loop": int(trainer.loop),
            }

    air_predictions = {}
    for label, path in checkpoints.items():
        trainer.load(path)
        air_predictions[label] = {
            case.case_id: base._air_predictions(trainer, case)
            for case in all_cases
        }
    endpoint_metrics = {}
    for case in all_cases:
        endpoint_metrics[case.case_id] = base._air_case_metrics(
            {label: air_predictions[label][case.case_id] for label in LABELS},
            case, labels=LABELS, pair_labels=PAIR_LABELS,
        )

    old_physics = {}
    for label, path in checkpoints.items():
        trainer.load(path)
        old_physics[label] = {
            case.case_id: base._old_own_mapping_physics(trainer, case)
            for case in all_cases
        }
    interface = {}
    initial_condition = {}
    for label, path in checkpoints.items():
        trainer.load(path)
        interface[label] = {
            case.case_id: phys._interface_errors(
                case, trainer.contact_net, trainer.air_net,
                points=int(args.interface_points),
            )
            for case in all_cases
        }
        initial_condition[label] = {
            case.case_id: base._initial_condition_metrics(trainer, case)
            for case in all_cases
        }

    reference = base._reference_physical_points(
        trainer, all_cases, int(args.points_per_case)
    )
    np.savez_compressed(
        output / "physical_reference_points.npz", **reference["arrays"]
    )
    (output / "physical_reference_metadata.json").write_text(
        json.dumps(reference["metadata"], indent=2, default=diag._jsonable),
        encoding="utf-8",
    )
    fixed_physics = base._fixed_physical_physics(
        trainer, all_cases, checkpoints, reference, labels=LABELS
    )
    metrics["physical_fixed_rows"] = fixed_physics
    metrics["physical_reference"] = {
        "npz": "physical_reference_points.npz",
        "metadata": "physical_reference_metadata.json",
    }

    for label in LABELS:
        metrics["endpoints"][label] = _make_endpoint_entry(
            trainer, checkpoints, label, all_cases, endpoint_metrics,
            old_physics, fixed_physics, interface, initial_condition, args,
        )

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=diag._jsonable), encoding="utf-8"
    )
    base._write_plots(
        output, metrics, trainer, all_cases,
        {
            "original": (4000, start),
            "A full objective lr=1e-4": (4500, checkpoints["A_4500"]),
            "D data-only objective": (4500, checkpoints["D_4500"]),
        },
    )
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
