#!/usr/bin/env python3
"""Audit per-case completeness of the existing C/D metrics.json.

This is read-only.  It does not evaluate checkpoints and does not rewrite the
old report.  The legacy file only stored a row for cases that produced an
``ok`` window result, so completeness must be checked from the recorded IDs
rather than inferred from the aggregate count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402
import evaluate_gate_pair as pair  # noqa: E402


WINDOWS = ("0_1pct", "0_5pct", "0_10pct")
DEFAULT_METRICS = (
    "outputs/fullframes/time_input_experiment/gate_pair/evaluation/metrics.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nonfinite_paths(value, path=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_nonfinite_paths(item, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_nonfinite_paths(item, f"{path}/{index}"))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument(
        "--output",
        default=(
            "outputs/fullframes/time_input_experiment/gate_pair/"
            "failure_semantics/completeness_audit.json"
        ),
    )
    args = parser.parse_args(argv)
    metrics_path = _REPO_ROOT / args.metrics
    output = _REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    trainer, _ = diag._build_trainer(pair.C_CONFIG)
    expected = {
        split: sorted(
            case.case_id
            for case in list(trainer.cases) + list(trainer.validation_cases)
            if case.role == split
        )
        for split in ("train", "validation")
    }
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    checks = []
    t0_checks = []
    for suffix in ("last", "best"):
        for model in ("C", "D"):
            cases = payload["sets"][suffix]["t0"][model].get("cases", {})
            expected_ids = sorted(
                expected["train"] + expected["validation"]
            )
            t0_checks.append({
                "set": suffix,
                "model": model,
                "n_cases": len(cases),
                "expected_case_ids": expected_ids,
                "recorded_case_ids": sorted(cases),
                "complete": sorted(cases) == expected_ids,
            })
        for window in WINDOWS:
            for split in ("train", "validation"):
                block = payload["sets"][suffix]["early"][window][
                    "by_split"
                ][split]
                rows = block.get("exclusions", [])
                ids = [row["case_id"] for row in rows]
                expected_ids = expected[split]
                missing = sorted(set(expected_ids) - set(ids))
                extra = sorted(set(ids) - set(expected_ids))
                empty_common = sorted(
                    row["case_id"] for row in rows
                    if int(row.get("common", 0)) == 0
                )
                zero_used = sorted(
                    row["case_id"] for row in rows
                    if int(row.get("used", 0)) == 0
                )
                fields_ok = True
                for side in ("C_own", "D_own", "C_common", "D_common"):
                    for field in ("rho", "u", "p"):
                        metric = block[side][field]
                        if not (
                            int(metric.get("n", 0)) > 0
                            and metric.get("mse") is not None
                            and math.isfinite(float(metric["mse"]))
                        ):
                            fields_ok = False
                complete = (
                    len(rows) == len(expected_ids) == 10
                    and not missing and not extra
                    and not empty_common and not zero_used
                    and fields_ok
                )
                checks.append({
                    "set": suffix,
                    "window": window,
                    "split": split,
                    "expected_case_ids": expected_ids,
                    "recorded_case_ids": ids,
                    "n_expected": len(expected_ids),
                    "n_recorded": len(rows),
                    "missing_cases": missing,
                    "extra_cases": extra,
                    "empty_common_cases": empty_common,
                    "zero_used_cases": zero_used,
                    "aggregate_fields_finite": fields_ok,
                    "complete": complete,
                })

    nonfinite = _nonfinite_paths(payload)
    failures = []
    for item in checks:
        if not item["complete"]:
            failures.append(
                f"{item['set']}/{item['window']}/{item['split']}: incomplete"
            )
        if item["empty_common_cases"]:
            failures.append(
                f"{item['set']}/{item['window']}/{item['split']}: "
                f"empty common {item['empty_common_cases']}"
            )
    for item in t0_checks:
        if not item["complete"]:
            failures.append(f"{item['set']}/t0/{item['model']}: incomplete")
    failures.extend(f"nonfinite:{path}" for path in nonfinite)
    all_complete = not failures
    result = {
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "schema": (
            "legacy early-window rows exist only for evaluable cases; the "
            "corrected evaluator now records expected/evaluated/missing cases "
            "explicitly.  This audit checks IDs because the old file did not."
        ),
        "all_checks_complete": all_complete,
        "failure_reasons": failures,
        "exit_code": 0 if all_complete else 1,
        "n_checks": len(checks),
        "n_t0_checks": len(t0_checks),
        "nonfinite_numeric_values": nonfinite,
        "checks": checks,
        "t0_checks": t0_checks,
        "conclusion": (
            "all 12 window/split/set combinations contain exactly the 10 "
            "expected cases, nonempty physical domains and finite aggregates; "
            "the old numbers are retained, not a newly discovered omission"
            if all_complete
            else "the old metrics are incomplete or non-finite"
        ),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(output),
        "sha256": _sha256(output),
        "all_checks_complete": result["all_checks_complete"],
        "failure_reasons": result["failure_reasons"],
        "exit_code": result["exit_code"],
        "n_checks": len(checks),
        "nonfinite": len(nonfinite),
    }, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
