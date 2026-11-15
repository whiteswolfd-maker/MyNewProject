#!/usr/bin/env python3
"""Audit per-case completeness of the existing C/D metrics.json.

Read-only: it does not evaluate checkpoints and does not rewrite the old
report.  Current metrics record ``n_nonfinite``/``complete``/``assessment``;
legacy metrics do not.  Legacy fields are checked with the old rules and are
explicitly marked as unrecorded, never as "proven zero non-finite".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
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
CURRENT_FIELD_KEYS = (
    "n_candidates", "n_nonfinite", "complete", "n_cases_skipped",
    "n_cases_present", "n_cases_expected",
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


def _is_current_metric(metric):
    return any(key in metric for key in CURRENT_FIELD_KEYS)


def _field_failures(metric, path):
    """Independent checks plus the current schema's recorded status."""
    failures = []
    if not isinstance(metric, dict):
        return [f"{path}: metric is not a dict"]
    n = metric.get("n")
    mse = metric.get("mse")
    n_valid = isinstance(n, (int, float)) and int(n) > 0
    mse_valid = (
        mse is not None
        and math.isfinite(float(mse))
        and float(mse) >= 0.0
    )
    if not n_valid or not mse_valid:
        failures.append(
            f"{path}: need n>0 and nonnegative finite MSE "
            f"(n={n}, mse={mse})"
        )
    if "complete" in metric and not bool(metric["complete"]):
        failures.append(f"{path}: recorded complete=false")
    if int(metric.get("n_nonfinite", 0)) > 0:
        failures.append(
            f"{path}: recorded n_nonfinite={metric.get('n_nonfinite')}"
        )
    if int(metric.get("n_cases_skipped", 0)) > 0:
        failures.append(
            f"{path}: recorded n_cases_skipped={metric.get('n_cases_skipped')}"
        )
    if metric.get("status") not in (None, "ok"):
        failures.append(f"{path}: recorded status={metric.get('status')}")

    if all(key in metric for key in ("n", "n_candidates", "n_nonfinite")):
        n_int = int(metric["n"])
        candidates = int(metric["n_candidates"])
        nonfinite = int(metric["n_nonfinite"])
        if n_int + nonfinite != candidates:
            failures.append(
                f"{path}: n+n_nonfinite={n_int + nonfinite} "
                f"!= n_candidates={candidates}"
            )
    if all(
        key in metric
        for key in ("n_cases", "n_cases_used", "n_cases_skipped")
    ):
        total = int(metric["n_cases"])
        used = int(metric["n_cases_used"])
        skipped = int(metric["n_cases_skipped"])
        if used + skipped != total:
            failures.append(
                f"{path}: n_cases_used+n_cases_skipped="
                f"{used + skipped} != n_cases={total}"
            )
    if all(
        key in metric for key in ("n_cases_present", "n_cases_expected")
    ):
        present = int(metric["n_cases_present"])
        expected = int(metric["n_cases_expected"])
        if present != expected:
            failures.append(
                f"{path}: n_cases_present={present} "
                f"!= n_cases_expected={expected}"
            )
    return failures


def _case_row_failures(rows, path):
    failures = []
    for row in rows:
        case_id = row.get("case_id", "?")
        prefix = f"{path}/{case_id}"
        if "evaluated" in row and not bool(row["evaluated"]):
            failures.append(f"{prefix}: evaluated=false")
        if "own_C_evaluated" in row and not bool(row["own_C_evaluated"]):
            failures.append(f"{prefix}: own_C_evaluated=false")
        if "own_D_evaluated" in row and not bool(row["own_D_evaluated"]):
            failures.append(f"{prefix}: own_D_evaluated=false")
        if "common_evaluated" in row and not bool(row["common_evaluated"]):
            failures.append(f"{prefix}: common_evaluated=false")
        if "common" in row and "C_inside" in row:
            if int(row["common"]) > int(row["C_inside"]):
                failures.append(f"{prefix}: common > C_inside")
        if "common" in row and "D_inside" in row:
            if int(row["common"]) > int(row["D_inside"]):
                failures.append(f"{prefix}: common > D_inside")
    return failures


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
    source_assessment = payload.get("assessment")
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
                path = f"{suffix}/{window}/{split}"
                rows = block.get("case_records") or block.get("exclusions", [])
                ids = [row["case_id"] for row in rows]
                id_counts = Counter(ids)
                duplicate_ids = {
                    case_id: count
                    for case_id, count in id_counts.items()
                    if count > 1
                }
                unique_ids = sorted(id_counts)
                expected_ids = expected[split]
                missing = sorted(set(expected_ids) - set(ids))
                extra = sorted(set(ids) - set(expected_ids))
                id_set_equal = set(ids) == set(expected_ids)
                recorded_count_equal = len(rows) == len(expected_ids)
                empty_common = sorted(
                    row["case_id"] for row in rows
                    if int(row.get("common", 0)) == 0
                )
                zero_used = sorted(
                    row["case_id"] for row in rows
                    if int(row.get("used", row.get("C_used", 0))) == 0
                )

                field_failures = []
                unrecorded = []
                current_fields = 0
                for side in ("C_own", "D_own", "C_common", "D_common"):
                    for field in ("rho", "u", "p"):
                        metric = block[side][field]
                        field_path = f"{path}/{side}/{field}"
                        if _is_current_metric(metric):
                            current_fields += 1
                        else:
                            unrecorded.append(field_path)
                        field_failures.extend(
                            _field_failures(metric, field_path)
                        )

                split_failures = []
                for key in ("complete", "own_complete", "common_complete"):
                    if key in block and not bool(block[key]):
                        split_failures.append(f"{path}: {key}=false")
                if int(block.get("n_missing", 0)) > 0:
                    split_failures.append(
                        f"{path}: n_missing={block.get('n_missing')}"
                    )
                if distinct := block.get("missing_cases"):
                    split_failures.append(
                        f"{path}: missing_cases={len(distinct)}"
                    )
                if block.get("incomplete_reasons"):
                    split_failures.append(
                        f"{path}: incomplete_reasons="
                        f"{block['incomplete_reasons']}"
                    )
                split_failures.extend(_case_row_failures(rows, path))
                if duplicate_ids:
                    split_failures.append(
                        f"{path}: duplicate case ids="
                        f"{duplicate_ids}"
                    )
                if not recorded_count_equal:
                    split_failures.append(
                        f"{path}: recorded_count={len(rows)} "
                        f"!= expected_count={len(expected_ids)}"
                    )
                if not id_set_equal:
                    split_failures.append(
                        f"{path}: id_set differs from expected"
                    )
                if missing:
                    split_failures.append(f"{path}: missing={missing}")
                if extra:
                    split_failures.append(f"{path}: extra={extra}")
                if empty_common:
                    split_failures.append(
                        f"{path}: empty_common={empty_common}"
                    )
                if zero_used:
                    split_failures.append(f"{path}: zero_used={zero_used}")

                failures = split_failures + field_failures
                if current_fields and unrecorded:
                    schema = "mixed"
                elif current_fields:
                    schema = "current"
                else:
                    schema = "legacy"
                checks.append({
                    "set": suffix,
                    "window": window,
                    "split": split,
                    "schema": schema,
                    "expected_case_ids": expected_ids,
                    "recorded_case_ids": ids,
                    "unique_recorded_case_ids": unique_ids,
                    "duplicate_case_ids": duplicate_ids,
                    "n_expected": len(expected_ids),
                    "n_recorded": len(rows),
                    "n_unique_recorded": len(unique_ids),
                    "recorded_count_equal": recorded_count_equal,
                    "id_set_equal": id_set_equal,
                    "missing_cases": missing,
                    "extra_cases": extra,
                    "empty_common_cases": empty_common,
                    "zero_used_cases": zero_used,
                    "current_field_count": current_fields,
                    "unrecorded_fields": unrecorded,
                    "aggregate_fields_finite": not any(
                        "need n>0 and nonnegative finite MSE" in item
                        for item in field_failures
                    ),
                    "failure_reasons": failures,
                    "complete": not failures,
                })

    failures = []
    for item in checks:
        failures.extend(item["failure_reasons"])
    for item in t0_checks:
        if not item["complete"]:
            failures.append(
                f"{item['set']}/t0/{item['model']}: incomplete"
            )
    nonfinite = _nonfinite_paths(payload)
    failures.extend(f"nonfinite:{path}" for path in nonfinite)
    if source_assessment is not None:
        if not bool(source_assessment.get("complete")):
            failures.append(
                "source assessment.complete=false: "
                f"{source_assessment.get('failure_reasons', [])}"
            )
        if source_assessment.get("exit_code") not in (None, 0):
            failures.append(
                f"source assessment.exit_code="
                f"{source_assessment.get('exit_code')}"
            )

    schemas = {item["schema"] for item in checks}
    schema = (
        "current" if schemas == {"current"}
        else "legacy" if schemas == {"legacy"}
        else "mixed"
    )
    unrecorded = sorted(
        f"{item['set']}/{item['window']}/{item['split']}/{field}"
        for item in checks
        for field in item["unrecorded_fields"]
    )
    all_complete = not failures
    result = {
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "schema": schema,
        "source_assessment_present": source_assessment is not None,
        "source_assessment": source_assessment,
        "mse_rule": (
            "n>0 and MSE nonnegative and finite; zero MSE is allowed"
        ),
        "legacy_scope_only": schema == "legacy",
        "unrecorded_fields": unrecorded,
        "unrecorded_note": (
            "Legacy metrics do not contain n_nonfinite/complete fields. "
            "Their absence is not evidence that non-finite counts were zero; "
            "only the old MSE/ID/domain checks are proven."
            if unrecorded else ""
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
            "current-schema completeness and independent case/point checks "
            "all pass"
            if all_complete and schema == "current"
            else (
                "legacy checks pass; recorded n_nonfinite/completeness fields "
                "are absent and are NOT proven zero"
                if all_complete and schema == "legacy"
                else "the metrics are incomplete, inconsistent, or non-finite"
            )
        ),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(output),
        "sha256": _sha256(output),
        "schema": result["schema"],
        "all_checks_complete": result["all_checks_complete"],
        "failure_reasons": result["failure_reasons"],
        "exit_code": result["exit_code"],
        "n_checks": len(checks),
        "nonfinite": len(nonfinite),
    }, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
