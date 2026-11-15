"""Completeness handling in the paired C/D early-window evaluation."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_gate_pair",
    ROOT / "scripts" / "evaluate_gate_pair.py",
)
assert _SPEC is not None and _SPEC.loader is not None
gate_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate_eval)


def _case(case_id, role, radius):
    return SimpleNamespace(
        case_id=case_id, role=role, charge_radius_m=radius, t0=1.0e-5,
        rho_scale=1.0, u_scale=1.0, p_scale=1.0,
    )


def _cases():
    train = [_case(f"r{50 * (i + 1)}_full", "train", 0.05 * (i + 1))
             for i in range(10)]
    validation = [_case(f"r{50 * (i + 11)}_full", "validation",
                        0.05 * (i + 11)) for i in range(10)]
    return train + validation


def _row(case, values=(1.0, 2.0), inside=(True, True)):
    time = np.array([float(case.t0 + i) for i in range(len(values))])
    radius = np.array([float(case.charge_radius_m + i) for i in range(len(values))])
    return {
        "case_id": case.case_id,
        "role": case.role,
        "status": "ok",
        "selected": len(values),
        "used": len(values),
        "time": time,
        "radius": radius,
        "inside": np.asarray(inside, dtype=bool),
        "pred": {
            "rho": np.asarray(values, dtype=np.float64),
            "u": np.asarray(values, dtype=np.float64),
            "p": np.asarray(values, dtype=np.float64),
        },
        "target": {
            "rho": np.zeros(len(values)),
            "u": np.zeros(len(values)),
            "p": np.zeros(len(values)),
        },
    }


def _install(monkeypatch, provider):
    def fake_window(trainer, case, lo, hi):
        return provider(case)

    monkeypatch.setattr(gate_eval.ana, "_window_case", fake_window)


def test_all_ten_train_and_ten_validation_cases_are_complete(monkeypatch):
    cases = _cases()
    _install(monkeypatch, lambda case: _row(case))
    result = gate_eval._early_pair(None, None, cases, "0_1pct")
    assert result["complete"] is True
    for split in ("train", "validation"):
        block = result["by_split"][split]
        assert block["n_expected"] == 10
        assert block["n_evaluated"] == 10
        assert block["n_missing"] == 0
        assert block["missing_cases"] == []
        assert block["complete"] is True
        assert len(block["expected_case_ids"]) == 10
        assert set(block["evaluated_case_ids"]) == set(block["expected_case_ids"])


def test_missing_case_is_recorded_instead_of_silently_skipped(monkeypatch):
    cases = _cases()
    missing_id = "r500_full"

    def provider(case):
        if case.case_id == missing_id:
            return {
                "case_id": case.case_id, "role": case.role,
                "status": "no_reference_rows", "selected": 0, "used": 0,
            }
        return _row(case)

    _install(monkeypatch, provider)
    result = gate_eval._early_pair(None, None, cases, "0_1pct")
    block = result["by_split"]["train"]
    assert result["complete"] is False
    assert block["n_expected"] == 10
    assert block["n_evaluated"] == 9
    assert block["n_missing"] == 1
    assert block["missing_cases"] == [{
        "case_id": missing_id,
        "status": "unevaluable_case",
        "failure_reasons": ["C:no_reference_rows", "D:no_reference_rows"],
    }]
    assert "evaluated case ids differ" in " ".join(block["incomplete_reasons"])


def test_empty_common_domain_is_marked_incomplete_and_not_nan(monkeypatch):
    cases = _cases()

    # Only D differs on the target case: make a side-specific provider.
    def provider_side(case, side):
        row = _row(case)
        if case.case_id == "r500_full":
            row["inside"] = (
                np.array([True, False]) if side == "C"
                else np.array([False, True])
            )
        return row

    calls = {"n": 0}

    def fake_window(trainer, case, lo, hi):
        side = "C" if calls["n"] % 2 == 0 else "D"
        calls["n"] += 1
        return provider_side(case, side)

    monkeypatch.setattr(gate_eval.ana, "_window_case", fake_window)
    result = gate_eval._early_pair(None, None, cases, "0_1pct")
    block = result["by_split"]["train"]
    assert result["complete"] is False
    assert block["n_evaluated"] == 9
    assert block["empty_common_domain_cases"] == [{
        "case_id": "r500_full",
        "status": "empty_common_domain",
        "failure_reasons": [
            "no physical points inside both models' [Rc,Rs] domains",
        ],
    }]
    for side in ("C_own", "D_own", "C_common", "D_common"):
        for field in ("rho", "u", "p"):
            metric = block[side][field]
            assert metric["mse"] is None or math.isfinite(metric["mse"])


def test_nonfinite_points_are_counted_not_put_into_the_mean(monkeypatch):
    cases = _cases()

    def provider(case):
        row = _row(case)
        if case.case_id == "r500_full":
            row["pred"]["rho"] = np.array([1.0, np.nan])
            row["pred"]["u"] = np.array([np.nan, np.nan])
        return row

    _install(monkeypatch, provider)
    result = gate_eval._early_pair(None, None, cases, "0_1pct")
    block = result["by_split"]["train"]
    assert result["complete"] is False
    rho = block["C_common"]["rho"]
    assert rho["n"] == 19
    assert rho["n_candidates"] == 20
    assert rho["n_nonfinite"] == 1
    assert math.isfinite(rho["mse"])
    assert rho["complete"] is False
    # All-NaN u for one case is dropped rather than multiplied by None.
    u = block["C_common"]["u"]
    assert u["n"] == 18
    assert u["n_candidates"] == 20
    assert u["n_nonfinite"] == 2
    assert u["n_cases_skipped"] == 1
    assert math.isfinite(u["mse"])
    assert u["complete"] is False


def test_aggregate_air_rows_skips_none_and_reports_completeness():
    rows = [
        {
            "case_id": "a",
            "pred": {"rho": np.array([1.0])},
            "target": {"rho": np.array([0.0])},
            "common_mask": np.array([True]),
            "scales": {"rho": 1.0},
        },
        {
            "case_id": "b",
            "pred": {"rho": np.array([np.nan])},
            "target": {"rho": np.array([0.0])},
            "common_mask": np.array([True]),
            "scales": {"rho": 1.0},
        },
    ]
    metric = gate_eval._aggregate_air_rows(
        rows, "rho", "", "common_mask", "rho"
    )
    assert metric["n"] == 1
    assert metric["n_candidates"] == 2
    assert metric["n_nonfinite"] == 1
    assert metric["n_cases_skipped"] == 1
    assert metric["mse"] == 1.0
    assert metric["complete"] is False
    empty = gate_eval._aggregate_air_rows(
        [rows[1]], "rho", "", "common_mask", "rho"
    )
    assert empty["status"] == "not_evaluable"
    assert empty["mse"] is None


def test_physical_exclusion_is_not_an_incomplete_case(monkeypatch):
    cases = _cases()

    def provider(case):
        row = _row(case)
        if case.case_id == "r500_full":
            row["inside"] = np.array([True, False])
        return row

    _install(monkeypatch, provider)
    result = gate_eval._early_pair(None, None, cases, "0_1pct")
    block = result["by_split"]["train"]
    record = next(
        row for row in block["case_records"] if row["case_id"] == "r500_full"
    )
    assert record["evaluated"] is True
    assert record["status"] == "ok"
    assert record["common"] == 1
    assert record["common_excluded"] == 1
    assert block["complete"] is True
