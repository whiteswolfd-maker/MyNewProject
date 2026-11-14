"""Summary must not crash on None/undefined and must expose source failures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "summarize_contact_training_diagnostics",
    ROOT / "scripts" / "summarize_contact_training_diagnostics.py",
)
assert _SPEC is not None and _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def _trial(macro_change=-1.0, pooled_change=1.0):
    return {
        "status": "ok",
        "contact_data_loss_change": {
            "weighted_current_batch": -0.5,
            "first_order_prediction_g_d_dot_delta": -0.4,
        },
        "reference_loss_change": {
            "macro_mse": macro_change,
            "pooled_mse": pooled_change,
            "coverage": 0.1,
        },
        "parameter_delta": {"l2": 0.01},
        "reference_contact_before": {
            "n": 4,
            "covered_count": 1,
            "cases": [{
                "case_id": "a", "n": 4, "covered_count": 1, "mse": 2.0,
                "thirds": [
                    {"n": 2, "macro_mse": 2.0, "pooled_mse": 2.0},
                    {"n": 1, "macro_mse": 2.0, "pooled_mse": 2.0},
                    {"n": 1, "macro_mse": 2.0, "pooled_mse": 2.0},
                ],
            }],
        },
        "reference_contact_after": {
            "n": 4,
            "covered_count": 2,
            "cases": [{
                "case_id": "a", "n": 4, "covered_count": 2, "mse": 1.0,
                "thirds": [
                    {"n": 2, "macro_mse": 1.0, "pooled_mse": 1.0},
                    {"n": 1, "macro_mse": 1.0, "pooled_mse": 1.0},
                    {"n": 1, "macro_mse": 1.0, "pooled_mse": 1.0},
                ],
            }],
        },
    }


def _payload(trials, **overrides):
    payload = {
        "version": 3,
        "metric_version": 3,
        "ok": True,
        "checkpoint_sha256": {"unchanged": True},
        "adam_requested": len(trials),
        "adam_completed": len(trials),
        "adam_failed": 0,
        "adam_available": True,
        "non_finite": False,
        "adam_trials": trials,
    }
    payload.update(overrides)
    return payload


def test_cosine_stats_ignore_none_and_nonfinite():
    batches = [
        {"aggregate_gradients": {"x": {"status": "ok", "cosine": 0.5}}},
        {"aggregate_gradients": {"x": {"status": "ok", "cosine": -0.5}}},
        {"aggregate_gradients": {"x": {"status": "direction_undefined",
                                       "cosine": None}}},
        {"aggregate_gradients": {"x": {"status": "non_finite",
                                       "cosine": None}}},
    ]
    stats = summary._cosine_stats(batches, "x")
    assert stats["requested"] == 4
    assert stats["valid_directions"] == 2
    assert stats["undefined"] == 1
    assert stats["non_finite"] == 1
    assert stats["median"] == 0.0
    assert stats["negative"] == 1
    assert stats["negative_fraction"] == 0.5


def test_change_stats_drop_none_and_nonfinite():
    stats = summary._change_stats(
        [None, float("nan"), -1.0, 0.0, 2.0], greater_is_better=False
    )
    assert stats["n_evaluable"] == 3
    assert stats["n_missing"] == 2
    assert stats["improved"] == 1
    assert stats["unchanged"] == 1
    assert stats["worsened"] == 1
    assert stats["median"] == 0.0


def test_adam_summary_uses_macro_pooled_and_exact_coverage(tmp_path):
    path = tmp_path / "adam_best_seed1.json"
    path.write_text(
        json.dumps(_payload([_trial(macro_change=1.0, pooled_change=-1.0)])),
        encoding="utf-8",
    )
    entry = summary._adam([path])
    assert entry["n_trials"] == 1
    assert entry["macro_mse"]["worsened"] == 1
    assert entry["pooled_mse"]["improved"] == 1
    assert entry["coverage"]["improved"] == 1
    assert entry["covered_before_total"] == 1
    assert entry["covered_after_total"] == 2
    assert entry["reference_n_total"] == 4
    assert entry["batch_good_macro_bad"] == 1


def test_all_failed_adam_has_no_improvement_rate(tmp_path):
    path = tmp_path / "adam_best_seed1.json"
    payload = _payload(
        [{"status": "failed", "error": "boom"}],
        adam_completed=0, adam_failed=1, ok=False,
        failure_reasons=["Adam failures: 1/1"],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    entry = summary._adam([path])
    assert entry["n_trials"] == 0
    assert entry["batch"]["n_evaluable"] == 0
    assert entry["batch"]["median"] is None
    assert entry["batch"]["improved_fraction"] is None
    assert entry["source_issues"]


def test_summary_returns_nonzero_for_failed_source(tmp_path, capsys):
    path = tmp_path / "static_best_seed1.json"
    path.write_text(json.dumps({
        "version": 3, "metric_version": 3, "ok": False,
        "failure_reasons": ["injected failure"],
        "requested_batches": 1, "completed_batches": 0,
        "failed_batches": 1, "non_finite": False,
        "checkpoint_sha256": {"unchanged": True},
        "batches": [],
    }), encoding="utf-8")
    code = summary.main(["--directory", str(tmp_path)])
    assert code == 1
    assert "source diagnostics incomplete" in capsys.readouterr().err
    assert "源诊断失败" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_all_undefined_directions_have_no_median_or_fraction():
    batches = [
        {"aggregate_gradients": {"x": {"status": "direction_undefined",
                                       "cosine": None}}}
        for _ in range(3)
    ]
    stats = summary._cosine_stats(batches, "x")
    assert stats["valid_directions"] == 0
    assert stats["median"] is None
    assert stats["negative_fraction"] is None
    assert stats["undefined"] == 3
