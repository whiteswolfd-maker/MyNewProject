"""main() exit-code tests for the gate-pair audit scripts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_gate_pair_metrics")
audit = _load("audit_gate_pair_completeness")
gate_eval = _load("evaluate_gate_pair")


def _case(case_id, role):
    return SimpleNamespace(case_id=case_id, role=role)


def _fake_trainer():
    train = [_case(f"r{50 * (i + 1)}", "train") for i in range(10)]
    validation = [
        _case(f"r{50 * (i + 11)}", "validation") for i in range(10)
    ]
    return SimpleNamespace(cases=train, validation_cases=validation)


def _audit_payload():
    expected = {
        split: [
            case.case_id
            for case in list(_fake_trainer().cases)
            + list(_fake_trainer().validation_cases)
            if case.role == split
        ]
        for split in ("train", "validation")
    }
    payload = {"sets": {}}
    for suffix in ("last", "best"):
        payload["sets"][suffix] = {
            "t0": {
                model: {"cases": {case_id: {} for case_id in expected["train"] + expected["validation"]}}
                for model in ("C", "D")
            },
            "early": {},
        }
        for window in ("0_1pct", "0_5pct", "0_10pct"):
            payload["sets"][suffix]["early"][window] = {"by_split": {}}
            for split in ("train", "validation"):
                rows = [
                    {
                        "case_id": case_id, "used": 1, "common": 1,
                    }
                    for case_id in expected[split]
                ]
                block = {"exclusions": rows}
                for side in ("C_own", "D_own", "C_common", "D_common"):
                    block[side] = {
                        field: {"n": 1, "mse": 1.0}
                        for field in ("rho", "u", "p")
                    }
                payload["sets"][suffix]["early"][window]["by_split"][
                    split
                ] = block
    return payload, expected


def test_compare_main_returns_nonzero_on_difference(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    out = tmp_path / "comparison.json"
    old.write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    new.write_text(json.dumps({"a": 2.0}), encoding="utf-8")
    code = compare.main([
        "--old", str(old), "--new", str(new), "--output", str(out),
    ])
    assert code == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["old_values_preserved"] is False
    assert saved["exit_code"] == 1


def test_compare_main_returns_zero_when_equal(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    out = tmp_path / "comparison.json"
    old.write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    new.write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    code = compare.main([
        "--old", str(old), "--new", str(new), "--output", str(out),
    ])
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["exit_code"] == 0


def test_audit_main_returns_nonzero_and_writes_failure(tmp_path, monkeypatch):
    payload, expected = _audit_payload()
    payload["sets"]["last"]["early"]["0_1pct"]["by_split"]["train"][
        "exclusions"
    ][0]["common"] = 0
    metrics = tmp_path / "metrics.json"
    out = tmp_path / "audit.json"
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        audit.diag, "_build_trainer", lambda config: (_fake_trainer(), {})
    )
    code = audit.main([
        "--metrics", str(metrics), "--output", str(out),
    ])
    assert code == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["all_checks_complete"] is False
    assert saved["exit_code"] == 1
    assert saved["failure_reasons"]
    assert len(expected["train"]) == 10


def test_audit_main_returns_zero_when_complete(tmp_path, monkeypatch):
    payload, _ = _audit_payload()
    metrics = tmp_path / "metrics.json"
    out = tmp_path / "audit.json"
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        audit.diag, "_build_trainer", lambda config: (_fake_trainer(), {})
    )
    code = audit.main([
        "--metrics", str(metrics), "--output", str(out),
    ])
    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["all_checks_complete"] is True
    assert saved["exit_code"] == 0


def _eval_metrics(complete):
    return {
        "loop_C": 4000,
        "loop_D": 4000,
        "early": {
            window: {
                "by_split": {
                    split: {
                        "complete": complete,
                        "incomplete_reasons": [] if complete else ["fake"],
                    }
                    for split in ("train", "validation")
                }
            }
            for window in ("0_1pct", "0_5pct", "0_10pct")
        },
    }


def _run_eval_main(tmp_path, monkeypatch, complete):
    monkeypatch.setattr(
        gate_eval.diag, "_build_trainer",
        lambda config: (_fake_trainer(), {}),
    )
    monkeypatch.setattr(gate_eval, "_sha256", lambda path: "fake")
    monkeypatch.setattr(
        gate_eval, "_metrics_for_suffix",
        lambda *args, **kwargs: _eval_metrics(complete),
    )
    monkeypatch.setattr(gate_eval, "_report", lambda metrics, output: None)
    return gate_eval.main(["--output", str(tmp_path / "evaluation")])


def test_evaluate_main_returns_nonzero_and_saves_incomplete(tmp_path, monkeypatch):
    code = _run_eval_main(tmp_path, monkeypatch, complete=False)
    assert code == 1
    saved = json.loads(
        (tmp_path / "evaluation" / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved["assessment"]["complete"] is False
    assert saved["assessment"]["exit_code"] == 1
    assert saved["assessment"]["failure_reasons"]


def test_evaluate_main_returns_zero_when_complete(tmp_path, monkeypatch):
    code = _run_eval_main(tmp_path, monkeypatch, complete=True)
    assert code == 0
    saved = json.loads(
        (tmp_path / "evaluation" / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved["assessment"]["complete"] is True
    assert saved["assessment"]["exit_code"] == 0
