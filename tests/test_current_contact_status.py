"""Diagnostic status propagation: trial failures cannot leave top-level ok."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step = _load("diagnose_current_contact_step")
decomposition = _load("diagnose_current_contact_update_decomposition")
first_moment = _load("diagnose_current_contact_first_moment_control")
multistep = _load("run_first_moment_multistep_control")


def _payload(status="ok", *, unchanged=True, finite=True,
             air_frozen=True, restored=True):
    return {
        "checkpoints": {
            "best": {
                "unchanged": unchanged,
                "trials": [{
                    "status": status,
                    "finite_gradients": finite,
                    "air_frozen_during_contact": air_frozen,
                    "contact_restored_after_evaluation": restored,
                    "checks": {
                        "finite_gradients": finite,
                        "air_frozen_during_contact": air_frozen,
                        "contact_restored_after_evaluation": restored,
                    },
                }],
            },
        },
    }


def test_clean_trials_have_no_failure_reasons():
    assert step._status_reasons(_payload()) == []
    assert decomposition._status_reasons(_payload()) == []


def test_failed_trial_propagates_even_when_checkpoint_is_unchanged():
    reasons = step._status_reasons(_payload("failed"))
    assert any("trial status failed" in reason for reason in reasons)

    reasons = decomposition._status_reasons(_payload("failed"))
    assert any("trial status failed" in reason for reason in reasons)


def test_nonfinite_or_unrestored_trial_is_a_failure():
    for kwargs in (
        {"finite": False},
        {"air_frozen": False},
        {"restored": False},
    ):
        payload = _payload(**kwargs)
        assert step._status_reasons(payload)
        assert decomposition._status_reasons(payload)


def test_first_moment_failed_trial_propagates():
    payload = _payload("failed")
    reasons = first_moment._status_reasons(payload)
    assert any("trial status failed" in reason for reason in reasons)


def test_first_moment_rejects_nonstandard_adam():
    standard = {
        "lr": 1.0e-4, "betas": (0.9, 0.999), "eps": 1.0e-8,
        "weight_decay": 0.0, "amsgrad": False, "maximize": False,
    }
    supported, unsupported = first_moment._optimizer_supported(standard)
    assert supported and unsupported == []
    for key in ("weight_decay", "amsgrad", "maximize"):
        changed = dict(standard)
        changed[key] = 0.1 if key == "weight_decay" else True
        supported, unsupported = first_moment._optimizer_supported(changed)
        assert not supported and unsupported == [key]


def test_multistep_failed_pair_propagates():
    payload = {
        "pairs": [{
            "checkpoint": "best", "sample_seed": 1, "status": "failed",
            "error": "state mismatch",
        }],
        "checkpoint_hashes": {"unchanged": True},
    }
    reasons = multistep._status_reasons(payload)
    assert any("state mismatch" in reason for reason in reasons)


def test_multistep_nonfinite_tree_detects_nan():
    assert multistep._finite_tree({"ok": [1.0, 2.0]})
    assert not multistep._finite_tree({"bad": [1.0, float("nan")]})
