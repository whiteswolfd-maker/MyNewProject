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
