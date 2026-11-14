"""Paired-experiment summary must validate records, not trust ``ok=true``."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "summarize_contact_fullcontact_pair",
    ROOT / "scripts" / "summarize_contact_fullcontact_pair.py",
)
assert _SPEC is not None and _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)

_CHECKPOINT_BYTES = b"test-checkpoint"
_CHECKPOINT_SHA = hashlib.sha256(_CHECKPOINT_BYTES).hexdigest()


def _terms(value=1.0):
    return {
        name: {"weighted_loss": value}
        for name in ("pde", "contact_ic", "interface", "contact_data")
    }


def _reference(n=10, evaluated=True, cases=("a", "b")):
    return {
        "n": n,
        "covered_count": 2,
        "complete_training_target": evaluated,
        "cases": [
            {"case_id": case, "n": n // 2, "covered_count": 1, "mse": 1.0}
            for case in cases
        ],
    }


def _trial(index, *, seed=1, macro=0.0, pooled=0.0, coverage=0.0,
           checkpoint_loop=1, paired_ok=True, status="ok"):
    return {
        "trial": index,
        "status": status,
        "seed": seed + index * 104729,
        "checkpoint_loop": checkpoint_loop,
        "full_contact_labels_per_case": {"a": 5, "b": 5},
        "A": {
            "reference_change": {
                "macro_mse": macro, "pooled_mse": pooled, "coverage": coverage,
            },
            "reference_before": _reference(),
            "reference_after": _reference(),
            "full_labels_before": {"a": 0.0},
            "full_labels_after": {"a": 0.0},
            "before": {"terms": _terms()},
            "after": {"terms": _terms()},
            "parameter_delta_l2": 0.1,
            "time_s": 0.1,
        },
        "B": {
            "reference_change": {
                "macro_mse": macro, "pooled_mse": pooled, "coverage": coverage,
            },
            "reference_before": _reference(),
            "reference_after": _reference(),
            "full_labels_before": {"a": 0.0},
            "full_labels_after": {"a": 0.0},
            "before": {"terms": _terms()},
            "after": {"terms": _terms()},
            "parameter_delta_l2": 0.1,
            "time_s": 0.1,
        },
        "paired": {
            "initial_param_equal": paired_ok,
            "initial_optimizer_equal": paired_ok,
            "pre_contact_param_equal": paired_ok,
            "pre_contact_optimizer_equal": paired_ok,
            "air_state_equal": paired_ok,
            "air_loss_equal": paired_ok,
            "initial_state_equal": paired_ok,
            "reference_macro_mse_difference_B_minus_A": macro,
            "reference_pooled_mse_difference_B_minus_A": pooled,
            "reference_coverage_difference_B_minus_A": coverage,
            "full_label_macro_mse_difference_B_minus_A": macro,
            "common_A_macro_mse_difference_B_minus_A": 0.0,
        },
        "non_contact_sampling_equal": paired_ok,
    }


def _payload(trials, *, requested=None, completed=None, failed=0,
             checkpoint="best.pt", seed=1, labels=None,
             reference_frames=10, ok=True):
    labels = labels or {
        "a": {"n": 5, "excluded_outside_time": 0, "excluded_no_slope": 0,
              "total_frame_candidates": 5},
        "b": {"n": 5, "excluded_outside_time": 0, "excluded_no_slope": 0,
              "total_frame_candidates": 5},
    }
    requested = len(trials) if requested is None else requested
    completed = len(trials) - failed if completed is None else completed
    return {
        "format": "spherical_blast_pinn.contact_fullcontact_pair",
        "version": 1,
        "config": "cfg",
        "checkpoint": checkpoint,
        "tag": "best",
        "checkpoint_loop": 1,
        "checkpoint_sha256": {"before": _CHECKPOINT_SHA, "after": _CHECKPOINT_SHA, "unchanged": True},
        "requested_trials": requested,
        "completed_trials": completed,
        "failed_trials": failed,
        "non_finite": False,
        "seed": seed,
        "labels": labels,
        "full_contact_labels": sum(int(v.get("n", 0)) for v in labels.values()),
        "full_contact_labels_per_case": {
            key: int(value.get("n", 0)) for key, value in labels.items()
        },
        "reference_frames_per_trial": reference_frames,
        "trials": trials,
        "ok": ok,
    }


def _manifest(directory, *, seeds=(1,), trials=1, checkpoints=None,
              reference_frames=10):
    checkpoints = checkpoints or {
        "best": {"path": "best.pt", "sha256": _CHECKPOINT_SHA, "loop": 1}
    }
    payload = {
        "format": "spherical_blast_pinn.contact_fullcontact_pair.config",
        "version": 1,
        "config": {"path": str(directory / "cfg")},
        "checkpoints": {
            tag: {**value, "path": str(directory / value["path"])}
            for tag, value in checkpoints.items()
        },
        "design": {
            "seeds": list(seeds),
            "trials_per_checkpoint_seed": trials,
        },
        "training_cases": ["a", "b"],
        "reference_frames_per_trial": reference_frames,
    }
    (directory / "best.pt").write_bytes(_CHECKPOINT_BYTES)
    (directory / "experiment_config.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write(directory, name, payload):
    if payload.get("checkpoint") in {"best.pt", "other.pt"}:
        payload["checkpoint"] = str(directory / payload["checkpoint"])
    if payload.get("config") == "cfg":
        payload["config"] = str(directory / "cfg")
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_declared_hundred_but_zero_records_is_a_failure(tmp_path, capsys):
    _manifest(tmp_path, trials=100)
    _write(tmp_path, "pair_best_seed1.json",
           _payload([], requested=100, completed=0))
    assert summary.main(["--directory", str(tmp_path)]) == 1
    assert "record count" in capsys.readouterr().err


def test_duplicate_trial_indices_are_a_failure(tmp_path):
    _manifest(tmp_path, trials=2)
    _write(tmp_path, "pair_best_seed1.json",
           _payload([_trial(0), _trial(0)], requested=2))
    assert summary.main(["--directory", str(tmp_path)]) == 1


def test_missing_expected_seed_file_is_a_failure(tmp_path):
    _manifest(tmp_path, seeds=(1, 2), trials=1)
    _write(tmp_path, "pair_best_seed1.json", _payload([_trial(0)]))
    assert summary.main(["--directory", str(tmp_path)]) == 1


def test_mixed_checkpoint_is_a_failure(tmp_path):
    _manifest(tmp_path, seeds=(1, 2), trials=1)
    _write(tmp_path, "pair_best_seed1.json", _payload([_trial(0)], seed=1))
    _write(tmp_path, "pair_best_seed2.json",
           _payload([_trial(0, seed=2)], checkpoint="other.pt", seed=2))
    assert summary.main(["--directory", str(tmp_path)]) == 1


def test_all_flat_is_reported_as_all_flat(tmp_path):
    _manifest(tmp_path, trials=3)
    _write(tmp_path, "pair_best_seed1.json",
           _payload([_trial(i) for i in range(3)], requested=3))
    assert summary.main(["--directory", str(tmp_path)]) == 0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "全部持平" in text
    assert "10 帧" in text
    assert "7165" not in text


def test_few_improved_but_mostly_flat_is_not_called_worse(tmp_path):
    _manifest(tmp_path, trials=3)
    trials = [
        _trial(0, macro=-1.0),
        _trial(1, macro=0.0),
        _trial(2, macro=0.0),
    ]
    _write(tmp_path, "pair_best_seed1.json", _payload(trials, requested=3))
    assert summary.main(["--directory", str(tmp_path)]) == 0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "多数持平" in text
    assert "少数改善（多数变差）" not in text


def test_reference_frame_change_is_a_failure(tmp_path):
    _manifest(tmp_path, trials=1)
    payload = _payload([_trial(0)])
    payload["reference_frames_per_trial"] = 12
    payload["full_contact_labels"] = 12
    payload["full_contact_labels_per_case"] = {"a": 6, "b": 6}
    payload["labels"] = {
        "a": {"n": 6}, "b": {"n": 6},
    }
    for trial in payload["trials"]:
        trial["full_contact_labels_per_case"] = {"a": 6, "b": 6}
        trial["A"]["reference_before"]["n"] = 12
        trial["B"]["reference_before"]["n"] = 12
    _write(tmp_path, "pair_best_seed1.json", payload)
    assert summary.main(["--directory", str(tmp_path)]) == 1


def test_broken_runtime_pairing_condition_is_a_failure(tmp_path):
    _manifest(tmp_path, trials=1)
    _write(tmp_path, "pair_best_seed1.json",
           _payload([_trial(0, paired_ok=False)]))
    assert summary.main(["--directory", str(tmp_path)]) == 1


def test_single_empty_training_case_is_a_failure(tmp_path):
    _manifest(tmp_path, trials=1)
    payload = _payload([_trial(0)])
    payload["labels"]["b"]["n"] = 0
    payload["full_contact_labels"] = 5
    payload["full_contact_labels_per_case"]["b"] = 0
    _write(tmp_path, "pair_best_seed1.json", payload)
    assert summary.main(["--directory", str(tmp_path)]) == 1


def test_failed_record_with_ok_true_is_a_failure(tmp_path):
    _manifest(tmp_path, trials=1)
    _write(tmp_path, "pair_best_seed1.json",
           _payload([_trial(0, status="failed")], failed=1))
    assert summary.main(["--directory", str(tmp_path)]) == 1
