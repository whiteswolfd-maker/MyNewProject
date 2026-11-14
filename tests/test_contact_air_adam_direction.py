"""Actual Adam direction trials: reconstruction, restoration and validation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.models import AirNet, ContactNet
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
from spherical_blast_pinn.contact_air.trainer import (
    AlternatingTrainer,
    TrainConfig,
    seed_everything,
)

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "contact_fullcontact_pair_direction",
    ROOT / "scripts" / "contact_fullcontact_pair.py",
)
assert _SPEC is not None and _SPEC.loader is not None
pair = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pair)


def _trainer(tmp_path: Path) -> AlternatingTrainer:
    train = [
        synthetic_case_for(0.05, case_id="s50", role="train",
                           n_frames=21, n_radial=128),
        synthetic_case_for(0.05, case_id="s50b", role="train",
                           n_frames=21, n_radial=128),
    ]
    validation = [
        synthetic_case_for(0.05, case_id="s150", role="validation",
                           n_frames=21, n_radial=128),
    ]
    seed_everything(7)
    return AlternatingTrainer(
        cases=train,
        contact_net=ContactNet(width=8, depth=2, time_reference_s=0.01),
        air_net=AirNet(width=8, depth=2, time_reference_s=0.01),
        config=TrainConfig(
            n_pde=40, n_ic=8, n_right=8, n_data=8, n_contact=8,
            loops=2, n_schedule=10, seed=7,
            validation_holdout_fraction=0.0,
            validation_data_points=8, validation_contact_points=8,
            contact_data_weight=3.0,
        ),
        output_dir=tmp_path / "direction",
        external_validation_cases=validation,
    )


def _checkpoint(tmp_path: Path):
    trainer = _trainer(tmp_path)
    trainer.step()
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    return trainer, checkpoint


def _trial(tmp_path: Path):
    trainer, checkpoint = _checkpoint(tmp_path)
    before_hash = pair._sha256(checkpoint)
    labels = {
        case.case_id: pair._full_labels(case) for case in trainer.cases
    }
    reference = pair.diag._reference_points(trainer)
    record = pair._direction_trial(
        trainer, checkpoint, reference, labels,
        SimpleNamespace(seed=999), 0,
    )
    return trainer, checkpoint, record, before_hash


def test_direction_trial_reconstructs_adam_and_gradients(tmp_path):
    _trainer_obj, checkpoint, record, before_hash = _trial(tmp_path)
    assert record["status"] == "ok", record.get("error")
    assert all(record["checks"].values())
    gradients = record["gradients"]
    assert gradients["term_sum_abs_error"] <= 1.0e-9
    assert abs(gradients["captured_gradient_abs_error"]) <= 1.0e-12
    decomposition = record["adam"]["decomposition"]
    assert decomposition["reconstruction_relative_error"] <= 1.0e-9
    assert decomposition["contribution_sum_relative_error"] <= 1.0e-9
    assert decomposition["gC_dot_reconstruction"] == pytest.approx(
        record["objectives"]["gC_dot_delta"], rel=1.0e-8, abs=1.0e-12
    )
    assert decomposition["term_contribution_sum"] == pytest.approx(
        decomposition["gC_dot_reconstruction"], rel=1.0e-8, abs=1.0e-12
    )
    assert [item["scale"] for item in record["step_scaling"]] == [
        1.0, 0.5, 0.25, 0.125
    ]
    assert len(record["finite_difference"]) == 2
    assert pair._sha256(checkpoint) == before_hash


def test_direction_trial_restores_parameters_optimizer_rng_and_air(tmp_path):
    _trainer_obj, _checkpoint_path, record, _before = _trial(tmp_path)
    checks = record["checks"]
    assert checks["parameters_restored"]
    assert checks["optimizer_state_unchanged_by_interpolation"]
    assert checks["rng_unchanged_by_interpolation"]
    assert checks["air_state_unchanged_by_interpolation"]
    assert checks["air_frozen_during_contact_step"]
    assert checks["step_scale_one_matches_actual"]


def test_direction_issues_expose_broken_conditions():
    issues = pair._direction_issues(
        {"air_loss_equal": False, "parameters_restored": True}, False
    )
    assert issues == ["air_loss_equal=false"]
    assert pair._direction_issues({}, True) == [
        "non-finite direction quantity"
    ]


def test_adam_snapshot_rejects_missing_optimizer_state():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-4)
    with pytest.raises(RuntimeError, match="optimizer state is missing"):
        pair._adam_snapshot(optimizer, [parameter])


def test_direction_main_fails_for_empty_label_case(tmp_path, monkeypatch):
    trainer, checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        pair.diag, "_build_trainer", lambda config: (trainer, {})
    )
    monkeypatch.setattr(pair, "_full_labels", lambda case: {
        "times": np.zeros(0), "low": np.zeros(0), "up": np.zeros(0),
        "n": 0, "excluded_outside_time": 0, "excluded_no_slope": 0,
        "total_frame_candidates": 0,
    })
    output = tmp_path / "direction.json"
    code = pair.main([
        "--config", "unused", "--checkpoint", str(checkpoint),
        "--trials", "1", "--analysis", "direction",
        "--output", str(output),
    ])
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["failed_trials"] == 1
    assert "empty" in payload["trials"][0]["error"]


def _direction_summary_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_contact_adam_direction",
        ROOT / "scripts" / "summarize_contact_adam_direction.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_direction_run(directory: Path):
    trainer, checkpoint = _checkpoint(directory)
    labels = {
        case.case_id: pair._full_labels(case) for case in trainer.cases
    }
    reference = pair.diag._reference_points(trainer)
    record = pair._direction_trial(
        trainer, checkpoint, reference, labels,
        SimpleNamespace(seed=999), 0,
    )
    payload = {
        "format": "spherical_blast_pinn.contact_adam_direction",
        "version": 1,
        "config": str(directory / "cfg.yaml"),
        "checkpoint": str(checkpoint),
        "tag": "best",
        "checkpoint_loop": int(trainer.loop),
        "checkpoint_sha256": {
            "before": pair._sha256(checkpoint),
            "after": pair._sha256(checkpoint),
            "unchanged": True,
        },
        "requested_trials": 1,
        "completed_trials": 1,
        "failed_trials": 0,
        "non_finite": False,
        "seed": 999,
        "labels": labels,
        "full_contact_labels_per_case": {
            key: item["n"] for key, item in labels.items()
        },
        "reference_frames_per_trial": sum(
            item["n"] for item in labels.values()
        ),
        "trials": [record],
        "ok": True,
    }
    (directory / "cfg.yaml").write_text("unused", encoding="utf-8")
    (directory / "direction_best_seed999.json").write_text(
        json.dumps(payload, default=pair.diag._jsonable), encoding="utf-8"
    )
    manifest = {
        "format": "spherical_blast_pinn.contact_adam_direction.manifest",
        "version": 1,
        "config": {"path": str(directory / "cfg.yaml")},
        "checkpoints": {
            "best": {
                "path": str(checkpoint),
                "sha256": pair._sha256(checkpoint),
                "loop": int(trainer.loop),
            }
        },
        "seeds": [999],
        "trials_per_checkpoint_seed": 1,
        "training_cases": [case.case_id for case in trainer.cases],
    }
    (directory / "direction_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return payload


def test_direction_summary_accepts_valid_records(tmp_path):
    module = _direction_summary_module()
    _write_direction_run(tmp_path)
    assert module.main(["--directory", str(tmp_path)]) == 0


def test_direction_summary_rejects_broken_check(tmp_path):
    module = _direction_summary_module()
    payload = _write_direction_run(tmp_path)
    payload["trials"][0]["checks"]["parameters_restored"] = False
    (tmp_path / "direction_best_seed999.json").write_text(
        json.dumps(payload, default=pair.diag._jsonable), encoding="utf-8"
    )
    assert module.main(["--directory", str(tmp_path)]) == 1


def test_direction_summary_rejects_bad_gradient_sum(tmp_path):
    module = _direction_summary_module()
    payload = _write_direction_run(tmp_path)
    payload["trials"][0]["gradients"]["term_sum_abs_error"] = 1.0
    (tmp_path / "direction_best_seed999.json").write_text(
        json.dumps(payload, default=pair.diag._jsonable), encoding="utf-8"
    )
    assert module.main(["--directory", str(tmp_path)]) == 1


def test_direction_summary_rejects_bad_reconstruction(tmp_path):
    module = _direction_summary_module()
    payload = _write_direction_run(tmp_path)
    payload["trials"][0]["adam"]["decomposition"][
        "reconstruction_relative_error"
    ] = 1.0
    (tmp_path / "direction_best_seed999.json").write_text(
        json.dumps(payload, default=pair.diag._jsonable), encoding="utf-8"
    )
    assert module.main(["--directory", str(tmp_path)]) == 1
