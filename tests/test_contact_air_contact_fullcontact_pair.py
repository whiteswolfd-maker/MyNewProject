"""Paired random-contact vs full-contact one-step experiment helpers."""
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
    "contact_fullcontact_pair",
    ROOT / "scripts" / "contact_fullcontact_pair.py",
)
assert _SPEC is not None and _SPEC.loader is not None
pair = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pair)


def _trainer(tmp_path: Path, seed: int = 7) -> AlternatingTrainer:
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
    seed_everything(seed)
    contact = ContactNet(width=8, depth=2, time_reference_s=0.01)
    air = AirNet(width=8, depth=2, time_reference_s=0.01)
    config = TrainConfig(
        n_pde=40, n_ic=8, n_right=8, n_data=8, n_contact=8,
        loops=2, n_schedule=10, seed=seed,
        validation_holdout_fraction=0.0,
        validation_data_points=8, validation_contact_points=8,
        contact_data_weight=3.0,
    )
    return AlternatingTrainer(
        cases=train, contact_net=contact, air_net=air, config=config,
        output_dir=tmp_path / f"pair{seed}",
        external_validation_cases=validation,
    )


def _labels(trainer):
    return {case.case_id: pair._full_labels(case) for case in trainer.cases}


def _mass_mean(trainer, case, labels, batch):
    t = torch.as_tensor(labels["times"], dtype=torch.float64)
    fields = pair.diag._contact_at(trainer, case, t, create_graph=True)
    low = torch.as_tensor(labels["low"], dtype=torch.float64)
    up = torch.as_tensor(labels["up"], dtype=torch.float64)
    scale = float(case.length_scale)
    below = torch.clamp(low - fields.radius.reshape(-1), min=0.0)
    above = torch.clamp(fields.radius.reshape(-1) - up, min=0.0)
    values = (below / scale) ** 2 + (above / scale) ** 2
    return float(values.mean())


def test_full_contact_loss_is_per_case_mean_then_equal_case_mean(tmp_path):
    trainer = _trainer(tmp_path)
    labels = _labels(trainer)
    batches = trainer._batches(generator=torch.Generator().manual_seed(3))
    full = pair._all_label_batches(trainer, batches, labels)
    case_means = []
    for case, batch in full:
        terms = trainer.contact_stage_terms(case, batch)
        assert terms["contact_data"].counts["contact_data"] == labels[case.case_id]["n"]
        independent = _mass_mean(
            trainer, case, labels[case.case_id], batch
        )
        assert float(terms["contact_data"].total.detach()) == pytest.approx(
            independent, rel=1.0e-12, abs=1.0e-15
        )
        case_means.append(independent)
    expected = sum(case_means) / len(case_means)
    record = pair.diag._static_record(
        trainer, full, compute_gradients=False, threshold=1.0e-12
    )
    assert record["terms"]["contact_data"]["raw_loss"] == pytest.approx(
        expected, rel=1.0e-12, abs=1.0e-15
    )


def test_full_labels_match_fixed_reference_selection(tmp_path):
    trainer = _trainer(tmp_path)
    labels = _labels(trainer)
    reference = pair.diag._reference_points(trainer)
    assert sum(item["n"] for item in labels.values()) == sum(
        len(point["times"]) for point in reference
    )
    for point in reference:
        assert labels[point["case_id"]]["n"] == len(point["times"])
    assert set(labels) == {case.case_id for case in trainer.cases}
    assert "s150" not in labels


def test_all_label_batches_change_only_contact_labels(tmp_path):
    trainer = _trainer(tmp_path)
    labels = _labels(trainer)
    batches = trainer._batches(generator=torch.Generator().manual_seed(4))
    full = pair._all_label_batches(trainer, batches, labels)
    assert pair._non_contact_equal(batches, full)
    for (case, original), (_, changed) in zip(batches, full):
        assert changed.t_contact.numel() == labels[case.case_id]["n"]


def test_paired_trial_starts_from_identical_state_and_preserves_checkpoint(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()  # ensure Adam state exists
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    before = pair._sha256(checkpoint)
    reference = pair.diag._reference_points(trainer)
    labels = _labels(trainer)
    args = SimpleNamespace(seed=999)
    record = pair._trial(
        trainer, checkpoint, reference, labels, args, 0
    )
    assert record["paired"]["initial_param_equal"] is True
    assert record["paired"]["initial_optimizer_equal"] is True
    assert record["paired"]["pre_contact_param_equal"] is True
    assert record["paired"]["pre_contact_optimizer_equal"] is True
    assert record["paired"]["air_state_equal"] is True
    assert record["paired"]["air_loss_equal"] is True
    assert record["non_contact_sampling_equal"] is True
    assert record["full_contact_labels"] > 0
    assert pair._sha256(checkpoint) == before


def test_paired_trials_are_repeatable(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    reference = pair.diag._reference_points(trainer)
    labels = _labels(trainer)
    args = SimpleNamespace(seed=321)
    first = pair._trial(trainer, checkpoint, reference, labels, args, 0)
    second = pair._trial(trainer, checkpoint, reference, labels, args, 0)
    assert first["paired"]["reference_macro_mse_difference_B_minus_A"] == pytest.approx(
        second["paired"]["reference_macro_mse_difference_B_minus_A"],
        rel=0.0, abs=0.0,
    )
    assert first["A"]["reference_change"] == second["A"]["reference_change"]
    assert first["B"]["reference_change"] == second["B"]["reference_change"]


def test_empty_full_label_set_is_rejected(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    reference = pair.diag._reference_points(trainer)
    labels = _labels(trainer)
    for value in labels.values():
        value["n"] = 0
        value["times"] = np.zeros(0)
        value["low"] = np.zeros(0)
        value["up"] = np.zeros(0)
    with pytest.raises(RuntimeError, match="empty"):
        pair._trial(
            trainer, checkpoint, reference, labels,
            SimpleNamespace(seed=1), 0,
        )


def test_pair_main_fails_when_full_label_set_is_empty(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.step()
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    monkeypatch.setattr(pair.diag, "_build_trainer", lambda config: (trainer, {}))
    monkeypatch.setattr(pair, "_full_labels", lambda case: {
        "times": np.zeros(0), "low": np.zeros(0), "up": np.zeros(0),
        "n": 0, "excluded_outside_time": 0, "excluded_no_slope": 0,
        "total_frame_candidates": 0,
    })
    output = tmp_path / "out.json"
    code = pair.main([
        "--config", "unused", "--checkpoint", str(checkpoint),
        "--trials", "1", "--output", str(output),
    ])
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["failed_trials"] == 1
    assert "empty" in payload["trials"][0]["error"]
