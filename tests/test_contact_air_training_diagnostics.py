"""Regression tests for the contact training diagnostic itself."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import dataclasses
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
    "diagnose_contact_gradients",
    ROOT / "scripts" / "diagnose_contact_gradients.py",
)
assert _SPEC is not None and _SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


def _trainer(tmp_path: Path, seed: int = 11) -> AlternatingTrainer:
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
        output_dir=tmp_path / f"run{seed}",
        external_validation_cases=validation,
    )


def test_requested_batches_are_not_truncated(tmp_path):
    trainer = _trainer(tmp_path)
    seen = []
    for index in range(100):
        batches, which = diagnostic._make_batches(
            trainer, "training", index, 123
        )
        assert len(batches) == len(trainer.cases)
        seen.append(which)
    assert len(seen) == 100
    assert all(value is None for value in seen)


def test_stratified_ranges_cover_pde_contact_and_keep_ic_at_t0(tmp_path):
    trainer = _trainer(tmp_path)
    for index, which in enumerate((0, 1, 2)):
        batches, reported = diagnostic._make_batches(
            trainer, "stratified", index, 123
        )
        assert reported == which
        for case, batch in batches:
            span = float(case.t_end) - float(case.t0)
            lo = float(case.t0) + (which / 3.0) * span
            hi = float(case.t0) + ((which + 1) / 3.0) * span
            if batch.t_pde.numel():
                assert float(batch.t_pde.min()) >= lo - 1.0e-12
                assert float(batch.t_pde.max()) <= hi + 1.0e-12
            if batch.t_contact.numel():
                assert float(batch.t_contact.min()) >= lo - 1.0e-12
                assert float(batch.t_contact.max()) <= hi + 1.0e-12
            assert torch.allclose(
                batch.t_ic, torch.full_like(batch.t_ic, float(case.t0))
            )


def test_empty_contact_slice_is_not_replaced_by_the_full_window(tmp_path):
    trainer = _trainer(tmp_path)
    batches, _ = diagnostic._make_batches(trainer, "training", 0, 123)
    empty = torch.zeros(0, dtype=torch.float64)
    batches = [
        (case, dataclasses.replace(batch, t_contact=empty,
                                  low_contact=empty, up_contact=empty))
        for case, batch in batches
    ]
    record = diagnostic._static_record(
        trainer, batches, compute_gradients=True, threshold=1.0e-12
    )
    assert all(not entry["contact_evaluable"] for entry in record["per_case"])
    assert record["contact_evaluable"] is False
    assert record["terms"]["contact_data"]["raw_loss"] == 0.0
    assert record["aggregate_gradients"]["cos_data_vs_other"]["status"] == (
        "direction_undefined"
    )
    record.pop("_vectors", None)
    record.pop("_vector_g_data", None)


def test_aggregate_gradient_matches_formal_total_backward(tmp_path):
    trainer = _trainer(tmp_path)
    batches, _ = diagnostic._make_batches(trainer, "training", 0, 123)
    record = diagnostic._static_record(
        trainer, batches, compute_gradients=True, threshold=1.0e-12
    )
    vectors = record.pop("_vectors")
    record.pop("_vector_g_data", None)
    per_case = diagnostic._per_case_terms(trainer, batches)
    aggregate = diagnostic._aggregate_terms(per_case)
    coeffs = diagnostic._coefficients(trainer)
    total = sum(coeffs[name] * aggregate[name].total for name in diagnostic._TERM_NAMES)
    params = [p for p in trainer.contact_net.parameters() if p.requires_grad]
    formal = torch.autograd.grad(total, params, retain_graph=True)
    combined = torch.cat([value.reshape(-1) for value in formal])
    summed = sum(vectors[name] for name in diagnostic._TERM_NAMES)
    assert torch.allclose(combined, summed, rtol=1.0e-10, atol=1.0e-12)


def test_validation_labels_do_not_change_training_gradient(tmp_path):
    trainer = _trainer(tmp_path)
    batches, _ = diagnostic._make_batches(trainer, "training", 0, 123)
    first = diagnostic._static_record(
        trainer, batches, compute_gradients=True, threshold=1.0e-12
    )
    g_first = first["_vectors"]["contact_data"].clone()
    first.pop("_vectors", None)
    first.pop("_vector_g_data", None)
    validation = trainer.validation_cases[0]
    validation.contact.low = validation.contact.low + 1.0
    validation.contact.up = validation.contact.up + 1.0
    second = diagnostic._static_record(
        trainer, batches, compute_gradients=True, threshold=1.0e-12
    )
    g_second = second["_vectors"]["contact_data"].clone()
    assert torch.equal(g_first, g_second)


def test_single_adam_trial_matches_formal_step(tmp_path):
    trainer = _trainer(tmp_path, seed=17)
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    reference = diagnostic._reference_points(trainer)
    args = SimpleNamespace(adam_seed=999, zero_threshold=1.0e-12,
                           prediction_dir=None, prediction_tag="best")
    trial = diagnostic._adam_trial(
        trainer, checkpoint, reference, args, 0
    )

    formal = _trainer(tmp_path, seed=17)
    formal.load(checkpoint)
    formal.generator = torch.Generator().manual_seed(999)
    formal.step()

    for left, right in zip(
        trainer.contact_net.parameters(), formal.contact_net.parameters()
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        trainer.air_net.parameters(), formal.air_net.parameters()
    ):
        assert torch.equal(left, right)
    assert diagnostic._adam_summary(trainer.opt_contact) == (
        diagnostic._adam_summary(formal.opt_contact)
    )
    assert trial["status"] == "ok"
    assert len(trial["parameter_after_sha256"]) == 64


def test_adam_trials_are_independent_and_do_not_modify_the_file(tmp_path):
    trainer = _trainer(tmp_path, seed=23)
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    before = diagnostic._sha256(checkpoint)
    reference = diagnostic._reference_points(trainer)
    args = SimpleNamespace(adam_seed=777, zero_threshold=1.0e-12,
                           prediction_dir=None, prediction_tag="best")
    first = diagnostic._adam_trial(trainer, checkpoint, reference, args, 0)
    second = diagnostic._adam_trial(trainer, checkpoint, reference, args, 0)
    assert first["parameter_after_sha256"] == second["parameter_after_sha256"]
    assert first["parameter_delta"] == second["parameter_delta"]
    assert diagnostic._sha256(checkpoint) == before


def test_diagnostic_batches_do_not_touch_the_trainer_generator(tmp_path):
    trainer = _trainer(tmp_path)
    state = trainer.generator.get_state().clone()
    diagnostic._make_batches(trainer, "training", 0, 123)
    diagnostic._make_batches(trainer, "stratified", 1, 123)
    assert torch.equal(trainer.generator.get_state(), state)


def test_nonfinite_gradient_is_not_reported_as_a_valid_angle():
    a = torch.tensor([1.0, float("nan")], dtype=torch.float64)
    b = torch.tensor([1.0, 0.0], dtype=torch.float64)
    result = diagnostic._cosine(a, b, 1.0e-12)
    assert result["status"] == "non_finite"
    assert result["cosine"] is None


def test_nonpositive_batch_count_is_rejected():
    with pytest.raises(SystemExit):
        diagnostic.main([
            "--config", "unused.yaml", "--checkpoint", "unused.pt",
            "--batches", "0", "--output", "unused.json",
        ])


def test_stratified_adam_is_rejected():
    with pytest.raises(SystemExit):
        diagnostic.main([
            "--config", "unused.yaml", "--checkpoint", "unused.pt",
            "--mode", "stratified", "--adam-trials", "1",
            "--output", "unused.json",
        ])
