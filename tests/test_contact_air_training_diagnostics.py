"""Regression tests for the contact training diagnostic itself."""
from __future__ import annotations

import importlib.util
import json
import math
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


def test_macro_and_pooled_use_different_weights():
    cases = [
        {"case_id": "a", "n": 10, "covered_count": 7, "mse": 4.0,
         "mean_violation_over_Rsep": 2.0, "max_violation_over_Rsep": 3.0},
        {"case_id": "b", "n": 2, "covered_count": 0, "mse": 1.0,
         "mean_violation_over_Rsep": 1.0, "max_violation_over_Rsep": 1.0},
    ]
    aggregate = diagnostic._aggregate_metrics(cases, expected_cases=2)
    assert aggregate["macro_mse"] == pytest.approx((4.0 + 1.0) / 2.0)
    assert aggregate["macro_rmse"] == pytest.approx(math.sqrt(2.5))
    assert aggregate["pooled_mse"] == pytest.approx((10 * 4.0 + 2 * 1.0) / 12)
    assert aggregate["pooled_rmse"] == pytest.approx(math.sqrt(42 / 12))
    assert aggregate["covered_count"] == 7
    assert aggregate["n"] == 12
    assert aggregate["coverage"] == pytest.approx(7 / 12)
    assert aggregate["complete_training_target"] is True


def test_macro_can_worsen_while_pooled_improves():
    before = diagnostic._aggregate_metrics([
        {"case_id": "a", "n": 10, "covered_count": 0, "mse": 4.0,
         "mean_violation_over_Rsep": 2.0, "max_violation_over_Rsep": 2.0},
        {"case_id": "b", "n": 2, "covered_count": 0, "mse": 1.0,
         "mean_violation_over_Rsep": 1.0, "max_violation_over_Rsep": 1.0},
    ], expected_cases=2)
    after = diagnostic._aggregate_metrics([
        {"case_id": "a", "n": 10, "covered_count": 0, "mse": 1.0,
         "mean_violation_over_Rsep": 1.0, "max_violation_over_Rsep": 1.0},
        {"case_id": "b", "n": 2, "covered_count": 0, "mse": 10.0,
         "mean_violation_over_Rsep": 10.0, "max_violation_over_Rsep": 10.0},
    ], expected_cases=2)
    change = diagnostic._reference_change(before, after)
    assert change["macro_mse"] > 0
    assert change["pooled_mse"] < 0


def test_exact_coverage_counts_are_integer_and_dtype_stable():
    for dtype in (torch.float32, torch.float64):
        radius = torch.tensor([0.0] * 7 + [1.0] * 3, dtype=dtype)
        low = torch.zeros(10, dtype=dtype)
        up = torch.zeros(10, dtype=dtype)
        stats = diagnostic._interval_stats(radius, low, up, 1.0)
        assert stats["covered_count"] == 7
        assert stats["n"] == 10
        assert stats["coverage"] == pytest.approx(0.7)
    assert diagnostic._interval_stats(
        torch.ones(3), torch.zeros(3), torch.zeros(3), 1.0
    )["covered_count"] == 0
    assert diagnostic._interval_stats(
        torch.zeros(3), torch.zeros(3), torch.zeros(3), 1.0
    )["covered_count"] == 3


def test_empty_case_is_missing_not_zero_error():
    aggregate = diagnostic._aggregate_metrics([
        {"case_id": "a", "n": 0, "covered_count": 0, "mse": None,
         "mean_violation_over_Rsep": None, "max_violation_over_Rsep": None},
        {"case_id": "b", "n": 4, "covered_count": 2, "mse": 2.0,
         "mean_violation_over_Rsep": 1.0, "max_violation_over_Rsep": 1.0},
    ], expected_cases=2)
    assert aggregate["macro_mse"] == pytest.approx(2.0)
    assert aggregate["pooled_mse"] == pytest.approx(2.0)
    assert aggregate["n_cases_missing"] == ["a"]
    assert aggregate["complete_training_target"] is False


def test_segment_boundaries_are_counted_once():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    masks = diagnostic._third_masks(times, 0.0, 3.0)
    assert [int(mask.sum()) for mask in masks] == [1, 1, 2]
    assert sum(int(mask.sum()) for mask in masks) == times.size
    assert not np.any(masks[0] & masks[1])
    assert not np.any(masks[1] & masks[2])


def test_failure_reasons_cover_adam_and_hash_conditions():
    reasons = diagnostic._failure_reasons(
        static_requested=5, static_completed=4, static_failed=1,
        adam_requested=10, adam_completed=9, adam_failed=1,
        adam_available=True, non_finite=False, checkpoint_unchanged=False,
    )
    text = " | ".join(reasons)
    assert "static failures" in text
    assert "Adam failures" in text
    assert "checkpoint SHA-256 changed" in text
    assert diagnostic._failure_reasons(
        static_requested=1, static_completed=1, static_failed=0,
        adam_requested=0, adam_completed=0, adam_failed=0,
        adam_available=False, non_finite=False, checkpoint_unchanged=True,
    ) == []
    missing = diagnostic._failure_reasons(
        static_requested=1, static_completed=1, static_failed=0,
        adam_requested=3, adam_completed=0, adam_failed=0,
        adam_available=False, non_finite=False, checkpoint_unchanged=True,
    )
    assert any("missing" in item for item in missing)


def test_adam_exception_returns_nonzero_and_keeps_reason(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, seed=31)
    trainer.step()  # create real Adam state in the checkpoint
    checkpoint = tmp_path / "base.pt"
    trainer.save(checkpoint)
    monkeypatch.setattr(
        diagnostic, "_build_trainer", lambda config: (trainer, {})
    )
    def boom(*args, **kwargs):
        raise RuntimeError("injected Adam failure")
    monkeypatch.setattr(diagnostic, "_adam_trial", boom)
    output = tmp_path / "out.json"
    code = diagnostic.main([
        "--config", "unused.yaml", "--checkpoint", str(checkpoint),
        "--batches", "1", "--adam-trials", "2", "--output", str(output),
    ])
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["adam_failed"] == 2
    assert payload["ok"] is False
    assert all("injected Adam failure" in item["error"]
               for item in payload["adam_trials"])
    assert any("Adam failures" in item for item in payload["failure_reasons"])


def _checkpoint_without_optimizer_state(tmp_path):
    trainer = _trainer(tmp_path, seed=41)
    path = tmp_path / "no_optimizer.pt"
    trainer.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["opt_contact"] = {"state": {}, "param_groups": []}
    torch.save(payload, path)
    return trainer, path


def test_adam_request_without_optimizer_state_fails(tmp_path, monkeypatch):
    trainer, checkpoint = _checkpoint_without_optimizer_state(tmp_path)
    monkeypatch.setattr(
        diagnostic, "_build_trainer", lambda config: (trainer, {})
    )
    output = tmp_path / "out.json"
    code = diagnostic.main([
        "--config", "unused.yaml", "--checkpoint", str(checkpoint),
        "--batches", "1", "--adam-trials", "1", "--output", str(output),
    ])
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["adam_available"] is False
    assert payload["adam_unavailable"] is True
    assert any("optimizer state" in item for item in payload["failure_reasons"])


def test_static_only_allows_checkpoint_without_optimizer_state(tmp_path, monkeypatch):
    trainer, checkpoint = _checkpoint_without_optimizer_state(tmp_path)
    monkeypatch.setattr(
        diagnostic, "_build_trainer", lambda config: (trainer, {})
    )
    output = tmp_path / "out.json"
    code = diagnostic.main([
        "--config", "unused.yaml", "--checkpoint", str(checkpoint),
        "--batches", "1", "--adam-trials", "0", "--output", str(output),
    ])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["adam_requested"] == 0
