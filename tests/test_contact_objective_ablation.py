"""Diagnostic ContactNet objective ablation: data-only target and isolation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
    "run_contact_objective_ablation",
    ROOT / "scripts" / "run_contact_objective_ablation.py",
)
assert _SPEC is not None and _SPEC.loader is not None
ablation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ablation)


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
        output_dir=tmp_path / "ablation",
        external_validation_cases=validation,
    )


def _checkpoint(tmp_path: Path):
    trainer = _trainer(tmp_path)
    trainer.step()
    trainer.loop = 2000  # lambda_data=0.3, coefficient=0.9 at n_schedule=10
    checkpoint = tmp_path / "start.pt"
    trainer.save(checkpoint)
    return trainer, checkpoint


def test_data_only_target_is_weighted_contact_data_and_air_is_frozen(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.loop = 2000
    batches = trainer._batches(generator=torch.Generator().manual_seed(13))
    air_total, lambda_pde, lambda_data = trainer._air_update(batches)
    assert lambda_data == 0.3
    params = [p for p in trainer.contact_net.parameters() if p.requires_grad]
    terms = [trainer.contact_stage_terms(case, batch) for case, batch in batches]
    aggregate_data = sum(term["contact_data"].total for term in terms) / len(terms)
    expected = ablation.diag._grad(
        lambda_data * trainer.config.contact_data_weight * aggregate_data,
        params,
    )[0]
    captured = {}

    def hook(_optimizer, _args, _kwargs):
        captured["gradient"] = torch.cat([
            parameter.grad.detach().reshape(-1) for parameter in params
        ])

    air_before = [parameter.detach().clone() for parameter in trainer.air_net.parameters()]
    contact_before = [parameter.detach().clone() for parameter in trainer.contact_net.parameters()]
    handle = trainer.opt_contact.register_step_pre_hook(hook)
    try:
        total, terms_out = trainer.contact_update(
            batches, lambda_pde=lambda_pde, lambda_data=lambda_data,
            objective="contact_data_only", return_terms=True,
        )
    finally:
        handle.remove()
    raw = ablation._term_values(terms_out)["contact_data"]["raw_mean"]
    assert float(total.total.detach()) == (
        lambda_data * trainer.config.contact_data_weight * raw
    )
    assert torch.allclose(captured["gradient"], expected, rtol=1e-10, atol=1e-12)
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(trainer.air_net.parameters(), air_before)
    )
    assert any(
        not torch.equal(parameter, before)
        for parameter, before in zip(trainer.contact_net.parameters(), contact_before)
    )
    assert torch.equal(air_total.total.detach(), air_total.total.detach())


def test_default_contact_objective_still_contains_all_four_terms(tmp_path):
    trainer = _trainer(tmp_path)
    batches = trainer._batches(generator=torch.Generator().manual_seed(17))
    _air, lambda_pde, lambda_data = trainer._air_update(batches)
    result = trainer.contact_update(
        batches, lambda_pde=lambda_pde, lambda_data=lambda_data
    )
    assert hasattr(result, "total")
    assert any(key.startswith("pde_") for key in result.parts)
    assert "contact_data" in result.parts


def test_preflight_checks_data_only_gradient_and_does_not_write_checkpoint(tmp_path):
    trainer, checkpoint = _checkpoint(tmp_path)
    before = ablation._sha256(checkpoint)
    output = tmp_path / "preflight"
    output.mkdir()
    preflight = ablation._preflight(trainer, checkpoint, output)
    assert all(preflight["checks"].values()), preflight["checks"]
    assert ablation._sha256(checkpoint) == before
    assert preflight["first_batch_equal_to_standard"]
    assert preflight["air_stage_equal"]
    assert preflight["airnet_frozen_during_contact"]
    assert abs(preflight["contact_data_coefficient"] - 0.9) < 1e-12
    saved = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert saved["checks"] == preflight["checks"]


def test_short_data_only_run_does_not_overwrite_start_checkpoint(tmp_path):
    trainer, checkpoint = _checkpoint(tmp_path)
    before = ablation._sha256(checkpoint)
    output = tmp_path / "run"
    result = ablation._train(
        trainer, checkpoint, 2002, output, checkpoint_every=1,
        objective="contact_data_only",
    )
    assert result["loops"] == 2002
    assert abs(result["history"][0]["contact_data_coefficient"] - 0.9) < 1e-12
    assert (output / "checkpoints" / "loop_2001.pt").is_file()
    assert (output / "checkpoints" / "loop_2002.pt").is_file()
    assert ablation._sha256(checkpoint) == before
    assert not (output / "last.pt").samefile(checkpoint)


def test_batch_from_copy_does_not_advance_trainer_generator(tmp_path):
    trainer = _trainer(tmp_path)
    state = trainer.generator.get_state().clone()
    first = ablation._batch_from_copy(trainer)
    second = ablation._batch_from_copy(trainer)
    assert ablation._batch_digest(first) == ablation._batch_digest(second)
    assert torch.equal(trainer.generator.get_state(), state)
