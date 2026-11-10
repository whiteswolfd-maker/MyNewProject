"""Alternating updates, gradient isolation, checkpoint and split roles."""
from __future__ import annotations

import torch
import pytest

from spherical_blast_pinn.contact_air.models import AirNet, ContactNet
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
from spherical_blast_pinn.contact_air.trainer import (
    AlternatingTrainer,
    TrainConfig,
    TrainingError,
)


def _trainer(tmp_path, *, cases=None, holdout=0.3):
    cases = cases or [
        synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=1600),
        synthetic_case_for(0.08, case_id="r080", n_frames=11, n_radial=1600),
    ]
    config = TrainConfig(
        n_pde=8, n_ic=4, n_right=4, n_data=8, n_contact=6,
        loops=2, learning_rate=1.0e-3, log_every=1, checkpoint_every=0,
        validate_every=0, seed=11, validation_holdout_fraction=holdout,
        validation_data_points=8, validation_contact_points=6,
    )
    return AlternatingTrainer(
        cases=cases, contact_net=ContactNet(width=8, depth=1),
        air_net=AirNet(width=8, depth=1), config=config, output_dir=tmp_path,
    )


def test_one_alternating_loop_updates_both_networks_and_keeps_gradients_isolated(tmp_path):
    trainer = _trainer(tmp_path)
    contact_before = [p.detach().clone() for p in trainer.contact_net.parameters()]
    air_before = [p.detach().clone() for p in trainer.air_net.parameters()]
    record = trainer.step()
    assert record["air_loss"] >= 0.0
    assert record["contact_loss"] >= 0.0
    assert any(not torch.equal(p, q) for p, q in zip(contact_before, trainer.contact_net.parameters()))
    assert any(not torch.equal(p, q) for p, q in zip(air_before, trainer.air_net.parameters()))
    assert record["counts"]["air.pde_points"] > 0
    assert record["counts"]["contact.contact_data"] > 0


def test_whole_frames_are_isolated_between_training_and_validation(tmp_path):
    trainer = _trainer(tmp_path)
    for train_case, validation_case in zip(trainer.cases, trainer.validation_cases):
        train_frames = set(train_case.contact.times.tolist())
        validation_frames = set(validation_case.contact.times.tolist())
        assert train_frames
        assert validation_frames
        assert train_frames.isdisjoint(validation_frames)
        assert train_case.t_end <= validation_case.contact.times.min()


def test_role_test_cannot_enter_training_but_radius_alone_does_not_ban_it(tmp_path):
    with pytest.raises(TrainingError, match="test"):
        _trainer(tmp_path, cases=[synthetic_case_for(0.05, role="test")])
    # There is no fixed 500 mm training ban; the explicit role decides.
    trainer = _trainer(
        tmp_path, cases=[synthetic_case_for(0.5, case_id="r500", role="train",
                                            n_frames=7, n_radial=1600)],
        holdout=0.0,
    )
    assert trainer.cases[0].charge_radius_m == pytest.approx(0.5)


def test_checkpoint_round_trip_and_resume(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()
    trainer.step()
    path = trainer.save(tmp_path / "last.pt")
    assert path.is_file()

    fresh = _trainer(tmp_path / "fresh")
    info = fresh.load(path)
    assert info["loop"] == 2
    assert info["stage_state"] == "alternating"
    for left, right in zip(trainer.contact_net.state_dict().values(),
                           fresh.contact_net.state_dict().values()):
        assert torch.equal(left, right)
    for left, right in zip(trainer.air_net.state_dict().values(),
                           fresh.air_net.state_dict().values()):
        assert torch.equal(left, right)
    assert fresh.opt_air.state_dict()["param_groups"][0]["lr"] == pytest.approx(
        trainer.opt_air.state_dict()["param_groups"][0]["lr"]
    )
    assert set(fresh.opt_air.state_dict()["state"]) == set(
        trainer.opt_air.state_dict()["state"]
    )
    fresh.step()
    assert fresh.loop == 3


def test_air_stage_pde_gradient_reaches_only_airnet(tmp_path):
    from spherical_blast_pinn.contact_air.losses import pde_loss
    from spherical_blast_pinn.contact_air.mapping import build_geometry
    from spherical_blast_pinn.contact_air.models import air_fields_with_derivatives
    from spherical_blast_pinn.contact_air.sampling import sample_case_batch

    trainer = _trainer(tmp_path)
    case = trainer.cases[0]
    batch = sample_case_batch(
        case, n_pde=8, n_ic=0, n_right=0, n_data=0, n_contact=0,
        generator=trainer.generator,
    )
    closure = trainer._closures[case.case_id]
    scales = trainer._scales[case.case_id]
    trainer.contact_net.zero_grad(set_to_none=True)
    trainer.air_net.zero_grad(set_to_none=True)
    r0, rs, ds = trainer._contact_inputs(case, batch.t_pde)
    with torch.enable_grad():
        c_fields = trainer.contact_net(
            batch.t_pde.reshape(-1, 1), r0, rs, ds, closure, create_graph=False
        )
    c_fields = type(c_fields)(
        radius=c_fields.radius.detach(), speed=c_fields.speed.detach(),
        sigmoid=c_fields.sigmoid.detach(), raw=c_fields.raw.detach(),
        pressure=c_fields.pressure.detach(),
        relative_volume=c_fields.relative_volume.detach(),
    )
    geometry = build_geometry(
        c_fields.radius, c_fields.speed, rs, ds, batch.zeta_pde.reshape(-1, 1)
    )
    fields, derivatives = air_fields_with_derivatives(
        trainer.air_net, batch.zeta_pde, batch.t_pde, r0, scales, create_graph=True
    )
    pde_loss(fields=fields, derivatives=derivatives, geometry=geometry, case=case).total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in trainer.air_net.parameters())
    assert all(p.grad is None for p in trainer.contact_net.parameters())


def test_contact_stage_shared_physics_gradient_reaches_contactnet_not_airnet(tmp_path):
    from spherical_blast_pinn.contact_air.losses import pde_loss
    from spherical_blast_pinn.contact_air.mapping import build_geometry
    from spherical_blast_pinn.contact_air.models import air_fields_with_derivatives
    from spherical_blast_pinn.contact_air.sampling import sample_case_batch

    trainer = _trainer(tmp_path)
    case = trainer.cases[0]
    batch = sample_case_batch(
        case, n_pde=8, n_ic=0, n_right=0, n_data=0, n_contact=0,
        generator=trainer.generator,
    )
    closure = trainer._closures[case.case_id]
    scales = trainer._scales[case.case_id]
    trainer.contact_net.zero_grad(set_to_none=True)
    trainer.air_net.zero_grad(set_to_none=True)
    r0, rs, ds = trainer._contact_inputs(case, batch.t_pde)
    with torch.enable_grad():
        c_fields = trainer.contact_net(
            batch.t_pde.reshape(-1, 1), r0, rs, ds, closure, create_graph=True
        )
        geometry = build_geometry(
            c_fields.radius, c_fields.speed, rs, ds, batch.zeta_pde.reshape(-1, 1)
        )
        a_fields, a_derivs = air_fields_with_derivatives(
            trainer.air_net, batch.zeta_pde, batch.t_pde, r0, scales,
            create_graph=True, functional=True,
        )
        pde_loss(
            fields=a_fields, derivatives=a_derivs, geometry=geometry, case=case
        ).total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in trainer.contact_net.parameters())
    assert all(p.grad is None for p in trainer.air_net.parameters())


def test_data_pde_schedule_moves_from_data_heavy_to_pde_heavy():
    from spherical_blast_pinn.contact_air.losses import schedule_weights

    pde0, data0 = schedule_weights(0, 100)
    pde50, data50 = schedule_weights(50, 100)
    pde100, data100 = schedule_weights(100, 100)
    assert data0 == pytest.approx(0.9)
    assert data50 == pytest.approx(0.6)
    assert data100 == pytest.approx(0.3)
    assert pde0 == pytest.approx(0.1)
    assert pde100 == pytest.approx(0.7)


def test_multi_radius_uses_one_shared_pair_of_networks(tmp_path):
    trainer = _trainer(tmp_path)
    assert len(trainer.cases) == 2
    assert not hasattr(trainer, "models_by_case")
