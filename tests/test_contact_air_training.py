"""Alternating updates, gradient isolation, checkpoint and split roles."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import pytest

from spherical_blast_pinn.contact_air.models import AirNet, ContactNet
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
from spherical_blast_pinn.contact_air.trainer import (
    AlternatingTrainer,
    TrainConfig,
    TrainingError,
    seed_everything,
)


def _trainer(tmp_path, *, cases=None, holdout=0.3, external=None,
             time_reference_s=1.0, contact_data_weight=1.0, log_every=1):
    cases = cases or [
        synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=1600),
        synthetic_case_for(0.08, case_id="r080", n_frames=11, n_radial=1600),
    ]
    config = TrainConfig(
        n_pde=8, n_ic=4, n_right=4, n_data=8, n_contact=6,
        loops=2, learning_rate=1.0e-3, log_every=log_every, checkpoint_every=0,
        validate_every=0, seed=11, validation_holdout_fraction=holdout,
        validation_data_points=8, validation_contact_points=6,
        contact_data_weight=contact_data_weight,
    )
    return AlternatingTrainer(
        cases=cases, contact_net=ContactNet(width=8, depth=1,
                                            time_reference_s=time_reference_s),
        air_net=AirNet(width=8, depth=1, time_reference_s=time_reference_s),
        config=config, output_dir=tmp_path,
        external_validation_cases=external or (),
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


def test_external_validation_band_is_scored_and_never_trained(tmp_path):
    train = [synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=1600)]
    external = synthetic_case_for(
        0.55, case_id="r550", role="validation", n_frames=7, n_radial=1600
    )
    trainer = _trainer(tmp_path, cases=train, holdout=0.0, external=[external])
    assert [case.case_id for case in trainer.cases] == ["r050"]
    assert [case.case_id for case in trainer.validation_cases] == ["r550"]

    verdict = trainer.evaluate()
    assert verdict["evaluable"] is True
    assert verdict["n_cases"] == 1
    assert verdict["cases"][0]["case_id"] == "r550"


def test_external_validation_requires_validation_role_and_disjoint_id(tmp_path):
    train = [synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=1600)]
    with pytest.raises(TrainingError, match="role: validation"):
        _trainer(
            tmp_path, cases=train, holdout=0.0,
            external=[synthetic_case_for(
                0.55, case_id="r550", role="train", n_frames=7, n_radial=1600
            )],
        )
    with pytest.raises(TrainingError, match="both"):
        _trainer(
            tmp_path, cases=train, holdout=0.0,
            external=[synthetic_case_for(
                0.05, case_id="r050", role="validation", n_frames=7, n_radial=1600
            )],
        )


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


_INIT_SCRIPT = """
import hashlib
from spherical_blast_pinn.contact_air.cli import _networks
from spherical_blast_pinn.contact_air.trainer import seed_everything

section = {
    "networks": {
        "dtype": "float64",
        "contact": {"width": 8, "depth": 1},
        "air": {"width": 8, "depth": 1},
    },
    "training": {"seed": 4242},
}
seed_everything(4242)
contact, air = _networks(section)
digest = hashlib.sha256()
for parameter in list(contact.parameters()) + list(air.parameters()):
    digest.update(parameter.detach().numpy().tobytes())
print(digest.hexdigest())
"""


def test_seeded_initialisation_is_identical_across_independent_processes():
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    digests = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", _INIT_SCRIPT],
            env=env, capture_output=True, text=True, check=True,
        )
        digests.append(completed.stdout.strip())
    assert digests[0]
    assert digests[0] == digests[1]


def test_training_log_records_the_actual_learning_rate(tmp_path):
    trainer = _trainer(tmp_path)
    result = trainer.train(loops=1)
    record = result["history"]["loops"][0]
    assert record["learning_rate"] == pytest.approx(1.0e-3)
    assert record["contact_learning_rate"] == pytest.approx(1.0e-3)
    assert record["air_learning_rate"] == pytest.approx(1.0e-3)


def _optimizer_state_copy(optimizer):
    return [
        {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in optimizer.state[parameter].items()
        }
        for parameter in optimizer.param_groups[0]["params"]
    ]


def _assert_optimizer_state_equal(optimizer, expected):
    states = [
        optimizer.state[parameter]
        for parameter in optimizer.param_groups[0]["params"]
    ]
    assert len(states) == len(expected)
    for state, reference in zip(states, expected):
        assert set(state) == set(reference)
        for name, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, reference[name])


def test_contact_learning_rate_override_only_changes_contact_optimizer(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()
    checkpoint = trainer.save(tmp_path / "base.pt")
    parameters_before = [
        parameter.detach().clone() for parameter in trainer.contact_net.parameters()
    ]
    contact_state = _optimizer_state_copy(trainer.opt_contact)
    air_state = _optimizer_state_copy(trainer.opt_air)
    contact_step = int(next(iter(trainer.opt_contact.state.values()))["step"])

    resumed = _trainer(tmp_path / "resumed")
    resumed.load(checkpoint)
    torch_rng = torch.get_rng_state().clone()
    generator_state = resumed.generator.get_state().clone()
    resumed.set_contact_learning_rate(2.5e-5)

    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(resumed.generator.get_state(), generator_state)
    assert resumed.effective_learning_rates() == {
        "contact": 2.5e-5, "air": 1.0e-3,
    }
    for parameter, expected in zip(
        resumed.contact_net.parameters(), parameters_before
    ):
        assert torch.equal(parameter, expected)
    _assert_optimizer_state_equal(resumed.opt_contact, contact_state)
    _assert_optimizer_state_equal(resumed.opt_air, air_state)
    assert int(next(iter(resumed.opt_contact.state.values()))["step"]) == contact_step


def test_contact_learning_rate_is_saved_and_restored(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()
    trainer.set_contact_learning_rate(2.5e-5)
    checkpoint = trainer.save(tmp_path / "saved.pt")

    restored = _trainer(tmp_path / "restored")
    restored.load(checkpoint)
    assert restored.effective_learning_rates() == {
        "contact": 2.5e-5, "air": 1.0e-3,
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["effective_learning_rates"]["contact"] == pytest.approx(2.5e-5)
    assert payload["effective_learning_rates"]["air"] == pytest.approx(1.0e-3)


def test_contact_learning_rate_is_recorded_in_history(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.set_contact_learning_rate(2.5e-5)
    result = trainer.train(loops=1)
    record = result["history"]["loops"][0]
    assert record["contact_learning_rate"] == pytest.approx(2.5e-5)
    assert record["air_learning_rate"] == pytest.approx(1.0e-3)
    assert record["learning_rate"] == pytest.approx(2.5e-5)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_contact_learning_rate_rejects_nonpositive_or_nonfinite(tmp_path, rate):
    trainer = _trainer(tmp_path)
    with pytest.raises(TrainingError, match="finite positive"):
        trainer.set_contact_learning_rate(rate)


def test_resume_accepts_the_same_time_reference(tmp_path):
    trainer = _trainer(tmp_path, time_reference_s=0.01)
    path = trainer.save(tmp_path / "scaled.pt")
    same = _trainer(tmp_path / "same", time_reference_s=0.01)
    assert same.load(path)["loop"] == 0


def test_resume_rejects_a_different_time_reference(tmp_path):
    trainer = _trainer(tmp_path, time_reference_s=0.01)
    path = trainer.save(tmp_path / "scaled.pt")
    different = _trainer(tmp_path / "different", time_reference_s=1.0)
    with pytest.raises(TrainingError, match="time_reference"):
        different.load(path)


def test_resume_treats_a_legacy_checkpoint_as_seconds(tmp_path):
    trainer = _trainer(tmp_path, time_reference_s=1.0)
    path = trainer.save(tmp_path / "legacy.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["contact_net_describe"].pop("time_reference_s")
    payload["air_net_describe"].pop("time_reference_s")
    torch.save(payload, path)
    seconds = _trainer(tmp_path / "seconds", time_reference_s=1.0)
    assert seconds.load(path)["loop"] == 0
    scaled = _trainer(tmp_path / "scaled", time_reference_s=0.01)
    with pytest.raises(TrainingError, match="time_reference"):
        scaled.load(path)


def test_contact_data_weight_is_recorded_and_scales_only_the_interval_term(tmp_path):
    from spherical_blast_pinn.contact_air.losses import TermResult, combine_contact

    pde = TermResult(total=torch.tensor(1.0, dtype=torch.float64))
    ic = TermResult(total=torch.tensor(2.0, dtype=torch.float64))
    interface = TermResult(total=torch.tensor(3.0, dtype=torch.float64))
    data = TermResult(total=torch.tensor(4.0, dtype=torch.float64))
    plain = combine_contact(
        pde=pde, ic=ic, interface=interface, data=data,
        lambda_pde=0.5, lambda_data=0.2,
    )
    weighted = combine_contact(
        pde=pde, ic=ic, interface=interface, data=data,
        lambda_pde=0.5, lambda_data=0.2, contact_data_weight=3.0,
    )
    assert plain.total.item() == pytest.approx(0.5 + 0.2 * 4.0 + 2.0 + 3.0)
    assert weighted.total.item() == pytest.approx(0.5 + 3.0 * 0.2 * 4.0 + 2.0 + 3.0)

    trainer = _trainer(tmp_path, contact_data_weight=3.0)
    result = trainer.train(loops=1)
    assert result["history"]["loops"][0]["contact_data_weight"] == pytest.approx(3.0)
    with pytest.raises(TrainingError, match="contact_data_weight"):
        TrainConfig(contact_data_weight=-1.0)


def test_resume_rejects_a_checkpoint_without_model_description(tmp_path):
    trainer = _trainer(tmp_path, time_reference_s=1.0)
    path = trainer.save(tmp_path / "no_describe.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("contact_net_describe")
    payload.pop("air_net_describe")
    torch.save(payload, path)
    same = _trainer(tmp_path / "same", time_reference_s=1.0)
    with pytest.raises(TrainingError, match="model metadata"):
        same.load(path)


def test_resume_rejects_a_different_contact_data_weight(tmp_path):
    trainer = _trainer(tmp_path, contact_data_weight=3.0)
    path = trainer.save(tmp_path / "weight3.pt")
    same = _trainer(tmp_path / "same", contact_data_weight=3.0)
    assert same.load(path)["loop"] == 0
    different = _trainer(tmp_path / "different", contact_data_weight=1.0)
    with pytest.raises(TrainingError, match="contact_data_weight"):
        different.load(path)


def test_resume_treats_a_legacy_checkpoint_weight_as_one(tmp_path):
    trainer = _trainer(tmp_path, contact_data_weight=1.0)
    path = trainer.save(tmp_path / "legacy_weight.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"].pop("contact_data_weight")
    torch.save(payload, path)
    plain = _trainer(tmp_path / "plain", contact_data_weight=1.0)
    assert plain.load(path)["loop"] == 0
    weighted = _trainer(tmp_path / "weighted", contact_data_weight=3.0)
    with pytest.raises(TrainingError, match="contact_data_weight"):
        weighted.load(path)


def test_pde_parts_include_normalised_components(tmp_path):
    trainer = _trainer(tmp_path)
    record = trainer.step()
    parts = record["air_parts"]
    for key in ("pde_mass_normalised", "pde_momentum_normalised",
                "pde_energy_normalised"):
        assert key in parts
        assert torch.isfinite(torch.tensor(parts[key]))


def test_logging_does_not_consume_sampling_randomness(tmp_path):
    # Seed before each network is constructed, so the two trainers start from
    # identical initial weights.  Seeding only inside the trainer is too late
    # for the parameter initialisation.
    seed_everything(11)
    logged = _trainer(tmp_path / "logged", log_every=1)
    seed_everything(11)
    quiet = _trainer(tmp_path / "quiet", log_every=0)
    first = logged.train(loops=2)["history"]["loops"]
    second = quiet.train(loops=2)["history"]["loops"]
    for left, right in zip(first, second):
        assert left["air_loss"] == pytest.approx(right["air_loss"])
        assert left["contact_loss"] == pytest.approx(right["contact_loss"])
        assert left["counts"] == right["counts"]
