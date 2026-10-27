"""New startup -> full physics batch -> total loss -> one backward pass.

This is an assembly check, not training: no optimizer, no parameter update, no
checkpoint, and ``training_qualified`` stays false.
"""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from spherical_blast_pinn.cli import check_startup_chain as chain_cli
from spherical_blast_pinn.cli.check_startup_chain import (
    _failure_reasons,
    _grad_summary,
)
from spherical_blast_pinn.evaluation.checks import build_check_set, run_checks
from spherical_blast_pinn.losses.objective import contact_pressure_scale, total_loss
from spherical_blast_pinn.models.air_net import AirNet
from spherical_blast_pinn.models.coupled_model import CoupledModel
from spherical_blast_pinn.models.front_net import FrontNet
from spherical_blast_pinn.models.product_net import ProductNet
from spherical_blast_pinn.physics.similitude import PlanarStartupScales
from spherical_blast_pinn.physics.two_medium_initial import (
    build_two_medium_initial_field,
)
from spherical_blast_pinn.physics.two_medium_startup_handoff import (
    load_planar_startup_handoff,
    write_planar_startup_handoff,
)
from spherical_blast_pinn.physics.two_medium_startup_planar import (
    PlanarStartupConfig,
    build_planar_startup_field,
)
from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig

R0 = 0.05
TAU_END = 1.0e-6
TIME_SCALE = 2.0


@pytest.fixture(scope="module")
def loaded_field(tmp_path_factory):
    initial = build_two_medium_initial_field(charge_radius=R0, outer_radius=2.0 * R0)
    field = build_planar_startup_field(
        initial,
        config=PlanarStartupConfig(
            delta=1.0 / 200.0,
            inner_cells=16,
            fan_cells=24,
            plateau_cells=3,
            air_cells=6,
            ambient_cells=10,
            dense_samples=1024,
        ),
    )
    directory = tmp_path_factory.mktemp("handoff")
    json_path, _ = write_planar_startup_handoff(field, directory / "handoff")
    return load_planar_startup_handoff(json_path), json_path


def _model(scales, field) -> CoupledModel:
    return CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(
            scales=scales,
            width=8,
            depth=1,
            initial_contact_hat=float(field.contact_radius / field.charge_radius),
            initial_gap_hat=float(
                (field.shock_radius - field.contact_radius) / field.charge_radius
            ),
            seed=2,
        ),
        scales=scales,
    )


def test_full_batch_total_loss_and_one_backward(loaded_field):
    torch.set_default_dtype(torch.float64)
    field, _ = loaded_field
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    sampler = RadiusSampler(
        field=field,
        scales=scales,
        tau_end=TAU_END,
        config=SamplingConfig(
            n_pde_product=64, n_pde_air=64, n_ic_product=24, n_ic_air=24,
            n_contact=16, n_shock=16,
        ),
        seed=4,
    )
    batch = sampler.batch()

    # s = 0 is the field's own epsilon, and the tau range is the declared one.
    assert float(scales.physical_time_from_network(0.0)) == field.epsilon
    assert batch.pinn_time_max == TAU_END
    assert batch.detonation_velocity is None
    assert batch.s_data_product is None and batch.s_data_air is None
    assert float(batch.s_product.max()) <= float(scales.network_time(TAU_END)) + 1.0e-14
    for values in (batch.s_product, batch.s_air, batch.s_contact, batch.s_shock):
        assert values.numel() > 0
    assert batch.xi_ic_product.numel() > 0 and batch.eta_ic_air.numel() > 0

    model = _model(scales, field)
    pressure = contact_pressure_scale(field)
    assert pressure == field.solution.p_star
    loss = total_loss([batch], model, contact_pressure=pressure)

    assert torch.isfinite(loss.total)
    for name, value in loss.group_tensors.items():
        assert bool(torch.isfinite(value).all()), name
    # No data supervision: those two groups are exactly absent, not zero-filled
    # observations. The physics groups must actually have run.
    assert float(loss.group_tensors["data"].detach()) == 0.0
    assert float(loss.group_tensors["geometry"].detach()) == 0.0
    for name in ("pde", "initial", "contact", "shock", "admissibility"):
        assert float(loss.group_tensors[name].detach()) > 0.0, name
    assert loss.monitors[f"{batch.case_id}.contact_n_points"] > 0
    assert loss.monitors[f"{batch.case_id}.shock_n_points"] > 0

    loss.total.backward()
    for name, parameters in model.parameter_groups().items():
        grads = [p.grad for p in parameters if p.grad is not None]
        assert grads, f"{name} received no gradient at all"
        assert all(bool(torch.isfinite(g).all()) for g in grads), name
        # Not every parameter must be non-zero, but the group as a whole must
        # carry information; a whole group of exact zeros would mean the loss
        # never touched that network.
        assert any(float(g.abs().max()) > 0.0 for g in grads), name

    assert field.training_qualified is False
    assert field.reference_field is False


def test_the_chain_entry_point_reports_finite_losses_and_gradients(
    loaded_field, tmp_path
):
    _, handoff_json = loaded_field
    output = tmp_path / "chain.json"
    code = chain_cli.main(
        [
            "--handoff-json", str(handoff_json),
            "--tau-end", str(TAU_END),
            "--n-pde", "16",
            "--n-ic", "8",
            "--n-interface", "6",
            "--width", "8",
            "--depth", "1",
            "--seed", "5",
            "--output-json", str(output),
        ]
    )
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["time_convention"] == "planar_startup_epsilon"
    assert report["s0_physical_time_s"] == report["field_epsilon_s"]
    assert report["batch_tau_max_s"] == TAU_END
    assert report["detonation_velocity"] is None
    assert report["point_counts"]["pde_product"] == 16
    assert report["point_counts"]["contact"] == 6
    assert report["loss"]["all_finite"] is True
    assert math.isfinite(report["loss"]["total"])
    for name, entry in report["gradients"].items():
        assert entry["n_with_gradient"] == entry["n_parameters"], name
        assert entry["all_finite"] is True, name
        assert entry["any_nonzero"] is True, name
    assert report["training_qualified"] is False
    assert report["ok"] is True
    assert report["failures"] == []


def _good_gradient_report() -> dict:
    return {
        "loss": {"all_finite": True},
        "gradients": {
            name: {
                "n_parameters": 2,
                "n_with_gradient": 2,
                "all_finite": True,
                "any_nonzero": True,
            }
            for name in ("product", "air", "front")
        },
    }


def test_the_success_gate_rejects_missing_non_finite_and_zero_gradients():
    assert _failure_reasons(_good_gradient_report()) == []

    missing = _good_gradient_report()
    del missing["gradients"]["front"]
    assert any("front: no gradient record" in reason for reason in _failure_reasons(missing))

    absent = _good_gradient_report()
    absent["gradients"]["air"]["n_with_gradient"] = 0
    absent["gradients"]["air"]["any_nonzero"] = False
    assert any(
        "air: no parameter received a gradient" in reason
        for reason in _failure_reasons(absent)
    )

    non_finite = _good_gradient_report()
    non_finite["gradients"]["product"]["all_finite"] = False
    assert any(
        "product: a produced gradient is not finite" in reason
        for reason in _failure_reasons(non_finite)
    )

    zero_group = _good_gradient_report()
    zero_group["gradients"]["front"]["any_nonzero"] = False
    assert any(
        "front: every produced gradient is exactly zero" in reason
        for reason in _failure_reasons(zero_group)
    )

    bad_loss = _good_gradient_report()
    bad_loss["loss"]["all_finite"] = False
    assert any("loss:" in reason for reason in _failure_reasons(bad_loss))


def test_a_really_zeroed_group_is_detected_after_a_real_backward(loaded_field):
    torch.set_default_dtype(torch.float64)
    field, _ = loaded_field
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    sampler = RadiusSampler(
        field=field,
        scales=scales,
        tau_end=TAU_END,
        config=SamplingConfig(
            n_pde_product=8, n_pde_air=8, n_ic_product=4, n_ic_air=4,
            n_contact=4, n_shock=4,
        ),
        seed=6,
    )
    model = _model(scales, field)
    loss = total_loss(
        [sampler.batch()], model, contact_pressure=contact_pressure_scale(field)
    )
    loss.total.backward()
    assert _failure_reasons(
        {"loss": {"all_finite": True}, "gradients": _grad_summary(model)}
    ) == []

    for parameter in model.air_net.parameters():
        parameter.grad.zero_()
    reasons = _failure_reasons(
        {"loss": {"all_finite": True}, "gradients": _grad_summary(model)}
    )
    assert any("air: every produced gradient is exactly zero" in reason for reason in reasons)


def test_the_check_set_uses_the_same_startup_field_and_pressure(loaded_field):
    field, _ = loaded_field
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    check_set = build_check_set(
        field=field,
        scales=scales,
        tau_end=TAU_END,
        config=SamplingConfig(
            n_pde_product=8, n_pde_air=8, n_ic_product=4, n_ic_air=4,
            n_contact=4, n_shock=4,
        ),
        seed=9,
    )
    assert check_set.contact_pressure == field.solution.p_star
    assert len(check_set.batches) == 1
    batch = check_set.batches[0]
    assert batch.detonation_velocity is None
    assert batch.s_data_product is None
    windowed = check_set.windowed(0.5)
    assert windowed.contact_pressure == check_set.contact_pressure
    assert float(windowed.batches[0].s_contact.max()) <= float(
        batch.scales.network_time(0.5 * float(batch.pinn_time_max))
    ) + 1.0e-15
    model = _model(scales, field)
    report = run_checks(model, check_set, steps=0)
    assert math.isfinite(report.metrics["physics_violation_score"])
    assert report.admissible is True


def test_the_entry_point_returns_nonzero_and_writes_the_failed_groups(
    loaded_field, tmp_path, monkeypatch
):
    _, handoff_json = loaded_field
    zeroed = {
        name: {
            "n_parameters": 2,
            "n_with_gradient": 2,
            "all_finite": True,
            "any_nonzero": False,
        }
        for name in ("product", "air", "front")
    }
    monkeypatch.setattr(chain_cli, "_grad_summary", lambda model: zeroed)
    output = tmp_path / "failed.json"
    code = chain_cli.main(
        [
            "--handoff-json", str(handoff_json),
            "--tau-end", str(TAU_END),
            "--n-pde", "8",
            "--n-ic", "4",
            "--n-interface", "2",
            "--width", "8",
            "--depth", "1",
            "--output-json", str(output),
        ]
    )
    assert code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["loss"]["all_finite"] is True
    assert len(report["failures"]) == 3
    for name in ("product", "air", "front"):
        assert any(
            reason.startswith(f"{name}: every produced gradient is exactly zero")
            for reason in report["failures"]
        )
