"""Optional AirNet R0 input gate: none, similarity_time and compatibility."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from spherical_blast_pinn.contact_air.losses import (
    air_ic_loss,
    finite_mach_rh_state,
    interface_loss,
    right_loss,
)
from spherical_blast_pinn.contact_air.models import (
    AirNet,
    ContactNet,
    ModelError,
    TimeInput,
    air_fields_with_derivatives,
    scales_from_precursor,
)
from spherical_blast_pinn.contact_air.testing import (
    default_precursor,
    synthetic_case_for,
)
from spherical_blast_pinn.contact_air.trainer import (
    AlternatingTrainer,
    TrainConfig,
    TrainingError,
    describe_matches,
)


def _scales():
    return scales_from_precursor(
        default_precursor(0.05).as_dict(),
        like=torch.zeros(1, 1, dtype=torch.float64),
    )


def _field_loss(fields):
    return fields.density.sum() + fields.velocity.sum() + fields.pressure.sum()


def _manual_none(net, zeta, t, r0, scales, time_input):
    hidden = net.trunk(torch.cat(
        [zeta, net.scaled_time(t, time_input), r0], dim=-1
    ))
    return net._heads(hidden, scales)


def test_none_mode_matches_historical_output_and_parameter_gradient():
    torch.manual_seed(31)
    net = AirNet(
        width=8, depth=2, time_input_mode="similarity",
        similarity_input_divisor=64.0, r0_input_gate="none",
    )
    scales = _scales()
    ti = TimeInput(origin=1.0e-4, scale=2.0e-5)
    zeta = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    t = torch.full_like(zeta, ti.origin)
    r0 = torch.full_like(zeta, 0.5)
    fields = net(zeta, t, r0, scales, time_input=ti)
    manual = _manual_none(net, zeta, t, r0, scales, ti)
    for a, b in (
        (fields.density, manual.density),
        (fields.velocity, manual.velocity),
        (fields.pressure, manual.pressure),
    ):
        assert torch.equal(a, b)
    params = list(net.parameters())
    g_net = torch.autograd.grad(
        _field_loss(fields), params, retain_graph=True, allow_unused=True
    )
    g_manual = torch.autograd.grad(
        _field_loss(manual), params, allow_unused=True
    )
    for a, b in zip(g_net, g_manual):
        if a is None or b is None:
            assert a is b
        else:
            assert torch.equal(a, b)
    assert net.describe()["r0_input_gate"] == "none"


def test_gate_is_zero_at_t0_and_soft_ic_remains_nonzero_and_trainable():
    torch.manual_seed(37)
    net = AirNet(
        width=8, depth=2, time_input_mode="similarity",
        similarity_input_divisor=64.0, r0_input_gate="similarity_time",
    )
    scales = _scales()
    ti = TimeInput(origin=2.0e-4, scale=5.0e-5)
    zeta = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float64)
    t = torch.full_like(zeta, ti.origin)
    first = net(zeta, t, torch.full_like(zeta, 0.1), scales, time_input=ti)
    second = net(zeta, t, torch.full_like(zeta, 1.0), scales, time_input=ti)
    for a, b in (
        (first.density, second.density),
        (first.velocity, second.velocity),
        (first.pressure, second.pressure),
    ):
        assert torch.equal(a, b)
    batch = SimpleNamespace(
        zeta_ic=zeta, t_ic=torch.full_like(zeta, ti.origin)
    )
    term = air_ic_loss(net, batch, SimpleNamespace(
        t0=ti.origin, charge_radius_m=0.5,
        rho_scale=float(scales.rho), u_scale=float(scales.u),
        p_scale=float(scales.p), time_input=ti,
    ), scales)
    assert float(term.total.detach()) > 0.0
    grads = torch.autograd.grad(
        term.total, list(net.parameters()), allow_unused=True
    )
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)
    assert net.describe()["r0_input_gate"] == "similarity_time"


def test_gate_physical_time_derivative_matches_finite_difference_at_t0_and_early():
    torch.manual_seed(41)
    net = AirNet(
        width=8, depth=2, time_input_mode="similarity",
        similarity_input_divisor=64.0, r0_input_gate="similarity_time",
    )
    scales = _scales()
    ti = TimeInput(origin=1.0e-4, scale=2.0e-5)
    zeta = torch.tensor([[0.4]], dtype=torch.float64)
    r0 = torch.tensor([[0.5]], dtype=torch.float64)
    for theta in (0.0, 0.25, 1.0):
        t_value = ti.origin + theta * ti.scale
        t = torch.tensor([[t_value]], dtype=torch.float64, requires_grad=True)
        fields, derivatives = air_fields_with_derivatives(
            net, zeta, t, r0, scales, create_graph=False, time_input=ti,
        )
        h = 1.0e-8 * ti.scale
        with torch.no_grad():
            plus = net(
                zeta, torch.tensor([[t_value + h]], dtype=torch.float64),
                r0, scales, time_input=ti,
            )
            if theta == 0.0:
                minus = net(zeta, t, r0, scales, time_input=ti)
                denominator = h
            else:
                minus = net(
                    zeta, torch.tensor([[t_value - h]], dtype=torch.float64),
                    r0, scales, time_input=ti,
                )
                denominator = 2.0 * h
        for name, auto, plus_v, minus_v in (
            ("rho", derivatives["rho_t"], plus.density, minus.density),
            ("u", derivatives["u_t"], plus.velocity, minus.velocity),
            ("p", derivatives["p_t"], plus.pressure, minus.pressure),
        ):
            finite_difference = (plus_v - minus_v) / denominator
            assert torch.allclose(
                auto, finite_difference, rtol=5.0e-5, atol=1.0e-6
            ), name


def test_ordinary_and_functional_gate_paths_agree_and_functional_is_frozen():
    torch.manual_seed(43)
    net = AirNet(
        width=8, depth=2, time_input_mode="similarity",
        similarity_input_divisor=64.0, r0_input_gate="similarity_time",
    )
    scales = _scales()
    ti = TimeInput(origin=1.0e-4, scale=2.0e-5)
    zeta = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
    t = torch.full_like(zeta, ti.origin + 0.25 * ti.scale)
    r0 = torch.full_like(zeta, 0.5)
    fields, derivatives = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False, time_input=ti,
    )
    func_fields, func_derivatives = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False, functional=True,
        time_input=ti,
    )
    for a, b in (
        (fields.density, func_fields.density),
        (fields.velocity, func_fields.velocity),
        (fields.pressure, func_fields.pressure),
    ):
        assert torch.equal(a, b)
    for name in ("rho", "u", "p"):
        assert torch.equal(derivatives[f"{name}_t"], func_derivatives[f"{name}_t"])
        assert torch.equal(derivatives[f"{name}_z"], func_derivatives[f"{name}_z"])
    grads = torch.autograd.grad(
        _field_loss(func_fields), list(net.parameters()), allow_unused=True
    )
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in grads)


def test_gate_checkpoint_metadata_is_rejected_by_none_and_old_none_is_accepted():
    none_net = AirNet(width=8, depth=1, time_input_mode="similarity")
    gate_net = AirNet(
        width=8, depth=1, time_input_mode="similarity",
        r0_input_gate="similarity_time",
    )
    assert describe_matches(none_net.describe(), none_net.describe())
    assert describe_matches(gate_net.describe(), gate_net.describe())
    assert not describe_matches(gate_net.describe(), none_net.describe())
    old = dict(none_net.describe())
    old.pop("r0_input_gate")
    old.pop("r0_gate_rule")
    assert describe_matches(old, none_net.describe())


def test_gate_requires_similarity_mode():
    with pytest.raises(ModelError, match="requires time_input='similarity'"):
        AirNet(time_input_mode="legacy", r0_input_gate="similarity_time")
    with pytest.raises(ModelError, match="R0 input gate must be one of"):
        AirNet(r0_input_gate="other")


def test_right_loss_uses_the_gated_airnet_path():
    torch.manual_seed(47)
    case = synthetic_case_for(
        0.05, case_id="r050", n_frames=11, n_radial=800
    )
    net = AirNet(
        width=8, depth=2, time_input_mode="similarity",
        similarity_input_divisor=64.0, r0_input_gate="similarity_time",
    )
    scales = _scales()
    span = float(case.t_end) - float(case.t0)
    t_right = torch.tensor(
        [float(case.t0) + 0.25 * span, float(case.t0) + 0.50 * span],
        dtype=torch.float64,
    )
    batch = SimpleNamespace(
        t_right=t_right, t_ic=torch.zeros(0, dtype=torch.float64)
    )
    term = right_loss(net=net, batch=batch, case=case, scales=scales)
    _, shock_speed = case.shock_tensors(t_right.reshape(-1))
    state = finite_mach_rh_state(
        shock_speed.reshape(-1, 1),
        gamma=case.gamma_air,
        ambient_density=case.ambient_density,
        ambient_pressure=case.ambient_pressure,
    )
    zeta = torch.ones_like(t_right).reshape(-1, 1)
    r0 = torch.full_like(zeta, float(case.charge_radius_m))
    fields = net(
        zeta, t_right.reshape(-1, 1), r0, scales,
        time_input=case.time_input,
    )
    manual = (
        ((fields.density - state["density"]) / float(case.rho_scale)).pow(2).mean()
        + ((fields.velocity - state["velocity"]) / float(case.u_scale)).pow(2).mean()
        + ((fields.pressure - state["pressure"]) / float(case.p_scale)).pow(2).mean()
    )
    assert torch.allclose(term.total, manual, rtol=1.0e-12, atol=1.0e-15)


def _small_trainer(tmp_path, *, gate=False):
    case = synthetic_case_for(
        0.05, case_id="r050", n_frames=11, n_radial=800
    )
    cfg = TrainConfig(
        n_pde=8, n_ic=4, n_right=4, n_data=8, n_contact=6,
        loops=0, learning_rate=1.0e-3, checkpoint_every=0,
        validate_every=0, seed=51, validation_holdout_fraction=0.0,
    )
    return AlternatingTrainer(
        cases=[case],
        contact_net=ContactNet(
            width=8, depth=1, time_input_mode="similarity",
        ),
        air_net=AirNet(
            width=8, depth=1, time_input_mode="similarity",
            r0_input_gate="similarity_time" if gate else "none",
        ),
        config=cfg, output_dir=tmp_path,
    )


def test_trainer_load_rejects_gate_checkpoint_in_none_mode(tmp_path):
    gate_trainer = _small_trainer(tmp_path / "gate", gate=True)
    checkpoint = gate_trainer.save(tmp_path / "gate.pt")
    none_trainer = _small_trainer(tmp_path / "none", gate=False)
    with pytest.raises(TrainingError, match="model metadata"):
        none_trainer.load(checkpoint)
    # The same checkpoint is accepted by the matching gate model.
    gate_again = _small_trainer(tmp_path / "gate_again", gate=True)
    gate_again.load(checkpoint)
    assert gate_again.air_net.r0_input_gate == "similarity_time"


def test_contact_stage_interface_uses_the_gated_functional_path(tmp_path):
    trainer = _small_trainer(tmp_path, gate=True)
    case, batch = trainer._batches()[0]
    terms = trainer.contact_stage_terms(case, batch)
    closure = trainer._closures[case.case_id]
    scales = trainer._scales[case.case_id]
    r0, rs, ds = trainer._contact_inputs(case, batch.t_pde)
    c_fields = trainer.contact_net(
        batch.t_pde.reshape(-1, 1), r0, rs, ds, closure,
        time_input=case.time_input, create_graph=True,
    )
    air0 = trainer.air_net.forward_functional(
        torch.zeros_like(batch.t_pde).reshape(-1, 1),
        batch.t_pde.reshape(-1, 1), r0.reshape(-1, 1), scales,
        detach_parameters=True, time_input=case.time_input,
    )
    manual = interface_loss(
        air_at_contact=air0, contact=c_fields, case=case
    )
    assert torch.allclose(
        terms["interface"].total, manual.total, rtol=1.0e-12, atol=1.0e-15
    )


def test_checkpoint_fields_have_distinct_path_and_checks():
    from scripts.check_r0_gate_design import _checkpoint_fields

    fields = _checkpoint_fields("outputs/example.pt", {"gate": "none"})
    assert set(fields) == {"checkpoint_path", "checkpoint_checks"}
    assert isinstance(fields["checkpoint_path"], str)
    assert isinstance(fields["checkpoint_checks"], dict)
    with pytest.raises(TypeError):
        _checkpoint_fields("outputs/example.pt", "not-a-dict")
