"""The MLP time input may be scaled, but physics stays in seconds."""
from __future__ import annotations

import torch
import pytest

from spherical_blast_pinn.contact_air.models import (
    AirNet,
    ContactNet,
    ModelError,
    TimeInput,
    air_fields_with_derivatives,
    closure_from_precursor,
    scales_from_precursor,
)
from spherical_blast_pinn.contact_air.testing import (
    default_precursor,
    synthetic_case_for,
)


def _closure():
    return closure_from_precursor(
        default_precursor(0.05).as_dict(),
        like=torch.zeros(1, 1, dtype=torch.float64),
    )


def test_contact_speed_stays_metres_per_second_with_scaled_time_input():
    torch.manual_seed(3)
    net = ContactNet(width=8, depth=1, time_reference_s=0.010)
    closure = _closure()

    shock_speed = 1500.0

    def evaluate(value):
        t = torch.tensor([[value]], dtype=torch.float64)
        rs = torch.full_like(t, 1.2 + shock_speed * (value - 3.0e-3))
        return net(
            t, torch.full_like(t, 0.05), rs,
            torch.full_like(t, shock_speed), closure, create_graph=False,
        )

    base = evaluate(3.0e-3)
    h = 1.0e-10
    finite_difference = (evaluate(3.0e-3 + h).radius - evaluate(3.0e-3 - h).radius) / (2.0 * h)
    assert base.speed.item() == pytest.approx(finite_difference.item(), rel=1.0e-6)


def test_time_reference_changes_the_network_input_but_not_the_shape():
    torch.manual_seed(5)
    physical = ContactNet(width=8, depth=1, time_reference_s=1.0)
    scaled = ContactNet(width=8, depth=1, time_reference_s=0.010)
    scaled.load_state_dict(physical.state_dict())
    closure = _closure()
    t = torch.tensor([[5.0e-3]], dtype=torch.float64)
    r0 = torch.full_like(t, 0.05)
    rs = torch.full_like(t, 12.0)
    ds = torch.full_like(t, 1500.0)
    assert physical.describe()["time_reference_s"] == pytest.approx(1.0)
    assert scaled.describe()["time_reference_s"] == pytest.approx(0.010)
    # Same weights, different tau -> different raw ratio.
    assert not torch.allclose(
        physical(t, r0, rs, ds, closure, create_graph=False).raw,
        scaled(t, r0, rs, ds, closure, create_graph=False).raw,
    )


def test_air_time_derivative_stays_per_second_with_scaled_time_input():
    torch.manual_seed(7)
    net = AirNet(width=8, depth=2, time_reference_s=0.010)
    scales = scales_from_precursor(
        default_precursor(0.05).as_dict(), like=torch.zeros(1, 1, dtype=torch.float64)
    )
    zeta = torch.tensor([[0.4]], dtype=torch.float64)
    r0 = torch.tensor([[0.05]], dtype=torch.float64)
    t = torch.tensor([[3.0e-3]], dtype=torch.float64)

    fields, derivatives = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False,
    )
    h = 1.0e-10
    plus = net(zeta, t + h, r0, scales).pressure
    minus = net(zeta, t - h, r0, scales).pressure
    finite_difference = (plus - minus) / (2.0 * h)
    assert derivatives["p_t"].item() == pytest.approx(
        finite_difference.item(), rel=1.0e-6
    )
    # The functional path used by the contact stage must scale time the same way.
    functional_fields, functional_derivatives = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False, functional=True,
    )
    assert torch.allclose(fields.pressure, functional_fields.pressure)
    assert torch.allclose(derivatives["p_t"], functional_derivatives["p_t"])


def test_case_and_prediction_objects_expose_the_same_time_input():
    case = synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=800)
    prediction = case.prediction_view()
    assert case.time_input.origin == pytest.approx(case.t0)
    assert case.time_input.scale == pytest.approx(case.time_scale)
    assert prediction.time_input.origin == pytest.approx(case.t0)
    assert prediction.time_input.scale == pytest.approx(case.time_scale)


def test_similarity_input_is_common_for_equal_theta_and_zero_at_t0():
    torch.manual_seed(11)
    net = ContactNet(
        width=8, depth=1, time_input_mode="similarity",
        similarity_input_divisor=64.0,
    )
    theta = torch.tensor([[7.5]], dtype=torch.float64)
    first = TimeInput(origin=1.0e-5, scale=2.0e-5)
    second = TimeInput(origin=2.0e-4, scale=4.0e-4)
    t_first = theta * first.scale + first.origin
    t_second = theta * second.scale + second.origin
    assert torch.allclose(
        net.scaled_time(t_first, first), net.scaled_time(t_second, second)
    )
    assert net.scaled_time(
        torch.tensor([[first.origin]], dtype=torch.float64), first
    ).item() == pytest.approx(0.0)
    # The conversion is linear in physical time; theta is not clamped.
    assert net.scaled_time(
        torch.tensor([[first.origin + 640.0 * first.scale]], dtype=torch.float64),
        first,
    ).item() == pytest.approx(10.0)


def test_similarity_scaling_requires_case_time_and_validates_inputs():
    net = AirNet(width=8, depth=1, time_input_mode="similarity")
    t = torch.tensor([[1.0e-3]], dtype=torch.float64)
    with pytest.raises(ModelError, match="requires TimeInput"):
        net.scaled_time(t)
    contact = ContactNet(width=8, depth=1, time_input_mode="similarity")
    with pytest.raises(ModelError, match="requires TimeInput"):
        contact.scaled_time(t)
    with pytest.raises(ModelError, match="origin must be finite"):
        TimeInput(float("nan"), 1.0e-5)
    with pytest.raises(ModelError, match="scale must be finite and positive"):
        TimeInput(0.0, 0.0)
    with pytest.raises(ModelError, match="divisor must be finite and positive"):
        AirNet(width=8, depth=1, time_input_mode="similarity",
               similarity_input_divisor=0.0)
    with pytest.raises(ModelError, match="time input mode"):
        AirNet(width=8, depth=1, time_input_mode="other")


def test_similarity_time_derivative_is_physical_seconds():
    torch.manual_seed(13)
    net = ContactNet(
        width=8, depth=1, time_input_mode="similarity",
        similarity_input_divisor=64.0,
    )
    time_input = TimeInput(origin=2.0e-3, scale=5.0e-4)
    t = torch.tensor([[2.0e-3], [2.5e-3]], dtype=torch.float64, requires_grad=True)
    scaled = net.scaled_time(t, time_input)
    grad = torch.autograd.grad(scaled.sum(), t)[0]
    expected = 1.0 / (64.0 * time_input.scale)
    assert torch.allclose(grad, torch.full_like(grad, expected))


def test_similarity_contact_speed_stays_physical_seconds():
    torch.manual_seed(17)
    net = ContactNet(
        width=8, depth=1, time_input_mode="similarity",
        similarity_input_divisor=64.0,
    )
    closure = _closure()
    time_input = TimeInput(origin=1.0e-3, scale=2.0e-5)
    shock_speed = 1500.0

    def evaluate(value):
        t = torch.tensor([[value]], dtype=torch.float64)
        rs = torch.full_like(t, 1.2 + shock_speed * (value - 3.0e-3))
        return net(
            t, torch.full_like(t, 0.05), rs,
            torch.full_like(t, shock_speed), closure,
            time_input=time_input, create_graph=False,
        )

    base = evaluate(3.0e-3)
    h = 1.0e-10
    finite_difference = (
        evaluate(3.0e-3 + h).radius - evaluate(3.0e-3 - h).radius
    ) / (2.0 * h)
    assert base.speed.item() == pytest.approx(finite_difference.item(), rel=1.0e-6)


def test_similarity_air_derivatives_and_functional_path_are_consistent():
    torch.manual_seed(19)
    net = AirNet(
        width=8, depth=2, time_input_mode="similarity",
        similarity_input_divisor=64.0,
    )
    scales = scales_from_precursor(
        default_precursor(0.05).as_dict(),
        like=torch.zeros(1, 1, dtype=torch.float64),
    )
    zeta = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    r0 = torch.tensor([[0.05]], dtype=torch.float64)
    t = torch.tensor([[3.0e-3]], dtype=torch.float64, requires_grad=True)
    time_input = TimeInput(origin=1.0e-3, scale=2.0e-5)

    fields, derivatives = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False, time_input=time_input,
    )
    h = 1.0e-10
    plus = net(zeta, t + h, r0, scales, time_input=time_input).pressure
    minus = net(zeta, t - h, r0, scales, time_input=time_input).pressure
    finite_difference = (plus - minus) / (2.0 * h)
    assert derivatives["p_t"].item() == pytest.approx(
        finite_difference.item(), rel=1.0e-6
    )
    functional_fields, functional_derivatives = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False, functional=True,
        time_input=time_input,
    )
    assert torch.allclose(fields.pressure, functional_fields.pressure)
    assert torch.allclose(derivatives["p_t"], functional_derivatives["p_t"])
    functional_direct = net.forward_functional(
        zeta, t, r0, scales, detach_parameters=True, time_input=time_input,
    )
    parameter_grads = torch.autograd.grad(
        functional_direct.pressure.sum(), tuple(net.parameters()),
        allow_unused=True, retain_graph=True,
    )
    assert all(g is None or not bool(g.any()) for g in parameter_grads)
    zeta_grad, t_grad = torch.autograd.grad(
        functional_direct.pressure.sum(), (zeta, t), retain_graph=True,
    )
    assert bool(zeta_grad.any()) and bool(t_grad.any())
