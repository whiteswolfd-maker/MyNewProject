"""The MLP time input may be scaled, but physics stays in seconds."""
from __future__ import annotations

import torch
import pytest

from spherical_blast_pinn.contact_air.models import (
    AirNet,
    ContactNet,
    air_fields_with_derivatives,
    closure_from_precursor,
    scales_from_precursor,
)
from spherical_blast_pinn.contact_air.testing import default_precursor


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
