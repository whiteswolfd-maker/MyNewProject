"""Mapped residual must equal the physical-coordinate Euler residual."""
from __future__ import annotations

import torch

from spherical_blast_pinn.contact_air.mapping import (
    build_geometry,
    physical_residuals,
    assemble_residuals,
)
from spherical_blast_pinn.contact_air.models import AirFields

DTYPE = torch.float64


def test_mapped_and_physical_residuals_agree_for_a_smooth_nonuniform_boundary():
    zeta = torch.tensor([[0.1], [0.4], [0.8]], dtype=DTYPE)
    t = torch.tensor([[1.0e-5], [2.0e-5], [3.0e-5]], dtype=DTYPE)
    # Non-uniform R_b and R_c: both the value and the grid speed vary.
    rc = 0.10 + 1000.0 * (t - 1.0e-5)
    rc_dot = torch.full_like(rc, 1000.0)
    rb = 0.20 + 1500.0 * (t - 1.0e-5) + 2.0e7 * (t - 1.0e-5) ** 2
    rb_dot = 1500.0 + 4.0e7 * (t - 1.0e-5)
    geometry = build_geometry(rc, rc_dot, rb, rb_dot, zeta)
    r = geometry.radius

    rho = 1.2 + 0.3 * r + 0.2 * t
    u = 1000.0 + 2.0 * r + 3.0 * t
    p = 1.0e5 + 10.0 * r + 20.0 * t
    fields = AirFields(density=rho, velocity=u, pressure=p,
                       specific_energy=p / (0.4 * rho))

    # Analytic physical derivatives at fixed (r,t).
    rho_t = torch.full_like(rho, 0.2)
    rho_r = torch.full_like(rho, 0.3)
    u_t = torch.full_like(u, 3.0)
    u_r = torch.full_like(u, 2.0)
    p_t = torch.full_like(p, 20.0)
    p_r = torch.full_like(p, 10.0)
    L = geometry.length
    vg = geometry.grid_speed
    derivatives = {
        "rho_t": rho_t + rho_r * vg,
        "rho_z": rho_r * L,
        "u_t": u_t + u_r * vg,
        "u_z": u_r * L,
        "p_t": p_t + p_r * vg,
        "p_z": p_r * L,
    }
    gamma = torch.full_like(rho, 1.4)
    mapped = assemble_residuals(fields, derivatives, geometry, gamma=gamma)
    physical = physical_residuals(
        rho, u, p, rho_t, rho_r, u_t, u_r, p_t, p_r, r,
        gamma=gamma,
    )
    for name in ("mass", "momentum", "energy"):
        actual = getattr(mapped, name)
        expected = getattr(physical, name)
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-9), (
            name, float((actual - expected).abs().max())
        )


def test_contact_speed_uses_the_full_product_rule_with_moving_rb():
    from spherical_blast_pinn.contact_air.models import ContactNet, closure_from_precursor
    from spherical_blast_pinn.contact_air.testing import default_precursor

    torch.manual_seed(3)
    net = ContactNet(width=8, depth=1)
    precursor = default_precursor(0.05)
    closure = closure_from_precursor(precursor.as_dict(), like=torch.zeros(1, 1, dtype=DTYPE))
    t = torch.tensor([[2.0e-5]], dtype=DTYPE)

    def rb_of(value):
        return 0.20 + 1500.0 * (value - 1.0e-5) + 2.0e7 * (value - 1.0e-5) ** 2

    def rb_dot_of(value):
        return 1500.0 + 4.0e7 * (value - 1.0e-5)

    r0 = torch.full_like(t, 0.05)
    with torch.enable_grad():
        fields = net(t, r0, rb_of(t), rb_dot_of(t), closure, create_graph=False)
    h = 1.0e-10
    with torch.enable_grad():
        plus = net(t + h, r0, rb_of(t + h), rb_dot_of(t + h), closure, create_graph=False)
        minus = net(t - h, r0, rb_of(t - h), rb_dot_of(t - h), closure, create_graph=False)
    finite_difference = (plus.radius - minus.radius) / (2.0 * h)
    assert torch.allclose(fields.speed, finite_difference, rtol=2.0e-5, atol=1.0e-3), (
        float(fields.speed), float(finite_difference)
    )


def test_air_time_derivatives_are_taken_at_fixed_zeta_and_r0():
    from spherical_blast_pinn.contact_air.models import (
        AirNet,
        air_fields_with_derivatives,
        scales_from_precursor,
    )
    from spherical_blast_pinn.contact_air.testing import default_precursor

    torch.manual_seed(5)
    net = AirNet(width=8, depth=2)
    scales = scales_from_precursor(default_precursor(0.05).as_dict(),
                                   like=torch.zeros(1, 1, dtype=DTYPE))
    zeta = torch.tensor([[0.2], [0.7]], dtype=DTYPE)
    t = torch.tensor([[2.0e-5], [3.0e-5]], dtype=DTYPE)
    r0 = torch.full_like(t, 0.05)
    fields, derivatives = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False
    )
    h = 1.0e-7
    plus = net(zeta, t + h, r0, scales)
    minus = net(zeta, t - h, r0, scales)
    attributes = {"rho": "density", "u": "velocity", "p": "pressure"}
    for name, plus_value, minus_value in (
        ("rho", plus.density, minus.density),
        ("u", plus.velocity, minus.velocity),
        ("p", plus.pressure, minus.pressure),
    ):
        finite = (plus_value - minus_value) / (2.0 * h)
        assert torch.allclose(derivatives[f"{name}_t"], finite, rtol=1.0e-5, atol=1.0e-6), name
        # Spatial derivative at fixed t and R0, checked independently.
        plus_z = getattr(net(zeta + h, t, r0, scales), attributes[name])
        minus_z = getattr(net(zeta - h, t, r0, scales), attributes[name])
        assert torch.allclose(derivatives[f"{name}_z"], (plus_z - minus_z) / (2.0 * h),
                              rtol=1.0e-5, atol=1.0e-6), name


def test_geometry_rejects_non_positive_layer():
    zeta = torch.tensor([[0.5]], dtype=DTYPE)
    rc = torch.tensor([[0.2]], dtype=DTYPE)
    rb = torch.tensor([[0.1]], dtype=DTYPE)
    try:
        build_geometry(rc, torch.zeros_like(rc), rb, torch.zeros_like(rb), zeta)
    except Exception as exc:
        assert "L = R_b - R_c" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-positive layer was accepted")
