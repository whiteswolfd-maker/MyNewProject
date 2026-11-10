"""Shock trajectory, contact interval and initial-geometry checks."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.geometry import (
    ContactIntervalTable,
    GeometryError,
    ShockTrajectory,
    build_initial_geometry,
)


def _shock() -> ShockTrajectory:
    times = np.linspace(1.0e-5, 2.0e-5, 11)
    radii = 0.15 + 1000.0 * (times - 1.0e-5) ** 2
    return ShockTrajectory.from_arrays(case_id="s", times=times, radii=radii)


def test_pchip_position_and_derivative_are_both_available_inside_a_segment():
    shock = _shock()
    value = shock.position(1.5e-5)
    slope = shock.derivative(1.5e-5)
    assert 0.15 < value < 0.16
    assert slope > 0.0


def test_no_extrapolation_and_no_gap_filling():
    shock = _shock()
    with pytest.raises(GeometryError, match="extrapolation"):
        shock.position(1.0e-6)
    with pytest.raises(GeometryError, match="dR_s/dt"):
        shock.derivative(1.0e-6)


def test_an_isolated_point_has_a_position_but_no_slope():
    shock = ShockTrajectory.from_arrays(
        case_id="one", times=[1.0e-5, 2.0e-5], radii=[0.15, 0.16],
        valid=[True, False],
    )
    assert shock.position(1.0e-5) == pytest.approx(0.15)
    with pytest.raises(GeometryError, match="isolated"):
        shock.derivative(1.0e-5)
    with pytest.raises(GeometryError, match="isolated"):
        shock.require_covering(1.0e-5)


def test_a_gap_splits_segments_instead_of_bridging_it():
    times = np.linspace(1.0e-5, 3.0e-5, 21)
    radii = 0.15 + 100.0 * (times - 1.0e-5)
    valid = np.ones(times.size, dtype=bool)
    valid[9:12] = False
    shock = ShockTrajectory.from_arrays(
        case_id="gap", times=times, radii=radii, valid=valid
    )
    assert len(shock.segments) == 2
    assert shock.position(1.5e-5) == pytest.approx(0.15 + 100.0 * 0.5e-5)
    with pytest.raises(GeometryError):
        shock.position(2.1e-5)


def test_torch_pchip_derivative_matches_scipy_and_autograd():
    shock = _shock()
    for value in (1.2e-5, 1.5e-5, 1.9e-5):
        point = torch.tensor([value], dtype=torch.float64)
        radius, speed = shock.position_and_derivative_torch(point)
        assert float(radius) == pytest.approx(float(shock.position(value)), rel=1e-14)
        assert float(speed) == pytest.approx(float(shock.derivative(value)), rel=1e-13)
        leaf = point.clone().requires_grad_(True)
        radius, speed = shock.position_and_derivative_torch(leaf)
        autograd = torch.autograd.grad(radius.sum(), leaf)[0]
        assert float(autograd) == pytest.approx(float(speed), rel=1e-12)


def test_contact_speed_contains_the_shock_speed_term():
    from spherical_blast_pinn.contact_air.models import ContactNet, closure_from_precursor
    from spherical_blast_pinn.contact_air.testing import default_precursor

    torch.manual_seed(3)
    net = ContactNet(width=8, depth=1)
    closure = closure_from_precursor(
        default_precursor(0.05).as_dict(), like=torch.zeros(1, 1, dtype=torch.float64)
    )

    def rs_of(value):
        return 0.20 + 1500.0 * (value - 1.0e-5) + 2.0e7 * (value - 1.0e-5) ** 2

    def ds_of(value):
        return 1500.0 + 4.0e7 * (value - 1.0e-5)

    def evaluate(value):
        t = torch.tensor([[value]], dtype=torch.float64)
        return net(t, torch.full_like(t, 0.05), rs_of(t), ds_of(t), closure,
                   create_graph=False)

    base = evaluate(2.0e-5)
    h = 1.0e-10
    finite_difference = (evaluate(2.0e-5 + h).radius - evaluate(2.0e-5 - h).radius) / (2.0 * h)
    assert torch.allclose(base.speed, finite_difference, rtol=2.0e-5, atol=1.0e-3)


def test_initial_geometry_uses_the_measured_shock_not_rb():
    shock = _shock()
    with pytest.raises(GeometryError, match="R_c"):
        build_initial_geometry(
            t0=1.5e-5, Rc0=0.16, Rs0=0.17, R_sep=0.1,
            u_c_sep=1000.0, D_s_sep=2000.0, shock=shock,
        )
    good = build_initial_geometry(
        t0=1.5e-5, Rc0=0.11, Rs0=0.12, R_sep=0.1,
        u_c_sep=1000.0, D_s_sep=2000.0, shock=shock,
    )
    assert good.Rs_data_at_t0 == pytest.approx(float(shock.position(1.5e-5)))
    assert np.isfinite(good.zeta_s0)


def test_t0_must_be_inside_one_continuous_shock_segment():
    shock = _shock()
    build_initial_geometry(
        t0=1.0e-5, Rc0=0.11, Rs0=0.12, R_sep=0.1,
        u_c_sep=1000.0, D_s_sep=2000.0, shock=shock,
    )
    with pytest.raises(GeometryError, match="outside every continuous"):
        build_initial_geometry(
            t0=2.5e-5, Rc0=0.11, Rs0=0.12, R_sep=0.1,
            u_c_sep=1000.0, D_s_sep=2000.0, shock=shock,
        )


def test_contact_interval_is_ordered_and_brackets_raw():
    table = ContactIntervalTable(
        case_id="c", times=np.array([1.0e-5, 2.0e-5]),
        low=np.array([0.10, 0.11]), up=np.array([0.12, 0.13]),
        raw=np.array([0.11, 0.12]), valid=np.array([True, False]),
    )
    assert table.n_valid == 1
    with pytest.raises(GeometryError):
        ContactIntervalTable(
            case_id="bad", times=np.array([1.0e-5]),
            low=np.array([0.12]), up=np.array([0.11]),
            raw=np.array([0.115]), valid=np.array([True]),
        )
    with pytest.raises(GeometryError):
        ContactIntervalTable(
            case_id="bad2", times=np.array([1.0e-5]),
            low=np.array([0.10]), up=np.array([0.12]),
            raw=np.array([0.13]), valid=np.array([True]),
        )
