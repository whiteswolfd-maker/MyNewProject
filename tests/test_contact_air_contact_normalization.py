"""Contact-interval loss normalization and robust loss shape."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.losses import (
    contact_data_loss,
    contact_violation_cells_rmse,
    contact_violation_rmse,
)
from spherical_blast_pinn.contact_air.models import ContactFields
from spherical_blast_pinn.contact_air.sampling import CaseBatch
from spherical_blast_pinn.contact_air.trainer import TrainConfig


def _case(low, up, *, length_scale=0.1, charge_radius_m=0.05):
    return SimpleNamespace(
        contact=SimpleNamespace(
            low=np.asarray(low, dtype=np.float64),
            up=np.asarray(up, dtype=np.float64),
        ),
        length_scale=float(length_scale),
        charge_radius_m=float(charge_radius_m),
    )


def _batch(t_contact, low, up):
    empty = torch.zeros(0, dtype=torch.float64)
    empty_bool = empty.bool()
    return CaseBatch(
        case_id="c", t_pde=empty, zeta_pde=empty, t_ic=empty,
        zeta_ic=empty, t_right=empty, t_data=empty, radius_data=empty,
        rho_data=empty, u_data=empty, p_data=empty,
        valid_rho=empty_bool, valid_u=empty_bool, valid_p=empty_bool,
        t_contact=torch.as_tensor(t_contact, dtype=torch.float64),
        low_contact=torch.as_tensor(low, dtype=torch.float64),
        up_contact=torch.as_tensor(up, dtype=torch.float64),
    )


def _fields(radius):
    values = torch.as_tensor(radius, dtype=torch.float64).reshape(-1, 1)
    zeros = torch.zeros_like(values)
    return ContactFields(
        radius=values, speed=zeros, sigmoid=zeros, raw=zeros,
        pressure=zeros, relative_volume=zeros,
    )


def test_default_contact_loss_is_the_historical_separation_radius_loss():
    case = _case([1.0, 1.0], [1.1, 1.1])
    batch = _batch([0.0, 0.0], [1.0, 1.0], [1.1, 1.1])
    fields = _fields([1.2, 0.9])
    result = contact_data_loss(fields, batch, case)
    # Both points are 0.1 m outside; divided by R_sep=0.1, each squared term is 1.
    assert result.total.item() == pytest.approx(1.0)
    assert result.parts["contact_data"] == pytest.approx(1.0)
    assert result.parts["contact_data_cells_rmse"] == pytest.approx(1.0)


def test_interval_width_smooth_l1_is_one_half_at_one_cell():
    case = _case([1.0], [1.001])
    batch = _batch([0.0], [1.0], [1.001])
    fields = _fields([1.002])  # 1 mm outside; one 1 mm cell
    result = contact_data_loss(
        fields, batch, case,
        normalization="interval_width", loss_shape="smooth_l1",
        beta=2.0, delta=1.0,
    )
    # r = 1 -> smooth_l1 = 0.5; beta=2 -> 1.0
    assert result.total.item() == pytest.approx(1.0)
    assert result.parts["contact_data_raw"] == pytest.approx(0.5)
    assert result.parts["contact_data_cells_rmse"] == pytest.approx(1.0)


def test_interval_width_uses_one_cell_for_zero_width_labels():
    # One positive cell (1 mm) and one zero-width label; the zero-width label
    # must use the 1 mm cell width, not divide by zero.
    case = _case([1.0, 1.0], [1.001, 1.0])
    batch = _batch([0.0, 0.0], [1.0, 1.0], [1.001, 1.0])
    fields = _fields([1.002, 1.002])  # 1 mm and 2 mm outside
    result = contact_data_loss(
        fields, batch, case,
        normalization="interval_width", loss_shape="smooth_l1",
        beta=1.0, delta=1.0,
    )
    # r = 1 and r = 2 -> smooth_l1 = 0.5 and 1.5 -> mean = 1.0
    assert result.total.item() == pytest.approx(1.0)
    assert result.parts["contact_data_cells_rmse"] == pytest.approx(
        np.sqrt((1.0 + 4.0) / 2.0)
    )


def test_cell_width_fallback_when_every_interval_is_zero_width():
    case = _case([1.0], [1.0], charge_radius_m=0.05)
    batch = _batch([0.0], [1.0], [1.0])
    fields = _fields([1.0002])  # 0.2 mm outside; fallback cell = R0/500 = 0.1 mm
    result = contact_data_loss(
        fields, batch, case,
        normalization="interval_width", loss_shape="smooth_l1",
        beta=1.0, delta=1.0,
    )
    # r = 2 -> smooth_l1 = 1.5
    assert result.total.item() == pytest.approx(1.5)


def test_cell_rmse_matches_the_contact_violation_in_cells():
    case = _case([1.0], [1.001])
    batch = _batch([0.0], [1.0], [1.001])
    fields = _fields([1.002])
    cells, count = contact_violation_cells_rmse(fields, batch, case)
    assert count == 1
    assert cells == pytest.approx(1.0)
    old, old_count = contact_violation_rmse(fields, batch, case)
    assert old_count == 1
    assert old == pytest.approx(0.01)  # 1 mm / R_sep=0.1 m


def test_bad_contact_loss_options_are_rejected():
    case = _case([1.0], [1.001])
    batch = _batch([0.0], [1.0], [1.001])
    fields = _fields([1.002])
    with pytest.raises(Exception, match="normalization"):
        contact_data_loss(fields, batch, case, normalization="other")
    with pytest.raises(Exception, match="loss shape"):
        contact_data_loss(fields, batch, case, loss_shape="other")
    with pytest.raises(Exception, match="beta"):
        contact_data_loss(fields, batch, case, beta=0.0)
    with pytest.raises(Exception, match="delta"):
        contact_data_loss(fields, batch, case, delta=0.0)


def test_train_config_validates_contact_loss_options():
    TrainConfig(
        contact_normalization="interval_width",
        contact_loss_shape="smooth_l1",
        contact_loss_beta=0.5,
        contact_smooth_l1_delta=1.0,
    )
    with pytest.raises(Exception, match="contact_normalization"):
        TrainConfig(contact_normalization="other")
    with pytest.raises(Exception, match="contact_loss_shape"):
        TrainConfig(contact_loss_shape="other")
