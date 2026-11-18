"""Spherical-shell contact reference and pointwise contact loss."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.data import load_case, save_case
from spherical_blast_pinn.contact_air.geometry import (
    ContactIntervalTable,
    GeometryError,
)
from spherical_blast_pinn.contact_air.losses import (
    LossError,
    contact_point_loss,
    contact_point_rmse,
)
from spherical_blast_pinn.contact_air.models import ContactFields
from spherical_blast_pinn.contact_air.sampling import CaseBatch, sample_case_batch
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
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


def _batch(t_contact, low, up, reference):
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
        contact_reference=torch.as_tensor(reference, dtype=torch.float64),
    )


def _fields(radius):
    values = torch.as_tensor(radius, dtype=torch.float64).reshape(-1, 1)
    zeros = torch.zeros_like(values)
    return ContactFields(
        radius=values, speed=zeros, sigmoid=zeros, raw=zeros,
        pressure=zeros, relative_volume=zeros,
    )


def test_point_loss_is_mean_squared_distance_to_reference():
    case = _case([1.0], [1.001])
    batch = _batch([0.0], [1.0], [1.001], [1.0004])
    fields = _fields([1.0008])
    result = contact_point_loss(fields, batch, case)
    # (0.0004 / 0.1)^2 = 1.6e-5; cell-scale RMSE = 0.4
    assert result.total.item() == pytest.approx(1.6e-5)
    assert result.parts["contact_data"] == pytest.approx(1.6e-5)
    assert result.parts["contact_data_cells_rmse"] == pytest.approx(0.4)
    rmse, count = contact_point_rmse(fields, batch, case)
    assert count == 1
    assert rmse == pytest.approx(0.004)


def test_point_loss_requires_a_reference():
    case = _case([1.0], [1.001])
    batch = _batch([0.0], [1.0], [1.001], [1.0004])
    batch.contact_reference = None
    with pytest.raises(LossError, match="contact_reference"):
        contact_point_loss(_fields([1.0008]), batch, case)
    assert contact_point_rmse(_fields([1.0008]), batch, case) == (None, 0)


def test_contact_table_rejects_reference_outside_the_interval():
    with pytest.raises(GeometryError, match="reference outside"):
        ContactIntervalTable(
            case_id="c", times=np.array([0.0]), low=np.array([1.0]),
            up=np.array([1.001]), valid=np.array([True]),
            reference=np.array([1.002]),
        )


def test_contact_table_accepts_zero_width_reference_at_the_boundary():
    table = ContactIntervalTable(
        case_id="c", times=np.array([0.0, 1.0]), low=np.array([1.0, 1.0]),
        up=np.array([1.0, 1.001]), valid=np.array([True, True]),
        reference=np.array([1.0, 1.0005]),
        reference_source=np.array(["shared_boundary", "mixed_cell_spherical"]),
    )
    assert table.reference is not None
    assert table.as_dict()["reference_source_counts"] == {
        "shared_boundary": 1, "mixed_cell_spherical": 1,
    }


def test_sample_batch_carries_the_reference_and_save_load_roundtrips(tmp_path):
    case = synthetic_case_for(0.05, case_id="r050", n_frames=7, n_radial=800)
    n = case.contact.times.size
    reference = np.array(case.contact.low, copy=True)
    source = np.full(n, "shared_boundary", dtype="<U24")
    case.contact = ContactIntervalTable(
        case_id=case.contact.case_id, times=case.contact.times,
        low=case.contact.low, up=case.contact.up, valid=case.contact.valid,
        reasons=list(case.contact.reasons),
        reference=reference, reference_source=source,
    )
    batch = sample_case_batch(
        case, n_pde=1, n_ic=1, n_right=1, n_data=1, n_contact=3,
        generator=torch.Generator().manual_seed(3),
    )
    assert batch.contact_reference is not None
    assert batch.contact_reference.numel() == batch.t_contact.numel()

    path = save_case(case, tmp_path / "point_cache.npz")
    loaded = load_case(path, role="train")
    assert loaded.contact.reference is not None
    assert np.allclose(loaded.contact.reference, reference)
    assert loaded.contact.reference_source is not None
    assert np.all(loaded.contact.reference_source == "shared_boundary")


def test_train_config_validates_point_target_options():
    TrainConfig(contact_target="point", contact_point_scale="cell_width")
    with pytest.raises(Exception, match="contact_target"):
        TrainConfig(contact_target="other")
    with pytest.raises(Exception, match="contact_point_scale"):
        TrainConfig(contact_point_scale="other")
