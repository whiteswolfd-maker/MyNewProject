"""Data contract, interval signs, R_b detection and cache round-trip."""
from __future__ import annotations

import numpy as np
import torch

from spherical_blast_pinn.contact_air.data import (
    AirObservationStore,
    CaseArrays,
    CaseData,
    build_case,
    contact_interval_from_material,
    detect_right_boundary,
    load_case,
    save_case,
)
from spherical_blast_pinn.contact_air.losses import contact_data_loss, contact_ic_loss
from spherical_blast_pinn.contact_air.models import ContactFields
from spherical_blast_pinn.contact_air.sampling import CaseBatch
from spherical_blast_pinn.contact_air.testing import (
    DEFAULT_TOLERANCES,
    default_precursor,
    synthetic_case_for,
)


def test_contact_interval_uses_the_single_mixed_segment():
    radius = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    material = np.array([0, 0, 2, 1, 1])
    low, up, raw, valid, reason = contact_interval_from_material(radius, material)
    assert valid
    assert low == 0.02
    assert up == 0.04
    assert low <= raw <= up
    assert "single mixed" in reason


def test_no_mixed_cell_is_not_a_contact_interval():
    radius = np.array([0.01, 0.02, 0.03])
    material = np.array([0, 1, 1])
    _, _, _, valid, reason = contact_interval_from_material(radius, material)
    assert not valid
    assert "no mixed" in reason


def test_multiple_mixed_segments_and_missing_brackets_are_invalid():
    radius = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    material = np.array([0, 2, 1, 0, 2, 1])
    assert not contact_interval_from_material(radius, material)[3]
    material_no_product = np.array([0, 0, 2, 1, 1, 1])
    radius2 = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    # The mixed segment is bracketed here; remove the product by moving it out.
    material_no_product[0] = 1
    material_no_product[1] = 1
    assert not contact_interval_from_material(radius2, material_no_product)[3]


def test_rb_detection_requires_a_consecutive_ambient_run():
    radius = np.array([0.10, 0.11, 0.12, 0.13, 0.14])
    pressure = np.array([2.0e5, 1.0132e5, 1.0132e5, 1.0132e5, 1.0132e5])
    velocity = np.zeros(5)
    density = np.full(5, 1.225)
    material = np.ones(5, dtype=np.int8)
    value, ok, _ = detect_right_boundary(
        radius, pressure, velocity, density, material,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=0.1, tol_density=1e-6, tol_velocity=1e-6,
        min_ambient_cells=2,
    )
    assert ok and value == 0.11
    # A single isolated ambient cell does not define a boundary.
    pressure2 = np.array([2.0e5, 1.0132e5, 2.0e5, 2.0e5, 2.0e5])
    _, ok2, reason2 = detect_right_boundary(
        radius, pressure2, velocity, density, material,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=0.1, tol_density=1e-6, tol_velocity=1e-6,
        min_ambient_cells=2,
    )
    assert not ok2
    assert "isolated" in reason2


def test_cache_round_trip_preserves_contact_observations_and_boundary(tmp_path):
    case = synthetic_case_for(0.05, case_id="cache", n_frames=9, n_radial=120)
    path = save_case(case, tmp_path / "case.npz")
    loaded = load_case(path)
    assert loaded.case_id == case.case_id
    assert loaded.t0 == case.t0
    assert loaded.t_end == case.t_end
    assert loaded.observations.size == case.observations.size
    assert np.allclose(loaded.contact.low, case.contact.low)
    assert np.allclose(loaded.boundary.radii, case.boundary.radii)


def test_contact_interval_penalty_is_zero_inside_and_positive_outside():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=100)
    batch = CaseBatch(
        case_id=case.case_id,
        t_pde=torch.zeros(0, dtype=torch.float64), zeta_pde=torch.zeros(0, dtype=torch.float64),
        t_ic=torch.zeros(0, dtype=torch.float64),
        zeta_ic_shock=torch.zeros(0, dtype=torch.float64),
        zeta_ic_ambient=torch.zeros(0, dtype=torch.float64),
        t_right=torch.zeros(0, dtype=torch.float64),
        t_data=torch.zeros(0, dtype=torch.float64),
        zeta_data=torch.zeros(0, dtype=torch.float64),
        rho_data=torch.zeros(0, dtype=torch.float64),
        u_data=torch.zeros(0, dtype=torch.float64),
        p_data=torch.zeros(0, dtype=torch.float64),
        valid_rho=torch.zeros(0, dtype=torch.bool),
        valid_u=torch.zeros(0, dtype=torch.bool),
        valid_p=torch.zeros(0, dtype=torch.bool),
        t_contact=torch.tensor([case.t0], dtype=torch.float64),
        low_contact=torch.tensor([0.10], dtype=torch.float64),
        up_contact=torch.tensor([0.12], dtype=torch.float64),
    )
    inside = ContactFields(
        radius=torch.tensor([[0.11]], dtype=torch.float64),
        speed=torch.zeros(1, 1, dtype=torch.float64),
        sigmoid=torch.full((1, 1), 0.5, dtype=torch.float64),
        raw=torch.zeros(1, 1, dtype=torch.float64),
        pressure=torch.ones(1, 1, dtype=torch.float64),
        relative_volume=torch.ones(1, 1, dtype=torch.float64),
    )
    assert float(contact_data_loss(inside, batch, case).total) == 0.0
    outside = ContactFields(
        radius=torch.tensor([[0.09]], dtype=torch.float64), speed=inside.speed,
        sigmoid=inside.sigmoid, raw=inside.raw, pressure=inside.pressure,
        relative_volume=inside.relative_volume,
    )
    assert float(contact_data_loss(outside, batch, case).total) > 0.0
    edge = ContactFields(
        radius=torch.tensor([[0.10]], dtype=torch.float64), speed=inside.speed,
        sigmoid=inside.sigmoid, raw=inside.raw, pressure=inside.pressure,
        relative_volume=inside.relative_volume,
    )
    assert float(contact_data_loss(edge, batch, case).total) == 0.0


def test_missing_pressure_field_is_skipped_and_counted_not_zero_filled():
    from spherical_blast_pinn.contact_air.losses import air_data_loss
    from spherical_blast_pinn.contact_air.models import AirNet, scales_from_precursor
    from spherical_blast_pinn.contact_air.sampling import sample_case_batch

    case = synthetic_case_for(0.05, n_frames=7, n_radial=100)
    case.observations.valid_pressure[:] = False
    batch = sample_case_batch(
        case, n_pde=0, n_ic_shock=0, n_ic_ambient=0, n_right=0,
        n_data=8, n_contact=0, generator=torch.Generator().manual_seed(3),
    )
    net = AirNet(width=8, depth=1)
    scales = scales_from_precursor(case.precursor, like=torch.zeros(1, 1, dtype=torch.float64))
    result = air_data_loss(net, batch, case, scales)
    assert result.counts["data_p"] == 0
    assert result.counts["data_rho"] > 0
    assert result.counts["data_u"] > 0


def test_missing_observation_field_is_counted_not_zero_filled():
    empty = AirObservationStore(
        time=np.zeros(0), radius=np.zeros(0), zeta=np.zeros(0),
        density=np.zeros(0), velocity=np.zeros(0), pressure=np.zeros(0),
        valid_density=np.zeros(0, bool), valid_velocity=np.zeros(0, bool),
        valid_pressure=np.zeros(0, bool), valid_domain=np.zeros(0, bool),
        frame_index=np.zeros(0, np.int64),
    )
    sample = empty.sample(4, None)
    assert sample["count"] == 0
    assert sample["valid_rho"].numel() == 0
