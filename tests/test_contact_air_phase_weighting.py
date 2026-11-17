"""Always-on theta cell weighting for contact labels and air observations."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.losses import (
    LossError,
    air_data_loss,
    contact_data_loss,
    contact_weighted_violation_rmse,
    masked_rmse,
)
from spherical_blast_pinn.contact_air.models import (
    AirNet,
    ContactFields,
    scales_from_precursor,
)
from spherical_blast_pinn.contact_air.sampling import (
    CaseBatch,
    air_observation_weights,
    phase_time_weights,
    sample_case_batch,
)
from spherical_blast_pinn.contact_air.testing import synthetic_case_for


def _case(times):
    return SimpleNamespace(
        contact=SimpleNamespace(times=np.asarray(times, dtype=np.float64)),
        t0=0.0,
        time_scale=1.0,
    )


def _batch(t_contact, low, up, weights=None):
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
        contact_weights=(
            None if weights is None
            else torch.as_tensor(weights, dtype=torch.float64)
        ),
    )


def _contact(radius):
    return SimpleNamespace(
        radius=torch.as_tensor(radius, dtype=torch.float64).reshape(-1, 1)
    )


def test_time_weights_are_local_raw_output_cells():
    weights = phase_time_weights(_case([0.0, 1.0, 2.0, 10.0, 11.0]))
    assert np.allclose(weights, [0.5, 1.0, 4.5, 4.5, 0.5])
    assert weights.sum() == pytest.approx(11.0)


def test_repeated_times_share_one_cell():
    weights = phase_time_weights(_case([0.0, 1.0, 1.0, 2.0]))
    assert np.allclose(weights, [0.5, 0.5, 0.5, 0.5])
    assert weights.sum() == pytest.approx(2.0)


def test_single_frame_has_positive_normalised_weight():
    assert np.array_equal(phase_time_weights(_case([5.0])), [1.0])
    assert phase_time_weights(_case([])).size == 0


def test_phase_weights_use_raw_cadence_across_invalid_frames():
    # Raw frames 0, 1, 2; the contact row at frame 1 is invalid and is absent
    # from the subset contact table.  The valid rows at 0 and 2 must get only
    # their half-cells, not the whole 0..2 interval.
    case = SimpleNamespace(
        contact=SimpleNamespace(times=np.asarray([0.0, 2.0])),
        raw_times_s=np.asarray([0.0, 1.0, 2.0]),
        event_alignment={"offset_s": 0.0},
        t0=0.0,
        time_scale=1.0,
    )
    assert np.allclose(phase_time_weights(case), [0.5, 0.5])


def test_weighted_masked_rmse_uses_only_selected_weights():
    pred = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    target = torch.zeros(3, dtype=torch.float64)
    mask = torch.tensor([False, True, True])
    value, count = masked_rmse(
        pred, target, mask, 1.0, weights=[100.0, 1.0, 3.0]
    )
    assert count == 2
    assert value == pytest.approx(np.sqrt((1.0 * 4.0 + 3.0 * 9.0) / 4.0))


def test_air_observation_weights_split_each_frame_cell_over_its_draws():
    # Frames 0, 1, 2 have theta cells 0.5, 1.0, 0.5.  Two draws come from
    # frame 0, one from frame 2, and none from frame 1.
    weights = air_observation_weights(_case([0.0, 1.0, 2.0]), [0.0, 0.0, 2.0])
    assert np.allclose(weights, [0.25, 0.25, 0.5])


def test_air_observation_weights_share_a_repeated_time_cell():
    # Contact rows 1 and 2 share the unique time cell of total size 1.0.
    # Two air draws at that time must split the whole cell (0.5 each), not
    # only the first duplicate row's half-cell (which would sum to 0.5).
    case = _case([0.0, 1.0, 1.0, 2.0])
    assert np.allclose(phase_time_weights(case), [0.5, 0.5, 0.5, 0.5])
    weights = air_observation_weights(case, [1.0, 1.0])
    assert np.allclose(weights, [0.5, 0.5])
    assert weights.sum() == pytest.approx(1.0)


def test_weighted_loss_is_a_weighted_mean_and_keeps_the_unweighted_value():
    batch = _batch([0.0, 1.0], [0.5, 0.5], [0.6, 0.6], weights=[1.0, 3.0])
    contact = _contact([0.55, 0.0])  # inside, then 0.25 violation
    result = contact_data_loss(contact, batch, SimpleNamespace(length_scale=1.0))
    assert float(result.total) == pytest.approx(0.1875)
    assert result.parts["contact_data_unweighted"] == pytest.approx(0.125)
    assert result.counts["contact_data"] == 2
    assert result.parts["contact_weight_ess"] == pytest.approx(1.6)
    weighted_rmse, count = contact_weighted_violation_rmse(
        contact, batch, SimpleNamespace(length_scale=1.0)
    )
    assert weighted_rmse == pytest.approx(np.sqrt(0.1875))
    assert count == 2


def test_uniform_weights_reproduce_the_old_loss():
    batch = _batch([0.0, 1.0], [0.5, 0.5], [0.6, 0.6], weights=[1.0, 1.0])
    contact = _contact([0.55, 0.0])
    result = contact_data_loss(contact, batch, SimpleNamespace(length_scale=1.0))
    assert float(result.total) == pytest.approx(0.125)


def test_invalid_weight_vectors_are_rejected():
    contact = _contact([0.55, 0.0])
    case = SimpleNamespace(length_scale=1.0)
    for weights in ([1.0], [1.0, -1.0], [0.0, 0.0], [1.0, float("nan")]):
        batch = _batch([0.0, 1.0], [0.5, 0.5], [0.6, 0.6], weights=weights)
        with pytest.raises(LossError, match="contact weights"):
            contact_data_loss(contact, batch, case)


def test_sampling_attaches_aligned_contact_and_air_weights():
    case = synthetic_case_for(0.05, n_frames=11, n_radial=200)
    batch = sample_case_batch(
        case, n_pde=0, n_ic=0, n_right=0, n_data=16, n_contact=4,
        generator=torch.Generator().manual_seed(7),
    )
    assert batch.contact_weights.numel() == batch.t_contact.numel()
    assert batch.data_weights.numel() == batch.t_data.numel()
    assert bool((batch.contact_weights > 0.0).all())
    assert bool((batch.data_weights > 0.0).all())
    indices = np.searchsorted(
        case.contact.times, batch.t_contact.detach().numpy()
    )
    expected = phase_time_weights(case)[indices]
    assert np.allclose(batch.contact_weights.detach().numpy(), expected)


def test_subset_keeps_full_raw_clock_for_theta_cells():
    case = synthetic_case_for(0.05, n_frames=11, n_radial=200)
    case.contact.valid[3] = False
    selected = np.flatnonzero(case.contact.valid)
    subset = case.subset_frames(selected, role="train")
    assert subset.raw_times_s.size == case.raw_times_s.size
    assert np.allclose(
        phase_time_weights(subset), phase_time_weights(case)[selected]
    )
    batch = sample_case_batch(
        subset, n_pde=0, n_ic=0, n_right=0, n_data=8, n_contact=4,
        generator=torch.Generator().manual_seed(15),
    )
    assert batch.contact_weights.numel() == batch.t_contact.numel()
    assert batch.data_weights.numel() == batch.t_data.numel()


def test_validation_frame_subset_maps_air_weights_by_time():
    case = synthetic_case_for(0.05, n_frames=11, n_radial=200)
    frames = np.unique(case.observations.frame_index)
    subset = case.subset_frames(frames[-5:], role="validation")
    batch = sample_case_batch(
        subset, n_pde=0, n_ic=0, n_right=0, n_data=8, n_contact=2,
        generator=torch.Generator().manual_seed(13),
    )
    assert batch.data_weights.numel() == batch.t_data.numel()
    assert bool((batch.data_weights > 0.0).all())


def test_air_data_loss_uses_frame_cell_weights():
    case = synthetic_case_for(0.05, n_frames=11, n_radial=400)
    batch = sample_case_batch(
        case, n_pde=0, n_ic=0, n_right=0, n_data=24, n_contact=0,
        generator=torch.Generator().manual_seed(11),
    )
    plain = dataclasses.replace(batch, data_weights=None)
    net = AirNet(width=8, depth=2, time_reference_s=1.0)
    scales = scales_from_precursor(case.precursor, like=torch.zeros(1, 1))
    contact = ContactFields(
        radius=torch.full(
            (batch.t_data.numel(), 1), float(case.initial.Rc0),
            dtype=torch.float64,
        ),
        speed=torch.zeros(batch.t_data.numel(), 1, dtype=torch.float64),
        sigmoid=torch.ones(batch.t_data.numel(), 1, dtype=torch.float64),
        raw=torch.zeros(batch.t_data.numel(), 1, dtype=torch.float64),
        pressure=torch.ones(batch.t_data.numel(), 1, dtype=torch.float64),
        relative_volume=torch.ones(batch.t_data.numel(), 1, dtype=torch.float64),
    )
    weighted = air_data_loss(net, batch, case, scales, contact)
    unweighted = air_data_loss(net, plain, case, scales, contact)
    assert "data_rho_unweighted" in weighted.parts
    assert "data_weight_ess" in weighted.parts
    assert "data_rho_unweighted" not in unweighted.parts
    assert weighted.counts["data_rho"] == unweighted.counts["data_rho"]
