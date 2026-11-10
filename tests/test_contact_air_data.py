"""1-D contact labels, fields/masks, cache and data-coordinate checks."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spherical_blast_pinn.contact_air.data import (
    CaseDataError,
    ContactLabelConfig,
    load_case,
    load_prediction_case,
    save_case,
)
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
from spherical_blast_pinn.contact_air.geometry import ContactIntervalTable
from spherical_blast_pinn.contact_air.losses import (
    LossError,
    contact_data_loss,
    data_coordinates,
)
from spherical_blast_pinn.contact_air.models import ContactFields
from spherical_blast_pinn.contact_air.one_d import (
    OneDDataError,
    contact_interval_from_fractions,
    read_standard_1d,
    write_standard_1d,
)
from tests.contact_air_fixtures import make_frame as _frame
from spherical_blast_pinn.contact_air.sampling import CaseBatch


def test_contact_interval_uses_actual_cell_boundaries_not_alpha_half_or_midpoint():
    frame = _frame(alpha=[1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    low, up, valid, reason = contact_interval_from_fractions(frame)
    assert valid
    assert low == pytest.approx(frame.r_left[3])
    assert up == pytest.approx(frame.r_right[4])
    assert low < up
    assert "actual cell boundaries" in reason


def test_direct_pure_pure_transition_uses_the_shared_boundary():
    frame = _frame(alpha=[1.0, 1.0, 0.0, 0.0])
    low, up, valid, reason = contact_interval_from_fractions(frame)
    assert valid
    assert low == up == pytest.approx(frame.r_left[2])
    assert "direct pure-product" in reason


def test_an_isolated_outer_product_fragment_is_not_the_contact():
    # Main product -> mixing -> air, then an isolated product fragment further out.
    frame = _frame(alpha=[1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0])
    low, up, valid, _ = contact_interval_from_fractions(frame)
    assert valid
    assert up == pytest.approx(frame.r_right[3])
    assert up < frame.r_left[6]


def test_nonfinite_volume_fraction_is_not_a_label():
    frame = _frame(alpha=[1.0, 1.0, float("nan"), 0.0])
    assert not contact_interval_from_fractions(frame)[2]


def test_standard_source_converts_declared_units_to_si(tmp_path):
    frame = {
        "radius": np.array([5.0, 8.0]), "r_left": np.array([4.0, 6.0]),
        "r_right": np.array([6.0, 10.0]), "alpha_product": np.array([1.0, 0.0]),
        "density": np.array([1.0, 1.0]),
        "velocity": np.array([2.0, 2.0]), "pressure": np.array([0.1, 0.1]),
    }
    path = write_standard_1d(
        tmp_path / "src.npz", case_id="u", charge_radius_m=0.05,
        frames=[frame, frame], frame_times=[1.0, 2.0],
        units={"length": "mm", "time": "ms", "density": "g/cm^3",
               "velocity": "km/s", "pressure": "GPa"},
    )
    source = read_standard_1d(path)
    cell = source.frames[0]
    assert cell.radius[0] == pytest.approx(5.0e-3)
    assert cell.r_left[0] == pytest.approx(4.0e-3)
    assert cell.density[0] == pytest.approx(1.0e3)
    assert cell.velocity[0] == pytest.approx(2.0e3)
    assert cell.pressure[0] == pytest.approx(1.0e8)
    assert cell.time_raw_s == pytest.approx(1.0e-3)
    assert source.charge_radius_m == pytest.approx(0.05)


def test_missing_units_and_gauge_pressure_are_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, metadata=np.asarray('{"format":"spherical_blast_pinn.standard_1d_source","version":1}'))
    with pytest.raises(OneDDataError):
        read_standard_1d(path)


def test_a_true_mask_over_nan_is_an_error_not_a_dropped_sample(tmp_path):
    frame = {
        "radius": np.array([1.0, 2.0, 3.0]), "r_left": np.array([0.5, 1.5, 2.5]),
        "r_right": np.array([1.5, 2.5, 3.5]), "alpha_product": np.array([1.0, 1.0, 0.0]),
        "density": np.array([1.0, np.nan, 1.0]), "velocity": np.zeros(3),
        "pressure": np.full(3, 1e5), "valid_density": np.ones(3, bool),
    }
    path = write_standard_1d(
        tmp_path / "badmask.npz", case_id="b", charge_radius_m=0.05,
        frames=[frame, frame], frame_times=[1.0, 2.0],
    )
    with pytest.raises(OneDDataError, match="non-finite density"):
        read_standard_1d(path)


def _base_source_arrays():
    return {
        "radius": np.array([5.0, 8.0]), "r_left": np.array([4.0, 6.0]),
        "r_right": np.array([6.0, 10.0]), "alpha_product": np.array([1.0, 0.0]),
        "density": np.array([1.0, 1.0]), "velocity": np.array([2.0, 2.0]),
        "pressure": np.array([0.1, 0.1]),
    }


def _rewrite_metadata(source, target, mutate):
    import json

    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(payload["metadata"]))
    mutate(metadata)
    payload["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(target, **payload)
    return target


def test_missing_coordinate_or_velocity_component_is_rejected(tmp_path):
    frame = _base_source_arrays()
    source = write_standard_1d(
        tmp_path / "source.npz", case_id="m", charge_radius_m=0.05,
        frames=[frame, frame], frame_times=[1.0, 2.0],
    )
    for key in ("coordinate", "velocity_component"):
        broken = _rewrite_metadata(
            source, tmp_path / f"no_{key}.npz", lambda md, key=key: md.pop(key)
        )
        with pytest.raises(OneDDataError, match=key):
            read_standard_1d(broken)


def test_node_source_requires_the_supported_preconversion(tmp_path):
    frame = _base_source_arrays()
    unsupported = write_standard_1d(
        tmp_path / "node_bad.npz", case_id="n", charge_radius_m=0.05,
        frames=[frame, frame], frame_times=[1.0, 2.0],
        position_type="node", node_to_cell_convention="midpoints_guess",
    )
    with pytest.raises(OneDDataError, match="node_to_cell_convention"):
        read_standard_1d(unsupported)
    supported = write_standard_1d(
        tmp_path / "node_ok.npz", case_id="n", charge_radius_m=0.05,
        frames=[frame, frame], frame_times=[1.0, 2.0],
        position_type="node",
        node_to_cell_convention="preconverted_cell_representatives",
    )
    source = read_standard_1d(supported)
    assert source.position_type == "node"
    assert source.source["node_to_cell_convention"] == "preconverted_cell_representatives"


def test_cache_round_trip_preserves_shock_and_physical_observation_radius(tmp_path):
    case = synthetic_case_for(0.05, case_id="cache", n_frames=9, n_radial=1600)
    path = save_case(case, tmp_path / "case.npz")
    loaded = load_case(path)
    assert loaded.case_id == case.case_id
    assert loaded.t0 == case.t0
    assert loaded.t_end == case.t_end
    assert loaded.observations.size == case.observations.size
    assert np.allclose(loaded.shock.radii, case.shock.radii, equal_nan=True)
    assert np.allclose(loaded.observations.radius, case.observations.radius)
    assert loaded.event_alignment["offset_s"] == pytest.approx(
        case.event_alignment["offset_s"]
    )


def test_prediction_loader_reads_only_the_shock_blocks(tmp_path):
    case = synthetic_case_for(0.05, case_id="pred", n_frames=9, n_radial=1600)
    full = save_case(case, tmp_path / "full.npz")
    with np.load(full, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in ("metadata", "shock_times",
                                                             "shock_radius", "shock_valid")}
    stripped = tmp_path / "pred.npz"
    np.savez(stripped, **payload)
    prediction = load_prediction_case(stripped)
    assert prediction.case_id == case.case_id
    assert prediction.event_alignment["offset_s"] == pytest.approx(
        case.event_alignment["offset_s"]
    )


def test_contact_interval_penalty_is_zero_inside_and_positive_outside():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=1200)
    batch = CaseBatch(
        case_id=case.case_id,
        t_pde=torch.zeros(0, dtype=torch.float64), zeta_pde=torch.zeros(0, dtype=torch.float64),
        t_ic=torch.zeros(0, dtype=torch.float64), zeta_ic=torch.zeros(0, dtype=torch.float64),
        t_right=torch.zeros(0, dtype=torch.float64),
        t_data=torch.zeros(0, dtype=torch.float64),
        radius_data=torch.zeros(0, dtype=torch.float64),
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

    def contact(radius):
        values = torch.tensor([[radius]], dtype=torch.float64)
        return ContactFields(
            radius=values, speed=torch.zeros_like(values),
            sigmoid=torch.ones_like(values), raw=torch.zeros_like(values),
            pressure=torch.ones_like(values), relative_volume=torch.ones_like(values),
        )

    assert float(contact_data_loss(contact(0.11), batch, case).total) == 0.0
    assert float(contact_data_loss(contact(0.09), batch, case).total) > 0.0
    assert float(contact_data_loss(contact(0.13), batch, case).total) > 0.0


def test_data_coordinates_use_the_predicted_contact_and_shock():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=1200)
    t = torch.tensor([case.t0], dtype=torch.float64)
    rs, _ = case.shock_tensors(t)

    def contact(rc):
        values = torch.tensor([[rc]], dtype=torch.float64)
        return ContactFields(
            radius=values, speed=torch.zeros_like(values),
            sigmoid=torch.ones_like(values), raw=torch.zeros_like(values),
            pressure=torch.ones_like(values), relative_volume=torch.ones_like(values),
        )

    radius = torch.tensor([[0.5 * (case.initial.Rc0 + float(rs))]], dtype=torch.float64)
    zeta_a, inside_a, _ = data_coordinates(
        radius=radius, t=t, contact=contact(case.initial.Rc0), case=case
    )
    zeta_b, inside_b, _ = data_coordinates(
        radius=radius, t=t, contact=contact(case.initial.Rc0 * 1.01), case=case
    )
    assert bool(inside_a.all()) and bool(inside_b.all())
    assert not torch.allclose(zeta_a, zeta_b)
    # The contact interval midpoint is not used: changing R_c changes zeta.
    midpoint = 0.5 * (float(case.contact.low[0]) + float(case.contact.up[0]))
    zeta_c, _, _ = data_coordinates(
        radius=radius, t=t, contact=contact(midpoint), case=case
    )
    assert not torch.allclose(zeta_a, zeta_c)


def test_all_data_points_outside_the_geometry_are_not_silently_dropped():
    from spherical_blast_pinn.contact_air.losses import air_data_loss
    from spherical_blast_pinn.contact_air.models import AirNet, scales_from_precursor

    case = synthetic_case_for(0.05, n_frames=5, n_radial=1200)
    scales = scales_from_precursor(case.precursor, like=torch.zeros(1, 1))
    batch = CaseBatch(
        case_id=case.case_id,
        t_pde=torch.zeros(0, dtype=torch.float64), zeta_pde=torch.zeros(0, dtype=torch.float64),
        t_ic=torch.zeros(0, dtype=torch.float64), zeta_ic=torch.zeros(0, dtype=torch.float64),
        t_right=torch.zeros(0, dtype=torch.float64),
        t_data=torch.tensor([case.t0], dtype=torch.float64),
        radius_data=torch.tensor([10.0], dtype=torch.float64),
        rho_data=torch.ones(1, dtype=torch.float64),
        u_data=torch.zeros(1, dtype=torch.float64),
        p_data=torch.ones(1, dtype=torch.float64),
        valid_rho=torch.ones(1, dtype=torch.bool),
        valid_u=torch.ones(1, dtype=torch.bool),
        valid_p=torch.ones(1, dtype=torch.bool),
        t_contact=torch.zeros(0, dtype=torch.float64),
        low_contact=torch.zeros(0, dtype=torch.float64),
        up_contact=torch.zeros(0, dtype=torch.float64),
    )
    contact = ContactFields(
        radius=torch.tensor([[case.initial.Rc0]], dtype=torch.float64),
        speed=torch.zeros(1, 1, dtype=torch.float64),
        sigmoid=torch.ones(1, 1, dtype=torch.float64),
        raw=torch.zeros(1, 1, dtype=torch.float64),
        pressure=torch.ones(1, 1, dtype=torch.float64),
        relative_volume=torch.ones(1, 1, dtype=torch.float64),
    )
    net = AirNet(width=8, depth=1)
    with pytest.raises(LossError, match="outside the current"):
        air_data_loss(net, batch, case, scales, contact)
