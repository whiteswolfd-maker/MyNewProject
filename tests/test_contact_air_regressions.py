"""Regression tests for the 2026-11-05 contact-air review fixes.

These are deliberately multi-point tests.  A single point cannot reveal the
time-label mismatch, the ``[N,1]`` vs ``[N]`` broadcast, or a gap bridged by
deleting NaN frames.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air import cli
from spherical_blast_pinn.contact_air.data import (
    PredictionCaseData,
    RightBoundaryDetection,
    build_case,
    classify_separation_frames,
    contact_interval_from_material,
    detect_right_boundary_report,
    load_prediction_case,
    save_case,
    select_air_native_resolution,
)
from spherical_blast_pinn.contact_air.geometry import (
    ContactIntervalTable,
    GeometryError,
    RightBoundaryTrajectory,
)
from spherical_blast_pinn.contact_air.losses import (
    LossError,
    _masked_mse,
    contact_data_loss,
    masked_rmse,
    residual_scales,
)
from spherical_blast_pinn.contact_air.models import AirNet, ContactFields, ContactNet
from spherical_blast_pinn.contact_air.sampling import CaseBatch, sample_case_batch
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
from spherical_blast_pinn.contact_air.trainer import (
    AlternatingTrainer,
    TrainConfig,
    TrainingError,
)

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "configs" / "contact_air" / "smoke.yaml"


def _batch(t_contact, low, up, *, n_pde=0) -> CaseBatch:
    empty = torch.zeros(0, dtype=torch.float64)
    return CaseBatch(
        case_id="c",
        t_pde=torch.zeros(n_pde, dtype=torch.float64),
        zeta_pde=torch.zeros(n_pde, dtype=torch.float64),
        t_ic=empty, zeta_ic_shock=empty, zeta_ic_ambient=empty, t_right=empty,
        t_data=empty, zeta_data=empty, rho_data=empty, u_data=empty, p_data=empty,
        valid_rho=empty.bool(), valid_u=empty.bool(), valid_p=empty.bool(),
        t_contact=torch.as_tensor(t_contact, dtype=torch.float64),
        low_contact=torch.as_tensor(low, dtype=torch.float64),
        up_contact=torch.as_tensor(up, dtype=torch.float64),
    )


def _contact(radius) -> ContactFields:
    values = torch.as_tensor(radius, dtype=torch.float64).reshape(-1, 1)
    return ContactFields(
        radius=values, speed=torch.zeros_like(values),
        sigmoid=torch.ones_like(values), raw=torch.zeros_like(values),
        pressure=torch.ones_like(values), relative_volume=torch.ones_like(values),
    )


class _TimeContactNet(nn.Module):
    """A deliberately trivial ContactNet: radius = t + parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, t, r0, rb, rb_dot, closure, *, create_graph=True):
        t = t.reshape(-1, 1)
        radius = t + self.offset
        zeros = torch.zeros_like(radius)
        ones = torch.ones_like(radius)
        return ContactFields(
            radius=radius, speed=zeros, sigmoid=ones, raw=zeros,
            pressure=ones, relative_volume=ones,
        )


def _trainer(tmp_path, *, cases=None, air_width=8, config=None, fraction=0.3):
    cases = cases or [synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=160)]
    config = config or TrainConfig(
        n_pde=8, n_ic_shock=4, n_ic_ambient=4, n_right=0, n_data=8, n_contact=6,
        loops=1, learning_rate=1.0e-3, log_every=0, checkpoint_every=0,
        validate_every=0, seed=11, validation_holdout_fraction=fraction,
        validation_data_points=8, validation_contact_points=6,
    )
    return AlternatingTrainer(
        cases=cases, contact_net=ContactNet(width=8, depth=1),
        air_net=AirNet(width=air_width, depth=1), config=config,
        output_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# 1. contact data loss must use batch.t_contact
# ---------------------------------------------------------------------------
def test_contact_data_loss_is_evaluated_at_the_label_times(tmp_path, monkeypatch):
    import spherical_blast_pinn.contact_air.trainer as trainer_module
    from spherical_blast_pinn.contact_air.losses import contact_data_loss as real_loss

    trainer = _trainer(tmp_path, fraction=0.0)
    trainer.contact_net = _TimeContactNet()
    trainer.opt_contact = torch.optim.Adam(trainer.contact_net.parameters(), lr=1.0e-3)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def wrapper(contact, batch, case):
        seen.append((batch.t_contact.detach().clone(), contact.radius.detach().clone()))
        return real_loss(contact, batch, case)

    monkeypatch.setattr(trainer_module, "contact_data_loss", wrapper)
    trainer.step()
    assert seen
    label_time, radius = seen[0]
    assert label_time.numel() == radius.numel() > 1
    # The dummy network returns radius=t+offset with offset=0 at this point.
    assert torch.allclose(radius.reshape(-1), label_time, atol=1.0e-15)


def test_contact_data_loss_with_empty_labels_is_skipped_and_independent_of_pde_count():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=80)
    empty = _batch([], [], [])
    contact = _contact([0.1])
    result = contact_data_loss(contact, empty, case)
    assert result.counts["contact_data"] == 0
    assert float(result.total) == 0.0
    # A non-empty PDE batch must not change the contact-data term.
    pde_only = _batch([], [], [], n_pde=17)
    result2 = contact_data_loss(contact, pde_only, case)
    assert result2.counts["contact_data"] == 0
    assert float(result2.total) == 0.0


# ---------------------------------------------------------------------------
# 2. shape/broadcast and NaN-mask regressions
# ---------------------------------------------------------------------------
def test_masked_mse_compares_pointwise_not_crosswise():
    pred = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    mask = torch.ones(3, dtype=torch.bool)
    value, count = _masked_mse(pred, target, mask, 1.0)
    assert count == 3
    assert float(value) == 0.0
    rmse, count = masked_rmse(pred, target, mask, 1.0)
    assert count == 3
    assert rmse == 0.0


def test_masked_mse_ignores_nan_in_invalid_labels_only():
    pred = torch.tensor([[1.0], [123.0], [3.0]], dtype=torch.float64)
    target = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64)
    mask = torch.tensor([True, False, True])
    value, count = _masked_mse(pred, target, mask, 1.0)
    assert count == 2
    assert float(value) == 0.0
    with pytest.raises(LossError, match="valid data point"):
        _masked_mse(
            pred, torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64),
            torch.ones(3, dtype=torch.bool), 1.0,
        )


def test_only_the_changed_label_contributes_to_the_error():
    pred = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    value, count = masked_rmse(pred, target, torch.ones(3, dtype=torch.bool), 1.0)
    assert count == 3
    assert value == pytest.approx(np.sqrt(1.0 / 3.0), rel=1.0e-15)


def test_contact_intervals_are_compared_one_to_one_for_several_points():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=80)
    t = np.linspace(case.t0, case.t_end, 4)
    low = [0.100, 0.200, 0.300, 0.400]
    up = [0.101, 0.205, 0.305, 0.410]
    radius = [0.1005, 0.204, 0.301, 0.409]
    result = contact_data_loss(_contact(radius), _batch(t, low, up), case)
    assert result.counts["contact_data"] == 4
    assert float(result.total) == 0.0
    # Changing one prediction must move only that point's contribution.
    moved = list(radius)
    moved[2] = 0.500
    result2 = contact_data_loss(_contact(moved), _batch(t, low, up), case)
    expected = ((0.500 - 0.305) / case.length_scale) ** 2 / 4.0
    assert float(result2.total) == pytest.approx(expected, rel=1.0e-12)


# ---------------------------------------------------------------------------
# 3. R_b gaps, plateaus and valid intervals
# ---------------------------------------------------------------------------
def test_a_nan_radius_frame_remains_a_break_marker_and_t2_fails():
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id="gap", times=[1.0, 2.0, 3.0],
        radii=[0.1, float("nan"), 0.3], valid=[True, False, True],
    )
    assert boundary.times.size == 3
    assert boundary.valid.tolist() == [True, False, True]
    with pytest.raises(GeometryError):
        boundary.position(2.0)
    with pytest.raises(GeometryError):
        boundary.require_interval(1.0, 3.0)
    assert boundary.invalid_frames == []


def test_a_nan_radius_marked_valid_is_still_a_break_not_a_bridge():
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id="nan", times=[1.0, 2.0, 3.0],
        radii=[0.1, float("nan"), 0.3], valid=[True, True, True],
    )
    assert boundary.valid.tolist() == [True, False, True]
    assert boundary.invalid_frames[0]["index"] == 1
    with pytest.raises(GeometryError):
        boundary.position(2.0)
    with pytest.raises(GeometryError):
        boundary.position(2.5)


def test_equal_radii_are_a_plateau_not_a_rejected_segment():
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id="plateau", times=[1.0, 2.0, 3.0],
        radii=[0.1, 0.1, 0.2], valid=[True, True, True],
    )
    assert len(boundary.segments) == 1
    assert boundary.rejected == []
    assert boundary.plateaus
    assert boundary.position(2.0) == pytest.approx(0.1)
    slope = boundary.derivative(1.5)
    assert np.isfinite(slope)


def test_a_decreasing_radius_is_rejected_instead_of_repaired():
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id="drop", times=[1.0, 2.0, 3.0],
        radii=[0.1, 0.2, 0.19], valid=[True, True, True],
    )
    assert boundary.segments == []
    assert boundary.rejected


def test_two_valid_runs_do_not_create_a_sampling_interval_across_the_gap():
    boundary = RightBoundaryTrajectory.from_arrays(
        case_id="two", times=[1.0, 2.0, 3.0, 4.0, 5.0],
        radii=[0.1, 0.2, float("nan"), 0.4, 0.5],
        valid=[True, True, False, True, True],
    )
    assert boundary.continuous_intervals == [(1.0, 2.0), (4.0, 5.0)]
    assert boundary.continuous_interval_containing(1.5) == (1.0, 2.0)
    with pytest.raises(GeometryError):
        boundary.require_interval(1.0, 5.0)
    with pytest.raises(GeometryError):
        boundary.position(3.0)


# ---------------------------------------------------------------------------
# 4. energy residual scale
# ---------------------------------------------------------------------------
def test_pde_loss_uses_the_energy_density_residual_scale():
    from spherical_blast_pinn.contact_air.losses import pde_loss
    from spherical_blast_pinn.contact_air.mapping import build_geometry
    from spherical_blast_pinn.contact_air.models import AirFields

    case = synthetic_case_for(0.05, n_frames=5, n_radial=80)
    scales = residual_scales(case)
    n = 4
    rho_value = 2.0 * case.rho_scale
    u_value = 0.3 * case.u_scale
    p_value = 1.5 * case.p_scale
    shape = (n, 1)
    fields = AirFields(
        density=torch.full(shape, rho_value, dtype=torch.float64),
        velocity=torch.full(shape, u_value, dtype=torch.float64),
        pressure=torch.full(shape, p_value, dtype=torch.float64),
        specific_energy=torch.full(shape, 1.0, dtype=torch.float64),
    )
    zeros = torch.zeros(shape, dtype=torch.float64)
    derivatives = {
        "rho_t": zeros, "rho_z": zeros, "u_t": zeros, "u_z": zeros,
        "p_t": zeros, "p_z": zeros,
    }
    zeta = torch.linspace(0.2, 0.8, n, dtype=torch.float64).reshape(-1, 1)
    geometry = build_geometry(
        torch.zeros(shape, dtype=torch.float64), zeros,
        torch.ones(shape, dtype=torch.float64), zeros, zeta,
    )
    result = pde_loss(fields=fields, derivatives=derivatives, geometry=geometry, case=case)
    r = geometry.radius
    mass = 2.0 * rho_value * u_value / r
    momentum = 2.0 * rho_value * u_value ** 2 / r
    E = p_value / (case.gamma_air - 1.0) + 0.5 * rho_value * u_value ** 2
    energy = 2.0 * u_value * (E + p_value) / r
    expected = (
        ((mass / scales["mass_residual"]) ** 2).mean()
        + ((momentum / scales["momentum_residual"]) ** 2).mean()
        + ((energy / scales["energy_residual"]) ** 2).mean()
    )
    assert float(result.total) == pytest.approx(float(expected), rel=1.0e-12)


def test_native_resolution_is_selected_per_frame_not_by_field_existence():
    values, sources, reasons = select_air_native_resolution(
        4,
        primary=np.array([np.nan, 0.002, 0.003, np.nan]),
        fallback=np.array([0.001, 0.001, 0.001, 0.001]),
        fallback_applicable=np.array([True, True, False, True]),
        fallback_source="air_side_cell",
    )
    assert not np.isnan(values[0]) and values[0] == pytest.approx(0.001)
    assert values[1] == pytest.approx(0.002)
    assert values[2] == pytest.approx(0.003)
    assert values[3] == pytest.approx(0.001)
    assert sources[0] == "air_side_cell"
    assert sources[1] == "shock_cell_thickness_m"
    assert sources[2] == "shock_cell_thickness_m"
    assert "fallback" in reasons[0]


def test_native_resolution_never_uses_an_unmarked_fallback():
    values, sources, reasons = select_air_native_resolution(
        3,
        primary=np.array([np.nan, np.nan, 0.0]),
        fallback=np.array([0.001, 0.001, 0.001]),
        fallback_applicable=np.array([False, False, False]),
    )
    assert np.isnan(values).all()
    assert all(source == "" for source in sources)
    assert all("not applicable" in reason or "non-positive" in reason for reason in reasons)


def test_native_resolution_both_sources_invalid_stays_unresolved():
    values, sources, reasons = select_air_native_resolution(
        2,
        primary=np.array([np.nan, np.nan]),
        fallback=np.array([np.nan, -1.0]),
        fallback_applicable=np.array([True, True]),
    )
    assert np.isnan(values).all()
    assert np.isnan(values).all()
    assert all(source == "" for source in sources)
    assert all("no native resolution claimed" in reason for reason in reasons)


def test_energy_residual_scale_is_energy_density_over_time():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=80)
    scales = residual_scales(case)
    expected_E = case.p_scale / (case.gamma_air - 1.0) + 0.5 * case.rho_scale * case.u_scale ** 2
    expected_T = case.length_scale / case.u_scale
    assert scales["energy_density"] == pytest.approx(expected_E, rel=1.0e-15)
    assert scales["time"] == pytest.approx(expected_T, rel=1.0e-15)
    assert scales["energy_residual"] == pytest.approx(expected_E / expected_T, rel=1.0e-15)
    assert scales["specific_energy"] == pytest.approx(case.e_scale, rel=1.0e-15)
    assert scales["energy_density"] != pytest.approx(scales["specific_energy"])
    assert case.scale_dict()["energy_density_pa"] == pytest.approx(expected_E)
    assert case.scale_dict()["energy_residual_w_m3"] == pytest.approx(expected_E / expected_T)


# ---------------------------------------------------------------------------
# 5. separation and right-boundary evidence
# ---------------------------------------------------------------------------
def _separation_case(
    material, *, contact_low, contact_up, rb, identified=True, times=None,
    density_value=2.0, pressure_value=2.0, native_resolution=0.01,
    coverage=True, ambient_pressure=1.0, ambient_density=1.0,
):
    r = np.linspace(0.10, 0.15, len(material))
    t = times if times is not None else np.array([1.0e-5])
    contact = ContactIntervalTable(
        case_id="s", times=t, low=np.array([contact_low]),
        up=np.array([contact_up]), raw=np.array([0.5 * (contact_low + contact_up)]),
        valid=np.array([True]),
    )
    report = RightBoundaryDetection(
        radius=rb, identified=identified, evidence="test evidence" if identified else "",
        reasons=() if identified else ("no boundary",), n_ambient_cells=2,
        n_disturbed_air_samples=3,
    )
    return classify_separation_frames(
        case_id="s", times=t, radius=r, material=np.asarray([material]),
        density=np.full((1, len(material)), density_value),
        velocity=np.zeros((1, len(material))),
        pressure=np.full((1, len(material)), pressure_value),
        contact=contact, boundary=None, boundary_reports=[report],
        valid_density=np.ones((1, len(material)), dtype=bool),
        valid_velocity=np.ones((1, len(material)), dtype=bool),
        valid_pressure=np.ones((1, len(material)), dtype=bool),
        ambient_pressure=ambient_pressure, ambient_density=ambient_density,
        tol_pressure=0.1, tol_density=0.1, tol_velocity=0.1,
        min_shocked_air_samples=2,
        air_native_resolution_m=(
            None if native_resolution is None
            else np.array([native_resolution], dtype=np.float64)
        ),
        air_coverage=(
            None if not coverage
            else np.ones((1, len(material)), dtype=np.float64)
        ),
    )


def test_separated_requires_contact_layer_and_boundary_together():
    separated = _separation_case(
        [0, 0, 1, 1, 1, 2], contact_low=0.10, contact_up=0.105, rb=0.15
    )
    assert separated["status"] == "separated"
    assert separated["frames"][0]["n_valid_air_shell_samples"] >= 2
    assert separated["frames"][0]["n_disturbed_air_samples"] >= 2
    assert separated["frames"][0]["resolution_verified"] is True
    assert separated["frame_times"].startswith("raw extraction frame times")

    not_separated = _separation_case(
        [0, 0, 1, 2, 2, 2], contact_low=0.10, contact_up=0.11, rb=0.12
    )
    assert not_separated["status"] == "not_separated"
    assert not_separated["frames"][0]["shocked_air_resolvable"] is False

    unresolved = _separation_case(
        [0, 0, 1, 1, 1, 2], contact_low=0.10, contact_up=0.105, rb=0.15,
        identified=False,
    )
    assert unresolved["status"] == "unresolvable"
    assert unresolved["frames"][0]["right_boundary_identified"] is False


def test_static_ambient_air_between_contact_and_rb_is_not_a_shocked_layer():
    result = _separation_case(
        [0, 0, 1, 1, 1, 2], contact_low=0.10, contact_up=0.105, rb=0.15,
        density_value=1.0, pressure_value=1.0,
    )
    frame = result["frames"][0]
    assert frame["shocked_air_resolvable"] is False
    assert frame["n_disturbed_air_samples"] == 0
    assert result["status"] == "unresolvable"


def test_more_shells_do_not_override_too_coarse_native_resolution():
    # Same measured layer, same original mesh resolution; one fixture has more
    # shell samples.  Shell count alone must not turn unresolvable into
    # separated.
    coarse = _separation_case(
        [0, 0, 1, 1, 2, 2], contact_low=0.10, contact_up=0.105, rb=0.15,
        native_resolution=0.10,
    )
    many = _separation_case(
        [0, 0] + [1] * 8 + [2] * 6, contact_low=0.10, contact_up=0.105, rb=0.15,
        native_resolution=0.10,
    )
    assert coarse["frames"][0]["n_disturbed_air_samples"] >= 2
    assert many["frames"][0]["n_disturbed_air_samples"] > coarse["frames"][0]["n_disturbed_air_samples"]
    assert coarse["frames"][0]["resolution_verified"] is False
    assert many["frames"][0]["resolution_verified"] is False
    assert coarse["status"] == "unresolvable"
    assert many["status"] == "unresolvable"


def test_shell_count_without_native_resolution_is_not_resolvable():
    result = _separation_case(
        [0, 0, 1, 1, 1, 2], contact_low=0.10, contact_up=0.105, rb=0.15,
        native_resolution=None,
    )
    frame = result["frames"][0]
    assert frame["shocked_air_resolvable"] is False
    assert frame["resolution_verified"] is False
    assert result["status"] == "unresolvable"


def test_an_all_ambient_profile_is_not_a_right_boundary():
    r = np.linspace(0.1, 0.2, 6)
    report = detect_right_boundary_report(
        r, np.full(6, 101320.0), np.zeros(6), np.full(6, 1.225), np.ones(6),
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert "all-ambient" in " ".join(report.reasons)


def test_material_2_candidate_with_missing_fields_is_not_identified():
    r = np.linspace(0.1, 0.2, 6)
    pressure = np.array([1.0e7, 1.0e7, 1.0e7, np.nan, np.nan, np.nan])
    density = np.array([5.0, 5.0, 5.0, np.nan, np.nan, np.nan])
    material = np.array([1, 1, 1, 2, 2, 2])
    report = detect_right_boundary_report(
        r, pressure, np.zeros(6), density, material,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert report.candidate_radius == pytest.approx(r[3])
    assert report.verification == "unverified_missing_fields"
    assert "material==2" not in report.evidence


def test_material_2_candidate_with_finite_non_ambient_fields_is_rejected():
    r = np.linspace(0.1, 0.2, 6)
    pressure = np.array([1.0e7, 1.0e7, 1.0e7, 5.0e6, 4.0e6, 3.0e6])
    density = np.array([5.0, 5.0, 5.0, 2.0, 1.8, 1.6])
    material = np.array([1, 1, 1, 2, 2, 2])
    report = detect_right_boundary_report(
        r, pressure, np.zeros(6), density, material,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert report.verification == "rejected_non_ambient_fields"
    assert "counter-evidence" in " ".join(report.reasons)


def test_material_2_with_ambient_fields_is_still_only_a_candidate():
    r = np.linspace(0.1, 0.2, 6)
    pressure = np.array([1.0e7, 1.0e7, 1.0e7, 101320.0, 101320.0, 101320.0])
    density = np.array([5.0, 5.0, 5.0, 1.225, 1.225, 1.225])
    material = np.array([1, 1, 1, 2, 2, 2])
    report = detect_right_boundary_report(
        r, pressure, np.zeros(6), density, material,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert report.candidate_radius == pytest.approx(r[3])
    assert report.verification == "unverified_material_label"


def test_explicit_phase_labels_can_verify_an_ambient_boundary():
    r = np.linspace(0.1, 0.2, 6)
    pressure = np.array([1.0e7, 1.0e7, 1.0e7, 101320.0, 101320.0, 101320.0])
    density = np.array([5.0, 5.0, 5.0, 1.225, 1.225, 1.225])
    phase = np.array([1, 1, 1, 1, 1, 1])
    report = detect_right_boundary_report(
        r, pressure, np.zeros(6), density, phase,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
        material_is_phase=True,
    )
    assert report.identified
    assert report.verification == "verified_ambient_state"
    assert report.radius == pytest.approx(r[3])


def test_material_2_missing_point_in_the_connectivity_path_is_not_verified():
    # positions [1,2,3,4], states disturbed / NaN / ambient / ambient,
    # labels [1,2,2,2].  The missing material==2 point must break the path.
    r = np.array([0.10, 0.12, 0.14, 0.16])
    report = detect_right_boundary_report(
        r,
        np.array([1.0e7, np.nan, 101320.0, 101320.0]),
        np.zeros(4),
        np.array([5.0, np.nan, 1.225, 1.225]),
        np.array([1, 2, 2, 2]),
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert report.candidate_radius == pytest.approx(0.12)
    assert report.verification == "unverified_missing_fields"


def test_material_2_missing_point_between_disturbed_and_ambient_breaks_path():
    r = np.array([0.10, 0.12, 0.14, 0.16, 0.18])
    report = detect_right_boundary_report(
        r,
        np.array([1.0e7, np.nan, 1.0e7, 101320.0, 101320.0]),
        np.zeros(5),
        np.array([5.0, np.nan, 4.0, 1.225, 1.225]),
        np.array([1, 2, 2, 1, 1]),
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert "non-air / mixed / unknown" in " ".join(report.reasons)


def test_material_2_mixed_region_in_the_connectivity_path_is_not_verified():
    r = np.array([0.10, 0.12, 0.14, 0.16, 0.18])
    report = detect_right_boundary_report(
        r,
        np.array([1.0e7, 1.0e7, 1.0e7, 101320.0, 101320.0]),
        np.zeros(5),
        np.array([5.0, 4.0, 4.0, 1.225, 1.225]),
        np.array([1, 2, 2, 1, 1]),
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert "non-air / mixed / unknown" in " ".join(report.reasons)


def test_product_or_missing_air_interrupts_radial_connectivity():
    r = np.linspace(0.1, 0.2, 6)
    ambient = np.array([101320.0, 101320.0, 101320.0])
    good_p = np.array([1.0e7, 1.0e7])
    good_rho = np.array([5.0, 5.0])
    material = np.array([1, 1, 0, 2, 2, 2])
    report = detect_right_boundary_report(
        r, np.concatenate([good_p, [1.0], ambient]),
        np.zeros(6), np.concatenate([good_rho, [5.0], [1.225] * 3]), material,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert "non-air / mixed / unknown" in " ".join(report.reasons)

    material = np.array([1, 1, 1, 2, 2, 2])
    pressure = np.array([1.0e7, 1.0e7, np.nan] + list(ambient))
    density = np.array([5.0, 5.0, np.nan, 1.225, 1.225, 1.225])
    report = detect_right_boundary_report(
        r, pressure, np.zeros(6), density, material,
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=1.0e-9, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert "missing or invalid air cells" in " ".join(report.reasons)


def test_material_2_outside_the_shock_is_not_treated_as_contact_mixing():
    from spherical_blast_pinn.contact_air.data import (
        contact_interval_from_band,
        contact_interval_from_material,
    )

    radius = np.linspace(0.10, 0.20, 11)
    material = np.array([0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 2])
    # The legacy label reader must refuse this: material==2 forms a band at
    # the contact and another unlabelled region outside the shock.
    assert not contact_interval_from_material(radius, material)[3]
    low, up, raw, ok, _ = contact_interval_from_band(
        radius, material, 0.12, 0.13
    )
    assert ok
    assert low == pytest.approx(radius[1])
    assert up == pytest.approx(radius[4])
    assert low <= raw <= up


def test_an_isolated_ambient_point_inside_a_disturbed_region_is_not_a_boundary():
    r = np.linspace(0.1, 0.2, 5)
    pressure = np.array([1.0e7, 101320.0, 1.0e7, 1.0e7, 1.0e7])
    report = detect_right_boundary_report(
        r, pressure, np.zeros(5), np.full(5, 5.0), np.ones(5),
        ambient_pressure=101320.0, ambient_density=1.225,
        tol_pressure=1.0, tol_density=5.0, tol_velocity=1.0e-9,
    )
    assert not report.identified
    assert "isolated" in " ".join(report.reasons)


# ---------------------------------------------------------------------------
# 6. boundary-only prediction loader
# ---------------------------------------------------------------------------
def _strip_cache(full: Path, target: Path, *, t_end_override=None) -> Path:
    with np.load(full, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if t_end_override is not None:
            metadata["case"]["t_end"] = float(t_end_override)
        np.savez(
            target,
            metadata=np.asarray(json.dumps(metadata)),
            rb_times=archive["rb_times"], rb_radius=archive["rb_radius"],
            rb_valid=archive["rb_valid"],
        )
    return target


def _config_for_cache(
    tmp_path, cache: Path, *, case_id: str = "tnt_50mm", radius: float = 0.05,
    synthetic: bool = True, precursor_override=None, air_width=None,
    role: str = "train",
) -> Path:
    section = load_config(SMOKE).section("contact_air")
    section["cases"] = [{
        "case_id": case_id, "role": role, "charge_radius_m": radius,
        "cache": str(cache), "synthetic": synthetic,
    }]
    if precursor_override:
        for key, value in precursor_override.items():
            if isinstance(value, dict):
                section["precursor"].setdefault(key, {}).update(value)
            else:
                section["precursor"][key] = value
    if air_width is not None:
        section["networks"]["air"]["width"] = int(air_width)
    path = tmp_path / "case_config.yaml"
    path.write_text(yaml.safe_dump({"contact_air": section}), encoding="utf-8")
    return path


def test_synthetic_false_and_no_synthetic_reject_synthetic_caches(tmp_path, capsys):
    case = synthetic_case_for(0.05, case_id="tnt_50mm", role="train",
                              n_frames=7, n_radial=100)
    cache = save_case(case, tmp_path / "synthetic.npz")

    config = _config_for_cache(tmp_path, cache, synthetic=False)
    assert cli.main([
        "train", "--config", str(config), "--cases", "50", "--max-loops", "0",
        "--output-dir", str(tmp_path / "train_false"),
    ]) == 6
    assert "synthetic" in capsys.readouterr().err

    config = _config_for_cache(tmp_path, cache, synthetic=True)
    assert cli.main([
        "train", "--config", str(config), "--cases", "50", "--max-loops", "0",
        "--no-synthetic", "--output-dir", str(tmp_path / "train_no_synth"),
    ]) == 6
    assert "no-synthetic" in capsys.readouterr().err.lower()


def test_changed_material_parameters_refuse_an_old_cache(tmp_path, capsys):
    case = synthetic_case_for(0.05, case_id="tnt_50mm", role="train",
                              n_frames=7, n_radial=100)
    cache = save_case(case, tmp_path / "synthetic.npz")
    config = _config_for_cache(
        tmp_path, cache, synthetic=True,
        precursor_override={"jwl": {"A": 3.7120e11 * 1.01}},
    )
    assert cli.main([
        "train", "--config", str(config), "--cases", "50", "--max-loops", "0",
        "--output-dir", str(tmp_path / "train_changed"),
    ]) == 6
    error = capsys.readouterr().err
    assert "jwl.A" in error
    assert "build-cache" in error


def test_old_version_1_case_cache_is_refused_not_silently_reinterpreted(tmp_path):
    from spherical_blast_pinn.contact_air.data import CaseDataError, load_case

    path = tmp_path / "old.npz"
    np.savez(
        path,
        metadata=np.asarray(json.dumps({
            "format": "spherical_blast_pinn.contact_air_case", "version": 1,
            "case": {}, "precursor": {},
        })),
    )
    with pytest.raises(CaseDataError, match="version"):
        load_case(path)


def test_prediction_loader_reads_only_boundary_fields_and_ignores_internal_t_end(tmp_path):
    case = synthetic_case_for(0.5, case_id="r500", role="test", n_frames=9, n_radial=160)
    full = save_case(case, tmp_path / "full.npz")
    mini = _strip_cache(full, tmp_path / "mini.npz", t_end_override=123456.0)
    prediction = load_prediction_case(mini)
    assert isinstance(prediction, PredictionCaseData)
    assert not hasattr(prediction, "observations")
    assert not hasattr(prediction, "contact")
    assert prediction.t_end == pytest.approx(case.boundary.continuous_intervals[-1][1])
    assert prediction.source["prediction_fields_read"] == [
        "metadata", "rb_times", "rb_radius", "rb_valid"
    ]


def test_predict_500_runs_without_any_internal_reference_arrays(tmp_path, monkeypatch):
    case = synthetic_case_for(0.5, case_id="tnt_500mm", role="test", n_frames=9, n_radial=160)
    full = save_case(case, tmp_path / "full.npz")
    mini = _strip_cache(full, tmp_path / "mini.npz", t_end_override=123456.0)

    trainer = _trainer(tmp_path / "train", fraction=0.0)
    trainer.step()
    checkpoint = trainer.save(tmp_path / "last.pt")

    section = load_config(SMOKE).section("contact_air")
    section["cases"] = [{
        "case_id": "tnt_500mm", "role": "test", "charge_radius_m": 0.5,
        "cache": str(mini), "synthetic": True,
    }]
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(yaml.safe_dump({"contact_air": section}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("predict-500 called load_case and read internal arrays")

    monkeypatch.setattr(cli, "load_case", forbidden)
    output = tmp_path / "pred.npz"
    code = cli.main([
        "predict-500", "--config", str(config_path), "--checkpoint", str(checkpoint),
        "--output", str(output), "--n-times", "7", "--n-zeta", "5",
    ])
    assert code == 0
    with np.load(output, allow_pickle=False) as archive:
        assert archive["density"].shape == (7, 5)
        times = archive["times"]
        assert times[0] == pytest.approx(case.t0)
        assert times[-1] == pytest.approx(case.boundary.continuous_intervals[-1][1])
        scope = json.loads(str(archive["prediction_data_scope"]))
        assert scope["internal_supervision_used_for_time_range"] is False


# ---------------------------------------------------------------------------
# 7. sampling, validation and resume
# ---------------------------------------------------------------------------
def test_air_data_sampling_is_frame_first_then_point():
    from spherical_blast_pinn.contact_air.data import AirObservationStore

    # Frame 0 has 500 points, frame 1 has one.  Row-uniform sampling would pick
    # frame 0 almost always; frame-first sampling gives each frame ~half.
    n0, n1 = 500, 1
    store = AirObservationStore(
        time=np.concatenate([np.zeros(n0), np.ones(n1)]),
        radius=np.concatenate([np.linspace(0.1, 0.2, n0), np.array([0.3])]),
        zeta=np.concatenate([np.linspace(0, 1, n0), np.array([0.5])]),
        density=np.ones(n0 + n1), velocity=np.zeros(n0 + n1),
        pressure=np.ones(n0 + n1),
        valid_density=np.ones(n0 + n1, dtype=bool),
        valid_velocity=np.ones(n0 + n1, dtype=bool),
        valid_pressure=np.ones(n0 + n1, dtype=bool),
        valid_domain=np.ones(n0 + n1, dtype=bool),
        frame_index=np.concatenate([np.zeros(n0, dtype=np.int64),
                                    np.ones(n1, dtype=np.int64)]),
    )
    sample = store.sample(200, torch.Generator().manual_seed(5), t_min=0.0, t_max=1.0)
    times = sample["t"].numpy()
    assert np.count_nonzero(times == 0.0) > 60
    assert np.count_nonzero(times == 1.0) > 60


def test_contact_labels_outside_the_solve_window_are_not_sampled():
    case = synthetic_case_for(0.05, n_frames=9, n_radial=120)
    extra = case.t_end + 10.0
    table = ContactIntervalTable(
        case_id=case.case_id,
        times=np.append(case.contact.times, extra),
        low=np.append(case.contact.low, case.contact.low[-1]),
        up=np.append(case.contact.up, case.contact.up[-1]),
        raw=np.append(case.contact.raw, case.contact.raw[-1]),
        valid=np.append(case.contact.valid, True),
    )
    extended = dataclasses.replace(case, contact=table)
    batch = sample_case_batch(
        extended, n_pde=0, n_ic_shock=0, n_ic_ambient=0, n_right=0,
        n_data=0, n_contact=100, generator=torch.Generator().manual_seed(1),
    )
    assert batch.t_contact.numel() > 0
    assert float(batch.t_contact.max()) <= extended.t_end


def test_validation_unavailable_is_reported_not_scored_as_zero(tmp_path):
    trainer = _trainer(tmp_path, fraction=0.0)
    assert trainer.validation_cases == []
    verdict = trainer.evaluate()
    assert verdict["evaluable"] is False
    assert verdict["score"] is None
    trainer.config.validate_every = 1
    trainer.train(loops=1)
    assert trainer.history.best_loop is None
    assert not np.isfinite(trainer.history.best_score)


def test_a_single_usable_frame_cannot_be_silently_reused_as_validation(tmp_path):
    case = synthetic_case_for(0.05, n_frames=7, n_radial=100)
    valid = np.zeros(case.contact.times.size, dtype=bool)
    valid[0] = True
    table = ContactIntervalTable(
        case_id=case.case_id, times=case.contact.times,
        low=case.contact.low, up=case.contact.up, raw=case.contact.raw, valid=valid,
    )
    single = dataclasses.replace(case, contact=table)
    with pytest.raises(TrainingError, match="at least two"):
        _trainer(tmp_path, cases=[single], fraction=0.3)


def test_all_generated_target_gradients_must_be_finite(tmp_path):
    trainer = _trainer(tmp_path, fraction=0.0)
    for parameter in trainer.air_net.parameters():
        parameter.grad = torch.zeros_like(parameter)
    first = next(trainer.air_net.parameters())
    first.grad[0] = float("nan")
    with pytest.raises(TrainingError, match="non-finite"):
        trainer._check_target_gradients(
            trainer.air_net, trainer.contact_net, stage="test"
        )


def test_resume_refuses_a_different_air_architecture_before_loading_anything(tmp_path):
    trainer = _trainer(tmp_path, fraction=0.0)
    trainer.step()
    checkpoint = trainer.save(tmp_path / "last.pt")
    fresh = _trainer(tmp_path / "fresh", air_width=9, fraction=0.0)
    contact_before = [p.detach().clone() for p in fresh.contact_net.parameters()]
    with pytest.raises(TrainingError, match="architecture"):
        fresh.load(checkpoint)
    assert all(torch.equal(p, q) for p, q in zip(contact_before, fresh.contact_net.parameters()))


def test_resume_refuses_changed_data_or_boundary(tmp_path):
    case = synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=160)
    trainer = _trainer(tmp_path, cases=[case], fraction=0.0)
    trainer.step()
    checkpoint = trainer.save(tmp_path / "last.pt")

    changed = synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=160)
    changed.observations.density[0] += 1.0
    other = _trainer(tmp_path / "other", cases=[changed], fraction=0.0)
    with pytest.raises(TrainingError, match="signature"):
        other.load(checkpoint)

    moved = synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=160)
    moved.boundary = RightBoundaryTrajectory.from_arrays(
        case_id=moved.case_id, times=moved.boundary.times,
        radii=moved.boundary.radii + 1.0e-6, valid=moved.boundary.valid,
    )
    moved.initial = dataclasses.replace(moved.initial, Rb0=moved.initial.Rb0 + 1.0e-6)
    moved.boundary_shift = None
    third = _trainer(tmp_path / "third", cases=[moved], fraction=0.0)
    with pytest.raises(TrainingError, match="signature"):
        third.load(checkpoint)


def test_resume_restores_the_fixed_validation_points_and_rng(tmp_path):
    trainer = _trainer(tmp_path, fraction=0.3)
    trainer.step()
    checkpoint = trainer.save(tmp_path / "last.pt")
    original = [
        (sample.batch.t_data.clone(), sample.batch.zeta_data.clone(),
         sample.batch.t_contact.clone())
        for sample in trainer._validation
    ]
    saved_rng = torch.get_rng_state().clone()
    fresh = _trainer(tmp_path / "fresh", fraction=0.3)
    fresh.load(checkpoint)
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert len(fresh._validation) == len(original)
    for sample, (t_data, zeta_data, t_contact) in zip(fresh._validation, original):
        assert torch.equal(sample.batch.t_data, t_data)
        assert torch.equal(sample.batch.zeta_data, zeta_data)
        assert torch.equal(sample.batch.t_contact, t_contact)


def test_best_score_zero_survives_checkpoint_round_trip(tmp_path):
    trainer = _trainer(tmp_path, fraction=0.3)
    trainer.step()
    trainer.history.best_score = 0.0
    trainer.history.best_loop = 7
    checkpoint = trainer.save(tmp_path / "zero_best.pt")
    fresh = _trainer(tmp_path / "fresh", fraction=0.3)
    fresh.load(checkpoint)
    assert fresh.history.best_score == 0.0
    assert fresh.history.best_loop == 7


def test_prediction_refuses_checkpoint_with_different_shared_physics(tmp_path, capsys):
    from spherical_blast_pinn.contact_air.data import synthetic_case
    from spherical_blast_pinn.contact_air.precursor import (
        PrecursorConfig,
        compute_precursor,
    )
    from spherical_blast_pinn.contact_air.testing import (
        DEFAULT_PRECURSOR, DEFAULT_TOLERANCES,
    )

    checkpoint = _trainer(tmp_path / "train", fraction=0.0)
    checkpoint.step()
    checkpoint_path = checkpoint.save(tmp_path / "last.pt")

    payload = dict(DEFAULT_PRECURSOR)
    payload["charge_radius_m"] = 0.5
    payload["epsilon_s"] = 2.0e-6
    predictor = compute_precursor(PrecursorConfig.from_mapping(payload))
    prediction_case = synthetic_case(
        case_id="tnt_500mm", charge_radius_m=0.5, precursor=predictor,
        role="test", n_frames=7, n_radial=120, tolerances=DEFAULT_TOLERANCES,
    )
    full = save_case(prediction_case, tmp_path / "prediction_full.npz")
    mini = _strip_cache(full, tmp_path / "prediction_mini.npz")
    config = _config_for_cache(
        tmp_path, mini, case_id="tnt_500mm", radius=0.5, role="test",
        synthetic=True, precursor_override={"epsilon_s": 2.0e-6},
    )
    code = cli.main([
        "predict-500", "--config", str(config), "--checkpoint", str(checkpoint_path),
        "--output", str(tmp_path / "bad_prediction.npz"),
    ])
    assert code == 6
    error = capsys.readouterr().err
    assert "epsilon_s" in error
    assert "prediction physics" in error


def test_prediction_refuses_incompatible_network_convention(tmp_path, capsys):
    prediction_case = synthetic_case_for(
        0.5, case_id="tnt_500mm", role="test", n_frames=7, n_radial=120
    )
    full = save_case(prediction_case, tmp_path / "prediction_full.npz")
    mini = _strip_cache(full, tmp_path / "prediction_mini.npz")
    checkpoint = _trainer(tmp_path / "train", fraction=0.0)
    checkpoint.step()
    checkpoint_path = checkpoint.save(tmp_path / "last.pt")
    config = _config_for_cache(
        tmp_path, mini, case_id="tnt_500mm", radius=0.5, role="test",
        synthetic=True, air_width=9,
    )
    assert cli.main([
        "predict-500", "--config", str(config), "--checkpoint", str(checkpoint_path),
        "--output", str(tmp_path / "bad_prediction.npz"),
    ]) == 6
    assert "architecture" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# 8. clock and initial compatibility reporting
# ---------------------------------------------------------------------------
def test_clock_and_material_provenance_are_marked_unverified():
    from spherical_blast_pinn.contact_air.precursor import (
        PrecursorConfig,
        PrecursorError,
        compute_precursor,
    )
    from spherical_blast_pinn.contact_air.testing import DEFAULT_PRECURSOR

    result = compute_precursor(
        PrecursorConfig.from_mapping(dict(DEFAULT_PRECURSOR, charge_radius_m=0.05))
    )
    assert result.clock["time_formula"] == "t_cj = R_cj / D"
    assert result.clock["time_formula_verified"] is False
    assert result.clock["d3plot_clock_verified"] is False
    assert result.clock["material_source_verified"] is False
    assert "not verified" in result.clock["d3plot_clock_relation"]
    compatibility = result.initial_pressure_compatibility
    assert compatibility["forced_equal"] is False
    assert compatibility["relative_difference"] == pytest.approx(
        abs(result.p_c_t0 - result.p_sep) / result.p_sep, rel=1.0e-15
    )
    assert compatibility["relative_difference"] > 1.0e-2

    payload = dict(DEFAULT_PRECURSOR, charge_radius_m=0.05,
                   material_source="unverified_config_defaults",
                   material_source_verified=True)
    with pytest.raises(PrecursorError, match="material-card source"):
        PrecursorConfig.from_mapping(payload)


def test_initial_pressure_difference_is_reported_by_its_actual_size():
    case = synthetic_case_for(0.05, n_frames=7, n_radial=100)
    compatibility = case.initial_compatibility()
    assert compatibility["forced_equal"] is False
    assert compatibility["relative_difference"] == pytest.approx(
        abs(case.p_c_at_t0 - case.p_scale) / case.p_scale, rel=1.0e-15
    )
    assert compatibility["relative_difference"] > 1.0e-2
    assert "non-negligible" in compatibility["assessment"]


# ---------------------------------------------------------------------------
# extraction-support provenance
# ---------------------------------------------------------------------------
def _write_real_style_profile(
    directory, *, times, precursor, shock_thickness=None,
    fallback_resolution=None, fallback_applicable=None,
    contact_native_resolution=None,
):
    """A fixture with the real extractor's material==2 ambiguity.

    material==2 is used for both the contact mixing band and every shell
    outside the detected shock.  The outside shells are still unlabelled for
    training, but their shell-averaged fields and per-field validity are
    retained so R_b can be verified.
    """
    directory.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.02, 0.20, 120)
    n_t = len(times)
    material = np.full((n_t, grid.size), 2, dtype=np.int8)
    material[:, grid < 0.105] = 0
    material[:, (grid >= 0.11) & (grid < 0.16)] = 1
    alpha = np.zeros((n_t, grid.size))
    alpha[:, grid < 0.105] = 1.0
    alpha[:, (grid >= 0.105) & (grid < 0.11)] = 0.5
    material_phase = np.full((n_t, grid.size), 1, dtype=np.int8)
    material_phase[:, grid < 0.105] = 0
    material_phase[:, (grid >= 0.105) & (grid < 0.11)] = 2
    pressure = np.full((n_t, grid.size), np.nan)
    density = np.full((n_t, grid.size), np.nan)
    velocity = np.zeros((n_t, grid.size))
    energy = np.full((n_t, grid.size), np.nan)
    air = material == 1
    product = material == 0
    outside_air = (material == 2) & (alpha <= 0.05)
    pressure[air] = precursor.p_sep
    density[air] = precursor.rho_a_sep
    energy[air] = precursor.e_a_sep
    pressure[product] = precursor.p_sep
    density[product] = precursor.rho_a_sep
    energy[product] = precursor.e_a_sep
    pressure[outside_air] = precursor.air_pressure
    density[outside_air] = precursor.air_density
    energy[outside_air] = precursor.air_pressure / (
        (precursor.gamma_air - 1.0) * precursor.air_density
    )
    np.savez(
        directory / "raw_line.npz", t=np.asarray(times, dtype=float), x=grid,
        rho=density, u=velocity, P=pressure, e=energy,
    )
    training_valid = material != 2
    field_valid = training_valid | outside_air
    coverage_air = np.ones((n_t, grid.size), dtype=np.float64)
    extra = {}
    if shock_thickness is None:
        shock_thickness = np.full(n_t, 0.002)
    if shock_thickness is not None:
        extra["shock_cell_thickness_m"] = np.asarray(shock_thickness, dtype=np.float64)
    if fallback_resolution is not None:
        extra["air_native_resolution_fallback_m"] = np.asarray(
            fallback_resolution, dtype=np.float64
        )
    if fallback_applicable is not None:
        extra["air_native_resolution_fallback_applicable"] = np.asarray(
            fallback_applicable, dtype=bool
        )
    if contact_native_resolution is not None:
        extra["contact_native_resolution_m"] = np.asarray(
            contact_native_resolution, dtype=np.float64
        )
    np.savez(
        directory / "material.npz",
        alpha_product=alpha, material=material, material_phase=material_phase,
        valid=training_valid,
        field_valid_density=field_valid, field_valid_velocity=field_valid,
        field_valid_energy=field_valid, field_valid_pressure=field_valid,
        boundary_air_location_valid=outside_air,
        coverage_air=coverage_air,
        contact_radius=np.full(n_t, 0.1075),
        shock_radius=np.full(n_t, 0.16),
        contact_mixing_band_inner_m=np.full(n_t, 0.105),
        contact_mixing_band_outer_m=np.full(n_t, 0.11),
        contact_has_supervision=np.ones(n_t, dtype=bool),
        contact_state=np.full(n_t, 2, dtype=np.int64),
        shock_state=np.full(n_t, 2, dtype=np.int64),
        shell_count=np.ones((n_t, grid.size), dtype=np.int64),
        **extra,
    )
    (directory / "metadata.json").write_text(json.dumps({
        "R_0": precursor.charge_radius, "t_0": float(times[0]),
        "t_end": float(times[-1]), "n_x": grid.size, "n_t": n_t,
        "x_end": float(grid[-1]), "symmetry_factor": 8.0,
        "material_partition_is_volume_fraction_based": True,
        "material_partition_source": {
            "method": "ale_volume_fraction", "producer": "cli.extract_profiles",
            "producer_version": 1, "fields": ["tnt_volume_fraction"],
            "input_identifier": "sha256:test",
            "detail": {"pure_air_threshold": 0.05, "pure_product_threshold": 0.95},
        },
        "field_map": {"time": "t", "radius": "x", "density": "rho",
                      "velocity": "u", "pressure": "P"},
        "profile_layout": {"version": 1, "layout": "frames_by_radius"},
    }), encoding="utf-8")
    return directory


def test_real_style_profile_verifies_outer_air_by_state_not_label(tmp_path):
    from spherical_blast_pinn.contact_air.data import build_case_from_profile
    from spherical_blast_pinn.contact_air.testing import (
        DEFAULT_TOLERANCES, default_precursor,
    )

    precursor = default_precursor(0.05)
    times = precursor.t0 + np.arange(6) * 2.0e-6
    directory = _write_real_style_profile(
        tmp_path / "profile", times=times, precursor=precursor
    )
    case = build_case_from_profile(
        directory, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
    )
    assert case.contact.n_valid == len(times)
    assert np.all(case.contact.low > 0.10) and np.all(case.contact.low < 0.105)
    assert np.all(case.contact.up > 0.107) and np.all(case.contact.up < 0.112)
    assert case.separation["status"] == "separated"
    assert case.separation["right_boundary_identified_count"] == len(times)
    assert all(
        frame["shocked_air_resolvable"]
        for frame in case.separation["frames"]
    )
    assert all(
        "finite ambient state" in frame["boundary_evidence"]
        for frame in case.separation["frames"]
    )
    assert case.source["fields_are_shell_averaged"] is True
    assert case.source["air_native_resolution_primary_source"] == "shock_cell_thickness_m"
    assert "shock_cell_thickness_m" in case.source["air_native_resolution_sources"]
    assert case.source["raw_cell_ids_available"] is False
    assert case.source["raw_cell_radial_ranges_available"] is False
    assert case.source["shell_count_is_original_cell_count"] is False


def test_extract_profiles_phase_survives_old_shock_supervision_withdrawal(tmp_path, monkeypatch):
    from tests.test_second_review_regressions import _run_extract_case
    from spherical_blast_pinn.cli import extract_profiles as ep
    from spherical_blast_pinn.contact_air.data import build_case_from_profile
    from spherical_blast_pinn.contact_air.testing import (
        DEFAULT_TOLERANCES, default_precursor,
    )
    from spherical_blast_pinn.data.extracted_case import load_extracted_case

    precursor = default_precursor(0.05)
    times = precursor.t0 + np.arange(4) * 2.0e-6
    contact = [0.115, 0.120, 0.126, 0.135]
    shock = [0.160, 0.180, 0.210, 0.260]

    normal = _run_extract_case(
        tmp_path / "normal", times=times, contact=contact, shock=shock,
        air_overpressure=True, monkeypatch=monkeypatch,
    )
    original_detect = ep.detect_shock

    def without_supervision(*args, **kwargs):
        detection = original_detect(*args, **kwargs)
        return dataclasses.replace(detection, supervision_radius_range=None)

    monkeypatch.setattr(ep, "detect_shock", without_supervision)
    nosup = _run_extract_case(
        tmp_path / "nosup", times=times, contact=contact, shock=shock,
        air_overpressure=True, monkeypatch=monkeypatch,
    )
    normal_z = np.load(Path(normal.output_dir) / "material.npz")
    nosup_z = np.load(Path(nosup.output_dir) / "material.npz")
    # The physical phase field is unchanged; only the old supervision/material
    # withdrawal differs.
    assert np.array_equal(
        np.asarray(normal_z["material_phase"]),
        np.asarray(nosup_z["material_phase"]),
    )
    assert np.any(
        (np.asarray(normal_z["material"]) == 1)
        & (np.asarray(nosup_z["material"]) == 2)
    )
    assert np.any(
        (np.asarray(nosup_z["material_phase"]) == 1)
        & (np.asarray(nosup_z["material"]) == 2)
    )
    extracted = load_extracted_case(nosup.output_dir)
    assert extracted.material_phase is not None
    assert np.array_equal(extracted.material_phase, np.asarray(nosup_z["material_phase"]))
    case = build_case_from_profile(
        nosup.output_dir, precursor=precursor, role="train",
        tolerances=DEFAULT_TOLERANCES,
    )
    assert case.contact.n_valid == len(times)
    assert case.separation["right_boundary_identified_count"] == len(times)
    assert case.source["material_phase_source"] == "material_phase"


def test_resolution_fallback_is_per_frame_and_requires_applicability(tmp_path):
    from spherical_blast_pinn.contact_air.data import build_case_from_profile
    from spherical_blast_pinn.contact_air.testing import (
        DEFAULT_TOLERANCES, default_precursor,
    )

    precursor = default_precursor(0.05)
    times = precursor.t0 + np.arange(6) * 2.0e-6
    n_t = len(times)
    primary = np.full(n_t, np.nan)
    fallback = np.full(n_t, 0.002)
    applicable = np.array([True, True, True, False, False, False])

    directory = _write_real_style_profile(
        tmp_path / "fallback", times=times, precursor=precursor,
        shock_thickness=primary, fallback_resolution=fallback,
        fallback_applicable=applicable,
    )
    case = build_case_from_profile(
        directory, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
    )
    frames = case.separation["frames"]
    assert frames[0]["native_resolution_source"] == "air_native_resolution_fallback_m"
    assert frames[0]["resolution_verified"] is True
    assert frames[3]["resolution_verified"] is False
    assert frames[3]["native_resolution_m"] is None
    assert "not applicable" in frames[3]["native_resolution_reason"]

    # A contact/product-side thickness is not used without an applicability mark.
    contact_only = _write_real_style_profile(
        tmp_path / "contact_only", times=times, precursor=precursor,
        shock_thickness=primary,
        contact_native_resolution=np.full(n_t, 0.002),
    )
    contact_case = build_case_from_profile(
        contact_only, precursor=precursor, role="train",
        tolerances=DEFAULT_TOLERANCES,
    )
    assert contact_case.separation["frames"][0]["resolution_verified"] is False
    assert contact_case.separation["status"] == "unresolvable"

    both_invalid = _write_real_style_profile(
        tmp_path / "both_invalid", times=times, precursor=precursor,
        shock_thickness=primary, fallback_resolution=np.full(n_t, np.nan),
        fallback_applicable=np.ones(n_t, dtype=bool),
    )
    invalid_case = build_case_from_profile(
        both_invalid, precursor=precursor, role="train",
        tolerances=DEFAULT_TOLERANCES,
    )
    assert invalid_case.separation["frames"][0]["resolution_verified"] is False
    assert invalid_case.separation["status"] == "unresolvable"


def test_profile_bridge_reports_shell_midpoint_support_not_cell_centres(tmp_path):
    from tests.test_contact_air_extraction_bridge import _write_profile
    from spherical_blast_pinn.contact_air.data import build_case_from_profile
    from spherical_blast_pinn.contact_air.testing import (
        DEFAULT_TOLERANCES, default_precursor,
    )

    precursor = default_precursor(0.05)
    times = precursor.t0 + np.arange(6) * 2.0e-6
    directory = _write_profile(tmp_path / "profile", times=times, precursor=precursor)
    case = build_case_from_profile(
        directory, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
    )
    assert case.source["fields_are_shell_averaged"] is True
    assert case.source["radial_coordinate_origin"] == "shell midpoint (geometric), not cell centroid"
    assert "shell" in case.source["radial_support"]
    assert any("not cell centroid" in note for note in case.notes)
