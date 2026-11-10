"""Shared regression checks for the one-dimensional contact-air scheme.

These are deliberately multi-point tests.  A single point cannot reveal the
time-label mismatch, the ``[N,1]`` vs ``[N]`` broadcast, or a shock gap bridged
by deleting NaN frames.
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
    load_case,
    save_case,
)
from spherical_blast_pinn.contact_air.testing import synthetic_case_for
from spherical_blast_pinn.contact_air.geometry import (
    GeometryError,
    ShockTrajectory,
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
        t_ic=empty, zeta_ic=empty, t_right=empty,
        t_data=empty, radius_data=empty,
        rho_data=empty, u_data=empty, p_data=empty,
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
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, t, r0, rs, ds, closure, *, create_graph=True):
        t = t.reshape(-1, 1)
        radius = t + self.offset
        zeros = torch.zeros_like(radius)
        ones = torch.ones_like(radius)
        return ContactFields(
            radius=radius, speed=zeros, sigmoid=ones, raw=zeros,
            pressure=ones, relative_volume=ones,
        )


def _trainer(tmp_path, *, cases=None, air_width=8, config=None, fraction=0.3):
    cases = cases or [synthetic_case_for(0.05, case_id="r050", n_frames=11, n_radial=1600)]
    config = config or TrainConfig(
        n_pde=8, n_ic=4, n_right=0, n_data=8, n_contact=6,
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
    assert torch.allclose(radius.reshape(-1), label_time, atol=1.0e-15)


def test_contact_data_loss_with_empty_labels_is_skipped_and_independent_of_pde_count():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=800)
    empty = _batch([], [], [])
    result = contact_data_loss(_contact([0.1]), empty, case)
    assert result.counts["contact_data"] == 0
    assert float(result.total) == 0.0
    pde_only = _batch([], [], [], n_pde=17)
    result2 = contact_data_loss(_contact([0.1]), pde_only, case)
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
    assert count == 3 and float(value) == 0.0
    rmse, count = masked_rmse(pred, target, mask, 1.0)
    assert count == 3 and rmse == 0.0


def test_masked_mse_ignores_nan_in_invalid_labels_only():
    pred = torch.tensor([[1.0], [123.0], [3.0]], dtype=torch.float64)
    target = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64)
    mask = torch.tensor([True, False, True])
    value, count = _masked_mse(pred, target, mask, 1.0)
    assert count == 2 and float(value) == 0.0
    with pytest.raises(LossError, match="valid data point"):
        _masked_mse(
            pred, torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64),
            torch.ones(3, dtype=torch.bool), 1.0,
        )


def test_contact_intervals_are_compared_one_to_one_for_several_points():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=800)
    times = [case.t0 + i * 1.0e-7 for i in range(4)]
    low = [0.10, 0.11, 0.12, 0.13]
    up = [0.105, 0.115, 0.125, 0.135]
    batch = _batch(times, low, up)
    inside = _contact([0.102, 0.112, 0.122, 0.132])
    assert float(contact_data_loss(inside, batch, case).total) == 0.0
    outside = _contact([0.099, 0.116, 0.121, 0.140])
    values, count = contact_data_loss(outside, batch, case).total, 4
    assert count == 4 and float(values) > 0.0


def test_contact_initial_condition_constrains_position_only():
    from spherical_blast_pinn.contact_air.losses import contact_ic_loss

    case = synthetic_case_for(0.05, n_frames=5, n_radial=800)
    radius = torch.tensor([[case.initial.Rc0]], dtype=torch.float64)

    def fields(speed):
        return ContactFields(
            radius=radius, speed=torch.full_like(radius, speed),
            sigmoid=torch.ones_like(radius), raw=torch.zeros_like(radius),
            pressure=torch.ones_like(radius),
            relative_volume=torch.ones_like(radius),
        )

    assert float(contact_ic_loss(fields(0.0), case).total) == 0.0
    assert float(contact_ic_loss(fields(1.0e9), case).total) == 0.0


def test_air_initial_condition_is_only_the_separated_layer():
    from spherical_blast_pinn.contact_air.losses import air_ic_loss
    from spherical_blast_pinn.contact_air.models import AirNet, scales_from_precursor

    case = synthetic_case_for(0.05, n_frames=5, n_radial=800)
    zeta = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    batch = CaseBatch(
        case_id=case.case_id,
        t_pde=torch.zeros(0, dtype=torch.float64), zeta_pde=torch.zeros(0, dtype=torch.float64),
        t_ic=torch.full_like(zeta, case.t0), zeta_ic=zeta,
        t_right=torch.zeros(0, dtype=torch.float64),
        t_data=torch.zeros(0, dtype=torch.float64),
        radius_data=torch.zeros(0, dtype=torch.float64),
        rho_data=torch.zeros(0, dtype=torch.float64),
        u_data=torch.zeros(0, dtype=torch.float64),
        p_data=torch.zeros(0, dtype=torch.float64),
        valid_rho=torch.zeros(0, dtype=torch.bool),
        valid_u=torch.zeros(0, dtype=torch.bool),
        valid_p=torch.zeros(0, dtype=torch.bool),
        t_contact=torch.zeros(0, dtype=torch.float64),
        low_contact=torch.zeros(0, dtype=torch.float64),
        up_contact=torch.zeros(0, dtype=torch.float64),
    )
    assert not hasattr(batch, "zeta_ic_ambient")
    result = air_ic_loss(
        AirNet(width=8, depth=1), batch, case,
        scales_from_precursor(case.precursor, like=torch.zeros(1, 1)),
    )
    assert result.counts["ic"] == zeta.numel()


# ---------------------------------------------------------------------------
# 3. shock trajectory break markers# ---------------------------------------------------------------------------
# 3. shock trajectory break markers
# ---------------------------------------------------------------------------
def test_a_nan_radius_frame_remains_a_break_marker():
    times = np.linspace(1.0e-5, 3.0e-5, 5)
    radii = np.array([0.15, np.nan, 0.17, 0.18, 0.19])
    valid = np.ones(times.size, dtype=bool)
    shock = ShockTrajectory.from_arrays(case_id="gap", times=times, radii=radii, valid=valid)
    assert len(shock.invalid_frames) == 1
    assert shock.position(1.0e-5) == pytest.approx(0.15)
    with pytest.raises(GeometryError):
        shock.position(1.5e-5)


def test_equal_radii_are_a_plateau_not_a_rejected_segment():
    times = np.linspace(1.0e-5, 2.0e-5, 4)
    radii = np.array([0.15, 0.15, 0.16, 0.17])
    shock = ShockTrajectory.from_arrays(case_id="p", times=times, radii=radii)
    assert shock.plateaus
    assert shock.derivative(1.1e-5) == pytest.approx(0.0)


def test_a_decreasing_radius_is_rejected_instead_of_repaired():
    times = np.linspace(1.0e-5, 2.0e-5, 4)
    radii = np.array([0.15, 0.14, 0.16, 0.17])
    shock = ShockTrajectory.from_arrays(case_id="d", times=times, radii=radii)
    assert shock.rejected
    assert not shock.segments


def test_a_gap_does_not_create_a_sampling_interval():
    case = synthetic_case_for(0.05, n_frames=9, n_radial=1200)
    assert len(case.shock.segments) == 1
    start, end = case.shock.continuous_intervals[0]
    assert start <= case.t0 <= end


def test_sampling_refuses_a_trajectory_without_supersonic_times():
    import dataclasses
    from spherical_blast_pinn.contact_air.geometry import ShockTrajectory
    from spherical_blast_pinn.contact_air.sampling import SamplingError, sample_case_batch

    case = synthetic_case_for(0.05, n_frames=7, n_radial=800)
    constant = ShockTrajectory.from_arrays(
        case_id=case.case_id, times=case.shock.times,
        radii=np.full(case.shock.times.size, case.initial.Rs_data_at_t0),
    )
    frozen = dataclasses.replace(case, shock=constant)
    with pytest.raises(SamplingError, match="D_s > c_inf"):
        sample_case_batch(
            frozen, n_pde=1, n_ic=0, n_right=4, n_data=0, n_contact=0,
            generator=torch.Generator().manual_seed(0),
        )


# ---------------------------------------------------------------------------
# 4. loss scales# ---------------------------------------------------------------------------
# 4. loss scales
# ---------------------------------------------------------------------------
def test_energy_residual_scale_is_energy_density_over_time():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=800)
    scales = residual_scales(case)
    expected_density = case.p_scale / (case.gamma_air - 1.0) + (
        0.5 * case.rho_scale * case.u_scale ** 2
    )
    assert scales["energy_density"] == pytest.approx(expected_density)
    assert scales["energy_residual"] == pytest.approx(
        expected_density / (case.length_scale / case.u_scale)
    )
    assert scales["energy_residual"] != pytest.approx(
        case.e_scale / (case.length_scale / case.u_scale)
    )


# ---------------------------------------------------------------------------
# 5. cache/split guards
# ---------------------------------------------------------------------------
def test_old_cache_versions_are_refused_not_silently_reinterpreted(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, metadata=np.asarray(json.dumps({
        "format": "spherical_blast_pinn.contact_air_case", "version": 6,
        "case": {}, "precursor": {},
    })))
    with pytest.raises(Exception, match="version 7"):
        load_case(path)


def _smoke_section():
    return load_config(SMOKE).section("contact_air")


def _write_config(tmp_path, section):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"contact_air": section}), encoding="utf-8")
    return path


def test_synthetic_false_and_no_synthetic_reject_synthetic_caches(tmp_path):
    case = synthetic_case_for(0.05, case_id="tnt_50mm", n_frames=7, n_radial=1000)
    cache = save_case(case, tmp_path / "synthetic.npz")
    section = _smoke_section()
    spec = dict(section["cases"][0])
    spec["cache"] = str(cache)
    spec["synthetic"] = False
    config = load_config(_write_config(tmp_path, section))
    with pytest.raises(cli.CommandError, match="synthetic"):
        cli._make_case(section, spec, config, allow_synthetic=True)


def test_changed_material_parameters_refuse_an_old_cache(tmp_path):
    case = synthetic_case_for(0.05, case_id="tnt_50mm", n_frames=7, n_radial=1000)
    cache = save_case(case, tmp_path / "cache.npz")
    section = _smoke_section()
    spec = dict(section["cases"][0])
    spec["cache"] = str(cache)
    spec["synthetic"] = True
    changed = json.loads(json.dumps(section))
    changed["precursor"]["air"]["gamma"] = 1.5
    config = load_config(_write_config(tmp_path, changed))
    with pytest.raises(cli.CommandError, match="cached precursor"):
        cli._make_case(changed, spec, config, allow_synthetic=True)


def test_synthetic_flag_is_visible_in_the_case_source(tmp_path):
    case = synthetic_case_for(0.05, case_id="s", n_frames=5, n_radial=800)
    assert case.source["synthetic"] is True


# ---------------------------------------------------------------------------
# 6. validation and checkpoints
# ---------------------------------------------------------------------------
def test_validation_unavailable_is_reported_not_scored_as_zero(tmp_path):
    trainer = _trainer(tmp_path, fraction=0.0)
    verdict = trainer.evaluate()
    assert verdict["evaluable"] is False
    assert verdict["score"] is None


def test_resume_refuses_a_different_air_architecture_before_loading_anything(tmp_path):
    trainer = _trainer(tmp_path)
    path = trainer.save(tmp_path / "last.pt")
    wider = _trainer(tmp_path / "wider", air_width=9)
    with pytest.raises(TrainingError, match="architecture|keys"):
        wider.load(path)


def test_resume_refuses_changed_data(tmp_path):
    trainer = _trainer(tmp_path)
    path = trainer.save(tmp_path / "last.pt")
    other = synthetic_case_for(0.05, case_id="r050", n_frames=13, n_radial=1600)
    fresh = _trainer(tmp_path / "fresh", cases=[other])
    with pytest.raises(TrainingError, match="signature|case ids|split"):
        fresh.load(path)


def test_resume_restores_the_fixed_validation_points_and_rng(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()
    path = trainer.save(tmp_path / "last.pt")
    saved_batch = trainer._batch_payload(trainer._validation[0].batch)
    fresh = _trainer(tmp_path / "fresh")
    fresh.load(path)
    restored = fresh._batch_payload(fresh._validation[0].batch)
    for key in saved_batch:
        if key == "case_id":
            assert saved_batch[key] == restored[key]
            continue
        assert np.array_equal(saved_batch[key], restored[key]), key


def test_best_score_zero_survives_checkpoint_round_trip(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.step()
    trainer.history.best_score = 0.0
    trainer.history.best_loop = 1
    path = trainer.save(tmp_path / "last.pt")
    fresh = _trainer(tmp_path / "fresh")
    fresh.load(path)
    assert fresh.history.best_score == 0.0
    assert fresh.history.best_loop == 1


# ---------------------------------------------------------------------------
# 7. prediction compatibility
# ---------------------------------------------------------------------------
def test_prediction_refuses_checkpoint_with_different_shared_physics(tmp_path):
    case = synthetic_case_for(0.05, case_id="r050", n_frames=7, n_radial=1000)
    trainer = _trainer(tmp_path, cases=[case], fraction=0.0)
    payload = torch.load(trainer.save(tmp_path / "last.pt"), weights_only=False)
    prediction = case.prediction_view()
    prediction.precursor = dict(prediction.precursor)
    prediction.precursor["u_c_sep"] *= 1.01
    contact, air = ContactNet(width=8, depth=1), AirNet(width=8, depth=1)
    with pytest.raises(cli.CommandError, match="dimensionless|physics"):
        cli._check_prediction_compatibility(prediction, payload, contact, air)


def test_prediction_compatibility_is_radius_independent(tmp_path):
    """A different target radius with the same shared physics is allowed."""
    case = synthetic_case_for(0.05, case_id="r050", n_frames=7, n_radial=1000)
    trainer = _trainer(tmp_path, cases=[case], fraction=0.0)
    payload = torch.load(trainer.save(tmp_path / "last.pt"), weights_only=False)
    other = synthetic_case_for(0.08, case_id="r080", role="test",
                               n_frames=7, n_radial=1000)
    prediction = other.prediction_view()
    contact, air = ContactNet(width=8, depth=1), AirNet(width=8, depth=1)
    # Must not raise: R_sep/t0/Rc0/Rs0 differ with radius, shared physics does not.
    cli._check_prediction_compatibility(prediction, payload, contact, air)


def test_prediction_refuses_incompatible_network_convention(tmp_path):
    case = synthetic_case_for(0.05, case_id="r050", n_frames=7, n_radial=1000)
    trainer = _trainer(tmp_path, cases=[case], fraction=0.0)
    payload = torch.load(trainer.save(tmp_path / "last.pt"), weights_only=False)
    prediction = case.prediction_view()
    contact, air = ContactNet(width=9, depth=1), AirNet(width=8, depth=1)
    with pytest.raises(cli.CommandError, match="convention"):
        cli._check_prediction_compatibility(prediction, payload, contact, air)


def test_right_boundary_sampling_excludes_plateau_and_subsonic_times():
    import dataclasses
    from spherical_blast_pinn.contact_air.geometry import ShockTrajectory
    from spherical_blast_pinn.contact_air.sampling import sample_case_batch

    case = synthetic_case_for(0.05, n_frames=9, n_radial=800)
    times = np.asarray(case.shock.times)
    mid = times.size // 2
    radii = np.empty(times.size)
    radii[:mid] = np.linspace(
        case.initial.Rs_data_at_t0, case.initial.Rs_data_at_t0 + 0.01, mid
    )
    radii[mid:] = radii[mid - 1]
    shock = ShockTrajectory.from_arrays(case_id=case.case_id, times=times, radii=radii)
    frozen = dataclasses.replace(case, shock=shock)
    c_inf = np.sqrt(case.gamma_air * case.ambient_pressure / case.ambient_density)
    batch = sample_case_batch(
        frozen, n_pde=0, n_ic=0, n_right=64, n_data=0, n_contact=0,
        generator=torch.Generator().manual_seed(3),
    )
    assert batch.t_right.numel() > 0
    speed = np.asarray(frozen.shock.derivative(batch.t_right.numpy()))
    assert np.all(speed > c_inf)


def test_precheck_all_reports_an_empty_split_instead_of_failing(tmp_path):
    output = tmp_path / "precheck.json"
    code = cli.main([
        "precheck", "--config", str(ROOT / "configs" / "contact_air" / "base.yaml"),
        "--all", "--output", str(output),
    ])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "no_cases_configured"
    assert payload["rows"] == []
    assert payload["missing"]


# ---------------------------------------------------------------------------
# 8. data-entry regressions# ---------------------------------------------------------------------------
# 8. data-entry regressions
# ---------------------------------------------------------------------------
def test_air_observations_keep_rho_and_u_when_pressure_is_invalid(tmp_path):
    from spherical_blast_pinn.contact_air.data import build_case
    from spherical_blast_pinn.contact_air.data import ContactLabelConfig
    from spherical_blast_pinn.contact_air.one_d import ShockDetectionConfig
    from spherical_blast_pinn.contact_air.testing import (
        DEFAULT_TOLERANCES,
        default_precursor,
    )
    from spherical_blast_pinn.contact_air.data import synthetic_radial_case
    precursor = default_precursor(0.05)
    source = synthetic_radial_case(
        case_id="mask", charge_radius_m=0.05, precursor=precursor,
        n_frames=7, n_radial=1200,
    )
    first = build_case(
        source, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
        contact_config=ContactLabelConfig(),
        shock_config=ShockDetectionConfig(relative_prominence=4.0,
                                          min_pressure_jump_fraction=1e-2),
    )
    observed = False
    for index, frame in enumerate(source.frames):
        try:
            up = float(first.contact.up[index])
            rs = float(first.shock.position(float(first.contact.times[index])))
        except Exception:
            continue
        target = up + 0.3 * (rs - up)
        cell = int(np.argmin(np.abs(frame.radius - target)))
        frame.pressure[cell] = np.nan
        frame.valid_pressure[cell] = False
        observed = True
    assert observed
    case = build_case(
        source, precursor=precursor, role="train", tolerances=DEFAULT_TOLERANCES,
        contact_config=ContactLabelConfig(),
        shock_config=ShockDetectionConfig(relative_prominence=4.0,
                                          min_pressure_jump_fraction=1e-2),
    )
    assert np.any(~case.observations.valid_pressure & case.observations.valid_density)
    assert np.any(~case.observations.valid_pressure & case.observations.valid_velocity)


def test_air_data_sampling_is_frame_first_then_point():
    from spherical_blast_pinn.contact_air.data import AirObservationStore

    # Frame 0 has 100 rows, frame 1 has two.  A uniform row draw would select
    # frame 1 only ~2% of the time; frame-first selection gives it ~50%.
    n0, n1 = 100, 2
    store = AirObservationStore(
        time=np.zeros(n0 + n1), radius=np.ones(n0 + n1),
        density=np.concatenate([np.zeros(n0), np.ones(n1)]),
        velocity=np.zeros(n0 + n1), pressure=np.ones(n0 + n1),
        valid_density=np.ones(n0 + n1, bool), valid_velocity=np.ones(n0 + n1, bool),
        valid_pressure=np.ones(n0 + n1, bool),
        frame_index=np.concatenate([np.zeros(n0, np.int64), np.ones(n1, np.int64)]),
    )
    generator = torch.Generator().manual_seed(5)
    draw = store.sample(400, generator)
    fraction_frame_one = float((draw["rho"] > 0.5).float().mean())
    assert draw["count"] == 400
    assert 0.30 < fraction_frame_one < 0.70


def test_initial_pressure_difference_is_reported_by_its_actual_size():
    case = synthetic_case_for(0.05, n_frames=5, n_radial=800)
    report = case.initial_compatibility()
    expected = abs(case.p_c_at_t0 - case.p_scale) / case.p_scale
    assert report["relative_difference"] == pytest.approx(expected)
    assert report["forced_equal"] is False
    assert "R_s_theory_minus_data_m" in report


def test_clock_and_material_provenance_are_marked_unverified():
    from spherical_blast_pinn.contact_air.testing import default_precursor

    result = default_precursor(0.05)
    assert result.clock["time_formula_verified"] is False
    assert result.clock["d3plot_clock_verified"] is False
    assert result.provenance["material_source"]["verified"] is False
