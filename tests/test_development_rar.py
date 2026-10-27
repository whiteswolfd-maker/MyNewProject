"""Continuous-frame development split and RAR correctness.

These tests do not run an optimizer update: they cover assembly, sampling,
fixed development batches, RAR scoring/selection, checkpoint state and failure
paths only.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spherical_blast_pinn.cli._common import build_context
from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.data.startup_supervision import (
    SupervisionError,
    sample_supervision,
    synthetic_dataset,
)
from spherical_blast_pinn.evaluation.checks import build_check_set, run_checks
from spherical_blast_pinn.sampling.rar import (
    RARConfig,
    RARError,
    RARPointPool,
    maybe_update_rar,
)
from spherical_blast_pinn.training.checkpoint import TrainingError
from spherical_blast_pinn.training.startup_cases import CaseSetupError
from spherical_blast_pinn.training.trainer import Trainer

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "startup_diagnostic.yaml"


def _config(
    tmp_path,
    *,
    cases=None,
    rar=None,
    fraction=0.0,
    group_frames=3,
    split_seed=7,
    data_counts=None,
):
    config = load_config(CONFIG)
    config = config.set("startup.handoff_dir", str(tmp_path / "handoffs"))
    config = config.set("training.output_dir", str(tmp_path / "run"))
    if cases is not None:
        config = config.set("startup.cases", cases)
    config = config.set("data.development_holdout_fraction", fraction)
    config = config.set("data.development_group_frames", group_frames)
    config = config.set("data.split_seed", split_seed)
    if rar is not None:
        config = config.set("training.rar", rar)
    if data_counts is not None:
        for key, value in data_counts.items():
            config = config.set(f"sampling.{key}", value)
    return config


def _case(case_id="diag_r050", radius=0.05, data=None):
    spec = {"case_id": case_id, "charge_radius_m": radius, "split": "train"}
    if data is not None:
        spec["data"] = data
    return spec


def _base_context(tmp_path, *, rar=None, cases=None, fraction=0.0):
    config = _config(
        tmp_path,
        cases=cases or [_case()],
        rar=rar,
        fraction=fraction,
    )
    return config, build_context(config, seed=3, diagnostic=True)


def _dataset(context, *, n_frames=12, radius_count=40):
    case = context.training_cases[0]
    eps = float(case.field.epsilon)
    radius = np.linspace(0.0005, 0.075, radius_count)
    shape = (n_frames, radius.size)
    medium = np.full(shape, -1, dtype=np.int8)
    medium[:, radius < 0.05] = 0
    medium[:, (radius > 0.05) & (radius < 0.06)] = 1
    fields = {
        "density": np.full(shape, 1.225),
        "velocity": np.zeros(shape),
        "specific_energy": np.full(shape, 2.0e5),
        "pressure": np.full(shape, 101320.0),
    }
    return synthetic_dataset(
        case_id=case.case_id,
        charge_radius_m=0.05,
        times_dyna=eps + np.arange(1, n_frames + 1) * 1.0e-7,
        radius=radius,
        origin_time_s=0.0,
        fields=fields,
        medium=medium,
        contact_radius=np.full(n_frames, 0.05),
        contact_valid=np.ones(n_frames, dtype=bool),
        shock_radius=np.full(n_frames, 0.06),
        shock_valid=np.ones(n_frames, dtype=bool),
    )


def _with_data_context(tmp_path, dataset, *, fraction=0.25, group_frames=2, split_seed=7, rar=None):
    path = dataset.save(tmp_path / "data.npz")
    config = _config(
        tmp_path,
        cases=[_case(data={"path": str(path), "synthetic": True})],
        rar=rar,
        fraction=fraction,
        group_frames=group_frames,
        split_seed=split_seed,
        data_counts={
            "n_data_product": 8,
            "n_data_air": 8,
            "n_data_contact": 4,
            "n_data_shock": 4,
        },
    )
    return config, build_context(config, seed=3, diagnostic=True)


def _tensor_equal(left, right):
    names = left.tensor_fields()
    return names == right.tensor_fields() and all(
        torch.equal(getattr(left, name), getattr(right, name)) for name in names
    )


_BASE_S = 1001.0
_BASE_COORDINATE = 0.987654321


def _install_pool(sampler, medium, *, eligible=4, future=0):
    window_s = float(sampler.scales.network_time(sampler.window_pinn_time_max))
    count = eligible + future
    s = np.concatenate(
        [
            np.linspace(0.0, window_s, eligible),
            np.full(future, window_s + 1.0),
        ]
    )
    coordinate = np.linspace(0.05, 0.95, count)
    pool = RARPointPool(medium=medium, capacity=count)
    pool.replace(s, coordinate, np.arange(count, dtype=np.float64))
    sampler.rar_pools[medium] = pool


def _install_base_sentinels(sampler):
    calls = []

    def base_time(n):
        calls.append(("s", int(n)))
        return np.full(n, _BASE_S, dtype=np.float64)

    def base_coordinate(n):
        calls.append(("coordinate", int(n)))
        return np.full(n, _BASE_COORDINATE, dtype=np.float64)

    sampler.sample_network_time = base_time
    sampler._stratified_coordinate = base_coordinate
    return calls


def _base_point_mask(s, coordinate):
    return np.isclose(s, _BASE_S, rtol=0.0, atol=0.0) & np.isclose(
        coordinate, _BASE_COORDINATE, rtol=0.0, atol=0.0
    )


def test_frame_split_is_contiguous_groupwise_and_reproducible(tmp_path):
    _, context = _base_context(tmp_path)
    dataset = _dataset(context)
    first = dataset.split_frames(group_frames=3, holdout_fraction=0.25, seed=7)
    second = dataset.split_frames(group_frames=3, holdout_fraction=0.25, seed=7)
    assert first.as_dict() == second.as_dict()
    assert [frame for group in first.groups for frame in group] == list(first.frames)
    assert set(first.train_frames).isdisjoint(first.development_frames)
    assert len(first.development_groups) == 1
    assert first.development_groups[0] > 0
    assert first.n_development_frames == 3
    assert first.actual_holdout_fraction == pytest.approx(0.25)
    assert set(first.train_frames) | set(first.development_frames) == set(first.frames)


def test_training_supervision_does_not_read_development_labels(tmp_path):
    _, context = _base_context(tmp_path)
    dataset = _dataset(context)
    dataset.assign_frame_split(group_frames=2, holdout_fraction=0.25, seed=7)
    changed = copy.deepcopy(dataset)
    dev = list(dataset.frame_split.development_frames)
    changed.density[dev, :] = 999.0
    changed.specific_energy[dev, :] = 9.0e8
    kwargs = dict(
        scales=context.training_cases[0].scales,
        field_epsilon=float(context.training_cases[0].field.epsilon),
        window_tau_max=1.0e-6,
        n_product=8,
        n_air=8,
        n_contact=4,
        n_shock=4,
        dtype=torch.float64,
        frame_mask=dataset.training_frame_mask(),
    )
    left = sample_supervision(dataset, rng=np.random.default_rng(11), **kwargs)
    right = sample_supervision(changed, rng=np.random.default_rng(11), **kwargs)
    assert left.counts == right.counts
    assert left.fields.keys() == right.fields.keys()
    for key in left.fields:
        if isinstance(left.fields[key], torch.Tensor):
            assert torch.equal(left.fields[key], right.fields[key]), key
        else:
            assert left.fields[key] == right.fields[key], key


def test_zero_fraction_and_missing_data_modes(tmp_path):
    _, context = _base_context(tmp_path, fraction=0.0)
    case = context.training_cases[0]
    assert case.data is None
    assert context.config.get("data.development_holdout_fraction") == 0.0
    batch = case.sampler.batch()
    assert batch.diagnostics["data_counts"] == {
        "product": 0, "air": 0, "contact": 0, "shock": 0
    }

    config = _config(tmp_path, cases=[_case()], fraction=0.2)
    with pytest.raises(CaseSetupError, match="no supervision data"):
        build_context(config, seed=3, diagnostic=True)


def test_insufficient_frame_groups_are_rejected(tmp_path):
    _, context = _base_context(tmp_path)
    dataset = _dataset(context, n_frames=2)
    with pytest.raises(SupervisionError, match="two frame groups"):
        dataset.assign_frame_split(group_frames=5, holdout_fraction=0.2, seed=1)


def test_development_check_set_is_fixed_and_uses_only_development_frames(tmp_path):
    _, base = _base_context(tmp_path)
    dataset = _dataset(base)
    _, context = _with_data_context(tmp_path, dataset)
    case = context.training_cases[0]
    split = case.data.frame_split
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=3,
        output_dir=tmp_path / "run",
        check_every=1,
        checkpoint_every=0,
        config=context.config,
        diagnostic=True,
    )
    check_set = trainer._ensure_check_sets()[case.case_id]
    assert len(check_set.data_batches) == 1
    dev_batch = check_set.data_batches[0]
    dev_tau = (
        float(dev_batch.s_data_product.max())
        if dev_batch.s_data_product is not None
        else -1.0
    )
    report = run_checks(context.model, check_set, with_conservation=False)
    assert np.isfinite(report.metrics["development_prediction_error"])
    assert check_set.definition["development_data"]["split"]["development_frames"] == list(
        split.development_frames
    )
    # The physics batch itself carries no data labels.
    assert dev_tau >= 0.0
    assert check_set.batches[0].s_data_product is None


def test_development_batch_can_be_empty_without_faking_a_metric(tmp_path):
    _, base = _base_context(tmp_path)
    dataset = _dataset(base)
    _, context = _with_data_context(tmp_path, dataset)
    case = context.training_cases[0]
    check_set = build_check_set(
        field=case.field,
        scales=case.scales,
        tau_end=1.0e-9,
        config=case.sampler.config,
        seed=1,
        development_case=case.data,
        development_config=case.sampler.config,
    )
    assert len(check_set.data_batches) == 1
    report = run_checks(context.model, check_set, with_conservation=False)
    assert np.isnan(report.metrics["development_prediction_error"])


def test_rar_selects_high_scores_and_pool_is_bounded(tmp_path, monkeypatch):
    config, context = _base_context(
        tmp_path,
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 16,
            "n_candidates_air": 16,
            "n_select_product": 4,
            "n_select_air": 4,
            "pool_capacity_product": 6,
            "pool_capacity_air": 6,
            "pool_fraction": 0.5,
        },
    )
    case = context.training_cases[0]
    captured = []

    def fake_score(sampler, *, medium, s, coordinate, model):
        captured.append((medium, np.asarray(s), np.asarray(coordinate)))
        base = 1.0 if medium == "product" else 2.0
        return base + np.arange(s.size, dtype=np.float64)

    monkeypatch.setattr("spherical_blast_pinn.sampling.rar._score_candidates", fake_score)
    summary = case.sampler.maybe_update_rar(context.model, 1)
    assert summary["media"]["product"]["selected"] == 4
    assert summary["media"]["air"]["selected"] == 4
    for medium in ("product", "air"):
        pool = case.sampler.rar_pools[medium]
        assert 0 < len(pool) <= 6
        assert np.all(np.diff(pool.score) <= 0.0)
        assert np.all(pool.coordinate > 1.0e-9)
        assert np.all(pool.coordinate < 1.0 - 1.0e-9)
        assert np.all(pool.s >= 0.0)
    window_s = float(
        case.sampler.scales.network_time(case.sampler.window_pinn_time_max)
    )
    for medium, s_values, coordinate in captured:
        assert np.all(s_values <= window_s + 1.0e-15)
        assert np.all(coordinate > 1.0e-9)
        assert np.all(coordinate < 1.0 - 1.0e-9)
    case.sampler.maybe_update_rar(context.model, 1)
    assert len(case.sampler.rar_pools["product"]) <= 6
    batch = case.sampler.batch()
    for medium, s_field, c_field in (
        ("product", batch.s_product, batch.xi_product),
        ("air", batch.s_air, batch.eta_air),
    ):
        pool = case.sampler.rar_pools[medium]
        s_values = s_field.detach().cpu().numpy().reshape(-1)
        c_values = c_field.detach().cpu().numpy().reshape(-1)
        for s_value, c_value in zip(pool.s, pool.coordinate):
            assert np.any(
                np.isclose(s_values, s_value, rtol=0.0, atol=0.0)
                & np.isclose(c_values, c_value, rtol=0.0, atol=0.0)
            ), (medium, s_value, c_value)


def test_rar_keeps_independent_pools_for_two_cases(tmp_path, monkeypatch):
    _, context = _base_context(
        tmp_path,
        cases=[_case("diag_r050", 0.05), _case("diag_r080", 0.08)],
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 8,
            "n_candidates_air": 8,
            "n_select_product": 2,
            "n_select_air": 2,
            "pool_capacity_product": 4,
            "pool_capacity_air": 4,
            "pool_fraction": 0.5,
        },
    )
    monkeypatch.setattr(
        "spherical_blast_pinn.sampling.rar._score_candidates",
        lambda sampler, *, medium, s, coordinate, model: np.ones(s.size),
    )
    for case in context.training_cases:
        case.sampler.maybe_update_rar(context.model, 1)
        assert len(case.sampler.rar_pools["product"]) == 2
        assert len(case.sampler.rar_pools["air"]) == 2
        assert case.sampler._rar_last_update_step == 1
    # The two cases have independent arrays (their own scales/time windows).
    first, second = context.training_cases
    assert not np.array_equal(
        first.sampler.rar_pools["product"].s,
        second.sampler.rar_pools["product"].s,
    )


def test_rar_real_scoring_leaves_parameters_grads_and_mode_unchanged(tmp_path):
    _, context = _base_context(
        tmp_path,
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 8,
            "n_candidates_air": 8,
            "n_select_product": 2,
            "n_select_air": 2,
            "pool_capacity_product": 4,
            "pool_capacity_air": 4,
            "pool_fraction": 0.5,
        },
    )
    case = context.training_cases[0]
    parameters = [p for p in context.model.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 3.0)
    before_params = [p.detach().clone() for p in parameters]
    before_grads = [p.grad.detach().clone() for p in parameters]
    before_mode = context.model.training
    case.sampler.maybe_update_rar(context.model, 1)
    assert context.model.training == before_mode
    for parameter, value, grad in zip(parameters, before_params, before_grads):
        assert torch.equal(parameter.detach(), value)
        assert torch.equal(parameter.grad, grad)
    assert len(case.sampler.rar_pools["product"]) > 0
    assert len(case.sampler.rar_pools["air"]) > 0


def test_rar_nonfinite_residual_is_an_error(tmp_path, monkeypatch):
    _, context = _base_context(
        tmp_path,
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 4,
            "n_candidates_air": 4,
            "n_select_product": 1,
            "n_select_air": 1,
            "pool_capacity_product": 2,
            "pool_capacity_air": 2,
            "pool_fraction": 0.5,
        },
    )
    monkeypatch.setattr(
        "spherical_blast_pinn.sampling.rar._score_candidates",
        lambda sampler, *, medium, s, coordinate, model: np.full(s.size, np.nan),
    )
    with pytest.raises(RARError, match="non-finite"):
        context.training_cases[0].sampler.maybe_update_rar(context.model, 1)


def test_rar_disabled_preserves_the_original_batch(tmp_path):
    _, disabled = _base_context(tmp_path / "a")
    _, enabled = _base_context(
        tmp_path / "b",
        rar={"enabled": True, "interval": 10},
    )
    left = disabled.training_cases[0].sampler.batch()
    right = enabled.training_cases[0].sampler.batch()
    assert _tensor_equal(left, right)


@pytest.mark.parametrize(
    "fraction", [1.0, 1, 0.0, -0.1, np.inf, np.nan, True, "0.5"]
)
def test_rar_pool_fraction_must_be_finite_and_strictly_between_zero_and_one(
    fraction,
):
    with pytest.raises(RARError, match="pool_fraction"):
        RARConfig(pool_fraction=fraction).validate()
    with pytest.raises(RARError, match="pool_fraction"):
        RARConfig.from_mapping({"pool_fraction": fraction})


@pytest.mark.parametrize(
    "count,fraction,expected_pool",
    [(4, 0.9, 3), (4, 0.999, 3), (2, 0.9, 1), (1, 0.75, 0)],
)
def test_rar_keeps_at_least_one_base_point(
    tmp_path, count, fraction, expected_pool
):
    _, context = _base_context(
        tmp_path, rar={"enabled": True, "pool_fraction": fraction}
    )
    sampler = context.training_cases[0].sampler
    for medium in ("product", "air"):
        _install_pool(sampler, medium)
        calls = _install_base_sentinels(sampler)
        s, coordinate = sampler._pde_points(medium, count)
        base = _base_point_mask(s, coordinate)
        expected_base = count - expected_pool
        assert s.size == count
        assert int(base.sum()) == expected_base
        assert expected_base >= 1
        assert calls == [("s", expected_base), ("coordinate", expected_base)]
        assert int((~base).sum()) == expected_pool


@pytest.mark.parametrize(
    "eligible,future,expected_pool,expected_base",
    [(0, 0, 0, 4), (1, 0, 1, 3), (1, 3, 1, 3), (2, 0, 2, 2)],
)
def test_rar_short_pool_is_filled_by_base_sampling(
    tmp_path, eligible, future, expected_pool, expected_base
):
    _, context = _base_context(
        tmp_path, rar={"enabled": True, "pool_fraction": 0.5}
    )
    sampler = context.training_cases[0].sampler
    for medium in ("product", "air"):
        if eligible or future:
            _install_pool(sampler, medium, eligible=eligible, future=future)
        calls = _install_base_sentinels(sampler)
        s, coordinate = sampler._pde_points(medium, 4)
        base = _base_point_mask(s, coordinate)
        assert s.size == 4
        assert int(base.sum()) == expected_base
        assert int((~base).sum()) == expected_pool
        assert calls == [("s", expected_base), ("coordinate", expected_base)]


def test_rar_zero_count_returns_empty_without_sampling(tmp_path):
    _, context = _base_context(
        tmp_path, rar={"enabled": True, "pool_fraction": 0.5}
    )
    sampler = context.training_cases[0].sampler
    for medium in ("product", "air"):
        _install_pool(sampler, medium)
        calls = _install_base_sentinels(sampler)
        s, coordinate = sampler._pde_points(medium, 0)
        assert s.size == coordinate.size == 0
        assert calls == []


def test_rar_disabled_ignores_populated_pool_and_uses_base_sampling(tmp_path):
    _, context = _base_context(tmp_path)
    sampler = context.training_cases[0].sampler
    for medium in ("product", "air"):
        _install_pool(sampler, medium)
        calls = _install_base_sentinels(sampler)
        s, coordinate = sampler._pde_points(medium, 4)
        assert s.size == coordinate.size == 4
        assert _base_point_mask(s, coordinate).all()
        assert calls == [("s", 4), ("coordinate", 4)]


def test_rar_checkpoint_restores_pool_split_window_and_next_batch(tmp_path, monkeypatch):
    _, base = _base_context(tmp_path)
    dataset = _dataset(base)
    _, context = _with_data_context(
        tmp_path, dataset,
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 8,
            "n_candidates_air": 8,
            "n_select_product": 2,
            "n_select_air": 2,
            "pool_capacity_product": 4,
            "pool_capacity_air": 4,
            "pool_fraction": 0.5,
        },
    )
    monkeypatch.setattr(
        "spherical_blast_pinn.sampling.rar._score_candidates",
        lambda sampler, *, medium, s, coordinate, model: np.ones(s.size),
    )
    case = context.training_cases[0]
    case.sampler.maybe_update_rar(context.model, 1)
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=3,
        output_dir=tmp_path / "run",
        check_every=0,
        checkpoint_every=0,
        config=context.config,
        diagnostic=True,
    )
    checkpoint = trainer.save(tmp_path / "state.pt")

    _, fresh_context = _with_data_context(
        tmp_path, dataset,
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 8,
            "n_candidates_air": 8,
            "n_select_product": 2,
            "n_select_air": 2,
            "pool_capacity_product": 4,
            "pool_capacity_air": 4,
            "pool_fraction": 0.5,
        },
    )
    fresh = Trainer(
        cases=fresh_context.training_cases,
        model=fresh_context.model,
        seed=3,
        output_dir=tmp_path / "run2",
        check_every=0,
        checkpoint_every=0,
        config=fresh_context.config,
        diagnostic=True,
    )
    info = fresh.load_checkpoint(checkpoint)
    assert info["step"] == 0
    saved = case.sampler.rar_state()
    restored = fresh.cases[0].sampler.rar_state()
    for medium in ("product", "air"):
        for key in ("s", "coordinate", "score"):
            assert np.array_equal(saved["pools"][medium][key], restored["pools"][medium][key])
    assert saved["last_update_step"] == restored["last_update_step"] == 1
    assert (
        case.data.identity()["development_split"]
        == fresh.cases[0].data.identity()["development_split"]
    )
    assert case.sampler.window_state() == fresh.cases[0].sampler.window_state()
    assert case.sampler.rng_state() == fresh.cases[0].sampler.rng_state()
    assert _tensor_equal(case.sampler.batch(), fresh.cases[0].sampler.batch())
    # Restoring at the same step must not perform the trigger a second time.
    before = fresh.cases[0].sampler.rar_state()
    assert fresh.cases[0].sampler.maybe_update_rar(fresh.model, 1) is None
    after = fresh.cases[0].sampler.rar_state()
    for medium in ("product", "air"):
        for key in ("s", "coordinate", "score"):
            assert np.array_equal(before["pools"][medium][key], after["pools"][medium][key])


def test_checkpoint_refuses_changed_development_split_or_labels(tmp_path):
    _, base = _base_context(tmp_path)
    dataset = _dataset(base)
    dataset.assign_frame_split(group_frames=2, holdout_fraction=0.25, seed=7)
    config, context = _with_data_context(tmp_path, dataset, split_seed=7)
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=3,
        output_dir=tmp_path / "run",
        check_every=0,
        checkpoint_every=0,
        config=config,
        diagnostic=True,
    )
    checkpoint = trainer.save(tmp_path / "split.pt")

    _, changed_split_context = _with_data_context(
        tmp_path / "split_changed", dataset, split_seed=8, fraction=0.5
    )
    changed_split = Trainer(
        cases=changed_split_context.training_cases,
        model=changed_split_context.model,
        seed=3,
        output_dir=tmp_path / "run_changed",
        check_every=0,
        checkpoint_every=0,
        config=changed_split_context.config,
        diagnostic=True,
    )
    before = {
        name: value.clone()
        for name, value in changed_split.model.state_dict().items()
    }
    with pytest.raises(TrainingError, match="supervision dataset|data"):
        changed_split.load_checkpoint(checkpoint)
    for name, value in changed_split.model.state_dict().items():
        assert torch.equal(value, before[name])

    changed_labels = copy.deepcopy(dataset)
    dev = list(dataset.frame_split.development_frames)
    changed_labels.density[dev, :] += 1.0
    _, changed_data_context = _with_data_context(
        tmp_path / "labels_changed", changed_labels
    )
    changed_data = Trainer(
        cases=changed_data_context.training_cases,
        model=changed_data_context.model,
        seed=3,
        output_dir=tmp_path / "run_labels",
        check_every=0,
        checkpoint_every=0,
        config=changed_data_context.config,
        diagnostic=True,
    )
    with pytest.raises(TrainingError, match="supervision dataset|data"):
        changed_data.load_checkpoint(checkpoint)


def test_init_from_loads_weights_without_restoring_rar_state(tmp_path, monkeypatch):
    _, context = _base_context(
        tmp_path,
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 4,
            "n_candidates_air": 4,
            "n_select_product": 1,
            "n_select_air": 1,
            "pool_capacity_product": 2,
            "pool_capacity_air": 2,
            "pool_fraction": 0.5,
        },
    )
    monkeypatch.setattr(
        "spherical_blast_pinn.sampling.rar._score_candidates",
        lambda sampler, *, medium, s, coordinate, model: np.ones(s.size),
    )
    case = context.training_cases[0]
    case.sampler.maybe_update_rar(context.model, 1)
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=3,
        output_dir=tmp_path / "run",
        check_every=0,
        checkpoint_every=0,
        config=context.config,
        diagnostic=True,
    )
    checkpoint = trainer.save(tmp_path / "state.pt")

    _, fresh_context = _base_context(
        tmp_path / "fresh",
        rar={
            "enabled": True,
            "interval": 1,
            "n_candidates_product": 4,
            "n_candidates_air": 4,
            "n_select_product": 1,
            "n_select_air": 1,
            "pool_capacity_product": 2,
            "pool_capacity_air": 2,
            "pool_fraction": 0.5,
        },
    )
    fresh = Trainer(
        cases=fresh_context.training_cases,
        model=fresh_context.model,
        seed=3,
        output_dir=tmp_path / "fresh_run",
        check_every=0,
        checkpoint_every=0,
        config=fresh_context.config,
        diagnostic=True,
    )
    fresh.load_weights(checkpoint)
    assert len(fresh.cases[0].sampler.rar_pools["product"]) == 0
    assert len(fresh.cases[0].sampler.rar_pools["air"]) == 0
    assert fresh.cases[0].sampler._rar_last_update_step is None
    assert fresh.step == 0


def test_rar_checkpoint_config_mismatch_is_refused_before_mutation(tmp_path):
    _, context = _base_context(
        tmp_path,
        rar={"enabled": True, "interval": 10},
    )
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=3,
        output_dir=tmp_path / "run",
        check_every=0,
        checkpoint_every=0,
        config=context.config,
        diagnostic=True,
    )
    checkpoint = trainer.save(tmp_path / "state.pt")

    _, other = _base_context(
        tmp_path / "other",
        rar={"enabled": True, "interval": 20},
    )
    fresh = Trainer(
        cases=other.training_cases,
        model=other.model,
        seed=3,
        output_dir=tmp_path / "other_run",
        check_every=0,
        checkpoint_every=0,
        config=other.config,
        diagnostic=True,
    )
    before = {name: value.clone() for name, value in fresh.model.state_dict().items()}
    with pytest.raises(TrainingError, match="RAR|sampling"):
        fresh.load_checkpoint(checkpoint)
    for name, value in fresh.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_rar_config_validation_and_pool_dedupe(tmp_path):
    with pytest.raises(RARError):
        RARConfig.from_mapping({"interval": 1.5})
    pool = RARPointPool(medium="product", capacity=3)
    pool.replace(
        [0.1, 0.1, 0.2, 0.3],
        [0.2, 0.2, 0.3, 0.4],
        [1.0, 2.0, 3.0, 4.0],
    )
    assert len(pool) == 3
    assert np.all(np.diff(pool.score) <= 0.0)
    assert pool.score[0] == pytest.approx(4.0)


def test_500mm_blind_radius_is_rejected_before_training(tmp_path):
    config = _config(tmp_path, cases=[_case("blind", radius=0.5)])
    with pytest.raises(CaseSetupError, match="final_validation"):
        build_context(config, seed=3, diagnostic=True)
