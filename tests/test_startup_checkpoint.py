"""New startup checkpoint contract: save/load, identity binding, resume."""

import copy
import dataclasses
import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spherical_blast_pinn.losses.objective import contact_pressure_scale, total_loss
from spherical_blast_pinn.models.air_net import AirNet
from spherical_blast_pinn.models.coupled_model import CoupledModel
from spherical_blast_pinn.models.front_net import FrontNet
from spherical_blast_pinn.models.product_net import ProductNet
from spherical_blast_pinn.physics.similitude import PlanarStartupScales
from spherical_blast_pinn.physics.two_medium_initial import (
    build_two_medium_initial_field,
)
from spherical_blast_pinn.physics.two_medium_startup_handoff import (
    load_planar_startup_handoff,
    write_planar_startup_handoff,
)
from spherical_blast_pinn.physics.two_medium_startup_planar import (
    PlanarStartupConfig,
    build_planar_startup_field,
)
from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig
from spherical_blast_pinn.training import checkpoint as checkpoint_module
from spherical_blast_pinn.training.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_PURPOSE,
    CHECKPOINT_VERSION,
    TrainingError,
    collect_rng_state,
    load_startup_checkpoint,
    planar_scales_from_record,
    save_startup_checkpoint,
    startup_scale_record,
)

R0 = 0.05
TAU_END = 1.0e-6
TIME_SCALE = 2.0


def _field(radius=R0, delta=1.0 / 200.0):
    initial = build_two_medium_initial_field(
        charge_radius=radius, outer_radius=2.0 * radius
    )
    return build_planar_startup_field(
        initial,
        config=PlanarStartupConfig(
            delta=delta, inner_cells=8, fan_cells=16, plateau_cells=2,
            air_cells=4, ambient_cells=8, dense_samples=512,
        ),
    )


def _model(scales, field, *, width=8, seed=0):
    return CoupledModel(
        product=ProductNet(scales=scales, width=width, depth=1, seed=seed),
        air=AirNet(scales=scales, width=width, depth=1, seed=seed + 1),
        front=FrontNet(
            scales=scales,
            width=width,
            depth=1,
            initial_contact_hat=float(field.contact_radius / field.charge_radius),
            initial_gap_hat=float(
                (field.shock_radius - field.contact_radius) / field.charge_radius
            ),
            seed=seed + 2,
        ),
        scales=scales,
    )


def _sampling_config(**overrides):
    values = dict(
        n_pde_product=8, n_pde_air=8, n_ic_product=4, n_ic_air=4,
        n_contact=4, n_shock=4,
    )
    values.update(overrides)
    return SamplingConfig(**values)


def _sampler(
    field, scales, *, seed=0, tau_end=TAU_END, config=None, r_min_fraction=1.0e-2
):
    return RadiusSampler(
        field=field,
        scales=scales,
        tau_end=tau_end,
        config=config or _sampling_config(),
        seed=seed,
        r_min_fraction=r_min_fraction,
    )


def _handoff(tmp_path, field, name="handoff"):
    directory = tmp_path / name
    json_path, _ = write_planar_startup_handoff(field, directory)
    return json_path


def _state_equal(left, right) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def test_scale_record_roundtrips_with_type_origin_and_conversions():
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    record = startup_scale_record(scales)
    rebuilt = planar_scales_from_record(record)
    assert isinstance(rebuilt, PlanarStartupScales)
    assert rebuilt.as_dict() == scales.as_dict()
    assert rebuilt.start_time() == field.epsilon
    for s_value in (0.0, 0.1, 1.0):
        assert float(rebuilt.physical_time_from_network(s_value)) == float(
            scales.physical_time_from_network(s_value)
        )
        assert float(rebuilt.network_time_from_physical(
            scales.physical_time_from_network(s_value)
        )) == pytest.approx(s_value, rel=1.0e-14, abs=1.0e-15)
    for bad in (
        {**record, "detonation_velocity": 0.0},
        {**record, "time_convention": "scheme_b_handover"},
        {**record, "startup_epsilon": -1.0},
    ):
        with pytest.raises(TrainingError):
            planar_scales_from_record(bad)


def test_save_load_restores_model_scales_and_pressure(tmp_path):
    torch.set_default_dtype(torch.float64)
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    handoff = _handoff(tmp_path, field)

    source = _model(scales, field, seed=3)
    optimizer = torch.optim.Adam(source.parameters(), lr=1.0e-3)
    sampler = _sampler(field, scales, seed=7)
    path = save_startup_checkpoint(
        tmp_path / "step3.pt",
        field=field,
        scales=scales,
        model=source,
        optimizer=optimizer,
        step=3,
        sampler=sampler,
    )

    target = _model(scales, field, seed=99)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=5.0e-4)
    target_sampler = _sampler(field, scales, seed=123)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    info = load_startup_checkpoint(
        path,
        handoff_json=handoff,
        model=target,
        optimizer=target_optimizer,
        sampler=target_sampler,
    )
    assert info["handoff_checked"] is True
    assert info["restored_model"] and info["restored_optimizer"] and info["restored_rng"]
    assert info["restored_sampling"] is True
    assert info["step"] == 3
    assert info["contact_pressure_pa"] == field.solution.p_star
    assert isinstance(info["scales"], PlanarStartupScales)
    assert info["scales"].start_time() == field.epsilon
    assert info["sampling"]["tau_end_s"] == TAU_END
    assert info["sampling"]["counts"]["n_pde_product"] == 8
    assert info["sampling"]["window"] == sampler.window_state()

    assert _state_equal(source.state_dict(), target.state_dict())
    assert not _state_equal(before, target.state_dict())
    assert target_sampler.rng_state() == sampler.rng_state()
    assert target_sampler.window_state() == sampler.window_state()

    s = torch.tensor([[0.0], [0.01]], dtype=torch.float64)
    xi = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    eta = torch.tensor([[0.3], [0.8]], dtype=torch.float64)
    for name in ("density", "velocity", "specific_energy"):
        assert torch.equal(
            getattr(source.product_fields(s, xi), name),
            getattr(target.product_fields(s, xi), name),
        ), name
        assert torch.equal(
            getattr(source.air_fields(s, eta), name),
            getattr(target.air_fields(s, eta), name),
        ), name
    assert torch.equal(
        source.front_positions(s).contact_hat, target.front_positions(s).contact_hat
    )


def test_old_checkpoints_are_rejected_without_migration(tmp_path):
    old = tmp_path / "old.pt"
    torch.save(
        {
            "format": "spherical_blast_pinn.checkpoint",
            "version": 8,
            "detonation_velocity": 6930.0,
            "scales": {"case": {"start_time": 8.2e-6}},
            "model": {},
        },
        old,
    )
    with pytest.raises(TrainingError, match="not migrated"):
        load_startup_checkpoint(old)

    wrong_version = tmp_path / "wrong_version.pt"
    torch.save(
        {"format": CHECKPOINT_FORMAT, "version": 99, "purpose": CHECKPOINT_PURPOSE},
        wrong_version,
    )
    with pytest.raises(TrainingError, match="version"):
        load_startup_checkpoint(wrong_version)

    wrong_purpose = tmp_path / "wrong_purpose.pt"
    torch.save(
        {"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION, "purpose": "other"},
        wrong_purpose,
    )
    with pytest.raises(TrainingError, match="purpose"):
        load_startup_checkpoint(wrong_purpose)


def test_a_mismatched_handoff_is_rejected(tmp_path):
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    model = _model(scales, field)
    path = save_startup_checkpoint(
        tmp_path / "a.pt", field=field, scales=scales, model=model
    )
    other = _field(radius=0.08)
    other_scales = PlanarStartupScales.for_startup_field(other, time_scale=TIME_SCALE)
    other_handoff = _handoff(tmp_path, other, name="other")
    with pytest.raises(TrainingError, match="cache_key|initial condition"):
        load_startup_checkpoint(
            path,
            handoff_json=other_handoff,
            model=_model(other_scales, other),
        )


def test_wrong_structure_or_scales_are_refused_before_mutation(tmp_path):
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    source = _model(scales, field, seed=1)
    path = save_startup_checkpoint(
        tmp_path / "model.pt", field=field, scales=scales, model=source
    )

    wide = _model(scales, field, width=12, seed=2)
    before = {k: v.clone() for k, v in wide.state_dict().items()}
    with pytest.raises(TrainingError, match="network structure"):
        load_startup_checkpoint(path, model=wide)
    assert _state_equal(before, wide.state_dict())

    normal = _model(scales, field, seed=3)
    before = {k: v.clone() for k, v in normal.state_dict().items()}
    with pytest.raises(TrainingError, match="scales differ|scales"):
        load_startup_checkpoint(
            path,
            scales=dataclasses.replace(scales, time_scale=3.0),
            model=normal,
        )
    assert _state_equal(before, normal.state_dict())


def test_missing_recovery_state_is_refused(tmp_path):
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    model = _model(scales, field)
    path = save_startup_checkpoint(
        tmp_path / "weights.pt", field=field, scales=scales, model=model
    )
    target = _model(scales, field, seed=5)
    optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
    with pytest.raises(TrainingError, match="no optimizer state"):
        load_startup_checkpoint(path, model=target, optimizer=optimizer)

    # 没有采样配置的检查点不能恢复到一个采样器上：点数与 tau_end 都无从核对。
    with pytest.raises(TrainingError, match="no sampling configuration"):
        load_startup_checkpoint(
            path, model=target, sampler=_sampler(field, scales)
        )

    sampler = _sampler(field, scales)
    with_sampler = save_startup_checkpoint(
        tmp_path / "with_sampler.pt",
        field=field,
        scales=scales,
        model=model,
        sampler=sampler,
    )

    payload = torch.load(with_sampler, map_location="cpu", weights_only=False)
    del payload["rng"]
    stripped = tmp_path / "no_rng.pt"
    torch.save(payload, stripped)
    with pytest.raises(TrainingError, match="RNG state"):
        load_startup_checkpoint(stripped, model=target)

    payload = torch.load(with_sampler, map_location="cpu", weights_only=False)
    payload["rng"]["sampler"] = None
    no_sampler = tmp_path / "no_sampler.pt"
    torch.save(payload, no_sampler)
    with pytest.raises(TrainingError, match="no sampler RNG state"):
        load_startup_checkpoint(
            no_sampler, model=target, sampler=_sampler(field, scales)
        )

    payload = torch.load(with_sampler, map_location="cpu", weights_only=False)
    del payload["sampling"]
    old_contract = tmp_path / "no_sampling_field.pt"
    torch.save(payload, old_contract)
    with pytest.raises(TrainingError, match="sampling"):
        load_startup_checkpoint(old_contract, model=target)


def _assert_same_batch(left, right):
    assert left.tensor_fields() == right.tensor_fields()
    for name in left.tensor_fields():
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    assert left.pinn_time_min == right.pinn_time_min
    assert left.pinn_time_max == right.pinn_time_max
    assert left.contact_ic == right.contact_ic
    assert left.shock_ic == right.shock_ic


def test_a_non_default_time_window_restores_the_next_batch(tmp_path):
    torch.set_default_dtype(torch.float64)
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    handoff = _handoff(tmp_path, field)

    source = _model(scales, field, seed=11)
    source_sampler = _sampler(field, scales, seed=21)
    source_sampler.set_time_window(0.4)
    assert source_sampler.window_history == [1.0, 0.4]
    path = save_startup_checkpoint(
        tmp_path / "window.pt",
        field=field,
        scales=scales,
        model=source,
        sampler=source_sampler,
    )

    target = _model(scales, field, seed=77)
    target_sampler = _sampler(field, scales, seed=888)
    assert target_sampler.window_fraction == 1.0
    info = load_startup_checkpoint(
        path,
        handoff_json=handoff,
        model=target,
        sampler=target_sampler,
    )
    assert info["restored_sampling"] is True
    # 恢复必须允许相对接收端的默认窗口**收窄**，否则阶段 1 的窗口无法重建。
    assert target_sampler.window_fraction == pytest.approx(0.4)
    assert target_sampler.window_history == [1.0, 0.4]
    assert target_sampler.window_pinn_time_max == pytest.approx(0.4 * TAU_END)
    assert target_sampler.window_state() == source_sampler.window_state()

    # 窗口、RNG 都与保存时相同，下一批采样必须逐位一致。
    _assert_same_batch(source_sampler.batch(), target_sampler.batch())


def _optimizer_structure(optimizer):
    state = optimizer.state_dict()
    return state["param_groups"], set(state["state"])


def test_a_different_sampling_configuration_is_refused_before_mutation(
    tmp_path,
):
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    source = _model(scales, field, seed=3)
    source_optimizer = torch.optim.Adam(source.parameters(), lr=1.0e-3)
    source_sampler = _sampler(field, scales, seed=7)
    path = save_startup_checkpoint(
        tmp_path / "sampling.pt",
        field=field,
        scales=scales,
        model=source,
        optimizer=source_optimizer,
        sampler=source_sampler,
    )

    variants = [
        ("tau_end", _sampler(field, scales, seed=1, tau_end=0.9e-6), "tau_end"),
        (
            "counts",
            _sampler(
                field, scales, seed=1,
                config=_sampling_config(n_pde_product=9, n_contact=5),
            ),
            "counts",
        ),
        (
            "stratification",
            _sampler(
                field, scales, seed=1,
                config=_sampling_config(near_fraction=0.2, near_span=0.05),
            ),
            "stratification",
        ),
        (
            "r_min_fraction",
            _sampler(field, scales, seed=1, r_min_fraction=2.0e-2),
            "r_min_fraction",
        ),
    ]
    for label, variant, match in variants:
        target = _model(scales, field, seed=99)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
        before_model = {k: v.clone() for k, v in target.state_dict().items()}
        before_optimizer = _optimizer_structure(target_optimizer)
        before_window = variant.window_state()
        before_sampler_rng = variant.rng_state()
        before_global = (
            torch.get_rng_state().clone(),
            np.random.get_state(),
            random.getstate(),
        )

        with pytest.raises(TrainingError, match=match):
            load_startup_checkpoint(
                path,
                model=target,
                optimizer=target_optimizer,
                sampler=variant,
            )

        assert _state_equal(before_model, target.state_dict()), label
        assert _optimizer_structure(target_optimizer) == before_optimizer, label
        assert variant.window_state() == before_window, label
        assert variant.rng_state() == before_sampler_rng, label
        now_global = (
            torch.get_rng_state(),
            np.random.get_state(),
            random.getstate(),
        )
        assert torch.equal(now_global[0], before_global[0]), label
        assert np.array_equal(now_global[1][1], before_global[1][1]), label
        assert now_global[1][0] == before_global[1][0], label
        assert now_global[1][2:] == before_global[1][2:], label
        assert now_global[2] == before_global[2], label


def test_a_sampler_on_a_different_startup_field_is_refused(tmp_path):
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    source = _model(scales, field, seed=3)
    path = save_startup_checkpoint(
        tmp_path / "field.pt",
        field=field,
        scales=scales,
        model=source,
        sampler=_sampler(field, scales, seed=7),
    )

    other = _field(radius=0.08)
    other_scales = PlanarStartupScales.for_startup_field(other, time_scale=TIME_SCALE)
    target = _model(scales, field, seed=99)
    other_sampler = _sampler(other, other_scales, seed=5)
    before_window = other_sampler.window_state()
    before_rng = other_sampler.rng_state()
    with pytest.raises(TrainingError, match="startup|cache_key"):
        load_startup_checkpoint(path, model=target, sampler=other_sampler)
    assert other_sampler.window_state() == before_window
    assert other_sampler.rng_state() == before_rng

    # 写端同样不能把"另一个启动场上的采样器"写进这份检查点。
    with pytest.raises(TrainingError, match="startup|cache_key"):
        save_startup_checkpoint(
            tmp_path / "bad_binding.pt",
            field=field,
            scales=scales,
            model=target,
            sampler=other_sampler,
        )


def test_a_failure_during_restore_rolls_back_the_target(tmp_path, monkeypatch):
    torch.set_default_dtype(torch.float64)
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    source = _model(scales, field, seed=1)
    optimizer = torch.optim.Adam(source.parameters(), lr=1.0e-3)
    source_sampler = _sampler(field, scales, seed=7)
    source_sampler.set_time_window(0.4)
    path = save_startup_checkpoint(
        tmp_path / "rollback.pt",
        field=field,
        scales=scales,
        model=source,
        optimizer=optimizer,
        step=1,
        sampler=source_sampler,
    )

    target = _model(scales, field, seed=42)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
    target_sampler = _sampler(field, scales, seed=123)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())
    before_window = target_sampler.window_state()
    before_sampler_rng = target_sampler.rng_state()
    assert before_window["fraction"] == 1.0

    original_apply = checkpoint_module._apply_rng_state
    calls = {"count": 0}

    def _boom(state, sampler):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected RNG failure")
        return original_apply(state, sampler)

    monkeypatch.setattr(checkpoint_module, "_apply_rng_state", _boom)
    with pytest.raises(RuntimeError, match="injected RNG failure"):
        load_startup_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            sampler=target_sampler,
            restore_rng=True,
        )
    assert _state_equal(before, target.state_dict())
    restored = target_optimizer.state_dict()
    assert restored["param_groups"] == before_optimizer["param_groups"]
    assert set(restored["state"]) == set(before_optimizer["state"])
    # 窗口在 RNG 之前被写入；失败时也必须回滚，不能留下"模型是旧的、窗口是
    # 检查点的"这种半恢复状态。
    assert target_sampler.window_state() == before_window
    assert target_sampler.rng_state() == before_sampler_rng


def _one_step(model, optimizer, sampler, pressure):
    batch = sampler.batch()
    loss = total_loss([batch], model, contact_pressure=pressure)
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    return batch, float(loss.total.detach())


def test_a_short_resume_matches_a_continuous_run(tmp_path):
    torch.set_default_dtype(torch.float64)
    field = _field()
    scales = PlanarStartupScales.for_startup_field(field, time_scale=TIME_SCALE)
    pressure = contact_pressure_scale(field)
    handoff = _handoff(tmp_path, field)

    # Continuous: four deterministic updates.
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    continuous = _model(scales, field, seed=11)
    continuous_optimizer = torch.optim.Adam(continuous.parameters(), lr=1.0e-3)
    continuous_sampler = _sampler(field, scales, seed=21)
    losses_continuous = []
    for _ in range(4):
        _, value = _one_step(continuous, continuous_optimizer, continuous_sampler, pressure)
        losses_continuous.append(value)

    # Interrupted: two updates, save, fresh objects, load, two more.
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    interrupted = _model(scales, field, seed=11)
    interrupted_optimizer = torch.optim.Adam(interrupted.parameters(), lr=1.0e-3)
    interrupted_sampler = _sampler(field, scales, seed=21)
    losses_interrupted = []
    for _ in range(2):
        _, value = _one_step(
            interrupted, interrupted_optimizer, interrupted_sampler, pressure
        )
        losses_interrupted.append(value)
    path = save_startup_checkpoint(
        tmp_path / "resume.pt",
        field=field,
        scales=scales,
        model=interrupted,
        optimizer=interrupted_optimizer,
        step=2,
        sampler=interrupted_sampler,
    )

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    resumed = _model(scales, field, seed=77)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1.0e-3)
    resumed_sampler = _sampler(field, scales, seed=888)
    info = load_startup_checkpoint(
        path,
        handoff_json=handoff,
        model=resumed,
        optimizer=resumed_optimizer,
        sampler=resumed_sampler,
    )
    assert info["step"] == 2
    assert _state_equal(interrupted.state_dict(), resumed.state_dict())
    for _ in range(2):
        _, value = _one_step(resumed, resumed_optimizer, resumed_sampler, pressure)
        losses_interrupted.append(value)

    assert losses_continuous == pytest.approx(losses_interrupted, rel=0, abs=0.0)
    assert _state_equal(continuous.state_dict(), resumed.state_dict())
    assert math.isfinite(losses_continuous[-1])

    saved_rng = collect_rng_state()
    assert saved_rng["torch"].dtype == torch.uint8
