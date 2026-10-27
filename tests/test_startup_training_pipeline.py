"""完整链路：诊断训练、恢复、多半径、合成监督、预测与对照、失败对照。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spherical_blast_pinn.cli import compare as compare_cli
from spherical_blast_pinn.cli import predict as predict_cli
from spherical_blast_pinn.cli._common import build_context
from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.data.startup_supervision import (
    SupervisionDataset,
    TimeOrigin,
    synthetic_dataset,
)
from spherical_blast_pinn.losses.objective import (
    LossBreakdown,
    total_loss,
    total_loss_for_cases,
)
from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.similitude import PlanarStartupScales
from spherical_blast_pinn.physics.two_medium_initial import (
    build_two_medium_initial_field,
)
from spherical_blast_pinn.physics.two_medium_startup_handoff import (
    write_planar_startup_handoff,
)
from spherical_blast_pinn.physics.two_medium_startup_planar import (
    PlanarStartupConfig,
    build_planar_startup_field,
)
from spherical_blast_pinn.training import trainer as trainer_module
from spherical_blast_pinn.training.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_PURPOSE,
    CHECKPOINT_VERSION,
    TrainingError,
    load_startup_checkpoint,
)
from spherical_blast_pinn.training.trainer import (
    RUN_DIAGNOSTIC,
    Trainer,
    TrainingError as TrainerError,
)
from spherical_blast_pinn.training.startup_cases import CaseSetupError

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "startup_diagnostic.yaml"


def _config(tmp_path, *, cases=None, data=None, sampling=None):
    config = load_config(CONFIG)
    config = config.set("startup.handoff_dir", str(tmp_path / "handoffs"))
    config = config.set("training.output_dir", str(tmp_path / "run"))
    if cases is not None:
        config = config.set("startup.cases", cases)
    if data is not None:
        specs = config.list_of("startup.cases")
        specs = [dict(spec, data=data) for spec in specs]
        config = config.set("startup.cases", specs)
    if sampling:
        for key, value in sampling.items():
            config = config.set(f"sampling.{key}", value)
    return config


def _single_case(config):
    return [config.list_of("startup.cases")[0]]


def _handoff(
    tmp_path,
    name,
    *,
    radius=0.05,
    jwl=None,
    product_density=None,
    air_density=None,
    air_pressure=None,
    air_gamma=None,
):
    initial = build_two_medium_initial_field(
        charge_radius=radius,
        outer_radius=4.0 * radius,
        jwl=jwl,
        product_density=product_density,
        air_density=air_density,
        air_pressure=air_pressure,
        air_gamma=air_gamma,
    )
    field = build_planar_startup_field(
        initial,
        config=PlanarStartupConfig(
            delta=0.005,
            inner_cells=16,
            fan_cells=24,
            plateau_cells=4,
            air_cells=6,
            ambient_cells=8,
            dense_samples=1024,
        ),
    )
    path, _ = write_planar_startup_handoff(field, Path(tmp_path) / name)
    return path, field


def _case_spec(case_id, handoff, *, radius=0.05):
    return {
        "case_id": case_id,
        "charge_radius_m": radius,
        "split": "train",
        "handoff_json": str(handoff),
    }


def _trainer(context, *, seed=0, config=None, output_dir=None, **kwargs):
    return Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=seed,
        output_dir=output_dir or (context.config.path_of("training.output_dir")),
        log_every=1,
        checkpoint_every=0,
        check_every=0,
        config=config or context.config,
        diagnostic=True,
        diagnostic_max_updates=20,
        **kwargs,
    )


def _fingerprint_model(model):
    return {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }


def _same_state(left, right):
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def test_single_radius_diagnostic_train_predict_roundtrip(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=11, diagnostic=True)
    trainer = _trainer(context, seed=11, config=config)
    before = _fingerprint_model(context.model)
    result = trainer.run(max_updates=10, diagnostic=True)

    assert result.status == RUN_DIAGNOSTIC
    assert result.updates == 10
    assert math.isfinite(result.final_loss)
    assert all(math.isfinite(row["total"]) for row in result.history)
    assert not _same_state(before, _fingerprint_model(context.model))
    checkpoint = Path(result.checkpoints["last"])
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training"]["diagnostic"]["enabled"] is True
    assert payload["training"]["resumable"] is True
    assert payload["training"]["step"] == 10

    case = context.training_cases[0]
    output = tmp_path / "predict.npz"
    code = predict_cli.main([
        "--checkpoint", str(checkpoint),
        "--handoff-json", str(case.handoff_json),
        "--case-id", case.case_id,
        "--tau", "1e-7,2e-7",
        "--output", str(output),
    ])
    assert code == 0
    assert output.exists()
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["case_id"] == case.case_id
    assert metadata["time_convention"] == "planar_startup_epsilon"
    assert metadata["epsilon_s"] == pytest.approx(float(case.field.epsilon))
    with np.load(output, allow_pickle=False) as archive:
        assert archive["contact_radius"].size == 2
        assert np.all(np.isfinite(archive["contact_radius"]))
        assert np.all(np.isfinite(archive["shock_radius"]))
        finite = np.isfinite(archive["pressure"])
        assert finite.any()
        assert (archive["region"][finite] >= 0).all()


def test_five_plus_five_resume_matches_a_continuous_ten_step_run(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))

    continuous_context = build_context(config, seed=17, diagnostic=True)
    continuous = _trainer(continuous_context, seed=17, config=config)
    continuous.run(max_updates=10, diagnostic=True)

    interrupted_context = build_context(config, seed=17, diagnostic=True)
    interrupted = _trainer(interrupted_context, seed=17, config=config)
    interrupted.run(max_updates=5, diagnostic=True)
    saved_window = {
        case.case_id: case.sampler.window_state() for case in interrupted.cases
    }
    saved_rng = {
        case.case_id: case.sampler.rng_state() for case in interrupted.cases
    }
    checkpoint = interrupted.save(tmp_path / "resume.pt")

    resumed_context = build_context(config, seed=17, diagnostic=True)
    resumed = _trainer(resumed_context, seed=17, config=config)
    info = resumed.load_checkpoint(checkpoint)
    assert info["step"] == 5
    assert resumed.step == 5
    assert resumed._run_origin["explicit_diagnostic"] is True
    assert resumed._current_stage == interrupted._current_stage
    assert resumed._stage_window == interrupted._stage_window
    assert resumed.balancer.state_dict() == interrupted.balancer.state_dict()
    assert resumed.history == interrupted.history

    # 只加载参数与"续训"必须区分：前者不恢复步号、优化器或采样状态。
    weights_context = build_context(config, seed=17, diagnostic=True)
    weights_only = _trainer(weights_context, seed=17, config=config)
    weights_only.load_weights(checkpoint)
    assert weights_only.step == 0
    assert _same_state(
        _fingerprint_model(interrupted.model), _fingerprint_model(weights_only.model)
    )

    # 采样配置改变后不能冒充精确续训。
    changed = config.set("sampling.n_pde_product", 9)
    changed_context = build_context(changed, seed=17, diagnostic=True)
    changed_trainer = _trainer(changed_context, seed=17, config=changed)
    with pytest.raises(TrainingError, match="counts|sampling"):
        changed_trainer.load_checkpoint(checkpoint)

    # 梯度裁剪/验收容差/平衡配置改变同样不能冒充精确续训。
    settings_context = build_context(config, seed=17, diagnostic=True)
    settings_trainer = _trainer(
        settings_context, seed=17, config=config, gradient_clip=0.5
    )
    with pytest.raises(TrainingError, match="settings"):
        settings_trainer.load_checkpoint(checkpoint)
    for case in resumed.cases:
        assert case.sampler.window_state() == saved_window[case.case_id]
        assert case.sampler.rng_state() == saved_rng[case.case_id]
    # 下一批采样：把一个探针采样器恢复成保存时的窗口与 RNG，结果必须与恢复后
    # 的采样器逐位一致。
    resumed_batch = resumed.cases[0].sampler.batch()
    probe_context = build_context(config, seed=17, diagnostic=True)
    probe = probe_context.training_cases[0].sampler
    probe.set_window_state(saved_window[probe.case_id])
    probe.set_rng_state(saved_rng[probe.case_id])
    probe_batch = probe.batch()
    for name in probe_batch.tensor_fields():
        assert torch.equal(
            getattr(probe_batch, name), getattr(resumed_batch, name)
        ), name
    # 刚才的核对消耗了一次采样；把恢复后的采样器退回保存时的状态，才能继续
    # 与"连续 10 步"逐步比较。
    resumed.cases[0].sampler.set_rng_state(saved_rng[resumed.cases[0].case_id])
    resumed.cases[0].sampler.set_window_state(saved_window[resumed.cases[0].case_id])
    resumed.run(max_updates=10, diagnostic=True)

    assert [row["total"] for row in continuous.history] == pytest.approx(
        [row["total"] for row in resumed.history], rel=0, abs=0.0
    )
    assert _same_state(
        _fingerprint_model(continuous.model), _fingerprint_model(resumed.model)
    ), "5 + save/load + 5 must reproduce the continuous 10-step parameters"


def test_multiple_radii_use_their_own_scales_and_restore_per_case_state(tmp_path):
    config = _config(tmp_path)  # two cases from the diagnostic config
    context = build_context(config, seed=5, diagnostic=True)
    assert len(context.training_cases) == 2
    epsilons = {case.case_id: float(case.field.epsilon) for case in context.training_cases}
    assert epsilons["diag_r050"] != epsilons["diag_r080"]

    trainer = _trainer(context, seed=5, config=config)
    # 一个共享模型、一个优化器；每个参数在 optimizer 中只出现一次。
    assert trainer.model is context.model
    optimizer_ids = [
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == {id(p) for p in trainer.model.parameters() if p.requires_grad}

    result = trainer.run(max_updates=2, diagnostic=True)
    per_case = result.history[-1]["per_case"]
    assert set(per_case) == {case.case_id for case in context.training_cases}
    assert all(math.isfinite(value) for value in per_case.values())
    for case in context.training_cases:
        batch = case.sampler.batch()
        assert batch.charge_radius == pytest.approx(case.charge_radius)
        assert float(batch.scales.charge_radius) == pytest.approx(case.charge_radius)
        assert float(batch.scales.startup_epsilon) == pytest.approx(
            float(case.field.epsilon)
        )
        # 共享网络的无量纲位置乘以**本半径**的 R0；速度与半径无关。
        z = torch.tensor([[0.0]])
        positions = trainer.model.front_positions(z)
        state = trainer.model.front_state(
            z.clone().requires_grad_(True),
            charge_radius=case.charge_radius,
            create_graph=False,
            scales=case.scales,
        )
        assert float(state.contact_radius) == pytest.approx(
            case.charge_radius * float(positions.contact_hat)
        )
        assert float(state.shock_radius) == pytest.approx(
            case.charge_radius * float(positions.shock_hat)
        )
    speeds = [
        float(trainer.model.front_state(
            torch.tensor([[0.0]], requires_grad=True),
            charge_radius=case.charge_radius,
            create_graph=False,
            scales=case.scales,
        ).contact_speed)
        for case in context.training_cases
    ]
    assert speeds[0] == pytest.approx(speeds[1])
    assert context.training_cases[0].charge_radius != context.training_cases[1].charge_radius

    checkpoint = trainer.save(tmp_path / "multi.pt")
    saved = {
        case.case_id: (case.sampler.window_state(), case.sampler.rng_state())
        for case in trainer.cases
    }
    fresh_context = build_context(config, seed=5, diagnostic=True)
    fresh = _trainer(fresh_context, seed=5, config=config)
    fresh.load_checkpoint(checkpoint)
    for case in fresh.cases:
        assert (case.sampler.window_state(), case.sampler.rng_state()) == saved[case.case_id]
    assert [case.case_id for case in fresh.cases] == ["diag_r050", "diag_r080"]
    with pytest.raises(TrainingError, match="case set|missing|extra"):
        from spherical_blast_pinn.training.checkpoint import load_training_checkpoint

        load_training_checkpoint(
            checkpoint,
            cases=[fresh.cases[0]],
            model=fresh.model,
            optimizer=fresh.optimizer,
            schedule=fresh.schedule,
        )


def _supervision_dataset(context, *, origin_unknown=False):
    case = context.training_cases[0]
    eps = float(case.field.epsilon)
    # 足够密，使窄空气层 [Rc, Rs] 内至少有可对照的参照点。
    radius = np.linspace(0.0005, 0.075, 1000)
    shape = (2, radius.size)
    medium = np.full(shape, -1, dtype=np.int8)
    medium[:, radius < 0.05] = 0
    medium[:, (radius > 0.05) & (radius < 0.06)] = 1
    fields = {
        "density": np.full(shape, 1.225),
        "velocity": np.zeros(shape),
        "specific_energy": np.full(shape, 2.0e5),
        "pressure": np.full(shape, 101320.0),
    }
    valid = {name: np.ones(shape, dtype=bool) for name in fields}
    # 空气侧的 velocity 整列无效：无论采到哪些空气点，u 标签都必须是 NaN，
    # 而同一行的 rho/e 仍然有效——逐字段有效性不能被"整行丢掉"替代。
    valid["velocity"][medium == 1] = False
    dataset = synthetic_dataset(
        case_id=case.case_id,
        charge_radius_m=0.05,
        times_dyna=np.asarray([eps + 1.0e-7, eps + 2.0e-7]),
        radius=radius,
        origin_time_s=0.0,
        fields=fields,
        valid=valid,
        medium=medium,
        contact_radius=[0.05, 0.05],
        contact_valid=[True, False],
        shock_radius=[0.06, 0.06],
        shock_valid=[False, False],
        contact_delta_m=1.0e-4,
        shock_delta_m=1.0e-4,
    )
    if origin_unknown:
        dataset = SupervisionDataset(
            **{
                **dataset.__dict__,
                "time_origin": TimeOrigin(
                    status="unknown",
                    origin_time_s=None,
                    method="not determined",
                    evidence="no DYNA t_phys=0 marker was provided",
                ),
            }
        )
    return dataset


def test_synthetic_supervision_uses_only_valid_points_and_compare_reports_nulls(tmp_path):
    base = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(base, seed=3, diagnostic=True)
    dataset = _supervision_dataset(context)
    path = dataset.save(tmp_path / "synthetic_reference.npz")

    config = _config(
        tmp_path,
        cases=_single_case(base),
        data={"path": str(path), "synthetic": True},
        sampling={
            "n_data_product": 4,
            "n_data_air": 4,
            "n_data_contact": 2,
            "n_data_shock": 2,
        },
    )
    with_data = build_context(config, seed=3, diagnostic=True)
    case = with_data.training_cases[0]
    assert case.data is not None and case.data.synthetic is True
    batch = case.sampler.batch()
    counts = batch.diagnostics["data_counts"]
    assert counts["product"] > 0 and counts["air"] > 0
    assert counts["contact"] == 1      # 只有第一帧的 contact 有效
    assert counts["shock"] == 0        # shock 全程无效 → 不产生样本
    # 无效的 velocity 只把该字段标成 NaN，不丢掉整行，也不填 0。
    assert torch.isnan(batch.u_data_air).all()
    assert torch.isfinite(batch.rho_data_air).all()

    trainer = _trainer(with_data, seed=3, config=config)
    trainer.run(max_updates=1, diagnostic=True)
    checkpoint = trainer.save(tmp_path / "with_data.pt")
    prediction = tmp_path / "synthetic_predict.npz"
    assert predict_cli.main([
        "--checkpoint", str(checkpoint),
        "--handoff-json", str(case.handoff_json),
        "--case-id", case.case_id,
        "--tau", "1e-7,2e-7",
        "--output", str(prediction),
    ]) == 0
    report_path = tmp_path / "compare.json"
    assert compare_cli.main([
        "--prediction", str(prediction),
        "--reference", str(path),
        "--output", str(report_path),
    ]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["matched_frames"] == 2
    assert report["reference_synthetic"] is True
    # 两帧的 shock 都无效 → 汇总必须是 null，而不是 0。
    assert report["summary"]["shock_relative_error_mean"] is None
    assert report["frames"][0]["fronts"]["contact_relative_error"] is not None
    assert report["frames"][1]["fronts"]["contact_available"] is False
    assert report["frames"][0]["overpressure"]["curve"]["radius_m"]
    # 参照超压恒为 0 时相对误差无定义 → null；rms 仍是有限的测量。
    if report["summary"]["overpressure_relative_l2_mean"] is None:
        assert report["frames"][0]["overpressure"]["rms"] is not None


def test_supervision_with_an_unknown_time_origin_is_refused(tmp_path):
    base = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(base, seed=3, diagnostic=True)
    bad = _supervision_dataset(context, origin_unknown=True)
    path = bad.save(tmp_path / "unknown_origin.npz")
    config = _config(
        tmp_path,
        cases=_single_case(base),
        data={"path": str(path), "synthetic": True},
    )
    with pytest.raises(CaseSetupError, match="time origin|origin"):
        build_context(config, seed=3, diagnostic=True)


def test_formal_assembly_no_longer_requires_training_qualified(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=False)
    assert context.training_cases[0].field.training_qualified is False
    assert context.model is not None
    # Trainer construction is an entry check; no optimizer step is performed.
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=0,
        output_dir=tmp_path / "formal_entry",
        check_every=0,
        checkpoint_every=0,
        config=config,
        diagnostic=False,
    )
    assert trainer.model is context.model


def test_predict_refuses_the_wrong_handoff_and_nonzero_exit(tmp_path):
    # 先用默认两半径配置生成两份 handoff，再只训练 50 mm 这一份。
    build_context(_config(tmp_path), seed=0, diagnostic=True)
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=True)
    trainer = _trainer(context, seed=0, config=config)
    trainer.run(max_updates=1, diagnostic=True)
    checkpoint = trainer.save(tmp_path / "one.pt")
    other_handoffs = sorted(
        (tmp_path / "handoffs").glob("**/handoff.json")
    )
    other = next(path for path in other_handoffs if path != context.training_cases[0].handoff_json)
    code = predict_cli.main([
        "--checkpoint", str(checkpoint),
        "--handoff-json", str(other),
        "--tau", "1e-7",
        "--output", str(tmp_path / "bad.npz"),
    ])
    assert code != 0


def test_incompatible_checkpoint_version_is_refused(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=True)
    trainer = _trainer(context, seed=0, config=config)
    trainer.run(max_updates=1, diagnostic=True)
    path = trainer.save(tmp_path / "v2.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = CHECKPOINT_VERSION - 1
    old = tmp_path / "v1.pt"
    torch.save(payload, old)
    with pytest.raises(TrainingError, match="version"):
        load_startup_checkpoint(old)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["purpose"] = "something_else"
    wrong = tmp_path / "wrong_purpose.pt"
    torch.save(payload, wrong)
    with pytest.raises(TrainingError, match="purpose"):
        load_startup_checkpoint(wrong)


def test_joint_gradient_is_the_equal_weight_mean_of_the_case_gradients(tmp_path):
    config = _config(tmp_path)
    context = build_context(config, seed=9, diagnostic=True)
    trainer = _trainer(context, seed=9, config=config)
    batches = [case.sampler.batch() for case in trainer.cases]
    pressures = [case.contact_pressure for case in trainer.cases]
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    case_grads = []
    for batch, pressure in zip(batches, pressures):
        loss = total_loss([batch], trainer.model, contact_pressure=pressure)
        case_grads.append(
            torch.autograd.grad(loss.total, parameters, retain_graph=True)
        )
    joint = total_loss_for_cases(
        batches, trainer.model, contact_pressures=pressures
    )
    joint_grads = torch.autograd.grad(joint.total, parameters, retain_graph=True)
    for index, joint_grad in enumerate(joint_grads):
        expected = sum(
            (grads[index] for grads in case_grads),
            torch.zeros_like(joint_grad),
        ) / len(case_grads)
        assert torch.allclose(joint_grad, expected, rtol=1e-12, atol=1e-12)


def test_frozen_parameters_predict_an_untrained_radius_without_updating(tmp_path):
    config = _config(tmp_path)
    context = build_context(config, seed=6, diagnostic=True)
    trainer = _trainer(context, seed=6, config=config)
    trainer.run(max_updates=2, diagnostic=True)
    checkpoint = trainer.save(tmp_path / "shared.pt")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    # 100 mm 不在训练配置里；现场构造其 handoff，仅作为预测目标。
    initial = build_two_medium_initial_field(
        charge_radius=0.10, outer_radius=0.40
    )
    field = build_planar_startup_field(
        initial,
        config=PlanarStartupConfig(
            delta=0.005, inner_cells=16, fan_cells=24, plateau_cells=4,
            air_cells=6, ambient_cells=8, dense_samples=1024,
        ),
    )
    handoff_dir = tmp_path / "untrained_100mm"
    handoff_json, _ = write_planar_startup_handoff(field, handoff_dir)

    output = tmp_path / "predict_100.npz"
    code = predict_cli.main([
        "--checkpoint", str(checkpoint),
        "--handoff-json", str(handoff_json),
        "--radius", "0.10",
        "--tau", "1e-7",
        "--output", str(output),
    ])
    assert code == 0
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "frozen_shared_parameters"
    assert metadata["charge_radius_m"] == pytest.approx(0.10)
    assert metadata["epsilon_s"] == pytest.approx(float(field.epsilon))
    assert metadata["trained_case_ids"] == ["diag_r050", "diag_r080"]
    with np.load(output, allow_pickle=False) as archive:
        assert np.all(np.isfinite(archive["contact_radius"]))
        assert np.isfinite(archive["pressure"]).any()
        predicted_contact = archive["contact_radius"][0]
        predicted_shock = archive["shock_radius"][0]

    # 预测使用的确实是检查点里的冻结权重：独立重建同一结构、装入同一份参数，
    # 用新半径的 scales 在 tau=1e-7 重算前沿，必须与输出一致。
    from spherical_blast_pinn.models.coupled_model import build_model_from_description
    from spherical_blast_pinn.training.checkpoint import (
        SCALES_KEY,
        inspect_startup_checkpoint,
        load_frozen_model,
        planar_scales_from_record,
    )

    inspected = inspect_startup_checkpoint(checkpoint)
    new_scales = PlanarStartupScales.for_startup_field(field, time_scale=2.0)
    rebuilt = build_model_from_description(
        inspected["model"]["model_structure"], scales=new_scales
    )
    load_frozen_model(
        checkpoint, handoff_json=handoff_json, scales=new_scales, model=rebuilt
    )
    tau = torch.tensor([[1.0e-7]], dtype=torch.float64)
    s = new_scales.network_time(tau).requires_grad_(True)
    state = rebuilt.front_net.state(
        s, charge_radius=field.charge_radius, create_graph=False, scales=new_scales
    )
    assert float(state.contact_radius) == pytest.approx(predicted_contact)
    assert float(state.shock_radius) == pytest.approx(predicted_shock)

    # 预测没有改检查点，也没有创建/更新任何参数。
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training"]["step"] == 2


def test_auto_built_startup_field_uses_configured_physical_conditions(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    config = (
        config.set("air.gamma", 1.35)
        .set("air.rho", 1.10)
        .set("air.pressure", 95000.0)
    )
    context = build_context(config, seed=0, diagnostic=True)
    case = context.training_cases[0]
    state = case.field.initial_state
    assert case.field.air_gamma == pytest.approx(1.35)
    assert state["air_density"] == pytest.approx(1.10)
    assert state["air_pressure"] == pytest.approx(95000.0)
    # The model is built from the field, not from a separate YAML reading.
    assert context.model.air_net.gamma_air == pytest.approx(1.35)
    assert context.model.undisturbed.density == pytest.approx(1.10)
    assert context.model.undisturbed.pressure == pytest.approx(95000.0)
    assert (
        context.model.product_net.jwl.as_dict()
        == case.field.product_eos.as_dict()
    )


def test_default_config_reproduces_the_direct_default_startup_field(tmp_path):
    """The default YAML must not perturb the existing initial field or Riemann state."""
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=True)
    built = context.training_cases[0].field

    direct_initial = build_two_medium_initial_field(
        charge_radius=0.05, outer_radius=0.20
    )
    direct = build_planar_startup_field(
        direct_initial,
        config=PlanarStartupConfig(
            delta=0.005, inner_cells=16, fan_cells=24, plateau_cells=4,
            air_cells=6, ambient_cells=8, dense_samples=1024,
        ),
    )
    assert built.cache_key() == direct.cache_key()
    assert float(built.epsilon) == float(direct.epsilon)
    for name in (
        "radii", "cell_volume", "density", "velocity", "specific_energy",
        "pressure", "sound_speed", "material", "region", "mass", "node_velocity",
    ):
        assert np.array_equal(getattr(built, name), getattr(direct, name)), name
    assert dict(built.initial_state) == dict(direct.initial_state)


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"air.gamma": 1.35}, "air.gamma"),
        ({"air.rho": 1.0}, "air.rho"),
        ({"air.pressure": 90000.0}, "air.pressure"),
        ({"jwl.A": 3.75e11}, "jwl.A"),
        ({"jwl.not_a_parameter": 1.0}, "unknown physical key"),
    ],
)
def test_handoff_physical_conflict_is_rejected_before_optimization(
    tmp_path, overrides, needle
):
    handoff, _ = _handoff(tmp_path, "default")
    config = _config(
        tmp_path, cases=[_case_spec("r050", handoff)]
    )
    for key, value in overrides.items():
        config = config.set(key, value)
    with pytest.raises(CaseSetupError, match=needle):
        build_context(config, seed=0, diagnostic=True)


def test_custom_jwl_handoff_is_bound_to_model_or_rejected(tmp_path):
    custom = replace(mat.product_jwl(), A=3.75e11, B=3.70e9)
    handoff, field = _handoff(tmp_path, "custom", jwl=custom)

    mismatched = _config(
        tmp_path, cases=[_case_spec("r050", handoff)]
    )
    with pytest.raises(CaseSetupError, match="jwl.A"):
        build_context(mismatched, seed=0, diagnostic=True)

    matching = mismatched.set("jwl.A", custom.A).set("jwl.B", custom.B)
    context = build_context(matching, seed=0, diagnostic=True)
    assert context.training_cases[0].field.product_eos.as_dict() == field.product_eos.as_dict()
    assert context.model.product_net.jwl.as_dict() == field.product_eos.as_dict()


def test_custom_jwl_auto_build_reaches_the_initial_field_and_model(tmp_path):
    custom = replace(mat.product_jwl(), A=3.75e11, B=3.70e9)
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    config = config.set("jwl.A", custom.A).set("jwl.B", custom.B)
    context = build_context(config, seed=0, diagnostic=True)
    field = context.training_cases[0].field
    assert field.product_eos.A == pytest.approx(custom.A)
    assert context.model.product_net.jwl.A == pytest.approx(custom.A)


def test_cases_with_different_initial_state_are_rejected(tmp_path):
    standard, _ = _handoff(tmp_path, "standard")
    different, _ = _handoff(
        tmp_path, "different_state", product_density=1600.0
    )
    config = _config(
        tmp_path,
        cases=[
            _case_spec("r050_standard", standard),
            _case_spec("r050_different", different),
        ],
    )
    with pytest.raises(CaseSetupError, match="initial_state.product_density"):
        build_context(config, seed=0, diagnostic=True)


def test_cases_with_different_jwl_are_rejected_even_without_config_values(tmp_path):
    custom = replace(mat.product_jwl(), A=3.75e11, B=3.70e9)
    standard, _ = _handoff(tmp_path, "standard")
    other, _ = _handoff(tmp_path, "other", jwl=custom)
    config = _config(
        tmp_path,
        cases=[
            _case_spec("r050_standard", standard),
            _case_spec("r050_other", other),
        ],
    ).set("jwl", {}).set("air", {})
    with pytest.raises(CaseSetupError, match="JWL.A"):
        build_context(config, seed=0, diagnostic=True)


def test_checkpoint_material_record_binds_initial_state_and_model(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=True)
    trainer = _trainer(context, seed=0, config=config)
    trainer.run(max_updates=1, diagnostic=True)
    checkpoint = trainer.save(tmp_path / "material.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    record = payload["cases"][0]["material"]
    assert "initial_state" in record
    assert record["initial_state"]["product_density"] == pytest.approx(
        context.training_cases[0].field.initial_state["product_density"]
    )
    assert record["jwl"] == context.training_cases[0].field.product_eos.as_dict()

    record["air_gamma"] = 1.35
    tampered = tmp_path / "tampered_material.pt"
    torch.save(payload, tampered)
    with pytest.raises(TrainingError, match="material|shared model"):
        load_startup_checkpoint(tampered, model=context.model)

    # The model structure itself is checked against the case record, even
    # without supplying a live model (inspection path).
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["model_structure"]["describe"]["product"]["jwl"]["A"] *= 1.01
    structure_tampered = tmp_path / "tampered_structure.pt"
    torch.save(payload, structure_tampered)
    with pytest.raises(TrainingError, match="model structure"):
        load_startup_checkpoint(structure_tampered)

    # A truncated material record is not silently accepted by skipping the
    # missing keys during comparison.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload["cases"][0]["material"]["initial_state"]["product_density"]
    truncated = tmp_path / "truncated_material.pt"
    torch.save(payload, truncated)
    with pytest.raises(TrainingError, match="missing"):
        load_startup_checkpoint(truncated)


def test_frozen_prediction_refuses_a_new_radius_with_a_different_initial_state(tmp_path):
    from spherical_blast_pinn.training.checkpoint import load_frozen_model

    config = _config(tmp_path)
    context = build_context(config, seed=6, diagnostic=True)
    trainer = _trainer(context, seed=6, config=config)
    trainer.run(max_updates=1, diagnostic=True)
    checkpoint = trainer.save(tmp_path / "shared.pt")

    handoff, field = _handoff(
        tmp_path, "untrained_100_different", radius=0.10, product_density=1600.0
    )
    scales = PlanarStartupScales.for_startup_field(field, time_scale=2.0)
    from spherical_blast_pinn.models.coupled_model import build_model_from_description
    from spherical_blast_pinn.training.checkpoint import inspect_startup_checkpoint

    inspected = inspect_startup_checkpoint(checkpoint)
    model = build_model_from_description(
        inspected["model"]["model_structure"], scales=scales
    )
    with pytest.raises(TrainingError, match="product_density|initial"):
        load_frozen_model(
            checkpoint, handoff_json=handoff, scales=scales, model=model
        )


def test_diagnostic_mode_cannot_be_used_to_run_the_formal_budget(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=True)
    trainer = _trainer(context, seed=0, config=config)
    with pytest.raises(TrainerError, match="capped"):
        trainer.run(max_updates=trainer.diagnostic_max_updates + 1, diagnostic=True)


def test_check_set_and_best_record_survive_resume(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=4, diagnostic=True)
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        seed=4,
        output_dir=config.path_of("training.output_dir"),
        log_every=1,
        checkpoint_every=0,
        check_every=1,
        config=config,
        diagnostic=True,
        diagnostic_max_updates=20,
    )
    trainer.run(max_updates=1, diagnostic=True)
    assert trainer.best is not None
    assert trainer._best_model is not None
    checkpoint = trainer.save(tmp_path / "best.pt")
    fresh_context = build_context(config, seed=4, diagnostic=True)
    fresh = Trainer(
        cases=fresh_context.training_cases,
        model=fresh_context.model,
        seed=4,
        output_dir=config.path_of("training.output_dir"),
        log_every=1,
        checkpoint_every=0,
        check_every=1,
        config=config,
        diagnostic=True,
        diagnostic_max_updates=20,
    )
    fresh.load_checkpoint(checkpoint)
    assert fresh.best is not None
    assert fresh._best_model is not None
    assert fresh.best["parameter_fingerprint"] == trainer.best["parameter_fingerprint"]
    for name, value in fresh._best_model.items():
        assert torch.equal(value, trainer._best_model[name])


def test_lbfgs_refinement_is_available_and_marks_the_run_diagnostic(tmp_path):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=2, diagnostic=True)
    trainer = _trainer(context, seed=2, config=config)
    trainer.run(max_updates=1, diagnostic=True)
    result = trainer.refine(iterations=1)
    assert result.status == RUN_DIAGNOSTIC
    assert result.metadata["refinement"]
    assert result.metadata["run_origin"]["refined"] is True
    payload = torch.load(result.checkpoints["last"], map_location="cpu", weights_only=False)
    assert payload["training"]["run_origin"]["refined"] is True
    assert payload["training"]["diagnostic"]["enabled"] is True


def test_a_nonfinite_loss_blocks_the_update_and_zero_loss_does_not(tmp_path, monkeypatch):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=True)
    trainer = _trainer(context, seed=0, config=config)
    before = _fingerprint_model(trainer.model)

    def _nan_loss(*args, **kwargs):
        total = torch.tensor(float("nan"), requires_grad=True)
        zero = torch.tensor(0.0, requires_grad=True)
        return LossBreakdown(
            total=total,
            per_case={case.case_id: float("nan") for case in trainer.cases},
            weights={"pde": 1.0},
            terms={},
            group_tensors={"pde": zero},
        )

    monkeypatch.setattr(trainer_module, "total_loss_for_cases", _nan_loss)
    with pytest.raises(trainer_module.NonFiniteUpdate):
        trainer._update()
    assert _same_state(before, _fingerprint_model(trainer.model))

    def _zero_loss(*args, **kwargs):
        zero = torch.zeros((), requires_grad=True)
        return LossBreakdown(
            total=zero,
            per_case={case.case_id: 0.0 for case in trainer.cases},
            weights={"pde": 1.0},
            terms={},
            group_tensors={"pde": zero},
        )

    monkeypatch.setattr(trainer_module, "total_loss_for_cases", _zero_loss)
    trainer.step = 0
    trainer._update()
    assert trainer._last_update_loss == 0.0


def test_a_nonfinite_gradient_with_finite_loss_blocks_the_update(tmp_path, monkeypatch):
    config = _config(tmp_path, cases=_single_case(_config(tmp_path)))
    context = build_context(config, seed=0, diagnostic=True)
    trainer = _trainer(context, seed=0, config=config)
    parameter = trainer.model.parameters().__next__()
    with torch.no_grad():
        parameter.zero_()
    before = _fingerprint_model(trainer.model)

    def _finite_loss_infinite_gradient(*args, **kwargs):
        # 在 0 处 sqrt(x^2) 的值是 0、导数是 inf/nan：损失有限而梯度不有限。
        total = torch.sqrt(parameter.sum() ** 2)
        return LossBreakdown(
            total=total,
            per_case={case.case_id: 0.0 for case in trainer.cases},
            weights={"pde": 1.0},
            terms={},
            group_tensors={"pde": total},
        )

    monkeypatch.setattr(
        trainer_module, "total_loss_for_cases", _finite_loss_infinite_gradient
    )
    with pytest.raises(trainer_module.NonFiniteUpdate, match="gradient"):
        trainer._update()
    assert _same_state(before, _fingerprint_model(trainer.model))
