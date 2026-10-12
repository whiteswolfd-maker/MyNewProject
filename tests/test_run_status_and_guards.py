"""运行终态、评估覆盖、恢复事务性与数值守卫（修改提示词 §1、§2、§3、§6、§7、§8）。

这一组走**真实调用路径**：成功的判定走真实的 ``run()`` 控制流，评估走真实的
``compare_fields``，恢复走真实的 ``load()``。只构造结果类或只调校验函数的测试
在这里只能当补充——复核已经指出过，那样的测试证明不了调用路径是对的。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.training.balancing import BalanceConfig
from spherical_blast_pinn.training.trainer import (
    NonFiniteUpdate,
    RUN_COMPLETED,
    RUN_CRITERIA_FAILED,
    RUN_DIAGNOSTIC,
    RUN_INCOMPLETE,
    TrainingError,
)


def _tiny(monkeypatch, tmp_path: Path, name: str = "w"):
    import test_review_regressions as base

    from spherical_blast_pinn.training import trainer as trainer_module
    from spherical_blast_pinn.training.schedule import SCHEDULE, Stage

    tiny = tuple(
        Stage(
            stage.name, 2,
            stage.pde_weight_start, stage.pde_weight_end,
            stage.learning_rate_start, stage.learning_rate_end,
            time_window_fraction=stage.time_window_fraction,
        )
        for stage in SCHEDULE
    )
    trainer = base._tiny_trainer(tmp_path / name, tmp_path / f"out_{name}")
    trainer.schedule = tiny
    trainer.grading = trainer_module  # 供下面按需替换判据
    return trainer, tiny


# ==========================================================================
# §1 成功判定与最终失败出口
# ==========================================================================


def test_a_full_run_with_passing_criteria_is_completed(monkeypatch, tmp_path):
    """预算跑满 + 判据通过 ⇒ ``completed``。"""
    from spherical_blast_pinn.training import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "criteria_for", lambda name: ())
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 1
    result = trainer.run(max_updates=sum(s.update for s in tiny) if False else sum(
        s.updates for s in tiny
    ), verbose=False)

    assert result.status == RUN_COMPLETED
    assert result.ok is True
    assert result.formally_accepted is True
    assert result.final_stage_verdict is not None
    assert result.final_stage_verdict["passed"] is True


def test_a_full_run_with_failing_criteria_is_not_a_success(monkeypatch, tmp_path):
    """**复核点名的原反例。** 预算跑满但最终判据失败——以前 ``ok`` 仍是 True。

    判据不通过时 ``_evaluate_stage_criteria(fatal=False)`` 只是记一笔，而
    ``TrainingResult.ok`` 只看 ``updates > 0 and isfinite(final_loss)``，于是 CLI
    返回 0。一次被判失败的运行于是和一次正式成功拿到同一个退出码。
    """
    from spherical_blast_pinn.training.schedule import Stage

    trainer, tiny = _tiny(monkeypatch, tmp_path)
    # 单阶段：这样"预算跑满"与"最终判据失败"能同时发生。多阶段时，中间那道
    # 阶段闸门会先以 fatal 的方式拦下运行，走不到"跑满但不通过"这一档。
    trainer.schedule = (Stage("joint", 2, 1.0, 1.0, 1.0e-4, 1.0e-5,
                              time_window_fraction=1.0),)
    trainer.check_every = 1
    trainer.tolerances = {"pde_rms": 0.0}
    total = 2
    result = trainer.run(max_updates=total, verbose=False)

    assert result.status == RUN_CRITERIA_FAILED
    assert result.ok is False
    assert result.formally_accepted is False
    assert result.final_stage_verdict is not None
    assert result.final_stage_verdict["passed"] is False
    # 控制台、结果、清单、检查点必须一致
    assert result.as_dict()["status"] == RUN_CRITERIA_FAILED
    assert result.as_dict()["formally_accepted"] is False
    manifest = trainer.output_dir / "run_manifest.json"
    assert manifest.exists()
    import json

    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == RUN_CRITERIA_FAILED
    payload = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert payload["run_status"] == RUN_CRITERIA_FAILED
    assert payload["final_stage_verdict"]["passed"] is False


def test_a_truncated_run_is_incomplete_and_makes_no_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "spherical_blast_pinn.training.trainer.criteria_for", lambda name: ()
    )
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 1
    result = trainer.run(max_updates=tiny[0].updates, verbose=False)

    assert result.status == RUN_INCOMPLETE
    assert result.ok is False
    assert result.final_stage_verdict is None, "截断的运行不评价任何阶段"
    assert result.metadata["truncated"] is True


def test_a_run_without_any_check_cannot_be_completed(monkeypatch, tmp_path):
    """**缺检查不是通过。** 一次验收检查都没跑，就没有任何结论可言。"""
    monkeypatch.setattr(
        "spherical_blast_pinn.training.trainer.criteria_for", lambda name: ()
    )
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 0
    result = trainer.run(max_updates=sum(s.updates for s in tiny), verbose=False)

    assert result.status == RUN_CRITERIA_FAILED
    assert "no acceptance check" in result.final_stage_verdict["reason"]


def test_an_explicit_diagnostic_run_claims_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "spherical_blast_pinn.training.trainer.criteria_for", lambda name: ()
    )
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 1
    result = trainer.run(
        max_updates=sum(s.updates for s in tiny), verbose=False, diagnostic=True
    )
    assert result.status == RUN_DIAGNOSTIC
    assert result.ok is False, "诊断运行不主张任何验收结论"


def test_the_final_loss_is_labelled_with_the_step_it_came_from(monkeypatch, tmp_path):
    """``final_loss`` 必须是**最后一次更新实测的**，不是最后一条日志。

    日志按 ``log_every`` 打，间隔大于 1 时最后一条日志来自更早的模型——而它与
    最终结论写在同一个字段里。
    """
    monkeypatch.setattr(
        "spherical_blast_pinn.training.trainer.criteria_for", lambda name: ()
    )
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 0
    trainer.log_every = 1
    total = sum(s.updates for s in tiny)
    result = trainer.run(max_updates=total, verbose=False)

    assert result.final_loss_step == total
    assert np.isfinite(result.final_loss)
    assert trainer._last_update_loss_step == total


# ==========================================================================
# §1 退出码
# ==========================================================================


def test_the_cli_exit_codes_distinguish_the_four_end_states():
    from spherical_blast_pinn.cli.train import EXIT_CODES

    assert EXIT_CODES[RUN_COMPLETED] == 0
    assert EXIT_CODES[RUN_CRITERIA_FAILED] != 0
    assert EXIT_CODES[RUN_INCOMPLETE] != 0
    assert EXIT_CODES[RUN_CRITERIA_FAILED] != EXIT_CODES[RUN_INCOMPLETE], (
        "'判据失败'与'被截断'必须能被调用方区分"
    )
    # 诊断运行不是失败，但它主张的东西写在状态里
    assert EXIT_CODES[RUN_DIAGNOSTIC] == 0


# ==========================================================================
# §6 零预算
# ==========================================================================


def test_zero_updates_means_zero_updates_not_the_whole_schedule(monkeypatch, tmp_path):
    """``--max-steps 0`` 以前是假值，被解释成"跑完整日程"。"""
    monkeypatch.setattr(
        "spherical_blast_pinn.training.trainer.criteria_for", lambda name: ()
    )
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 1
    result = trainer.run(max_updates=0, verbose=False)

    assert trainer.step == 0, "零预算必须真的零更新"
    assert result.updates == 0
    assert result.status == RUN_INCOMPLETE


def test_a_negative_budget_is_refused_before_the_expensive_context(tmp_path, monkeypatch):
    """负数预算在构造上下文之前就被拒绝。"""
    from spherical_blast_pinn.cli import train as train_module

    config_path = tmp_path / "c.yaml"
    base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config_path.write_text(f"inherits: {base.as_posix()}\n", encoding="utf-8")

    called = []
    monkeypatch.setattr(
        train_module, "build_context",
        lambda *a, **k: called.append(1),  # 被调用就说明没提前返回
    )
    code = train_module.main(["--config", str(config_path), "--max-steps", "-3"])
    assert code != 0
    assert not called, "负数预算应当在昂贵的上下文构造之前被拒绝"


# ==========================================================================
# §2 评估覆盖与非有限预测
# ==========================================================================


class _FakeCase:
    """``compare_fields`` 用到的最小接口。"""

    def __init__(self, *, n_t=2, n_r=40, n_frames_valid=2):
        self.radius = np.linspace(0.01, 0.20, n_r)
        self.times = np.linspace(1.0e-5, 1.0e-4, n_t)
        self.material = np.where(self.radius < 0.05, 0, 1)[None, :].repeat(n_t, axis=0)
        self.valid = np.ones((n_t, n_r), dtype=bool)
        self.field_validity = {
            name: np.ones((n_t, n_r), dtype=bool)
            for name in ("density", "velocity", "specific_energy", "pressure")
        }
        self.contact_radius = np.full(n_t, 0.05)
        self.shock_radius = np.full(n_t, 0.12)
        self._scale = 1600.0

    def supervisable_frames(self):
        return np.ones(self.times.size, dtype=bool)

    def sample_at(self, frame, radii):
        radii = np.asarray(radii, dtype=np.float64)
        product = radii < 0.05
        return {
            "density": np.where(product, self._scale, 6.0),
            "velocity": np.where(product, 100.0, 10.0),
            "specific_energy": np.where(product, 1.0e7, 2.0e5),
            "pressure": np.where(product, 1.0e10, 1.0e5),
        }


def _prediction(case: _FakeCase, *, density_offset=0.0, non_finite_at=None):
    density = np.where(case.material == 0, 1600.0 + density_offset, 6.0)
    if non_finite_at is not None:
        density = density.copy()
        density[:, non_finite_at] = np.nan
    pred = {
        "physical_time": case.times.copy(),
        "pinn_time": case.times - 1.0e-6,
        "start_time": 1.0e-6,
        "radius": case.radius.copy(),
        "density": density,
        "velocity": np.where(case.material == 0, 100.0, 10.0),
        "specific_energy": np.where(case.material == 0, 1.0e7, 2.0e5),
        "pressure": np.where(case.material == 0, 1.0e10, 1.0e5),
        "contact_radius": case.contact_radius.copy(),
        "shock_radius": case.shock_radius.copy(),
        # ``_field_at`` 按预测自己的区域标签判定边界，跨间断不做插值
        "region": np.where(
            case.material == 0, 0,
            np.where(case.radius[None, :] < case.shock_radius[:, None], 1, 2),
        ).astype(np.int64),
        "scale_product_density": 1600.0,
        "scale_air_density": 6.0,
        "scale_velocity": 100.0,
        "scale_specific_energy": 1.0e7,
    }
    return pred


def test_compare_uses_per_field_validity():
    """**复核点名的原反例。** 联合掩码为假、密度自身有效 ⇒ 密度仍参与评价。

    训练采样早已按字段各自取点，评估却还在用联合掩码——两边不是同一个有效性，
    算出来的误差也不是同一个量。
    """
    from spherical_blast_pinn.cli.compare import compare_fields

    case = _FakeCase()
    product = case.material[0] == 0
    case.valid[:, product] = False                      # 联合掩码：产物区全不可用
    for name, mask in case.field_validity.items():
        mask[:, product] = name == "density"            # 只有密度自己有効
    out = compare_fields(_prediction(case), case)

    assert out["counts"]["product_density"]["compared"] > 0, (
        "联合掩码为假时密度标签被整片丢掉了；逐字段有效性没有生效"
    )
    assert out["counts"]["product_velocity"]["compared"] == 0
    assert "per-field validity" in out["validity_source"]


def test_a_non_finite_prediction_is_counted_not_silently_dropped():
    """**复核点名的原反例。** 一个预测是 NaN 时，失败数量必须记下来。

    ``good = isfinite(reference) & isfinite(model)`` 会把这一项从分子分母里一起
    删掉：三个候选点里一个预测是 NaN，剩下两个正确，于是报出零误差、``available``
    仍然为真——失败点消失得无影无踪。
    """
    from spherical_blast_pinn.cli.compare import compare_fields

    case = _FakeCase()
    out = compare_fields(_prediction(case, non_finite_at=5), case)

    non_finite = out["counts"]["product_density"]["model_non_finite"]
    assert non_finite > 0, "非有限预测被静默删掉了"
    assert out["complete"] is False
    assert any("model non-finite" in item for item in out["unavailable_fields"])


def test_every_dropped_point_is_accounted_for():
    """计数必须闭合：候选 = 缺测 + 域外 + 错配 + 保护区 + 比较（+ 未分类）。"""
    from spherical_blast_pinn.cli.compare import compare_fields

    case = _FakeCase()
    out = compare_fields(_prediction(case), case)
    for key, entry in out["counts"].items():
        accounted = (
            entry["reference_missing"] + entry["outside_domain"]
            + entry["region_mismatch"] + entry["buffer_zone"] + entry["compared"]
        )
        assert accounted == entry["candidates"], f"{key}: {entry}"


def test_a_field_with_no_comparable_point_is_unavailable_not_zero_error():
    """零样本不是零误差。"""
    from spherical_blast_pinn.cli.compare import compare_fields

    case = _FakeCase()
    for name, mask in case.field_validity.items():
        mask[:, :] = False
    out = compare_fields(_prediction(case), case)

    assert out["complete"] is False
    assert out["unavailable_fields"], "全部不可比时必须有字段被标为不可用"
    assert np.isnan(out["product_density_relative_l2"]), (
        "没有可比点时给出了一个数值——那会被读成零误差"
    )


# ==========================================================================
# §3 恢复的事务性
# ==========================================================================


def _writer(monkeypatch, tmp_path):
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="w")
    trainer.check_every = 0
    trainer.checkpoint_every = 10_000
    trainer._update()
    trainer.step += 1
    trainer._save(trainer.output_dir / "one.pt")
    return trainer, tiny


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda p: p.update({"sampler_rng": {"version": 1}}), "samplers"),
        (lambda p: p["sampler_rng"]["samplers"].pop("frame-check"), "missing"),
        (lambda p: p["sampler_rng"].update({"version": 99}), "version"),
        (lambda p: p["stage"].pop("window_fraction"), "window_fraction"),
        (lambda p: p["stage"].pop("blocked"), "blocked"),
        (lambda p: p.pop("balancing"), "adaptive-weight"),
        (lambda p: p["model"].pop(next(iter(p["model"]))), "does not match"),
    ],
)
def test_a_broken_checkpoint_fails_without_touching_the_target(
    monkeypatch, tmp_path, mutate, needle
):
    """**复核点名的原反例。** 各种损坏的检查点都必须：拒绝，且**对象原封不动**。

    以前只把窗口检查提前：非法采样器 RNG、缺 ``window_fraction``、模型键不匹配，
    都在模型、优化器、步数与全局 RNG 已经改掉之后才报错。调用者收到"恢复失败"，
    手里的对象却已经是半加载的。
    """
    writer, tiny = _writer(monkeypatch, tmp_path)
    payload = torch.load(
        writer.output_dir / "one.pt", map_location="cpu", weights_only=False
    )
    mutate(payload)
    torch.save(payload, writer.output_dir / "broken.pt")

    import test_review_regressions as base

    reader = base._tiny_trainer(tmp_path / "r", tmp_path / "out_r")
    reader.schedule = tiny
    before_model = {
        name: value.clone() for name, value in reader.model.state_dict().items()
    }
    before_step = reader.step
    before_window = reader.sampler.samplers[0].window_fraction
    before_rng = reader.sampler.rng_state()
    before_numpy = np.random.get_state()[1].copy()

    with pytest.raises(TrainingError, match=needle):
        reader.load(writer.output_dir / "broken.pt")

    assert reader.step == before_step
    assert reader.sampler.samplers[0].window_fraction == pytest.approx(before_window)
    assert reader.sampler.rng_state() == before_rng
    np.testing.assert_array_equal(np.random.get_state()[1], before_numpy)
    for name, value in before_model.items():
        assert torch.equal(value, reader.model.state_dict()[name]), (
            f"{name} 在恢复失败时已经被改掉了"
        )


def test_a_failure_inside_the_commit_rolls_everything_back(monkeypatch, tmp_path):
    """提交阶段**内部**出错也要完整回滚，不只是事前检查不通过的那几条路径。"""
    writer, tiny = _writer(monkeypatch, tmp_path)
    import test_review_regressions as base

    reader = base._tiny_trainer(tmp_path / "r", tmp_path / "out_r")
    reader.schedule = tiny

    before_model = {
        name: value.clone() for name, value in reader.model.state_dict().items()
    }
    before_step = reader.step

    original = reader.sampler.set_rng_state

    def explode(payload):
        original(payload)
        raise RuntimeError("boom, after the sampler RNG was already replaced")

    monkeypatch.setattr(reader.sampler, "set_rng_state", explode)
    with pytest.raises(RuntimeError, match="boom"):
        reader.load(writer.output_dir / "one.pt")

    assert reader.step == before_step
    for name, value in before_model.items():
        assert torch.equal(value, reader.model.state_dict()[name]), (
            f"{name} 在提交中途失败后没有被回滚"
        )


# ==========================================================================
# §7 精修后不能沿用旧模型的检查结果
# ==========================================================================


def test_refinement_invalidates_the_previous_check(monkeypatch, tmp_path):
    """精修改参数却不改 ``step``；按"这一步查过了"判断会把旧结论留下。"""
    from spherical_blast_pinn.training.schedule import Stage

    monkeypatch.setattr(
        "spherical_blast_pinn.training.trainer.criteria_for", lambda name: ()
    )
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.schedule = (Stage("joint", 2, 1.0, 1.0, 1.0e-4, 1.0e-5,
                              time_window_fraction=1.0),)
    trainer.check_every = 1
    trainer.checkpoint_every = 10_000
    trainer.run(max_updates=2, verbose=False)

    fingerprint_before = trainer.check_history[-1]["parameter_fingerprint"]
    assert fingerprint_before == trainer.parameter_fingerprint()

    result = trainer.refine(iterations=2, verbose=False)

    # 精修动过参数 ⇒ 指纹变了 ⇒ 检查必须重做过
    assert trainer.check_history[-1]["parameter_fingerprint"] == trainer.parameter_fingerprint()
    assert trainer.check_history[-1]["parameter_fingerprint"] != fingerprint_before, (
        "精修后的检查结论仍对应精修前的参数"
    )
    # 结论必须与当前参数一致，并如实说明精修改过目标函数
    assert result.status == RUN_DIAGNOSTIC
    assert result.metadata["refinement"][0]["closures"] >= 1
    assert result.metadata["refinement"][0]["parameters_changed"] is True


def test_refinement_records_the_weights_it_used(monkeypatch, tmp_path):
    """精修不能**无说明地**从自适应权重退到默认权重。"""
    from spherical_blast_pinn.training.schedule import Stage

    monkeypatch.setattr(
        "spherical_blast_pinn.training.trainer.criteria_for", lambda name: ()
    )
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.schedule = (Stage("joint", 2, 1.0, 1.0, 1.0e-4, 1.0e-5,
                              time_window_fraction=1.0),)
    trainer.check_every = 0
    trainer.checkpoint_every = 10_000
    trainer.balance_config = BalanceConfig(every=1, warmup_steps=0)
    trainer.balancer.config = trainer.balance_config
    trainer.run(max_updates=2, verbose=False)
    trainer.refine(iterations=1, verbose=False)

    recorded = trainer.refine_history[0]["weights"]
    assert set(recorded) >= {"data", "initial", "contact"}
    assert recorded == trainer.balancer.weights(), (
        "精修记录的权重与实际用的不是同一套"
    )


def test_refinement_is_refused_while_a_stage_is_blocked(monkeypatch, tmp_path):
    """在"判据没过、不允许推进"的状态上精修，等于绕过那道闸门。"""
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 0
    trainer.tolerances = {"initial_relative_l2": 0.0, "initial_gap_relative": 0.0}
    from spherical_blast_pinn.training.trainer import StageNotPassed

    with pytest.raises(StageNotPassed):
        trainer.run(verbose=False)
    with pytest.raises(TrainingError, match="blocked"):
        trainer.refine(iterations=1, verbose=False)


# ==========================================================================
# §8 非有限损失/梯度的更新前检查
# ==========================================================================


def test_a_non_finite_gradient_blocks_the_update(monkeypatch, tmp_path):
    """**参数更新之前**识别非有限梯度，阻止该次更新并定位参数。"""
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 0

    before = {
        name: value.clone() for name, value in trainer.model.state_dict().items()
    }

    real_loss = trainer.__class__._update

    def poisoned(self):
        # 让其中一个参数拿到 NaN 梯度：不改 ``_update`` 的结构，只污染反向传播
        # 经过的那条路径。
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: grad * float("nan"))
                break
        return real_loss(self)

    monkeypatch.setattr(trainer.__class__, "_update", poisoned)
    with pytest.raises(NonFiniteUpdate) as error:
        trainer.run(max_updates=1, verbose=False)

    assert error.value.bad_parameters, "必须报出是哪个参数"
    assert error.value.step == 1
    for name, value in before.items():
        assert torch.equal(value, trainer.model.state_dict()[name]), (
            f"{name} 在更新被拦截后仍被改动了"
        )


def test_the_divergence_stops_the_run_instead_of_being_absorbed(monkeypatch, tmp_path):
    """异常必须**终止运行**，不能被 ``nan_to_num`` / 过滤 / 裁剪吸收掉。

    这条用行为而不是源码字面检查：一个被掩盖的异常会表现为"运行继续跑完了、
    损失是 nan"，而正确实现会抛错并且**不再往下走**。
    """
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 0
    real_loss = trainer.__class__._update

    def poisoned(self):
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: grad * float("nan"))
                break
        return real_loss(self)

    monkeypatch.setattr(trainer.__class__, "_update", poisoned)
    with pytest.raises(NonFiniteUpdate):
        trainer.run(max_updates=3, verbose=False)
    # 运行停在第一步：没有被"跳过这一步"继续跑完
    assert trainer.step == 0
    assert all(
        bool(torch.isfinite(value).all())
        for value in trainer.model.state_dict().values()
    ), "模型里留下了 NaN——异常被吸收进了参数"
