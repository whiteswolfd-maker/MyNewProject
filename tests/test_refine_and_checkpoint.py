"""精修的事务性、终态一致性与检查点完整性（提示词 §2、§3、§4、§6）。

四条被复核复现过的行为，逐条钉住：

* **§2** 一次 NaN 梯度能让 L-BFGS 把模型污染成 NaN 并写出含 NaN 的 ``last.pt``
  ——Adam 的守卫没有覆盖精修；
* **§3** 返回值与清单是 ``diagnostic``，``last.pt.run_status`` 却是 ``completed``，
  清单的 ``metadata.diagnostic`` 还是 ``None``，检查点里没有精修历史；
* **§4** 精修改了参数但收尾复检被 ``self.step > start_step`` 挡住，最后一条检查
  记的还是精修前的指纹；
* **§6** 最后一次 Adam 更新的损失没进检查点（加载后零更新得到
  ``final_loss=nan, final_loss_step=0``）、同名阶段换个预算/学习率照样加载、
  删掉全局 RNG 记录照样加载、回滚会在窗口历史里留下一件从未发生过的事。

所有这些都走真实调用路径：真实 ``run()``/``refine()``/``load()``，真实落盘文件。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.training.balancing import BalanceConfig
from spherical_blast_pinn.training.trainer import (
    NonFiniteUpdate,
    RUN_COMPLETED,
    RUN_DIAGNOSTIC,
    TrainingError,
)


def _tiny(monkeypatch, tmp_path: Path, name: str = "w"):
    import test_review_regressions as base

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
    trainer.check_every = 1
    trainer.checkpoint_every = 10_000
    trainer.balance_config = BalanceConfig(every=1, warmup_steps=0)
    trainer.balancer.config = trainer.balance_config
    return trainer, tiny


def _total(tiny) -> int:
    return sum(stage.updates for stage in tiny)


def _inject_non_finite_gradient(trainer):
    """给第一个可训练参数挂一个把梯度改成 NaN 的 hook，返回移除函数。"""
    parameter = next(p for p in trainer.model.parameters() if p.requires_grad)
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    return handle.remove


class _Criteria:
    """把阶段判据换成空集：这一组测的是状态机与事务，不是物理闸门。

    **它不能证明任何物理判据通过**，正式入口也不借用它。
    """

    def __init__(self, monkeypatch, result=()):
        from spherical_blast_pinn.training import trainer as module

        monkeypatch.setattr(module, "criteria_for", lambda name: result)


# ==========================================================================
# §2 L-BFGS 的非有限保护与失败回退
# ==========================================================================


def test_a_non_finite_gradient_blocks_refinement_and_leaves_the_model_alone(
    monkeypatch, tmp_path
):
    """**复核的原反例。** 先做一次正常 Adam 更新，再注入 NaN 梯度并精修。

    修前：精修不拦截，模型参数出现 NaN，随后写出的 ``last.pt`` 也含 NaN。
    修后：抛出 ``NonFiniteUpdate``，参数**一个都没变**，检查点保持有效。
    """
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    before = {
        name: value.detach().clone()
        for name, value in trainer.model.state_dict().items()
    }
    checkpoint_before = (trainer.output_dir / "last.pt").read_bytes()

    remove = _inject_non_finite_gradient(trainer)
    try:
        with pytest.raises(NonFiniteUpdate) as caught:
            trainer.refine(iterations=1, verbose=False)
    finally:
        remove()

    bad = [
        name for name, value in trainer.model.state_dict().items()
        if not bool(torch.isfinite(value).all())
    ]
    assert bad == [], f"参数被污染成非有限值: {bad}"
    for name, value in before.items():
        assert torch.equal(value, trainer.model.state_dict()[name])
    assert (trainer.output_dir / "last.pt").read_bytes() == checkpoint_before, (
        "失败的精修覆盖了最后一个有效检查点"
    )
    # 原因要留住：参数名、精修编号与 closure 次数都在异常里。
    assert "refinement #1" in caught.value.detail
    assert "closure #1" in caught.value.detail
    assert caught.value.bad_parameters


def test_a_non_finite_loss_blocks_refinement_before_backward(monkeypatch, tmp_path):
    """损失本身非有限时，在反传**之前**就停下。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    before = trainer.parameter_fingerprint()

    from spherical_blast_pinn.training import trainer as module

    real_loss = module.total_loss

    def exploding(*args, **kwargs):
        breakdown = real_loss(*args, **kwargs)
        breakdown.total = breakdown.total * float("nan")
        return breakdown

    monkeypatch.setattr(module, "total_loss", exploding)
    with pytest.raises(NonFiniteUpdate, match="loss is not finite"):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.undo()
    assert trainer.parameter_fingerprint() == before


def test_a_failure_after_the_line_search_moved_the_parameters_is_rolled_back(
    monkeypatch, tmp_path
):
    """**只在闭包里抛错不足以保证模型还原。** L-BFGS 的线搜索会临时改参数。

    这里让闭包正常跑完（参数确实被移动过），再让"在被接受的参数上求值"这一步
    失败——回滚必须把参数、步数、检查历史、采样器窗口与随机状态一并还原。
    """
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=2, verbose=False)

    before_fingerprint = trainer.parameter_fingerprint()
    before_step = trainer.step
    before_checks = len(trainer.check_history)
    before_history = list(trainer.history)
    before_window = [s.window_fraction for s in trainer.sampler.samplers]
    before_rng = [s.rng_state() for s in trainer.sampler.samplers]

    from spherical_blast_pinn.training import trainer as module

    real_loss = module.total_loss
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        # 闭包第一次调用照常；之后的调用（也就是"在被接受的参数上求值"）
        # 让目标变成 NaN，触发校验失败。
        if calls["n"] > 1:
            breakdown = real_loss(*args, **kwargs)
            breakdown.total = breakdown.total * float("nan")
            return breakdown
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(module, "total_loss", flaky)
    with pytest.raises(NonFiniteUpdate):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.undo()

    assert trainer.parameter_fingerprint() == before_fingerprint, (
        "线搜索已经把参数挪走了，回滚没有还原"
    )
    assert trainer.step == before_step
    assert len(trainer.check_history) == before_checks
    assert trainer.history == before_history
    assert [s.window_fraction for s in trainer.sampler.samplers] == before_window
    assert [s.rng_state() for s in trainer.sampler.samplers] == before_rng
    assert trainer.refine_history == [], "失败的精修不该出现在精修历史里"


def test_a_failed_refinement_writes_a_separate_diagnostics_file(monkeypatch, tmp_path):
    """失败诊断单独成文件，**不**写进可恢复检查点。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    checkpoint_before = (trainer.output_dir / "last.pt").read_bytes()

    remove = _inject_non_finite_gradient(trainer)
    try:
        with pytest.raises(NonFiniteUpdate):
            trainer.refine(iterations=1, verbose=False)
    finally:
        remove()

    path = trainer.output_dir / "refinement_failures.json"
    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8"))[-1]
    assert record["parameters_restored"] is True
    assert record["fingerprint_restored"] is True
    assert record["closures"] >= 1
    assert record["objective"]["optimizer_state_checkpointed"] is False
    assert (trainer.output_dir / "last.pt").read_bytes() == checkpoint_before


def test_a_successful_refinement_records_who_it_really_changed(monkeypatch, tmp_path):
    """成功路径：指纹前后不同、``parameters_changed`` 由比较产生、权重被记录。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=2, verbose=False)
    result = trainer.refine(iterations=2, verbose=False)

    record = trainer.refine_history[-1]
    assert record["fingerprint_before"] != record["fingerprint_after"]
    assert record["parameters_changed"] is True
    assert record["weights"] == {k: float(v) for k, v in trainer.balancer.weights().items()}
    assert record["closures"] >= 1
    assert record["iterations_requested"] == 2
    # 最终目标值取自被接受的参数，不是线搜索最后一次试探值。
    assert record["loss_source"].startswith("evaluated on the parameters the line search")
    assert np.isfinite(record["loss_on_accepted_parameters"])
    assert result.status == RUN_DIAGNOSTIC


def test_resuming_an_interrupted_refinement_is_refused_explicitly(monkeypatch, tmp_path):
    """§3.5：三种操作必须分开，第三种**明确拒绝**而不是只写一行说明。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    with pytest.raises(TrainingError, match="not supported"):
        trainer.resume_refinement()


# ==========================================================================
# §3 精修终态在返回值、清单与检查点里一致
# ==========================================================================


def test_the_refinement_identity_is_the_same_in_every_artifact(monkeypatch, tmp_path):
    """**复核的原反例。** 修前：返回值 ``diagnostic``、``last.pt.run_status``
    ``completed``、清单 ``metadata.diagnostic`` ``None``、检查点没有精修历史。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    result = trainer.run(max_updates=_total(tiny), verbose=False)
    assert result.status == RUN_COMPLETED, "基线：没有精修时应当是 completed"

    result = trainer.refine(iterations=1, verbose=False)
    manifest = json.loads(
        (trainer.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )

    assert result.status == RUN_DIAGNOSTIC
    assert manifest["status"] == RUN_DIAGNOSTIC
    assert manifest["diagnostic"] is True
    assert manifest["refined"] is True
    assert manifest["refinement_count"] == 1
    assert checkpoint["run_status"] == RUN_DIAGNOSTIC
    assert len(checkpoint["refinement"]) == 1
    # 判据通过的事实仍然保留——"判据通过"与"正式接受"是两件事。
    assert manifest["final_stage_verdict_passed"] is True
    assert manifest["formally_accepted"] is False
    assert result.formally_accepted is False


def test_a_new_instance_loading_a_refined_checkpoint_knows_it_was_refined(
    monkeypatch, tmp_path
):
    """§3.2：精修身份随模型来源持久化，一次默认 ``run()`` 不能把它洗掉。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)
    trainer.refine(iterations=1, verbose=False)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh")
    fresh.load(trainer.output_dir / "last.pt")
    assert len(fresh.refine_history) == 1
    assert fresh.refine_history[0]["parameters_changed"] is True

    # 再跑一次（零更新）也不能变成"未经精修的正式完成结果"。
    again = fresh.run(max_updates=fresh.step, verbose=False)
    assert again.status == RUN_DIAGNOSTIC
    assert again.metadata["refined"] is True or again.metadata["diagnostic"] is True


def test_a_truncated_then_refined_run_is_diagnostic(monkeypatch, tmp_path):
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    result = trainer.refine(iterations=1, verbose=False)
    assert result.status == RUN_DIAGNOSTIC
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["run_status"] == RUN_DIAGNOSTIC


def test_an_explicit_diagnostic_run_stays_diagnostic_through_save_and_reload(
    monkeypatch, tmp_path
):
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    result = trainer.run(max_updates=_total(tiny), verbose=False, diagnostic=True)
    assert result.status == RUN_DIAGNOSTIC
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["run_status"] == RUN_DIAGNOSTIC
    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_diag")
    fresh.load(trainer.output_dir / "last.pt")
    assert fresh._final_status == RUN_DIAGNOSTIC


# ==========================================================================
# §4 参数变化后的复检
# ==========================================================================


def test_refinement_triggers_a_full_recheck_on_the_new_parameters(monkeypatch, tmp_path):
    """**复核的原反例。** 修前：模型参数变了，但新增检查数是 0，
    最后记录的参数指纹仍是精修前的。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    checks_before = len(trainer.check_history)

    trainer.refine(iterations=1, verbose=False)

    assert len(trainer.check_history) > checks_before, "精修之后没有重查"
    last = trainer.check_history[-1]
    assert last["parameter_fingerprint"] == trainer.parameter_fingerprint(), (
        "最后一条检查记录的不是当前参数"
    )
    assert last["scope"] == "full", "复检必须是完整检查集，不是窗口检查"


def test_a_zero_budget_call_does_not_recheck(monkeypatch, tmp_path):
    """对照组：没动过参数的调用不该重查——那只会往 ``best.pt`` 里写一份
    "从未训练过"的快照。判据是**参数变没变**，不是步数涨没涨。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    checks_before = len(trainer.check_history)

    result = trainer.run(max_updates=trainer.step, verbose=False)
    assert len(trainer.check_history) == checks_before
    assert result.metadata["check_state"] == "no_parameter_change_in_this_call"
    assert result.metadata["check_state_meaning"]


def test_checks_disabled_is_reported_as_unchecked_not_as_passed(monkeypatch, tmp_path):
    """§4.4：显式关掉检查时状态必须如实标记未检查。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.check_every = 0
    result = trainer.run(max_updates=_total(tiny), verbose=False)
    assert result.metadata["check_state"] == "checks_disabled"
    assert result.status != RUN_COMPLETED


def test_a_windowed_check_does_not_substitute_for_a_full_one(monkeypatch, tmp_path):
    """§4.3：窗口检查与全时段检查回答的不是同一个问题，不能靠"步数和指纹相同"顶替。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    trainer.check_history[-1]["scope"] = "windowed"
    assert trainer._check_stale() is True


def test_a_check_from_another_check_set_is_stale(monkeypatch, tmp_path):
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    trainer.check_history[-1]["check_set_identity"] = "deadbeefdeadbeef"
    assert trainer._check_stale() is True


def test_best_may_keep_an_earlier_model_and_the_new_one_still_competes(
    monkeypatch, tmp_path
):
    """§4.5：best 允许保留历史上更好的模型，不能强制 ``best`` 指纹等于 ``last``。

    这里不假定谁更好——只要求**新模型确实参与了比较**：两条分支各跑一次，
    每次都核对 best 里的指纹要么是新的、要么是旧的，且 best 文件与其记录一致。
    """
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=1, verbose=False)
    trainer.refine(iterations=1, verbose=False)

    best = trainer.best
    assert best is not None
    current = trainer.parameter_fingerprint()
    assert best["parameter_fingerprint"] in {current} | {
        record["parameter_fingerprint"] for record in trainer.check_history[:-1]
    }
    loaded = torch.load(
        trainer.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    best_model = loaded["model"]
    digest = _fingerprint_of_state(best_model)
    assert digest == best["parameter_fingerprint"], (
        "best.pt 里的参数与 best 记录里的指纹不是同一个模型"
    )


def _fingerprint_of_state(state: dict) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


# ==========================================================================
# §6 检查点完整性、恢复兼容性与回滚
# ==========================================================================


def test_the_last_measured_loss_survives_a_checkpoint_round_trip(monkeypatch, tmp_path):
    """**复核的原反例。** 修前：加载后不再更新，结果变成
    ``final_loss=NaN, final_loss_step=0``——看起来像发散，其实是记录丢了。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)
    assert np.isfinite(trainer._last_update_loss)
    assert trainer._last_update_loss_step == _total(tiny)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_loss")
    fresh.load(trainer.output_dir / "last.pt")
    result = fresh.run(max_updates=fresh.step, verbose=False)
    assert np.isfinite(result.final_loss)
    assert result.final_loss == pytest.approx(trainer._last_update_loss)
    assert result.final_loss_step == _total(tiny)


def test_a_checkpoint_records_model_data_and_check_set_identity(monkeypatch, tmp_path):
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["schedule"]
    assert checkpoint["check_set"]
    assert checkpoint["data_identity"]
    assert checkpoint["last_update_loss"] == pytest.approx(trainer._last_update_loss)
    assert checkpoint["last_update_loss_step"] == trainer._last_update_loss_step


@pytest.mark.parametrize("change", ["budget", "learning_rate", "pde_weight"])
def test_a_different_schedule_is_refused_even_when_the_window_matches(
    monkeypatch, tmp_path, change
):
    """**复核的原反例。** 同名 ``joint`` 阶段把预算从 2 改成 5、学习率与 PDE 权重
    一起改掉，只要那一刻的窗口相同，修前照常恢复。"""
    from spherical_blast_pinn.training.schedule import Stage

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)

    def mutate(stage):
        return Stage(
            stage.name,
            stage.updates + 3 if change == "budget" else stage.updates,
            stage.pde_weight_start * 2.0 if change == "pde_weight" else stage.pde_weight_start,
            stage.pde_weight_end * 2.0 if change == "pde_weight" else stage.pde_weight_end,
            stage.learning_rate_start * 5.0 if change == "learning_rate" else stage.learning_rate_start,
            stage.learning_rate_end * 5.0 if change == "learning_rate" else stage.learning_rate_end,
            time_window_fraction=stage.time_window_fraction,
        )

    fresh, _ = _tiny(monkeypatch, tmp_path, name=f"fresh_{change}")
    fresh.schedule = tuple(mutate(stage) for stage in tiny)
    with pytest.raises(TrainingError, match="different schedule"):
        fresh.load(trainer.output_dir / "last.pt")


def test_a_checkpoint_without_the_global_rng_record_is_refused(monkeypatch, tmp_path):
    """**复核的原反例。** 删掉全局 ``rng`` 记录，修前仍能加载。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    stripped = {key: value for key, value in checkpoint.items() if key != "rng"}
    path = tmp_path / "no_rng.pt"
    torch.save(stripped, path)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_rng")
    with pytest.raises(TrainingError, match="global RNG state"):
        fresh.load(path)


def test_a_partial_global_rng_record_is_refused(monkeypatch, tmp_path):
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    checkpoint["rng"] = {"seed": 0, "python": None, "numpy": None, "torch": None}
    path = tmp_path / "partial_rng.pt"
    torch.save(checkpoint, path)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_prng")
    with pytest.raises(TrainingError, match="partial record"):
        fresh.load(path)


def test_a_checkpoint_without_the_last_update_loss_is_refused(monkeypatch, tmp_path):
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    del checkpoint["last_update_loss"]
    path = tmp_path / "no_loss.pt"
    torch.save(checkpoint, path)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_loss2")
    with pytest.raises(TrainingError, match="last_update_loss"):
        fresh.load(path)


def test_an_old_checkpoint_version_is_refused_with_a_migration_note(monkeypatch, tmp_path):
    """§6.7：结构升级要升版本，并说明**没有**迁移路径而不是硬读。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    checkpoint["version"] = 2
    path = tmp_path / "old_version.pt"
    torch.save(checkpoint, path)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_ver")
    with pytest.raises(TrainingError, match="no migration path"):
        fresh.load(path)


def test_a_rollback_does_not_invent_a_window_history_event(monkeypatch, tmp_path):
    """**复核的原反例。** 提交中途失败后，窗口值恢复了，但 ``_window_history``
    从 ``[1.0, 0.25]`` 变成 ``[1.0, 0.25, 1.0]``——一次从未生效过的变化被记进了
    历史。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_rollback")
    fresh.sampler.samplers[0].set_time_window(0.25)
    before = fresh.sampler.samplers[0].window_history

    original = fresh.balancer.load_state_dict
    calls = {"n": 0}

    def once(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected: the balancer fails mid-commit")
        return original(state)

    fresh.balancer.load_state_dict = once
    with pytest.raises(RuntimeError, match="injected"):
        fresh.load(trainer.output_dir / "last.pt")
    fresh.balancer.load_state_dict = original

    assert fresh.sampler.samplers[0].window_history == before
    assert fresh.sampler.samplers[0].window_fraction == pytest.approx(0.25)


def test_a_failed_rollback_marks_the_instance_unusable(monkeypatch, tmp_path):
    """§6.5：回滚也失败时，对象处于未知状态——**不能**当成还能继续用。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_poison")
    original = fresh.balancer.load_state_dict

    def always_fails(_state):
        raise RuntimeError("injected: the balancer never loads")

    fresh.balancer.load_state_dict = always_fails
    with pytest.raises(TrainingError, match="unknown state"):
        fresh.load(trainer.output_dir / "last.pt")
    fresh.balancer.load_state_dict = original

    with pytest.raises(TrainingError, match="unknown state"):
        fresh.run(max_updates=fresh.step)
    with pytest.raises(TrainingError, match="unknown state"):
        fresh._save(tmp_path / "never.pt")


def test_resuming_matches_a_continuous_run_across_a_stage_boundary(monkeypatch, tmp_path):
    """§6.2/6.3：连续跑与"断在阶段边界再恢复"必须逐位相同，包括采样器状态。"""
    import test_review_regressions as base

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    total = _total(tiny)
    continuous = trainer.run(max_updates=total, verbose=False)
    assert continuous.updates == total

    interrupted, _ = _tiny(monkeypatch, tmp_path, name="interrupted")
    interrupted.run(max_updates=tiny[0].updates, verbose=False)
    boundary = interrupted.output_dir / "boundary.pt"
    interrupted._save(boundary)

    resumed, _ = _tiny(monkeypatch, tmp_path, name="resumed")
    resumed.load(boundary)
    resumed.run(max_updates=total, verbose=False)

    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name])
    assert resumed.sampler.rng_state() == trainer.sampler.rng_state()
    assert resumed.step == trainer.step
    assert resumed._last_update_loss_step == trainer._last_update_loss_step
    assert resumed._current_stage == trainer._current_stage


def test_a_blocked_stage_can_be_resumed_and_re_evaluated(monkeypatch, tmp_path):
    """§6.6：失败阶段仍必须能恢复，并且恢复后仍然被闸门拦下。"""
    import test_review_regressions as base
    from spherical_blast_pinn.training import trainer as module

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=tiny[0].updates - 1, verbose=False)
    trainer._save(trainer.output_dir / "before_block.pt")

    # 让第一阶段判据必然不过，跑完它。
    from spherical_blast_pinn.training.schedule import StageCriterion

    monkeypatch.setattr(
        module, "criteria_for",
        lambda name: (StageCriterion("pde_rms_max", "max", fallback=-1.0),),
    )
    with pytest.raises(Exception):
        trainer.run(max_updates=tiny[0].updates, verbose=False)

    assert trainer._stage_blocked is True
    path = trainer.output_dir / "blocked.pt"
    trainer._save(path)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_blocked")
    fresh.load(path)
    assert fresh._stage_blocked is True
    assert fresh._pending_stage == tiny[0].name
    # 恢复之后仍然被拦下：判据没过就不许推进。
    with pytest.raises(Exception):
        fresh.run(max_updates=sum(s.updates for s in tiny), verbose=False)
    # 精修也要被拦住。
    with pytest.raises(TrainingError, match="blocked"):
        fresh.refine(iterations=1, verbose=False)


def test_moving_the_output_directory_does_not_change_data_identity(monkeypatch, tmp_path):
    """§6.3：不要靠目录字符串相同来证明"还是同一份数据"。

    把检查点搬到另一个目录再恢复，必须照常工作；把**同一路径下的数据**换掉，
    必须拒绝。
    """
    import shutil

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)

    moved = tmp_path / "moved" / "last.pt"
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trainer.output_dir / "last.pt", moved)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_moved")
    fresh.load(moved)
    assert fresh.step == trainer.step

    # 同一份数据、不同的 case_id 标签也不行——身份是按算例记的。
    #
    # 第 3 轮起这条拒绝由**逐算例关联**先说出口（"checkpoint is for cases [...],
    # but the receiving trainer holds [...]"），比"数据身份不同"更具体；两者都
    # 是拒绝，所以这里接受任一诊断。
    renamed, _ = _tiny(monkeypatch, tmp_path, name="fresh_renamed")
    renamed.cases[0].case_id = "renamed"
    with pytest.raises(
        TrainingError, match="different data|resuming across different cases"
    ):
        renamed.load(moved)


def test_changing_the_data_behind_the_same_path_is_refused(monkeypatch, tmp_path):
    """同一目录字符串下换了剖面：数据身份必须变。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path)
    trainer.run(max_updates=_total(tiny), verbose=False)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="fresh_data")
    fresh.cases[0].times = np.asarray(fresh.cases[0].times) * 1.5
    with pytest.raises(TrainingError, match="different data"):
        fresh.load(trainer.output_dir / "last.pt")


# ==========================================================================
# 第七轮 §1 精修的**整条流程**都在事务里
# ==========================================================================


def _state_fingerprint(trainer):
    """参数 + 采样器随机状态 + 历史长度 + 检查点字节，一次取齐。"""
    best = trainer.output_dir / "best.pt"
    return {
        "parameters": trainer.parameter_fingerprint(),
        "sampler_rng": [s.rng_state() for s in trainer.sampler.samplers],
        "window": [s.window_fraction for s in trainer.sampler.samplers],
        "refine_history": len(trainer.refine_history),
        "check_history": len(trainer.check_history),
        "last_pt": (trainer.output_dir / "last.pt").read_bytes(),
        "best_pt": best.read_bytes() if best.exists() else None,
    }


@pytest.mark.parametrize("where", ["sampling", "optimisation", "recheck", "save"])
def test_a_refinement_failure_at_any_stage_rolls_the_whole_flow_back(
    monkeypatch, tmp_path, where
):
    """**复核的原反例（收尾 ``verify()`` 抛错）。**

    修前：``try`` 只罩住"线搜索 + 最终求值"，于是收尾的 ``verify`` 抛错会留下
    —— 参数没恢复、``refine_history`` 多一条、没有失败诊断文件、内存模型与
    ``last.pt`` 不一致。

    这里把失败注入在**四个不同阶段**：采样、优化、复检、写盘。
    """
    import torch

    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path, name="roll_" + where)
    trainer.run(max_updates=1, verbose=False)
    before = _state_fingerprint(trainer)

    if where == "sampling":
        def boom(*_a, **_k):
            raise RuntimeError("injected: sampling")
        monkeypatch.setattr(trainer.sampler, "batch", boom)
    elif where == "optimisation":
        def boom(*_a, **_k):
            raise RuntimeError("injected: L-BFGS step")
        monkeypatch.setattr(torch.optim.LBFGS, "step", boom)
    elif where == "recheck":
        def boom(*_a, **_k):
            raise RuntimeError("injected: the post-refinement re-check")
        monkeypatch.setattr(trainer, "verify", boom)
    else:
        def boom(*_a, **_k):
            raise RuntimeError("injected: writing the checkpoint")
        monkeypatch.setattr(trainer, "_save", boom)

    with pytest.raises(Exception):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.undo()

    after = _state_fingerprint(trainer)
    assert after["parameters"] == before["parameters"], where + ": 参数没有恢复"
    assert after["sampler_rng"] == before["sampler_rng"], where + ": 采样序列没有恢复"
    assert after["window"] == before["window"]
    assert after["refine_history"] == before["refine_history"], where + ": 精修历史多了记录"
    assert after["check_history"] == before["check_history"]
    assert after["last_pt"] == before["last_pt"], where + ": last.pt 被改了"
    assert after["best_pt"] == before["best_pt"], where + ": best.pt 被改了"

    diagnostics = trainer.output_dir / "refinement_failures.json"
    assert diagnostics.exists(), where + ": 没有失败诊断文件"
    record = json.loads(diagnostics.read_text(encoding="utf-8"))[-1]
    assert record["stage"] == {
        "sampling": "sampling",
        "optimisation": "optimisation",
        "recheck": "recheck and publish",
        "save": "recheck and publish",
    }[where]
    assert record["parameters_restored"] is True
    assert record["artifacts_restored"] is True
    # 失败之后实例仍然可用（回滚成功了）。
    trainer.run(max_updates=trainer.step, verbose=False)


def test_a_failed_rollback_during_refinement_marks_the_instance_unusable(
    monkeypatch, tmp_path
):
    """§1：回滚本身失败时，保留原始异常与回滚异常，并禁止继续使用。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path, name="roll_poison")
    trainer.run(max_updates=1, verbose=False)

    def boom(*_a, **_k):
        raise RuntimeError("injected: the re-check fails")

    original_restore = trainer._restore_state

    def also_boom(_snapshot):
        raise RuntimeError("injected: rolling back fails too")

    monkeypatch.setattr(trainer, "verify", boom)
    monkeypatch.setattr(trainer, "_restore_state", also_boom)
    with pytest.raises(TrainingError, match="unknown state"):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.setattr(trainer, "_restore_state", original_restore)
    monkeypatch.undo()

    with pytest.raises(TrainingError, match="unknown state"):
        trainer.run(max_updates=trainer.step)
    with pytest.raises(TrainingError, match="unknown state"):
        trainer._save(tmp_path / "never.pt")


# ==========================================================================
# 第七轮 §2 best.pt 的终态必须与选出它的那次运行一致
# ==========================================================================


def test_best_pt_carries_the_status_of_the_run_that_selected_it(monkeypatch, tmp_path):
    """**复核的原反例。** 修前：``best.pt`` 写 ``completed``、``last.pt`` 写
    ``diagnostic``，两份检查点的参数却完全相同。"""
    import torch

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="best_status")
    trainer.run(max_updates=_total(tiny), verbose=False)
    trainer.best = None                     # 让精修后的复检一定成为新的最佳
    trainer.refine(iterations=1, verbose=False)

    best = torch.load(
        trainer.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    last = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert best["run_status"] == RUN_DIAGNOSTIC
    assert last["run_status"] == RUN_DIAGNOSTIC
    assert best["run_origin"]["refined"] is True
    assert trainer.best["run_origin"]["refined"] is True


def test_a_kept_earlier_best_keeps_its_own_status(monkeypatch, tmp_path):
    """§2：保留旧最佳时**不重写** ``best.pt``——它自己的来源属于选出它的那次运行。"""
    import torch

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="best_kept")
    trainer.run(max_updates=_total(tiny), verbose=False)
    assert (trainer.output_dir / "best.pt").exists()

    # 把当前最佳设成**打不败**的（score 越小越好），于是精修之后的复检不会
    # 取代它 —— 这正是"保留旧最佳"那条分支。
    trainer.best = dict(trainer.best or {})
    trainer.best["score"] = [-1.0e9, -1.0e9, -1.0e9]
    trainer.best["parameter_fingerprint"] = "kept-earlier-model"
    trainer._best_dirty = False
    (trainer.output_dir / "best.pt").unlink()
    trainer._save(trainer.output_dir / "best.pt")
    best_before = (trainer.output_dir / "best.pt").read_bytes()
    status_before = torch.load(
        trainer.output_dir / "best.pt", map_location="cpu", weights_only=False
    )["run_status"]

    trainer.refine(iterations=1, verbose=False)

    assert trainer.best["parameter_fingerprint"] == "kept-earlier-model"
    assert (trainer.output_dir / "best.pt").read_bytes() == best_before, (
        "保留旧最佳时 best.pt 被重写了——它的来源属于选出它的那次运行"
    )
    kept = torch.load(
        trainer.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    assert kept["run_status"] == status_before


# ==========================================================================
# 第七轮 §3 显式诊断身份要跟着参数走
# ==========================================================================


def test_an_explicit_diagnostic_identity_survives_save_load_and_further_runs(
    monkeypatch, tmp_path
):
    """**复核的原反例。** 修前：``diagnostic=True`` 跑完存盘，新实例加载得到
    ``diagnostic``，但一次**零更新**的 ``run()`` 会把它算成 ``completed``、
    ``formally_accepted=True``。"""
    import torch

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="origin")
    trainer.run(max_updates=_total(tiny), verbose=False, diagnostic=True)
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["run_origin"]["explicit_diagnostic"] is True

    fresh, _ = _tiny(monkeypatch, tmp_path, name="origin_fresh")
    fresh.load(trainer.output_dir / "last.pt")
    assert fresh._run_origin["explicit_diagnostic"] is True

    zero = fresh.run(max_updates=fresh.step, verbose=False)
    assert zero.status == RUN_DIAGNOSTIC
    assert zero.formally_accepted is False

    # 继续更新也不行：身份跟着参数走，不跟"这一次调用"走。
    more = fresh.run(max_updates=fresh.step + 1, verbose=False)
    assert more.status == RUN_DIAGNOSTIC
    assert more.formally_accepted is False


def test_a_plain_run_keeps_its_normal_status(monkeypatch, tmp_path):
    """对照组：没有诊断来源的运行照常可以被判为完成。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="origin_plain")
    result = trainer.run(max_updates=_total(tiny), verbose=False)
    assert result.status == RUN_COMPLETED
    assert result.metadata["run_origin"] == {
        "explicit_diagnostic": False, "refined": False
    }
    assert result.metadata["diagnostic"] is False


def test_a_checkpoint_without_a_run_origin_is_refused(monkeypatch, tmp_path):
    import torch

    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="origin_missing")
    trainer.run(max_updates=_total(tiny), verbose=False)
    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    del checkpoint["run_origin"]
    path = tmp_path / "no_origin.pt"
    torch.save(checkpoint, path)
    fresh, _ = _tiny(monkeypatch, tmp_path, name="origin_missing2")
    with pytest.raises(TrainingError, match="run_origin"):
        fresh.load(path)


# ==========================================================================
# 第七轮 §4 数据身份必须覆盖真正参与训练的数据
# ==========================================================================


@pytest.mark.parametrize(
    "mutate",
    [
        lambda case: setattr(case, "density", np.asarray(case.density) * 2.0),
        lambda case: setattr(case, "velocity", np.asarray(case.velocity) + 123.0),
        lambda case: setattr(case, "pressure", np.asarray(case.pressure) * 0.5),
        lambda case: setattr(
            case, "specific_energy", np.asarray(case.specific_energy) * 0.5
        ),
        # 整体轮转 0→1→2→0：每个单元都变，且仍在合法取值域里。
        lambda case: setattr(
            case, "material", (np.asarray(case.material) + 1) % 3
        ),
        lambda case: setattr(case, "valid", ~np.asarray(case.valid)),
    ],
)
def test_a_changed_field_is_refused_by_the_data_identity(monkeypatch, tmp_path, mutate):
    """**复核的原反例。** 修前只哈希坐标轴、前沿与材料码的**计数**：把密度乘 2、
    速度加 123，哈希一字不变，旧检查点照常加载。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="data")
    trainer.run(max_updates=_total(tiny), verbose=False)

    fresh, _ = _tiny(monkeypatch, tmp_path, name="data_fresh")
    mutate(fresh.cases[0])
    with pytest.raises(TrainingError, match="different data"):
        fresh.load(trainer.output_dir / "last.pt")


def test_missing_and_empty_are_different_in_the_data_identity(monkeypatch, tmp_path):
    """§4：``None`` 与"空数组"必须分开——"没记录逐字段有效性"和"全 False"是两件事。"""
    _Criteria(monkeypatch)
    trainer, _ = _tiny(monkeypatch, tmp_path, name="id_missing")
    case = trainer.cases[0]
    case.field_validity = None
    identity_missing = trainer._data_identity()[case.case_id]["content_sha256"]
    case.field_validity = {}
    identity_empty = trainer._data_identity()[case.case_id]["content_sha256"]
    assert identity_missing != identity_empty


def test_the_data_identity_ignores_the_case_id_label(monkeypatch, tmp_path):
    """身份按 case_id 索引，所以改名要被拒绝（不是同一份数据的同一个位置）。"""
    _Criteria(monkeypatch)
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="id_label")
    trainer.run(max_updates=_total(tiny), verbose=False)
    fresh, _ = _tiny(monkeypatch, tmp_path, name="id_label2")
    fresh.cases[0].case_id = "renamed"
    # 第 3 轮起逐算例关联先报出更具体的原因（见上一条测试的说明）。
    with pytest.raises(
        TrainingError, match="different data|resuming across different cases"
    ):
        fresh.load(trainer.output_dir / "last.pt")
