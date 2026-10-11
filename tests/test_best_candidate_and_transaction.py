"""最佳候选、事务完整性、内存/磁盘恢复的独立性（第八轮 §1、§2、§3）。

三条被复核复现过的行为：

* **§1** 中途选出的最佳，发布时写的是**末步**参数——``best_check.step = 1``
  而 ``best.pt`` 里的 ``step = 6``，且 ``best_check.parameter_fingerprint``
  与文件里的实际参数对不上。
* **§2** 事务的磁盘清单只有两份检查点：``checks.json`` 已经发布、
  ``run_manifest.json`` 才失败时，``checks.json`` 与 ``_final_check_state``
  都没有恢复。
* **§3** 内存恢复与磁盘恢复写在一个 ``try`` 里：内存恢复一失败，磁盘恢复
  根本不执行，本次失败精修写出的 ``last.pt`` 留在盘上，**另一个实例照样能
  加载成功**。

所有测试都走真实调用路径：真实参数更新、真实候选快照、真实落盘。控制评分与
注入故障只用于安排顺序，**不代表物理验收通过**。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.evaluation.checks import CheckReport
from spherical_blast_pinn.training.balancing import BalanceConfig
from spherical_blast_pinn.training.trainer import (
    RUN_COMPLETED,
    RUN_DIAGNOSTIC,
    TrainingError,
)
from spherical_blast_pinn.training import trainer as trainer_module


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
    monkeypatch.setattr(trainer_module, "criteria_for", lambda name: ())
    return trainer, tiny


def _total(tiny) -> int:
    return sum(stage.updates for stage in tiny)


def _fingerprint(state: dict) -> str:
    """**独立**算一遍参数指纹，不调被测对象的方法。"""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _score_at(step: int):
    """按步号给分：用来把"最佳"安排在指定的一步。"""
    def fake(model, check_set, **kwargs):
        recorded = int(kwargs.get("steps", 0))
        value = 0.0 if recorded == step else float(recorded)
        return CheckReport(step=recorded, metrics={
            "nonphysical_violation_count": 0.0,
            "physics_violation_score": value,
            "development_prediction_error": value,
            "total_loss": value,
        })
    return fake


def _record_every_step(trainer) -> dict[int, str]:
    """每一步更新**之后**的参数指纹。``_update()` 在 ``step += 1`` 之前跑，
    所以记的是 ``step + 1``。"""
    per_step: dict[int, str] = {}
    original = trainer._update

    def recording():
        breakdown = original()
        per_step[int(trainer.step) + 1] = trainer.parameter_fingerprint()
        return breakdown

    trainer._update = recording
    return per_step


# ==========================================================================
# §1 最佳候选必须保存入选时的状态
# ==========================================================================


def test_a_mid_run_best_is_published_with_its_own_parameters(monkeypatch, tmp_path):
    """**复核的原反例。** 第 1 步最好，继续训到第 6 步。

    修前：``best_check.step = 1`` 而 ``best.pt`` 里的 ``step = 6``，
    ``best_check.parameter_fingerprint`` 与文件里的实际参数对不上。
    """
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(1))
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="mid_best")
    per_step = _record_every_step(trainer)
    trainer.run(max_updates=_total(tiny), verbose=False)

    best = torch.load(trainer.output_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(trainer.output_dir / "last.pt", map_location="cpu", weights_only=False)
    best_actual = _fingerprint(best["model"])
    last_actual = _fingerprint(last["model"])

    assert trainer.best["step"] == 1
    assert best["step"] == 1, "best.pt 存的是末步参数"
    assert best_actual == per_step[1], "best.pt 的参数不是第 1 步那一刻的参数"
    assert best_actual == trainer.best["parameter_fingerprint"], (
        "best_check 的指纹与 best.pt 里的实际参数不是同一组"
    )
    # last.pt 仍然是末步的。
    assert last["step"] == _total(tiny)
    assert last_actual == per_step[_total(tiny)]
    assert best_actual != last_actual


def test_the_last_step_becoming_best_keeps_everything_self_consistent(monkeypatch, tmp_path):
    """末步成为最佳：两份参数可以相同，但来源、指纹与状态仍要自洽。"""
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(_total(_tiny(monkeypatch, tmp_path, "probe")[1])))
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="last_best")
    per_step = _record_every_step(trainer)
    trainer.run(max_updates=_total(tiny), verbose=False)

    best = torch.load(trainer.output_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(trainer.output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert _fingerprint(best["model"]) == _fingerprint(last["model"])
    assert best["step"] == last["step"] == _total(tiny)
    assert best["run_status"] == last["run_status"]
    assert best["run_origin"] == last["run_origin"]
    assert best["candidate"]["parameter_fingerprint"] == trainer.best["parameter_fingerprint"]


def test_a_candidate_snapshot_does_not_follow_later_parameter_updates(monkeypatch, tmp_path):
    """候选快照必须是**独立副本**：后续的原地更新不能改到它。"""
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(1))
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="snapshot")
    trainer.run(max_updates=1, verbose=False)

    assert trainer._best_candidate is not None
    captured = trainer._best_candidate["step"]
    frozen = _fingerprint(trainer._best_candidate["model"])
    assert captured == 1

    trainer.run(max_updates=_total(tiny), verbose=False)   # 继续原地更新参数
    assert trainer._best_candidate["step"] == captured
    assert _fingerprint(trainer._best_candidate["model"]) == frozen, (
        "候选快照跟着后续的原地更新变了——它是引用，不是副本"
    )
    assert trainer.parameter_fingerprint() != frozen


def test_an_interrupted_run_keeps_the_unpublished_candidate(monkeypatch, tmp_path):
    """周期保存 + 运行中断：恢复出来的实例仍能看到那个尚未发布的最佳候选。"""
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(1))
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="interrupted")
    trainer.checkpoint_every = 1
    # 真正模拟"跑到一半被打断"：第 1 步的 verify 选出候选、周期检查点把它写进
    # ``last.pt``（**发布块还没走到**），第 2 步的 verify 直接中断。
    # 正常跑完时 ``run()`` 会在收尾发布候选并把 ``_best_dirty`` 清掉——那样就
    # 测不到"尚未最终发布"这一档了。
    calls = {"n": 0}
    original_verify = trainer.verify

    def verify_then_abort(*args, **kwargs):
        report = original_verify(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt("simulated interruption")
        return report

    monkeypatch.setattr(trainer, "verify", verify_then_abort)
    with pytest.raises(KeyboardInterrupt):
        trainer.run(max_updates=2, verbose=False)
    monkeypatch.undo()

    assert trainer._best_candidate is not None
    candidate_fingerprint = _fingerprint(trainer._best_candidate["model"])
    assert trainer._best_dirty is True, "中断时候选还没发布"
    assert not (trainer.output_dir / "best.pt").exists()
    intermediate = trainer.output_dir / "last.pt"

    fresh, _ = _tiny(monkeypatch, tmp_path, name="interrupted_fresh")
    fresh.load(intermediate)
    assert fresh._best_candidate is not None, "候选随检查点一起丢了"
    assert _fingerprint(fresh._best_candidate["model"]) == candidate_fingerprint
    assert fresh._best_dirty is True
    assert fresh.best is not None and fresh.best["step"] == 1


def test_resuming_from_best_matches_resuming_from_an_independent_checkpoint(
    monkeypatch, tmp_path
):
    """§1.2：从 ``best.pt`` 继续，必须与"在候选那一刻独立存一份再继续"一致。"""
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(1))
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="resume_best")
    trainer.checkpoint_every = 1
    trainer.run(max_updates=1, verbose=False)
    independent = trainer.output_dir / "independent_at_candidate.pt"
    trainer._save(independent)
    trainer.run(max_updates=_total(tiny), verbose=False)

    from_best, _ = _tiny(monkeypatch, tmp_path, name="from_best")
    from_best.load(trainer.output_dir / "best.pt")
    from_best.run(max_updates=_total(tiny), verbose=False)

    from_independent, _ = _tiny(monkeypatch, tmp_path, name="from_independent")
    from_independent.load(independent)
    from_independent.run(max_updates=_total(tiny), verbose=False)

    for name, value in from_best.model.state_dict().items():
        assert torch.equal(value, from_independent.model.state_dict()[name]), name
    assert from_best.sampler.rng_state() == from_independent.sampler.rng_state()
    assert from_best.step == from_independent.step


def test_a_kept_earlier_best_is_not_rewritten(monkeypatch, tmp_path):
    """保留上一轮的旧最佳时，文件字节不变。"""
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(-1))   # 永不被超越
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="kept")
    trainer.run(max_updates=_total(tiny), verbose=False)
    best_before = (trainer.output_dir / "best.pt").read_bytes()
    status_before = torch.load(
        trainer.output_dir / "best.pt", map_location="cpu", weights_only=False
    )["run_status"]

    trainer.run(max_updates=trainer.step, verbose=False)
    assert (trainer.output_dir / "best.pt").read_bytes() == best_before
    assert torch.load(
        trainer.output_dir / "best.pt", map_location="cpu", weights_only=False
    )["run_status"] == status_before


@pytest.mark.parametrize("refine_wins", [True, False])
def test_refined_candidates_keep_or_yield_the_diagnostic_identity(
    monkeypatch, tmp_path, refine_wins
):
    """精修候选获选时保留诊断身份；未获选时不把旧最佳改成精修来源。"""
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(1))
    trainer, tiny = _tiny(monkeypatch, tmp_path, name=f"refine_best_{refine_wins}")
    trainer.run(max_updates=_total(tiny), verbose=False)

    if not refine_wins:
        # 让当前最佳打不败，于是精修之后的复检不会取代它。
        trainer.best = dict(trainer.best or {})
        trainer.best["score"] = [-1.0e9, -1.0e9, -1.0e9, -1.0e9]
        trainer.best["parameter_fingerprint"] = "kept-earlier-model"
        trainer._best_candidate = None
        trainer._best_dirty = False
        (trainer.output_dir / "best.pt").unlink()
        trainer._save(trainer.output_dir / "best.pt")
        before = (trainer.output_dir / "best.pt").read_bytes()
    else:
        trainer.best = None
        before = None

    trainer.refine(iterations=1, verbose=False)

    best = torch.load(trainer.output_dir / "best.pt", map_location="cpu", weights_only=False)
    if refine_wins:
        assert best["run_status"] == RUN_DIAGNOSTIC
        assert best["run_origin"]["refined"] is True
        assert best["candidate"]["selected_in_run_with_status"] == RUN_DIAGNOSTIC
    else:
        assert (trainer.output_dir / "best.pt").read_bytes() == before
        assert best["run_origin"]["refined"] is False
        assert best["run_status"] != RUN_DIAGNOSTIC or best.get("candidate") is None


# ==========================================================================
# §2 事务必须覆盖整套发布产物与相关内存状态
# ==========================================================================


def _block_manifest(trainer):
    """让 ``run_manifest.json`` 的写出失败——**前面的文件已经真的写完了**。

    直接把整个保存函数换成一进来就抛错是不行的：那样测不到"部分发布"。
    """
    real = pathlib.Path.write_text
    target = str((trainer.output_dir / "run_manifest.json").resolve())

    def guarded(self, *args, **kwargs):
        if str(Path(self).resolve()) == target:
            raise OSError("injected: run_manifest.json cannot be written")
        return real(self, *args, **kwargs)

    pathlib.Path.write_text = guarded
    return lambda: setattr(pathlib.Path, "write_text", real)


def _artifact_bytes(trainer) -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {}
    for name in ("last.pt", "best.pt", "checks.json", "run_manifest.json"):
        path = trainer.output_dir / name
        out[name] = path.read_bytes() if path.exists() else None
    return out


def test_a_failure_after_earlier_artifacts_were_published_rolls_all_of_them_back(
    monkeypatch, tmp_path
):
    """**复核的原反例。** ``checks.json`` 已经写出、``run_manifest.json`` 才失败。

    修前：``checks.json`` 未恢复、``_final_check_state`` 未恢复。
    """
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="late_fail")
    trainer.run(max_updates=1, verbose=False)
    before = _artifact_bytes(trainer)
    before_state = trainer._final_check_state
    before_fingerprint = trainer.parameter_fingerprint()
    before_history = len(trainer.history)
    before_refine = len(trainer.refine_history)

    restore = _block_manifest(trainer)
    with pytest.raises(OSError, match="run_manifest"):
        trainer.refine(iterations=1, verbose=False)
    restore()

    after = _artifact_bytes(trainer)
    for name, payload in before.items():
        if payload is not None:
            assert after[name] == payload, f"{name} 没有恢复原字节"
    assert trainer._final_check_state == before_state
    assert trainer.parameter_fingerprint() == before_fingerprint
    assert len(trainer.history) == before_history
    assert len(trainer.refine_history) == before_refine
    # 失败诊断单独保留，不冒充成功检查历史。
    diagnostics = trainer.output_dir / "refinement_failures.json"
    assert diagnostics.exists()
    record = json.loads(diagnostics.read_text(encoding="utf-8"))[-1]
    assert record["stage"] == "recheck and publish"
    assert record["artifacts_restored"] is True
    assert set(record["artifact_rollback"]) == {
        "last.pt", "best.pt", "checks.json", "run_manifest.json"
    }


def test_a_failure_after_a_checkpoint_was_really_replaced_restores_it(monkeypatch, tmp_path):
    """至少一份检查点已经**真的被替换过**之后才失败。"""
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="late_fail2")
    trainer.run(max_updates=1, verbose=False)
    before_last = (trainer.output_dir / "last.pt").read_bytes()

    original_save = trainer._save
    state = {"replaced": False}

    def save_then_fail(path):
        result = original_save(path)          # 真实的写盘先发生
        if Path(path).name == "last.pt":
            state["replaced"] = True
            raise OSError("injected: fails after last.pt was replaced")
        return result

    monkeypatch.setattr(trainer, "_save", save_then_fail)
    with pytest.raises(OSError, match="replaced"):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.undo()

    assert state["replaced"] is True, "注入没有走到真实的写盘"
    assert (trainer.output_dir / "last.pt").read_bytes() == before_last
    assert trainer._unusable is None or "rollback" not in trainer._unusable


def test_a_transaction_that_never_creates_a_file_leaves_none_behind(monkeypatch, tmp_path):
    """原先不存在的产物，失败后不能留下本次创建的版本。"""
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="absent")
    trainer.best = None
    trainer._best_candidate = None
    trainer._best_dirty = False
    trainer.run(max_updates=1, verbose=False)
    if (trainer.output_dir / "best.pt").exists():
        (trainer.output_dir / "best.pt").unlink()

    restore = _block_manifest(trainer)
    with pytest.raises(OSError):
        trainer.refine(iterations=1, verbose=False)
    restore()
    assert not (trainer.output_dir / "best.pt").exists(), (
        "失败的事务留下了一份它自己创建的正式产物"
    )


def test_a_successful_transaction_leaves_consistent_artifacts(monkeypatch, tmp_path):
    """成功事务：四份产物互相一致，且没有留下未提交标记。"""
    monkeypatch.setattr(trainer_module, "run_checks", _score_at(1))
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="ok_tx")
    trainer.run(max_updates=_total(tiny), verbose=False)
    trainer.refine(iterations=1, verbose=False)

    manifest = json.loads((trainer.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checks = json.loads((trainer.output_dir / "checks.json").read_text(encoding="utf-8"))
    last = torch.load(trainer.output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert manifest["status"] == last["run_status"] == RUN_DIAGNOSTIC
    assert manifest["metadata"]["check_state"] == last["check_state"]
    assert checks[-1]["parameter_fingerprint"] == _fingerprint(last["model"]), (
        "最后一条检查结论不是落在 last.pt 的参数上的"
    )
    assert not (trainer.output_dir / ".publish_in_progress.json").exists(), (
        "成功之后未提交标记还在，加载端会拒绝一份本来有效的检查点"
    )


# ==========================================================================
# §3 内存恢复与磁盘恢复必须独立尝试
# ==========================================================================


def test_a_memory_restore_failure_still_restores_the_disk(monkeypatch, tmp_path):
    """**复核的原反例。** 精修写出 ``last.pt`` 之后失败，同时内存恢复也失败。

    修前：磁盘恢复根本没执行，``last.pt`` 仍带本次精修记录，**新实例加载成功**。
    """
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="mem_fail")
    trainer.run(max_updates=1, verbose=False)
    good_last = (trainer.output_dir / "last.pt").read_bytes()

    restore = _block_manifest(trainer)
    original_restore = trainer._restore_state

    def broken(_snapshot):
        raise RuntimeError("injected: the in-memory restore fails")

    monkeypatch.setattr(trainer, "_restore_state", broken)
    with pytest.raises(TrainingError, match="Rolling back did not restore everything"):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.setattr(trainer, "_restore_state", original_restore)
    restore()

    assert trainer._unusable is not None
    # **磁盘恢复仍然执行了。**
    assert (trainer.output_dir / "last.pt").read_bytes() == good_last
    last = torch.load(trainer.output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert len(last.get("refinement") or []) == 0

    # 盘上回到旧的有效检查点，新实例看到的就是它。
    fresh, _ = _tiny(monkeypatch, tmp_path, name="mem_fail_fresh")
    fresh.load(trainer.output_dir / "last.pt")
    assert len(fresh.refine_history) == 0

    # 诊断文件里原始错误与恢复错误都在。
    record = json.loads(
        (trainer.output_dir / "refinement_failures.json").read_text(encoding="utf-8")
    )[-1]
    assert "run_manifest" in record["error"]
    assert "in-memory restore" in record["memory_rollback_error"]
    assert record["parameters_restored"] is False


def test_one_failing_artifact_does_not_stop_the_others(monkeypatch, tmp_path):
    """一份产物恢复失败，其余仍要尝试恢复。"""
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="one_fail")
    trainer.run(max_updates=1, verbose=False)
    before = _artifact_bytes(trainer)

    real_restore = trainer._restore_artifacts

    def partial(captured):
        results = real_restore(captured)
        results["checks.json"] = "FAILED: injected: cannot restore checks.json"
        return results

    restore = _block_manifest(trainer)
    monkeypatch.setattr(trainer, "_restore_artifacts", partial)
    with pytest.raises(TrainingError, match="Rolling back did not restore everything"):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.setattr(trainer, "_restore_artifacts", real_restore)
    restore()

    after = _artifact_bytes(trainer)
    assert after["last.pt"] == before["last.pt"], "其余产物没有被尝试恢复"
    record = json.loads(
        (trainer.output_dir / "refinement_failures.json").read_text(encoding="utf-8")
    )[-1]
    assert record["artifacts_restored"] is False
    assert "FAILED" in record["artifact_rollback"]["checks.json"]


def test_an_unrestorable_publish_cannot_be_loaded_by_a_new_instance(monkeypatch, tmp_path):
    """磁盘无法恢复时，未提交产物必须被**明确标为不可加载**，而且加载端要落实。"""
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="unrestorable")
    trainer.run(max_updates=1, verbose=False)

    real_restore = trainer._restore_artifacts

    def failing(captured):
        results = real_restore(captured)
        results["last.pt"] = "FAILED: injected: cannot restore last.pt"
        return results

    restore = _block_manifest(trainer)
    monkeypatch.setattr(trainer, "_restore_artifacts", failing)
    with pytest.raises(TrainingError, match="Rolling back did not restore everything"):
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.setattr(trainer, "_restore_artifacts", real_restore)
    restore()

    assert (trainer.output_dir / ".publish_in_progress.json").exists()
    fresh, _ = _tiny(monkeypatch, tmp_path, name="unrestorable_fresh")
    with pytest.raises(TrainingError, match="uncommitted"):
        fresh.load(trainer.output_dir / "last.pt")


def test_a_cleaned_up_rollback_removes_the_marker(monkeypatch, tmp_path):
    """回滚全部成功的目录必须恢复可加载——标记不能把好目录永久锁死。"""
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="marker_ok")
    trainer.run(max_updates=1, verbose=False)
    restore = _block_manifest(trainer)
    with pytest.raises(OSError):
        trainer.refine(iterations=1, verbose=False)
    restore()
    assert not (trainer.output_dir / ".publish_in_progress.json").exists()
    fresh, _ = _tiny(monkeypatch, tmp_path, name="marker_ok_fresh")
    fresh.load(trainer.output_dir / "last.pt")


def test_a_failing_diagnostics_write_still_surfaces_both_errors(monkeypatch, tmp_path):
    """§3.7：诊断文件写不进去时，原始错误与恢复错误仍要看得见。"""
    trainer, tiny = _tiny(monkeypatch, tmp_path, name="diag_fail")
    trainer.run(max_updates=1, verbose=False)

    def broken_diagnostics(_failure):
        raise OSError("injected: the diagnostics file cannot be written")

    restore = _block_manifest(trainer)
    monkeypatch.setattr(trainer, "_write_refinement_failure", broken_diagnostics)
    with pytest.raises(TrainingError) as caught:
        trainer.refine(iterations=1, verbose=False)
    monkeypatch.undo()
    restore()

    message = str(caught.value)
    assert "run_manifest" in message, "原始错误被最后一条日志异常盖掉了"
    assert "diagnostics file could not be written" in message
