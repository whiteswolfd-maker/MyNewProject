"""检查点字段、``训练→保存→预测`` 链路与发布标记的归属（第九轮 §1–§4）。

四条被复核独立复现过的行为：

* **§1** ``Trainer._build_payload()`` 从来不写 ``detonation_velocity``，而
  ``predict.load_model_from_checkpoint()`` 要求它——一份**方案 B** 正常训练出来的
  检查点，被预测入口当成"方案 A 的旧文件"拒绝（``SetupError: this checkpoint
  carries no detonation velocity ... it was written by the scheme-A pipeline``）。
* **§2** 只有 ``Trainer.load()`` 查 ``.publish_in_progress.json``，预测入口直接
  ``torch.load()``：恢复训练被拒绝的文件，仍能用于预测。
* **§3** ``refine()`` 的回滚分支无条件删除标记。一次 ``run()`` 发布失败留下的
  标记，会被随后一次**失败的精修**清掉——四份文件字节一字未变，目录却由
  "拒绝加载"变成"加载成功"。
* **§4** ``best_dirty=True`` 而候选为空的检查点被**静默接受**：标志被自动改成
  ``False``，最佳记录的分数还在、参数却没有，再次运行也不再产出 ``best.pt``。

**为什么 27 项联检过了、预测仍然是坏的**：联检的 ``compare`` 用例直接把参考
数组写成预测 NPZ，没有经过模型预测入口。它证明了对照命令能跑，没有证明
``训练 → 保存检查点 → 预测 → 对照报告`` 这条链。§1 的测试因此走完整的
``predict.main(argv)``，不许绕开。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.cli import predict as predict_module
from spherical_blast_pinn.cli._common import SetupError
from spherical_blast_pinn.physics.similitude import SimilitudeScales
from spherical_blast_pinn.training.checkpoint import (
    DETONATION_VELOCITY_FIELD,
    PUBLISH_MARKER,
    PUBLISH_OWNER_KEY,
)
from spherical_blast_pinn.training.trainer import TrainingError

import test_review_regressions as base
from test_best_candidate_and_transaction import _block_manifest, _tiny

FROZEN_D = 6930.0  # 夹具 `_scales()` 冻结的卡片爆速，见 test_review_regressions


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


def _trainer(monkeypatch, tmp_path, name: str = "chain"):
    """一份真实的小 Trainer：真实参数更新、真实落盘、真实检查点格式。"""
    return _tiny(monkeypatch, tmp_path, name=name)


def _radius(trainer) -> float:
    return float(trainer.cases[0].charge_radius)


def _write_config(tmp_path: Path) -> Path:
    """预测入口要一个真实配置。``--output`` 另行给出，这里只保证配置可加载。"""
    base_yaml = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    path = tmp_path / "predict_config.yaml"
    path.write_text(
        f"inherits: {base_yaml.as_posix()}\n"
        f"training:\n  output_dir: {(tmp_path / 'run').as_posix()}\n",
        encoding="utf-8",
    )
    return path


def _rewrite_checkpoint(path: Path, **changes) -> None:
    """按原样读出一份真实检查点，改几个字段再写回去（**只用于构造坏输入**）。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update(changes)
    torch.save(payload, path)


def _place_marker(trainer, artifacts, *, owner="some-other-transaction") -> Path:
    """放一个"上一次发布没做完"的标记。"""
    marker = trainer.output_dir / PUBLISH_MARKER
    marker.write_text(
        json.dumps({"artifacts": list(artifacts), PUBLISH_OWNER_KEY: owner}),
        encoding="utf-8",
    )
    return marker


def _artifact_bytes(trainer) -> dict[str, bytes]:
    return {
        name: (trainer.output_dir / name).read_bytes()
        for name in ("last.pt", "best.pt", "checks.json", "run_manifest.json")
        if (trainer.output_dir / name).exists()
    }


def _fresh(monkeypatch, tmp_path, name: str):
    """另一个实例：它读到的就是盘上那一份文件。"""
    trainer, _ = _tiny(monkeypatch, tmp_path, name=f"fresh_{name}")
    return trainer


# ==========================================================================
# §1 训练 → 保存检查点 → 预测 → 生成预测文件
# ==========================================================================


def test_the_training_save_predict_chain_runs_end_to_end(monkeypatch, tmp_path):
    """**复核点名的断链。** 走真实 ``predict.main(argv)``，检查预测文件真的产生。

    修前：这一行第一步就抛 ``SetupError``（"carries no detonation velocity"），
    而且把一份方案 B 的检查点说成是方案 A 写的。
    """
    trainer, _ = _trainer(monkeypatch, tmp_path, "chain")
    trainer.run(max_updates=2, verbose=False)
    checkpoint = trainer.output_dir / "last.pt"
    radius = _radius(trainer)

    # 保存端：字段确实在，而且是**冻结的卡片值**，不是某个默认常量。
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload[DETONATION_VELOCITY_FIELD] == pytest.approx(FROZEN_D)

    config = _write_config(tmp_path)
    output = tmp_path / "prediction"
    code = predict_module.main([
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--radius", str(radius),
        "--output", str(output),
        "--n-times", "5",
        "--n-radii", "20",
    ])
    assert code == 0

    written = list(output.glob("*.npz"))
    assert written, "预测入口没有生成预测文件"
    with np.load(written[0], allow_pickle=True) as handle:
        result = {key: handle[key] for key in handle.files}

    # **时间原点是拿检查点里的 D 重建出来的**，不是默认常量补的：独立算一遍
    # T_h = R_0/D + epsilon，逐位对上。
    expected_start = SimilitudeScales.for_radius(
        radius, time_scale=base.TIME_SCALE, detonation_velocity=FROZEN_D
    ).start_time()
    assert float(result["start_time"]) == pytest.approx(expected_start, rel=0, abs=1e-18)
    assert float(result["charge_radius"]) == pytest.approx(radius)
    assert np.asarray(result["density"]).shape == (5, 20)


def test_prediction_refuses_a_checkpoint_without_the_frozen_detonation_velocity(
    monkeypatch, tmp_path
):
    """§1 的拒绝路径：没有这个字段就必须拒绝，**不能用默认常量补齐**。"""
    trainer, _ = _trainer(monkeypatch, tmp_path, "no_d")
    trainer.run(max_updates=1, verbose=False)
    checkpoint = trainer.output_dir / "last.pt"

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop(DETONATION_VELOCITY_FIELD)
    torch.save(payload, checkpoint)

    with pytest.raises(SetupError, match=DETONATION_VELOCITY_FIELD):
        predict_module.load_model_from_checkpoint(
            checkpoint, charge_radius=_radius(trainer)
        )


@pytest.mark.parametrize("bad", [0.0, -6930.0, float("nan"), "fast"])
def test_prediction_refuses_a_non_physical_detonation_velocity(monkeypatch, tmp_path, bad):
    """字段在但不可用时同样拒绝：``D`` 进 ``T_h = R_0/D + epsilon``。"""
    trainer, _ = _trainer(monkeypatch, tmp_path, f"bad_d_{abs(hash(bad)) % 1000}")
    trainer.run(max_updates=1, verbose=False)
    checkpoint = trainer.output_dir / "last.pt"
    _rewrite_checkpoint(checkpoint, **{DETONATION_VELOCITY_FIELD: bad})

    with pytest.raises(SetupError, match=DETONATION_VELOCITY_FIELD):
        predict_module.load_model_from_checkpoint(
            checkpoint, charge_radius=_radius(trainer)
        )


# ==========================================================================
# §2 发布完整性检查必须是**共享**读取规则
# ==========================================================================


def test_both_readers_refuse_the_same_uncommitted_checkpoint(monkeypatch, tmp_path):
    """**复核的原反例。** 同一个未提交检查点，训练恢复与预测都必须拒绝。

    修前：``Trainer.load`` 拒绝，``predict.load_model_from_checkpoint`` 成功。
    """
    trainer, _ = _trainer(monkeypatch, tmp_path, "shared_refuse")
    trainer.run(max_updates=1, verbose=False)
    checkpoint = trainer.output_dir / "last.pt"
    _place_marker(trainer, ["last.pt"])

    fresh = _fresh(monkeypatch, tmp_path, "shared_refuse")
    with pytest.raises(TrainingError, match="uncommitted"):
        fresh.load(checkpoint)

    with pytest.raises(SetupError, match="uncommitted"):
        predict_module.load_model_from_checkpoint(
            checkpoint, charge_radius=_radius(trainer)
        )


def test_an_unreadable_marker_is_refused_not_ignored(monkeypatch, tmp_path):
    """"读不懂标记"不能当成"没有标记"——两个入口都保守拒绝。"""
    trainer, _ = _trainer(monkeypatch, tmp_path, "bad_marker")
    trainer.run(max_updates=1, verbose=False)
    checkpoint = trainer.output_dir / "last.pt"
    (trainer.output_dir / PUBLISH_MARKER).write_text("{not json", encoding="utf-8")

    fresh = _fresh(monkeypatch, tmp_path, "bad_marker")
    with pytest.raises(TrainingError, match="cannot be read"):
        fresh.load(checkpoint)
    with pytest.raises(SetupError, match="cannot be read"):
        predict_module.load_model_from_checkpoint(
            checkpoint, charge_radius=_radius(trainer)
        )


# ==========================================================================
# §3 标记的归属：后来的事务不许清理别人的标记
# ==========================================================================


def test_a_later_failed_operation_does_not_delete_an_earlier_uncommitted_marker(
    monkeypatch, tmp_path
):
    """**复核的原反例。** 一次 ``run()`` 发布失败留下标记，随后另一项失败操作。

    修前：精修的回滚把四份文件还原成**上一次失败时的字节**（一字未变），然后
    无条件删掉标记——目录由"拒绝加载"变成"加载成功"，而那套产物从未提交成功。
    """
    trainer, _ = _trainer(monkeypatch, tmp_path, "stale_marker")
    trainer.run(max_updates=2, verbose=False)

    # 1) 一次 run() 在写清单时失败 → 留下未提交标记，检查点是第 2 步、清单还是第 1 步。
    restore = _block_manifest(trainer)
    with pytest.raises(OSError):
        trainer.run(max_updates=2, verbose=False)
    restore()

    marker = trainer.output_dir / PUBLISH_MARKER
    assert marker.exists(), "失败的发布没有留下未提交标记"
    before = _artifact_bytes(trainer)

    fresh = _fresh(monkeypatch, tmp_path, "stale_marker")
    with pytest.raises(TrainingError, match="uncommitted"):
        fresh.load(trainer.output_dir / "last.pt")

    # 2) 再执行另一项失败操作。
    with pytest.raises(TrainingError, match="unfinished publish"):
        trainer.refine(iterations=1, verbose=False)

    # 标记**没有被删掉**，四份文件也一字未变，新实例仍然拒绝加载。
    assert marker.exists(), "后来的失败操作删掉了不是它写的标记"
    assert _artifact_bytes(trainer) == before, "回滚改变了文件内容"
    fresh_again = _fresh(monkeypatch, tmp_path, "stale_marker_2")
    with pytest.raises(TrainingError, match="uncommitted"):
        fresh_again.load(trainer.output_dir / "last.pt")


def test_end_publish_only_removes_its_own_marker(monkeypatch, tmp_path):
    """归属规则本身：``_end_publish`` 见到别人的标记一律留着。"""
    trainer, _ = _trainer(monkeypatch, tmp_path, "ownership")
    trainer.run(max_updates=1, verbose=False)

    marker = _place_marker(trainer, ["last.pt"], owner="not-mine")
    trainer._publish_token = "mine"
    assert trainer._end_publish() is False
    assert marker.exists(), "撤掉了不是本次事务写的标记"

    # 自己的标记才删。
    trainer._begin_publish(["last.pt"])
    assert trainer._end_publish() is True
    assert not marker.exists()


def test_a_new_run_supersedes_an_unfinished_publish_and_records_it(monkeypatch, tmp_path):
    """``run()`` 会重写全部产物，覆盖未完成的发布是合法的——但必须留下痕迹。"""
    trainer, _ = _trainer(monkeypatch, tmp_path, "supersede")
    trainer.run(max_updates=1, verbose=False)
    _place_marker(trainer, ["last.pt"], owner="the-previous-failure")

    result = trainer.run(max_updates=2, verbose=False)

    manifest = json.loads(
        (trainer.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    superseded = manifest["metadata"]["superseded_uncommitted_publish"]
    assert superseded is not None, "覆盖了一次未完成的发布却没有记录"
    assert superseded[PUBLISH_OWNER_KEY] == "the-previous-failure"
    assert superseded["artifacts"] == ["last.pt"]
    assert result is not None
    # 成功之后标记撤掉：四份产物是一起重写过的，目录一致。
    assert not (trainer.output_dir / PUBLISH_MARKER).exists()


# ==========================================================================
# §4 best_dirty 与候选必须互相印证
# ==========================================================================


def _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, name: str) -> tuple:
    """一份**真实**的、带着未发布最佳候选的中间检查点。

    走真实路径取候选：``verify()`` 在固定检查集上评分之后登记候选与状态快照，
    ``_save()`` 把这个状态写成文件。不手工拼一个"看起来像"的载荷，也不改
    ``best_dirty`` 本身——**这一份检查点本来就该带着标志**：还没有任何一次
    ``run()`` 把候选发布出去。

    （``self.best`` 一开始是 ``None``，所以第一次 ``verify()`` 必然入选。）
    """
    trainer, _ = _trainer(monkeypatch, tmp_path, name)
    trainer.verify(verbose=False)
    path = trainer.output_dir / "pending.pt"
    trainer._save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["best_dirty"] is True
    assert payload["pending_best_candidate"] is not None
    return trainer, path


def test_a_dirty_flag_without_a_candidate_is_refused(monkeypatch, tmp_path):
    """**复核的原反例。** ``best_dirty=True`` + 候选为空。

    修前：加载**成功**，``best_dirty`` 被自动改成 ``False``，最佳记录的分数还在、
    参数却没有，再次运行也不再产出 ``best.pt``。
    """
    trainer, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "dirty")
    _rewrite_checkpoint(path, pending_best_candidate=None)

    fresh = _fresh(monkeypatch, tmp_path, "dirty")
    with pytest.raises(TrainingError, match="pending_best_candidate"):
        fresh.load(path)
    assert fresh._best_dirty is False, "拒绝之前就把对象改掉了"


def test_a_candidate_that_does_not_match_the_best_record_is_refused(monkeypatch, tmp_path):
    """候选与最佳记录必须描述**同一刻**：步号对不上就是坏输入。"""
    trainer, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidate = dict(payload["pending_best_candidate"])
    candidate["step"] = int(candidate["step"]) + 5
    _rewrite_checkpoint(path, pending_best_candidate=candidate)

    fresh = _fresh(monkeypatch, tmp_path, "mismatch")
    with pytest.raises(TrainingError, match="different moments"):
        fresh.load(path)


def test_an_honest_pending_candidate_still_loads_and_keeps_its_flag(monkeypatch, tmp_path):
    """反面：**合法**的候选不能被这条新检查误伤，标志要原样保住。"""
    trainer, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "honest")

    fresh = _fresh(monkeypatch, tmp_path, "honest")
    fresh.load(path)
    assert fresh._best_dirty is True, "合法候选的待发布标志被清掉了"
    assert fresh._best_candidate is not None
    assert int(fresh._best_candidate["step"]) == int(fresh.best["step"])


# ==========================================================================
# 反面对照：干净的目录仍然一切照旧
# ==========================================================================


def test_a_clean_directory_loads_in_both_readers(monkeypatch, tmp_path):
    """没有标记、字段齐全的目录：两个入口都必须正常读进去。"""
    trainer, _ = _trainer(monkeypatch, tmp_path, "clean")
    trainer.run(max_updates=2, verbose=False)
    checkpoint = trainer.output_dir / "last.pt"
    assert not (trainer.output_dir / PUBLISH_MARKER).exists()

    fresh = _fresh(monkeypatch, tmp_path, "clean")
    fresh.load(checkpoint)
    assert fresh.step == trainer.step

    model, payload = predict_module.load_model_from_checkpoint(
        checkpoint, charge_radius=_radius(trainer)
    )
    assert float(payload[DETONATION_VELOCITY_FIELD]) == pytest.approx(FROZEN_D)
    assert model is not None
