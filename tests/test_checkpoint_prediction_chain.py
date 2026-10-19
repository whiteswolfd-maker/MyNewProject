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


# ==========================================================================
# 第三项：未发布最佳候选的**完整性**、**指纹一致性**与**发布前校验**
# ==========================================================================
#
# 复核实测（修前）：在真实生成的待发布候选检查点上，把候选 ``model`` 改成空字典、
# 改一个候选参数但不改指纹、删掉候选 ``optimizer``——三种都被共享校验与
# ``Trainer.load()`` **接受**。原因很直接：那一版只查了"候选是 Mapping、步号相同、
# ``model`` 是 Mapping、指纹是非空字符串"，既没有验候选的**模型结构**，也没有把
# 记录的指纹与候选里**实际**的参数对上。
#
# 这一节因此覆盖三件事：
#   1. 候选必须走**与顶层载荷同一份**状态校验；
#   2. 指纹必须**从候选的实际参数重算**；
#   3. 发布 ``best.pt`` 之前就要查完，不能等坏文件已经落盘。


def _mutate_candidate(path: Path, mutate) -> dict:
    """把候选取出来改一改再写回，返回**读回来的**候选。

    走"读一份真实检查点 → 改 → 写回"：控制流是真的，只有这一个字段被动手脚。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload["pending_best_candidate"])
    torch.save(payload, path)
    return torch.load(path, map_location="cpu", weights_only=False)["pending_best_candidate"]


def _global_rng_snapshot() -> dict:
    """进程的全局随机状态。``load`` 会改它，所以拒绝时它也必须原样。"""
    import random

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def _same_global_rng(left: dict, right: dict) -> bool:
    return (
        left["python"] == right["python"]
        and all(
            (np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b)
            for a, b in zip(left["numpy"], right["numpy"])
        )
        and torch.equal(left["torch"], right["torch"])
    )


def _receiver_snapshot(trainer) -> dict:
    """接收实例上**任何一次失败加载都不该改变**的那些东西。

    覆盖复核点名的那些：模型、**完整优化器状态**、阶段、候选、历史、采样器与
    **全局随机状态**。只比学习率是不够的——动量才是恢复失败时最先被写坏的东西。
    """
    import copy

    return {
        "model": {k: v.detach().clone() for k, v in trainer.model.state_dict().items()},
        "step": int(trainer.step),
        "best_dirty": bool(trainer._best_dirty),
        "best_candidate": copy.deepcopy(trainer._best_candidate),
        "best": copy.deepcopy(trainer.best),
        "history": copy.deepcopy(trainer.history),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "optimizer_lr": [g["lr"] for g in trainer.optimizer.param_groups],
        "stage": (trainer._current_stage, float(trainer._stage_window),
                  bool(trainer._stage_blocked), trainer._pending_stage),
        "sampler_rng": trainer.sampler.rng_state(),
        "global_rng": _global_rng_snapshot(),
    }


def _same_receiver(left: dict, right: dict) -> bool:
    return (
        set(left["model"]) == set(right["model"])
        and all(torch.equal(left["model"][k], right["model"][k]) for k in left["model"])
        and left["step"] == right["step"]
        and left["best_dirty"] == right["best_dirty"]
        and left["best_candidate"] == right["best_candidate"]
        and left["best"] == right["best"]
        and left["history"] == right["history"]
        and left["optimizer_lr"] == right["optimizer_lr"]
        and _same_optimizer(left["optimizer"], right["optimizer"])
        and left["stage"] == right["stage"]
        and left["sampler_rng"] == right["sampler_rng"]
        and _same_global_rng(left["global_rng"], right["global_rng"])
    )


def _same_optimizer(left: dict, right: dict) -> bool:
    """优化器的**完整**状态：参数组逐键相等，动量逐参数逐张量相等。"""
    if len(left["param_groups"]) != len(right["param_groups"]):
        return False
    for a, b in zip(left["param_groups"], right["param_groups"]):
        if set(a) != set(b):
            return False
        for key in a:
            if key == "params":
                if list(a[key]) != list(b[key]):
                    return False
            elif isinstance(a[key], torch.Tensor):
                if not torch.equal(a[key], b[key]):
                    return False
            elif a[key] != b[key]:
                return False
    if set(left["state"]) != set(right["state"]):
        return False
    for pid in left["state"]:
        a, b = left["state"][pid], right["state"][pid]
        if set(a) != set(b):
            return False
        for key in a:
            if isinstance(a[key], torch.Tensor):
                if not torch.equal(a[key], b[key]):
                    return False
            elif a[key] != b[key]:
                return False
    return True


def test_a_candidate_with_an_empty_model_is_refused(monkeypatch, tmp_path):
    """空候选模型：任何模型都能"装"进去，装完参数一个没变，读起来像成功。"""
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "empty_model")
    _mutate_candidate(path, lambda c: c.__setitem__("model", {}))

    fresh = _fresh(monkeypatch, tmp_path, "empty_model")
    with pytest.raises(TrainingError, match="model state is empty"):
        fresh.load(path)


def test_a_candidate_whose_parameters_do_not_match_its_fingerprint_is_refused(
    monkeypatch, tmp_path
):
    """**复核的原反例。** 改一个候选参数、指纹不动。

    只检查"指纹是非空字符串"时这种改法无人能发现；``best.pt`` 的
    ``candidate.parameter_fingerprint`` 却正是靠它声称"这一份就是被选中的那组
    参数"。
    """
    from spherical_blast_pinn.training.checkpoint import parameter_fingerprint_of

    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "fp")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    recorded = payload["pending_best_candidate"]["best_check"]["parameter_fingerprint"]

    def bump(candidate):
        key = sorted(candidate["model"])[0]
        candidate["model"][key] = candidate["model"][key] + 1.0

    mutated = _mutate_candidate(path, bump)
    assert parameter_fingerprint_of(mutated["model"]) != recorded, (
        "变异没有生效——参数没被改动"
    )
    assert mutated["best_check"]["parameter_fingerprint"] == recorded, (
        "变异意外改动了指纹——那就不是这个反例了"
    )

    fresh = _fresh(monkeypatch, tmp_path, "fp")
    with pytest.raises(TrainingError, match="parameter fingerprint"):
        fresh.load(path)

    # **不把记录改成重算值来"修复"**：文件必须保持原样，否则一个坏文件会被
    # 就地"修好"，下一次读它就是合法的了。
    after = torch.load(path, map_location="cpu", weights_only=False)
    assert (
        after["pending_best_candidate"]["best_check"]["parameter_fingerprint"] == recorded
    ), "校验过程把记录里的指纹改写了——那是伪造一致性，不是拒绝"


def test_a_candidate_without_its_optimizer_is_refused(monkeypatch, tmp_path):
    """**复核的原反例。** 删掉候选 optimizer。"""
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "no_opt")
    _mutate_candidate(path, lambda c: c.pop("optimizer"))

    fresh = _fresh(monkeypatch, tmp_path, "no_opt")
    with pytest.raises(TrainingError, match="optimizer state"):
        fresh.load(path)


def test_a_candidate_with_a_missing_key_or_wrong_shape_is_refused(monkeypatch, tmp_path):
    """候选参数的键、形状与当前模型对不上：先比，别等 load_state_dict 装到一半。"""
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "shape")
    name = None

    def drop(candidate):
        nonlocal name
        name = sorted(candidate["model"])[0]
        candidate["model"].pop(name)

    _mutate_candidate(path, drop)
    with pytest.raises(TrainingError, match="missing"):
        _fresh(monkeypatch, tmp_path, "shape_a").load(path)


def test_a_candidate_with_a_wrong_shape_is_refused(monkeypatch, tmp_path):
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "shape_b")

    def grow(candidate):
        key = sorted(candidate["model"])[0]
        tensor = candidate["model"][key]
        candidate["model"][key] = torch.cat([tensor, tensor], dim=0)

    _mutate_candidate(path, grow)
    with pytest.raises(TrainingError, match="has shape"):
        _fresh(monkeypatch, tmp_path, "shape_b").load(path)


def test_a_candidate_with_a_non_finite_parameter_is_refused(monkeypatch, tmp_path):
    """NaN/Inf 权重是真损坏：更新前就有 ``NonFiniteUpdate`` 拦截，正常轨迹不该有。"""
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "nan")

    def poison(candidate):
        key = sorted(candidate["model"])[0]
        poisoned = candidate["model"][key].clone()
        poisoned.view(-1)[0] = float("nan")
        candidate["model"][key] = poisoned

    _mutate_candidate(path, poison)
    with pytest.raises(TrainingError, match="non-finite"):
        _fresh(monkeypatch, tmp_path, "nan").load(path)


@pytest.mark.parametrize(
    "field",
    ["optimizer", "balancing", "rng", "sampler_rng", "stage", "run_origin",
     "history", "checks", "refinement", "last_update_loss"],
)
def test_a_candidate_missing_required_state_is_refused(monkeypatch, tmp_path, field):
    """候选快照的**每一项必需状态**：缺一个就拒绝。

    清单的依据是"写入端写了、恢复端读了"——``_checkpoint_state()`` 产生它们，
    ``_commit_checkpoint()`` 消费它们。少一项，恢复出来的候选就在那一项上静默
    缺省（例如没有 ``stage`` 就重建不出阶段与窗口）。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, f"miss_{field}")
    mutated = _mutate_candidate(path, lambda c: c.pop(field))
    assert field not in mutated, "变异没有生效"

    with pytest.raises(TrainingError):
        _fresh(monkeypatch, tmp_path, f"miss_{field}_r").load(path)


def test_a_candidate_naming_an_unknown_stage_is_refused(monkeypatch, tmp_path):
    """阶段**名字**要与这份日程相容，不能只查"键在不在"。

    ``current = "stage-99"`` 或 ``123`` 这类值会让"这一份是哪个阶段的结论"变成
    一句无从判断的话。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "stage_name")

    def rename(candidate):
        candidate["stage"]["current"] = "no-such-stage"

    _mutate_candidate(path, rename)
    with pytest.raises(TrainingError, match="does not define"):
        _fresh(monkeypatch, tmp_path, "stage_name").load(path)


def test_refusing_a_bad_candidate_leaves_the_receiver_and_the_files_untouched(
    monkeypatch, tmp_path
):
    """拒绝要发生在**接收实例被改动之前**，盘上的文件也不能被动。

    先把候选加载进真实实例再回滚，是另一种做法——它要求回滚本身完美，而这里的
    约定是"压根不要碰"。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "untouched")
    _mutate_candidate(path, lambda c: c.__setitem__("model", {}))
    before_bytes = path.read_bytes()

    fresh = _fresh(monkeypatch, tmp_path, "untouched")
    before = _receiver_snapshot(fresh)
    with pytest.raises(TrainingError):
        fresh.load(path)
    after = _receiver_snapshot(fresh)

    assert _same_receiver(before, after), "拒绝之后接收实例被改动了"
    assert path.read_bytes() == before_bytes, "拒绝过程改动了检查点文件"


def test_a_candidate_from_an_earlier_step_still_loads(monkeypatch, tmp_path):
    """**候选参数与当前参数不同是合法的**：它本来就可能来自更早的一步。

    要求两者数值相同会把正常情形判成损坏——所以这条测试把参数往前推，再断言
    一份"候选停在第 0 步、当前已经到第 2 步"的检查点仍然能加载。
    """
    trainer, _ = _trainer(monkeypatch, tmp_path, "earlier")
    trainer.verify(verbose=False)          # 候选登记在第 0 步
    assert trainer._best_candidate is not None

    # 当前参数继续往前走**一步**。``run()`` 的循环是"先 ``_update()``、再把
    # ``step`` 加一"（``_update`` 自己不动步号），这里照做。
    #
    # 只走一步不是偷懒：这份小日程每 2 步一阶段，走到第 2 步就跨进下一阶段，
    # 而阶段窗口是靠 ``run()`` 的循环推进的——手工加步号会让 ``stage.window_fraction``
    # 与日程对不上，那是**夹具**造出来的不一致，不是被测行为。
    trainer._update()
    trainer.step += 1
    path = trainer.output_dir / "later.pt"
    trainer._save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidate = payload["pending_best_candidate"]
    assert int(candidate["step"]) < int(payload["step"]), "夹具没有把当前参数推得更远"
    assert any(
        not torch.equal(candidate["model"][k], payload["model"][k])
        for k in candidate["model"]
    ), "候选参数与当前参数一模一样——测不到'允许不同'这件事"

    fresh = _fresh(monkeypatch, tmp_path, "earlier_r")
    fresh.load(path)
    assert fresh._best_dirty is True
    assert int(fresh._best_candidate["step"]) == int(candidate["step"])


def test_a_loaded_candidate_publishes_a_consistent_best_checkpoint(monkeypatch, tmp_path):
    """合法候选恢复之后走**真实发布路径**：``best.pt`` 的指纹、步号、最佳记录一致。"""
    from spherical_blast_pinn.training.checkpoint import parameter_fingerprint_of

    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "publish")
    expected = torch.load(path, map_location="cpu", weights_only=False)
    expected_step = int(expected["pending_best_candidate"]["step"])

    fresh = _fresh(monkeypatch, tmp_path, "publish")
    fresh.load(path)
    fresh.run(max_updates=fresh.step, verbose=False)

    best_path = fresh.output_dir / "best.pt"
    assert best_path.exists(), "恢复出来的候选没有走发布路径"
    best = torch.load(best_path, map_location="cpu", weights_only=False)

    assert best["candidate"]["step"] == best["step"] == expected_step
    assert best["candidate"]["parameter_fingerprint"] == parameter_fingerprint_of(best["model"]), (
        "best.pt 里的参数与它声称的候选指纹不是同一组"
    )
    assert best["best_check"]["parameter_fingerprint"] == best["candidate"]["parameter_fingerprint"]
    assert best["best_check"]["step"] == best["step"]


def test_a_published_best_checkpoint_is_loadable_and_predictable(monkeypatch, tmp_path):
    """发布出来的 ``best.pt`` 要能被新实例恢复，**并且**能过预测入口。"""
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "published")
    publisher = _fresh(monkeypatch, tmp_path, "published")
    publisher.load(path)
    publisher.run(max_updates=publisher.step, verbose=False)
    best_path = publisher.output_dir / "best.pt"

    fresh = _fresh(monkeypatch, tmp_path, "published_r")
    fresh.load(best_path)
    assert fresh._best_dirty is False, "已发布的 best.pt 还带着待发布候选"

    model, payload = predict_module.load_model_from_checkpoint(
        best_path, charge_radius=_radius(publisher)
    )
    assert model is not None
    assert float(payload[DETONATION_VELOCITY_FIELD]) == pytest.approx(FROZEN_D)


def test_a_corrupt_in_memory_candidate_never_reaches_best_pt(monkeypatch, tmp_path):
    """**发布前**就要查完：坏候选不能留下任何发布副作用。

    候选走完 ``load`` 之后在下游被改坏，同样会走到发布这一步。这里让候选在内存里
    是坏的、并且**不让后续 ``verify()`` 把它替换掉**（评分固定成更差），于是发布
    流程面对的就是这个坏候选——它必须在写标记之前停下。

    若不控制评分顺序，一次新的 ``verify()`` 会选出一个**新的合法候选**顶掉它，
    那条路径本来就该成功，测出来的不是"旧候选的发布路径被修好了"。
    """
    from spherical_blast_pinn.evaluation.checks import CheckReport
    from test_best_candidate_and_transaction import trainer_module

    trainer, _ = _trainer(monkeypatch, tmp_path, "corrupt_publish")
    trainer.verify(verbose=False)
    assert trainer._best_dirty and trainer._best_candidate is not None

    trainer._best_candidate["model"] = {}     # 下游把它改坏

    def worse(model, check_set, **kwargs):
        # **必须真的更差**：分数是字典序比较的，第一个分量（非物理违规数）先比，
        # 相等才比第二个。合成算例上的真实分数在第二项就有 ~9.6e6，所以"更差"
        # 得给一个明显更大的数——给 1e6 反而更优，新候选就会把坏的顶掉，
        # 于是这条测试测到的是"新候选正常发布"，而不是"旧候选的发布路径"。
        value = 1.0e18
        return CheckReport(step=int(kwargs.get("steps", 0)), metrics={
            "nonphysical_violation_count": value,
            "physics_violation_score": value,
            "development_prediction_error": value,
            "total_loss": value,
        })

    monkeypatch.setattr(trainer_module, "run_checks", worse)
    trainer.check_every = 1

    with pytest.raises(TrainingError, match="model state is empty"):
        trainer.run(max_updates=1, verbose=False)

    assert not (trainer.output_dir / PUBLISH_MARKER).exists(), (
        "坏候选已经产生了发布副作用（标记）"
    )
    assert not (trainer.output_dir / "best.pt").exists(), "坏候选被写成了 best.pt"


# ==========================================================================
# 第三项（补完）：候选的**非模型**状态——优化器结构、全局随机、阶段窗口
# ==========================================================================
#
# 复核实测（修前）：在真实生成的待发布候选检查点上，把候选 ``optimizer`` 换成
# 空字典 ``{}``、把 ``rng['torch']`` 换成整数 ``0``、把 ``stage.window_fraction``
# 改成 ``0.123``——三种都被共享校验与 ``Trainer.load()`` **接受**，而且 ``run()``
# 照样把它们发布成 ``best.pt``。错误只在**下一个**实例恢复 ``best.pt`` 时才出现：
#
#   A. ``KeyError: 'param_groups'``
#   B. ``TypeError: expected a torch.ByteTensor, but got int``
#   C. 窗口与候选步号对应的日程不一致
#
# 原因同上一轮：那一版只查了"键在不在、是不是 Mapping"。
#   ``{"state": {}, ...}`` 与 ``{}`` 都通过"是 Mapping"那一关；
#   ``0`` 不是 ``None``，于是"两项非空"也过了；窗口只查了"是不是有限数"。
#
# 这一节覆盖：
#   1. 优化器内部结构、参数标识对应关系与逐参数状态的相容性；
#   2. 全局随机状态**能不能真的被生成器接受**（独立生成器，不动全局）；
#   3. 阶段与窗口跟**这份快照自己的**步号一致（不是接收实例的步号）；
#   4. 三条都在**接收实例被改动之前**、以及**发布副作用之前**被拒绝。


def _worse_than_any_candidate(model, check_set, **kwargs):
    """永远更差的选择分数：让已登记的候选**没有被新候选替换**。

    给 1e18 而不是 1e6：合成算例上的真实分数在第二项就有 ~9.6e6，"更差"得给一个
    明显更大的数，否则新候选反而更优、把我们要观察的那个顶掉。
    """
    from spherical_blast_pinn.evaluation.checks import CheckReport

    value = 1.0e18
    return CheckReport(step=int(kwargs.get("steps", 0)), metrics={
        "nonphysical_violation_count": value,
        "physics_violation_score": value,
        "development_prediction_error": value,
        "total_loss": value,
    })


def _partial_candidates():
    """复核点名的三条变异。每条给出"改哪里"与断言变异生效的办法。"""
    def empty_optimizer(candidate):
        candidate["optimizer"] = {}

    def untyped_rng(candidate):
        candidate["rng"]["torch"] = 0

    def wrong_window(candidate):
        candidate["stage"]["window_fraction"] = 0.123

    return {
        "optimizer": (empty_optimizer, lambda c: c["optimizer"] == {}),
        "rng": (untyped_rng, lambda c: c["rng"]["torch"] == 0),
        "window": (wrong_window, lambda c: c["stage"]["window_fraction"] == 0.123),
    }


@pytest.mark.parametrize("case", sorted(_partial_candidates()))
def test_a_candidate_with_a_corrupt_non_model_field_is_refused(monkeypatch, tmp_path, case):
    """**复核的三条反例。** 共享校验与 ``Trainer.load()`` 都必须拦下。

    走真实控制流：``verify()`` 登记候选、``_save()`` 写成文件，再对独立副本动手。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    mutate, applied = _partial_candidates()[case]
    trainer, path = _checkpoint_with_a_pending_candidate(
        monkeypatch, tmp_path, f"part_{case}"
    )
    mutated = _mutate_candidate(path, mutate)
    assert applied(mutated), f"{case}: 变异没有生效"

    payload = torch.load(path, map_location="cpu", weights_only=False)
    with pytest.raises(TrainingError):
        require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())

    fresh = _fresh(monkeypatch, tmp_path, f"part_{case}_r")
    with pytest.raises(TrainingError):
        fresh.load(path)


def test_a_candidate_with_an_empty_optimizer_is_refused(monkeypatch, tmp_path):
    """``{}`` 与"合法的空 ``state``"是两回事：前者缺恢复必需的 ``param_groups``。

    修前这条能一路走到发布：``load`` 接受、``run()`` 写出 ``best.pt``，只有恢复
    ``best.pt`` 的**下一个**实例才拿到 ``KeyError: 'param_groups'``。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "empty_opt")
    _mutate_candidate(path, lambda c: c.__setitem__("optimizer", {}))

    fresh = _fresh(monkeypatch, tmp_path, "empty_opt_r")
    with pytest.raises(TrainingError, match="param_groups"):
        fresh.load(path)


def test_a_candidate_whose_global_rng_is_not_usable_is_refused(monkeypatch, tmp_path):
    """``0`` 不是 ``None``，但生成器收不下它。

    "两项非空"这条检查放行了整数 ``0``，真正的错误（``expected a
    torch.ByteTensor``）要等到恢复时才抛——那时坏候选已经发布成 ``best.pt`` 了。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "bad_rng")
    _mutate_candidate(path, lambda c: c["rng"].__setitem__("torch", 0))

    fresh = _fresh(monkeypatch, tmp_path, "bad_rng_r")
    with pytest.raises(TrainingError, match="global RNG state is not usable"):
        fresh.load(path)


def test_a_candidate_whose_stage_window_does_not_match_its_step_is_refused(
    monkeypatch, tmp_path
):
    """窗口要与**这份快照自己的**步号一致，不是与接收实例的步号一致。

    候选在第 0 步，正确窗口是 0.25；改成 0.123 之后要让校验自己算出来对不上。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "bad_window")
    mutated = _mutate_candidate(
        path, lambda c: c["stage"].__setitem__("window_fraction", 0.123)
    )
    assert int(mutated["step"]) == 0, "夹具的候选不在第 0 步——这条测试的前提变了"

    fresh = _fresh(monkeypatch, tmp_path, "bad_window_r")
    with pytest.raises(TrainingError, match="not the same schedule"):
        fresh.load(path)


@pytest.mark.parametrize("case", sorted(_partial_candidates()))
def test_a_corrupt_non_model_field_in_memory_is_refused_before_publishing(
    monkeypatch, tmp_path, case
):
    """候选在**内存里**被改坏（例如被下游代码动过），发布前也要拦下。

    控制评分顺序，保证坏候选没有被一次新的 ``verify()`` 替换掉——否则测到的是
    "新候选正常发布"，而不是"坏候选的发布路径被堵住了"。
    """
    from test_best_candidate_and_transaction import trainer_module

    mutate, applied = _partial_candidates()[case]
    trainer, _ = _trainer(monkeypatch, tmp_path, f"mem_{case}")
    trainer.verify(verbose=False)
    assert trainer._best_dirty and trainer._best_candidate is not None

    mutate(trainer._best_candidate)
    assert applied(trainer._best_candidate), f"{case}: 变异没有生效"

    # 上一次运行留下的产物：拒绝发布不能覆盖它们。
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    for name in ("last.pt", "checks.json", "run_manifest.json"):
        (trainer.output_dir / name).write_bytes(f"previous {name}".encode())
        existing[name] = (trainer.output_dir / name).read_bytes()

    monkeypatch.setattr(trainer_module, "run_checks", _worse_than_any_candidate)
    trainer.check_every = 1

    with pytest.raises(TrainingError):
        trainer.run(max_updates=1, verbose=False)

    assert not (trainer.output_dir / PUBLISH_MARKER).exists(), (
        f"{case}: 坏候选产生了发布副作用（标记）"
    )
    assert not (trainer.output_dir / "best.pt").exists(), f"{case}: 坏候选被写成了 best.pt"
    for name, before in existing.items():
        assert (trainer.output_dir / name).read_bytes() == before, (
            f"{case}: 拒绝发布时覆盖了已有的 {name}"
        )


def test_refusing_a_corrupt_candidate_leaves_every_piece_of_the_receiver_alone(
    monkeypatch, tmp_path
):
    """拒绝发生在改动**之前**：完整优化器状态、阶段、采样器与全局随机都不动。

    上一版的快照只比了模型、步号与学习率——那样即使"回滚"把动量写坏了也看不出来。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "atomic")
    _mutate_candidate(path, lambda c: c.__setitem__("optimizer", {}))
    before_bytes = path.read_bytes()

    fresh = _fresh(monkeypatch, tmp_path, "atomic_r")
    before = _receiver_snapshot(fresh)
    with pytest.raises(TrainingError):
        fresh.load(path)
    after = _receiver_snapshot(fresh)

    assert _same_receiver(before, after), "拒绝路径改动了接收实例"
    assert path.read_bytes() == before_bytes, "拒绝路径改动了输入文件"


def test_a_candidate_with_an_empty_optimizer_state_is_still_legal(monkeypatch, tmp_path):
    """**不要为了拦坏数据把合法初始状态一起拦掉。**

    一次更新都没做过的训练，优化器 ``state`` 就是空的——那是合法的第一份检查点。
    被禁止掉的话，"训练刚开始就中断、恢复"这条最普通的路走不通。
    """
    _, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "empty_state")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = payload["pending_best_candidate"]["optimizer"]
    assert optimizer["state"] == {}, "夹具的候选已经更新过——测不到空 state"
    assert optimizer["param_groups"], "参数组本身不能为空"

    fresh = _fresh(monkeypatch, tmp_path, "empty_state_r")
    fresh.load(path)
    assert fresh._best_dirty is True
    # 恢复之后要**真的还能往前走一步**，而不只是"加载没报错"。
    fresh._update()
    fresh.step += 1
    assert fresh.optimizer.state_dict()["state"], "恢复后的优化器一步都走不了"


def _momentum_trainer(monkeypatch, tmp_path, name: str, *, updates: int = 8):
    """阶段更长的小 Trainer：让"候选带动量、当前参数还在同一阶段里"成为可能。

    ``updates=8`` 而不是夹具默认的 2：默认日程每 2 步一阶段，"候选在第 1 步、
    当前在第 2 步"会正好跨过阶段边界，手工推进的步号会让窗口与日程对不上——
    那是**夹具**造出来的不一致，不是被测行为。
    """
    from spherical_blast_pinn.training.schedule import Stage

    trainer, _ = _trainer(monkeypatch, tmp_path, name)
    trainer.schedule = tuple(
        Stage(
            stage.name, updates,
            stage.pde_weight_start, stage.pde_weight_end,
            stage.learning_rate_start, stage.learning_rate_end,
            time_window_fraction=stage.time_window_fraction,
        )
        for stage in trainer.schedule
    )
    return trainer


def _fresh_momentum(monkeypatch, tmp_path, name: str):
    """恢复端的实例必须用**同一份日程**——检查点里存着日程签名。"""
    return _momentum_trainer(monkeypatch, tmp_path, name)


def _checkpoint_with_a_candidate_that_has_momentum(monkeypatch, tmp_path, name: str):
    """候选来自第 1 步（**已经更新过**），当前参数走到第 2 步——同一阶段内。"""
    trainer = _momentum_trainer(monkeypatch, tmp_path, name)
    trainer._update()
    trainer.step += 1
    trainer.verify(verbose=False)          # 候选登记在第 1 步，带着 Adam 动量
    trainer._update()
    trainer.step += 1
    path = trainer.output_dir / "momentum.pt"
    trainer._save(path)
    return trainer, path


def test_a_candidate_with_momentum_can_be_restored_and_keeps_updating(
    monkeypatch, tmp_path
):
    """带优化器动量的合法候选：恢复出**同样的**动量，并且能继续更新。

    这一条与"空 ``state`` 合法"是一对：放宽成"优化器只要是 Mapping 就行"会放过
    ``{}``，收紧成"状态必须完整"会拦掉这里。两条都得成立。
    """
    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "momentum"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidate = payload["pending_best_candidate"]
    assert int(candidate["step"]) == 1 < int(payload["step"])
    assert candidate["optimizer"]["state"], "候选没有动量——这条测试的前提不成立"

    fresh = _fresh_momentum(monkeypatch, tmp_path, "momentum_r")
    fresh.load(path)
    # ``load`` 恢复的是**顶层**状态（当前参数，第 2 步）；候选作为"待发布最佳"
    # 原样带着走。两者本来就是两个时刻——把候选的动量和恢复出来的动量混为一谈，
    # 正是这一项要防的那类错。
    assert _same_optimizer(
        fresh.optimizer.state_dict(), payload["optimizer"]
    ), "恢复出来的动量与检查点里的当前状态不是同一份"
    assert _same_optimizer(
        fresh._best_candidate["optimizer"], candidate["optimizer"]
    ), "跟着恢复进来的候选动量被改动了"

    moments_before = {
        pid: entry["exp_avg"].clone()
        for pid, entry in payload["optimizer"]["state"].items()
    }

    fresh._update()
    fresh.step += 1
    after = fresh.optimizer.state_dict()["state"]
    assert all(
        bool(torch.isfinite(entry["exp_avg"]).all()) for entry in after.values()
    ), "继续更新之后动量变成了非有限值"
    assert any(
        not torch.equal(after[pid]["exp_avg"], moments_before[pid])
        for pid in moments_before
    ), "继续更新没有真的动用恢复出来的动量"


def test_an_earlier_candidate_with_momentum_is_published_and_predictable(
    monkeypatch, tmp_path
):
    """较早、带动量的合法候选仍可发布；发布出来的 ``best.pt`` 能恢复**并**预测。"""
    from spherical_blast_pinn.training.checkpoint import parameter_fingerprint_of

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "publish_momentum"
    )
    expected = torch.load(path, map_location="cpu", weights_only=False)
    expected_step = int(expected["pending_best_candidate"]["step"])

    publisher = _fresh_momentum(monkeypatch, tmp_path, "publish_momentum_r")
    publisher.load(path)
    publisher.run(max_updates=publisher.step, verbose=False)

    best_path = publisher.output_dir / "best.pt"
    assert best_path.exists(), "合法候选没有走发布路径"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert best["candidate"]["step"] == expected_step
    assert best["candidate"]["parameter_fingerprint"] == parameter_fingerprint_of(
        best["model"]
    )
    # ``best.pt`` **就是**已发布的候选：它的顶层状态就是候选那一刻的状态，
    # ``pending_best_candidate`` 因此是 ``None``（没有"还没发布的最佳"了）。
    assert best["step"] == expected_step
    assert best["pending_best_candidate"] is None
    assert _same_optimizer(
        best["optimizer"], expected["pending_best_candidate"]["optimizer"]
    ), "best.pt 里的优化器状态不是候选那一刻的——发布的不是同一组动量"

    fresh = _fresh_momentum(monkeypatch, tmp_path, "publish_momentum_rr")
    fresh.load(best_path)
    assert fresh._best_dirty is False
    model, payload = predict_module.load_model_from_checkpoint(
        best_path, charge_radius=_radius(publisher)
    )
    assert model is not None
    assert float(payload[DETONATION_VELOCITY_FIELD]) == pytest.approx(FROZEN_D)


def test_the_original_three_counterexamples_are_still_refused(monkeypatch, tmp_path):
    """**上一轮已通过的三条不能因为这一轮放宽掉。**

    空候选模型 / 改候选参数留旧指纹 / 删候选 optimizer——共享校验与
    ``Trainer.load()`` 都还要拦。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    def empty_model(candidate):
        candidate["model"] = {}

    def bump_parameter(candidate):
        key = sorted(candidate["model"])[0]
        candidate["model"][key] = candidate["model"][key] + 1.0

    def drop_optimizer(candidate):
        candidate.pop("optimizer")

    for label, mutate in (
        ("empty_model", empty_model),
        ("bumped_fingerprint", bump_parameter),
        ("no_optimizer", drop_optimizer),
    ):
        trainer, path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, label)
        _mutate_candidate(path, mutate)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        with pytest.raises(TrainingError):
            require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())
        with pytest.raises(TrainingError):
            _fresh(monkeypatch, tmp_path, f"{label}_r").load(path)


#: v4（27 项）→ v5（30 项）新增的三项。对照的是 `REVIEW_RESPONSE_3e4e1c61.md`
#: 里逐项列出的 v4 清单：其余 27 项与 v5 完全同名且保持相对次序。
V5_ADDED_CHECKS = (
    "prediction_entry_runs_from_a_training_checkpoint",
    "the_model_prediction_can_be_compared_against_the_reference",
    "the_same_uncommitted_checkpoint_is_refused_by_both_entries",
)

#: v4 清单里那 27 项。写死在这里，是为了让"脚本被改回 v4"或"某一项被删掉"
#: 都会被这条测试当场抓住，而不是靠人去比对报告。
V4_CHECK_NAMES = (
    "continuous_run_reaches_the_end_of_the_schedule",
    "every_shortened_stage_boundary_is_crossed",
    "resume_at_a_stage_boundary_matches_a_continuous_run",
    "a_resumed_run_reports_the_measured_loss_of_its_last_update",
    "a_failed_stage_stays_blocked_after_a_resume",
    "a_completed_refinement_reports_the_same_identity_everywhere",
    "a_refinement_rechecks_the_parameters_it_changed",
    "the_refinement_records_what_it_did",
    "a_refinement_after_a_truncated_run_is_still_diagnostic",
    "an_injected_refinement_failure_leaves_the_state_alone",
    "the_impulse_counterexample_is_2_point_5",
    "a_contact_sweeping_between_frames_is_not_interpolated_across",
    "outside_the_prediction_grid_or_time_coverage_says_why",
    "compare_cli_runs_end_to_end_and_writes_a_readable_report",
    "a_refinement_failure_after_the_update_leaves_no_half_changed_state",
    "an_explicit_diagnostic_identity_survives_a_reload",
    "best_pt_carries_the_status_of_the_run_that_selected_it",
    "a_changed_field_is_refused_by_the_data_identity",
    "a_zero_threshold_triangle_reports_area_2_and_duration_2",
    "an_exact_frame_ignores_a_missing_neighbour_frame",
    "an_incomplete_phase_removes_the_phase_and_arrival_errors",
    "a_mid_run_best_is_published_with_its_own_parameters",
    "best_pt_can_be_resumed",
    "a_failure_after_earlier_artifacts_were_published_rolls_all_of_them_back",
    "a_memory_restore_failure_still_restores_the_disk",
    "an_unrestorable_publish_cannot_be_loaded",
    "a_missing_left_edge_is_not_a_complete_positive_phase",
)


def test_the_delivered_snake_script_declares_v5_and_carries_the_three_added_checks():
    """报告里写的联检结果必须对应**交付的那份脚本**。

    上一轮的报告写了 v5 / 30 项，而复核看到的远端基线脚本是 v4 / 27 项。差异的
    来源是"改动从没被交付出去"（这个工作树没有 ``.git``，无法提交），不是检查项
    对不上——所以这里直接读脚本本体，把版本号、检查项数、以及 v4 那 27 项一并
    钉住：任何一项被删或版本被改回 4，这条测试立刻失败。
    """
    import ast

    source = Path(__file__).resolve().parent.parent / "scripts" / "bounded_snake_check.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    version = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "SNAKE_CHECK_VERSION" for target in node.targets
        ):
            version = ast.literal_eval(node.value)
    assert version == 5, f"联检脚本声明的版本是 {version}，报告里写的是 5"

    names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "check"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert set(V4_CHECK_NAMES) <= names, (
        f"v4 的检查项丢了：{sorted(set(V4_CHECK_NAMES) - names)}"
    )
    assert set(V5_ADDED_CHECKS) <= names, (
        f"v4 → v5 新增的三项不在脚本里：{sorted(set(V5_ADDED_CHECKS) - names)}"
    )
    assert len(V4_CHECK_NAMES) == 27, "v4 清单本身不是 27 项——底稿写错了"


# ==========================================================================
# 第四项：优化器状态的**数值语义**，与探针更新的**结果**检查
# ==========================================================================
#
# 复核实测（修前）：真实训练产生的候选带着 Adam 动量，只把
# ``pending_best_candidate.optimizer.state[参数号].exp_avg_sq`` 填成 ``-1``，于是：
#
#   共享候选校验通过 → ``Trainer.load()`` 通过 → ``run()`` 发布 ``best.pt``
#   → 新实例重新加载 ``best.pt`` 通过 → **继续一次真实 ``_update()`` 之后参数变成
#   非有限值**。
#
# 两个直接原因：
#
#   1. **``-1`` 是有限值。** 当时只查张量有限性，而 ``exp_avg_sq`` 估计的是**梯度
#      的平方**，Adam 要开它的平方根再除——负值开方得 NaN。实测：一个被污染的
#      状态项对应**正好一个**参数被写成 NaN（12 个参数号逐个试过），而再下一次
#      迭代会以 ``NonFiniteUpdate``（"the total loss is not finite"）硬失败。
#   2. **探针只查了"``step()`` 有没有抛异常"。** PyTorch 可以不抛异常、直接产出
#      NaN。"跑完了"和"结果是好的"是两件事。
#
# 这一节因此有三块：非负语义、探针结果检查、以及"合法情形不能被一起拦掉"。


def _first_parameter_id(payload: dict) -> int:
    return int(sorted(payload["pending_best_candidate"]["optimizer"]["state"])[0])


def _mutate_payload(path: Path, mutate) -> dict:
    """改**整份载荷**的任意位置（候选或顶层），返回读回来的载荷。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _poison_candidate_second_moment(payload: dict, value: float = -1.0) -> None:
    pid = _first_parameter_id(payload)
    payload["pending_best_candidate"]["optimizer"]["state"][pid]["exp_avg_sq"].fill_(value)


def _poison_top_level_second_moment(payload: dict, value: float = -1.0) -> None:
    pid = int(sorted(payload["optimizer"]["state"])[0])
    payload["optimizer"]["state"][pid]["exp_avg_sq"].fill_(value)


# --------------------------------------------------------------------------
# 1. 负的二阶动量：共享校验与 Trainer.load() 都要拒绝
# --------------------------------------------------------------------------


def test_a_candidate_with_a_negative_second_moment_is_refused(monkeypatch, tmp_path):
    """**复核的原反例。** ``exp_avg_sq = -1`` 必须被拒绝，而不是一路发布出去。

    走真实控制流取候选（``_update()`` 一步 → ``verify()`` 登记 → 再走一步 →
    ``_save()``），只把这一个状态项填成 ``-1``。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "neg_sq"
    )
    pid = _first_parameter_id(torch.load(path, map_location="cpu", weights_only=False))
    mutated = _mutate_payload(path, _poison_candidate_second_moment)
    entry = mutated["pending_best_candidate"]["optimizer"]["state"][pid]
    assert bool((entry["exp_avg_sq"] == -1).all()), "变异没有生效"
    assert bool(torch.isfinite(entry["exp_avg_sq"]).all()), (
        "变异必须是**有限**的负值——非有限的话测到的是另一条检查"
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    with pytest.raises(TrainingError, match="exp_avg_sq"):
        require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())

    fresh = _fresh_momentum(monkeypatch, tmp_path, "neg_sq_r")
    with pytest.raises(TrainingError, match="exp_avg_sq") as excinfo:
        fresh.load(path)

    # 报错要点名**状态字段**与**参数编号**，否则拿到报错的人还得去读代码。
    message = str(excinfo.value)
    assert "exp_avg_sq" in message and f"parameter {pid}" in message, message
    assert "negative" in message, message


def test_the_top_level_optimizer_with_a_negative_second_moment_is_refused(
    monkeypatch, tmp_path
):
    """**同一套规则也要管顶层载荷。** 不能只在候选那一路加特判。

    顶层与候选走的是同一个 ``validate_state_snapshot``（``_snapshot_rules()``
    是唯一的上下文构造点），这一条把"同一套"这件事钉死。
    """
    _, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "neg_top"
    )
    mutated = _mutate_payload(path, _poison_top_level_second_moment)
    pid = int(sorted(mutated["optimizer"]["state"])[0])
    assert bool((mutated["optimizer"]["state"][pid]["exp_avg_sq"] == -1).all()), (
        "顶层的变异没有生效"
    )

    fresh = _fresh_momentum(monkeypatch, tmp_path, "neg_top_r")
    with pytest.raises(TrainingError, match="exp_avg_sq") as excinfo:
        fresh.load(path)
    assert f"parameter {pid}" in str(excinfo.value), str(excinfo.value)


def test_a_negative_second_moment_never_reaches_best_pt(monkeypatch, tmp_path):
    """端到端：坏候选既不能被加载，也不能被发布。"""
    from test_best_candidate_and_transaction import trainer_module

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "neg_e2e"
    )
    _mutate_payload(path, _poison_candidate_second_moment)

    publisher = _fresh_momentum(monkeypatch, tmp_path, "neg_e2e_r")
    with pytest.raises(TrainingError, match="exp_avg_sq"):
        publisher.load(path)

    assert not (publisher.output_dir / "best.pt").exists(), "坏检查点被发布成了 best.pt"
    assert not (publisher.output_dir / PUBLISH_MARKER).exists(), "留下了发布标记"


# --------------------------------------------------------------------------
# 2. 加载合法检查点之后，**在内存里**破坏候选：发布前必须拒绝
# --------------------------------------------------------------------------


def test_a_negative_second_moment_in_memory_is_refused_before_publishing(
    monkeypatch, tmp_path
):
    """候选被下游代码改坏之后，发布路径要在 ``_begin_publish()`` **之前**停下。

    控制评分顺序（``run_checks`` 固定成更差），保证坏候选没有被一次新的
    ``verify()`` 换成合法候选——否则测到的是"新候选正常发布"。
    """
    from test_best_candidate_and_transaction import trainer_module

    trainer, _ = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "neg_mem"
    )
    assert trainer._best_dirty and trainer._best_candidate is not None

    pid = int(sorted(trainer._best_candidate["optimizer"]["state"])[0])
    trainer._best_candidate["optimizer"]["state"][pid]["exp_avg_sq"].fill_(-1.0)

    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    for name in ("last.pt", "checks.json", "run_manifest.json"):
        (trainer.output_dir / name).write_bytes(f"previous {name}".encode())
        existing[name] = (trainer.output_dir / name).read_bytes()

    monkeypatch.setattr(trainer_module, "run_checks", _worse_than_any_candidate)
    trainer.check_every = 1

    with pytest.raises(TrainingError, match="exp_avg_sq"):
        trainer.run(max_updates=trainer.step + 1, verbose=False)

    assert not (trainer.output_dir / PUBLISH_MARKER).exists(), (
        "坏候选已经产生了发布副作用（标记）"
    )
    assert not (trainer.output_dir / "best.pt").exists(), "坏候选被写成了 best.pt"
    for name, before in existing.items():
        assert (trainer.output_dir / name).read_bytes() == before, (
            f"拒绝发布时覆盖了已有的 {name}"
        )


def test_refusing_a_negative_second_moment_leaves_everything_alone(monkeypatch, tmp_path):
    """拒绝要发生在**改动之前**：接收实例、训练状态、RNG、候选张量、文件字节。"""
    _, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "neg_atomic"
    )
    _mutate_payload(path, _poison_candidate_second_moment)
    before_bytes = path.read_bytes()

    fresh = _fresh_momentum(monkeypatch, tmp_path, "neg_atomic_r")
    before = _receiver_snapshot(fresh)
    with pytest.raises(TrainingError):
        fresh.load(path)
    after = _receiver_snapshot(fresh)

    assert _same_receiver(before, after), "拒绝路径改动了接收实例"
    assert path.read_bytes() == before_bytes, "拒绝路径改动了输入检查点"


# --------------------------------------------------------------------------
# 3. 合法情形不能被一起拦掉
# --------------------------------------------------------------------------


def test_a_legal_negative_first_moment_is_accepted(monkeypatch, tmp_path):
    """``exp_avg`` 是**一阶矩**，允许为负。

    把非负规则套到它身上，会把一段完全正常的训练历史判成损坏——这正是这一项
    最容易过头的方向。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "neg_avg"
    )

    def negative_first_moment(payload: dict) -> None:
        pid = _first_parameter_id(payload)
        payload["pending_best_candidate"]["optimizer"]["state"][pid]["exp_avg"].fill_(-0.5)

    mutated = _mutate_payload(path, negative_first_moment)
    pid = _first_parameter_id(mutated)
    assert bool((mutated["pending_best_candidate"]["optimizer"]["state"][pid]["exp_avg"] == -0.5).all())

    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())

    fresh = _fresh_momentum(monkeypatch, tmp_path, "neg_avg_r")
    fresh.load(path)
    assert fresh._best_dirty is True


def test_a_legal_zero_second_moment_is_accepted(monkeypatch, tmp_path):
    """零的二阶动量是合法的（还没积累过梯度的那一档），不是"负值"的边界。"""
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "zero_sq"
    )

    def zero_second_moment(payload: dict) -> None:
        for entry in payload["pending_best_candidate"]["optimizer"]["state"].values():
            entry["exp_avg_sq"].zero_()

    _mutate_payload(path, zero_second_moment)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())

    fresh = _fresh_momentum(monkeypatch, tmp_path, "zero_sq_r")
    fresh.load(path)


def test_legal_candidates_resume_with_finite_parameters_and_moments(monkeypatch, tmp_path):
    """合法空状态 + 合法带动量候选：恢复后走一次**真实更新**，全程有限。

    这是修前反例的**对照组**：同一条控制流（load → 发布 → 新实例恢复 →
    ``_update()``），只有状态是合法的。修前修后都必须通过。
    """
    # 空状态那一档：``_checkpoint_with_a_pending_candidate`` 的候选在第 0 步。
    _, empty_path = _checkpoint_with_a_pending_candidate(monkeypatch, tmp_path, "ok_empty")
    empty_payload = torch.load(empty_path, map_location="cpu", weights_only=False)
    assert empty_payload["pending_best_candidate"]["optimizer"]["state"] == {}
    empty_reader = _fresh(monkeypatch, tmp_path, "ok_empty_r")
    empty_reader.load(empty_path)
    empty_reader._update()
    empty_reader.step += 1
    assert _all_finite(empty_reader), "合法空状态恢复之后参数就非有限了"

    # 带动量那一档：完整链条。
    _, path = _checkpoint_with_a_candidate_that_has_momentum(monkeypatch, tmp_path, "ok_mom")
    publisher = _fresh_momentum(monkeypatch, tmp_path, "ok_mom_r")
    publisher.load(path)
    publisher.run(max_updates=publisher.step, verbose=False)
    best_path = publisher.output_dir / "best.pt"
    assert best_path.exists(), "合法候选没有走发布路径"

    fresh = _fresh_momentum(monkeypatch, tmp_path, "ok_mom_rr")
    fresh.load(best_path)
    assert _all_finite(fresh), "恢复 best.pt 之后参数就非有限了"
    fresh._update()
    fresh.step += 1
    assert _all_finite(fresh), "一次真实更新之后参数变成非有限值"


def _all_finite(trainer) -> bool:
    """参数与优化器的浮点状态是否全部有限。"""
    if not all(
        bool(torch.isfinite(value).all())
        for value in trainer.model.state_dict().values()
    ):
        return False
    for entry in trainer.optimizer.state_dict()["state"].values():
        for value in entry.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    return False
    return True


# --------------------------------------------------------------------------
# 4. 探针更新**之后**的结果检查
# --------------------------------------------------------------------------


def test_the_probe_refuses_a_step_that_yields_non_finite_without_raising(
    monkeypatch, tmp_path
):
    """探针分支的真实触发：状态**有限且非负**，但更新结果非有限。

    ``exp_avg = 1e308`` 配上 ``exp_avg_sq = 0``：两个值都有限、二阶矩也不是负的，
    于是非负语义那一条放它过去。Adam 的分母是 ``sqrt(0)/sqrt(bias) + eps``，更新
    量放大约 ``1e6`` 倍，参数直接变 ``inf``——**``step()`` 全程不抛异常**。

    这条测试证明"更新之后看结果"是**独立生效**的，不是被前置的非负检查顺带覆盖。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "probe_real"
    )

    def overwhelming_first_moment(payload: dict) -> None:
        pid = _first_parameter_id(payload)
        entry = payload["pending_best_candidate"]["optimizer"]["state"][pid]
        entry["exp_avg"].fill_(1.0e308)
        entry["exp_avg_sq"].zero_()

    mutated = _mutate_payload(path, overwhelming_first_moment)
    pid = _first_parameter_id(mutated)
    entry = mutated["pending_best_candidate"]["optimizer"]["state"][pid]
    assert bool(torch.isfinite(entry["exp_avg"]).all()), "一阶矩必须仍然是有限的"
    assert bool((entry["exp_avg_sq"] >= 0).all()), "二阶矩必须仍然非负——否则测到的是另一条检查"

    payload = torch.load(path, map_location="cpu", weights_only=False)
    with pytest.raises(TrainingError, match="non-finite value on the very next update"):
        require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())


def test_the_probe_branch_refuses_an_injected_non_finite_step(monkeypatch, tmp_path):
    """**探针分支的专门测试（受控故障注入）。** 状态完全合法，问题出在 `step()`。

    上面那条走的是真实 Adam 会走到的路径；这一条故意把 ``step()`` 换成"不抛异常、
    但把参数写成 NaN"的实现，用来单独证明**这个分支本身**有效——正如复核提醒的，
    只靠负二阶动量那几条测试，可能只测到了前置的非负检查。

    注入只影响本进程里这个测试，``monkeypatch`` 结束即还原；被验证的状态是
    **合法**的，所以能走到探针那一步。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    import copy as _copy

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "probe_inject"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    real_step = torch.optim.Adam.step

    def poisoned_step(self, closure=None):
        result = real_step(self, closure)
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.data.fill_(float("nan"))
        return result

    before_model = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    before_optimizer = _copy.deepcopy(trainer.optimizer.state_dict())
    before_rng = _global_rng_snapshot()

    # 用**独立的** MonkeyPatch：注入只在这几行里生效，也不会顺手撤掉夹具自己的
    # monkeypatch（``_tiny`` 把阶段判据换成了空集）。
    injector = pytest.MonkeyPatch()
    injector.setattr(torch.optim.Adam, "step", poisoned_step)
    try:
        with pytest.raises(TrainingError, match="non-finite value on the very next update"):
            require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())
    finally:
        injector.undo()

    # 探针**隔离**：真模型、真优化器、全局随机状态都没被碰过。
    assert all(
        torch.equal(before_model[key], value)
        for key, value in trainer.model.state_dict().items()
    ), "探针污染了真模型的参数"
    assert _same_optimizer(before_optimizer, trainer.optimizer.state_dict()), (
        "探针污染了真优化器的状态"
    )
    assert _same_global_rng(before_rng, _global_rng_snapshot()), "校验改动了全局随机状态"


def test_the_probe_branch_refuses_an_injected_non_finite_optimizer_state(
    monkeypatch, tmp_path
):
    """**探针分支测试（受控故障注入，状态侧）。** 参数是好的，优化器状态被写坏。

    与上一条一对：检查要覆盖"探针参数"和"探针优化器浮点状态"两边。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "probe_state_inject"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    real_step = torch.optim.Adam.step

    def poisoned_state_step(self, closure=None):
        result = real_step(self, closure)
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    value.fill_(float("nan"))
        return result

    injector = pytest.MonkeyPatch()
    injector.setattr(torch.optim.Adam, "step", poisoned_state_step)
    try:
        with pytest.raises(TrainingError, match="floating-point state non-finite"):
            require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())
    finally:
        injector.undo()


def test_validating_a_candidate_does_not_touch_its_tensors_or_the_global_rng(
    monkeypatch, tmp_path
):
    """**只读性与隔离性。** 校验一份**合法**候选时，输入张量与全局随机状态不变。

    探针会在隔离的优化器上真走一步；``load_state_dict`` 在同 dtype 时会把张量
    **原样接过去**，所以不深拷贝的话那一步的原地更新就改到了候选自己的动量上。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "readonly"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    before_state = {
        pid: {key: value.detach().clone() for key, value in entry.items()}
        for pid, entry in payload["pending_best_candidate"]["optimizer"]["state"].items()
    }
    before_model = {
        key: value.detach().clone()
        for key, value in payload["pending_best_candidate"]["model"].items()
    }
    before_rng = _global_rng_snapshot()

    require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())

    for pid, entry in payload["pending_best_candidate"]["optimizer"]["state"].items():
        for key, value in entry.items():
            assert torch.equal(before_state[pid][key], value), (
                f"校验改动了输入候选的优化器状态 {pid}.{key}"
            )
    for key, value in payload["pending_best_candidate"]["model"].items():
        assert torch.equal(before_model[key], value), f"校验改动了输入候选的参数 {key}"
    assert _same_global_rng(before_rng, _global_rng_snapshot()), "校验改动了全局随机状态"


# ==========================================================================
# 第五项：步号字段的**存在性、类型与范围**
# ==========================================================================
#
# 复核实测（修前）：真实流程登记候选并 `_save()` 之后，对独立副本做
#
#   A. 删除 `pending_best_candidate.best_check.step`
#   B. 把它改成 `"broken"`
#
# 两件事都能：共享候选校验通过 → `Trainer.load()` 通过 → `run()` 发布 `best.pt`
# （发布文件里的 `best_check.step` 仍然缺失 / 为 `"broken"`）→ 新实例再次加载成功。
# 另有一档：**已发布**的 `best.pt` 没有待发布候选，`best_dirty=False` 让
# `require_best_candidate_consistency()` 在第一道判据就返回，于是文件**自身**的
# `best_check.step` 完全没人看。
#
# 根因就在 `_require_one_moment()`：非整数步号直接 `return`/`continue`，注释说
# "后续状态校验负责类型检查"——但后续只查了快照自身的 `step`。
#
# 现在的规则：三处步号**各自**先验原始类型与范围（不 `int()` 强转），再比数值；
# 快照自带的 `best_check.step` 在共享只读校验里也验。

#: 三处步号：名字 → 取出承载它的 mapping 的函数。
_STEP_LOCATIONS = {
    "candidate.step": lambda payload: payload["pending_best_candidate"],
    "candidate.best_check.step": lambda payload: payload["pending_best_candidate"]["best_check"],
    "best_check.step": lambda payload: payload["best_check"],
}

_ABSENT = object()

#: 有区别的坏值。**不**用同一个变异重复堆数量：缺失、None、字符串、浮点、布尔
#: （Python 与 NumPy）、范围各自是一类问题，而且 `int()` 分别会把 "1"、1.0、True
#: 悄悄变成 1。
_BAD_STEP_VALUES = {
    "missing": _ABSENT,
    "none": None,
    "string": "broken",
    "float": 1.0,
    "bool": True,
    "numpy_bool": np.bool_(True),
    "negative": -1,
}


def _apply_step_mutation(payload: dict, location: str, mutation: str) -> None:
    holder = _STEP_LOCATIONS[location](payload)
    value = _BAD_STEP_VALUES[mutation]
    if value is _ABSENT:
        holder.pop("step", None)
    else:
        holder["step"] = value


def _step_triple(payload: dict) -> tuple:
    return (
        payload["pending_best_candidate"].get("step"),
        payload["pending_best_candidate"]["best_check"].get("step"),
        payload.get("best_check", {}).get("step"),
    )


@pytest.mark.parametrize("location", sorted(_STEP_LOCATIONS))
@pytest.mark.parametrize("mutation", sorted(_BAD_STEP_VALUES))
def test_every_step_field_must_be_a_valid_integer(monkeypatch, tmp_path, location, mutation):
    """三处步号**各自**：缺失 / None / 字符串 / 浮点 / 布尔 / 负值都必须被拒绝。

    共享校验与 ``Trainer.load()`` 都要拦；报错要指出**字段路径**。
    修前：``candidate.best_check.step`` 与 ``best_check.step`` 这两处根本没验。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, f"step_{location}_{mutation}".replace(".", "_")
    )
    mutated = _mutate_payload(path, lambda payload: _apply_step_mutation(payload, location, mutation))

    # 变异确实生效（缺失与其余两种写法分开断言）。
    holder = _STEP_LOCATIONS[location](mutated)
    if mutation == "missing":
        assert "step" not in holder, "变异没有生效"
    else:
        assert holder["step"] is not None or mutation == "none", "变异没有生效"
        expected = _BAD_STEP_VALUES[mutation]
        assert type(holder["step"]) is type(expected), (
            f"变异没有生效：{holder['step']!r} 的类型是 {type(holder['step']).__name__}"
        )

    with pytest.raises(TrainingError) as shared_exc:
        require_best_candidate_consistency(
            torch.load(path, map_location="cpu", weights_only=False),
            path,
            **trainer._snapshot_rules(),
        )
    assert location in str(shared_exc.value), (
        f"共享校验的报错没有指出字段路径 {location!r}：{shared_exc.value}"
    )

    fresh = _fresh_momentum(monkeypatch, tmp_path, f"step_r_{location}_{mutation}".replace(".", "_"))
    with pytest.raises(TrainingError) as load_exc:
        fresh.load(path)
    assert location in str(load_exc.value), (
        f"load 的报错没有指出字段路径 {location!r}：{load_exc.value}"
    )


@pytest.mark.parametrize("location", sorted(_STEP_LOCATIONS))
def test_three_legal_but_disagreeing_steps_are_refused(monkeypatch, tmp_path, location):
    """三处都是**合法整数**、但数值不一致：仍然是坏输入。

    这一条与上一组分工明确：上一组测"类型/范围"，这一组测"关系"。修前
    ``_require_one_moment()`` 只比了 ``candidate.step`` 与另外两处，而那两处
    一旦不是整数就被 `continue` 掉——现在类型先过，数值再比。
    """
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, f"disagree_{location}".replace(".", "_")
    )
    pristine = torch.load(path, map_location="cpu", weights_only=False)
    assert _step_triple(pristine) == (1, 1, 1), (
        f"夹具的三处步号本应都是 1：{_step_triple(pristine)}"
    )

    mutated = _mutate_payload(
        path, lambda payload: _STEP_LOCATIONS[location](payload).__setitem__("step", 0)
    )
    assert 0 in _step_triple(mutated), "变异没有生效"

    with pytest.raises(TrainingError, match="different moments"):
        require_best_candidate_consistency(
            torch.load(path, map_location="cpu", weights_only=False),
            path,
            **trainer._snapshot_rules(),
        )


@pytest.mark.parametrize("mutation", ["missing", "string", "float", "bool", "none"])
def test_a_published_checkpoint_with_a_corrupt_best_record_step_is_refused(
    monkeypatch, tmp_path, mutation
):
    """**已发布**的 `best.pt` 没有待发布候选，也要查它自己的最佳记录。

    `best_dirty=False` 让 `require_best_candidate_consistency()` 直接返回；文件自身
    的 `best_check.step` 因此必须在共享只读校验里验，不能因为"没有候选"就跳过。
    """
    _, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, f"pub_{mutation}"
    )
    publisher = _fresh_momentum(monkeypatch, tmp_path, f"pub_{mutation}_r")
    publisher.load(path)
    publisher.run(max_updates=publisher.step, verbose=False)
    best_path = publisher.output_dir / "best.pt"
    assert best_path.exists(), "没有先发布出一份合法 best.pt"

    def corrupt(payload: dict) -> None:
        value = _BAD_STEP_VALUES[mutation]
        if value is _ABSENT:
            payload["best_check"].pop("step", None)
        else:
            payload["best_check"]["step"] = value

    mutated = _mutate_payload(best_path, corrupt)
    assert mutated["pending_best_candidate"] is None and mutated["best_dirty"] is False, (
        "这条测试的前提是'没有待发布候选'"
    )
    if mutation == "missing":
        assert "step" not in mutated["best_check"]
    else:
        assert type(mutated["best_check"]["step"]) is type(_BAD_STEP_VALUES[mutation])

    fresh = _fresh_momentum(monkeypatch, tmp_path, f"pub_{mutation}_reader")
    with pytest.raises(TrainingError) as excinfo:
        fresh.load(best_path)
    assert "best_check.step" in str(excinfo.value), str(excinfo.value)


# --------------------------------------------------------------------------
# 合法情形：初始无最佳记录、整数步号、较早候选、发布后恢复并预测
# --------------------------------------------------------------------------


def test_a_checkpoint_without_a_best_record_is_still_legal(monkeypatch, tmp_path):
    """**合法初始状态**：还没产生过最佳记录时 ``best_check`` 是 ``None``。

    写入端在这种情况下写的就是 ``None``，不是缺键——把 ``None`` 判成损坏会让
    "刚开始训练就中断"这条最普通的路走不通。
    """
    trainer = _momentum_trainer(monkeypatch, tmp_path, "no_best")
    trainer._update()
    trainer.step += 1
    path = trainer.output_dir / "no_best.pt"
    trainer._save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["best_check"] is None, "夹具没有造出'尚无最佳记录'的状态"
    assert payload["pending_best_candidate"] is None
    assert payload["best_dirty"] is False

    fresh = _fresh_momentum(monkeypatch, tmp_path, "no_best_r")
    fresh.load(path)
    assert fresh.best is None


def test_legal_integer_steps_and_an_earlier_candidate_still_load(monkeypatch, tmp_path):
    """合法整数步号通过；候选来自更早一步、参数**确实**与当前模型不同，也通过。"""
    _, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "legal_steps"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert _step_triple(payload) == (1, 1, 1), f"夹具的步号三元组变了：{_step_triple(payload)}"
    assert int(payload["pending_best_candidate"]["step"]) < int(payload["step"]), (
        "候选应当来自更早的一步"
    )
    assert any(
        not torch.equal(payload["pending_best_candidate"]["model"][key], payload["model"][key])
        for key in payload["model"]
    ), "候选参数与当前参数一模一样——测不到'允许不同'"

    fresh = _fresh_momentum(monkeypatch, tmp_path, "legal_steps_r")
    fresh.load(path)
    assert fresh._best_dirty is True
    assert int(fresh._best_candidate["step"]) == 1


def test_a_legal_candidate_publishes_and_the_published_steps_are_intact(monkeypatch, tmp_path):
    """合法候选发布后：三处步号仍在、新实例可恢复、预测入口可用。"""
    _, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "legal_publish"
    )
    publisher = _fresh_momentum(monkeypatch, tmp_path, "legal_publish_r")
    publisher.load(path)
    publisher.run(max_updates=publisher.step, verbose=False)

    best_path = publisher.output_dir / "best.pt"
    assert best_path.exists()
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert best["best_check"]["step"] == best["step"] == 1
    assert best["best_check"]["step"] == best["candidate"]["step"]

    fresh = _fresh_momentum(monkeypatch, tmp_path, "legal_publish_rr")
    fresh.load(best_path)
    model, predict_payload = predict_module.load_model_from_checkpoint(
        best_path, charge_radius=_radius(publisher)
    )
    assert model is not None
    assert float(predict_payload[DETONATION_VELOCITY_FIELD]) == pytest.approx(FROZEN_D)


# --------------------------------------------------------------------------
# 拒绝的时机：接收实例之前、发布副作用之前
# --------------------------------------------------------------------------


def test_refusing_a_bad_step_leaves_the_receiver_and_the_file_untouched(monkeypatch, tmp_path):
    """拒绝发生在**改动接收实例之前**：模型、完整优化器、步号、最佳记录、候选、
    标志、历史、采样器与全局 RNG 都不变；输入文件字节也不变。"""
    _, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "step_atomic"
    )
    _mutate_payload(
        path,
        lambda payload: payload["pending_best_candidate"]["best_check"].__setitem__(
            "step", "broken"
        ),
    )
    before_bytes = path.read_bytes()

    fresh = _fresh_momentum(monkeypatch, tmp_path, "step_atomic_r")
    before = _receiver_snapshot(fresh)
    with pytest.raises(TrainingError):
        fresh.load(path)
    after = _receiver_snapshot(fresh)

    assert _same_receiver(before, after), "拒绝路径改动了接收实例"
    assert path.read_bytes() == before_bytes, "拒绝路径改动了输入检查点"


def test_a_corrupt_in_memory_step_is_refused_before_begin_publish(monkeypatch, tmp_path):
    """加载合法检查点之后**在内存里**改坏候选步号：发布路径必须拒绝。

    **直接监测 `_begin_publish()` 未被调用**，而不是靠"最后没有标记文件"推断——
    回滚也可能把标记删掉，那两种情况的证据强度不一样。
    """
    from test_best_candidate_and_transaction import trainer_module

    trainer, _ = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "step_mem"
    )
    assert trainer._best_dirty and trainer._best_candidate is not None
    trainer._best_candidate["best_check"]["step"] = "broken"

    starts: list[list[str]] = []
    real_begin = trainer._begin_publish

    def spy_begin(names):
        starts.append(list(names))
        return real_begin(names)

    monkeypatch.setattr(trainer, "_begin_publish", spy_begin)
    monkeypatch.setattr(trainer_module, "run_checks", _worse_than_any_candidate)
    trainer.check_every = 1

    with pytest.raises(TrainingError, match="pending_best_candidate.best_check.step"):
        trainer.run(max_updates=trainer.step + 1, verbose=False)

    assert starts == [], f"_begin_publish() 被调用了：{starts}"
    assert not (trainer.output_dir / PUBLISH_MARKER).exists()
    assert not (trainer.output_dir / "best.pt").exists()


# --------------------------------------------------------------------------
# 只读性：校验不得修改输入载荷（含嵌套张量）
# --------------------------------------------------------------------------


def _deep_equal(left, right) -> bool:
    """递归比较，**张量逐元素**。

    不能对含多元素 Tensor 的字典直接用 ``==``：那会抛
    ``Boolean value of Tensor with more than one value is ambiguous``。
    """
    from collections.abc import Mapping

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _deep_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _deep_equal(a, b) for a, b in zip(left, right)
        )
    if torch.is_tensor(left) and torch.is_tensor(right):
        return (
            tuple(left.shape) == tuple(right.shape)
            and left.dtype == right.dtype
            and bool(torch.equal(left, right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
    # ``best_check.score`` 里带 NaN 是合法的；``nan != nan``，直接 ``==`` 会把
    # 一份**没被碰过**的载荷判成"被修改了"。
    import math

    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_validation_does_not_modify_the_input_payload(monkeypatch, tmp_path):
    """校验一份**合法**载荷不改变它：模型张量、优化器动量、步号、最佳记录……"""
    from spherical_blast_pinn.training.checkpoint import (
        require_best_candidate_consistency,
    )

    trainer, path = _checkpoint_with_a_candidate_that_has_momentum(
        monkeypatch, tmp_path, "readonly_step"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    import copy as _copy

    snapshot = _copy.deepcopy(payload)
    # 比较器本身要先能"看出"差异，否则下面的断言没有意义。
    probe = _copy.deepcopy(payload)
    probe["pending_best_candidate"]["optimizer"]["state"][
        _first_parameter_id(probe)
    ]["exp_avg"].view(-1)[0] += 1.0
    assert not _deep_equal(probe, snapshot), "比较器看不出张量差异"

    require_best_candidate_consistency(payload, path, **trainer._snapshot_rules())

    assert _deep_equal(payload, snapshot), "校验修改了输入载荷"
