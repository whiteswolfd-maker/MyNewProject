"""运行终态、评估覆盖、恢复事务性与数值守卫（修改提示词 §1、§2、§3、§6、§7、§8）。

这一组走**真实调用路径**：成功的判定走真实的 ``run()`` 控制流，恢复走真实的
``load()``。只构造结果类或只调校验函数的测试
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






# ==========================================================================
# §7 精修后不能沿用旧模型的检查结果
# ==========================================================================








# ==========================================================================
# §8 非有限损失/梯度的更新前检查
# ==========================================================================




