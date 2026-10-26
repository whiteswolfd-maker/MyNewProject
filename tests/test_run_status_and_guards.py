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






# ==========================================================================
# §7 精修后不能沿用旧模型的检查结果
# ==========================================================================








# ==========================================================================
# §8 非有限损失/梯度的更新前检查
# ==========================================================================




