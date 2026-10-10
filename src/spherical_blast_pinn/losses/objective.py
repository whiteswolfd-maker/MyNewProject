"""总损失（提示词 §10）。

.. math::

    L_i = \\lambda_{PDE} L_{PDE} + 10 L_{IC} + 10 L_c + 10 L_s + 1 L_d + 1 L_{adm}

.. math::

    L = \\frac{L_{50} + L_{80} + L_{100}}{3}

三条约化规则，缺一不可：

1. **组内先均值**：每一项在自己的样本上取均值，所以采样数量不改变隐含权重。
2. **半径等权**：先对每个半径算出 ``L_i``，再对半径取平均。三个半径在每次
   参数更新里都参与且等权。
3. **不引入规范外的项**：没有弱形式、没有全局积分守恒训练项、没有经验曲线
   损失。整体质量与能量平衡只作为**诊断**报告，不进梯度。

第一版固定基础权重，不自动加入权重学习算法。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..sampling.batch import SampleBatch
from .terms import (
    TermBreakdown,
    admissibility_loss,
    contact_loss,
    data_loss,
    geometry_loss,
    initial_condition_loss,
    pde_loss,
    pde_states,
    shock_loss,
)

#: 基础权重，逐字对应提示词 §10。
BASE_WEIGHTS: dict[str, float] = {
    "pde": 1.0,
    "initial": 10.0,
    "contact": 10.0,
    "shock": 10.0,
    "data": 1.0,
    "admissibility": 1.0,
    # L_geometry（§11.2）与独立数据、admissibility 并列，单独加权——不并入
    # 数据组，否则会悄悄改变产物/空气/前沿三组的等权关系。
    "geometry": 1.0,
}

#: ``lambda_PDE`` 的三个阶段值由训练阶段提供，其余权重恒定。
TUNABLE = ("pde",)


@dataclass
class LossBreakdown:
    """一个优化步的总损失及其全部监控量。"""

    total: Any
    per_case: dict[str, float]
    weights: dict[str, float]
    terms: dict[str, float]
    monitors: dict[str, Any] = field(default_factory=dict)
    #: 各损失组**未加权**的半径平均张量，供有界梯度平衡测量 ``d(组)/d(theta)``
    #: （§14）。``terms`` 是它们的 ``float()``，只能看数值、不能求导，所以两组
    #: 数据都要留。某一组没有可用样本时，它的张量是一个 ``requires_grad=False``
    #: 的常数——这正是平衡器判断"该组本步不存在"的依据，不必再另设一个标志位。
    group_tensors: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "per_case": self.per_case,
            "weights": self.weights,
            "terms": self.terms,
            "monitors": self.monitors,
        }


def contact_pressure_scale(
    reference, *, charge_radius: float, detonation_velocity: float
) -> float:
    """``Pc_*``：参考 1 µs 初值场在接触面上的正值压力，Pa。

    提取方法固定为：取参考场在**交接时刻** ``T_h = T_b + epsilon`` 时刻、接触面
    **产物侧**最后一个单元的状态，用一般 JWL 算出压力。它不来自验证真值，也不沿用
    任何其它时刻的界面压力。

    方案 B 下 ``T_h`` 才是 PINN 的起点：A 用的 ``epsilon``（50 mm 时 1 µs）对应
    "整球已经是均匀产物"，而真实算例到 ``T_b``（7.215 µs）爆轰波才走到表面。

    为什么必须有一个共用尺度：两条前沿的压力尺度相差三个数量级（产物
    ``rho_*,p U_*^2 = 6.0e9``，空气 ``rho_*,a U_*^2 = 4.5e6``），直接令两个
    无量纲压力输出相等是没有意义的。
    """
    from ..physics.similitude import handover_time

    start = handover_time(charge_radius, detonation_velocity=detonation_velocity)
    value = float(reference.contact_pressure_at(start))
    if not (value > 0.0):
        raise ValueError(
            f"the reference contact pressure came out non-positive ({value} Pa); "
            "the reference field is not usable as an initial-condition source"
        )
    return value


def case_loss(
    batch: SampleBatch,
    model,
    *,
    weights: Mapping[str, float],
    contact_pressure: float,
) -> tuple[
    Any,
    TermBreakdown,
    TermBreakdown,
    TermBreakdown,
    TermBreakdown,
    TermBreakdown,
    TermBreakdown,
    TermBreakdown,
]:
    """一个半径的 ``L_i`` 与五项分解。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")

    pde = pde_loss(batch, model)
    initial = initial_condition_loss(batch, model)
    contact = contact_loss(batch, model, contact_pressure=contact_pressure)
    shock = shock_loss(batch, model)
    data = data_loss(batch, model)
    geometry = geometry_loss(batch, model)
    product_state, air_state = pde_states(batch, model)
    admissibility = admissibility_loss(
        batch, model, product_fields=product_state, air_fields=air_state
    )

    total = (
        float(weights.get("pde", 1.0)) * pde.total
        + float(weights.get("initial", 10.0)) * initial.total
        + float(weights.get("contact", 10.0)) * contact.total
        + float(weights.get("shock", 10.0)) * shock.total
        + float(weights.get("data", 1.0)) * data.total
        + float(weights.get("geometry", 1.0)) * geometry.total
        + float(weights.get("admissibility", 1.0)) * admissibility.total
    )
    return total, pde, initial, contact, shock, data, geometry, admissibility


def total_loss(
    batches: Sequence[SampleBatch],
    model,
    *,
    contact_pressure: float,
    pde_weight: float = 1.0,
    weights: Mapping[str, float] | None = None,
) -> LossBreakdown:
    """三个半径等权平均后的总损失。

    ``pde_weight`` 就是 ``lambda_PDE``：它在建立初值阶段是 0.1，在增强阶段平滑
    升到 1，在完整联合训练阶段固定在 1。其余权重不随阶段变。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    if not batches:
        raise ValueError("total_loss needs at least one radius batch")

    effective = dict(BASE_WEIGHTS)
    if weights:
        effective.update({k: float(v) for k, v in weights.items()})
    effective["pde"] = float(pde_weight)

    totals: list[Any] = []
    per_case: dict[str, float] = {}
    terms: dict[str, float] = {}
    monitors: dict[str, Any] = {}
    group_values: dict[str, list[Any]] = {}

    for batch in batches:
        value, pde, initial, contact, shock, data, geometry, admissibility = case_loss(
            batch, model, weights=effective, contact_pressure=contact_pressure
        )
        totals.append(value)
        per_case[batch.case_id] = float(value.detach())
        for name, breakdown in (
            ("pde", pde),
            ("initial", initial),
            ("contact", contact),
            ("shock", shock),
            ("data", data),
            ("geometry", geometry),
            ("admissibility", admissibility),
        ):
            terms[f"{batch.case_id}.{name}"] = float(breakdown.total.detach())
            group_values.setdefault(name, []).append(breakdown.total)
            for key, monitor in breakdown.monitors.items():
                monitors[f"{batch.case_id}.{key}"] = (
                    float(monitor) if isinstance(monitor, (int, float)) else monitor
                )

    total = sum(totals) / len(totals)
    # 半径等权（与 ``total`` 同一条约化规则），但**不加权**：平衡器要量的正是
    # "这一组自己有多大"。
    group_tensors = {
        name: sum(values) / len(values) for name, values in group_values.items()
    }
    return LossBreakdown(
        total=total,
        per_case=per_case,
        weights=effective,
        terms=terms,
        monitors=monitors,
        group_tensors=group_tensors,
    )


__all__ = [
    "BASE_WEIGHTS",
    "TUNABLE",
    "LossBreakdown",
    "case_loss",
    "contact_pressure_scale",
    "total_loss",
]
