r"""目标函数层。

:mod:`terms` 给出各项损失（初始条件、接触面、主激波、物理解、数据监督、PDE），
:mod:`objective` 按提示词 §10 把它们装配成

.. math::

    L_i = \lambda_{PDE} L_{PDE} + 10 L_{IC} + 10 L_c + 10 L_s + 1 L_d + 1 L_{adm},
    \qquad L = \frac{L_{50} + L_{80} + L_{100}}{3}

本层不含任何规范外的项：没有弱形式、没有全局积分守恒训练项、没有经验曲线
损失。整体质量与能量平衡只作为诊断，不进梯度。
"""
from __future__ import annotations

from .objective import (
    BASE_WEIGHTS,
    TUNABLE,
    LossBreakdown,
    case_loss,
    contact_pressure_scale,
    total_loss,
)
from .terms import (
    LossError,
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

__all__ = [
    "BASE_WEIGHTS",
    "TUNABLE",
    "LossBreakdown",
    "LossError",
    "TermBreakdown",
    "admissibility_loss",
    "case_loss",
    "contact_loss",
    "contact_pressure_scale",
    "data_loss",
    "geometry_loss",
    "initial_condition_loss",
    "pde_loss",
    "pde_states",
    "shock_loss",
    "total_loss",
]
