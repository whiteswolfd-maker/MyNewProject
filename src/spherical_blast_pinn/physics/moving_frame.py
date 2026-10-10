"""移动坐标与物理导数链式法则。

提示词 §6 要求的两个坐标映射：

.. math::

    \\xi = \\frac{r}{R_c} \\quad (0 < \\xi < 1, \\text{产物域}) \\\\
    h = R_s - R_c, \\qquad \\eta = \\frac{r - R_c}{h} \\quad (0 < \\eta < 1, \\text{空气域})

以及它们带来的物理导数：

对 ``f(s, xi)``：

.. math::

    \\partial_r f &= \\frac{f_\\xi}{R_c} \\\\
    \\left.\\partial_t f\\right|_r &= \\frac{U_*}{R_0 \\hat T_*} f_s
        - \\xi \\frac{\\dot R_c}{R_c} f_\\xi

对 ``f(s, eta)``：

.. math::

    \\partial_r f &= \\frac{f_\\eta}{h} \\\\
    \\left.\\partial_t f\\right|_r &= \\frac{U_*}{R_0 \\hat T_*} f_s
        - \\frac{\\dot R_c + \\eta (\\dot R_s - \\dot R_c)}{h} f_\\eta

``t_phys`` 与 ``t_PINN`` 在固定半径下只差常数，所以两者的时间导数相同。

前沿速度由前沿网络对 ``s`` 求导得到：

.. math::

    \\dot R_c = \\frac{U_*}{\\hat T_*} \\frac{d\\hat R_c}{ds}, \\qquad
    \\dot R_s = \\frac{U_*}{\\hat T_*} \\frac{d\\hat R_s}{ds}

**不要把这里的链式项和自动求导重复叠加一次。** 本模块提供两条路径：

* :func:`physical_derivatives` —— 显式链式法则，生产路径。它接受网络对
  ``(s, xi)`` 的偏导，因此不依赖 ``r`` 或 ``t`` 是否是可微叶子。
* :func:`autograd_physical_derivatives` —— 把 ``(r, t)`` 作为叶子、用可微
  包装器直接求导。它是**检验路径**：两条路径必须给出同一个数，
  ``tests/test_moving_frame.py`` 用解析光滑函数把这一点钉死。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .similitude import SimilitudeScales

PRODUCT = "product"
AIR = "air"


class MovingFrameError(ValueError):
    """移动坐标不合法（前沿次序错误、零厚度等）。"""


@dataclass
class FrontData:
    """一个时间点上的两个前沿及其物理速度。全部 SI。"""

    contact_radius: Any
    shock_radius: Any
    contact_speed: Any
    shock_speed: Any

    @property
    def gap(self):
        return self.shock_radius - self.contact_radius

    def as_dict(self) -> dict[str, Any]:
        return {
            "R_c": self.contact_radius,
            "R_s": self.shock_radius,
            "dR_c_dt": self.contact_speed,
            "dR_s_dt": self.shock_speed,
            "h": self.gap,
        }


def front_speeds_from_network_derivative(
    d_hat_contact_ds, d_hat_shock_ds, scales: SimilitudeScales
):
    """``Rc_dot = (U_*/hat T_*) d(hat Rc)/ds``，``Rs_dot`` 同理。"""
    factor = scales.reference_velocity / scales.time_scale
    return factor * d_hat_contact_ds, factor * d_hat_shock_ds


def check_moving_frame(
    fronts: FrontData,
    scales: SimilitudeScales,
    *,
    name: str = "",
    charge_radius: float | None = None,
) -> None:
    """前沿次序与有限性检查。

    非法移动域必须**硬失败**，不能警告后继续：``R_s <= R_c`` 会让空气域
    为空，而空域上的残差是精确的零，在损失里读起来和"方程已满足"一模一样。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    label = f"{name}: " if name else ""
    for field in ("contact_radius", "shock_radius", "contact_speed", "shock_speed"):
        value = getattr(fronts, field)
        if not bool(torch.isfinite(value).all()):
            raise MovingFrameError(f"{label}{field} contains non-finite values")
    if not bool((fronts.contact_radius > 0.0).all()):
        raise MovingFrameError(f"{label}R_c must stay positive")
    if not bool((fronts.shock_radius > fronts.contact_radius).all()):
        raise MovingFrameError(
            f"{label}R_s <= R_c at some points; the disturbed-air domain would be empty"
        )


# --------------------------------------------------------------------------
# 相似坐标
# --------------------------------------------------------------------------


def product_coordinate(radius, front_radius):
    """``xi = r / R_c``。"""
    return radius / front_radius


def air_coordinate(radius, front_radius, gap):
    """``eta = (r - R_c) / h``。"""
    return (radius - front_radius) / gap


def network_time(physical_time, scales: SimilitudeScales):
    """``s = hat t / hat T_*``，其中 ``hat t = U_* t_PINN / R_0``。"""
    return scales.network_time(physical_time)


# --------------------------------------------------------------------------
# 显式链式法则：生产路径
# --------------------------------------------------------------------------


@dataclass
class ChainFactors:
    """``d(coord)/dr`` 与 ``d(coord)/dt|_r``，加上时间项系数。"""

    coordinate: str
    d_coord_dr: Any
    d_coord_dt: Any
    d_time_ds: float
    coordinate_value: Any

    def physical_derivatives(self, f_s, f_coord):
        """由 ``f_s``、``f_coord`` 得到 ``(f_r, f_t)``。"""
        f_r = f_coord * self.d_coord_dr
        f_t = self.d_time_ds * f_s + f_coord * self.d_coord_dt
        return f_r, f_t


def product_chain_factors(fronts: FrontData, scales: SimilitudeScales, radius) -> ChainFactors:
    """产物域 ``xi`` 的链式系数。

    .. math::

        \\partial_r \\xi = 1/R_c, \\qquad
        \\partial_t \\xi|_r = -\\xi \\dot R_c / R_c
    """
    xi = product_coordinate(radius, fronts.contact_radius)
    d_coord_dr = 1.0 / fronts.contact_radius
    d_coord_dt = -xi * fronts.contact_speed / fronts.contact_radius
    return ChainFactors(
        coordinate="xi",
        d_coord_dr=d_coord_dr,
        d_coord_dt=d_coord_dt,
        d_time_ds=scales.reference_velocity / (scales.charge_radius * scales.time_scale),
        coordinate_value=xi,
    )


def air_chain_factors(fronts: FrontData, scales: SimilitudeScales, radius) -> ChainFactors:
    """空气域 ``eta`` 的链式系数。

    .. math::

        \\partial_r \\eta = 1/h, \\qquad
        \\partial_t \\eta|_r = -\\frac{\\dot R_c + \\eta(\\dot R_s - \\dot R_c)}{h}
    """
    gap = fronts.gap
    eta = air_coordinate(radius, fronts.contact_radius, gap)
    d_coord_dr = 1.0 / gap
    d_coord_dt = -(fronts.contact_speed + eta * (fronts.shock_speed - fronts.contact_speed)) / gap
    return ChainFactors(
        coordinate="eta",
        d_coord_dr=d_coord_dr,
        d_coord_dt=d_coord_dt,
        d_time_ds=scales.reference_velocity / (scales.charge_radius * scales.time_scale),
        coordinate_value=eta,
    )


def chain_factors(medium: str, fronts: FrontData, scales: SimilitudeScales, radius) -> ChainFactors:
    if medium == PRODUCT:
        return product_chain_factors(fronts, scales, radius)
    if medium == AIR:
        return air_chain_factors(fronts, scales, radius)
    raise MovingFrameError(f"unknown medium {medium!r}")


def physical_derivatives(f_s, f_coord, factors: ChainFactors):
    """一层标量场 ``f(s, coord)`` 的 ``(f_r, f_t)``。"""
    return factors.physical_derivatives(f_s, f_coord)


# --------------------------------------------------------------------------
# 检验路径：``(r, t)`` 作为叶子直接求导
# --------------------------------------------------------------------------


def autograd_physical_derivatives(
    field_fn,
    radius,
    physical_time,
    scales: SimilitudeScales,
    fronts_fn,
    *,
    create_graph: bool = True,
):
    """把 ``(r, t_phys)`` 作为可微叶子，由可微包装器直接求物理导数。

    这是**独立于显式链式法则**的第二条路径。两条路径必须一致；用其中一条
    去验证另一条，比任何一条自己和自己比都可信。

    ``field_fn(s, coord)`` 接受网络输入；``fronts_fn(t_phys)`` 返回
    :class:`FrontData`，且其 ``R_c``/``R_s`` 必须对 ``t_phys`` 可微。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")

    radius = _as_leaf(radius)
    physical_time = _as_leaf(physical_time)
    fronts = fronts_fn(physical_time)

    s = network_time(physical_time, scales)
    xi = product_coordinate(radius, fronts.contact_radius)
    eta = air_coordinate(radius, fronts.contact_radius, fronts.gap)
    return {"s": s, "xi": xi, "eta": eta, "fronts": fronts}


def _as_leaf(value):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    if not isinstance(value, torch.Tensor):
        raise TypeError("autograd derivatives need tensor inputs")
    return value.detach().requires_grad_(True)


def differentiate_scalar(field, wrt) -> Any:
    """``d field / d wrt``，对 ``wrt=None`` 返回零。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    (grad,) = torch.autograd.grad(
        field,
        wrt,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    return grad if grad is not None else torch.zeros_like(field)


__all__ = [
    "AIR",
    "PRODUCT",
    "ChainFactors",
    "FrontData",
    "MovingFrameError",
    "air_chain_factors",
    "air_coordinate",
    "autograd_physical_derivatives",
    "chain_factors",
    "check_moving_frame",
    "differentiate_scalar",
    "front_speeds_from_network_derivative",
    "network_time",
    "physical_derivatives",
    "product_chain_factors",
    "product_coordinate",
]
