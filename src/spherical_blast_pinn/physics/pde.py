"""移动坐标下的球对称 Euler 强形式残差。

提示词 §8：

.. math::

    E &= e + u^2/2 \\\\
    R_\\rho &= \\partial_t \\rho + \\frac{1}{r^2}\\partial_r(r^2 \\rho u) \\\\
    R_m &= \\partial_t(\\rho u) + \\frac{1}{r^2}\\partial_r(r^2 \\rho u^2) + \\partial_r p \\\\
    R_E &= \\partial_t(\\rho E) + \\frac{1}{r^2}\\partial_r\\!\\left[r^2 u(\\rho E + p)\\right]

动量通量写成 ``r^2(rho u^2 + p)`` 时必须配套右端的几何源项，**不能漏掉也不能
重复添加**。本模块按上式的展开形式实现：

.. math::

    \\frac{1}{r^2}\\partial_r(r^2 \\rho u^2) = \\partial_r(\\rho u^2) + \\frac{2\\rho u^2}{r}

所以几何源项恰好出现一次。``tests/test_pde_residual.py`` 用均匀静止场和
解析光滑场把这一点钉死。

残差按材料各自的尺度归一化：

.. math::

    S_{\\rho,k} = \\frac{\\rho_{*,k} U_*}{R_0}, \\quad
    S_{m,k} = \\frac{\\rho_{*,k} U_*^2}{R_0}, \\quad
    S_{E,k} = \\frac{\\rho_{*,k} U_*^3}{R_0}

三个残差分别均方后等权平均，再对两材料等权平均。这样空气残差不会因为空气
密度小三数量级而自动失去权重。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .moving_frame import ChainFactors, differentiate_scalar
from .similitude import SimilitudeScales

PRODUCT = "product"
AIR = "air"


class PDEError(ValueError):
    """残差评价的前提不成立。"""


@dataclass
class ScaledResidual:
    """一个材料的三个残差，以及它们各自的归一化尺度。"""

    mass: Any
    momentum: Any
    energy: Any
    mass_scale: float
    momentum_scale: float
    energy_scale: float
    n_points: int
    medium: str

    def scaled(self):
        return (
            self.mass / self.mass_scale,
            self.momentum / self.momentum_scale,
            self.energy / self.energy_scale,
        )

    def mean_square(self):
        """三个残差分别均方后等权平均。"""
        mass, momentum, energy = self.scaled()
        return ((mass**2).mean() + (momentum**2).mean() + (energy**2).mean()) / 3.0

    def diagnostics(self) -> dict[str, Any]:
        mass, momentum, energy = self.scaled()
        return {
            f"pde_{self.medium}_points": int(self.n_points),
            f"pde_{self.medium}_mass_rms": float(torch.sqrt((mass**2).mean()).detach()),
            f"pde_{self.medium}_momentum_rms": float(torch.sqrt((momentum**2).mean()).detach()),
            f"pde_{self.medium}_energy_rms": float(torch.sqrt((energy**2).mean()).detach()),
        }


def residuals_from_physical_derivatives(
    *,
    rho,
    u,
    e,
    p,
    drho_dr,
    du_dr,
    de_dr,
    dp_dr,
    drho_dt,
    du_dt,
    de_dt,
    dp_dt,
    r,
):
    """由原始场量及其**物理**一阶导数装配三个残差。纯函数，便于解析测试。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    inv_r = 1.0 / r

    # -- 连续方程 ---------------------------------------------------------
    mass = drho_dt + (drho_dr * u + rho * du_dr) + 2.0 * rho * u * inv_r

    # -- 径向动量 ---------------------------------------------------------
    momentum = (
        (drho_dt * u + rho * du_dt)
        + (drho_dr * u * u + 2.0 * rho * u * du_dr + dp_dr)
        + 2.0 * rho * u * u * inv_r
    )

    # -- 总能量 -----------------------------------------------------------
    total_energy = e + 0.5 * u * u
    rho_E = rho * total_energy
    d_rhoE_dt = drho_dt * total_energy + rho * (de_dt + u * du_dt)
    d_rhoE_dr = drho_dr * total_energy + rho * (de_dr + u * du_dr)
    energy_flux = u * (rho_E + p)
    energy = (
        d_rhoE_dt
        + (du_dr * (rho_E + p) + u * (d_rhoE_dr + dp_dr))
        + 2.0 * energy_flux * inv_r
    )

    return mass, momentum, energy


def medium_residual(
    *,
    fields,
    s,
    coordinate,
    factors: ChainFactors,
    radius,
    scales: SimilitudeScales,
    medium: str,
    pressure_derivatives: Callable[[Any], tuple[Any, Any]],
) -> ScaledResidual:
    """一个材料在移动坐标下的三个残差。

    ``fields`` 需要按名字提供 ``density`` / ``velocity`` / ``specific_energy`` /
    ``pressure``。``pressure_derivatives`` 给出 ``(dp/drho|_e, dp/de|_rho)``，
    于是压力的物理导数由链式法则从 ``rho``、``e`` 的导数得到——**压力不是
    独立的网络输出**，它的导数也不该被独立地求出来。

    ``s`` 与 ``coordinate`` 必须是带计算图的叶子。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")

    rho = fields.density
    u = fields.velocity
    e = fields.specific_energy
    p = fields.pressure

    n_points = int(rho.numel())
    if n_points == 0:
        raise PDEError("medium_residual requires at least one collocation point")

    # 网络输出对两个**输入**的偏导。
    d_rho_s = differentiate_scalar(rho, s)
    d_rho_c = differentiate_scalar(rho, coordinate)
    d_u_s = differentiate_scalar(u, s)
    d_u_c = differentiate_scalar(u, coordinate)
    d_e_s = differentiate_scalar(e, s)
    d_e_c = differentiate_scalar(e, coordinate)

    # 压力的偏导沿 EOS 接起来。
    dp_drho, dp_de = pressure_derivatives(fields)
    d_p_s = dp_drho * d_rho_s + dp_de * d_e_s
    d_p_c = dp_drho * d_rho_c + dp_de * d_e_c

    # 显式链式法则：物理坐标下的导数。
    d_dr, d_dt = factors.d_coord_dr, factors.d_coord_dt
    time_s = factors.d_time_ds

    def physical(d_s, d_c):
        return d_c * d_dr, time_s * d_s + d_c * d_dt

    drho_dr, drho_dt = physical(d_rho_s, d_rho_c)
    du_dr, du_dt = physical(d_u_s, d_u_c)
    de_dr, de_dt = physical(d_e_s, d_e_c)
    dp_dr, dp_dt = physical(d_p_s, d_p_c)

    mass, momentum, energy = residuals_from_physical_derivatives(
        rho=rho,
        u=u,
        e=e,
        p=p,
        drho_dr=drho_dr,
        du_dr=du_dr,
        de_dr=de_dr,
        dp_dr=dp_dr,
        drho_dt=drho_dt,
        du_dt=du_dt,
        de_dt=de_dt,
        dp_dt=dp_dt,
        r=radius,
    )

    return ScaledResidual(
        mass=mass,
        momentum=momentum,
        energy=energy,
        mass_scale=scales.mass_residual_scale(medium),
        momentum_scale=scales.momentum_residual_scale(medium),
        energy_scale=scales.energy_residual_scale(medium),
        n_points=n_points,
        medium=medium,
    )


def combined_pde_loss(product_residual: ScaledResidual, air_residual: ScaledResidual):
    """两材料的 PDE 损失：各自三残差等权平均，再对两材料等权平均。"""
    return 0.5 * (product_residual.mean_square() + air_residual.mean_square())


__all__ = [
    "AIR",
    "PRODUCT",
    "PDEError",
    "ScaledResidual",
    "combined_pde_loss",
    "medium_residual",
    "residuals_from_physical_derivatives",
]
