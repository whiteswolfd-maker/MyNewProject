"""``ProductNet``：接触面以内的爆轰产物状态。

提示词 §7 的输出构造：

.. math::

    (s, \\xi^2) &\\longrightarrow (z_\\rho, z_u, z_e) \\\\
    \\rho_p &= \\rho_{*,p} \\operatorname{softplus}(z_\\rho) \\\\
    u_p &= U_* \\, \\xi \\, z_u \\\\
    e_p &= e_* \\operatorname{softplus}(z_e)

三处硬结构，都不是惩罚项：

* ``softplus`` 使密度与比内能恒为正，不需要截断，也不需要正值惩罚去追一个
  结构上已经成立的性质；
* 显式的 ``xi`` 因子给出 ``u_p(0) = 0``，即球心速度条件；
* 输入用 ``\\xi^2`` 而不是 ``\\xi``，因为球对称光滑场关于 ``r=0`` 是偶函数，
  在 ``\\xi=0`` 处 ``\\partial_r \\rho = \\partial_r e = 0``。把 ``\\xi^2`` 喂进去,
  让"偶函数"成为网络可以表达的结构，而不是让它去学一个奇函数的展开。

压力**不是**网络输出：它由一般 JWL 状态方程 ``p_p = p_{JWL}(\\rho_p, e_p)``
得到。因此比内能始终是独立的自由度，永远不会被冻结到某条固定等熵线上。

速度允许为负，不做 ``tanh`` 限幅，也不人为封顶。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from ..physics import material as mat
from ..physics.eos import (
    JWLParameters,
    jwl_pressure,
    jwl_pressure_derivatives,
    jwl_sound_speed,
    jwl_sound_speed_squared,
)
from ..physics.similitude import SimilitudeScales
from .mlp import MLP, MLPConfig, ModelError, as_column


@dataclass
class ProductFields:
    """产物场的物理量，SI。"""

    density: Any
    velocity: Any
    specific_energy: Any
    pressure: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "rho_product": self.density,
            "u_product": self.velocity,
            "e_product": self.specific_energy,
            "p_product": self.pressure,
        }


class ProductNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """``f(s, xi^2) -> (rho, u, e)``，压力由 JWL 得到。"""

    def __init__(
        self,
        *,
        scales: SimilitudeScales,
        jwl: JWLParameters | None = None,
        width: int = 128,
        depth: int = 4,
        activation: str = "tanh",
        seed: int = 0,
        dtype=None,
    ) -> None:
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        super().__init__()
        self.jwl = jwl or mat.product_jwl()
        self.jwl.validate()
        self.scales = scales
        self.dtype_ = dtype or torch.float64
        self.width = int(width)
        self.depth = int(depth)
        self.activation = str(activation)
        self.seed = int(seed)

        if self.depth < 1:
            raise ModelError(f"ProductNet needs at least one hidden layer, got depth={self.depth}")

        self.mlp = MLP(
            MLPConfig(
                input_dim=2,
                output_dim=3,
                width=self.width,
                depth=self.depth,
                activation=self.activation,
                # 第一版不加入 Fourier 特征（§7.5）。
                fourier_features=0,
            ),
            seed=seed,
            dtype=self.dtype_,
        )

    # -- 尺度 -------------------------------------------------------------
    @property
    def density_scale(self) -> float:
        """``rho_{*,p}``。"""
        return self.scales.product_density

    @property
    def velocity_scale(self) -> float:
        """``U_*``。"""
        return self.scales.reference_velocity

    @property
    def energy_scale(self) -> float:
        """``e_* = E_0/rho_0``。"""
        return self.scales.reference_specific_energy

    # -- 前向 -------------------------------------------------------------
    def forward(self, s, xi, scales=None) -> ProductFields:
        """``s`` 与 ``xi`` 为形状 ``(N, 1)`` 的张量。

        ``scales`` 是**这条样本所属算例**的尺度。网络本身无量纲；输出换算用
        该算例的 ``rho_*, U_*, e_*``。同一组权重可以服务多个半径，但材料常数
        必须相同（由调用方在装配/加载时核对），本方法不偷偷混用。
        """
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        active = self.scales if scales is None else scales
        s = as_column(s, dtype=self.dtype_)
        xi = as_column(xi, dtype=self.dtype_)
        raw = self.mlp(torch.cat([s, xi * xi], dim=-1))

        density = float(active.product_density) * F.softplus(raw[:, 0:1])
        velocity = float(active.reference_velocity) * xi * raw[:, 1:2]
        energy = float(active.reference_specific_energy) * F.softplus(raw[:, 2:3])
        pressure = jwl_pressure(density, energy, self.jwl)
        return ProductFields(
            density=density,
            velocity=velocity,
            specific_energy=energy,
            pressure=pressure,
        )

    # -- 派生量 -----------------------------------------------------------
    def sound_speed(self, fields: ProductFields):
        return jwl_sound_speed(fields.density, fields.specific_energy, self.jwl)

    def sound_speed_squared(self, fields: ProductFields):
        """**未裁剪**的 ``c^2``。物理解条件必须用它判断有效性。

        ``sound_speed`` 会把负的 ``c^2`` 裁成 0，所以拿它去检查 ``c^2 >= 0``
        永远成立——那是一个把无效状态报告成有效状态的检查。
        """
        return jwl_sound_speed_squared(fields.density, fields.specific_energy, self.jwl)

    def pressure_derivatives(self, fields: ProductFields):
        """``(dp/drho|_e, dp/de|_rho)``，供链式法则把 ``p`` 的导数接起来。"""
        return jwl_pressure_derivatives(fields.density, fields.specific_energy, self.jwl)

    # -- 报告 -------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict[str, Any]:
        return {
            "network": "product",
            "inputs": ["s", "xi^2"],
            "outputs": ["rho", "u", "e"],
            "output_construction": {
                "rho": "rho_*,p * softplus(z_rho)",
                "u": "U_* * xi * z_u",
                "e": "e_* * softplus(z_e)",
            },
            "density_scale": self.density_scale,
            "velocity_scale": self.velocity_scale,
            "energy_scale": self.energy_scale,
            "width": self.width,
            "depth": self.depth,
            "activation": self.activation,
            "parameters": self.parameter_count(),
            "jwl": self.jwl.as_dict(),
        }


__all__ = ["ProductFields", "ProductNet"]
