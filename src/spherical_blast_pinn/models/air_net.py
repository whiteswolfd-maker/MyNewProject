"""``AirNet``：接触面与主激波之间的受扰动空气。

提示词 §7 的输出构造：

.. math::

    (s, \\eta) &\\longrightarrow (z_\\rho, z_u, z_e) \\\\
    \\rho_a &= \\rho_{*,a} \\operatorname{softplus}(z_\\rho) \\\\
    u_a &= U_* z_u \\\\
    e_a &= e_* \\operatorname{softplus}(z_e)

与 ``ProductNet`` 的两处结构差别都是有物理原因的：

* 空气没有球心硬约束，所以速度输出不乘 ``\\xi``；空气域也不含 ``r=0``。
* 输入是 ``\\eta`` 本身而不是 ``\\eta^2``。产物域的 ``\\xi=0`` 是球心，是
  一个对称点；空气域的 ``\\eta=0`` 是接触面，是一个间断，没有理由要求场
  在那里是偶函数。

压力由理想气体状态方程 ``p_a = (\\gamma_a - 1) \\rho_a e_a`` 得到，同样不是
网络输出。

两个场网络**不共享权重**。它们通过接触面条件与主激波条件耦合，不是通过
共享参数耦合。
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
from ..physics.eos import ideal_gas_pressure, ideal_gas_sound_speed
from ..physics.similitude import SimilitudeScales
from .mlp import MLP, MLPConfig, ModelError, as_column


@dataclass
class AirFields:
    """受扰动空气的物理量，SI。"""

    density: Any
    velocity: Any
    specific_energy: Any
    pressure: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "rho_air": self.density,
            "u_air": self.velocity,
            "e_air": self.specific_energy,
            "p_air": self.pressure,
        }


class AirNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """``f(s, eta) -> (rho, u, e)``，压力由理想气体 EOS 得到。"""

    def __init__(
        self,
        *,
        scales: SimilitudeScales,
        gamma_air: float = mat.AIR_GAMMA,
        width: int = 128,
        depth: int = 4,
        activation: str = "tanh",
        seed: int = 0,
        dtype=None,
    ) -> None:
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        super().__init__()
        if gamma_air <= 1.0:
            raise ModelError(f"gamma_air must exceed 1, got {gamma_air}")
        self.gamma_air = float(gamma_air)
        self.scales = scales
        self.dtype_ = dtype or torch.float64
        self.width = int(width)
        self.depth = int(depth)
        self.activation = str(activation)
        self.seed = int(seed)

        if self.depth < 1:
            raise ModelError(f"AirNet needs at least one hidden layer, got depth={self.depth}")

        self.mlp = MLP(
            MLPConfig(
                input_dim=2,
                output_dim=3,
                width=self.width,
                depth=self.depth,
                activation=self.activation,
                fourier_features=0,
            ),
            seed=seed,
            dtype=self.dtype_,
        )

    # -- 尺度 -------------------------------------------------------------
    @property
    def density_scale(self) -> float:
        """``rho_{*,a}``。"""
        return self.scales.air_density

    @property
    def velocity_scale(self) -> float:
        """``U_*``，与产物共用同一个速度尺度。"""
        return self.scales.reference_velocity

    @property
    def energy_scale(self) -> float:
        """``e_*``，与产物共用同一个比内能尺度。"""
        return self.scales.reference_specific_energy

    # -- 前向 -------------------------------------------------------------
    def forward(self, s, eta) -> AirFields:
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        s = as_column(s, dtype=self.dtype_)
        eta = as_column(eta, dtype=self.dtype_)
        raw = self.mlp(torch.cat([s, eta], dim=-1))

        density = self.density_scale * F.softplus(raw[:, 0:1])
        velocity = self.velocity_scale * raw[:, 1:2]
        energy = self.energy_scale * F.softplus(raw[:, 2:3])
        pressure = ideal_gas_pressure(density, energy, self.gamma_air)
        return AirFields(
            density=density,
            velocity=velocity,
            specific_energy=energy,
            pressure=pressure,
        )

    # -- 派生量 -----------------------------------------------------------
    def sound_speed(self, fields: AirFields):
        return ideal_gas_sound_speed(fields.density, fields.specific_energy, self.gamma_air)

    def pressure_derivatives(self, fields: AirFields):
        """``(dp/drho|_e, dp/de|_rho) = ((gamma-1) e, (gamma-1) rho)``。"""
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        gamma_minus_one = self.gamma_air - 1.0
        return gamma_minus_one * fields.specific_energy, gamma_minus_one * fields.density

    # -- 报告 -------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict[str, Any]:
        return {
            "network": "air",
            "inputs": ["s", "eta"],
            "outputs": ["rho", "u", "e"],
            "output_construction": {
                "rho": "rho_*,a * softplus(z_rho)",
                "u": "U_* * z_u",
                "e": "e_* * softplus(z_e)",
            },
            "density_scale": self.density_scale,
            "velocity_scale": self.velocity_scale,
            "energy_scale": self.energy_scale,
            "gamma_air": self.gamma_air,
            "width": self.width,
            "depth": self.depth,
            "activation": self.activation,
            "parameters": self.parameter_count(),
        }


__all__ = ["AirFields", "AirNet"]
