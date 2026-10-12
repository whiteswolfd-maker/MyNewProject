"""``FrontNet``：由统一时间 ``s`` 直接预测两条前沿。

提示词 §7.3、§9：

* 2 个隐藏层，每层 64，``tanh``；
* **只输入统一时间 ``s``**，不输入半径——半径的作用已经由尺度化吸收，
  非相似起爆条件不在本版的泛化范围内；
* 输出两个量，用平滑正值变换构造 ``\\hat R_c > 0``、``\\hat h > 0``，
  再令 ``\\hat R_s = \\hat R_c + \\hat h``；
* **不强制** ``R_c`` 单调增加；
* 不额外输出一个不受约束的波速网络。

前沿速度由对 ``s`` 求导得到：

.. math::

    \\dot R_c = \\frac{U_*}{\\hat T_*} \\frac{d\\hat R_c}{ds}, \\qquad
    \\dot R_s = \\frac{U_*}{\\hat T_*} \\frac{d\\hat R_s}{ds}

这与提示词禁止的"用 RK 推进前沿"是两回事：这里没有任何时间积分，前沿就是
网络输出的代数函数。

初始位置不写死在这里，而是由初始条件损失约束。输出层偏置只用来让**未训练**
的网络从一个参考初值附近起步，从而避免第一帧就把某个域压成空集；它不限制
网络之后能学到什么。
"""
from __future__ import annotations

import math
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

from ..physics.similitude import SimilitudeScales
from .mlp import MLP, MLPConfig, ModelError, as_column

#: ``softplus(z) = 1`` 的 ``z``：``log(e - 1)``。把输出层偏置设成它，
#: 就使 ``scale * softplus(bias) == scale`` 精确成立。
SOFTPLUS_UNITY_ARGUMENT = 0.5413248546129181

#: ``hat h`` 初值的下界。``softplus`` 只趋近 0 而不到达 0，所以初值必须严格
#: 为正；这个下界是**数值**保护，不是物理假设。空气层在参考启动时刻的真实
#: 厚度由参考初值给出，通常远大于它。
MINIMUM_GAP_HAT = 1.0e-6

#: 输出层权重的初始缩放。**很小，但不是零。**
#:
#: 取零会让未训练的前沿精确等于参考初值，但它同时让 ``dL/dW_hidden`` 精确为零
#: ——梯度要穿过零输出矩阵，隐藏层在第一步上一点梯度都收不到。取 ``1e-3`` 时
#: 隐藏层从第一次调用起就与输出相连，而前沿仍然落在参考初值的千分之几以内。
FRONT_OUTPUT_WEIGHT_SCALE = 1.0e-3


@dataclass
class FrontPositions:
    """``hat R_c``、``hat h``、``hat R_s``，全部以 ``R_0`` 为单位。"""

    contact_hat: Any
    gap_hat: Any
    shock_hat: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "hat_Rc": self.contact_hat,
            "hat_h": self.gap_hat,
            "hat_Rs": self.shock_hat,
        }


@dataclass
class FrontState:
    """物理单位的前沿位置与速度。"""

    contact_radius: Any
    shock_radius: Any
    contact_speed: Any
    shock_speed: Any
    positions: FrontPositions

    @property
    def gap(self):
        return self.shock_radius - self.contact_radius

    def as_dict(self) -> dict[str, Any]:
        return {
            "R_c": self.contact_radius,
            "R_s": self.shock_radius,
            "dR_c_dt": self.contact_speed,
            "dR_s_dt": self.shock_speed,
        }


class FrontNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """``s -> (hat R_c, hat h)``。

    接触面的输出构造
    ----------------

    提示词 §7 的约定是**平滑正值变换**，即

    .. math::  \\hat R_c = a\\, \\mathrm{softplus}(z_c), \\qquad a = \\hat R_c^{\\rm IC} > 0

    初始位置由初值损失约束，而不是写进输出构造。这是**默认**，也是
    ``networks.front.contact_anchor`` 关闭时的行为。

    可选的早时段锚定
    ----------------

    另有一档可选的额外约束，**默认关闭**，通过配置开启：

    .. math::

        \\hat R_c = a\\, \\mathrm{softplus}(z_c) + w(s), \\qquad
        w(s) = \\tfrac12\\left(1 + \\cos\\pi\\,\\min(s/s_{\\rm anchor}, 1)\\right)

    * **适用时段**：只有 ``0 <= s <= s_anchor`` 上有 ``w > 0``；``s >= s_anchor``
      时 ``w`` 恒为 0，网络回到字面的正值构造。``w`` 在两端都有一阶零导数，
      所以开关本身不引入折点。``s_anchor = inf``（``until_s: null``）退化成
      全时段 ``+1``。
    * **依据**：接触面就是原装药表面所在的那个物质界面。参考前置求解给出的
      ``R_c(t)`` 在整个训练时间区间上严格大于 ``R_0``，且初值损失是 ``s = 0``
      上的**一个点**约束——它在初始化时的量级远小于同一时刻的接触面条件，
      于是 ``hat R_c(0)`` 会被其它项的梯度推着单调下滑。这一档是为了让早期
      几步的候选轨迹不越出那条已知的物理下界，**不是**一个关于整个演化的
      物理假设，也不强制单调：``hat R_c`` 仍可随 ``s`` 上下起伏。

    为什么要写成一个带时段的开关而不是默认行为：把"初值验收"或"优化初期"
    的问题直接升级成"全时域的物理假设"，是把自己实现的数值困难伪装成物理。
    默认档必须逐字对应提示词 §7；额外约束要有适用时段和依据，并且可以被关掉。
    """

    def __init__(
        self,
        *,
        scales: SimilitudeScales,
        width: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        initial_contact_hat: float = 1.0,
        initial_gap_hat: float = 0.1,
        anchor_contact_at_charge_surface: bool = False,
        anchor_until_s: float | None = None,
        seed: int = 0,
        dtype=None,
    ) -> None:
        """``initial_contact_hat`` / ``initial_gap_hat`` 来自参考初值。

        它们只设置输出层偏置，使未训练网络的初始前沿落在参考初值附近。参考
        初值由 :mod:`spherical_blast_pinn.physics.precursor` 在
        ``epsilon_ref`` 时刻给出，经 ``k`` 缩放到目标半径：

        ``Rc_IC(R_0) = k Rc_ref(epsilon_ref)``，``Rs_IC`` 同理。
        """
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        super().__init__()
        self.scales = scales
        self.dtype_ = dtype or torch.float64
        self.width = int(width)
        self.depth = int(depth)
        self.activation = str(activation)
        self.seed = int(seed)

        if self.depth < 1:
            raise ModelError(f"FrontNet needs at least one hidden layer, got depth={self.depth}")

        contact_hat = float(initial_contact_hat)
        gap_hat = max(float(initial_gap_hat), MINIMUM_GAP_HAT)
        if not math.isfinite(contact_hat) or contact_hat <= 0.0:
            raise ModelError(f"initial_contact_hat must be positive and finite, got {contact_hat}")

        self.anchor_contact_at_charge_surface = bool(anchor_contact_at_charge_surface)
        if self.anchor_contact_at_charge_surface:
            if contact_hat <= 1.0:
                raise ModelError(
                    "the early-time contact anchor offsets hat_Rc by +1, so it needs "
                    f"initial_contact_hat > 1; the reference initial field gave {contact_hat}"
                )
            if anchor_until_s is not None:
                until = float(anchor_until_s)
                if not math.isfinite(until) or until <= 0.0:
                    raise ModelError(
                        f"anchor_until_s must be finite and positive when given, got {anchor_until_s}"
                    )
                self.anchor_until_s: float | None = until
            else:
                self.anchor_until_s = None  # 全时段
        else:
            self.anchor_until_s = None

        # 尺度取初值本身，于是偏置恒为 `softplus^-1(1)`，初值可精确表示。
        # 锚定档下偏移量是 1，所以缩放取 `contact_hat - 1`；两档都满足
        # `hat_Rc(0) == contact_hat`。
        self.contact_scale = (
            contact_hat - 1.0 if self.anchor_contact_at_charge_surface else contact_hat
        )
        self.gap_scale = gap_hat
        self.initial_contact_hat = contact_hat
        self.initial_gap_hat = gap_hat

        self.mlp = MLP(
            MLPConfig(
                input_dim=1,
                output_dim=2,
                width=self.width,
                depth=self.depth,
                activation=self.activation,
                fourier_features=0,
                output_bias=SOFTPLUS_UNITY_ARGUMENT,
                output_weight_scale=FRONT_OUTPUT_WEIGHT_SCALE,
            ),
            seed=seed,
            dtype=self.dtype_,
        )

    # -- 前向 -------------------------------------------------------------
    def offset(self, s):
        """可选早时段锚定的平滑偏移 ``w(s)``，见类文档。

        关闭时恒为 0（返回 ``None`` 的便宜版本：调用方直接跳过）。开启且未给
        ``anchor_until_s`` 时恒为 1。
        """
        if not self.anchor_contact_at_charge_surface:
            return None
        if self.anchor_until_s is None:
            return 1.0

        def ramp(time):
            fraction = (time / self.anchor_until_s).clamp(0.0, 1.0)
            return 0.5 * (1.0 + torch.cos(math.pi * fraction))

        return ramp

    def positions(self, s) -> FrontPositions:
        """前沿位置，以 ``R_0`` 为单位。"""
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        s = as_column(s, dtype=self.dtype_)
        raw = self.mlp(s)
        contact_hat = self.contact_scale * F.softplus(raw[:, 0:1])
        ramp = self.offset(s)
        if ramp is not None:
            contact_hat = contact_hat + (ramp if isinstance(ramp, float) else ramp(s))
        gap_hat = self.gap_scale * F.softplus(raw[:, 1:2])
        return FrontPositions(
            contact_hat=contact_hat,
            gap_hat=gap_hat,
            shock_hat=contact_hat + gap_hat,
        )

    def state(
        self,
        s,
        *,
        charge_radius: float | None = None,
        create_graph: bool = True,
    ) -> FrontState:
        """物理单位的位置与速度；要求 ``s`` 可微。

        ``charge_radius`` 是**这个样本所属的**装药半径。三个训练半径共用同一套
        网络，所以半径不能烘进网络，也不能烘进模块的默认尺度——那会让 80 mm 的
        样本拿到用 50 mm 还原出来的物理半径，``R_c`` 于是看起来掉到 ``R_0`` 以下，
        一个本来健康的前沿会被判成非法移动域。不传时退回模块默认尺度里的半径，
        只应在"模型就是为这个半径建的"这种场合使用（例如 predict）。

        前沿速度与半径无关：

        .. math::

            \\dot R_c = \\frac{d}{dt}\\left(R_0 \\hat R_c(s)\\right)
            = R_0 \\frac{d\\hat R_c}{ds}\\frac{ds}{d\\hat t}\\frac{d\\hat t}{dt}
            = \\frac{U_*}{\\hat T_*}\\frac{d\\hat R_c}{ds}

        因为 ``ds/dhat t = 1/\\hat T_*`` 而 ``dhat t/dt = U_*/R_0``，两个 ``R_0``
        正好约掉。这不是巧合，而是"半径只进尺度化、不进网络"的直接后果。
        """
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        s = as_column(s, dtype=self.dtype_)
        if not s.requires_grad:
            raise ModelError(
                "FrontNet.state() needs s to require grad so the front speeds can be "
                "differentiated. Call s.requires_grad_(True) on the sampled time, or use "
                "positions() if only R_c and R_s are needed."
            )
        positions = self.positions(s)

        (d_contact,) = torch.autograd.grad(
            positions.contact_hat,
            s,
            grad_outputs=torch.ones_like(positions.contact_hat),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )
        (d_shock,) = torch.autograd.grad(
            positions.shock_hat,
            s,
            grad_outputs=torch.ones_like(positions.shock_hat),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )
        zero = torch.zeros_like(positions.contact_hat)
        d_contact = d_contact if d_contact is not None else zero
        d_shock = d_shock if d_shock is not None else zero

        factor = self.scales.reference_velocity / self.scales.time_scale
        radius = float(charge_radius if charge_radius is not None else self.scales.charge_radius)
        return FrontState(
            contact_radius=radius * positions.contact_hat,
            shock_radius=radius * positions.shock_hat,
            contact_speed=factor * d_contact,
            shock_speed=factor * d_shock,
            positions=positions,
        )

    # -- 报告 -------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict[str, Any]:
        return {
            "network": "front",
            "inputs": ["s"],
            "outputs": ["hat_Rc", "hat_h"],
            "shock": "hat_Rs = hat_Rc + hat_h",
            "width": self.width,
            "depth": self.depth,
            "activation": self.activation,
            "initial_contact_hat": self.initial_contact_hat,
            "initial_gap_hat": self.initial_gap_hat,
            "anchored_contact": self.anchor_contact_at_charge_surface,
            "anchor_until_s": self.anchor_until_s,
            "contact_construction": (
                "hat_Rc = a softplus(z_c) + w(s)"
                if self.anchor_contact_at_charge_surface
                else "hat_Rc = a softplus(z_c)"
            ),
            "contact_anchor_window": (
                None
                if not self.anchor_contact_at_charge_surface
                else ("all s" if self.anchor_until_s is None else f"0 <= s <= {self.anchor_until_s}")
            ),
            "monotonicity_imposed": False,
            "contact_floor_imposed": False,
            "parameters": self.parameter_count(),
        }


__all__ = [
    "MINIMUM_GAP_HAT",
    "SOFTPLUS_UNITY_ARGUMENT",
    "FrontNet",
    "FrontPositions",
    "FrontState",
]
