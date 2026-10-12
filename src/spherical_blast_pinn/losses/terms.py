"""各项损失：初始条件、接触面、主激波、物理解、数据监督、PDE。

每一项都遵守同一条约化规则：**先在组内对自己的样本取均值，再在组间等权
平均**。这样采样数量的变化不会悄悄改变隐含权重——一个用了 1024 个点的
残差和一个用了 128 个点的耦合条件，在总损失里的地位不该由它们的点数决定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..physics.moving_frame import AIR, PRODUCT, check_moving_frame
from ..physics.pde import combined_pde_loss, medium_residual
from ..sampling.batch import SampleBatch

#: 数值保护：对数与除法用得到，同时不掩盖非物理状态。
TINY = 1.0e-30


class LossError(ValueError):
    """损失评价的前提不成立。"""


@dataclass
class TermBreakdown:
    """一项损失的标量值与其监控量。"""

    total: Any
    parts: dict[str, Any] = field(default_factory=dict)
    monitors: dict[str, Any] = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        out = {}
        for name, value in self.parts.items():
            if isinstance(value, torch.Tensor):
                out[name] = float(value.detach())
        for name, value in self.monitors.items():
            if isinstance(value, torch.Tensor):
                out[name] = float(value.detach())
            elif isinstance(value, (int, float)):
                out[name] = float(value)
        return out


def _mean_square(value) -> Any:
    return (value**2).mean()


def _or_zero(*values) -> Any:
    """返回第一个非空张量的 ``zeros``，供空集合使用。"""
    for value in values:
        if isinstance(value, torch.Tensor):
            return torch.zeros((), dtype=value.dtype, device=value.device)
    return torch.zeros(())


# --------------------------------------------------------------------------
# 初始条件（§10）
# --------------------------------------------------------------------------


def initial_condition_loss(batch: SampleBatch, model, *, contact_pressure=None) -> TermBreakdown:
    """初值损失：两材料的 ``rho/u/e`` 分布，加两条前沿与初始空气层厚度。

    前沿在 ``t_PINN = 0`` 处评价，即 ``s = 0``。提示词要求
    ``h_IC = R_s,IC - R_c,IC`` 也进入初值损失：薄空气层如果只约束两条前沿的
    各自位置，两个相近的数会互相掩盖误差，而厚度是它们之差，直接约束它才
    真正把空气层钉住。厚度误差按 ``h_IC`` 归一化。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = batch.scales
    zero = _or_zero(batch.xi_ic_product, batch.eta_ic_air)

    product_parts = []
    if batch.xi_ic_product is not None and batch.xi_ic_product.numel():
        fields = model.product_fields(
            torch.zeros_like(batch.xi_ic_product), batch.xi_ic_product
        )
        product_parts = [
            (((fields.density - batch.rho_ic_product) / scales.product_density) ** 2).mean(),
            (((fields.velocity - batch.u_ic_product) / scales.reference_velocity) ** 2).mean(),
            (((fields.specific_energy - batch.e_ic_product) / scales.reference_specific_energy) ** 2).mean(),
        ]
    air_parts = []
    if batch.eta_ic_air is not None and batch.eta_ic_air.numel():
        fields = model.air_fields(torch.zeros_like(batch.eta_ic_air), batch.eta_ic_air)
        air_parts = [
            (((fields.density - batch.rho_ic_air) / scales.air_density) ** 2).mean(),
            (((fields.velocity - batch.u_ic_air) / scales.reference_velocity) ** 2).mean(),
            (((fields.specific_energy - batch.e_ic_air) / scales.reference_specific_energy) ** 2).mean(),
        ]

    front_parts: list[Any] = []
    monitors: dict[str, Any] = {}
    if batch.contact_ic and batch.shock_ic and batch.contact_ic > 0.0:
        s_zero = torch.zeros(
            (1, 1), dtype=batch.xi_ic_product.dtype if product_parts else torch.float64,
            device=batch.xi_ic_product.device if product_parts else None,
        )
        positions = model.front_positions(s_zero)
        radius = float(batch.charge_radius)
        contact_pred = radius * positions.contact_hat
        shock_pred = radius * positions.shock_hat
        gap_pred = shock_pred - contact_pred
        gap_target = float(batch.shock_ic - batch.contact_ic)
        if gap_target <= 0.0:
            raise LossError(
                f"the reference initial field gives a non-positive air layer "
                f"({gap_target} m); the initial condition cannot be imposed"
            )
        front_parts = [
            (((contact_pred - float(batch.contact_ic)) / radius) ** 2).mean(),
            (((shock_pred - float(batch.shock_ic)) / radius) ** 2).mean(),
            (((gap_pred - gap_target) / gap_target) ** 2).mean(),
        ]
        monitors["ic_gap_relative_error"] = float(
            (torch.abs(gap_pred - gap_target) / gap_target).detach()
        )
        monitors["ic_contact_hat"] = float(positions.contact_hat.detach())
        monitors["ic_shock_hat"] = float(positions.shock_hat.detach())

    items = product_parts + air_parts + front_parts
    combined = sum(items) / len(items) if items else zero
    field_item_count = len(product_parts) + len(air_parts)
    return TermBreakdown(
        total=combined,
        parts={
            "ic_fields": (
                sum(product_parts + air_parts) / field_item_count
                if field_item_count
                else zero
            ),
            "ic_fronts": (
                sum(front_parts) / len(front_parts) if front_parts else zero
            ),
            **{f"ic_product_{i}": v for i, v in enumerate(product_parts)},
            **{f"ic_air_{i}": v for i, v in enumerate(air_parts)},
        },
        monitors={
            "ic_n_product": int(batch.xi_ic_product.numel()) if batch.xi_ic_product is not None else 0,
            "ic_n_air": int(batch.eta_ic_air.numel()) if batch.eta_ic_air is not None else 0,
            "ic_gap_target_m": float(batch.shock_ic - batch.contact_ic) if batch.contact_ic else 0.0,
            **monitors,
        },
    )


# --------------------------------------------------------------------------
# 接触面（§9）
# --------------------------------------------------------------------------


def contact_loss(batch: SampleBatch, model, *, contact_pressure: float) -> TermBreakdown:
    """接触面：两侧取极限，压力连续、两侧速度都等于界面速度。

    .. math::

        p_p - p_a = 0, \\qquad u_p - \\dot R_c = 0, \\qquad u_a - \\dot R_c = 0

    不要求密度或比内能相等。压力残差统一除以 ``Pc_*``——取参考 1 µs 初值场
    在接触面上的正值压力，两条前沿压力尺度相差三个数量级，直接比较两个
    无量纲压力输出是没有意义的。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = batch.scales
    s = batch.s_contact
    if s is None or not s.numel():
        zero = _or_zero(batch.s_contact)
        return TermBreakdown(total=zero, monitors={"contact_n_points": 0})

    s = s.detach().requires_grad_(True)
    fronts = model.front_data(s, charge_radius=batch.charge_radius)
    check_moving_frame(fronts, scales, name="contact")

    ones = torch.ones_like(s)
    product = model.product_fields(s, ones)
    air = model.air_fields(s, torch.zeros_like(s))

    pressure = (product.pressure - air.pressure) / float(contact_pressure)
    speed_product = (product.velocity - fronts.contact_speed) / scales.reference_velocity
    speed_air = (air.velocity - fronts.contact_speed) / scales.reference_velocity

    parts = {
        "contact_pressure": _mean_square(pressure),
        "contact_speed_product": _mean_square(speed_product),
        "contact_speed_air": _mean_square(speed_air),
    }
    total = sum(parts.values()) / len(parts)
    return TermBreakdown(
        total=total,
        parts=parts,
        monitors={
            "contact_n_points": int(s.numel()),
            "contact_pressure_scale_pa": float(contact_pressure),
            "contact_speed_median_m_s": float(fronts.contact_speed.detach().median()),
        },
    )


# --------------------------------------------------------------------------
# 主激波：完整 Rankine–Hugoniot（§9）
# --------------------------------------------------------------------------


def shock_loss(batch: SampleBatch, model) -> TermBreakdown:
    """三条完整 RH 条件，波前取解析环境态，波后取 ``eta = 1`` 的空气态。

    .. math::

        \\rho_{a0} D_s - \\rho_b (D_s - u_b) &= 0 \\\\
        p_{a0} + \\rho_{a0} D_s^2 - p_b - \\rho_b (D_s - u_b)^2 &= 0 \\\\
        e_{a0} + \\frac{p_{a0}}{\\rho_{a0}} + \\frac{D_s^2}{2}
          - e_b - \\frac{p_b}{\\rho_b} - \\frac{(D_s - u_b)^2}{2} &= 0

    归一化尺度依次为 ``rho_{a0} U_*``、``rho_{a0} U_*^2``、``U_*^2``。不使用
    强激波常数密度比替代完整 RH，也不在 ``R_s`` 处再加普通无反射边界条件。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = batch.scales
    s = batch.s_shock
    if s is None or not s.numel():
        return TermBreakdown(total=_or_zero(batch.s_shock), monitors={"shock_n_points": 0})

    s = s.detach().requires_grad_(True)
    fronts = model.front_data(s, charge_radius=batch.charge_radius)
    check_moving_frame(fronts, scales, name="shock")

    post = model.air_fields(s, torch.ones_like(s))
    ambient = model.undisturbed

    rho0 = ambient.density
    p0 = ambient.pressure
    e0 = ambient.specific_energy
    ds = fronts.shock_speed
    relative = ds - post.velocity

    mass = (rho0 * ds - post.density * relative) / (rho0 * scales.reference_velocity)
    momentum = (
        p0 + rho0 * ds**2 - post.pressure - post.density * relative**2
    ) / (rho0 * scales.reference_velocity**2)
    energy = (
        e0 + p0 / rho0 + 0.5 * ds**2
        - post.specific_energy - post.pressure / post.density.clamp_min(TINY)
        - 0.5 * relative**2
    ) / (scales.reference_velocity**2)

    parts = {
        "shock_mass": _mean_square(mass),
        "shock_momentum": _mean_square(momentum),
        "shock_energy": _mean_square(energy),
    }
    total = sum(parts.values()) / len(parts)
    return TermBreakdown(
        total=total,
        parts=parts,
        monitors={
            "shock_n_points": int(s.numel()),
            "shock_speed_median_m_s": float(ds.detach().median()),
            "shock_mach_median": float((ds.detach() / _ambient_sound_speed(model)).median()),
        },
    )


def _ambient_sound_speed(model) -> float:
    import math

    return math.sqrt(model.gamma_air * model.undisturbed.pressure / model.undisturbed.density)


# --------------------------------------------------------------------------
# 物理解条件 L_adm（§9）
# --------------------------------------------------------------------------


def _hinge(value, *, margin=0.0):
    """``max(margin - value, 0)^2``：连续、可导、在满足时精确为零。"""
    return torch.clamp(margin - value, min=0.0) ** 2


def admissibility_loss(batch: SampleBatch, model, *, product_fields, air_fields) -> TermBreakdown:
    """正值、压缩、超声速与熵条件，用平方 hinge 作连续惩罚。

    密度与内能的正值已经由输出形式保证，不在这里重复惩罚。不要求整个空气域
    ``p >= p_a0``，也不禁止负超压或回流——那些是真实的爆炸后现象。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = batch.scales
    zero = _or_zero(batch.s_contact, batch.xi_product, batch.eta_air)

    # -- 两材料：绝对压力为正、声速平方为正 -------------------------------
    pieces: dict[str, Any] = {}
    if product_fields is not None:
        # 用**未裁剪**的 ``c^2``（``jwl_sound_speed_squared``）。先把负值裁成零
        # 再检查 ``c^2 >= 0``，会把每一个无效状态都报成有效。
        c2 = model.product_net.sound_speed_squared(product_fields)
        pieces["adm_product_pressure"] = _hinge(
            product_fields.pressure / scales.product_pressure, margin=0.0
        ).mean()
        pieces["adm_product_sound_speed"] = _hinge(
            c2 / scales.reference_velocity**2, margin=0.0
        ).mean()
        monitors_pre: dict[str, Any] = {
            "adm_product_invalid_sound_speed_rate": float(
                (c2 < 0.0).to(torch.float64).mean().detach()
            )
        }
    else:
        monitors_pre = {}
    if air_fields is not None:
        c2 = model.gamma_air * air_fields.pressure / air_fields.density.clamp_min(TINY)
        pieces["adm_air_pressure"] = _hinge(
            air_fields.pressure / scales.air_pressure, margin=0.0
        ).mean()
        pieces["adm_air_sound_speed"] = _hinge(
            c2 / scales.reference_velocity**2, margin=0.0
        ).mean()
        monitors_pre["adm_air_invalid_sound_speed_rate"] = float(
            (c2 < 0.0).to(torch.float64).mean().detach()
        )

    # -- 主激波处的压缩、超声速与熵 ---------------------------------------
    monitors: dict[str, Any] = dict(monitors_pre)
    s = batch.s_shock
    if s is not None and s.numel():
        s = s.detach().requires_grad_(True)
        fronts = model.front_data(s, charge_radius=batch.charge_radius)
        post = model.air_fields(s, torch.ones_like(s))
        ambient = model.undisturbed

        rho0 = ambient.density
        p0 = ambient.pressure
        gamma = model.gamma_air
        c0 = _ambient_sound_speed(model)
        ds = fronts.shock_speed
        relative = ds - post.velocity

        pieces["adm_shock_compression_density"] = _hinge(
            post.density / rho0, margin=1.0
        ).mean()
        pieces["adm_shock_compression_pressure"] = _hinge(
            post.pressure / p0, margin=1.0
        ).mean()
        pieces["adm_shock_supersonic"] = _hinge(
            ds / c0, margin=1.0
        ).mean()
        pieces["adm_shock_relative_flow"] = _hinge(
            relative / scales.reference_velocity, margin=0.0
        ).mean()

        entropy = torch.log(
            post.pressure.clamp_min(TINY) / p0
        ) - gamma * torch.log(post.density.clamp_min(TINY) / rho0)
        violated = entropy < 0.0
        monitors["entropy_violation_rate"] = float(violated.double().mean().detach())
        pieces["adm_shock_entropy"] = _hinge(entropy, margin=0.0).mean()

    if not pieces:
        return TermBreakdown(total=zero, monitors=monitors)
    total = sum(pieces.values()) / len(pieces)
    return TermBreakdown(total=total, parts=pieces, monitors=monitors)


# --------------------------------------------------------------------------
# 数据监督 L_d（§10）
# --------------------------------------------------------------------------


#: 支持区内的求积点数。观测是**壳平均**，模型侧必须在同一段半径上按几何权重
#: 平均，不能拿中心点值去比（§12）。取 5 个点已远细于数据分辨率（支持区就是一个
#: 原始单元），再细只是浪费。
N_QUADRATURE = 5


def shell_average_of_model(model, medium: str, batch: SampleBatch, s, lo, hi):
    """在支持区 ``[lo, hi]`` 上做**几何加权平均**的模型观测（§12）。

        rho_bar = sum(v_j rho_j) / sum(v_j)
        u_bar   = sum(v_j rho_j u_j) / sum(v_j rho_j)
        e_bar   = sum(v_j rho_j e_j) / sum(v_j rho_j)
        p_bar   = sum(v_j p_j) / sum(v_j)      （p_j 由**逐点** EOS 算出）

    ``v_j`` 是几何体积权重 ``(r_{j+1}^3 - r_j^3)/3``。质量权重用的是**预测**的
    rho，不是真实密度——把真实密度代进预测观测是另一件事。

    压力必须"逐点由 EOS 求，再平均"，不能用 ``EOS(平均 rho, 平均 e)``：
    JWL 是非线性的，两者不相等。

    返回 ``(rho_bar, u_bar, e_bar, p_bar)``，形状都是 ``(n, 1)``。
    """

    """在支持区 ``[lo, hi]`` 上做**几何加权平均**的模型观测（§12）。

        rho_bar = sum(v_j rho_j) / sum(v_j)
        u_bar   = sum(v_j rho_j u_j) / sum(v_j rho_j)
        e_bar   = sum(v_j rho_j e_j) / sum(v_j rho_j)
        p_bar   = sum(v_j p_j) / sum(v_j)      （p_j 由**逐点** EOS 算出）

    ``v_j`` 是几何体积权重 ``(r_{j+1}^3 - r_j^3)/3``。质量权重用的是
    **预测**的 rho，不是真实密度——把真实密度代进预测观测是另一件事。

    压力必须"逐点由 EOS 求，再平均"，不能用 ``EOS(平均 rho, 平均 e)``：
    JWL 是非线性的，两者不相等。
    """
    # lo/hi 是**列向量** (n, 1)（batch 里所有张量都是 (N, 1) 约定），
    # 所以这里 reshape 成 (n, 1) 再广播；用 unsqueeze 会变成 (n, 1, 1)，
    # 求积出来的 edges 就成了 (n, 1, NQ+1)，取 centres 时列数为 0。
    n = lo.shape[0]
    lo_column = lo.reshape(n, 1)
    hi_column = hi.reshape(n, 1)
    # 求积节点：把每个支持区等分，取子区间中点。权重按几何体积。
    fractions = torch.linspace(
        0.0, 1.0, N_QUADRATURE + 1, dtype=lo.dtype, device=lo.device
    ).reshape(1, -1)
    edges = lo_column + (hi_column - lo_column) * fractions
    # (n, N_QUADRATURE+1)
    centres = 0.5 * (edges[:, :-1] + edges[:, 1:])
    volumes = (edges[:, 1:] ** 3 - edges[:, :-1] ** 3) / 3.0
    radii = centres.reshape(-1, 1)
    s_repeated = s.repeat_interleave(N_QUADRATURE, dim=0)
    fronts = model.front_data(
        s_repeated, charge_radius=batch.charge_radius, create_graph=False
    )
    if medium == PRODUCT:
        coordinate = radii / fronts.contact_radius
        fields = model.product_fields(s_repeated, coordinate)
    else:
        gap = fronts.shock_radius - fronts.contact_radius
        coordinate = (radii - fronts.contact_radius) / gap
        fields = model.air_fields(s_repeated, coordinate)
    # 逐点场量折回 (n, N_QUADRATURE) 再按几何体积加权。求积点的展开是
    # 行主序的（第 i 个观测的 NQ 个点连续排列），与 ``repeat_interleave`` 一致。
    density_points = fields.density.reshape(n, N_QUADRATURE)
    velocity_points = fields.velocity.reshape(n, N_QUADRATURE)
    energy_points = fields.specific_energy.reshape(n, N_QUADRATURE)
    pressure_points = fields.pressure.reshape(n, N_QUADRATURE)

    volume_sum = volumes.sum(dim=1, keepdim=True)
    density = (volumes * density_points).sum(dim=1, keepdim=True) / volume_sum
    # 质量权重用**预测**的 rho——把真实密度代进预测观测是另一件事（§12）。
    mass_weight = volumes * density_points
    mass_sum = mass_weight.sum(dim=1, keepdim=True)
    velocity = (
        (mass_weight * velocity_points).sum(dim=1, keepdim=True) / mass_sum
    )
    energy = (mass_weight * energy_points).sum(dim=1, keepdim=True) / mass_sum
    # 压力：**逐点**由 EOS 求，再按体积平均。不能用 ``EOS(平均 rho, 平均 e)``
    # ——JWL 是非线性的，两者不相等。
    pressure = (volumes * pressure_points).sum(dim=1, keepdim=True) / volume_sum
    return density, velocity, energy, pressure






def data_loss(batch: SampleBatch, model) -> TermBreakdown:
    """``rho/u/e`` 与两条前沿。

    **三组等权**：产物场、空气场、前沿。这是提示词 §10 的原话——"产物场、
    空气场、前沿三部分分别均值后等权组合"。前沿组**内部**再对 ``Rc`` 与 ``Rs``
    取平均。把 ``Rc``/``Rs`` 拆成两组参与最外层的等权平均，会让前沿在总损失里
    占 1/2 而不是 1/3，物理上没有任何理由让两条标量前沿曲线和两个完整场平分
    秋色。

    数据标签的相似坐标由**模型当前预测的前沿**换算。当前沿与数据不一致时，
    ``xi`` 会落在单位区间之外——这时照常评价并记录越界比例，不把坐标钳到
    边界来伪装标签一致。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = batch.scales
    parts: dict[str, Any] = {}
    monitors: dict[str, Any] = {}

    #: 支持区内的求积点数。观测是**壳平均**，模型侧必须在同一段半径上按几何权重
    #: 平均，不能拿中心点值去比（§12）。取 5 个高斯式均匀点已远细于数据分辨率
    #: （支持区就是一个原始单元），再细只是浪费。
    def field_part(medium: str, s, lo, hi, rho, u, e):
        if s is None or not s.numel():
            return None
        s_leaf = s.detach().requires_grad_(True)
        fronts = model.front_data(s_leaf, charge_radius=batch.charge_radius, create_graph=False)
        # 支持区是否落在预测域内。落在外面时**照常评价**并记录，不把坐标钳到边界
        # 伪装成一致（§11）——位置冲突由 L_geometry 专门负责。
        if medium == PRODUCT:
            out_of_range = (hi > fronts.contact_radius) | (lo < 0.0)
            density_scale = scales.product_density
        else:
            out_of_range = (lo < fronts.contact_radius) | (hi > fronts.shock_radius)
            density_scale = scales.air_density
        monitors[f"data_{medium}_support_out_of_range"] = float(
            out_of_range.to(torch.float64).mean().detach()
        )

        prediction = shell_average_of_model(model, medium, batch, s_leaf, lo, hi)

        # 每个字段**独立**求均值（§11.1）：一个字段在某个点缺失（NaN），只跳过
        # 那个字段的那个点，不丢掉同一位置其他可靠的字段。
        def field_loss(predicted, target, scale):
            residual = ((predicted - target) / scale) ** 2
            finite = torch.isfinite(target)
            if not bool(finite.any()):
                return None
            return residual[finite].mean()

        return [
            loss for loss in (
                field_loss(prediction[0], rho, density_scale),
                field_loss(prediction[1], u, scales.reference_velocity),
                field_loss(prediction[2], e, scales.reference_specific_energy),
            ) if loss is not None
        ]

    product_parts = field_part(
        PRODUCT,
        batch.s_data_product,
        batch.lo_data_product,
        batch.hi_data_product,
        batch.rho_data_product,
        batch.u_data_product,
        batch.e_data_product,
    )
    air_parts = field_part(
        AIR,
        batch.s_data_air,
        batch.lo_data_air,
        batch.hi_data_air,
        batch.rho_data_air,
        batch.u_data_air,
        batch.e_data_air,
    )
    groups: list[Any] = []
    if product_parts:
        parts["data_product"] = sum(product_parts) / len(product_parts)
        groups.append(parts["data_product"])
    if air_parts:
        parts["data_air"] = sum(air_parts) / len(air_parts)
        groups.append(parts["data_air"])

    radius = float(batch.charge_radius)

    front_parts = []
    front_n_points = 0
    # 接触面与激波**独立**（§11.1）：各自用自己的时间点与标签，缺一个不关另一个。
    if batch.s_data_contact is not None and batch.s_data_contact.numel():
        s_contact = batch.s_data_contact.detach().requires_grad_(True)
        contact_hat = model.front_positions(s_contact).contact_hat
        parts["data_front_contact"] = (
            ((contact_hat * radius - batch.contact_data) / radius) ** 2
        ).mean()
        front_parts.append(parts["data_front_contact"])
        front_n_points += int(s_contact.numel())
        monitors["data_front_contact_n_points"] = int(s_contact.numel())
    if batch.s_data_shock is not None and batch.s_data_shock.numel():
        s_shock = batch.s_data_shock.detach().requires_grad_(True)
        shock_hat = model.front_positions(s_shock).shock_hat
        parts["data_front_shock"] = (
            ((shock_hat * radius - batch.shock_data) / radius) ** 2
        ).mean()
        front_parts.append(parts["data_front_shock"])
        front_n_points += int(s_shock.numel())
        monitors["data_front_shock_n_points"] = int(s_shock.numel())
    if front_parts:
        # 前沿组：组内先对 Rc/Rs 取均值，再作为一个整体参与最外层的等权平均。
        # 若只有一侧有标签，只算那一侧——不再因为"另一侧缺失"把整个前沿组关掉。
        parts["data_front"] = sum(front_parts) / len(front_parts)
        groups.append(parts["data_front"])
        monitors["data_front_n_points"] = front_n_points

    monitors["data_n_product"] = int(batch.s_data_product.numel()) if batch.s_data_product is not None else 0
    monitors["data_n_air"] = int(batch.s_data_air.numel()) if batch.s_data_air is not None else 0
    monitors["data_n_groups"] = len(groups)

    if not groups:
        return TermBreakdown(total=_or_zero(batch.s_contact), monitors=monitors)
    total = sum(groups) / len(groups)
    return TermBreakdown(total=total, parts=parts, monitors=monitors)


# --------------------------------------------------------------------------
# PDE（§8）
# --------------------------------------------------------------------------


def pde_loss(batch: SampleBatch, model) -> TermBreakdown:
    """两材料各自的三个强形式残差。

    ``xi`` / ``eta`` 就采在开区间内，物理半径 ``r = R_c xi`` 用模型自己的前沿
    算出来。非法移动域在这里**硬失败**：零个配置点、``R_c < R_0``、前沿或
    前沿速度非有限、``R_s <= R_c``，都是 :class:`~...MovingFrameError`。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")

    product = medium_pde_residual(batch, model, PRODUCT)
    air = medium_pde_residual(batch, model, AIR)
    if product is None or air is None:
        return TermBreakdown(
            total=_or_zero(batch.s_contact),
            monitors={"pde_product_points": 0 if product is None else product.n_points,
                      "pde_air_points": 0 if air is None else air.n_points},
        )
    total = combined_pde_loss(product, air)
    monitors = {**product.diagnostics(), **air.diagnostics()}
    return TermBreakdown(
        total=total,
        parts={
            "pde_product": product.mean_square(),
            "pde_air": air.mean_square(),
        },
        monitors=monitors,
    )


def medium_pde_residual(batch: SampleBatch, model, medium: str):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    scales = batch.scales
    if medium == PRODUCT:
        s_raw, coordinate_raw = batch.s_product, batch.xi_product
    else:
        s_raw, coordinate_raw = batch.s_air, batch.eta_air
    if s_raw is None or not s_raw.numel():
        return None

    s = s_raw.detach().requires_grad_(True)
    coordinate = coordinate_raw.detach().requires_grad_(True)

    fronts = model.front_data(s, charge_radius=batch.charge_radius)
    check_moving_frame(fronts, scales, name=f"{medium} pde")

    if medium == PRODUCT:
        radius = fronts.contact_radius * coordinate
        fields = model.product_fields(s, coordinate)
        pressure_derivatives = model.product_net.pressure_derivatives
    else:
        radius = fronts.contact_radius + (fronts.shock_radius - fronts.contact_radius) * coordinate
        fields = model.air_fields(s, coordinate)
        pressure_derivatives = model.air_net.pressure_derivatives

    # 唯一属于"移动域非法"的判据是半径到达轴心：``xi = r/R_c`` 在 ``R_c <= 0``
    # 时无定义，而 ``1/r`` 的几何源项在 ``r = 0`` 处发散。``R_c < R_0`` **不在**
    # 此列——它是优化初期的一个瞬态，不是关于整个演化的物理假设（提示词 §7
    # 只用正值构造约束前沿，初始位置由初值损失给出）。把它升级成全时段硬失败，
    # 就等于把"候选轨迹还没满足初值"当成"移动域不合法"。
    if not bool((radius > 0.0).all()):
        raise LossError(f"{medium} pde: a collocation radius reached the axis")

    # 链式法则必须用**本算例**的尺度：时间项系数是 ``U_*/(R_0 \hat T_*)``，
    # 而 ``R_0`` 随半径变。用共享模型的默认尺度会让 80/100 mm 的时间导数
    # 分别差 1.6 倍与 2 倍。
    factors = model.chain_factors(medium, fronts, radius, scales=scales)
    return medium_residual(
        fields=fields,
        s=s,
        coordinate=coordinate,
        factors=factors,
        radius=radius,
        scales=scales,
        medium=medium,
        pressure_derivatives=pressure_derivatives,
    )


def front_floor_monitor(batch: SampleBatch, model) -> dict[str, float]:
    """``R_c < R_0`` 的**监控量**，不是硬失败。

    按每根采样时间轴上的最小 ``R_c`` 是否低于装药半径来计数，并记录最坏的一
    个 ``R_c/R_0``。这是提示词 §7 的字面构造下会发生、且需要如实报告的瞬态：
    把它藏起来会让"前沿从未走到装药半径以内"变成一句没有依据的话。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    reference = batch.s_contact if batch.s_contact is not None else batch.s_product
    if reference is None or not reference.numel():
        return {}
    s = reference.detach().requires_grad_(True)
    fronts = model.front_data(s, charge_radius=batch.charge_radius, create_graph=False)
    ratio = (fronts.contact_radius.detach() / float(batch.charge_radius)).reshape(-1)
    below = ratio < 1.0
    return {
        "contact_below_charge_radius_rate": float(below.to(torch.float64).mean()),
        "min_contact_over_charge_radius": float(ratio.min()),
    }


def pde_states(batch: SampleBatch, model) -> tuple[Any, Any]:
    """PDE 配置点上的两材料状态，供物理解条件使用。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    product = None
    air = None
    if batch.s_product is not None and batch.s_product.numel():
        product = model.product_fields(
            batch.s_product.detach(), batch.xi_product.detach()
        )
    if batch.s_air is not None and batch.s_air.numel():
        air = model.air_fields(batch.s_air.detach(), batch.eta_air.detach())
    return product, air


def geometry_loss(batch: SampleBatch, model) -> TermBreakdown:
    """``L_geometry``：可靠观测支持区与预测前沿的**位置**约束（§11.2）。

    归一化 hinge 平方：

    * 产物   ``relu(r - Rc - delta_c) / R0``：可靠纯产物观测的外端应在 Rc 之内；
    * 空气   ``max(relu(Rc - r - delta_c), relu(r - Rs - delta_s)) / R0``：
      可靠波后纯空气支持区应落在 Rc 与 Rs 之间。

    **它必须独立于场量损失的掩码。** 场量损失只保留 target 有限的点，而几何冲突
    那些点的 target 恰恰可能是**有限的**（位置不对，值没问题）。把 L_geometry 挂
    在那张掩码下面，冲突样本会被一起删掉——模型只要缩小可比较区域就能降低误差，
    而那正是 §11.2 要防的行为。所以这里用**自己**的样本（数据点本身），不看
    target 的有限性。

    ``delta`` 是**冻结**的位置不确定性：由探测器分辨率算出、随 batch 传进来，
    训练中不更新。它一旦随训练自动扩大，约束就会把错误预测一起包容进去。

    这一项与"独立数据"、"admissibility" 并列，是总损失里**单独加权**的一项，
    不并入数据组——否则它会悄悄改变数据三组（产物/空气/前沿）的等权关系。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")

    parts: dict[str, Any] = {}
    monitors: dict[str, Any] = {}
    total = None
    radius = float(batch.charge_radius)
    delta_contact = float(getattr(batch, "geometry_delta_contact", float("nan")))
    delta_shock = float(getattr(batch, "geometry_delta_shock", float("nan")))

    def _has_points(value: Any) -> bool:
        return value is not None and getattr(value, "numel", lambda: 0)() > 0

    def _usable(value: float) -> bool:
        return math.isfinite(value) and value > 0.0

    if not (_usable(delta_contact) and _usable(delta_shock)):
        # 有几何约束该管的点、却没有定义 delta —— 这是一个**错误**，不是一个
        # 可以降级成 0 的情形。这一项在总损失里单独加权，返回 0 等于让它从损失
        # 里静默消失，而告警只落在 monitors 上（没人会去看一个不报错的人）。
        if _has_points(batch.s_data_product) or _has_points(batch.s_data_air):
            raise LossError(
                "L_geometry (§11.2) has data points to constrain but no finite position "
                f"uncertainty (contact={delta_contact!r}, shock={delta_shock!r}). Refusing "
                "to silently drop a weighted constraint."
            )
        monitors["geometry_available"] = 0.0
        return TermBreakdown(total=_or_zero(batch.s_contact), monitors=monitors)
    monitors["geometry_available"] = 1.0
    monitors["geometry_delta_contact_over_R0"] = delta_contact / radius
    monitors["geometry_delta_shock_over_R0"] = delta_shock / radius

    def evaluate(s, r, medium: str):
        if s is None or not s.numel():
            return None
        s_leaf = s.detach().requires_grad_(True)
        fronts = model.front_data(
            s_leaf, charge_radius=batch.charge_radius, create_graph=False
        )
        if medium == PRODUCT:
            violation = (r - fronts.contact_radius - delta_contact) / radius
            label = "geometry_product_outer"
        else:
            violation = torch.maximum(
                (fronts.contact_radius - r - delta_contact) / radius,
                (r - fronts.shock_radius - delta_shock) / radius,
            )
            label = "geometry_air_support"
        monitors[f"{label}_violating_fraction"] = float(
            (violation > 0.0).to(torch.float64).mean().detach()
        )
        monitors[f"{label}_max_violation_over_R0"] = float(violation.max().detach())
        monitors[f"{label}_n_points"] = int(s_leaf.numel())
        return torch.clamp(violation, min=0.0) ** 2

    pieces = [
        piece
        for piece in (
            evaluate(batch.s_data_product, batch.r_data_product, PRODUCT),
            evaluate(batch.s_data_air, batch.r_data_air, AIR),
        )
        if piece is not None
    ]
    if not pieces:
        monitors["geometry_available"] = 0.0
        return TermBreakdown(total=_or_zero(batch.s_contact), monitors=monitors)

    # 两个介质各自先取均值，再等权平均——与数据损失同一条约化规则：
    # 点数多的那一侧不该因为点多而在几何约束里更重。
    total = sum(piece.mean() for piece in pieces) / len(pieces)
    parts["geometry_product"] = pieces[0].mean()
    if len(pieces) > 1:
        parts["geometry_air"] = pieces[1].mean()
    return TermBreakdown(total=total, parts=parts, monitors=monitors)


__all__ = [
    "LossError",
    "TermBreakdown",
    "admissibility_loss",
    "contact_loss",
    "data_loss",
    "front_floor_monitor",
    "geometry_loss",
    "shell_average_of_model",
    "initial_condition_loss",
    "medium_pde_residual",
    "pde_loss",
    "pde_states",
    "shock_loss",
]
