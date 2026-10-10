"""双场耦合模型：产物网络 + 空气网络 + 前沿网络。

三者**全程**联合优化，没有阶段切换，没有产物先跑、空气后跑的分离，也没有
任何时刻关闭产物网络。

.. math::

    0 \\le r < R_c &: \\texttt{ProductNet},\\; p = p_{JWL}(\\rho, e) \\\\
    R_c < r < R_s &: \\texttt{AirNet},\\; p = (\\gamma_a - 1)\\rho e \\\\
    r \\ge R_s &: \\text{解析环境态，无网络、无标签}

两条前沿都由**同一个** ``FrontNet`` 给出。它只吃统一时间 ``s``，两个场网络
只吃各自的相似坐标。半径不进入任何网络：材料相同、环境相同、初始化相同的
尺度相似模型里，``R_0`` 的作用已经被尺度化吸收。

前沿在计算 PDE 与耦合损失时**不得 detach**。``FrontState`` 的每一个速度都是
``autograd`` 对 ``s`` 求导得到的，计算图一直连到 ``FrontNet`` 的参数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:  # pragma: no cover
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from ..physics import material as mat
from ..physics.moving_frame import (
    AIR,
    PRODUCT,
    FrontData,
    air_chain_factors,
    product_chain_factors,
)
from ..physics.similitude import SimilitudeScales
from .air_net import AirFields, AirNet
from .front_net import FrontNet, FrontState
from .mlp import ModelError, resolve_device, resolve_dtype
from .product_net import ProductFields, ProductNet

#: 三个可训练模块的名字，供优化器分组与冻结使用。
GROUP_PRODUCT = "product"
GROUP_AIR = "air"
GROUP_FRONT = "front"
PARAMETER_GROUPS = (GROUP_PRODUCT, GROUP_AIR, GROUP_FRONT)


@dataclass
class UndisturbedState:
    """``r >= R_s`` 的解析环境态，空间与时间上都是常数。"""

    density: float
    velocity: float
    specific_energy: float
    pressure: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rho_ambient": self.density,
            "u_ambient": self.velocity,
            "e_ambient": self.specific_energy,
            "p_ambient": self.pressure,
        }


class CoupledModel(nn.Module if nn is not None else object):  # type: ignore[misc]
    """产物网络 + 空气网络 + 前沿网络。"""

    def __init__(
        self,
        *,
        product: ProductNet,
        air: AirNet,
        front: FrontNet,
        scales: SimilitudeScales,
        gamma_air: float = mat.AIR_GAMMA,
        ambient_density: float = mat.AIR_DENSITY,
        ambient_pressure: float = mat.AIR_PRESSURE,
    ) -> None:
        if nn is None:  # pragma: no cover
            raise ModelError("PyTorch is required")
        super().__init__()
        self.product_net = product
        self.air_net = air
        self.front_net = front
        self.scales = scales
        self.gamma_air = float(gamma_air)
        self.undisturbed = UndisturbedState(
            density=float(ambient_density),
            velocity=0.0,
            specific_energy=float(ambient_pressure)
            / ((float(gamma_air) - 1.0) * float(ambient_density)),
            pressure=float(ambient_pressure),
        )

    # -- 场 ---------------------------------------------------------------
    def product_fields(self, s, xi) -> ProductFields:
        return self.product_net(s, xi)

    def air_fields(self, s, eta) -> AirFields:
        return self.air_net(s, eta)

    # -- 前沿 -------------------------------------------------------------
    def front_state(
        self, s, *, charge_radius: float | None = None, create_graph: bool = True
    ) -> FrontState:
        """前沿位置与速度。``s`` 必须可微。

        ``charge_radius`` 由**该样本所属的算例**给出：三个训练半径共用这一套
        网络，物理半径只在还原时才乘上去。不传则退回模块默认尺度里的半径。
        """
        return self.front_net.state(
            s, charge_radius=charge_radius, create_graph=create_graph
        )

    def front_positions(self, s):
        return self.front_net.positions(s)

    def front_data(
        self, s, *, charge_radius: float | None = None, create_graph: bool = True
    ) -> FrontData:
        """前沿的**物理**位置与速度，供链式法则使用。"""
        state = self.front_state(
            s, charge_radius=charge_radius, create_graph=create_graph
        )
        return FrontData(
            contact_radius=state.contact_radius,
            shock_radius=state.shock_radius,
            contact_speed=state.contact_speed,
            shock_speed=state.shock_speed,
        )

    # -- 移动坐标 ---------------------------------------------------------
    def chain_factors(self, medium: str, fronts: FrontData, radius, *, scales):
        """链式系数。``scales`` 是**必填**的，且必须是本样本所属算例的尺度。

        时间项系数是 ``U_*/(R_0 \\hat T_*)``，其中 ``R_0`` 是**这个算例**的装药
        半径。用一个共享模型自带的、来自第一个训练半径的默认尺度，会让 80/100 mm
        的时间导数分别差 1.6 倍与 2 倍——而空间项与移动坐标项并不会替它抵消，
        那只是时间导数一项错了，残差其余部分照旧。

        所以这里没有默认值：调用方必须显式给出 ``scales``，最自然的来源就是
        ``SampleBatch.scales``（见 :class:`~...sampling.batch.SampleBatch`）。
        一个可以省略的参数，迟早会被省略。
        """
        if medium == PRODUCT:
            return product_chain_factors(fronts, scales, radius)
        if medium == AIR:
            return air_chain_factors(fronts, scales, radius)
        raise ModelError(f"unknown medium {medium!r}")

    # -- 分区 -------------------------------------------------------------
    def label_regions(self, radius, contact_radius, shock_radius):
        """按当前预测前沿分区，返回 ``(product, air, undisturbed)`` 三个掩码。"""
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        product = radius < contact_radius
        air = (~product) & (radius < shock_radius)
        return product, air, ~(product | air)

    def piecewise_profile(self, radius, s, *, charge_radius: float | None = None):
        """整条径向剖面；每个场量只在自己的域上评价，域外为 ``nan``。

        ``charge_radius`` 与 :meth:`front_state` 同理：三个训练半径共用一套网络，
        物理半径只能在还原时乘上去。不传时退回模块默认尺度里的半径，只适用于
        "模型就是为这个半径建的"这种场合。
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        positions = self.front_positions(s)
        radius = torch.as_tensor(radius, dtype=self.product_net.dtype_)
        if radius.dim() == 1:
            # 一维输入按"一个时刻上的一整条剖面"处理。前沿是 ``(n_t, 1)``，半径
            # 若是 ``(n_r,)`` 就会与它广播成 ``(n_r, n_r)``——一个不报错、只是把
            # 每个查询点都变成对所有半径取值的失误。
            radius = radius.reshape(1, -1)
        r0 = float(charge_radius if charge_radius is not None else self.scales.charge_radius)
        contact_column = radius.new_tensor(r0) * positions.contact_hat  # (n_t, 1)
        shock_column = radius.new_tensor(r0) * positions.shock_hat
        if contact_column.shape[0] == 1:
            contact_column = contact_column.expand(radius.shape[0], 1)
            shock_column = shock_column.expand(radius.shape[0], 1)
        # 掩码与索引都要二维前沿：``contact[product_mask]`` 要求两者形状一致，
        # ``(n_t, 1)`` 的前沿配 ``(n_t, n_r)`` 的掩码会在索引处直接报错。
        contact = contact_column.expand_as(radius)
        shock = shock_column.expand_as(radius)
        product_mask, air_mask, undisturbed_mask = self.label_regions(radius, contact, shock)

        shape = radius.shape
        fields = {
            "density": torch.full(shape, float("nan"), dtype=self.product_net.dtype_, device=radius.device),
            "velocity": torch.full(shape, float("nan"), dtype=self.product_net.dtype_, device=radius.device),
            "specific_energy": torch.full(shape, float("nan"), dtype=self.product_net.dtype_, device=radius.device),
            "pressure": torch.full(shape, float("nan"), dtype=self.product_net.dtype_, device=radius.device),
        }
        if bool(product_mask.any()):
            sub_r = radius[product_mask].reshape(-1, 1)
            xi = sub_r / contact[product_mask].reshape(-1, 1)
            out = self.product_fields(_broadcast_s(s, sub_r), xi)
            for name, value in (
                ("density", out.density),
                ("velocity", out.velocity),
                ("specific_energy", out.specific_energy),
                ("pressure", out.pressure),
            ):
                fields[name][product_mask] = value.reshape(-1)
        if bool(air_mask.any()):
            sub_r = radius[air_mask].reshape(-1, 1)
            c = contact[air_mask].reshape(-1, 1)
            h = shock[air_mask].reshape(-1, 1) - c
            eta = (sub_r - c) / h
            out = self.air_fields(_broadcast_s(s, sub_r), eta)
            for name, value in (
                ("density", out.density),
                ("velocity", out.velocity),
                ("specific_energy", out.specific_energy),
                ("pressure", out.pressure),
            ):
                fields[name][air_mask] = value.reshape(-1)
        for name, value in (
            ("density", self.undisturbed.density),
            ("velocity", self.undisturbed.velocity),
            ("specific_energy", self.undisturbed.specific_energy),
            ("pressure", self.undisturbed.pressure),
        ):
            fields[name][undisturbed_mask] = value
        # 前沿按列返回：它对每个时刻只有一个值，铺成 ``(n_t, n_r)`` 只是索引的需要。
        fields["contact_radius"] = contact_column
        fields["shock_radius"] = shock_column
        return fields

    # -- 训练辅助 ---------------------------------------------------------
    def parameter_groups(self) -> dict[str, list[Any]]:
        return {
            GROUP_PRODUCT: list(self.product_net.parameters()),
            GROUP_AIR: list(self.air_net.parameters()),
            GROUP_FRONT: list(self.front_net.parameters()),
        }

    def set_requires_grad(self, groups: Mapping[str, bool]) -> None:
        for name, parameters in self.parameter_groups().items():
            flag = bool(groups.get(name, True))
            for parameter in parameters:
                parameter.requires_grad_(flag)

    def describe(self) -> dict[str, Any]:
        return {
            "variant": "dual_network",
            "product": self.product_net.describe(),
            "air": self.air_net.describe(),
            "front": self.front_net.describe(),
            "undisturbed": self.undisturbed.as_dict(),
            "scales": self.scales.as_dict(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


def _broadcast_s(s, like):
    """把标量或 ``(1,1)`` 的时间张量广播到与 ``like`` 相同的行数。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    s = torch.as_tensor(s, dtype=like.dtype, device=like.device)
    if s.numel() == 1:
        return s.reshape(1, 1).expand(like.shape[0], 1)
    return s.reshape(-1, 1)


def resolve_front_anchor(config, *, charge_radius: float | None = None):
    """读取 ``networks.front.contact_anchor``，返回 ``(enabled, until_s)``。

    默认 ``enabled = False``：接触面用提示词 §7 的字面正值构造
    ``hat_Rc = a softplus(z_c)``，初始位置由初值损失约束。早时段锚定是一档
    **可选**的额外约束，必须显式打开，并且可以带一个以 ``s`` 计的适用时段
    （``until_s``）；``until_s`` 省略表示全时段。细节与依据见
    :class:`~...models.front_net.FrontNet` 的类文档。
    """
    section = config.section("networks.front")
    raw = section.get("contact_anchor", None)
    if raw is None:
        return False, None
    if isinstance(raw, bool):
        return bool(raw), None
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", False))
        until = raw.get("until_s", None)
        return enabled, (None if until is None else float(until))
    raise ModelError(f"networks.front.contact_anchor must be a bool or a mapping, got {raw!r}")


def build_model(
    config,
    *,
    scales: SimilitudeScales,
    initial_contact_hat: float = 1.0,
    initial_gap_hat: float = 0.1,
    seed: int | None = None,
    device=None,
) -> CoupledModel:
    """由配置构造整套模型。"""
    chosen_seed = int(seed if seed is not None else _first_seed(config))
    dtype = resolve_dtype(str(config.get("networks.dtype", "float64")))
    product_section = config.section("networks.product")
    air_section = config.section("networks.air")
    front_section = config.section("networks.front")

    product = ProductNet(
        scales=scales,
        width=int(product_section.get("width", 128)),
        depth=int(product_section.get("depth", 4)),
        activation=str(product_section.get("activation", "tanh")),
        seed=chosen_seed,
        dtype=dtype,
    )
    air = AirNet(
        scales=scales,
        gamma_air=float(config.get("air.gamma", mat.AIR_GAMMA)),
        width=int(air_section.get("width", 128)),
        depth=int(air_section.get("depth", 4)),
        activation=str(air_section.get("activation", "tanh")),
        seed=chosen_seed + 7,
        dtype=dtype,
    )
    anchor_enabled, anchor_until_s = resolve_front_anchor(config)
    front = FrontNet(
        scales=scales,
        width=int(front_section.get("width", 64)),
        depth=int(front_section.get("depth", 2)),
        activation=str(front_section.get("activation", "tanh")),
        initial_contact_hat=initial_contact_hat,
        initial_gap_hat=initial_gap_hat,
        anchor_contact_at_charge_surface=anchor_enabled,
        anchor_until_s=anchor_until_s,
        seed=chosen_seed + 13,
        dtype=dtype,
    )
    model = CoupledModel(
        product=product,
        air=air,
        front=front,
        scales=scales,
        gamma_air=float(config.get("air.gamma", mat.AIR_GAMMA)),
        ambient_density=float(config.get("air.rho", mat.AIR_DENSITY)),
        ambient_pressure=float(config.get("air.pressure", mat.AIR_PRESSURE)),
    )
    target = device if device is not None else resolve_device(str(config.get("training.device", "auto")))
    return model.to(target)


def _first_seed(config) -> int:
    seeds: Sequence[Any] = config.list_of("training.seeds", [0])
    return int(seeds[0]) if seeds else 0


__all__ = [
    "GROUP_AIR",
    "GROUP_FRONT",
    "GROUP_PRODUCT",
    "PARAMETER_GROUPS",
    "CoupledModel",
    "UndisturbedState",
    "build_model",
    "resolve_front_anchor",
]
