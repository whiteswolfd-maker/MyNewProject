"""一个半径在一个优化步上的全部样本。

字段名是采样器与损失之间唯一的契约。采样点一律以**网络输入坐标**
（``s``、``xi``、``eta``）存放，物理半径只在与 EOS、几何源项和数据标签有关
的地方保留。

这样做有一个具体好处：链式法则所需的两个坐标恰好就是损失手里已有的两个
坐标，不需要在损失里再做一次从物理坐标到相似坐标的换算——那种二次换算是
漏乘或重复乘链式项最常见的来源。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..physics.similitude import (
    TIME_CONVENTION_PLANAR_STARTUP,
    SimilitudeScales,
)


@dataclass
class SampleBatch:
    """一个半径的全部采样点。张量形状一律 ``(N, 1)``。"""

    case_id: str
    charge_radius: float
    #: 卡片爆速 ``D``，m/s。方案 B 的时间原点 ``T_h = R_0/D + epsilon`` 与接触面压力
    #: 尺度都要用它；从算例带上来，**不给默认值**。
    #:
    #: 平面黎曼启动场的时间原点是它自己的 ``epsilon``，不需要 ``D``。那条路径必须
    #: 显式传 ``None``（"不适用"），不能用 1、6930 之类数值填满旧接口；方案 B 的
    #: batch 仍要求有限正爆速。两种约定由 ``scales.time_convention`` 区分。
    detonation_velocity: float | None
    scales: SimilitudeScales
    # 该半径的有效时间范围（``t_PINN``），s。
    pinn_time_min: float
    pinn_time_max: float

    # -- PDE 配置点（严格位于各自域内部） --------------------------------
    s_product: Any = None
    xi_product: Any = None
    r_product: Any = None
    s_air: Any = None
    eta_air: Any = None
    r_air: Any = None

    # -- 初始条件 ---------------------------------------------------------
    xi_ic_product: Any = None
    rho_ic_product: Any = None
    u_ic_product: Any = None
    e_ic_product: Any = None
    eta_ic_air: Any = None
    rho_ic_air: Any = None
    u_ic_air: Any = None
    e_ic_air: Any = None
    contact_ic: float = float("nan")
    shock_ic: float = float("nan")

    # -- 接触面与主激波时间点 ---------------------------------------------
    s_contact: Any = None
    s_shock: Any = None

    # -- 数据监督 ---------------------------------------------------------
    # 数据标签的物理位置是 ``(物理半径 r, 物理时间 t)``，材料归属由数据预处理
    # 给出。这里**只存 ``r`` 和 ``s``，不存 ``xi``/``eta``**：相似坐标要由
    # **模型当前预测的前沿**换算，因为标签是要拿去评价模型自己的场的。若模型
    # 的前沿与数据不一致，换算出的坐标会落在单位区间之外——这时损失照常评价并
    # 记录不匹配比例，而不是把坐标钳到边界伪装成一致（提示词 §11）。
    #: 观测**支持区**（§12）。``r_data_*`` 是支持区中心，``lo_``/``hi_`` 是它的
    #: 两端。标签 ``rho/u/e`` 是这一段上的**平均值**（数据侧是单元平均），所以模型
    #: 侧必须在**同一段**上做同样的几何加权平均——拿中心点值去比壳平均，比的是两个
    #: 不同的观测。
    s_data_product: Any = None
    r_data_product: Any = None
    lo_data_product: Any = None
    hi_data_product: Any = None
    coverage_data_product: Any = None
    rho_data_product: Any = None
    u_data_product: Any = None
    e_data_product: Any = None
    s_data_air: Any = None
    r_data_air: Any = None
    lo_data_air: Any = None
    hi_data_air: Any = None
    coverage_data_air: Any = None
    rho_data_air: Any = None
    u_data_air: Any = None
    e_data_air: Any = None
    # 前沿标签**独立**（§11.1）：contact 与 shock 各用自己的时间点与标签，一个
    # 有效不要求另一个也有效。以前是 ``s_data_front / contact_data / shock_data``
    # 三列平行、长度相同——那意味着"contact 有效但 shock 无效"的整帧都被丢掉。
    s_data_contact: Any = None
    contact_data: Any = None
    s_data_shock: Any = None
    shock_data: Any = None

    # -- L_geometry（§11.2）----------------------------------------------
    # 冻结的位置不确定性，m。由数据与探测器分辨率决定，**训练中不更新**：
    # delta 一旦随训练自动扩大，就会把错误预测一起包容进去，约束失去意义。
    geometry_delta_contact: float = float("nan")
    geometry_delta_shock: float = float("nan")

    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """爆速必须与时间约定一致：没有"恰好为零就当作不适用"的猜测。"""
        convention = getattr(self.scales, "time_convention", None)
        if self.detonation_velocity is None:
            if convention != TIME_CONVENTION_PLANAR_STARTUP:
                raise ValueError(
                    "a batch may omit the detonation velocity only under the "
                    f"planar-startup time convention ({TIME_CONVENTION_PLANAR_STARTUP!r}); "
                    f"this batch carries {convention!r}"
                )
            return
        velocity = float(self.detonation_velocity)
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError(
                "the detonation velocity must be finite and positive, got "
                f"{self.detonation_velocity!r}"
            )
        if convention == TIME_CONVENTION_PLANAR_STARTUP:
            raise ValueError(
                "a planar-startup batch must not carry a detonation velocity: its "
                "time origin is epsilon_field and has no R_0/D term"
            )

    # -- 工具 -------------------------------------------------------------
    def tensor_fields(self) -> list[str]:
        return [name for name, value in self.__dict__.items()
                if torch is not None and isinstance(value, torch.Tensor)]

    def to(self, device) -> "SampleBatch":
        for name in self.tensor_fields():
            setattr(self, name, getattr(self, name).to(device))
        return self

    def detach(self) -> "SampleBatch":
        for name in self.tensor_fields():
            setattr(self, name, getattr(self, name).detach())
        return self

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "charge_radius": self.charge_radius,
            "pinn_time_min": self.pinn_time_min,
            "pinn_time_max": self.pinn_time_max,
        }
        for name in self.tensor_fields():
            value = getattr(self, name)
            payload[f"n_{name}"] = int(value.reshape(-1).numel())
        payload.update(self.diagnostics)
        return payload


#: 每个张量字段的**时间坐标**字段名（§14.2 的窗口裁剪用）。
#:
#: 必须逐个声明，不能按"长度相同就一起裁"来猜：``s_product`` 与 ``s_air`` 的
#: 点数常常相同，猜错不会报错，只会把一个介质的 PDE 点裁到另一个介质的时间
#: 坐标上去。初始条件两组**不在**表里——它们在 ``t_PINN = 0`` 上评价，与时间
#: 窗口无关，任何阶段都必须全部保留，否则"初值预训练"会在扩窗时被自己削掉。
TIME_COORDINATES: dict[str, str] = {
    "s_product": "s_product",
    "xi_product": "s_product",
    "r_product": "s_product",
    "s_air": "s_air",
    "eta_air": "s_air",
    "r_air": "s_air",
    "s_contact": "s_contact",
    "s_shock": "s_shock",
    "s_data_product": "s_data_product",
    "r_data_product": "s_data_product",
    "lo_data_product": "s_data_product",
    "hi_data_product": "s_data_product",
    "coverage_data_product": "s_data_product",
    "rho_data_product": "s_data_product",
    "u_data_product": "s_data_product",
    "e_data_product": "s_data_product",
    "s_data_air": "s_data_air",
    "r_data_air": "s_data_air",
    "lo_data_air": "s_data_air",
    "hi_data_air": "s_data_air",
    "coverage_data_air": "s_data_air",
    "rho_data_air": "s_data_air",
    "u_data_air": "s_data_air",
    "e_data_air": "s_data_air",
    "s_data_contact": "s_data_contact",
    "contact_data": "s_data_contact",
    "s_data_shock": "s_data_shock",
    "shock_data": "s_data_shock",
}


def restrict_to_time_window(batch: "SampleBatch", s_max: float) -> "SampleBatch":
    """只保留 ``s <= s_max`` 的样本，返回**新**的 batch（原 batch 不动）。

    用途是阶段验收：检查集只建一次、全程复用（§13），但阶段判据必须在**本阶段
    的窗口**内评价。若拿完整时段的检查点去评价 25 % 窗口的阶段 1，模型在还没
    训到的那四分之三上当然不满足方程，判据永远不可能通过。

    裁剪只按**时间坐标**做，各字段用自己的坐标，见 :data:`TIME_COORDINATES`。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    limit = float(s_max)
    if not (limit > 0.0):
        raise ValueError(f"s_max must be positive, got {s_max!r}")

    out = SampleBatch(
        case_id=batch.case_id,
        charge_radius=batch.charge_radius,
        detonation_velocity=batch.detonation_velocity,
        scales=batch.scales,
        pinn_time_min=batch.pinn_time_min,
        # ``limit`` 是无量纲的 ``s`` 上界；批次的 ``pinn_time_max`` 是 ``tau``
        # 秒。两者不能直接比大小——换算只经由批次自己的尺度。
        pinn_time_max=min(
            float(batch.pinn_time_max),
            float(batch.scales.pinn_time_from_network(limit)),
        ),
        contact_ic=batch.contact_ic,
        shock_ic=batch.shock_ic,
        geometry_delta_contact=batch.geometry_delta_contact,
        geometry_delta_shock=batch.geometry_delta_shock,
        diagnostics=dict(batch.diagnostics),
    )

    masks: dict[str, Any] = {}

    def mask_for(coordinate: str):
        if coordinate not in masks:
            reference = getattr(batch, coordinate, None)
            if reference is None:
                raise ValueError(
                    f"batch {batch.case_id!r} declares {coordinate!r} as a time "
                    "coordinate but carries no such field"
                )
            masks[coordinate] = (reference.reshape(-1) <= limit)
        return masks[coordinate]

    for name, value in batch.__dict__.items():
        if name in ("case_id", "charge_radius", "detonation_velocity", "scales",
                    "pinn_time_min", "pinn_time_max", "contact_ic", "shock_ic",
                    "geometry_delta_contact", "geometry_delta_shock", "diagnostics"):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        coordinate = TIME_COORDINATES.get(name)
        if coordinate is None:
            # 与时间无关的字段（初始条件的两组）原样保留。
            setattr(out, name, value)
            continue
        mask = mask_for(coordinate)
        if mask.numel() != value.shape[0]:
            raise ValueError(
                f"field {name!r} has {value.shape[0]} rows but its time coordinate "
                f"{coordinate!r} has {mask.numel()}"
            )
        setattr(out, name, value[mask])

    out.diagnostics["time_window_s_max"] = limit
    return out


__all__ = ["SampleBatch", "TIME_COORDINATES", "restrict_to_time_window"]
