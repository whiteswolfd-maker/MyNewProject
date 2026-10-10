"""尺度化：时间原点、统一参考量与相似坐标的归一化输入。

提示词 §5、§6 的全部公式在这里，且**只在这里**实现一次。任何模块若要
把物理时间换成网络输入，都必须调用本模块，不能自己再推一遍。

时间原点
--------

.. math::

    k &= R_0 / R_{\\rm ref} \\\\
    \\epsilon(R_0) &= k \\, \\epsilon_{\\rm ref} \\\\
    t_{\\rm PINN} &= t_{\\rm phys} - \\epsilon(R_0)

参考半径 ``R_ref = 0.05 m``、参考启动时间 ``epsilon_ref = 1e-6 s``，因此
50 / 80 / 100 mm 的启动时间分别是 1 / 1.6 / 2 微秒。

这不是从数据里测出来的燃烧时间。它只依赖半径，是规范固定下来的约定；
把数据里的 ``t_b`` 混进来会让不同半径的时间原点不再同比例缩放。

基础尺度
--------

.. math::

    e_* = E_0/\\rho_0, \\quad U_* = \\sqrt{e_*}, \\quad
    \\hat t = \\frac{U_* t_{\\rm PINN}}{R_0}, \\quad
    \\hat r = \\frac{r}{R_0}

``U_*`` 只是参考速度，既不是指定的产物速度也不是波速。

两个材料各自使用自己的密度尺度 ``rho_{*,p}=1630``、``rho_{*,a}=1.225``，
速度尺度同为 ``U_*``，比内能尺度同为 ``U_*^2``，压力尺度分别为
``P_{*,k} = rho_{*,k} U_*^2``。

时间输入
--------

.. math::  s = \\hat t / \\hat T_*

``hat T_*`` 由**训练半径**有效时间范围内最大的 ``hat t`` 确定，训练前冻结
并写入检查点。验证时不重新拟合，不把每个算例的最后一帧映射成 1。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from . import material as mat

#: 参考半径，m。
REFERENCE_RADIUS = 0.05

#: 参考启动时间，s。
REFERENCE_START_TIME = 1.0e-6


class SimilitudeError(ValueError):
    """尺度化输入不合法。"""


# --------------------------------------------------------------------------
# 时间原点
# --------------------------------------------------------------------------


def radius_ratio(charge_radius: float) -> float:
    """``k = R_0 / R_ref``。"""
    if not math.isfinite(charge_radius) or charge_radius <= 0.0:
        raise SimilitudeError(f"charge radius must be finite and positive, got {charge_radius}")
    return float(charge_radius) / REFERENCE_RADIUS


def start_delay(charge_radius: float, *, reference_start: float = REFERENCE_START_TIME) -> float:
    """``epsilon(R_0) = k epsilon_ref``，s：交接延迟。

    与半径同比例缩放，因此四个半径互为相似变换下的对应量。它是**选定的默认交接
    延迟**，不是从数据里测出的燃烧时间，需要数值验收。
    """
    return radius_ratio(charge_radius) * float(reference_start)


def detonation_transit_time(charge_radius: float, *, detonation_velocity: float) -> float:
    """``T_b = R_0 / D``，s：理论爆轰渡越时间。

    ``D`` 来自材料卡（§4），不给模块默认值。卡片上的 ``D`` 与 JWL+E0 的理想 CJ
    解相差 −0.68%、与卡片 PCJ 相差 −1.72%（实测），属于标称值与标定值的正常差异；
    **不能**再要求 JWL 同时穿过 ``(D, PCJ)`` 那个点，那才会超定。
    """
    velocity = float(detonation_velocity)
    if not math.isfinite(velocity) or velocity <= 0.0:
        raise SimilitudeError(
            f"the detonation velocity must be finite and positive, got {detonation_velocity!r}"
        )
    return float(charge_radius) / velocity


def handover_time(
    charge_radius: float,
    *,
    detonation_velocity: float,
    reference_start: float = REFERENCE_START_TIME,
) -> float:
    """``T_h = T_b + epsilon(R_0)``，s：PINN 的启动时刻（方案 B 的时间原点）。"""
    return detonation_transit_time(
        charge_radius, detonation_velocity=detonation_velocity
    ) + start_delay(charge_radius, reference_start=reference_start)


def physical_to_pinn_time(
    physical_time,
    charge_radius: float,
    *,
    detonation_velocity: float,
    reference_start: float = REFERENCE_START_TIME,
):
    """``tau = T - T_h``，s。**只转换一次。**"""
    return physical_time - handover_time(
        charge_radius,
        detonation_velocity=detonation_velocity,
        reference_start=reference_start,
    )


def pinn_to_physical_time(
    pinn_time,
    charge_radius: float,
    *,
    detonation_velocity: float,
    reference_start: float = REFERENCE_START_TIME,
):
    """``T = tau + T_h``，s。"""
    return pinn_time + handover_time(
        charge_radius,
        detonation_velocity=detonation_velocity,
        reference_start=reference_start,
    )


def surface_arrival_relative_time(pinn_time, charge_radius: float, *, detonation_velocity: float):
    """``t_surface = T - T_b = tau + epsilon``，s。

    需要"表面到达之后过了多久"时用它，**不要**再对 ``tau`` 平移一次。
    """
    return pinn_time + start_delay(charge_radius)


# --------------------------------------------------------------------------
# 尺度集合
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilitudeScales:
    """一个半径的完整尺度集合。全部为 SI。

    ``time_scale`` 即 ``hat T_*``，是整个运行共用的一个数，不是每个半径一个。
    它在训练前由 :func:`freeze_time_scale` 从**训练**半径冻结得到，之后不允许
    再变。
    """

    charge_radius: float
    reference_velocity: float
    reference_specific_energy: float
    product_density: float
    air_density: float
    time_scale: float
    #: 卡片上的爆速 ``D``，m/s。时间原点 ``T_h = R_0/D + epsilon`` 需要它。
    #:
    #: **不给默认值。** 省略它等于悄悄退回方案 A 的时间轴（原点只有 ``epsilon``），
    #: 而两条时间轴都只有秒的量纲，错了不会报错，只会把整条轨迹平移一个 ``T_b``。
    detonation_velocity: float = 0.0

    # -- 构造 -------------------------------------------------------------
    def __post_init__(self) -> None:
        for name in (
            "charge_radius",
            "reference_velocity",
            "reference_specific_energy",
            "product_density",
            "air_density",
            "time_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SimilitudeError(f"scale {name!r} must be finite and positive, got {value}")

    @classmethod
    def for_radius(
        cls,
        charge_radius: float,
        *,
        time_scale: float,
        detonation_velocity: float,
    ) -> "SimilitudeScales":
        """``detonation_velocity`` 必填：它来自材料卡，方案 B 的时间原点依赖它。"""
        return cls(
            charge_radius=float(charge_radius),
            reference_velocity=mat.REFERENCE_VELOCITY,
            reference_specific_energy=mat.PRODUCT_SPECIFIC_ENERGY,
            product_density=mat.PRODUCT_REFERENCE_DENSITY,
            air_density=mat.AIR_DENSITY,
            time_scale=float(time_scale),
            detonation_velocity=float(detonation_velocity),
        )

    # -- 压力尺度 ---------------------------------------------------------
    @property
    def product_pressure(self) -> float:
        """``P_{*,p} = rho_{*,p} U_*^2``，Pa。等于 E0（按参考体积）。"""
        return self.product_density * self.reference_velocity**2

    @property
    def air_pressure(self) -> float:
        """``P_{*,a} = rho_{*,a} U_*^2``，Pa。"""
        return self.air_density * self.reference_velocity**2

    def pressure_scale(self, medium: str) -> float:
        if medium == "product":
            return self.product_pressure
        if medium == "air":
            return self.air_pressure
        raise SimilitudeError(f"unknown medium {medium!r}")

    def density_scale(self, medium: str) -> float:
        if medium == "product":
            return self.product_density
        if medium == "air":
            return self.air_density
        raise SimilitudeError(f"unknown medium {medium!r}")

    # -- PDE 残差尺度（§8） ----------------------------------------------
    def mass_residual_scale(self, medium: str) -> float:
        """``S_rho,k = rho_{*,k} U_* / R_0``。"""
        return self.density_scale(medium) * self.reference_velocity / self.charge_radius

    def momentum_residual_scale(self, medium: str) -> float:
        """``S_m,k = rho_{*,k} U_*^2 / R_0``。"""
        return self.density_scale(medium) * self.reference_velocity**2 / self.charge_radius

    def energy_residual_scale(self, medium: str) -> float:
        """``S_E,k = rho_{*,k} U_*^3 / R_0``。"""
        return self.density_scale(medium) * self.reference_velocity**3 / self.charge_radius

    # -- 时间换算 ---------------------------------------------------------
    def start_delay(self) -> float:
        """``epsilon(R_0)``，s：交接延迟。"""
        return start_delay(self.charge_radius)

    def transit_time(self) -> float:
        """``T_b = R_0 / D``，s。"""
        return detonation_transit_time(
            self.charge_radius, detonation_velocity=self.detonation_velocity
        )

    def start_time(self) -> float:
        """``T_h = T_b + epsilon(R_0)``，s：PINN 的时间原点（方案 B）。"""
        return handover_time(
            self.charge_radius, detonation_velocity=self.detonation_velocity
        )

    def hat_time(self, pinn_time):
        """``hat t = U_* t_PINN / R_0``。"""
        return self.reference_velocity * pinn_time / self.charge_radius

    def pinn_time_from_hat(self, hat_time):
        """``t_PINN = hat t R_0 / U_*``。"""
        return hat_time * self.charge_radius / self.reference_velocity

    def network_time(self, pinn_time):
        """``s = hat t / hat T_*``，网络的时间输入。"""
        return self.hat_time(pinn_time) / self.time_scale

    def hat_time_from_network(self, s):
        """``hat t = s hat T_*``。"""
        return s * self.time_scale

    def pinn_time_from_network(self, s):
        """``t_PINN = s hat T_* R_0 / U_*``。"""
        return self.pinn_time_from_hat(self.hat_time_from_network(s))

    def physical_time_from_network(self, s):
        """``t_phys = t_PINN + epsilon(R_0)``。"""
        return self.pinn_time_from_network(s) + self.start_time()

    # -- 相似坐标 ---------------------------------------------------------
    def hat_radius(self, radius):
        """``hat r = r / R_0``。"""
        return radius / self.charge_radius

    def radius_from_hat(self, hat_radius):
        return hat_radius * self.charge_radius

    # -- 报告 -------------------------------------------------------------
    def as_dict(self) -> dict[str, float]:
        return {
            "charge_radius": self.charge_radius,
            "reference_velocity": self.reference_velocity,
            "reference_specific_energy": self.reference_specific_energy,
            "product_density": self.product_density,
            "air_density": self.air_density,
            "product_pressure": self.product_pressure,
            "air_pressure": self.air_pressure,
            "time_scale": self.time_scale,
            "start_time": self.start_time(),
            "radius_ratio": radius_ratio(self.charge_radius),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SimilitudeScales":
        return cls(
            charge_radius=float(payload["charge_radius"]),
            reference_velocity=float(payload["reference_velocity"]),
            reference_specific_energy=float(payload["reference_specific_energy"]),
            product_density=float(payload["product_density"]),
            air_density=float(payload["air_density"]),
            time_scale=float(payload["time_scale"]),
        )

    def with_time_scale(self, time_scale: float) -> "SimilitudeScales":
        return replace(self, time_scale=float(time_scale))


# --------------------------------------------------------------------------
# hat T_*：由训练半径冻结
# --------------------------------------------------------------------------


def freeze_time_scale(
    training_valid_pinn_times: Mapping[float, float] | Iterable[tuple[float, float]],
    *,
    charge_radius_of=None,
) -> tuple[float, dict[str, Any]]:
    """由**训练**半径的有效时间范围确定并冻结 ``hat T_*``。

    参数
    ----
    training_valid_pinn_times:
        ``{R_0: t_PINN_max}`` 或 ``[(R_0, t_PINN_max), ...]``。``t_PINN_max``
        是该半径最后一个有效仿真帧对应的 ``t_PINN``。

    返回 ``(hat_T_*, provenance)``。取的是各半径 ``hat t`` 的最大值，而不是
    某个平均或中位数：``s <= 1`` 必须覆盖每一个训练半径的整个有效区间，否则
    训练区间末端会被静默地推到分布之外。

    验证半径与盲测半径**不得**进入这个估计。传入它们会改变 ``hat T_*``，
    从而改变整个训练的时间输入分布。
    """
    items = (
        list(training_valid_pinn_times.items())
        if isinstance(training_valid_pinn_times, Mapping)
        else list(training_valid_pinn_times)
    )
    if not items:
        raise SimilitudeError("at least one training radius is required to freeze hat T_*")

    per_radius: dict[str, float] = {}
    best = 0.0
    best_radius = None
    for radius, pinn_time in items:
        radius = float(radius)
        pinn_time = float(pinn_time)
        if radius <= 0.0 or not math.isfinite(radius):
            raise SimilitudeError(f"invalid charge radius {radius!r}")
        if not math.isfinite(pinn_time) or pinn_time <= 0.0:
            raise SimilitudeError(
                f"charge radius {radius} has a non-positive valid t_PINN range ({pinn_time!r}); "
                "a radius whose data ends at or before its own start time cannot define the "
                "time scale"
            )
        hat = mat.REFERENCE_VELOCITY * pinn_time / radius
        per_radius[repr(radius)] = hat
        if hat > best:
            best, best_radius = hat, radius

    provenance = {
        "definition": "max over training radii of hat_t = U_* t_PINN / R_0 at the last valid frame",
        "hat_time_per_radius": per_radius,
        "governing_radius": best_radius,
        "time_scale": best,
        "training_radii_only": True,
    }
    return best, provenance


def scaled_time_coverage(
    valid_pinn_time_max: Mapping[Any, float],
    *,
    time_scale: float,
) -> tuple[float, dict[str, Any]]:
    """训练覆盖的**无量纲**时间上界 ``s_max``，以及它的来源。

    ``s = hat_t / hat T_*``，``hat_t = U_* t_PINN / R_0``。判断一个查询是否属于
    训练覆盖，必须在 ``s``（或 ``hat_t``）上做，而不是拿秒数比大小：同样的
    ``s`` 在大半径上对应的物理秒数更长——球对称相似解的时间尺度就是
    ``R_0/U_*``——所以拿 50 mm 的秒数上限去卡 100 mm 的查询，会把一个完全处在
    训练覆盖内的点标成外推，反过来也一样。

    ``s_max`` 是**一个与半径无关的数**：给定 ``s``，无论目标半径是多少，是否
    超出覆盖的判断都相同。
    """
    if not valid_pinn_time_max:
        raise SimilitudeError(
            "the scaled-time coverage needs at least one case's last valid t_PINN"
        )
    time_scale = float(time_scale)
    if not math.isfinite(time_scale) or time_scale <= 0.0:
        raise SimilitudeError(f"hat T_* must be finite and positive, got {time_scale}")

    per_case: dict[str, dict[str, float]] = {}
    best = 0.0
    governing = None
    for key, pinn_time in valid_pinn_time_max.items():
        radius = float(key)
        pinn_time = float(pinn_time)
        if radius <= 0.0 or not math.isfinite(radius):
            raise SimilitudeError(f"invalid charge radius {radius!r}")
        if not math.isfinite(pinn_time) or pinn_time < 0.0:
            raise SimilitudeError(
                f"charge radius {radius} has an invalid last valid t_PINN ({pinn_time!r})"
            )
        hat = mat.REFERENCE_VELOCITY * pinn_time / radius
        s_value = hat / time_scale
        per_case[repr(radius)] = {"pinn_time_max": pinn_time, "hat_t_max": hat, "s_max": s_value}
        if s_value > best:
            best, governing = s_value, radius

    provenance = {
        "definition": "max over training cases of s = hat_t / hat T_*, hat_t = U_* t_PINN / R_0",
        "time_scale": time_scale,
        "s_max": best,
        "governing_radius": governing,
        "per_case": per_case,
        "radius_independent": True,
    }
    return best, provenance


def physical_time_from_scaled(
    s_value: float, charge_radius: float, *, time_scale: float, detonation_velocity: float
) -> float:
    """``s -> T``，供"默认预测到训练覆盖的末端"这类用途。

    ``T = s hat T_* R_0 / U_* + T_h``，``T_h = R_0/D + epsilon``。方案 B 下这里加的
    是**交接时刻**而不是 ``epsilon``：加错一个会正好差一个 ``T_b``，而两者都只有秒
    的量纲，不会报错。目标半径只参与还原，不参与覆盖判断本身。
    """
    scales = SimilitudeScales.for_radius(
        charge_radius, time_scale=time_scale, detonation_velocity=detonation_velocity
    )
    return float(scales.physical_time_from_network(float(s_value)))


def scales_for_cases(
    charge_radii: Sequence[float], *, time_scale: float, detonation_velocity: float
) -> dict[float, SimilitudeScales]:
    """每个半径一套尺度，共用同一个 ``hat T_*`` 与同一个卡片爆速。"""
    return {
        float(radius): SimilitudeScales.for_radius(float(radius), time_scale=time_scale)
        for radius in charge_radii
    }


__all__ = [
    "REFERENCE_RADIUS",
    "REFERENCE_START_TIME",
    "SimilitudeError",
    "SimilitudeScales",
    "freeze_time_scale",
    "physical_time_from_scaled",
    "physical_to_pinn_time",
    "pinn_to_physical_time",
    "radius_ratio",
    "scaled_time_coverage",
    "scales_for_cases",
    "start_delay",
    "detonation_transit_time",
    "handover_time",
    "surface_arrival_relative_time",
]
