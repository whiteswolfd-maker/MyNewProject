"""均匀高压产物球与静止空气的分段常数初值（``t_phys = 0``）。

这个模块只构造初值，不做时间推进，不调用旧的爆轰求解器，也不写参考场缓存。

物理模型（2026-10-21 用户确认）
--------------------------------

在物理时间 ``t_phys = 0``：

* ``0 <= r < R_0``：爆轰产物。常数密度 ``rho_p0``、零速度、常数比内能
  ``e_p0``；压力由一般 JWL EOS 从 ``(rho_p0, e_p0)`` 计算，**不是**独立输入。
* ``r > R_0``：静止空气。常数密度 ``rho_air``、零速度；比内能由空气 EOS 从
  给定环境压力反解，``e_air = p_amb / ((gamma - 1) rho_air)``。
* ``r = R_0`` 是一个**单元界面**：左侧单元属于产物，右侧单元属于空气。
  不做左右状态平均，不加过渡层，不强制两侧压力相等。

这不是平面 Sod 问题：体积权重是球对称的 ``(r_{i+1}^3 - r_i^3)/3``，不是一维
管道的单位面积体积。也不是 Taylor 初场，不模拟炸药内部的爆轰波传播。

``E0`` 的口径
-------------

``material.PRODUCT_VOLUME_ENERGY`` 是 ``*EOS_JWL`` 的 ``E0``，按**参考体积**
定义（J/m^3）；对应的比内能是 ``e_p0 = E0 / rho_ref``（J/kg）。本模块用
``JWLParameters.initial_specific_energy`` 做这次换算，不把历史报告里的压力、
内能或 CJ 数值抄进来。若调用方直接给出 ``product_specific_energy``，就按
J/kg 使用，**不再**除密度；压力仍然由 EOS 计算。

这里的 ``rho_ref`` 是**实际使用的 EOS 参数集**的 ``rho0``，不是全局材料常量，
也与显式的 ``product_density`` 无关：初始密度和 EOS 参考密度是两个独立的量。
未显式给出 ``product_density`` 时，初始密度默认取同一个 ``rho_ref``。

模型标识
--------

``TWO_MEDIUM_MODEL_ID`` 与旧的方案 B 参考场构造字符串
``"scheme_b_products_only_detonation"`` 不同。这个初值场的 metadata 标记
``reference_field = False``、``acceptance_status = "NOT APPLICABLE ..."``，
不能被旧参考场验收闸门当成已验收场。

与旧方案 B 的关系
-----------------

仓库的 ``AGENTS.md`` 仍把"B 初场由 ``physics/detonation.py`` 构造"列为规范。
2026-10-21 的决策记录 ``DECISION_2026-10-21_UNIFORM_PRODUCT_SPHERE.md`` 明确
改掉了**新初值模块**的这一要求；旧求解器与旧参考场代码保留，但本模块不导入、
不调用它们。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import material as mat
from .eos import (
    EOSError,
    JWLParameters,
    ideal_gas_pressure_numpy,
    ideal_gas_specific_energy_numpy,
    ideal_gas_sound_speed_numpy,
    jwl_pressure_numpy,
    jwl_sound_speed_numpy,
)

#: 新初值模型的唯一标识。不能与旧方案 B 的 ``REFERENCE_CONSTRUCTION`` 相同。
TWO_MEDIUM_MODEL_ID = "uniform_product_sphere_air_v1"
TWO_MEDIUM_CONSTRUCTION = TWO_MEDIUM_MODEL_ID

PRODUCT = 0
AIR = 1
MATERIAL_NAMES = {PRODUCT: "product", AIR: "air"}

#: 几何/质量一致性的相对容差。工厂用同一组节点半径计算 edges、centres、
#: volume 与 mass，合法输入的差值为 0；1e-12 约为双精度 ulp 的 4000 倍，
#: 远低于任何真正的几何错误（例如体积错一倍 = 100%）。
_GEOMETRY_REL_TOL = 1.0e-12

#: 位置比较（外边界、产物/空气分界面）的容差。半径是 O(0.05..2) m，
#: 双精度在该尺度的 ulp 约 1e-17..4e-16；1e-12 相对 + 1e-15 绝对
#: 仍远小于单元宽度（约 1e-4 m），不会把合法的舍入差当成几何错误。
_POSITION_REL_TOL = 1.0e-12
_POSITION_ABS_TOL = 1.0e-15

#: EOS 一致性的相对容差。工厂用**同一组** ``eos.py`` 函数、同一组 float64
#: 操作数计算所存值，合法输入的差值恒为 0；1e-12 只用于吸收不同平台的
#: ``exp``/``sqrt`` 末位差异，仍远小于任何真正的 EOS 错误（压力乘 2 是 50%
#: 相对偏差，声速乘 2 是 50%）。压力与声速在进入比较前已确认有限且为正，
#: 因此量级不会接近 0，纯相对容差就足够，不需要再加绝对项。
_EOS_REL_TOL = 1.0e-12


class TwoMediumInitialError(ValueError):
    """初值参数或构造结果不合法。"""


def _finite_positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TwoMediumInitialError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise TwoMediumInitialError(f"{name} must be finite and positive, got {value!r}")
    return number


def _positive_int(name: str, value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TwoMediumInitialError(f"{name} must be an integer, got {value!r}") from exc
    if number != value or number < 1:
        raise TwoMediumInitialError(f"{name} must be an integer >= 1, got {value!r}")
    return number


def _require_eos_field(
    side: str,
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    eos_label: str,
) -> None:
    """要求所存物性与该侧 EOS 重新计算的结果一致。

    **先**检查 EOS 结果有限且为正：``nan``/``inf`` 会让任何 ``deviation > tol``
    比较静默为 False，负的声速平方也被 ``sqrt`` 夹成 0，二者都必须在此之前
    明确失败。比较用相对量级 ``max(|actual|, |expected|)``，两边都是同一个
    正量，不需要额外的绝对容差。
    """
    if not np.all(np.isfinite(expected)):
        raise TwoMediumInitialError(
            f"{side} {name} recomputed from the {eos_label} EOS is not finite; "
            f"fields: {name} ({side} cells), density, specific_energy"
        )
    if np.any(expected <= 0.0):
        raise TwoMediumInitialError(
            f"{side} {name} recomputed from the {eos_label} EOS is not positive; "
            f"fields: {name} ({side} cells), density, specific_energy"
        )
    if not np.all(np.isfinite(actual)):
        raise TwoMediumInitialError(
            f"{side} {name} must be finite; fields: {name} ({side} cells)"
        )
    scale = np.maximum(np.abs(actual), np.abs(expected))
    deviation = np.abs(actual - expected) / np.maximum(scale, 1.0e-300)
    worst = float(np.max(deviation))
    if worst > _EOS_REL_TOL:
        index = int(np.argmax(deviation))
        raise TwoMediumInitialError(
            f"{side} {name} does not match the {eos_label} EOS evaluated at "
            f"(density, specific_energy): max relative deviation {worst:.3e} > "
            f"{_EOS_REL_TOL:.1e} at cell {index} "
            f"(stored {float(actual[index])!r}, EOS {float(expected[index])!r}); "
            f"fields: {name} ({side} cells), density, specific_energy"
        )


@dataclass(frozen=True)
class TwoMediumInitialField:
    """``t_phys = 0`` 的球对称两介质分段常数初值。

    所有数组都是**单元量**（``cell_edges`` 与 ``node_velocity`` 除外）。密度、
    比内能、压力、声速、速度在每个单元内是常数；材料身份是单元属性，界面是
    单元边界，所以不会出现"一个单元里一半产物一半空气"的混合状态。

    质量与能量均按**单位立体角**给出：``mass_i = rho_i * (r_{i+1}^3-r_i^3)/3``。

    ``cell_centres`` 采用与 ``LagrangianMesh.centres`` 相同的半体积半径
    ``(0.5*(r_i^3 + r_{i+1}^3))^(1/3)``；它必须位于所属单元内部。

    对象自带构造时使用的 ``product_eos``（JWL 参数集）与 ``air_gamma``。
    ``pressure`` 与 ``sound_speed`` 都会被重新计算并逐单元核对：产物用一般
    JWL，空气用该 gamma 的理想气体。构造完成后改写这两列不会静默通过。
    """

    time: float
    charge_radius: float
    outer_radius: float
    cell_edges: np.ndarray
    cell_centres: np.ndarray
    cell_volume: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    specific_energy: np.ndarray
    pressure: np.ndarray
    sound_speed: np.ndarray
    material: np.ndarray
    mass: np.ndarray
    node_velocity: np.ndarray
    #: 构造本场时**实际使用**的 JWL 参数集与空气 gamma。校验必须拿它们
    #: 重新计算，而不是拿全局材料默认值，否则自定义材料会被误判。
    product_eos: JWLParameters
    air_gamma: float
    product_state: dict[str, float] = field(default_factory=dict)
    air_state: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- 结构校验 ---------------------------------------------------------
    def __post_init__(self) -> None:
        for name in (
            "cell_edges",
            "cell_centres",
            "cell_volume",
            "density",
            "velocity",
            "specific_energy",
            "pressure",
            "sound_speed",
            "mass",
            "node_velocity",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        object.__setattr__(self, "material", np.asarray(self.material, dtype=np.int64))

        # EOS 参数必须随对象保存：校验时用它们重算，不能用全局材料默认值。
        if not isinstance(self.product_eos, JWLParameters):
            raise TwoMediumInitialError(
                "product_eos must be a JWLParameters instance holding the parameters "
                f"actually used to build this field, got {type(self.product_eos).__name__}"
            )
        try:
            self.product_eos.validate()
        except EOSError as exc:
            raise TwoMediumInitialError(f"product_eos is invalid: {exc}") from exc
        try:
            gamma_air = float(self.air_gamma)
        except (TypeError, ValueError) as exc:
            raise TwoMediumInitialError(
                f"air_gamma must be a real number, got {self.air_gamma!r}"
            ) from exc
        if not math.isfinite(gamma_air) or gamma_air <= 1.0:
            raise TwoMediumInitialError(
                f"air_gamma must be finite and > 1, got {self.air_gamma!r}"
            )
        object.__setattr__(self, "air_gamma", gamma_air)

        n = int(self.material.size)
        if self.material.ndim != 1:
            raise TwoMediumInitialError(
                f"material must be one-dimensional, got shape {self.material.shape}"
            )
        if self.cell_edges.ndim != 1 or self.cell_edges.size != n + 1:
            raise TwoMediumInitialError(
                f"cell_edges must be one-dimensional with {n + 1} entries, "
                f"got shape {self.cell_edges.shape}"
            )
        if self.node_velocity.ndim != 1 or self.node_velocity.size != n + 1:
            raise TwoMediumInitialError(
                f"node_velocity must be one-dimensional with {n + 1} entries, "
                f"got shape {self.node_velocity.shape}"
            )
        for name in (
            "cell_centres",
            "cell_volume",
            "density",
            "velocity",
            "specific_energy",
            "pressure",
            "sound_speed",
            "mass",
        ):
            array = getattr(self, name)
            if array.ndim != 1 or array.shape != (n,):
                raise TwoMediumInitialError(
                    f"{name} must be one-dimensional with shape {(n,)}, got {array.shape}"
                )

        # -- 时间与标量半径 ------------------------------------------------
        if not math.isfinite(self.time):
            raise TwoMediumInitialError(f"time must be finite, got {self.time!r}")
        # 本模块只表示物理时间零；严格拒绝非零时间，不把它修正为 0。
        if self.time != 0.0:
            raise TwoMediumInitialError(
                "time must be exactly 0.0 (this module represents the physical "
                f"initial time only), got {self.time!r}"
            )
        if not math.isfinite(self.charge_radius) or not math.isfinite(self.outer_radius):
            raise TwoMediumInitialError(
                "charge_radius and outer_radius must be finite; "
                f"got charge_radius={self.charge_radius!r}, outer_radius={self.outer_radius!r}"
            )
        if self.charge_radius <= 0.0:
            raise TwoMediumInitialError(
                f"charge_radius must be positive, got {self.charge_radius!r}"
            )
        if self.outer_radius <= self.charge_radius:
            raise TwoMediumInitialError(
                "outer_radius must exceed charge_radius; "
                f"got outer_radius={self.outer_radius!r}, charge_radius={self.charge_radius!r}"
            )

        # -- 材料分区 ------------------------------------------------------
        if not np.all(np.isin(self.material, (PRODUCT, AIR))):
            raise TwoMediumInitialError(
                f"material tags must be {PRODUCT} (product) or {AIR} (air)"
            )
        product = self.material == PRODUCT
        air = self.material == AIR
        if not np.any(product) or not np.any(air):
            raise TwoMediumInitialError("the field must contain both product and air cells")
        n_inside = int(np.count_nonzero(product))
        if not (np.all(product[:n_inside]) and np.all(air[n_inside:])):
            raise TwoMediumInitialError(
                "material must be sorted: all product cells first, then all air cells; "
                "fields: material"
            )

        # -- 网格边界 ------------------------------------------------------
        # 容差依据：半径 O(0.05..2) m，双精度 ulp 约 1e-17..4e-16；1e-12 相对
        # 约 2500..5000 ulp，仍远小于单元宽度（约 1e-4 m）。
        if not np.all(np.isfinite(self.cell_edges)):
            raise TwoMediumInitialError("cell_edges must be finite; fields: cell_edges")
        if np.any(np.diff(self.cell_edges) <= 0.0):
            raise TwoMediumInitialError(
                "cell_edges must be strictly increasing; fields: cell_edges"
            )
        if abs(float(self.cell_edges[0])) > 0.0:
            raise TwoMediumInitialError(
                f"cell_edges[0] must be exactly 0 (the centre), got {self.cell_edges[0]!r}"
            )
        if not math.isclose(
            float(self.cell_edges[-1]),
            float(self.outer_radius),
            rel_tol=_POSITION_REL_TOL,
            abs_tol=_POSITION_ABS_TOL,
        ):
            raise TwoMediumInitialError(
                "cell_edges[-1] must equal outer_radius; "
                f"got {self.cell_edges[-1]!r} vs {self.outer_radius!r}"
            )

        # -- 单元中心 ------------------------------------------------------
        # 先查有限性，避免 NaN 绕过后面的误差比较。
        if not np.all(np.isfinite(self.cell_centres)):
            raise TwoMediumInitialError(
                "cell_centres must be finite; fields: cell_centres"
            )
        expected_centres = (
            0.5 * (self.cell_edges[1:] ** 3 + self.cell_edges[:-1] ** 3)
        ) ** (1.0 / 3.0)
        centre_scale = np.maximum(np.abs(expected_centres), np.abs(self.cell_centres))
        if np.any(
            np.abs(self.cell_centres - expected_centres)
            > _GEOMETRY_REL_TOL * np.maximum(centre_scale, 1.0e-300)
        ):
            raise TwoMediumInitialError(
                "cell_centres must match the declared half-volume centre "
                "(0.5*(r_left^3+r_right^3))^(1/3); fields: cell_centres, cell_edges"
            )
        if np.any(self.cell_centres <= self.cell_edges[:-1]) or np.any(
            self.cell_centres >= self.cell_edges[1:]
        ):
            raise TwoMediumInitialError(
                "every cell_centre must lie strictly inside its cell; "
                "fields: cell_centres, cell_edges"
            )

        # -- 单元体积 ------------------------------------------------------
        # 先查有限性：NaN 会让任何 `error > tol` 比较静默为 False。
        if not np.all(np.isfinite(self.cell_volume)):
            raise TwoMediumInitialError(
                "cell_volume must be finite; fields: cell_volume"
            )
        if np.any(self.cell_volume <= 0.0):
            raise TwoMediumInitialError(
                "cell_volume must be positive; fields: cell_volume"
            )
        geometric_volume = (
            self.cell_edges[1:] ** 3 - self.cell_edges[:-1] ** 3
        ) / 3.0
        volume_scale = np.maximum(
            np.abs(geometric_volume), np.abs(self.cell_volume)
        )
        if np.any(
            np.abs(self.cell_volume - geometric_volume)
            > _GEOMETRY_REL_TOL * np.maximum(volume_scale, 1.0e-300)
        ):
            raise TwoMediumInitialError(
                "cell_volume must equal the spherical cell volume "
                "(r_right^3 - r_left^3)/3; fields: cell_volume, cell_edges"
            )

        # -- 产物/空气分界面 -----------------------------------------------
        if not math.isclose(
            float(self.cell_edges[n_inside]),
            float(self.charge_radius),
            rel_tol=_POSITION_REL_TOL,
            abs_tol=_POSITION_ABS_TOL,
        ):
            raise TwoMediumInitialError(
                "the product/air interface cell_edges[n_product] must equal "
                f"charge_radius; got {self.cell_edges[n_inside]!r} vs "
                f"{self.charge_radius!r} (fields: cell_edges, material, charge_radius)"
            )

        # -- 物性 ----------------------------------------------------------
        if np.any(~np.isfinite(self.density)) or np.any(self.density <= 0.0):
            raise TwoMediumInitialError(
                "density must be finite and positive everywhere; fields: density"
            )
        if np.any(~np.isfinite(self.specific_energy)) or np.any(self.specific_energy <= 0.0):
            raise TwoMediumInitialError(
                "specific_energy must be finite and positive everywhere; "
                "fields: specific_energy"
            )
        if np.any(~np.isfinite(self.pressure)) or np.any(self.pressure <= 0.0):
            raise TwoMediumInitialError(
                "pressure must be finite and positive everywhere; fields: pressure"
            )
        if np.any(~np.isfinite(self.sound_speed)) or np.any(self.sound_speed <= 0.0):
            raise TwoMediumInitialError(
                "sound_speed must be finite and positive everywhere; fields: sound_speed"
            )

        # -- 质量 ----------------------------------------------------------
        if not np.all(np.isfinite(self.mass)):
            raise TwoMediumInitialError("mass must be finite; fields: mass")
        if np.any(self.mass <= 0.0):
            raise TwoMediumInitialError("mass must be positive; fields: mass")
        expected_mass = self.density * geometric_volume
        mass_scale = np.maximum(np.abs(expected_mass), np.abs(self.mass))
        if np.any(
            np.abs(self.mass - expected_mass)
            > _GEOMETRY_REL_TOL * np.maximum(mass_scale, 1.0e-300)
        ):
            raise TwoMediumInitialError(
                "mass must equal density * (r_right^3 - r_left^3)/3; "
                "fields: mass, density, cell_edges"
            )

        # -- 速度 ----------------------------------------------------------
        if not np.all(np.isfinite(self.velocity)) or not np.all(
            np.isfinite(self.node_velocity)
        ):
            raise TwoMediumInitialError(
                "velocity and node_velocity must be finite; "
                "fields: velocity, node_velocity"
            )
        if np.any(self.velocity != 0.0) or np.any(self.node_velocity != 0.0):
            raise TwoMediumInitialError(
                "the initial velocity field must be identically zero; "
                "fields: velocity, node_velocity"
            )

        # -- EOS 一致性 ----------------------------------------------------
        # 压力与声速不是独立自由度：它们必须由本对象自带的 (rho, e, EOS 参数)
        # 重新算出。产物用一般 JWL，空气用本对象自己的 gamma 的理想气体。
        # 只与 EOS 比较，不拿 product_state/air_state 里同样可改写的数当依据。
        rho_p = self.density[product]
        e_p = self.specific_energy[product]
        _require_eos_field(
            "product",
            "pressure",
            self.pressure[product],
            jwl_pressure_numpy(rho_p, e_p, self.product_eos),
            "general JWL",
        )
        _require_eos_field(
            "product",
            "sound_speed",
            self.sound_speed[product],
            jwl_sound_speed_numpy(rho_p, e_p, self.product_eos),
            "general JWL",
        )

        rho_a = self.density[air]
        e_a = self.specific_energy[air]
        air_eos = f"ideal gas with gamma = {self.air_gamma!r}"
        _require_eos_field(
            "air",
            "pressure",
            self.pressure[air],
            ideal_gas_pressure_numpy(rho_a, e_a, self.air_gamma),
            air_eos,
        )
        _require_eos_field(
            "air",
            "sound_speed",
            self.sound_speed[air],
            ideal_gas_sound_speed_numpy(rho_a, e_a, self.air_gamma),
            air_eos,
        )

    # -- 基本属性 ---------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(self.material.size)

    @property
    def n_inside(self) -> int:
        """产物单元数；``cell_edges[n_inside]`` 就是 ``r = R_0``。"""
        return int(np.count_nonzero(self.material == PRODUCT))

    @property
    def product_mask(self) -> np.ndarray:
        return self.material == PRODUCT

    @property
    def air_mask(self) -> np.ndarray:
        return self.material == AIR

    @property
    def node_mass(self) -> np.ndarray:
        """节点质量 ``M_j = (m_{j-1}+m_j)/2``，两端取相邻单元的一半。"""
        node = np.zeros(self.n_cells + 1, dtype=np.float64)
        node[0] = 0.5 * self.mass[0]
        node[1:-1] = 0.5 * (self.mass[:-1] + self.mass[1:])
        node[-1] = 0.5 * self.mass[-1]
        return node

    @property
    def product_mass(self) -> float:
        return float(self.mass[self.product_mask].sum())

    @property
    def air_mass(self) -> float:
        return float(self.mass[self.air_mask].sum())

    @property
    def internal_energy(self) -> float:
        return float(np.sum(self.mass * self.specific_energy))

    @property
    def product_internal_energy(self) -> float:
        mask = self.product_mask
        return float(np.sum(self.mass[mask] * self.specific_energy[mask]))

    @property
    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.node_mass * self.node_velocity**2))

    @property
    def total_energy(self) -> float:
        return self.internal_energy + self.kinetic_energy

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.metadata.get("model_id", TWO_MEDIUM_MODEL_ID),
            "physical_time_s": self.time,
            "charge_radius_m": self.charge_radius,
            "outer_radius_m": self.outer_radius,
            "n_cells": self.n_cells,
            "n_product_cells": self.n_inside,
            "n_air_cells": self.n_cells - self.n_inside,
            "product_mass": self.product_mass,
            "air_mass": self.air_mass,
            "product_internal_energy": self.product_internal_energy,
            "internal_energy": self.internal_energy,
            "kinetic_energy": self.kinetic_energy,
            "total_energy": self.total_energy,
            "product_pressure_pa": self.product_state.get("pressure"),
            "air_pressure_pa": self.air_state.get("pressure"),
            "pressure_jump_pa": (
                float(self.product_state.get("pressure", 0.0))
                - float(self.air_state.get("pressure", 0.0))
            ),
            **self.metadata,
        }


def _build_edges(
    *,
    charge_radius: float,
    outer_radius: float,
    cells_inside: int,
    cells_outside: int,
    outside_growth: float,
) -> np.ndarray:
    """建一套在 ``R_0`` 处**恰好有节点**的球对称网格。

    内部 ``[0, R_0]`` 均匀；外部 ``[R_0, R_outer]`` 按固定比 ``outside_growth``
    几何放宽。界面永远是单元边界，因此材料归属没有歧义。
    """
    inner = np.linspace(0.0, charge_radius, cells_inside + 1)
    span = outer_radius - charge_radius
    if outside_growth == 1.0:
        widths = np.full(cells_outside, span / cells_outside, dtype=np.float64)
    else:
        log_growth = math.log(outside_growth)
        if cells_outside * log_growth > 700.0:
            raise TwoMediumInitialError(
                "outside_growth ** cells_outside overflows; choose a smaller growth or fewer air cells"
            )
        first = span * math.expm1(log_growth) / math.expm1(cells_outside * log_growth)
        widths = first * np.exp(np.arange(cells_outside, dtype=np.float64) * log_growth)
    outer = charge_radius + np.concatenate(([0.0], np.cumsum(widths)))
    edges = np.concatenate([inner, outer[1:]])
    # 让外边界逐位等于用户给的值；cumsum 的舍入不应该改变域的边界。
    edges[-1] = outer_radius
    if not np.all(np.diff(edges) > 0.0):
        raise TwoMediumInitialError("the generated grid is not strictly increasing")
    return edges


def build_two_medium_initial_field(
    *,
    charge_radius: float,
    product_density: float | None = None,
    product_specific_energy: float | None = None,
    air_density: float | None = None,
    air_pressure: float | None = None,
    air_gamma: float | None = None,
    outer_radius: float | None = None,
    cells_inside: int = 64,
    cells_outside: int = 96,
    outside_growth: float = 1.005,
    jwl: JWLParameters | None = None,
) -> TwoMediumInitialField:
    """构造 ``t_phys = 0`` 的球对称两介质分段常数初值。

    参数
    ----
    charge_radius:
        装药半径 ``R_0``，m。
    product_density:
        产物**初始**常数密度 ``rho_p0``，kg/m^3。默认取实际 EOS 参数集的参考
        密度 ``jwl.rho0``。显式给出时按调用者的值使用，可以是合法但不等于
        参考密度的值；它不会改变 ``e_p0 = E0/rho_ref`` 中的 ``rho_ref``。
    product_specific_energy:
        产物常数比内能 ``e_p0``，J/kg。默认为 ``E0/rho_ref``；若给出，则**按
        J/kg 直接使用**，不再除密度。
    air_density, air_pressure, air_gamma:
        环境空气参数。默认取材料卡片的 ``1.225 kg/m^3``、``101320 Pa``、
        ``1.4``。空气比内能由 ``e_air = p_amb/((gamma-1) rho_air)`` 反解，
        因而与给定环境压力严格一致。
    outer_radius:
        外边界半径，m。默认 ``40 R_0``。必须严格大于 ``R_0``。
    cells_inside, cells_outside:
        产物区与空气区的单元数。
    outside_growth:
        空气区宽度增长比；``1`` 表示均匀。
    jwl:
        JWL 参数集。默认使用 ``material.product_jwl()``。

    返回
    ----
    :class:`TwoMediumInitialField`
        只含初值与元数据，不含任何推进结果。
    """
    params = jwl or mat.product_jwl()
    params.validate()

    R0 = _finite_positive("charge_radius", charge_radius)
    # 初始密度与 EOS 参考密度是两个量：未显式给出初始密度时，取自**实际**
    # 参数集的 rho0，而不是全局材料默认值；否则自定义 JWL 会出现“密度按
    # 1630、比内能按 E0/rho0(custom)”这种自相矛盾的初值。
    rho_product = _finite_positive(
        "product_density",
        float(params.rho0) if product_density is None else product_density,
    )
    if product_specific_energy is None:
        e_product = float(params.initial_specific_energy(mat.PRODUCT_VOLUME_ENERGY))
        e_source = "E0/rho_ref via JWLParameters.initial_specific_energy"
    else:
        e_product = _finite_positive("product_specific_energy", product_specific_energy)
        e_source = "user-provided specific energy in J/kg (not divided by density again)"
    if not math.isfinite(e_product) or e_product <= 0.0:
        raise TwoMediumInitialError(
            f"product specific energy must be finite and positive, got {e_product!r}"
        )

    rho_air = _finite_positive(
        "air_density", mat.AIR_DENSITY if air_density is None else air_density
    )
    p_air = _finite_positive(
        "air_pressure", mat.AIR_PRESSURE if air_pressure is None else air_pressure
    )
    gamma = float(mat.AIR_GAMMA if air_gamma is None else air_gamma)
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise TwoMediumInitialError(f"air_gamma must be finite and > 1, got {air_gamma!r}")

    outer = _finite_positive(
        "outer_radius", 40.0 * R0 if outer_radius is None else outer_radius
    )
    if outer <= R0:
        raise TwoMediumInitialError(
            f"outer_radius must exceed charge_radius; got {outer!r} <= {R0!r}"
        )

    n_in = _positive_int("cells_inside", cells_inside)
    n_out = _positive_int("cells_outside", cells_outside)
    growth = float(outside_growth)
    if not math.isfinite(growth) or growth < 1.0:
        raise TwoMediumInitialError(
            f"outside_growth must be finite and >= 1, got {outside_growth!r}"
        )

    edges = _build_edges(
        charge_radius=R0,
        outer_radius=outer,
        cells_inside=n_in,
        cells_outside=n_out,
        outside_growth=growth,
    )
    volume = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    centres = (0.5 * (edges[1:] ** 3 + edges[:-1] ** 3)) ** (1.0 / 3.0)

    material = np.concatenate(
        [
            np.full(n_in, PRODUCT, dtype=np.int64),
            np.full(n_out, AIR, dtype=np.int64),
        ]
    )
    density = np.concatenate(
        [np.full(n_in, rho_product), np.full(n_out, rho_air)]
    )

    # 空气：从给定的环境压力反解比内能；压力再由 EOS 重算，二者必须一致。
    e_air = float(
        ideal_gas_specific_energy_numpy(
            np.array([rho_air]), np.array([p_air]), gamma
        )[0]
    )
    if not math.isfinite(e_air) or e_air <= 0.0:
        raise TwoMediumInitialError(
            f"air specific energy from the ambient pressure is inadmissible: {e_air!r}"
        )
    p_air_check = float(
        ideal_gas_pressure_numpy(np.array([rho_air]), np.array([e_air]), gamma)[0]
    )
    if not math.isclose(p_air_check, p_air, rel_tol=1.0e-12, abs_tol=0.0):
        raise TwoMediumInitialError(
            "air pressure recomputed from (rho, e) does not match the ambient pressure: "
            f"{p_air_check!r} vs {p_air!r}"
        )

    # 产物：压力只由 EOS 计算，不接受独立的压力输入。
    p_product = float(
        jwl_pressure_numpy(
            np.array([rho_product]), np.array([e_product]), params
        )[0]
    )
    if not math.isfinite(p_product) or p_product <= 0.0:
        raise TwoMediumInitialError(
            "the product (rho, e) state gives a non-positive or non-finite JWL pressure: "
            f"rho={rho_product!r}, e={e_product!r}, p={p_product!r}"
        )

    c_product = float(
        jwl_sound_speed_numpy(
            np.array([rho_product]), np.array([e_product]), params
        )[0]
    )
    c_air = float(
        ideal_gas_sound_speed_numpy(
            np.array([rho_air]), np.array([e_air]), gamma
        )[0]
    )
    if not math.isfinite(c_product) or c_product <= 0.0:
        raise TwoMediumInitialError(f"product sound speed is inadmissible: {c_product!r}")
    if not math.isfinite(c_air) or c_air <= 0.0:
        raise TwoMediumInitialError(f"air sound speed is inadmissible: {c_air!r}")

    specific_energy = np.concatenate(
        [np.full(n_in, e_product), np.full(n_out, e_air)]
    )
    pressure = np.concatenate(
        [np.full(n_in, p_product), np.full(n_out, p_air)]
    )
    sound_speed = np.concatenate(
        [np.full(n_in, c_product), np.full(n_out, c_air)]
    )
    velocity = np.zeros(edges.size - 1, dtype=np.float64)
    node_velocity = np.zeros(edges.size, dtype=np.float64)
    mass = density * volume

    product_state = {
        "material": "product",
        "density": rho_product,
        "velocity": 0.0,
        "specific_energy": e_product,
        "pressure": p_product,
        "sound_speed": c_product,
        "volume": float(volume[:n_in].sum()),
        "mass": float(mass[:n_in].sum()),
        "internal_energy": float(np.sum(mass[:n_in] * e_product)),
    }
    air_state = {
        "material": "air",
        "density": rho_air,
        "velocity": 0.0,
        "specific_energy": e_air,
        "pressure": p_air,
        "sound_speed": c_air,
        "volume": float(volume[n_in:].sum()),
        "mass": float(mass[n_in:].sum()),
        "internal_energy": float(np.sum(mass[n_in:] * e_air)),
    }

    metadata: dict[str, Any] = {
        "model_id": TWO_MEDIUM_MODEL_ID,
        "construction": TWO_MEDIUM_CONSTRUCTION,
        "physical_time_s": 0.0,
        "time_convention": (
            "t_phys = 0; this module does not define a PINN handoff time"
        ),
        "interface_rule": (
            "r = R0 is a cell face: the left cell is product, the right cell is air; "
            "no averaging, no smoothing layer, no forced pressure equality"
        ),
        "interface_radius_m": R0,
        "pressure_jump_pa": p_product - p_air,
        # 初始密度（场里实际用的）与 EOS 参考密度（只进 V=rho0/rho 和
        # e_p0=E0/rho0）分开记录，二者可以不同。
        "product_density_kg_m3": rho_product,
        "product_density_source": (
            "user-provided initial density; may differ from the EOS reference density"
            if product_density is not None
            else "product_eos.rho0 (EOS reference density)"
        ),
        "eos_reference_density_kg_m3": float(params.rho0),
        "eos_reference_density_role": (
            "used for V = rho0/rho in the JWL EOS and for e_p0 = E0/rho0; "
            "it is not the initial density when product_density is given"
        ),
        "product_specific_energy_j_kg": e_product,
        "product_specific_energy_source": e_source,
        "product_pressure_pa": p_product,
        "product_pressure_source": "JWL EOS evaluated at (rho_p0, e_p0)",
        "E0_volume_energy_j_m3": float(mat.PRODUCT_VOLUME_ENERGY),
        "E0_definition": "per reference volume; e_p0 = E0/rho_ref",
        "rho_ref_kg_m3": float(params.rho0),
        "air_density_kg_m3": rho_air,
        "air_specific_energy_j_kg": e_air,
        "air_pressure_pa": p_air,
        "air_pressure_source": (
            "ideal gas; e_air = p_amb/((gamma-1) rho_air), then p = (gamma-1) rho e"
        ),
        "gamma_air": gamma,
        "acceptance_status": (
            "NOT APPLICABLE: initial field only; this is not a ReferenceField"
        ),
        "reference_field": False,
        "handoff_time_s": None,
        "solver_dependency": (
            "none; does not import or call ProductsDetonation or SphericalPrecursor"
        ),
        "units": (
            "SI (m, kg, s, Pa, J/kg); mass and energy are per unit solid angle"
        ),
        "cell_volume_rule": "(r_{i+1}^3 - r_i^3)/3 (spherical, per unit solid angle)",
    }

    return TwoMediumInitialField(
        time=0.0,
        charge_radius=R0,
        outer_radius=outer,
        cell_edges=edges,
        cell_centres=centres,
        cell_volume=volume,
        density=density,
        velocity=velocity,
        specific_energy=specific_energy,
        pressure=pressure,
        sound_speed=sound_speed,
        material=material,
        mass=mass,
        node_velocity=node_velocity,
        product_eos=params,
        air_gamma=gamma,
        product_state=product_state,
        air_state=air_state,
        metadata=metadata,
    )


__all__ = [
    "AIR",
    "MATERIAL_NAMES",
    "PRODUCT",
    "TWO_MEDIUM_CONSTRUCTION",
    "TWO_MEDIUM_MODEL_ID",
    "TwoMediumInitialError",
    "TwoMediumInitialField",
    "build_two_medium_initial_field",
]
