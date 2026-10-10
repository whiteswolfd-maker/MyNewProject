"""材料参数与物理零点：本仓库唯一的一份数值来源。

提示词 §4 的表格是这里每一个数字的唯一出处。任何模块都不得自己再写一遍
JWL 系数或空气常数——第二份副本就是两份数值开始漂移的方式。

注意 `B = 3.7471e9 Pa`。旧项目里出现过的 `3.231e9 Pa` 不是本规范采用的值，
不在此处，也不应被重新引入。

全部内部计算采用 SI：m、kg、s、Pa、J/kg。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .eos import JWLParameters

# --------------------------------------------------------------------------
# 爆轰产物（TNT）：*MAT_HIGH_EXPLOSIVE_BURN + *EOS_JWL
# --------------------------------------------------------------------------

JWL_A = 3.7120e11
JWL_B = 3.7471e9
JWL_R1 = 4.15
JWL_R2 = 0.95
JWL_OMEGA = 0.30

#: 产物参考（未反应装药）密度，kg/m^3。JWL 的体积比 V = rho0/rho 用它。
PRODUCT_REFERENCE_DENSITY = 1630.0

#: *EOS_JWL 的 E0，按**参考体积**定义，J/m^3。
PRODUCT_VOLUME_ENERGY = 6.0e9

#: *EOS_JWL 的 V0。
PRODUCT_V0 = 1.0

#: LS-DYNA 能量参考零点修正，J/kg。本算例为 0。
ENERGY_SHIFT = 0.0

#: 仅供参考记录：CJ 压力与爆速。不用于整球初始压力，也不用 CJ 重新标定 E0。
CJ_PRESSURE_REFERENCE = 2.1e10
DETONATION_VELOCITY_REFERENCE = 6930.0

# --------------------------------------------------------------------------
# 环境空气：*MAT_NULL + *EOS_LINEAR_POLYNOMIAL
#
# 卡片（四个算例完全一致）：
#   *MAT_NULL                ro = 1.225E-12        -> 1.225 kg/m^3
#   *EOS_LINEAR_POLYNOMIAL   c0..c3 = 0, c4 = c5 = 0.4, c6 = 0, E0 = 0.2533, V0 = 1
#
# LS-DYNA 的线性多项式 EOS 是
#   p = c0 + c1*mu + c2*mu^2 + c3*mu^3 + (c4 + c5*mu + c6*mu^2)*E,  mu = rho/rho0 - 1
# 其中 E = rho0*e 是单位**参考体积**的内能。这里 c0..c3 = 0、c6 = 0，而且
# **c5 = c4**，于是
#   p = c4*E + c4*mu*E = c4*E*(1 + mu) = c4*(rho/rho0)*(rho0*e) = c4*rho*e
# 也就是说它**恰好**是 gamma - 1 = c4 的理想气体。这个归约只在 c5 = c4 时成立，
# 所以 :func:`air_from_linear_polynomial` 会强制检查，不满足就报错——不能悄悄把
# 一个非理想气体的卡片当成理想气体用。
# --------------------------------------------------------------------------

#: 卡片 E0，单位参考体积，J/m^3。注意它是 0.2533 MPa 而不是精确的 1 atm。
AIR_VOLUME_ENERGY = 0.2533e6

AIR_DENSITY = 1.225

#: gamma - 1 = c4（并要求 c5 = c4）。卡片 c4 = c5 = 0.4，故 gamma = 1.4。
AIR_GAMMA = 1.4

#: 标准大气压，Pa。**这是参考值，不是本算例的环境压力。**
#:
#: 卡片用 E0 = 0.2533 MPa 表示 1 atm，保留四位有效数字。由它反推的环境压力是
#: 101320.0 Pa，比标准大气压小 5 Pa。DYNA 实际初始化出来的就是 101320 Pa——这一点
#: 在 d3plot 里可以直接读到（空气单元的 `p` 最小值恒为 101320）。
#:
#: 二者不能混用。凡是要与 DYNA 数据比较的量（环境态、超压基准、激波阈值）一律用
#: :data:`AIR_PRESSURE`（卡片值）；标准大气压只作为文档里的参考保留。
STANDARD_ATMOSPHERE = 101325.0

def _air_from_card() -> tuple[float, float]:
    """由卡片 E0 与理想气体关系得到 (e0, p0)。"""
    gamma_minus_one = AIR_GAMMA - 1.0
    e0 = AIR_VOLUME_ENERGY / AIR_DENSITY
    return e0, gamma_minus_one * AIR_DENSITY * e0


#: 空气静止比内能，e0 = E0/rho0 = 205... J/kg（卡片值，不是 p_std/[(gamma-1)rho]）。
AIR_SPECIFIC_ENERGY, AIR_PRESSURE = _air_from_card()


def ambient_state_from_card() -> "AmbientAir":
    """本算例的环境态，四个量全部由上面的卡片值推出。"""
    return AmbientAir(
        density=AIR_DENSITY,
        velocity=0.0,
        specific_energy=AIR_SPECIFIC_ENERGY,
        pressure=AIR_PRESSURE,
    )

# --------------------------------------------------------------------------
# 由上面推出的量
# --------------------------------------------------------------------------


def product_jwl() -> JWLParameters:
    """规范 §4 的 JWL 参数集。"""
    params = JWLParameters(
        A=JWL_A,
        B=JWL_B,
        R1=JWL_R1,
        R2=JWL_R2,
        omega=JWL_OMEGA,
        rho0=PRODUCT_REFERENCE_DENSITY,
        energy_shift=ENERGY_SHIFT,
    )
    params.validate()
    return params


#: e_* = E0 / rho0，产物的比内能尺度，J/kg。
PRODUCT_SPECIFIC_ENERGY = PRODUCT_VOLUME_ENERGY / PRODUCT_REFERENCE_DENSITY

#: U_* = sqrt(e_*)，唯一的速度尺度，m/s。两个材料共用。
REFERENCE_VELOCITY = math.sqrt(PRODUCT_SPECIFIC_ENERGY)


def product_pressure_at_reference_state() -> float:
    """参考状态 ``(rho_ref, e_*)`` 处的 JWL 压力，Pa。

    **这不是初值压力。** 方案 A 把这个数（8.220261897e9 Pa）当成整球均匀初压，
    方案 B 没有这个量：初值场是中心起爆的有限爆速产物场，每个位置的 ``(rho, e)``
    都不同，压力也就不同。这里保留它只有一个用途——给 EOS 一个固定的往返参照点，
    确认 JWL 实现没有漂移。

    这里**通过 EOS 计算**，而不是作为一个独立的输入常数。把 8.220261897e9 写成
    常量，就等于允许它和 EOS 漂移；这是本仓库刻意避免的一类不一致。
    """
    from .eos import jwl_pressure_numpy
    import numpy as np

    return float(
        jwl_pressure_numpy(
            np.array([PRODUCT_REFERENCE_DENSITY]),
            np.array([PRODUCT_SPECIFIC_ENERGY]),
            product_jwl(),
        )[0]
    )


@dataclass(frozen=True)
class AmbientAir:
    """静止环境空气的四个状态量。"""

    density: float
    velocity: float
    specific_energy: float
    pressure: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.density, self.velocity, self.specific_energy, self.pressure)

    def as_dict(self) -> dict[str, float]:
        return {
            "rho_air_ambient": self.density,
            "u_air_ambient": self.velocity,
            "e_air_ambient": self.specific_energy,
            "p_air_ambient": self.pressure,
        }


def ambient_air() -> AmbientAir:
    return AmbientAir(
        density=AIR_DENSITY,
        velocity=0.0,
        specific_energy=AIR_SPECIFIC_ENERGY,
        pressure=AIR_PRESSURE,
    )


__all__ = [
    "AIR_DENSITY",
    "AIR_GAMMA",
    "AIR_PRESSURE",
    "AIR_SPECIFIC_ENERGY",
    "AIR_VOLUME_ENERGY",
    "STANDARD_ATMOSPHERE",
    "AmbientAir",
    "CJ_PRESSURE_REFERENCE",
    "DETONATION_VELOCITY_REFERENCE",
    "ENERGY_SHIFT",
    "JWL_A",
    "JWL_B",
    "JWL_OMEGA",
    "JWL_R1",
    "JWL_R2",
    "PRODUCT_REFERENCE_DENSITY",
    "PRODUCT_SPECIFIC_ENERGY",
    "PRODUCT_V0",
    "PRODUCT_VOLUME_ENERGY",
    "REFERENCE_VELOCITY",
    "ambient_air",
    "ambient_state_from_card",
    "product_pressure_at_reference_state",
    "product_jwl",
]
