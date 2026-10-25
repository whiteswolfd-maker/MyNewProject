"""方案 B 的核心：中心起爆、有限爆速传播的球对称产物场。

.. warning::

   **遗留链路（2026-10-21 起）**。新初值是
   ``physics/two_medium_initial.py`` 的均匀产物球 + 静止空气，短时推进是
   ``physics/two_medium_startup.py``；它们都不导入本模块。本模块现在只服务
   旧参考场链路（``cli/reference_field.py`` →
   ``precursor.solve_reference_field`` → ``cli._common.build_context``）。
   在 ``build_context`` 切换到 ``TwoMediumHandoff`` 之前不能删除，否则会留下
   悬空导入；删除顺序见 ``DECISION_2026-10-21_UNIFORM_PRODUCT_SPHERE.md``。

这个模块回答提示词 §2 的那一问——"B 的产物场到底是怎么来的"。它**不是**把某个
仿真帧改名，也不是把一个均匀状态铺满装药球。

模型
----

**只解爆轰产物区，不解未反应装药。** 理由在下面"为什么不建未反应材料"一节。

未知量是爆轰波前**后面**的产物场 ``rho(r), u(r), e(r), p(r)``；波前本身以卡片
爆速 ``D`` 从中心向外推进，位置

.. math::  r_f(t) = D t

波前两侧跨越的是质量、动量与（带放热的）能量条件。反应前态取

.. math::  \\rho = \\rho_{\\rm ref},\\quad u = 0,\\quad e = 0,\\quad p = 0

也就是**以未反应装药静止、零压、零内能为能量零点**。放热按单位质量

.. math::  Q = E_0/\\rho_{\\rm ref}

其中 ``E0`` 是卡片 ``*EOS_JWL`` 的爆轰能量（单位参考体积）。产物用同一个卡片的
一般 JWL

.. math::  p = f(V) + \\omega \\rho e,\\qquad V = \\rho_{\\rm ref}/\\rho

波前关系（前向质量、动量、能量）
--------------------------------

在波前参考系里，来流速度是 ``-D``、去流速度是 ``u - D``。质量守恒给出

.. math::  \\rho_{\\rm ref} D = \\rho (D - u) \\;\\Longrightarrow\\; u = D(1-V)

动量守恒（``p_up = 0``）给出

.. math::  p = \\rho_{\\rm ref} D\\, u

能量守恒（带放热 ``Q``、能量零点 ``e_up = 0``）给出

.. math::  e = p\\,(1-V)/(2\\rho_{\\rm ref}) + Q

把后两式代入 JWL 并消去 ``e``，得到**一条只含 V 的方程**

.. math::

    p_H(V) = \\frac{f(V) + \\omega E_0/V}{1 - \\omega(1-V)/(2V)},
    \\qquad D_{\\rm RH}(V)^2 = \\frac{p_H(V)}{\\rho_{\\rm ref}(1-V)}

于是"给定 D 求波前状态"就是解 ``D_RH(V) = D``。

**两个根与物理选支**
--------------------

``D_RH(V)`` 在 ``V_CJ`` 处取极小值，极小值就是理想 CJ 爆速。因此：

* ``D < D_CJ``：无解（自持爆轰不能低于 CJ 速度）；
* ``D = D_CJ``：一个根，即 CJ 点；
* ``D > D_CJ``：两个根，分别对应激波理论上熟悉的强、弱两支。

**卡片给的是 ``D = 6930 m/s``，而本卡片 JWL + E0 的 ``D_CJ = 6883.14 m/s``，
所以 ``D > D_CJ``，有两个根。选支判据是声速比**

.. math::  \\frac{c}{D-u} = \\frac{c}{D V}

其中 ``c`` 由一般 JWL 的声速公式给出。**平面弱激波后流动相对波前必为亚声速**，
所以物理支是 ``c/(D-u) > 1`` 的那一个。实测：两支的声速比是 1.084 与 0.918，
取前者。

这一点必须写清：**它不是 CJ 状态**。把 CJ 状态当成波前状态是在一个 D 与卡片的
JWL/E0 不相容的模型里强行施加 CJ 声速条件，而这个模型并不满足它（见
``ideal_cj_state`` 与 §3 的相容性核查）。

能量与质量的**精确**积分不变量
------------------------------

这套方程的产物区在任意时刻满足两条闭式不变量（单位立体角）：

.. math::

    M(t) = \\rho_{\\rm ref} r_f(t)^3/3, \\qquad
    E(t) = E_0\\, r_f(t)^3/3

第一条是"波前扫过的装药全在波后"。

第二条要**带运动边界项**才能得到，这里的推导必须写全。产物区 ``[0, r_f]`` 是
控制体，它的能量方程（守恒形式）是

.. math::

    \\frac{dE}{dt} = -\\oint \\big[ \\rho E (u - v_b) + p\\,u \\big]\\cdot n\\,dA,
    \\qquad E = e + u^2/2

其中 ``v_b`` 是边界速度、``n`` 是外法线。波前处 ``v_b = D``、``u = u_f``、
``n`` 指向 ``+r``，质量通量 ``\\rho_f (u_f - D) = -\\rho_{\\rm ref} D``（向内），
所以（单位立体角）

.. math::

    \\frac{dE}{dt} = \\rho_{\\rm ref} D r_f^2 (e_f + u_f^2/2) - p_f u_f r_f^2

**两项**，不是三项。``p u \\cdot n`` 这一项**已经**是完整的压力功；若再把它拆成
"流动功 ``+p_f/\\rho_f\\cdot\\rho_{\\rm ref}D``"加一项、又减一个 ``-p_f u_f``，
就是在 ``p u`` 之外重复计了一次压力功。代入能量跳跃 ``e_f = Q + u_f^2/2`` 与
动量跳跃 ``p_f = \\rho_{\\rm ref} D u_f``：

.. math::

    \\rho_{\\rm ref} D\\,(Q + u_f^2) - p_f u_f
      = \\rho_{\\rm ref} D Q + \\rho_{\\rm ref} D u_f^2 - \\rho_{\\rm ref} D u_f^2
      = \\rho_{\\rm ref} D Q

即 ``dE/dt = rho_ref D Q r_f^2``，积分得 ``E_0 r_f^3/3``。

**数值验证（两项）**：按实测波前状态（``e_f=5.992187e6``、``u_f=2149.979``、
``p_f=2.428595e10``），``(e_f+u_f^2/2)*rho_ref*D - p_f*u_f`` 除以
``rho_ref*D*Q`` 得 **0.9999999999999997**。若误加第三个流动功项
``+(p_f/rho_f)*rho_ref*D``，同一个比值变成 **3.79**——那是混用了总比能与总焓、
把压力功重复计了一次。

``_step_budget`` 的 ``physical_input`` 用的正是这两项抵消后的 ``\\Delta m\\, Q``，
与上述推导一致。

**这两条与 D 的取值无关**，所以它们是数值格式的强独立核查：任何质量吸积、边界
通量或几何项写错都会立刻破坏它们。见 ``ProductsDetonation`` 的验收输出。

离散收支的定位结果（§3.2）
--------------------------

``detonation_energy_budget()`` 逐步输出离散收支。离散总能量取

.. math::

    E_h = \\sum_i m_i e_i + \\frac{1}{2}\\sum_j M_j u_j^2,
    \\qquad M_j = \\frac{m_{j-1}+m_j}{2}

物理输入只有一项：本步被波前扫过的那部分装药释放的化学能 ``\\Delta m\\, Q``。
逐步残差 ``R = \\Delta E_h - \\Delta m\\, Q``。

**一个已验证的关系**（逐步，相对 ``E_exact`` 最大偏离 1.7e-8，是实测近似值）：

.. math::

    \\sum_j M_j^{\\rm new} u_j^{\\rm old} \\Delta u_j
      = \\sum_{i<k} P_i \\Delta V_i - P_k \\Delta V_k^{\\rm node}

左边是动量方程做的功（``work_momentum``），右边是内部单元的 ``p dV``
（``work_interior``）减去节点 k 扫过体积的那一项（``work_front_inner``）。
这一条只说明**这两套定义彼此自洽**；它**不**等于"时间、几何、压力功问题都已排除"。

**实测收支**（n=500 cfl=0.2，累计除以 ``E_exact``）：

===========  ==========
量           值
===========  ==========
物理输入      +1.000000
内能增量      +0.915458
动能增量      +0.088630
残差          +0.004088
===========  ==========

**排除性实验**（改一项、测一次）：

* 前端功改成分面 ``p_f (\\Delta m/\\rho_{\\rm ref}) - P_k \\Delta V_k^{\\rm node}``
  -> ``-4.4e-2``（更差）。
* 节点 k 补成动量守恒 -> ``+2.56e-1``（更差）。
* 节点 k 加来流动量 -> ``+1.68e-2``（更差）。

**这些实验只排除那几个特定的单点改动，不证明基准里对应项是对的**：残差可能是
多个误差互相抵消后的净值，改掉一个、抵消被打破，残差反而变大，**不能**反推出
"被改的那一项原先是正确的"。

**残差的来源尚未定论。** 一个候选是"吸积在离散能量里怎么分摊"——新物质实际速度
是 ``u_f``，节点 k 因质量增长被稀释。但 ``velocity_lumping_deficit`` 只有
1.9e-3，远小于残差 4.1e-3，不足以解释全部。**不能写"残差已定位"**；只能说这个
候选被证伪得最少。在此之前**不宣称相容离散已经完成**。

数值方法
--------

一维球对称**拉格朗日有限体积 + 质量吸积波前**：

* 网格按**原始半径**均匀划分，``radii0[j] = j * dr``。未反应装药在波前到达之前
  是静止未扰动的，所以"原来在半径 x 处的物质，在波前到达它的那一刻仍在 x 处"，
  于是原始半径同时是物质坐标：**波前在 ``t = j dr / D`` 到达节点 j**；
* 节点 1..n 是物质节点，按动量方程 ``du/dt = -r^2 dp/dm`` 运动；节点 0 是球心
  （``u = 0``）；节点 n+1 是**非物质的波前节点**，位置**给定**为 ``D t``；
* 单元 0..n-1 是拉格朗日单元（质量固定）；单元 n 是**正在被波前扫过**的前端单元，
  质量随时间按 ``m_p = rho_ref (r_f^3 - (n dr)^3)/3`` 增长，新质量以波前状态的
  **比焓** ``h_f = e_f + p_f/rho_f`` 进入，内能按带流入的控制体平衡更新
  ``d(m e) = dm h_f - p dV``；
* 人工粘性用 von Neumann–Richtmyer，系数与卡片 ``*CONTROL_BULK_VISCOSITY`` 一致。

前端单元的**近似**：反应区没有分辨，一个单元内被扫过的物质被当成一个均匀状态，
质量加权混合。这是 ZND 的"瞬时反应"极限，也是没有未反应 EOS 时唯一能写下来的
东西。它会随网格加密收敛，验收里按三套网格给出。

为什么不建未反应材料
--------------------

卡片只给**产物**的 JWL（``*EOS_JWL``）与爆速，**没有未反应装药的 EOS**。JWL 在
``V = 1, e = 0`` 处给 ``f(1) = 6.42 GPa`` 而不是零，所以拿它当未反应态会凭空造出
一个 6.4 GPa 的静止炸药。凭空补一个未反应 EOS 属于提示词 §2 禁止的"偷偷改变材料
卡"。因此未反应区**不建模**：波前之外什么都没有，产物的外边界就是波前本身。

适用范围
--------

* 不分辨反应区结构与未反应 Hugoniot，因此**不预测**起爆判据、失效直径或起爆延迟；
* 波前速度取卡片值且**恒定**，因此不描述爆轰波的曲率衰减；
* 中心附近（前几个单元）的分辨率最低，且球对称假设排除了任何非球扰动；
* 只用于生成**交接时刻的初始场**，不作为对 DYNA 结果的独立预测。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import material as mat
from .eos import JWLParameters, jwl_pressure_numpy, jwl_sound_speed_numpy


#: 产物场**物理模型与数值格式**的版本。改动波前关系、选支、前端单元的吸积或
#: 能量记账，都必须 +1。
#:
#: 为什么要有这个常量：参考场缓存是按"求解参数"做键的，而上面那些改动**不会**
#: 改变任何参数——网格数、CFL、增长比、爆速全都一样，产出的场却不同。少了这个
#: 版本，一次修好能量记账之后，缓存里那份偏差 7.4% 的旧场会被原样复用，而报告
#: 里连一个字都不会变。
#:
#: 第 1 版：前端单元的压力功按**波前压力**扣（此前按本单元压力扣，欠扣，能量
#: 不变量偏高 5.6%~7.4%）。
#: 第 2 版：加入**逐步离散收支**（``detonation_energy_budget`` / ``_step_budget``）
#: 并把物理输入改成 ``Δm Q``。推进本身没变（同一网格给出逐位相同的场），但报告
#: 里的残差口径变了——旧口径把 ``inflow_enthalpy - work_interior - work_front``
#: 当作输入，实测它恒等于 ``ΔKE``，也就是说那个"残差"什么都不检验。
DETONATION_MODEL_VERSION = 2


class DetonationError(RuntimeError):
    """产物场构造失败。"""


def _node_masses(cell_mass: np.ndarray) -> np.ndarray:
    """节点质量 ``M_j = (m_{j-1} + m_j)/2``，两端各取相邻单元的一半。"""
    node = np.zeros(cell_mass.size + 1, dtype=np.float64)
    node[0] = 0.5 * cell_mass[0]
    node[1:-1] = 0.5 * (cell_mass[:-1] + cell_mass[1:])
    node[-1] = 0.5 * cell_mass[-1]
    return node


def _physical_volume_ratio_range(jwl: JWLParameters) -> tuple[float, float]:
    """``V`` 上物理有意义的一段的端点。

    ``p_H(V)`` 的分母 ``1 - omega(1-V)/(2V)`` 在 ``V = omega/(2+omega)``（本卡片约
    0.1304）处过零并在其下变号。那里 ``p_H`` 是一个极点而不是物理状态：越过它去
    求 ``D_RH`` 的极小值或者找交点，都会落在"平方根被钳成 0"的假根上。所以搜索区间
    从极点之上开始。
    """
    pole = jwl.omega / (2.0 + jwl.omega)
    return pole * 1.05, 1.0 - 1.0e-9


# --------------------------------------------------------------------------
# 冷压项与波前关系
# --------------------------------------------------------------------------


def cold_pressure(volume_ratio, jwl: JWLParameters):
    """JWL 的冷压项 ``f(V)``，``V = rho0/rho``。"""
    V = np.asarray(volume_ratio, dtype=np.float64)
    if np.any(V <= 0.0):
        raise DetonationError("JWL 的体积比必须为正")
    return (
        jwl.A * (1.0 - jwl.omega / (jwl.R1 * V)) * np.exp(-jwl.R1 * V)
        + jwl.B * (1.0 - jwl.omega / (jwl.R2 * V)) * np.exp(-jwl.R2 * V)
    )


def cold_pressure_derivative(volume_ratio, jwl: JWLParameters):
    """``df/dV``。声速要用到它（``df/drho = -(V/rho) df/dV``）。"""
    V = np.asarray(volume_ratio, dtype=np.float64)
    return (
        jwl.A
        * np.exp(-jwl.R1 * V)
        * (jwl.omega / (jwl.R1 * V**2) - jwl.R1 * (1.0 - jwl.omega / (jwl.R1 * V)))
        + jwl.B
        * np.exp(-jwl.R2 * V)
        * (jwl.omega / (jwl.R2 * V**2) - jwl.R2 * (1.0 - jwl.omega / (jwl.R2 * V)))
    )


def jwl_pressure_from_volume_ratio(volume_ratio, specific_energy, jwl: JWLParameters):
    """``p(V, e) = f(V) + omega rho e``，``rho = rho0/V``。"""
    V = np.asarray(volume_ratio, dtype=np.float64)
    rho = jwl.rho0 / V
    return cold_pressure(V, jwl) + jwl.omega * rho * np.asarray(specific_energy)


def rayleigh_hugoniot_pressure(volume_ratio, *, jwl: JWLParameters, volume_energy: float):
    """``p_H(V)``：Rayleigh 线与 JWL 及带放热能量关系的联立解。

    推导：``e = p(1-V)/(2 rho0) + E0/rho0``，代入 ``p = f(V) + omega rho e``，
    解出 ``p``。
    """
    V = np.asarray(volume_ratio, dtype=np.float64)
    numerator = cold_pressure(V, jwl) + jwl.omega * volume_energy / V
    denominator = 1.0 - jwl.omega * (1.0 - V) / (2.0 * V)
    if np.any(np.abs(denominator) < 1.0e-14):
        raise DetonationError("Rayleigh-Hugoniot 关系的分母为零")
    return numerator / denominator


def rayleigh_hugoniot_velocity(volume_ratio, *, jwl: JWLParameters, volume_energy: float):
    """``D_RH(V)``：使 (V, p_H(V)) 落在 Rayleigh 线上的爆速。"""
    V = np.asarray(volume_ratio, dtype=np.float64)
    p = rayleigh_hugoniot_pressure(V, jwl=jwl, volume_energy=volume_energy)
    argument = p / (jwl.rho0 * (1.0 - V))
    return np.sqrt(np.maximum(argument, 0.0))


@dataclass(frozen=True)
class DetonationFrontState:
    """跨越爆轰波前之后**紧邻波后**的产物状态。

    ``sonic_ratio`` 是 ``c/(D-u) = c/(D V)``：小于 1 表示相对波前超声速，大于 1
    表示亚声速。平面弱激波的波后必为亚声速，这个比值就是选支判据。
    """

    volume_ratio: float
    density: float
    velocity: float
    specific_energy: float
    pressure: float
    sound_speed: float
    sonic_ratio: float
    branch: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "V_front": self.volume_ratio,
            "rho_front": self.density,
            "u_front": self.velocity,
            "e_front": self.specific_energy,
            "p_front": self.pressure,
            "c_front": self.sound_speed,
            "sonic_ratio_c_over_D_minus_u": self.sonic_ratio,
            "branch": self.branch,
        }


def _state_from_volume_ratio(V, *, detonation_velocity, jwl, volume_energy, branch):
    p = float(rayleigh_hugoniot_pressure(V, jwl=jwl, volume_energy=volume_energy))
    rho = jwl.rho0 / V
    e = (volume_energy + p * (1.0 - V) / 2.0) / jwl.rho0
    u = detonation_velocity * (1.0 - V)
    c = float(
        jwl_sound_speed_numpy(np.array([rho]), np.array([e]), jwl)[0]
    )
    sonic = c / (detonation_velocity - u) if detonation_velocity > u else math.inf
    return DetonationFrontState(
        volume_ratio=float(V),
        density=float(rho),
        velocity=float(u),
        specific_energy=float(e),
        pressure=float(p),
        sound_speed=float(c),
        sonic_ratio=float(sonic),
        branch=branch,
    )


def rayleigh_hugoniot_states(
    *,
    detonation_velocity: float,
    jwl: JWLParameters | None = None,
    volume_energy: float = mat.PRODUCT_VOLUME_ENERGY,
) -> tuple[DetonationFrontState, ...]:
    """给定爆速，求所有满足波前关系的 ``V``。

    ``D < D_CJ`` 时返回空元组——那是"这个 JWL/E0 组合撑不起这个爆速"的物理回答，
    不是数值失败。
    """
    params = jwl or mat.product_jwl()
    if detonation_velocity <= 0.0:
        raise DetonationError(f"爆速必须为正，收到 {detonation_velocity!r}")

    # D_RH(V) 在 V_CJ 取极小；先在细网格上定位所有与目标 D 的交点。
    lower, upper = _physical_volume_ratio_range(params)
    V = np.linspace(lower, upper, 20000)
    D_rh = rayleigh_hugoniot_velocity(V, jwl=params, volume_energy=volume_energy)
    difference = D_rh - detonation_velocity
    crossings = np.flatnonzero(np.diff(np.sign(difference)) != 0)
    roots: list[float] = []
    for index in crossings:
        lo, hi = float(V[index]), float(V[index + 1])

        def residual(value: float) -> float:
            return float(
                rayleigh_hugoniot_velocity(
                    np.array([value]), jwl=params, volume_energy=volume_energy
                )[0]
                - detonation_velocity
            )

        # 二分：稳健，不依赖 scipy。
        f_lo = residual(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if hi - lo < 1.0e-15:
                break
            f_mid = residual(mid)
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        roots.append(0.5 * (lo + hi))

    states = []
    for root in roots:
        branch = "subsonic" if _sonic_ratio(root, detonation_velocity, params, volume_energy) > 1.0 else "supersonic"
        states.append(
            _state_from_volume_ratio(
                root,
                detonation_velocity=detonation_velocity,
                jwl=params,
                volume_energy=volume_energy,
                branch=branch,
            )
        )
    return tuple(states)


def _sonic_ratio(V, detonation_velocity, jwl, volume_energy) -> float:
    p = float(rayleigh_hugoniot_pressure(np.array([V]), jwl=jwl, volume_energy=volume_energy)[0])
    rho = jwl.rho0 / V
    e = (volume_energy + p * (1.0 - V) / 2.0) / jwl.rho0
    c = float(jwl_sound_speed_numpy(np.array([rho]), np.array([e]), jwl)[0])
    return c / (detonation_velocity * V)


def detonation_front_state(
    *,
    detonation_velocity: float,
    jwl: JWLParameters | None = None,
    volume_energy: float = mat.PRODUCT_VOLUME_ENERGY,
) -> DetonationFrontState:
    """物理的波前状态：``D > D_CJ`` 时取亚声速那一支。

    选支不是约定，是激波理论的结果：波后流动相对波前必为亚声速，所以
    ``c/(D-u) > 1`` 的那一支才是可实现的。多于两支或没有亚声速支都直接失败——
    那说明模型或卡片参数出了问题，不该悄悄挑一个。
    """
    states = rayleigh_hugoniot_states(
        detonation_velocity=detonation_velocity, jwl=jwl, volume_energy=volume_energy
    )
    if not states:
        raise DetonationError(
            f"卡片 JWL/E0 在这个爆速下没有波前解：D = {detonation_velocity!r} m/s 低于"
            f"该材料组合的理想 CJ 爆速 {ideal_cj_state(jwl=jwl, volume_energy=volume_energy).detonation_velocity:.6g} m/s"
        )
    subsonic = [s for s in states if s.sonic_ratio > 1.0]
    if len(subsonic) != 1:
        raise DetonationError(
            f"期望恰好一个亚声速波前支，得到 {len(subsonic)} 个（共 {len(states)} 个根）；"
            "这说明波前关系或卡片参数不自洽，不能自动挑一个"
        )
    return subsonic[0]


# --------------------------------------------------------------------------
# 理想 CJ（自由爆速相切）——诊断与 §3 相容性核查用
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IdealCJState:
    """自由爆速下的理想 CJ 相切解。**不用于构造波前状态。**"""

    volume_ratio: float
    detonation_velocity: float
    pressure: float
    density: float
    velocity: float
    specific_energy: float
    sonic_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "V_CJ": self.volume_ratio,
            "D_CJ": self.detonation_velocity,
            "p_CJ": self.pressure,
            "rho_CJ": self.density,
            "u_CJ": self.velocity,
            "e_CJ": self.specific_energy,
            "sonic_ratio": self.sonic_ratio,
        }


def ideal_cj_state(
    *,
    jwl: JWLParameters | None = None,
    volume_energy: float = mat.PRODUCT_VOLUME_ENERGY,
) -> IdealCJState:
    """自由爆速相切解：``D_RH(V)`` 的极小点，等价于声速条件 ``c = D V``。

    CJ 的定义是 **Rayleigh 线与 JWL-Hugoniot 相切**，不是"两线相交"。相切与声速
    条件等价，这里直接在 ``dD_RH/dV = 0`` 上求根，并用声速比复核。
    """
    params = jwl or mat.product_jwl()
    lower, upper = _physical_volume_ratio_range(params)
    V = np.linspace(lower, upper, 40000)
    D_rh = rayleigh_hugoniot_velocity(V, jwl=params, volume_energy=volume_energy)
    index = int(np.argmin(D_rh))
    lo = float(V[max(index - 1, 0)])
    hi = float(V[min(index + 1, V.size - 1)])

    def value(at: float) -> float:
        return float(
            rayleigh_hugoniot_velocity(
                np.array([at]), jwl=params, volume_energy=volume_energy
            )[0]
        )

    # D_RH(V) 在 C V_CJ 附近是单谷的，三分搜索直接逼近极小点；
    # 相切条件 dD/dV = 0 与声速条件 c = D V 等价，最后用声速比复核。
    for _ in range(400):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        if value(left) < value(right):
            hi = right
        else:
            lo = left
        if hi - lo < 1.0e-15:
            break
    V_cj = 0.5 * (lo + hi)

    p = float(rayleigh_hugoniot_pressure(np.array([V_cj]), jwl=params, volume_energy=volume_energy)[0])
    D_cj = float(rayleigh_hugoniot_velocity(np.array([V_cj]), jwl=params, volume_energy=volume_energy)[0])
    rho = params.rho0 / V_cj
    e = (volume_energy + p * (1.0 - V_cj) / 2.0) / params.rho0
    c = float(jwl_sound_speed_numpy(np.array([rho]), np.array([e]), params)[0])
    return IdealCJState(
        volume_ratio=float(V_cj),
        detonation_velocity=D_cj,
        pressure=p,
        density=float(rho),
        velocity=D_cj * (1.0 - V_cj),
        specific_energy=float(e),
        sonic_ratio=c / (D_cj * V_cj),
    )


# --------------------------------------------------------------------------
# 产物区拉格朗日有限体积
# --------------------------------------------------------------------------


@dataclass
class DetonationConfig:
    """产物场求解参数。粘性系数与卡片 ``*CONTROL_BULK_VISCOSITY`` 一致。"""

    cells: int = 2000
    cfl: float = 0.2
    q1: float = 1.5
    q2: float = 0.06
    #: 逐步记录离散收支（§3.1/§3.2）。只用于诊断，不改变推进。
    record_budget: bool = False
    max_steps: int = 40_000_000

    def validate(self) -> None:
        if self.cells < 8:
            raise DetonationError(f"cells 至少为 8，收到 {self.cells}")
        if not (0.0 < self.cfl < 1.0):
            raise DetonationError(f"cfl 必须落在 (0, 1)，收到 {self.cfl}")
        if self.q1 < 0.0 or self.q2 < 0.0:
            raise DetonationError("人工粘性系数必须非负")


@dataclass
class DetonationProfile:
    """波前到达装药表面那一刻的产物场。"""

    time: float
    charge_radius: float
    detonation_velocity: float
    radii: np.ndarray            # 单元中心半径，m
    edges: np.ndarray            # 单元边界半径，(n_cells + 1,)，m
    cell_mass: np.ndarray        # 单元质量（单位立体角），kg/sr
    node_velocity: np.ndarray    # 节点速度，(n_cells + 1,)，m/s
    density: np.ndarray
    velocity: np.ndarray
    specific_energy: np.ndarray
    pressure: np.ndarray
    front_state: DetonationFrontState
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time,
            "charge_radius_m": self.charge_radius,
            "detonation_velocity_m_s": self.detonation_velocity,
            "n_cells": int(self.density.size),
            "edges_first_last": [float(self.edges[0]), float(self.edges[-1])],
            "density_min": float(self.density.min()),
            "density_max": float(self.density.max()),
            "pressure_min": float(self.pressure.min()),
            "pressure_max": float(self.pressure.max()),
            "max_speed": float(np.abs(self.velocity).max()),
            **self.front_state.as_dict(),
            **self.diagnostics,
        }


class ProductsDetonation:
    """中心起爆、有限爆速、产物区球对称拉格朗日有限体积。"""

    def __init__(
        self,
        *,
        charge_radius: float,
        detonation_velocity: float,
        config: DetonationConfig | None = None,
        jwl: JWLParameters | None = None,
        volume_energy: float = mat.PRODUCT_VOLUME_ENERGY,
        reference_density: float = mat.PRODUCT_REFERENCE_DENSITY,
    ) -> None:
        self.config = config or DetonationConfig()
        self.config.validate()
        self.charge_radius = float(charge_radius)
        self.detonation_velocity = float(detonation_velocity)
        self.jwl = jwl or mat.product_jwl()
        self.volume_energy = float(volume_energy)
        self.reference_density = float(reference_density)

        if self.charge_radius <= 0.0:
            raise DetonationError(f"装药半径必须为正，收到 {charge_radius!r}")

        self.front_state = detonation_front_state(
            detonation_velocity=self.detonation_velocity,
            jwl=self.jwl,
            volume_energy=self.volume_energy,
        )
        self.heat_release = self.volume_energy / self.reference_density
        #: 波前状态的比焓，``h_f = e_f + p_f/rho_f``。前端单元吸积进来的质量按
        #: 这个比焓进能量方程（标准的带流入控制体写法）。
        #:
        #: 曾经在这里放过一个 ``front_energy_flux`` 常数（域的总能量通量
        #: ``rho_ref D r_f^2 Q``），但推进里**从未使用**——注释说"采用精确通量"而
        #: 代码走的是另一条路。那种不一致比没有注释更坏：它让读代码的人以为自己
        #: 知道格式在做什么。要么用它，要么别写。
        self.front_enthalpy = (
            self.front_state.specific_energy
            + self.front_state.pressure / self.front_state.density
        )

        n = self.config.cells
        self.n_cells = n
        self.spacing = self.charge_radius / n
        self.radii_original = np.arange(n + 1.0) * self.spacing
        self.mass_original = (
            self.reference_density
            * (self.radii_original[1:] ** 3 - self.radii_original[:-1] ** 3)
            / 3.0
        )
        self.surface_time = self.charge_radius / self.detonation_velocity

        self._reset()

    # -- 状态初始化 -------------------------------------------------------
    def _reset(self) -> None:
        """从"波前刚扫过第一个单元的一部分"开始。

        从 ``t = 0`` 开始会让第一个单元的质量与体积同时为零（前端单元在 ``r_f = 0``
        处是退化的）。取一个非零的起始扫过分数既避开退化，又不改变物理。
        """
        n = self.n_cells
        self.radii = np.zeros(n + 2, dtype=np.float64)
        self.velocity = np.zeros(n + 2, dtype=np.float64)
        self.cell_mass = self.mass_original.copy()
        self.specific_volume = np.full(n + 1, 1.0 / self.reference_density)
        self.specific_energy = np.zeros(n + 1)

        start_fraction = 0.5
        self.time = start_fraction * self.spacing / self.detonation_velocity
        self.front_cell = 0
        self.radii[0] = 0.0
        self.radii[1] = self.detonation_velocity * self.time
        self.cell_mass[0] = (
            self.reference_density * self.radii[1] ** 3 / 3.0
        )
        self.specific_volume[0] = (
            (self.radii[1] ** 3 - self.radii[0] ** 3) / 3.0
        ) / self.cell_mass[0]
        self.specific_energy[0] = self.front_state.specific_energy
        self.velocity[0] = 0.0
        # 波前节点是**非物质的**：它的位置被给定在 D t，但那里的**物质**速度是
        # 波后速度 u_f。把 D 当成物质速度会凭空给最外层单元一个 6930 m/s 的动能，
        # 也会在人工粘性里造出假的压缩。
        self.velocity[1] = self.front_state.velocity
        self.steps = 0
        self.budget: list[dict[str, float]] = []

    # -- 状态方程 ---------------------------------------------------------
    def _pressure_of(self, density, specific_energy):
        return jwl_pressure_numpy(density, specific_energy, self.jwl)

    def _sound_speed_of(self, density, specific_energy):
        return jwl_sound_speed_numpy(density, specific_energy, self.jwl)

    @property
    def _n_active_cells(self) -> int:
        """包括正在被扫过的前端单元。"""
        return self.front_cell + 1

    def _cell_pressure(self) -> np.ndarray:
        """活动单元的压力。前端单元与内部单元一样由 EOS 给出。"""
        count = self._n_active_cells
        density = 1.0 / self.specific_volume[:count]
        return self._pressure_of(density, self.specific_energy[:count])

    # -- 时间步 -----------------------------------------------------------
    def _timestep(self) -> float:
        count = self._n_active_cells
        pressure = self._cell_pressure()
        sound = self._sound_speed_of(1.0 / self.specific_volume[:count], self.specific_energy[:count])
        width = np.abs(np.diff(self.radii[: count + 1]))
        # 刚跨过一个节点时，新的前端单元宽度为零——它的质量与体积都还没有。
        # 这是一个退化但不非法的状态。给它一个下限，使"刚点燃"的那一步不会一步
        # 长到单元尺度：前端单元的状态由吸积（入流）决定，不是由内部单元的波速
        # 决定，所以对内部单元成立的 CFL 判据在这里没有意义。
        width[-1] = max(float(width[-1]), 0.1 * self.spacing)
        speed = sound + np.abs(0.5 * (self.velocity[: count + 1][:-1] + self.velocity[: count + 1][1:]))
        with np.errstate(divide="ignore", invalid="ignore"):
            candidates = width / np.maximum(speed, 1.0e-30)
        candidates = candidates[np.isfinite(candidates)]
        dt = self.config.cfl * float(np.min(candidates)) if candidates.size else math.inf
        if not math.isfinite(dt) or dt <= 0.0:
            raise DetonationError(f"时间步非正：{dt!r}")
        return dt

    # -- 一步推进 ---------------------------------------------------------
    def _advance(self, dt: float) -> None:
        k = self.front_cell
        count = k + 1
        record = self.config.record_budget
        if record:
            # 逐步收支要在**改动任何状态之前**取快照。取晚了读到的就是更新后的
            # 状态，"前后之差"会恒等于零。
            snap = {
                "mass": self.cell_mass[:count].copy(),
                "energy": self.specific_energy[:count].copy(),
                "velocity": self.velocity[: count + 1].copy(),
                # 比容也必须快照：第 8 步会把它覆盖成新值，之后再算
                # ``v_new - v_old`` 就恒等于零——诊断自己会给出"内部单元没做功"
                # 这种一眼假的结果。
                "specific_volume": self.specific_volume[:count].copy(),
                # 节点半径同样要快照：收支里的**几何增量**（节点扫过的体积、
                # 单元体积变化）只能由前后两套半径算出来，而第 1 步就把半径改掉了。
                "radii": self.radii[: count + 1].copy(),
                "front_cell": k,
            }
        time_new = self.time + dt
        front_new = self.detonation_velocity * time_new

        # 波前到达节点是一个**离散事件**。主循环已经把 dt 夹到它，但
        # ``D * (t + dt)`` 与节点半径之间还差一次浮点舍入。判据若写成
        # ``front_new >= 节点 - 1e-18``，就会偶发地整次漏掉事件——而漏掉之后前端
        # 单元的质量永远长不满、下一个单元永远不激活。后果不是"差一个单元"，而是
        # 剖面出现与网格相关的台阶，看起来很像数值振荡。所以把事件吸附到节点上。
        crossing_at = self.radii_original[k + 1]
        event_snapped = False
        if k + 1 < self.n_cells and abs(front_new - crossing_at) <= 1.0e-9 * self.spacing:
            time_new = crossing_at / self.detonation_velocity
            dt = time_new - self.time
            front_new = crossing_at
            event_snapped = True

        # 1) 物质节点按流体速度运动；波前节点给定在 D t。
        self.radii[1 : k + 1] += dt * self.velocity[1 : k + 1]
        self.radii[k + 1] = front_new

        # 2) 前端单元吸积：新质量按波前状态进入。
        mass_new = (
            self.reference_density
            * (min(front_new, self.radii_original[k + 1]) ** 3 - self.radii_original[k] ** 3)
            / 3.0
        )
        mass_old = self.cell_mass[k]
        added = mass_new - mass_old
        if added < -1.0e-18 * max(mass_new, 1.0):
            raise DetonationError("前端单元的质量在减少；吸积逻辑有误")

        # 3) 几何：比容由当前几何与质量给出。
        volume = (self.radii[1 : count + 1] ** 3 - self.radii[:count] ** 3) / 3.0
        self.cell_mass[:count] = np.concatenate(
            [self.mass_original[:k], np.array([mass_new])]
        )
        specific_volume_new = np.empty(count)
        if k >= 1:
            specific_volume_new[:k] = volume[:k] / self.cell_mass[:k]
        # 刚跨过节点的那一步前端单元的质量可能仍为零（几何退化），此时保留它在
        # 跨节点时被赋的播种比容，等下一步有了体积再回到几何定义。
        if mass_new > 0.0:
            specific_volume_new[k] = volume[k] / mass_new
        else:
            specific_volume_new[k] = self.specific_volume[k]

        # 4) 压力与声速取旧内能（能量随后更新）
        density_old = 1.0 / self.specific_volume[:count]
        pressure = self._pressure_of(density_old, self.specific_energy[:count])
        sound = self._sound_speed_of(density_old, self.specific_energy[:count])

        # 5) 人工粘性
        q = np.zeros(count)
        du = np.diff(self.velocity[: count + 1])
        compression = np.minimum(du, 0.0)
        active = compression < -1.0e-8
        q[active] = (
            self.config.q1 * density_old[active] * sound[active] * np.abs(compression[active])
            + self.config.q2 * density_old[active] * compression[active] ** 2
        )

        # 6) 内能。
        #    内部单元：拉格朗日 de = -(p+q) dv。拉格朗日格式的这一条**精确**守恒
        #    内能+动能，所以内部单元不需要再补任何东西。
        #    前端单元放到 7b) 去算——它的更新要用到**更新后**的动能。
        internal = self.specific_energy[:count].copy()
        internal[:k] -= (pressure[:k] + q[:k]) * (
            specific_volume_new[:k] - self.specific_volume[:k]
        )

        # 7) 节点加速度 du/dt = -r^2 dp/dm（节点质量取两侧单元质量的一半）。
        node_mass = np.zeros(count + 1)
        if k >= 1:
            node_mass[1:k] = 0.5 * (self.cell_mass[: k - 1] + self.cell_mass[1:k])
        node_mass[k] = (
            0.5 * (self.cell_mass[k - 1] + self.cell_mass[k])
            if k >= 1
            else 0.5 * self.cell_mass[0]
        )
        node_mass[k + 1] = 0.5 * self.cell_mass[k]
        total_pressure = pressure + q
        jump = np.zeros(count + 1)
        jump[1:count] = total_pressure[1:] - total_pressure[:-1]
        acceleration = -(self.radii[: count + 1] ** 2) * jump / np.maximum(node_mass, 1.0e-300)
        acceleration[0] = 0.0
        acceleration[k + 1] = 0.0  # 波前节点位置给定，不积分
        self.velocity[: count + 1] += dt * acceleration
        self.velocity[0] = 0.0

        # 节点 k 的形式是 ``M_k^new Δu_k = -dt F_k``（不是守恒形式
        # ``M_k^old u_k^old`` 那一支）。**这是实测选出来的**，不是随手写的：
        #
        # 节点 k 的质量每步增长 ``ΔM_k = added/2``，所以 "M^new Δu = -dtF" 与守恒
        # 形式 "M^new u^new = M^old u^old - dtF" 相差一个莱布尼茨项 ``-u_k ΔM_k``
        # （Caramana 等 1998 §2.1.4 明确指出这一项"often neglected in practice"）。
        # 三种写法都实现过并实测（n=500/cfl=0.2，总能量不变量偏差）：
        #
        #   ``M^new Δu = -dtF``                       -> +4.09e-3   ← 现在用的
        #   ``M^new Δu = -dtF - u_k ΔM_k``（守恒）    -> +2.56e-1
        #   ``M^new Δu = -dtF + ΔM_k (u_f - u_k)``    -> +1.68e-2
        #
        # 保守形式反而差 60 倍，原因是**波前节点 k+1 的位置是给定的**：它不满足
        # 动量方程，多出来的那部分动量被它吸收了。在节点 k 上单独"改正"守恒性，
        # 等于把一份动量从 k 搬到 k+1，而 k+1 又不许动，于是能量账对不上。
        # 也就是说：**在前端节点被约束的情况下，节点 k 上"漏掉的"莱布尼茨项正是
        # 与那个约束配套的量**。要真正改这一项，必须同时改波前节点的处理，那是
        # 一次格式重写，不是补丁。
        self.velocity[k + 1] = self.front_state.velocity

        # 7b) 前端单元：带流入的控制体的**内能**形式
        #       d(m e) = dm h_f - (压力功)
        if mass_new > 0.0:
            # 带流入的控制体，**内能**形式
            #
            #     d(m e) = dm * h_f - W_k
            #
            # 入流比焓 ``h_f = e_f + p_f/rho_f`` 是标准的带流入写法。
            #
            # ``W_k`` 是**本单元几何体积变化**乘波前压力 p_f：
            #     W_k = p_f * (v_k^new m_k^new - v_k^old m_k^old)
            # 两面的连续推导给出的**分面**形式是
            #     W_face = p_f * (dm/rho_ref) - P_k * (节点 k 扫过的体积)
            # 两个形式都实现过并实测（n=500/cfl=0.2，总能量不变量偏差）：
            #     W_k    -> +4.1e-3
            #     W_face -> -4.4e-2（更差）
            # 差值恰好是 (p_f - P_k) x (节点 k 扫过的体积)，累计 +4.3e-2。
            #
            # **"更差"只排除分面形式这一种单点改动，不证明 W_k 是对的**：残差可能
            # 是多个误差互相抵消后的净值，改掉一个、抵消被打破，残差反而变大。
            # 一个候选解释是动量侧恒等式（``_step_budget`` 里验证到 1.7e-8 的
            # ``work_momentum = work_interior - work_front_inner``）已经含了
            # P_k x (节点 k 扫过的体积)，在这里再减一次就重复扣了——但这是候选，
            # 不是证明。所以保留实测更好的整项形式，同时**不宣称这一项已经闭合**。
            # 残差来源见模块文档"离散收支的定位结果"，结论是**尚未定论**。
            #
            # 这里**不**加人工粘性：前端单元一边被扫过一边在膨胀，q 只在压缩时非零，
            # 实测 q[k] 恒为 0（加与不加逐位相同）。留一个不生效的系数只会让人以为
            # 它在起作用。
            work = self.front_state.pressure * (
                specific_volume_new[k] * mass_new - self.specific_volume[k] * mass_old
            )
            internal[k] = (
                mass_old * self.specific_energy[k] + added * self.front_enthalpy - work
            ) / mass_new
            if internal[k] <= 0.0:
                raise DetonationError(
                    f"前端单元的比内能在 t = {time_new!r} 处变成 {internal[k]!r}；"
                    "吸积能量收支不再相容"
                )

        # 8) 接受新状态
        self.specific_volume[:count] = specific_volume_new
        self.specific_energy[:count] = internal
        self.time = time_new
        self.steps += 1

        if not np.all(np.isfinite(self.specific_volume[:count])) or not np.all(
            np.isfinite(self.specific_energy[:count])
        ):
            raise DetonationError("产物场出现非有限值")
        if np.any(self.specific_volume[:count] <= 0.0):
            raise DetonationError("产物场出现非正比容")

        if record:
            self.budget.append(
                self._step_budget(
                    dt=dt,
                    k=k,
                    snap=snap,
                    pressure=pressure,
                    q=q,
                    crossed=(k + 1 < self.n_cells and front_new >= crossing_at),
                    event_snapped=event_snapped,
                )
            )

        # 9) 前端单元跨过节点时，它变成普通拉格朗日单元，波前进入下一个单元。
        if k + 1 < self.n_cells and front_new >= crossing_at:
            self.front_cell = k + 1
            self.radii[k + 2] = front_new
            self.velocity[k + 2] = self.front_state.velocity
            # 刚离开波前的那个节点带上波后物质速度。
            self.velocity[k + 1] = self.front_state.velocity
            self.cell_mass[k + 1] = 0.0
            self.specific_volume[k + 1] = 1.0 / self.front_state.density
            self.specific_energy[k + 1] = self.front_state.specific_energy

    # -- 离散收支（§3.1 / §3.2） ------------------------------------------
    def _step_budget(self, *, dt, k, snap, pressure, q, crossed, event_snapped):
        """本步的离散收支各项——**事后审计**，不采信推进过程传进来的中间量。

        所有几何增量、质量变化和能量变化都由"本步前的快照"与"本步后的状态"
        重新算出。推进过程里算过的那些量（比焓入流、单元体积变化）**不参与**
        记账，否则诊断只会复述实现，发现不了实现本身的错。

        离散能量（单位立体角，与提示词 §3.1 的约定一致）：

        .. math::

            E_h = \\sum_i m_i e_i + \\frac{1}{2}\\sum_j M_j u_j^2,
            \\qquad M_j = \\frac{m_{j-1}+m_j}{2}

        物理输入只有一个：**本步被波前扫过的那部分装药释放的化学能**

        .. math::  \\Delta_{\\rm in} = \\Delta m \\cdot Q, \\qquad Q = E_0/\\rho_{\\rm ref}

        这一条不是"选出来让残差最小的数"，而是连续方程的直接结果。产物区
        ``[0, r_f]`` 的拉格朗日控制体满足（已数值复核到 1e-16）

        .. math::

            \\frac{dE}{dt} = \\rho_{\\rm ref} D r_f^2 (e_f + u_f^2/2) - p_f u_f r_f^2
                          = \\rho_{\\rm ref} D Q r_f^2

        也就是：波前坐标系里的总能流 ``rho D r_f^2 (h_f + u_f^2/2)`` 减掉运动边界上
        的 ``p_f u_f`` 做功之后，只剩化学能一项。**这个抵消关系是记账的关键**，
        少了 ``-p_f u_f r_f^2`` 会以为输入是 ``ṁ(h_f+u_f²/2)``，比真值大 2.25 倍。

        于是逐步残差

        .. math::  R = \\Delta E_h - \\Delta m\\, Q

        就是"这一步的离散格式有没有把化学能正确地变成内能+动能"的判据。
        """
        count = k + 1
        m_old, e_old, u_old = snap["mass"], snap["energy"], snap["velocity"]
        r_old = snap["radii"]
        m_new = self.cell_mass[:count].copy()
        e_new = self.specific_energy[:count].copy()
        u_new = self.velocity[: count + 1].copy()
        r_new = self.radii[: count + 1].copy()

        def node_ke(mass, velocity):
            return 0.5 * float(np.sum(_node_masses(mass) * velocity**2))

        added = float(m_new[k] - m_old[k])
        ke_old = node_ke(m_old, u_old)
        ke_new = node_ke(m_new, u_new)
        internal_old = float(np.sum(m_old * e_old))
        internal_new = float(np.sum(m_new * e_new))
        d_energy = (internal_new + ke_new) - (internal_old + ke_old)
        physical_input = added * self.heat_release

        # 几何增量：全部由前后两套节点半径算出，不用实现里的比容。
        def swept(radius_new, radius_old):
            return (radius_new**3 - radius_old**3) / 3.0

        node_sweep = swept(r_new[: count + 1], r_old[: count + 1])
        cell_volume_old = swept(r_old[1:], r_old[:-1])
        cell_volume_new = swept(r_new[1:], r_new[:-1])
        d_cell_volume = cell_volume_new - cell_volume_old

        total_pressure = pressure + q
        work_interior = (
            float(np.sum(total_pressure[:k] * d_cell_volume[:k])) if k >= 1 else 0.0
        )
        # 波前（外侧面）扫过的体积 = 新吸积的质量 / 参考密度，精确成立。
        front_volume_added = added / self.reference_density
        work_front_outer = self.front_state.pressure * front_volume_added
        # 节点 k（内侧面）扫过的体积。这一面在动量方程里用的压力是 P_k，所以功也
        # 必须用 P_k——用 p_f 就造成 §3.3 说的"力与功不匹配"。
        work_front_inner = float(total_pressure[k]) * float(node_sweep[k])
        force_work_mismatch = (
            self.front_state.pressure - float(total_pressure[k])
        ) * float(node_sweep[k])

        # 动能变化的三分解。恒等式 A + B + C = ΔKE（实测 1e-16）：
        du = u_new - u_old
        node_new = _node_masses(m_new)
        node_old_mass = _node_masses(m_old)
        node_new_mass = node_new
        work_momentum = float(np.sum(node_new * u_old * du))
        euler_error = 0.5 * float(np.sum(node_new * du**2))
        mass_partition = node_ke(m_new, u_old) - node_ke(m_old, u_old)

        return {
            "time": float(self.time),
            "dt": float(dt),
            "front_cell": int(k),
            "crossed": bool(crossed),
            "event_snapped": bool(event_snapped),
            "front_cell_created": bool(m_old[k] <= 0.0),
            "mass_added": added,
            "mass_total_before": float(m_old.sum()),
            "mass_total_after": float(m_new.sum()),
            "internal_before": internal_old,
            "internal_after": internal_new,
            "kinetic_before": ke_old,
            "kinetic_after": ke_new,
            # 唯一的物理输入
            "physical_input": physical_input,
            # 逐步残差：离散能量变化 - 物理输入
            "energy_residual": d_energy - physical_input,
            "work_interior": work_interior,
            "work_front_outer": work_front_outer,
            "work_front_inner": work_front_inner,
            "force_work_mismatch": force_work_mismatch,
            "node_k_sweep": float(node_sweep[k]),
            "front_volume_added": front_volume_added,
            # 动能的三分解（A + B + C = ΔKE）
            "work_momentum": work_momentum,
            "euler_error": euler_error,
            "mass_partition_kinetic": mass_partition,
            # 速度分片缺陷：把吸积质量当成跟着旧节点速度走时，动能比
            # "新物质以 u_f 进入"少掉的那一块。二阶量，网格加密时应按 h^2 消失。
            "velocity_lumping_deficit": (
                0.25 * added * (self.front_state.velocity - float(u_old[k])) ** 2
            ),
            "u_k_before": float(u_old[k]),
            "u_front": float(self.front_state.velocity),
            "p_front": float(self.front_state.pressure),
            "P_k": float(total_pressure[k]),
            # 节点 k 的动量账：M_k^new u_k^new - M_k^old u_k^old 应当**只**由压力
            # 冲量解释。若有人再加一个"新质量带着 u_f 进来"的来流动量项，这一条
            # 立刻不成立（差值会多出 0.5*dm*u_f）。
            "momentum_before": float(node_old_mass[k] * u_old[k]),
            "momentum_after": float(node_new_mass[k] * u_new[k]),
            "impulse": -dt
            * float(r_new[k]) ** 2
            * (
                float(total_pressure[k]) - float(total_pressure[k - 1])
                if k >= 1
                else 0.0
            ),
        }

    # -- 主循环 -----------------------------------------------------------
    def run_to_surface(self) -> DetonationProfile:
        """推进到 ``t = R_0/D``，即波前恰好到达装药表面的时刻。"""
        while self.time < self.surface_time - 1.0e-18:
            dt = self._timestep()
            next_crossing = (
                self.radii_original[self.front_cell + 1] / self.detonation_velocity - self.time
            )
            remaining = self.surface_time - self.time
            if next_crossing > 0.0:
                dt = min(dt, next_crossing)
            dt = min(dt, remaining)
            if dt <= 0.0:
                raise DetonationError(
                    f"时间步塌缩到 {dt!r}，t = {self.time!r}，前沿单元 = {self.front_cell}"
                )
            self._advance(dt)
            if self.steps > self.config.max_steps:
                raise DetonationError(
                    f"步数超过 max_steps = {self.config.max_steps}，尚未到达 t = {self.surface_time}"
                )
        return self.profile()

    # -- 输出 -------------------------------------------------------------
    def profile(self) -> DetonationProfile:
        count = self._n_active_cells
        centres = (
            0.5 * (self.radii[1 : count + 1] ** 3 + self.radii[:count] ** 3)
        ) ** (1.0 / 3.0)
        density = 1.0 / self.specific_volume[:count]
        velocity = 0.5 * (self.velocity[:count] + self.velocity[1 : count + 1])
        energy = self.specific_energy[:count].copy()
        pressure = self._pressure_of(density, energy)

        total_mass = float(self.cell_mass[:count].sum())
        exact_mass = self.reference_density * self.charge_radius**3 / 3.0
        # 动能有两种求和约定，而**格式守恒的是节点加权那一种**：
        #   节点约定 KE = sum_j M_j u_j^2/2,  M_j = 0.5(m_{j-1}+m_j)
        # （拉格朗日格式的动量方程就是作用在节点上的，它的能量恒等式配的是这个）
        #   单元约定 KE = sum_i m_i [(u_i+u_{i+1})/2]^2/2
        # 报告里两个都出，因为**用错约定去比对一个精确不变量，会读出一个根本不在
        # 格式里的偏差**。
        #
        # 两者之差有**闭式**，不是"线性速度下相等"：
        #
        #     K_node - K_cell = (1/8) * sum_i m_i (u_{i+1} - u_i)^2
        #
        # 推导：把 sum_j M_j u_j^2/2 按单元重排得 ¼ sum_i m_i (u_i^2 + u_{i+1}^2)，
        # 减去 (1/8) sum_i m_i (u_i+u_{i+1})^2，差恰好是上式。实测相对偏差 1.5e-13。
        #
        # **所以非零的速度梯度下两者通常不相等**，线性剖面也一样。曾经这里写的是
        # "两者在速度剖面线性且网格均匀时相等"——**错的**：线性剖面下
        # (1/8) sum m (du)^2 是个正常数，K_node 比 K_cell 大这个量。
        # 交接时用**同一种**约定比前后，只能说明交接没换口径，
        # **不能**说明两种口径彼此相等。
        node_mass = np.zeros(count + 1)
        node_mass[0] = 0.5 * self.cell_mass[0]
        node_mass[1:count] = 0.5 * (self.cell_mass[: count - 1] + self.cell_mass[1:count])
        node_mass[count] = 0.5 * self.cell_mass[count - 1]
        kinetic_node = 0.5 * float(np.sum(node_mass * self.velocity[: count + 1] ** 2))
        total_energy_node = float(np.sum(self.cell_mass[:count] * energy)) + kinetic_node
        total_energy = float(
            np.sum(
                self.cell_mass[:count]
                * (energy + 0.5 * velocity**2)
            )
        )
        exact_energy = self.volume_energy * self.charge_radius**3 / 3.0

        diagnostics = {
            "method": (
                "1D spherical Lagrangian finite volume for the products only; the "
                "detonation front is a non-material node pinned at r = D t, carrying "
                "the Rayleigh-Hugoniot state; the front cell accretes mass"
            ),
            "cells": self.n_cells,
            "cfl": self.config.cfl,
            "q1": self.config.q1,
            "q2": self.config.q2,
            "steps": self.steps,
            "front_cells_swept": self.front_cell,
            "mass_total": total_mass,
            "mass_exact": exact_mass,
            "mass_relative_error": abs(total_mass - exact_mass) / exact_mass,
            # **独立复核**：用带出去的单元边界重新积分 rho r^2 dr，而不是再用一遍
            # 内部的质量数组。两者若不一致，说明密度场与单元质量脱钩——而"质量
            # 相对偏差 = 0"只证明了后者。上游那里已经因此漏掉过一次。
            "mass_from_density": float(
                np.sum(density * (self.radii[1 : count + 1] ** 3 - self.radii[:count] ** 3) / 3.0)
            ),
            "energy_total": total_energy,
            "energy_exact": exact_energy,
            "energy_relative_error": abs(total_energy - exact_energy) / exact_energy,
            # **格式守恒的约定**。上面那个是单元中心约定，用它会读出一个不在格式里的偏差。
            "energy_total_node_weighted": total_energy_node,
            "energy_relative_error_node_weighted": abs(total_energy_node - exact_energy)
            / exact_energy,
            "energy_from_density": float(
                np.sum(
                    density
                    * (self.radii[1 : count + 1] ** 3 - self.radii[:count] ** 3)
                    / 3.0
                    * (energy + 0.5 * velocity**2)
                )
            ),
            # **两种动能口径都记**。格式守恒的是节点加权那一个
            # （``sum_j M_j u_j^2/2``）；单元中心约定（``sum_i m_i u_i^2/2``，
            # ``u_i`` 取单元中心速度）是场量的积分。两者差 ``O(du^2)``，而
            # "交接守不守恒"必须**在同一口径下**问，否则读出来的是口径差。
            "energy_node_weighted": float(
                np.sum(self.cell_mass[:count] * energy)
                + 0.5
                * float(
                    np.sum(
                        _node_masses(self.cell_mass[:count]) * self.velocity[: count + 1] ** 2
                    )
                )
            ),
            "unreacted_modelled": False,
            "reaction_zone_resolved": False,
        }
        return DetonationProfile(
            time=self.time,
            edges=self.radii[: count + 1].copy(),
            cell_mass=self.cell_mass[:count].copy(),
            node_velocity=self.velocity[: count + 1].copy(),
            charge_radius=self.charge_radius,
            detonation_velocity=self.detonation_velocity,
            radii=centres,
            density=density,
            velocity=velocity,
            specific_energy=energy,
            pressure=pressure,
            front_state=self.front_state,
            diagnostics=diagnostics,
        )


def detonation_energy_budget(
    *,
    charge_radius: float,
    detonation_velocity: float,
    config: DetonationConfig | None = None,
) -> dict[str, Any]:
    """跑一遍产物构造并返回**逐步离散收支**（§3.1/§3.2）。

    返回的每一项都以 ``E_exact = E0 R_0^3/3`` 归一。

    两个**恒等式**在逐步上精确成立（实测 1e-16 量级），它们是记账的骨架：

    * ``work_momentum + euler_error + mass_partition_kinetic == ΔKE``
      ——动能的功/欧拉/质量分配三分解；
    * ``internal_after - internal_before + ΔKE == energy_residual + physical_input``
      ——总能量变化的定义式。

    真正的判据是 ``energy_residual``：离散总能量的变化减去**物理输入**
    ``Δm Q``。它不依赖任何"让残差最小"的选择。残差越接近零，说明格式把化学能
    变成内能加动能的过程越正确。逐步残差按 ``front_cell`` / ``crossed`` /
    ``front_cell_created`` 分组，就能看出误差是在普通吸积步上累积，还是集中在
    跨节点事件的那一步。
    """
    # 不修改调用者传进来的配置。曾经这里是 ``settings.record_budget = True`` 直接改
    # 传进来的对象，调用者的配置会被悄悄改掉一个字段——副作用最难查的一种。
    settings = replace(config or DetonationConfig(), record_budget=True)
    solver = ProductsDetonation(
        charge_radius=charge_radius,
        detonation_velocity=detonation_velocity,
        config=settings,
    )
    profile = solver.run_to_surface()
    exact = mat.PRODUCT_VOLUME_ENERGY * charge_radius**3 / 3.0
    steps = solver.budget
    keys = (
        "physical_input",
        "energy_residual",
        "work_interior",
        "work_front_outer",
        "work_front_inner",
        "force_work_mismatch",
        "work_momentum",
        "euler_error",
        "mass_partition_kinetic",
        "velocity_lumping_deficit",
        "momentum_before",
        "momentum_after",
        "impulse",
    )
    cumulative = {k: float(np.sum([r[k] for r in steps])) / exact for k in keys}

    def total(key):
        return float(np.sum([r[key] for r in steps]))

    # 事件分组：逐步残差出现在哪一类步上。
    def group(predicate):
        rows = [r for r in steps if predicate(r)]
        return {
            "steps": len(rows),
            "energy_residual": float(np.sum([r["energy_residual"] for r in rows])) / exact,
        }

    return {
        "n_steps": len(steps),
        "energy_exact": exact,
        # **逐步记录**。诊断的价值主要在逐步上（哪一类步、哪个位置出错），
        # 只给累计值等于把定位能力丢掉。这里原样返回，调用者要画曲线、要按事件
        # 分组、要写文件都行。记录条数等于步数（本卡片 500 单元约 1.2 万条）。
        "steps": steps,
        # 逐步记录的字段名，方便调用者在不加载记录的情况下知道有哪些量。
        "step_keys": sorted(steps[0]) if steps else [],
        "cumulative": cumulative,
        # 恒等式 1：A + B + C = ΔKE
        "kinetic_identity_residual": (
            total("work_momentum")
            + total("euler_error")
            + total("mass_partition_kinetic")
            - (steps[-1]["kinetic_after"] - steps[0]["kinetic_before"])
        )
        / exact,
        "residual_by_event": {
            "ordinary": group(lambda r: not r["crossed"] and not r["front_cell_created"]),
            "crossing": group(lambda r: r["crossed"]),
            "cell_created": group(lambda r: r["front_cell_created"]),
            "event_snapped": group(lambda r: r["event_snapped"]),
            "front_cell_0": group(lambda r: r["front_cell"] == 0),
        },
        "final_invariant_relative_error": float(
            profile.diagnostics["energy_relative_error_node_weighted"]
        ),
    }


def detonation_product_profile(
    *,
    charge_radius: float,
    detonation_velocity: float,
    config: DetonationConfig | None = None,
    jwl: JWLParameters | None = None,
    volume_energy: float = mat.PRODUCT_VOLUME_ENERGY,
) -> DetonationProfile:
    """构造波前到达装药表面那一刻的产物场 ``rho, u, e, p``。"""
    solver = ProductsDetonation(
        charge_radius=charge_radius,
        detonation_velocity=detonation_velocity,
        config=config,
        jwl=jwl,
        volume_energy=volume_energy,
    )
    return solver.run_to_surface()


__all__ = [
    "DetonationConfig",
    "DetonationError",
    "DetonationFrontState",
    "DetonationProfile",
    "IdealCJState",
    "ProductsDetonation",
    "cold_pressure",
    "cold_pressure_derivative",
    "detonation_energy_budget",
    "detonation_front_state",
    "detonation_product_profile",
    "ideal_cj_state",
    "jwl_pressure_from_volume_ratio",
    "rayleigh_hugoniot_pressure",
    "rayleigh_hugoniot_states",
    "rayleigh_hugoniot_velocity",
]
