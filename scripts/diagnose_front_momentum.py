"""独立的径向动量收支诊断（只读，不改变 ``ProductsDetonation`` 的推进）。

它回答一个问题：**产物区 [0, r_f(t)] 的总离散径向动量，按球对称 Euler 的
积分动量方程，能不能用"边界输运 + 球面几何源项 + 人工粘性 + 变质量"解释清楚？**

控制域与变量
------------

单位立体角（与 ``physics/detonation.py`` 一致）。质量坐标下径向动量是

    P = ∫_0^{r_f} ρ u r² dr  ≈  Σ_j M_j u_j ,   M_j = (m_{j-1}+m_j)/2

对球对称 Euler 的**守恒形式**在移动控制域 [0, r_f(t)] 上积分（Leibniz），
保留球面几何源项：

    dP/dt = -r_f² [ ρ u (u - D) + p ]_{r_f} + ∫_0^{r_f} 2 r p dr

波前处用 RH：``ρ_f u_f (u_f - D) + p_f = 0``，所以**边界通量项在连续极限下为零**，
动量只能由球面几何源 ``∫2rp dr`` 与质量带入的动量产生。本诊断**不预设**这一项在
离散下也成立，而是把它单独量出来。

上游静止零压时还有 ``p_f = ρ0 D u_f``，因此被吸积质量带进来的动量
``Δm·u_f`` 与边界压力冲量 ``dt·r_f²·p_f`` **相互抵消**。所以"``p_f`` 没有显式
出现在节点更新里"本身**不能**当作缺力项的证据——要算的是这两项在离散里如何分摊。

每一步实际计算（全部由前后快照独立算出）
-----------------------------------------

令 ``count = k+1``（k 为当前前端单元）：

1. ``ΔP = Σ_j [M_j^{n+1} u_j^{n+1} - M_j^n u_j^n]``（节点 0..k+1）。
2. 压力/粘性冲量（就是节点动量方程的右端，用**推进后的半径**）：
   ``I = -dt Σ_{j=1}^{k} r_j² (P_j - P_{j-1})``，其中 ``P = p + q``；
   再按 ``p``、``q`` 拆成 ``I_p``、``I_q``。
3. 变质量/节点重分配：``A = Σ_j ΔM_j u_j^n``。
   **自检**：恒等式 ``ΔP = I_p + I_q + A`` 必须成立（否则诊断本身记账有误）。
4. 连续方程预测（独立定义，不含任何"总变化减其它项"的补偿）：
   ``pred = dt[ Σ_{i=0}^{k} p_i (r_{i+1}² - r_i²) - r_f² p_f ] + Δm u_f``
   其中 ``p_f, u_f`` 来自 RH 波前状态，``Δm`` 来自单元质量数组，``r`` 取推进后半径。
5. 尚未解释的残差：``R = (ΔP - I_q) - pred``。
   可以证明（见下）``R = dt·r_f²(p_f - p_k) + (Δm/2)(u_k^n - u_f)``；
   脚本同时独立计算右端，与 ``R`` 比较（``closed_form_check``）。

量纲与汇总口径
--------------

``geometric_source = Σ p_i (r_{i+1}² - r_i²)`` 是**力**，单位 **N/sr**
（全部量按单位立体角）。逐步把力相加没有物理意义；跨步累计的是冲量

    geometric_impulse = dt_actual · geometric_source      [N·s/sr]

``aggregate`` **只汇总冲量**（有符号累计、逐步绝对值累计）；
逐步的 ``geometric_source`` 仍写在 ``steps.csv`` 里，用于检查单步受力。
汇总中的未加权 ``sum(geometric_source)`` 已删除并标注为 deprecated。

边界时间积分的两种口径（上一轮的修正，保留）
--------------------------------------------

第 4 步里的边界压力冲量用的是**步末矩形**：

    J_old = dt · r_f(t2)² · p_f          ← 旧口径，保留，字段名不变

它把整个时间步的吸积动量 ``Δm·u_f`` 与一个只在 ``t2`` 取值的面积相乘，因此
"这两项精确抵消"这句话在离散层面**不成立**。本轮新增对恒定 ``D``、恒定 ``p_f``
的**解析时间积分**：

    J_exact = p_f · ∫_{t1}^{t2} r_f(τ)² dτ = p_f · (r_f2³ - r_f1³) / (3D)

因为 ``r_f(τ) = D τ`` 在一步内精确成立（包括跨节点事件吸附与末步夹取），
``r_f1, r_f2`` 取**实际步前/步后**波前位置。

上游静止零压时 RH 给出 ``p_f = ρ0 D u_f``，而前端单元吸积质量恰好是
``Δm = ρ0 (r_f2³ - r_f1³)/3``，所以解析上 ``J_exact ≡ Δm·u_f``。脚本仍
**分别**计算这两个量并报告差值 ``exact_minus_accreted``——**不**把后者赋给前者。

新旧预测只差边界项，因此

    pred_new - pred_old = J_old - J_exact
    R_new - R_old       = J_exact - J_old = -边界求积差额

脚本逐步核对这一恒等式（``residual_shift_check``）。

**重要**：``R_new`` 仍含**几何源项的求积误差**（``dt·Σ p_i Δ(r²)`` 是
``∫∫ 2rp dr dt`` 的一种离散），所以它**不是纯边界误差**；它只是把边界压力
冲量换成了与吸积动量同一时间区间的解析积分。

**不做**的事：不把 ``R`` 定义成"总变化减掉其它项"就宣布闭合。``R`` 有两部分：
一个是物理残差（离散值与连续预测之差），另一个是诊断自检（恒等式 2+3 是否精确）。
两者都写进输出。

全域几何源项的压力时间重构（本轮新增，B 口径）
----------------------------------------------

原来的 ``geometric_time_integral`` 只对几何因子 ``∫r²dt`` 做时间积分，压力冻结在
步前值（A 口径）。本轮增加 B 口径：始终对齐本步**原有活动单元 0..k**（跨节点时
排除新播种的单元 k+1），两端压力分别由步前/步后状态经**同一个 JWL EOS** 算出，
压力在步内线性，节点仍按本步已有的同一条线性轨迹运动，对乘积解析积分：

    J_G^B = Σ_i ∫_0^1 [p0_i + (p1_i-p0_i)s] [r_{i+1}(s)² - r_i(s)²] ds · dt

其中 ``r_j(s) = r0_j + s(r1_j-r0_j)``。A 口径是 ``p1_i = p0_i`` 的特例。
**B 只是基于数值端点的时间重构，不是连续真解；残差更小也不代表更正确。**

B 只替换 ``J_G``；``ΔP``、``I_q``、``J_exact``、``Δm·u_f`` 与 A 完全相同。因此

    R_B - R_A = -(J_G^B - J_G^A)

脚本逐步核对这个恒等式。另有两项退化核对：压力恒定时 B 退化为现有冻结压力的
几何时间积分；几何恒定时 B 退化为压力梯形积分。

自检与退出码
------------

``run_self_checks`` 只检查**记账关系**（离散恒等式、两个闭式分解、新旧残差
之差、解析冲量与吸积动量的一致性、波前运动、几何冲量分组可加性）。它**不**把
``residual`` / ``residual_exact_boundary`` 的大小当失败条件——物理残差非零正是
本诊断要报告的对象。

每个检查的容差是 ``atol + rtol * |scale|``，``scale`` 取该关系各参与项绝对值
之和（逐 step）；报告里逐项给出 ``max_abs_error``、``max_scale``、
``tolerance_at_max_scale``、``worst_error_over_tolerance``、单位与依据。
非有限值（NaN/Inf）**显式判失败**，不依赖 ``nan <= tol`` 的比较结果。

``summary.json`` 里的 ``self_checks_passed`` / ``failed_checks`` 是退出码的
唯一依据：全部通过时 ``main()`` 返回 0，否则**仍然写出完整报告**并返回 1。

JSON 序列化
-----------

写文件与打控制台之前都会先复制一份可序列化结构：NaN/±Inf 换成 ``null``，
并把"字段路径 + 原类型"记进 ``non_finite_fields``；原始 ``summary`` / ``rows``
与 ``self_checks_passed`` 不改。两处都使用 ``allow_nan=False``，漏网的非有限值
会直接抛错，而不是被写成非标准 JSON 常量。

用法::

    PYTHONPATH=<repo>/src python scripts/diagnose_front_momentum.py \
        --cells 400 --cfl 0.2 --output <仓库外目录>

输出与源码分开：``--output`` 下写 ``summary.json`` 与 ``steps.csv``。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spherical_blast_pinn.physics import material as mat  # noqa: E402
from spherical_blast_pinn.physics.detonation import (  # noqa: E402
    ProductsDetonation,
    DetonationConfig,
)
from spherical_blast_pinn.physics.eos import (  # noqa: E402
    jwl_pressure_numpy,
    jwl_sound_speed_numpy,
)


def node_masses(cell_mass: np.ndarray) -> np.ndarray:
    """``M_j = (m_{j-1}+m_j)/2``，两端各取相邻单元的一半。

    独立实现（不 import 私有 ``_node_masses``），公式与模块文档 §1 一致。
    """
    cell_mass = np.asarray(cell_mass, dtype=np.float64)
    node = np.zeros(cell_mass.size + 1, dtype=np.float64)
    node[0] = 0.5 * cell_mass[0]
    node[1:-1] = 0.5 * (cell_mass[:-1] + cell_mass[1:])
    node[-1] = 0.5 * cell_mass[-1]
    return node


def pressure_and_viscosity(
    solver: ProductsDetonation,
    specific_volume: np.ndarray,
    specific_energy: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按 ``_advance`` 第 4/5 步的同一公式，从**步前**状态重算 p 与 q。

    这样诊断用的是"解算器看到的那套旧状态"，而不是推进后回头猜的东西。
    """
    density = 1.0 / specific_volume
    pressure = jwl_pressure_numpy(density, specific_energy, solver.jwl)
    sound = jwl_sound_speed_numpy(density, specific_energy, solver.jwl)
    q = np.zeros_like(pressure)
    compression = np.minimum(np.diff(velocity), 0.0)
    active = compression < -1.0e-8
    q[active] = (
        solver.config.q1 * density[active] * sound[active] * np.abs(compression[active])
        + solver.config.q2 * density[active] * compression[active] ** 2
    )
    return pressure, q


def geometric_impulse(dt_actual: float, geometric_source: float) -> float:
    """把逐步的几何源项**力**乘上实际时长，得到冲量。

    ``geometric_source = Σ p_i Δ(r²)`` 的量纲是 **N/sr**（单位立体角下的力）。
    逐步把力相加没有物理意义；只有乘上 ``dt_actual`` 之后（**N·s/sr**）才能
    跨步累加。累计一律使用本函数，避免把 ``sum(G)`` 误当成冲量。
    """
    return float(dt_actual) * float(geometric_source)


def geometric_source_time_integral(
    radii_before: np.ndarray,
    radii_after: np.ndarray,
    pressure_before: np.ndarray,
    pressure_after: np.ndarray,
    dt_actual: float,
) -> float:
    """``∫∫ 2 r p dr dt`` 的离散时间积分：线性压力 × 线性节点轨迹。

    单元 ``i`` 贡献 ``∫ p_i(t) [r_{i+1}(t)² - r_i(t)²] dt``。节点按本步已有的
    线性运动 ``r_j(s) = r0_j + s(r1_j-r0_j)``（``s∈[0,1]``），压力在步前/步后
    端点值之间线性 ``p_i(s) = p0_i + s(p1_i-p0_i)``。两个积分都解析求出：

        ∫_0^1 r_j(s)² ds       = (r0² + r0 r1 + r1²)/3
        ∫_0^1 s r_j(s)² ds     = r0²/2 + 2 r0 Δr/3 + Δr²/4

    所以单元贡献是 ``dt·[p0_i·ΔM0_i + (p1_i-p0_i)·ΔM1_i]``。
    ``p1 = p0`` 时退化为冻结压力的几何时间积分；``r1 = r0`` 时退化为压力
    梯形积分。这只是端点数值重构，不是连续真解。
    """
    r0 = np.asarray(radii_before, dtype=np.float64)
    r1 = np.asarray(radii_after, dtype=np.float64)
    p0 = np.asarray(pressure_before, dtype=np.float64)
    p1 = np.asarray(pressure_after, dtype=np.float64)
    if not (r0.shape == r1.shape and p0.shape == p1.shape == (r0.size - 1,)):
        raise ValueError(
            "geometric_source_time_integral: 需要节点数组长 n+1、压力数组长 n"
        )
    delta_r = r1 - r0
    moment0 = (r0**2 + r0 * r1 + r1**2) / 3.0
    moment1 = 0.5 * r0**2 + (2.0 / 3.0) * r0 * delta_r + 0.25 * delta_r**2
    cell0 = float(dt_actual) * (moment0[1:] - moment0[:-1])
    cell1 = float(dt_actual) * (moment1[1:] - moment1[:-1])
    return float(np.sum(p0 * cell0 + (p1 - p0) * cell1))


def run_case(
    *,
    cells: int,
    cfl: float,
    charge_radius: float,
    detonation_velocity: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """逐行走完 ``run_to_surface`` 的循环，并在每步前后取快照。"""
    solver = ProductsDetonation(
        charge_radius=charge_radius,
        detonation_velocity=detonation_velocity,
        config=DetonationConfig(cells=cells, cfl=cfl),
    )
    front = solver.front_state
    D = float(detonation_velocity)
    rows: list[dict[str, Any]] = []

    while solver.time < solver.surface_time - 1.0e-18:
        # 与 ``run_to_surface`` 完全相同的步长夹取。
        dt_requested = solver._timestep()
        next_crossing = (
            solver.radii_original[solver.front_cell + 1] / D - solver.time
        )
        remaining = solver.surface_time - solver.time
        dt = dt_requested
        if next_crossing > 0.0:
            dt = min(dt, next_crossing)
        dt = min(dt, remaining)
        if dt <= 0.0:
            raise RuntimeError(f"时间步塌缩：dt={dt!r}, t={solver.time!r}")

        k = int(solver.front_cell)
        count = k + 1
        t_before = float(solver.time)
        m0 = solver.cell_mass[:count].copy()
        v0 = solver.specific_volume[:count].copy()
        e0 = solver.specific_energy[:count].copy()
        u0 = solver.velocity[: count + 1].copy()
        r0 = solver.radii[: count + 1].copy()
        mass_before = node_masses(m0)
        momentum_before = float(np.sum(mass_before * u0))

        # 解算器在这一步里使用的旧状态量
        p, q = pressure_and_viscosity(solver, v0, e0, u0)

        solver._advance(dt)

        t_after = float(solver.time)
        dt_actual = t_after - t_before
        k_after = int(solver.front_cell)
        crossed = k_after != k

        # ---- 步后状态：为与"旧节点集"对齐，取节点 0..k+1 ----
        m1 = solver.cell_mass[:count].copy()
        u1 = solver.velocity[: count + 1].copy()
        r1 = solver.radii[: count + 1].copy()
        # 单元 0..k 的步后热力学状态；跨节点时 k+1 是新播种单元，已被 count
        # 切片排除。两端压力都用同一个 JWL EOS，不借用推进过程里的数组。
        v1 = solver.specific_volume[:count].copy()
        e1 = solver.specific_energy[:count].copy()
        p_after = jwl_pressure_numpy(1.0 / v1, e1, solver.jwl)
        mass_after = node_masses(m1)
        momentum_after = float(np.sum(mass_after * u1))
        delta_momentum = momentum_after - momentum_before

        # ---- 1) 压力/粘性冲量（节点动量方程的右端）----
        force_p = np.zeros(count + 1)
        force_q = np.zeros(count + 1)
        force_p[1:count] = p[1:] - p[:-1]
        force_q[1:count] = q[1:] - q[:-1]
        weights = r1**2
        impulse_p = float(-dt_actual * np.sum(weights * force_p))
        impulse_q = float(-dt_actual * np.sum(weights * force_q))

        # ---- 2) 变质量/节点重分配 ----
        partition = float(np.sum((mass_after - mass_before) * u0))

        # ---- 3) 自检恒等式 ----
        identity_residual = delta_momentum - (impulse_p + impulse_q + partition)

        # ---- 4) 连续方程预测（独立定义）----
        r_f = float(r1[k + 1])
        r_f_before = float(r0[k + 1])
        p_front = float(front.pressure)
        u_front = float(front.velocity)
        # ∫2rp dr 的离散：整个 [0, r_f]，**包括前端单元 k**。
        # 几何源项求积方式**本轮不改**，新旧两个预测都用它。
        # 单位：**N/sr**（单位立体角下的力），不是冲量。
        geometric_source = float(np.sum(p[:count] * (r1[1 : count + 1] ** 2 - r1[:count] ** 2)))
        # 只有乘上实际时长才是可跨步累计的冲量（N·s/sr）。
        geometric_impulse_value = geometric_impulse(dt_actual, geometric_source)
        delta_mass = float(m1[k] - m0[k])

        # 旧求积口径（**保留作对照**）：边界压力冲量取步末矩形。
        boundary_impulse_old = dt_actual * r_f**2 * p_front
        accreted_momentum = delta_mass * u_front
        prediction = (
            dt_actual * (geometric_source - r_f**2 * p_front)
            + delta_mass * u_front
        )

        # ---- 4b) 边界压力冲量的解析时间积分（恒定 D、恒定 p_f）----
        # r_f(τ) = D·τ 在这一步内精确成立，所以
        #   ∫_{t1}^{t2} r_f² dτ = (r_f2³ - r_f1³) / (3D)。
        # r_f1、r_f2 取实际步前/步后波前位置，而不是 dt 和步末面积。
        boundary_impulse_exact = p_front * (r_f**3 - r_f_before**3) / (3.0 * D)
        boundary_quadrature_error = boundary_impulse_old - boundary_impulse_exact
        # 独立核对：解析冲量 与 吸积动量。**不**用后者给前者赋值。
        exact_minus_accreted = boundary_impulse_exact - accreted_momentum
        # 波前是否真的按 D·t 走（事件吸附/末步夹取下应为零）
        front_motion_check = (r_f - r_f_before) - D * dt_actual

        prediction_exact_boundary = (
            dt_actual * geometric_source
            - boundary_impulse_exact
            + accreted_momentum
        )

        # ---- 5) 未解释残差与其闭式分解 ----
        # 旧口径（保留）：R_old = (ΔP - I_q) - pred_old
        residual = (delta_momentum - impulse_q) - prediction
        residual_boundary = dt_actual * r_f**2 * (p_front - float(p[k]))
        residual_lumping = 0.5 * delta_mass * (float(u0[k]) - u_front)
        closed_form_check = residual - (residual_boundary + residual_lumping)

        # 新口径：只把边界压力冲量换成解析积分，几何源项求积原样保留。
        # R_new = (ΔP - I_q) - pred_new，且 pred_new ≈ dt·G，
        # 所以 R_new **仍含几何源项求积误差**，不是纯边界误差。
        residual_exact_boundary = (delta_momentum - impulse_q) - prediction_exact_boundary
        # 恒等式：R_new - R_old = J_exact - J_old = -boundary_quadrature_error
        residual_shift_check = (
            residual_exact_boundary - residual + boundary_quadrature_error
        )
        # 新残差的闭式：由 I_p = dt(G - r_f² p_k) 精确推出
        #   R_new = A - dt·r_f²·p_k + (J_exact - Δm·u_f)
        residual_exact_closed_form = (
            partition - dt_actual * r_f**2 * float(p[k]) + exact_minus_accreted
        )
        closed_form_check_exact = residual_exact_boundary - residual_exact_closed_form

        # ---- 4c) 几何源项的时间积分（**冻结步前压力**）----
        # 一步内节点位置是线性的（detonation.py::_advance 第 1 步）：
        #   r_j(t2) = r_j(t1) + u_j^old * dt   (j = 1..k)
        #   r_{k+1}(t) = D*t                    (波前节点，含事件吸附/末步夹取)
        # 对线性 r(τ)：∫_{t1}^{t2} r² dτ = dt*(a² + a*c + c²)/3，
        # 其中 a = r(t1)、c = r(t2)。这里只对**几何因子**做时间积分，
        # 压力仍冻结在步前值 p_i(t1)——不是完整连续解的精确源项。
        node_geom_integral = (r0**2 + r0 * r1 + r1**2) / 3.0
        cell_geom_integral = dt_actual * (
            node_geom_integral[1 : count + 1] - node_geom_integral[:count]
        )
        geometric_time_integral = float(np.sum(p[:count] * cell_geom_integral))
        # 与现有"步末几何 × 步前压力 × dt"的差额（N·s/sr）。
        geometric_quadrature_difference = (
            dt_actual * geometric_source - geometric_time_integral
        )
        prediction_geom_time = (
            geometric_time_integral
            - boundary_impulse_exact
            + accreted_momentum
        )
        residual_geom_time = (delta_momentum - impulse_q) - prediction_geom_time
        # 核对：R_geom - R_new = ΔJ_geom（逐 step 应为 0）。
        residual_geom_check = (
            residual_geom_time - residual_exact_boundary
            - geometric_quadrature_difference
        )

        # ---- 4d) 几何源项的时间积分（**压力在步前/步后 EOS 值之间线性重构**）----
        # 单元仍取原有活动单元 0..k（跨节点时新播种的 k+1 已被 count 排除），
        # 节点仍用同一条线性轨迹。B 只替换 J_G，ΔP、I_q、J_exact、Δm*u_f 与 A
        # 完全相同；所以 R_B - R_A 必须精确等于 -(J_G^B - J_G^A)。
        # 这是端点数值重构，不是连续真解；残差更小不代表更正确。
        geometric_time_integral_linear = geometric_source_time_integral(
            r0[: count + 1], r1[: count + 1], p[:count], p_after, dt_actual
        )
        geometric_time_difference = (
            geometric_time_integral_linear - geometric_time_integral
        )
        prediction_geom_time_linear = (
            geometric_time_integral_linear
            - boundary_impulse_exact
            + accreted_momentum
        )
        residual_geom_time_linear = (
            (delta_momentum - impulse_q) - prediction_geom_time_linear
        )
        residual_geom_time_change = (
            residual_geom_time_linear - residual_geom_time
        )
        # 独立核对：R_linear - R_frozen = -(J_G_linear - J_G_frozen)。
        residual_linear_check = (
            residual_geom_time_change + geometric_time_difference
        )
        # 核对物质节点确实按 r(t2) = r(t1) + u_old*dt 运动（j = 1..k）。
        if count > 1:
            node_motion_check = float(np.max(np.abs(
                (r1[1:count] - r0[1:count]) - u0[1:count] * dt_actual
            )))
        else:
            node_motion_check = 0.0

        rows.append({
            "step": int(solver.steps),
            "t_before": t_before,
            "t_after": t_after,
            "dt_requested": float(dt_requested),
            "dt_actual": dt_actual,
            "front_cell": k,
            "crossed": bool(crossed),
            "n_nodes": count + 1,
            "momentum_before": momentum_before,
            "momentum_after": momentum_after,
            "delta_momentum": delta_momentum,
            "impulse_pressure": impulse_p,
            "impulse_viscous": impulse_q,
            "mass_partition": partition,
            "identity_residual": identity_residual,
            # 逐步力（N/sr）与其乘 dt 后的冲量（N·s/sr）。汇总只用后者。
            "geometric_source": geometric_source,
            "geometric_impulse": geometric_impulse_value,
            "delta_mass": delta_mass,
            "continuous_prediction": prediction,
            "residual": residual,
            "residual_boundary_pressure": residual_boundary,
            "residual_velocity_lumping": residual_lumping,
            "closed_form_check": closed_form_check,
            # ---- 新增：边界时间积分的解析口径 ----
            "boundary_impulse_old": boundary_impulse_old,
            "boundary_impulse_exact": boundary_impulse_exact,
            "boundary_quadrature_error": boundary_quadrature_error,
            "accreted_momentum": accreted_momentum,
            "exact_minus_accreted": exact_minus_accreted,
            "front_motion_check": front_motion_check,
            "continuous_prediction_exact": prediction_exact_boundary,
            "residual_exact_boundary": residual_exact_boundary,
            "residual_shift_check": residual_shift_check,
            "residual_exact_closed_form": residual_exact_closed_form,
            "closed_form_check_exact": closed_form_check_exact,
            # ---- 本轮：几何源项时间积分（冻结步前压力）----
            "geometric_time_integral": geometric_time_integral,
            "geometric_quadrature_difference": geometric_quadrature_difference,
            "continuous_prediction_geom_time": prediction_geom_time,
            "residual_geom_time": residual_geom_time,
            "residual_geom_check": residual_geom_check,
            # ---- 本轮：压力线性重构（B 口径）----
            "geometric_time_integral_linear": geometric_time_integral_linear,
            "geometric_time_difference": geometric_time_difference,
            "continuous_prediction_geom_time_linear": prediction_geom_time_linear,
            "residual_geom_time_linear": residual_geom_time_linear,
            "residual_geom_time_change": residual_geom_time_change,
            "residual_linear_check": residual_linear_check,
            "node_motion_check": node_motion_check,
            "r_front": r_f,
            "r_front_before": r_f_before,
            "r_k": float(r1[k]),
            "p_k": float(p[k]),
            "p_front": float(front.pressure),
            "u_k_before": float(u0[k]),
            "u_front": float(front.velocity),
            "q_max_abs": float(np.max(np.abs(q))) if q.size else 0.0,
        })

        if solver.steps > solver.config.max_steps:
            raise RuntimeError("步数超过 max_steps")

    return {
        "cells": int(cells),
        "cfl": float(cfl),
        "charge_radius_m": float(charge_radius),
        "detonation_velocity_m_s": float(detonation_velocity),
        "reference_density_kg_m3": float(solver.reference_density),
        # 解析冲量 = 吸积动量 的前提是 p_f = ρ0 D u_f。RH 求根是数值的，
        # 这里把偏差量出来，避免把"前提"当成"恒等式"。
        "front_consistency": {
            "p_front": float(front.pressure),
            "u_front": float(front.velocity),
            "rho0_D_u_front": float(solver.reference_density * D * front.velocity),
            "p_front_minus_rho0_D_u_front": float(
                front.pressure - solver.reference_density * D * front.velocity
            ),
            "relative_gap": float(
                (front.pressure - solver.reference_density * D * front.velocity)
                / front.pressure
            ),
        },
        "front_state": {
            "volume_ratio": float(front.volume_ratio),
            "density": float(front.density),
            "velocity": float(front.velocity),
            "specific_energy": float(front.specific_energy),
            "pressure": float(front.pressure),
            "sound_speed": float(front.sound_speed),
            "sonic_ratio": float(front.sonic_ratio),
            "branch": front.branch,
        },
        "final_time_s": float(solver.time),
        "surface_time_s": float(solver.surface_time),
        "steps": int(solver.steps),
        "final_front_cell": int(solver.front_cell),
    }, rows


def aggregate(meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按"普通推进/跨节点事件"分组，累加**有符号**与**绝对**两种口径。

    **只汇总冲量，不汇总力。** ``geometric_source`` 是逐步的力（N/sr），
    ``sum(geometric_source)`` 没有物理意义，已从汇总键中移除；跨步累计的是
    ``geometric_impulse = dt_actual * geometric_source``（N·s/sr）。
    逐步的 ``geometric_source`` 仍保留在 ``steps.csv``，用于检查单步受力。
    """
    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"steps": 0}
        keys_signed = (
            "delta_momentum", "impulse_pressure", "impulse_viscous",
            "mass_partition", "identity_residual",
            "geometric_impulse",
            "delta_mass", "continuous_prediction", "residual",
            "residual_boundary_pressure", "residual_velocity_lumping",
            "closed_form_check",
            # 边界时间积分修正后的量
            "boundary_impulse_old", "boundary_impulse_exact",
            "boundary_quadrature_error", "accreted_momentum",
            "exact_minus_accreted", "front_motion_check",
            "continuous_prediction_exact", "residual_exact_boundary",
            "residual_shift_check", "residual_exact_closed_form",
            "closed_form_check_exact",
            # 上一轮：几何源项的时间积分（冻结步前压力，A 口径）
            "geometric_time_integral", "geometric_quadrature_difference",
            "continuous_prediction_geom_time", "residual_geom_time",
            "residual_geom_check",
            # 本轮：压力线性重构（B 口径）与两者对照
            "geometric_time_integral_linear", "geometric_time_difference",
            "continuous_prediction_geom_time_linear",
            "residual_geom_time_linear", "residual_geom_time_change",
            "residual_linear_check",
        )
        out = {"steps": len(subset)}
        for key in keys_signed:
            values = np.array([row[key] for row in subset], dtype=np.float64)
            out[key] = float(values.sum())
            out[key + "_abs"] = float(np.abs(values).sum())
            out[key + "_max_abs"] = float(np.max(np.abs(values)))
        out["crossed_steps"] = int(sum(1 for row in subset if row["crossed"]))
        out["dt_min"] = float(min(row["dt_actual"] for row in subset))
        out["dt_mean"] = float(np.mean([row["dt_actual"] for row in subset]))
        out["dt_max"] = float(max(row["dt_actual"] for row in subset))
        out["p_k_over_p_front_mean"] = float(np.mean([row["p_k"] / row["p_front"] for row in subset]))
        out["u_k_over_u_front_mean"] = float(np.mean([row["u_k_before"] / row["u_front"] for row in subset]))
        out["q_max_abs"] = float(max(row["q_max_abs"] for row in subset))
        return out

    ordinary = [row for row in rows if not row["crossed"]]
    crossing = [row for row in rows if row["crossed"]]
    total = block(rows)
    result = dict(meta)
    result.update({
        "momentum_initial": float(rows[0]["momentum_before"]) if rows else 0.0,
        "momentum_final": float(rows[-1]["momentum_after"]) if rows else 0.0,
        "momentum_change_total": (
            float(rows[-1]["momentum_after"] - rows[0]["momentum_before"]) if rows else 0.0
        ),
        "all_steps": total,
        "ordinary_steps": block(ordinary),
        "crossing_steps": block(crossing),
    })
    # 明确弃用：逐步 geometric_source 是力（N/sr），它的未加权求和不是冲量。
    # 汇总里只保留 geometric_impulse（N·s/sr）。
    result["geometric_source_aggregation"] = (
        "deprecated: per-step force (N/sr); aggregate only "
        "geometric_impulse = dt_actual * geometric_source (N*s/sr)"
    )
    # 分组可加性自检：普通步 + 跨节点步 应等于总计。
    result["geometric_impulse_group_sum_check"] = (
        result["ordinary_steps"].get("geometric_impulse", 0.0)
        + result["crossing_steps"].get("geometric_impulse", 0.0)
        - result["all_steps"].get("geometric_impulse", 0.0)
    )

    # 诊断自检的归一化强度：恒等式与闭式分解各自多大。
    scale = max(abs(result["momentum_final"]), 1.0)
    result["identity_max_abs_over_final_momentum"] = total.get("identity_residual_max_abs", 0.0) / scale
    result["closed_form_max_abs_over_final_momentum"] = total.get("closed_form_check_max_abs", 0.0) / scale
    # 新口径的三项自检
    result["residual_shift_max_abs"] = total.get("residual_shift_check_max_abs", 0.0)
    result["closed_form_exact_max_abs"] = total.get("closed_form_check_exact_max_abs", 0.0)
    result["exact_minus_accreted_abs_sum"] = total.get("exact_minus_accreted_abs", 0.0)
    result["exact_minus_accreted_max_abs"] = total.get("exact_minus_accreted_max_abs", 0.0)
    result["front_motion_max_abs"] = total.get("front_motion_check_max_abs", 0.0)
    # 本轮：几何源项时间积分的自洽核对与节点运动假设核对
    result["residual_geom_check_max_abs"] = total.get("residual_geom_check_max_abs", 0.0)
    # 本轮：B 口径与 A 口径的差，及其恒等式核对
    result["residual_linear_check_max_abs"] = total.get("residual_linear_check_max_abs", 0.0)
    result["geometric_time_difference_abs_sum"] = total.get("geometric_time_difference_abs", 0.0)
    result["node_motion_max_abs"] = (
        max(abs(float(row["node_motion_check"])) for row in rows) if rows else 0.0
    )
    return result


# --------------------------------------------------------------------------
# 自检：诊断本身是否可信（**不是**物理残差阈值）
# --------------------------------------------------------------------------
#
# 这里只检查记账关系是否在容差内成立，**不**检查 ``residual`` /
# ``residual_exact_boundary`` 的大小。物理残差非零正是本诊断要报告的对象；
# 把它当失败条件，会把"求解器有残差"误判成"诊断算错了"。
#
# 容差 = ``atol + rtol * |scale|``，``scale`` 是该关系各参与项绝对值之和
# （逐 step），因此容差随该项的量级缩放。依据：这些关系都是同一批浮点数之间
# 的恒等式或恒等式的重排，误差只来自舍入与求和顺序；实测相对误差在 1e-16
# 量级，取 ``rtol=1e-10`` 留约 6 个数量级余量，同时仍能抓住 O(1) 的记账错误。
# 动量类量的单位是 ``kg*m/s per sr``，几何量是 ``m``，二者**不共用**绝对容差。

def _relation_check(
    name: str,
    errors,
    scales,
    *,
    atol: float,
    rtol: float,
    unit: str,
    basis: str,
) -> dict[str, Any]:
    """检查 ``|error| <= atol + rtol*|scale|``，并显式处理非有限值。"""
    err = np.asarray(errors, dtype=np.float64)
    scl = np.asarray(scales, dtype=np.float64)
    non_finite = not (
        bool(np.all(np.isfinite(err))) and bool(np.all(np.isfinite(scl)))
    )
    record: dict[str, Any] = {
        "name": name,
        "unit": unit,
        "basis": basis,
        "atol": float(atol),
        "rtol": float(rtol),
        "samples": int(err.size),
        "non_finite": bool(non_finite),
        "passed": False,
        "max_abs_error": None,
        "max_scale": None,
        "tolerance_at_max_scale": None,
        "worst_error_over_tolerance": None,
    }
    if non_finite:
        # 必须显式失败，不能只靠 ``nan <= tol``：它在 Python 里求值为 False，
        # 表面上"失败"，但报告里没有原因；而任何 ``nan != nan`` 形式的写法都
        # 可能把故障藏起来。
        return record
    if err.size == 0:
        return record
    abs_err = np.abs(err)
    abs_scale = np.abs(scl)
    # 调用方保证 atol > 0，因此 tolerance > 0，不会除零。
    tolerance = atol + rtol * abs_scale
    record.update({
        "max_abs_error": float(np.max(abs_err)),
        "max_scale": float(np.max(abs_scale)),
        "tolerance_at_max_scale": float(atol + rtol * float(np.max(abs_scale))),
        "worst_error_over_tolerance": float(np.max(abs_err / tolerance)),
        "passed": bool(np.all(abs_err <= tolerance)),
    })
    return record


def degenerate_time_integral_checks() -> list[dict[str, Any]]:
    """线性重构公式的两个退化核对（用合成数组，不依赖真实算例）。

    1. 压力恒定（``p1 = p0``）：必须恢复**现有冻结压力**的几何时间积分。
    2. 几何恒定（``r1 = r0``）：必须恢复**压力梯形积分**
       ``dt·Σ (p0+p1)/2·(r_{i+1}²-r_i²)``。

    这两项检查的是积分公式本身。真实算例里 ``p`` 与 ``r`` 都在变，无法直接
    退化，所以用确定性的合成端点。它们**不**说明真实算例更正确。
    """
    r0 = np.array([0.0, 0.013, 0.031, 0.052, 0.079, 0.110], dtype=np.float64)
    r1 = np.array([0.0, 0.016, 0.037, 0.060, 0.090, 0.124], dtype=np.float64)
    p0 = np.array([3.0e10, 2.1e10, 1.2e10, 6.0e9, 2.0e9], dtype=np.float64)
    p1 = np.array([2.7e10, 1.8e10, 1.05e10, 5.2e9, 1.7e9], dtype=np.float64)
    dt = 2.0e-9

    # 现有 A 口径的几何时间积分（与 run_case 里的表达式相同）。
    moment0 = (r0**2 + r0 * r1 + r1**2) / 3.0
    frozen = float(np.sum(p0 * dt * (moment0[1:] - moment0[:-1])))
    # 压力恒定：B 公式输入 p1=p0。
    constant_pressure = geometric_source_time_integral(r0, r1, p0, p0, dt)
    # 几何恒定：B 公式输入 r1=r0，对照压力梯形积分。
    trapezoid = float(np.sum(
        dt * 0.5 * (p0 + p1) * (r0[1:] ** 2 - r0[:-1] ** 2)
    ))
    constant_geometry = geometric_source_time_integral(r0, r0, p0, p1, dt)

    return [
        _relation_check(
            "degenerate_constant_pressure",
            [constant_pressure - frozen],
            [abs(constant_pressure) + abs(frozen)],
            atol=1.0e-12, rtol=1.0e-12, unit="N*s/sr",
            basis="p1=p0 时，B 应退化为 A（冻结压力的几何时间积分）",
        ),
        _relation_check(
            "degenerate_constant_geometry",
            [constant_geometry - trapezoid],
            [abs(constant_geometry) + abs(trapezoid)],
            atol=1.0e-12, rtol=1.0e-12, unit="N*s/sr",
            basis="r1=r0 时，B 应退化为压力梯形积分",
        ),
    ]


def run_self_checks(
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """逐项检查记账关系；物理残差的大小**不**参与判定。"""
    if not rows:
        return [{
            "name": "rows_present", "unit": "1", "basis": "没有可检查的步",
            "atol": 0.0, "rtol": 0.0, "samples": 0, "non_finite": False,
            "passed": False, "max_abs_error": None, "max_scale": None,
            "tolerance_at_max_scale": None, "worst_error_over_tolerance": None,
        }]

    def col(key: str) -> np.ndarray:
        return np.array([float(row[key]) for row in rows], dtype=np.float64)

    dt = col("dt_actual")
    delta_momentum = col("delta_momentum")
    impulse_p = col("impulse_pressure")
    impulse_q = col("impulse_viscous")
    partition = col("mass_partition")

    residual_old = col("residual")
    residual_boundary = col("residual_boundary_pressure")
    residual_lumping = col("residual_velocity_lumping")

    residual_new = col("residual_exact_boundary")
    residual_geom_frozen = col("residual_geom_time")
    residual_geom_linear = col("residual_geom_time_linear")
    geometric_time_difference = col("geometric_time_difference")
    p_k = col("p_k")
    r_front = col("r_front")
    exact_minus_accreted = col("exact_minus_accreted")

    boundary_quadrature_error = col("boundary_quadrature_error")
    boundary_impulse_exact = col("boundary_impulse_exact")
    accreted_momentum = col("accreted_momentum")
    r_front_before = col("r_front_before")
    D = float(meta["detonation_velocity_m_s"])

    momentum_unit = "kg*m/s per sr"
    return [
        _relation_check(
            "accounting_identity",
            col("identity_residual"),
            np.abs(delta_momentum) + np.abs(impulse_p)
            + np.abs(impulse_q) + np.abs(partition),
            atol=1.0e-10, rtol=1.0e-10, unit=momentum_unit,
            basis="ΔP = I_p + I_q + A；尺度取四项绝对值之和（逐 step）",
        ),
        _relation_check(
            "closed_form_old",
            col("closed_form_check"),
            np.abs(residual_old) + np.abs(residual_boundary)
            + np.abs(residual_lumping),
            atol=1.0e-10, rtol=1.0e-10, unit=momentum_unit,
            basis="R_old = R_b + R_l；尺度取三项绝对值之和（逐 step）",
        ),
        _relation_check(
            "closed_form_new",
            col("closed_form_check_exact"),
            np.abs(residual_new) + np.abs(partition)
            + np.abs(dt * r_front**2 * p_k) + np.abs(exact_minus_accreted),
            atol=1.0e-10, rtol=1.0e-10, unit=momentum_unit,
            basis=(
                "R_new = A - dt*r_f^2*p_k + (J_exact - Δm*u_f)；"
                "尺度取四项绝对值之和（逐 step）"
            ),
        ),
        _relation_check(
            "residual_shift",
            col("residual_shift_check"),
            np.abs(residual_new) + np.abs(residual_old)
            + np.abs(boundary_quadrature_error),
            atol=1.0e-10, rtol=1.0e-10, unit=momentum_unit,
            basis="R_new - R_old = -ΔJ_quad；尺度取三项绝对值之和（逐 step）",
        ),
        _relation_check(
            "residual_linear_shift",
            col("residual_linear_check"),
            np.abs(residual_geom_linear) + np.abs(residual_geom_frozen)
            + np.abs(geometric_time_difference),
            atol=1.0e-10, rtol=1.0e-10, unit=momentum_unit,
            basis=(
                "R_linear - R_frozen = -(J_G_linear - J_G_frozen)；"
                "尺度取三项绝对值之和（逐 step）"
            ),
        ),
        _relation_check(
            "exact_impulse_vs_accreted",
            exact_minus_accreted,
            np.abs(boundary_impulse_exact) + np.abs(accreted_momentum),
            atol=1.0e-10, rtol=1.0e-10, unit=momentum_unit,
            basis=(
                "J_exact 与 Δm*u_f 分开计算，二者应一致；"
                "尺度取两项绝对值之和（逐 step）"
            ),
        ),
        _relation_check(
            "front_motion",
            col("front_motion_check"),
            np.abs(r_front) + np.abs(r_front_before) + np.abs(D * dt),
            atol=1.0e-14, rtol=1.0e-12, unit="m",
            basis="(r_f2 - r_f1) = D*dt；尺度取位置量之和（逐 step）",
        ),
        _relation_check(
            "geometric_impulse_group_sum",
            [float(summary["geometric_impulse_group_sum_check"])],
            [abs(float(summary["ordinary_steps"].get("geometric_impulse", 0.0)))
             + abs(float(summary["crossing_steps"].get("geometric_impulse", 0.0)))
             + abs(float(summary["all_steps"].get("geometric_impulse", 0.0)))],
            atol=1.0e-10, rtol=1.0e-10, unit=momentum_unit,
            basis="普通步 + 跨节点步 = 总计（对累计冲量）",
        ),
    ]


# --------------------------------------------------------------------------
# JSON 序列化：非有限值 -> null，但保留故障信息
# --------------------------------------------------------------------------

def _non_finite_label(value: float) -> str | None:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0.0 else "-Inf"
    return None


def json_safe_copy(
    value: Any,
    path: str = "",
    findings: list[dict[str, str]] | None = None,
) -> tuple[Any, list[dict[str, str]]]:
    """返回 ``(可 JSON 序列化的副本, 非有限字段清单)``。

    只把 NaN/±Inf 替换成 ``None``，并在清单里记录**原字段路径**与**原类型**
    （``NaN`` / ``+Inf`` / ``-Inf``）；其余值原样复制。**不修改输入对象**，
    也不把非有限值改成 0。``json.dumps(..., allow_nan=False)`` 会在最后一道
    关卡上拒绝任何漏网的非有限值。
    """
    if findings is None:
        findings = []
    if isinstance(value, dict):
        return {
            key: json_safe_copy(
                item, f"{path}.{key}" if path else str(key), findings
            )[0]
            for key, item in value.items()
        }, findings
    if isinstance(value, (list, tuple)):
        return [
            json_safe_copy(item, f"{path}[{index}]", findings)[0]
            for index, item in enumerate(value)
        ], findings
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return bool(value), findings
    if isinstance(value, (int, np.integer)):
        return int(value), findings
    if isinstance(value, (float, np.floating)):
        number = float(value)
        label = _non_finite_label(number)
        if label is not None:
            findings.append({"path": path, "value": label})
            return None, findings
        return number, findings
    return value, findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--charge-radius", type=float, default=None)
    parser.add_argument("--detonation-velocity", type=float, default=None)
    parser.add_argument("--output", required=True, help="仓库外的结果目录")
    args = parser.parse_args(argv)

    charge_radius = (
        float(args.charge_radius) if args.charge_radius is not None
        else float(getattr(mat, "CHARGE_RADIUS_REFERENCE", 0.05))
    )
    detonation_velocity = (
        float(args.detonation_velocity) if args.detonation_velocity is not None
        else float(mat.DETONATION_VELOCITY_REFERENCE)
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    meta, rows = run_case(
        cells=args.cells,
        cfl=args.cfl,
        charge_radius=charge_radius,
        detonation_velocity=detonation_velocity,
    )
    summary = aggregate(meta, rows)
    # 自检只看记账可信度；residual / residual_exact_boundary 的大小不参与判定。
    summary["self_checks"] = run_self_checks(meta, rows, summary)
    # 线性重构公式的两个退化核对（合成端点，与真实算例无关）。
    summary["self_checks"].extend(degenerate_time_integral_checks())
    summary["self_checks_passed"] = all(
        check["passed"] for check in summary["self_checks"]
    )
    summary["failed_checks"] = [
        check["name"] for check in summary["self_checks"] if not check["passed"]
    ]
    summary["elapsed_s"] = round(time.time() - started, 3)
    summary["script"] = str(Path(__file__).relative_to(REPO))
    summary["note"] = (
        "独立的径向动量收支诊断；只读调用 ProductsDetonation，不改推进公式。"
        "全部量按单位立体角。字段名不带后缀的 residual/continuous_prediction/"
        "closed_form_check 是**旧求积口径**（边界压力冲量取步末矩形 dt*r_f(t2)^2*p_f），"
        "保留作对照；residual_exact_boundary/continuous_prediction_exact 是"
        "**解析时间积分**口径（p_f*∫r_f^2 dt = p_f*(r_f2^3-r_f1^3)/(3D)）。"
        "新残差仍含几何源项求积误差，不是纯边界误差。"
        "geometric_time_integral 是 A 口径（压力冻结在步前值）；"
        "geometric_time_integral_linear 是 B 口径（压力在步前/步后 EOS 值之间线性，"
        "节点仍按同一条线性轨迹）；B 只是端点数值重构，不是连续真解。"
        "两种口径只替换 J_G，故 R_linear-R_frozen=-(J_G_linear-J_G_frozen)。"
        "self_checks 只判**记账关系是否可信**（恒等式/闭式/差值/冲量一致/波前运动/"
        "分组可加性）；residual 非零是诊断要报告的对象，不是失败条件。"
        "JSON 输出使用 allow_nan=False；非有限字段写成 null，并在 "
        "non_finite_fields 中记录原字段路径与原类型（NaN/+Inf/-Inf）。"
    )

    # 为 JSON 输出生成**单独**的可序列化副本；原始 summary/rows 不改。
    # 非有限值 -> null，并记录字段路径与原类型（NaN/+Inf/-Inf）。
    serializable, non_finite_fields = json_safe_copy(summary)
    serializable["non_finite_fields"] = non_finite_fields

    (output / "summary.json").write_text(
        json.dumps(
            serializable, indent=2, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8",
    )
    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 控制台摘要（也便于人工抄录）；同样先转成可序列化副本。
    a = summary["all_steps"]
    console_payload = {
        "cells": summary["cells"],
        "cfl": summary["cfl"],
        "steps": summary["steps"],
        "dt_mean_s": a.get("dt_mean"),
        "momentum_initial": summary["momentum_initial"],
        "momentum_final": summary["momentum_final"],
        "momentum_change_total": summary["momentum_change_total"],
        "delta_momentum_signed": a.get("delta_momentum"),
        "impulse_pressure_signed": a.get("impulse_pressure"),
        "impulse_viscous_signed": a.get("impulse_viscous"),
        "mass_partition_signed": a.get("mass_partition"),
        # 几何源项：逐步是力（N/sr），汇总只给乘 dt 后的冲量（N·s/sr）
        "geometric_impulse_signed": a.get("geometric_impulse"),
        "geometric_impulse_abs": a.get("geometric_impulse_abs"),
        "geometric_impulse_group_sum_check": summary["geometric_impulse_group_sum_check"],
        "continuous_prediction_signed": a.get("continuous_prediction"),
        "residual_signed": a.get("residual"),
        "residual_abs": a.get("residual_abs"),
        "residual_boundary_signed": a.get("residual_boundary_pressure"),
        "residual_lumping_signed": a.get("residual_velocity_lumping"),
        "identity_max_abs": a.get("identity_residual_max_abs"),
        "closed_form_max_abs": a.get("closed_form_check_max_abs"),
        # ---- 新口径 ----
        "boundary_impulse_old_signed": a.get("boundary_impulse_old"),
        "boundary_impulse_exact_signed": a.get("boundary_impulse_exact"),
        "boundary_quadrature_error_signed": a.get("boundary_quadrature_error"),
        "boundary_quadrature_error_abs": a.get("boundary_quadrature_error_abs"),
        "accreted_momentum_signed": a.get("accreted_momentum"),
        "exact_minus_accreted_abs_sum": summary["exact_minus_accreted_abs_sum"],
        "exact_minus_accreted_max_abs": summary["exact_minus_accreted_max_abs"],
        "front_motion_max_abs": summary["front_motion_max_abs"],
        "continuous_prediction_exact_signed": a.get("continuous_prediction_exact"),
        "residual_exact_boundary_signed": a.get("residual_exact_boundary"),
        "residual_exact_boundary_abs": a.get("residual_exact_boundary_abs"),
        "residual_shift_max_abs": summary["residual_shift_max_abs"],
        "closed_form_exact_max_abs": summary["closed_form_exact_max_abs"],
        # ---- 本轮：几何源项时间积分（冻结步前压力）----
        "geometric_time_integral_signed": a.get("geometric_time_integral"),
        "geometric_quadrature_difference_signed": a.get("geometric_quadrature_difference"),
        "geometric_quadrature_difference_abs": a.get("geometric_quadrature_difference_abs"),
        "continuous_prediction_geom_time_signed": a.get("continuous_prediction_geom_time"),
        "residual_geom_time_signed": a.get("residual_geom_time"),
        "residual_geom_time_abs": a.get("residual_geom_time_abs"),
        "residual_geom_check_max_abs": summary["residual_geom_check_max_abs"],
        "node_motion_max_abs": summary["node_motion_max_abs"],
        # ---- 本轮：两种 J_G 与两种 R（A 冻结 / B 压力线性重构）----
        "geometric_time_integral_frozen_signed": a.get("geometric_time_integral"),
        "geometric_time_integral_linear_signed": a.get("geometric_time_integral_linear"),
        "geometric_time_integral_difference_signed": a.get("geometric_time_difference"),
        "geometric_time_integral_difference_abs": a.get("geometric_time_difference_abs"),
        "residual_geom_time_frozen_signed": a.get("residual_geom_time"),
        "residual_geom_time_frozen_abs": a.get("residual_geom_time_abs"),
        "residual_geom_time_linear_signed": a.get("residual_geom_time_linear"),
        "residual_geom_time_linear_abs": a.get("residual_geom_time_linear_abs"),
        "residual_geom_time_change_signed": a.get("residual_geom_time_change"),
        "residual_geom_time_change_abs": a.get("residual_geom_time_change_abs"),
        "residual_linear_check_max_abs": summary.get("residual_linear_check_max_abs"),
        "geometric_time_difference_abs_sum": summary.get("geometric_time_difference_abs_sum"),
        "ordinary_geometric_time_integral_frozen": summary["ordinary_steps"].get("geometric_time_integral"),
        "ordinary_geometric_time_integral_linear": summary["ordinary_steps"].get("geometric_time_integral_linear"),
        "ordinary_residual_geom_time_frozen": summary["ordinary_steps"].get("residual_geom_time"),
        "ordinary_residual_geom_time_linear": summary["ordinary_steps"].get("residual_geom_time_linear"),
        "crossing_geometric_time_integral_frozen": summary["crossing_steps"].get("geometric_time_integral"),
        "crossing_geometric_time_integral_linear": summary["crossing_steps"].get("geometric_time_integral_linear"),
        "crossing_residual_geom_time_frozen": summary["crossing_steps"].get("residual_geom_time"),
        "crossing_residual_geom_time_linear": summary["crossing_steps"].get("residual_geom_time_linear"),
        "ordinary_steps": summary["ordinary_steps"].get("steps"),
        "crossing_steps": summary["crossing_steps"].get("steps"),
        "ordinary_residual_signed": summary["ordinary_steps"].get("residual"),
        "crossing_residual_signed": summary["crossing_steps"].get("residual"),
        "ordinary_residual_exact_boundary_signed": summary["ordinary_steps"].get("residual_exact_boundary"),
        "crossing_residual_exact_boundary_signed": summary["crossing_steps"].get("residual_exact_boundary"),
        # ---- 自检（记账可信度；不是物理残差阈值）----
        "self_checks_passed": summary["self_checks_passed"],
        "failed_checks": summary["failed_checks"],
        "self_check_worst_ratios": {
            check["name"]: check["worst_error_over_tolerance"]
            for check in summary["self_checks"]
        },
    }
    console_serializable, console_non_finite = json_safe_copy(console_payload)
    console_serializable["non_finite_fields"] = console_non_finite
    print(json.dumps(
        console_serializable, indent=2, ensure_ascii=False, allow_nan=False
    ))
    # 自检失败时**仍然**已经写出 summary.json / steps.csv，然后返回非零。
    return 0 if summary["self_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
