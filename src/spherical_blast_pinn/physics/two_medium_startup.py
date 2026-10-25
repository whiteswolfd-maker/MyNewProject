r"""短时两介质启动求解器：球对称拉格朗日 + 相容能量更新。

这个模块只做一件事：把
:class:`~spherical_blast_pinn.physics.two_medium_initial.TwoMediumInitialField`
给出的分段常数初值推进到一个物理时刻，输出同一时刻的完整两介质场与两个前沿，
供双网络传播阶段接收。它不训练、不写旧参考场缓存、不调用旧爆轰求解器。

数值方法
--------

一维球对称拉格朗日格式，交错布置：

* 节点 ``j = 0..N``：半径 ``r_j``、速度 ``u_j``；
* 单元 ``i = 0..N-1``：固定质量 ``m_i``、比容 ``v_i``、密度 ``rho_i``、
  比内能 ``e_i``、压力 ``p_i``、声速 ``c_i``、固定材料标签。

材料界面永远是**一个节点**：产物单元与空气单元各自保持自己的热力学状态，
节点只有一个速度。没有混合单元，因此 EOS 分支不会用错。

相容能量更新
------------

本模块不采用「先按旧速度移动节点、再按几何体积变化更新内能」的写法。那种写法
在 ``u = 0`` 的静止压力跳变初值上，第一步内能不变而节点速度已经变成 ``dt·a``，
即凭空产生 ``dt²`` 量级的动能，且之后每一步都会留下不配对的功。

这里采用力—功配对（compatible energy）的形式：内能更新使用的面通量，与动量
方程使用的是**同一组**压力、同一组人工粘性和同一组面积，并且使用时间中心速度

.. math::

    \bar u_j = \tfrac12\left(u_j^{n-1/2} + u_j^{n+1/2}\right),

.. math::

    e_i^{n+1} = e_i^n
      - \frac{\Delta t}{m_i}\,(p_i + q_i)
        \left(A_{i+1}\bar u_{i+1} - A_i \bar u_i\right),
    \qquad A_j = r_j^2 .

动量方程是

.. math::

    M_j\left(u_j^{n+1/2} - u_j^{n-1/2}\right)
      = -\Delta t\,A_j\left[(p+q)_j - (p+q)_{j-1}\right],

在 ``j = 0`` 处取对称条件 ``A_0 = 0, a_0 = 0``，在 ``j = N`` 处取环境压力。
两式相加（分部求和）给出**逐步恒等式**

.. math::

    \sum_i m_i\left(e_i^{n+1} - e_i^n\right)
      = -\left(K^{n+1/2} - K^{n-1/2}\right)
        - \Delta t\,p_{\mathrm{amb}}\,A_N \bar u_N ,

其中 ``K = ½ Σ_j M_j u_j²``，``M_j = ½(m_{j-1} + m_j)``。最后一项就是**实际边界功**：
外边界对系统做的功率乘时间。于是 ``Σ m e + K + W_boundary`` 在每一步都保持不变
（舍入误差以内），第一步也不会出现「内能不变、动能增加」。

方法出处与核实程度
------------------

力—功配对与边界功的标准表述见 Caramana, Burton, Shashkov & Whalen (1998),
*The Construction of Compatible Hydrodynamics Algorithms Utilizing Conservation of
Total Energy*, J. Comput. Phys. 146, 227–262
(https://cnls.lanl.gov/~shashkov/papers/comp.pdf)。

**本环境无法抓取该文正文**（网络策略阻断了所有外部 PDF），因此上面的离散关系是
从守恒恒等式逐步推导并**由测试逐步核对**的，不是对论文 §4.1/§4.2 的逐条复述。
项目没有宣称「已按原文实现」；论文正文的核对仍是未完成项。

时间层与精度
------------

**空间交错、时间不交错。** 节点存速度、单元存热力学量是空间交错；但 ``r`` 与
``u`` 都取自同一整数时刻，**没有半时间层速度**，也不存在 ``time - dt/2`` 的标记。
所有量都在整数时刻 ``t^n = state.time``。一步是

.. math::

    a^n = a\left(r^n, p^n
ight),\qquad
    u^{n+1} = u^n + \Delta t\, a^n,\qquad
    r^{n+1} = r^n + \Delta t\,	frac12\left(u^n + u^{n+1}
ight).

* 位置更新是梯形公式：**恒加速度下精确**（见测试）；
* 但力、压力、几何面积 ``A = (r^n)^2`` 和能量功里的 ``(p+q)^n`` 都取**旧时刻**
  ``t^n``，没有预测—校正；速度更新只有一阶，加速度随时间变化时 ``u^{n+1}`` 有
  ``O(Δt²)`` 误差，能量功同样带旧时刻压力的一阶误差；
* 因此整个格式**不叫二阶 Verlet**：只有位置推进用了新旧速度平均；
* 可变步长：每一步用各自的 ``Δt``，最后一步截断到恰好 ``t_h``；梯形公式对
  任意 ``Δt`` 都成立（恒加速度下连乘两步也精确，见测试）；
* 几何体积增量 ``V(r^{n+1}) - V(r^n)`` 与功中体积增量
  ``dt·(A_{i+1}ū_{i+1} - A_iū_i)`` 只在 ``O(dt)`` 内一致，二者单独报告；
  能量恒等式成立**不能**代替这项检查。

与旧求解器的关系
----------------

本模块不导入 ``physics/detonation.py``、``ProductDetonation`` 或
``SphericalPrecursor``，也不复用它们的推进循环：旧循环正是「先移节点、再按几何
体积更新内能」，与本模块的相容能量配对不同。网格、状态字段与 EOS 调用方式保持
一致，便于下游替换。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Sequence

import numpy as np

from . import material as mat
from .eos import (
    JWLParameters,
    jwl_pressure_numpy,
    jwl_sound_speed_squared_numpy,
)
from .two_medium_initial import AIR, PRODUCT, TwoMediumInitialField

#: 方法版本，进入缓存标识。改了离散格式就必须改它。
#: 方法版本。v2 相对 v1 的离散变化：位置改用梯形更新（与能量功同一个新旧速度
#: 平均）、速度按节点重构、激波按局部窗口提取。旧版本产出的验证证据不再适用。
METHOD_VERSION = "compatible_lagrangian_spherical_v2"

#: 启动链路自己的模型标识，与旧参考场构造字符串不同。
STARTUP_MODEL_ID = "uniform_product_sphere_air_startup_v1"

#: 逐步能量恒等式的相对漂移上限。它是**代数守恒**的检查，不是物理精度指标：
#: 合法步长下实测约 1e-15；1e-10 只用来抓住「某处忘了把边界功计入」这类错误。
_ENERGY_DRIFT_TOL = 1.0e-10

#: 激波定位时判为「未扰动」的相对压力余量。环境区的舍入级起伏（~1e-8）
#: 不是激波尾部；0.01% 只用来排除这类噪声，不参与激波判据本身。
_SHOCK_EXCESS_TOL = 1.0e-4

#: 激波展宽层判定：向内每单元相对压力增量小于它即认为已离开展宽层。
_RAMP_TOL = 0.05

#: 展宽层搜索窗口：从最外侧扰动单元向内最多这么多个单元。
_MAX_RAMP_CELLS = 12

#: 激波距外边界少于这么多个单元时，判为无法可靠提取。
_SHOCK_BOUNDARY_CELLS = 2


#: 三条 Rankine–Hugoniot 归一化残差的上限（仅用于「诊断是否有效」）。
_RH_REL_TOL = 0.05

#: 轨迹速度与 RH 速度的相对偏差上限（仅用于「诊断是否有效」）。
_FRONT_SPEED_REL_TOL = 0.05

#: 交接资格：能量恒等式漂移上限（代数守恒）。
_QUALITY_ENERGY_DRIFT = 1.0e-10

#: 交接资格：几何体积增量与功中体积增量的最大相对差上限（步长离散误差）。
_QUALITY_VOLUME_MISMATCH = 1.0e-2

#: 交接资格：接触面与激波之间至少要有这么多当地单元。
_QUALITY_AIR_GAP_CELLS = 8.0

#: 外部验证阈值。它们只作用于**随证据传入的数值**，不接受布尔值。
_VALIDATION_THRESHOLDS = {
    "grid_front_relative": 0.01,
    "grid_field_relative": 0.02,
    "timestep_front_relative": 0.01,
    "timestep_field_relative": 0.02,
    "contact_pressure_relative": 0.01,
    "contact_velocity_relative": 0.01,
}


def _evidence_number(
    section: dict,
    key: str,
    where: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """证据项必须是有限实数；``True``/``"ok"`` 之类的值一律拒绝。"""
    if key not in section:
        raise StartupError(f"{where}: missing {key!r}")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StartupError(
            f"{where}.{key} must be a number, got {type(value).__name__}: {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise StartupError(f"{where}.{key} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise StartupError(f"{where}.{key} must be >= {minimum}, got {value!r}")
    if maximum is not None and number > maximum:
        raise StartupError(f"{where}.{key} must be <= {maximum}, got {value!r}")
    return number


def _evidence_integer(section: dict, key: str, where: str) -> int:
    """单元数必须是正整数；不允许 ``int()`` 把 800.7 静默截断成 800。"""
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StartupError(
            f"{where}.{key} must be a number, got {type(value).__name__}: {value!r}"
        )
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise StartupError(f"{where}.{key} must be an integer, got {value!r}")
    integer = int(number)
    if integer < 1:
        raise StartupError(f"{where}.{key} must be positive, got {value!r}")
    return integer


def _binding_mismatches(expected: dict, actual: Any, where: str) -> list[str]:
    if not isinstance(actual, dict):
        return [f"evidence.{where} must be a mapping"]
    reasons: list[str] = []
    for key, want in expected.items():
        if key not in actual:
            reasons.append(f"evidence.{where}.{key} is missing")
            continue
        got = actual[key]
        if isinstance(want, float):
            if isinstance(got, bool) or not isinstance(got, (int, float)):
                reasons.append(f"evidence.{where}.{key} must be a number, got {got!r}")
                continue
            if not math.isfinite(float(got)):
                reasons.append(f"evidence.{where}.{key} must be finite, got {got!r}")
                continue
            if abs(float(got) - want) > 1.0e-12 * max(abs(want), 1.0):
                reasons.append(
                    f"evidence.{where}.{key} mismatch: {got!r} != {want!r}"
                )
        elif got != want:
            reasons.append(f"evidence.{where}.{key} mismatch: {got!r} != {want!r}")
    return reasons


def assess_validation_evidence(
    evidence: Any,
    *,
    binding: dict[str, Any],
) -> dict[str, Any]:
    """核对**外部**收敛/交接证据是否绑定到这次具体运行。

    绑定项：方法版本、初值、EOS、空气参数、人工粘性、交接半径/时间。比较组必须
    包含待交接运行，并且网格对照与时间步对照之间有明确对应关系。误差必须是
    有限非负数值，单元数必须是正整数。任何一项不满足都返回 ``accepted=False``。
    """
    if evidence is None:
        return {
            "provided": False,
            "accepted": False,
            "reasons": ["no external validation evidence"],
            "evidence": None,
        }
    if not isinstance(evidence, dict):
        return {
            "provided": True,
            "accepted": False,
            "reasons": ["evidence must be a mapping"],
            "evidence": evidence,
        }

    reasons: list[str] = []
    if evidence.get("method_version") != binding["method_version"]:
        reasons.append(
            f"method_version mismatch: {evidence.get('method_version')!r} "
            f"!= {binding['method_version']!r}"
        )
    for key in ("charge_radius", "handoff_time_s"):
        try:
            got = _evidence_number(evidence, key, "evidence", minimum=0.0)
        except StartupError as exc:
            reasons.append(str(exc))
            continue
        want = float(binding["charge_radius"] if key == "charge_radius" else binding["handoff_time"])
        tolerance = 1.0e-12 if key == "charge_radius" else 1.0e-9
        if abs(got - want) > tolerance * max(abs(want), 1.0e-30):
            reasons.append(f"evidence.{key} mismatch: {got!r} != {want!r}")

    for section in ("initial_state", "eos", "artificial_viscosity"):
        reasons.extend(_binding_mismatches(binding[section], evidence.get(section), section))

    source = evidence.get("source")
    if not isinstance(source, str) or not source.strip():
        reasons.append("evidence.source must be a non-empty string")

    runs = evidence.get("runs")
    parsed: dict[str, dict[str, float | int]] = {}
    if not isinstance(runs, list) or len(runs) < 2:
        reasons.append("evidence.runs must be a list with at least two runs")
        runs = []
    for index, run in enumerate(runs):
        where = f"runs[{index}]"
        if not isinstance(run, dict):
            reasons.append(f"{where} must be a mapping")
            continue
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id.strip():
            reasons.append(f"{where}.id must be a non-empty string")
            continue
        if run_id in parsed:
            reasons.append(f"duplicate run id {run_id!r}")
        try:
            cells = _evidence_integer(run, "cells_inside", where)
            growth = _evidence_number(run, "outside_growth", where, minimum=0.0)
            outer = _evidence_number(run, "outer_radius_factor", where, minimum=0.0)
            cfl = _evidence_number(run, "cfl", where, minimum=0.0, maximum=1.0)
            radius = _evidence_number(run, "charge_radius", where, minimum=0.0)
            time = _evidence_number(run, "handoff_time_s", where, minimum=0.0)
        except StartupError as exc:
            reasons.append(str(exc))
            continue
        if growth <= 1.0:
            reasons.append(f"{where}.outside_growth must exceed 1, got {growth!r}")
        if outer <= 1.0:
            reasons.append(f"{where}.outer_radius_factor must exceed 1, got {outer!r}")
        if not (0.0 < cfl < 1.0):
            reasons.append(f"{where}.cfl must lie in (0, 1), got {cfl!r}")
        if abs(radius - binding["charge_radius"]) > 1.0e-12 * max(
            abs(binding["charge_radius"]), 1.0
        ):
            reasons.append(f"{where}.charge_radius mismatch: {radius!r}")
        if abs(time - binding["handoff_time"]) > 1.0e-9 * max(
            abs(binding["handoff_time"]), 1.0e-30
        ):
            reasons.append(f"{where}.handoff_time_s mismatch: {time!r}")
        parsed[run_id] = {
            "cells_inside": cells,
            "outside_growth": growth,
            "outer_radius_factor": outer,
            "cfl": cfl,
        }

    handoff_run = evidence.get("handoff_run")
    if handoff_run not in parsed:
        reasons.append("evidence.handoff_run must name one of runs")
        handoff_cfl = float(binding["handoff_config"]["cfl"])
    else:
        handoff_cfl = float(parsed[handoff_run]["cfl"])
        for key in ("cells_inside", "outside_growth", "outer_radius_factor", "cfl"):
            want = binding["handoff_config"][key]
            got = parsed[handoff_run][key]
            if isinstance(want, int):
                if got != want:
                    reasons.append(
                        f"handoff_run {key} mismatch: {got!r} != {want!r}"
                    )
            elif abs(float(got) - float(want)) > 1.0e-12 * max(abs(float(want)), 1.0):
                reasons.append(f"handoff_run {key} mismatch: {got!r} != {want!r}")

    grid = evidence.get("grid_convergence")
    grid_coarse_id = grid_fine_id = None
    if not isinstance(grid, dict):
        reasons.append("evidence.grid_convergence must be a mapping")
    else:
        grid_coarse_id = grid.get("coarse_run")
        grid_fine_id = grid.get("fine_run")
        if grid_coarse_id not in parsed or grid_fine_id not in parsed:
            reasons.append("grid_convergence must name two known runs")
        else:
            coarse = parsed[grid_coarse_id]
            fine = parsed[grid_fine_id]
            if fine["cells_inside"] <= coarse["cells_inside"]:
                reasons.append("grid_convergence.fine run must use more cells")
            if coarse["cfl"] != handoff_cfl or fine["cfl"] != handoff_cfl:
                reasons.append(
                    "grid_convergence must compare runs at the handoff CFL"
                )
            if handoff_run not in (grid_coarse_id, grid_fine_id):
                reasons.append(
                    "grid_convergence must include the handoff run"
                )
        for key in ("front_relative", "field_relative"):
            try:
                value = _evidence_number(grid, key, "grid_convergence", minimum=0.0)
            except StartupError as exc:
                reasons.append(str(exc))
                continue
            limit = _VALIDATION_THRESHOLDS["grid_" + key]
            if value > limit:
                reasons.append(
                    f"grid_convergence.{key} {value:.3e} > {limit:.1e}"
                )

    timestep = evidence.get("timestep_sensitivity")
    if not isinstance(timestep, dict):
        reasons.append("evidence.timestep_sensitivity must be a mapping")
    else:
        coarse_id = timestep.get("coarse_run")
        fine_id = timestep.get("fine_run")
        if coarse_id not in parsed or fine_id not in parsed:
            reasons.append("timestep_sensitivity must name two known runs")
        else:
            coarse = parsed[coarse_id]
            fine = parsed[fine_id]
            if coarse_id != grid_fine_id:
                reasons.append(
                    "timestep_sensitivity.coarse_run must be the grid fine run"
                )
            if fine["cells_inside"] != coarse["cells_inside"]:
                reasons.append(
                    "timestep_sensitivity must compare runs on the same grid"
                )
            if coarse["cfl"] != handoff_cfl:
                reasons.append(
                    "timestep_sensitivity.coarse_run must use the handoff CFL"
                )
            if not fine["cfl"] < coarse["cfl"]:
                reasons.append(
                    "timestep_sensitivity.fine run must use a smaller CFL"
                )
        for key in ("front_relative", "field_relative"):
            try:
                value = _evidence_number(
                    timestep, key, "timestep_sensitivity", minimum=0.0
                )
            except StartupError as exc:
                reasons.append(str(exc))
                continue
            limit = _VALIDATION_THRESHOLDS["timestep_" + key]
            if value > limit:
                reasons.append(
                    f"timestep_sensitivity.{key} {value:.3e} > {limit:.1e}"
                )

    contact = evidence.get("contact_consistency")
    if not isinstance(contact, dict):
        reasons.append("evidence.contact_consistency must be a mapping")
    else:
        for key in ("pressure_relative", "velocity_relative"):
            try:
                value = _evidence_number(
                    contact, key, "contact_consistency", minimum=0.0
                )
            except StartupError as exc:
                reasons.append(str(exc))
                continue
            limit = _VALIDATION_THRESHOLDS["contact_" + key]
            if value > limit:
                reasons.append(
                    f"contact_consistency.{key} {value:.3e} > {limit:.1e}"
                )

    return {
        "provided": True,
        "accepted": not reasons,
        "reasons": reasons,
        "evidence": evidence,
        "thresholds": dict(_VALIDATION_THRESHOLDS),
    }



def shock_rh_residuals(
    *,
    post_density: float,
    post_velocity: float,
    post_pressure: float,
    post_specific_energy: float,
    ambient_density: float,
    ambient_pressure: float,
    ambient_specific_energy: float,
    shock_speed: float,
) -> dict[str, float]:
    """三条 Rankine–Hugoniot 的原始残差与**量纲一致**的归一化残差。

    归一化尺度分别取该项本身在波前的量级：

    ========  ===========================  ====================
    方程      原始残差量纲                 归一化尺度
    ========  ===========================  ====================
    质量      kg m^-2 s^-1                 ``rho_0 D``
    动量      Pa                           ``rho_0 D^2``
    能量      J kg^-1                      ``D^2``
    ========  ===========================  ====================

    只看原始残差无法判断「1e5 是大还是小」；只看相对残差又会掩盖尺度错误。
    两个都返回。
    """
    values = (
        post_density,
        post_velocity,
        post_pressure,
        post_specific_energy,
        ambient_density,
        ambient_pressure,
        ambient_specific_energy,
        shock_speed,
    )
    if not all(math.isfinite(v) for v in values):
        return {
            "shock_rh_mass": float("nan"),
            "shock_rh_momentum": float("nan"),
            "shock_rh_energy": float("nan"),
            "shock_rh_mass_rel": float("nan"),
            "shock_rh_momentum_rel": float("nan"),
            "shock_rh_energy_rel": float("nan"),
        }
    relative = shock_speed - post_velocity
    mass = ambient_density * shock_speed - post_density * relative
    momentum = (
        ambient_pressure
        + ambient_density * shock_speed**2
        - post_pressure
        - post_density * relative**2
    )
    energy = (
        ambient_specific_energy
        + ambient_pressure / ambient_density
        + 0.5 * shock_speed**2
        - post_specific_energy
        - post_pressure / post_density
        - 0.5 * relative**2
    )
    mass_scale = max(abs(ambient_density * shock_speed), 1.0e-300)
    momentum_scale = max(abs(ambient_density * shock_speed**2), 1.0e-300)
    energy_scale = max(abs(shock_speed**2), 1.0e-300)
    return {
        "shock_rh_mass": float(mass),
        "shock_rh_momentum": float(momentum),
        "shock_rh_energy": float(energy),
        "shock_rh_mass_rel": float(abs(mass) / mass_scale),
        "shock_rh_momentum_rel": float(abs(momentum) / momentum_scale),
        "shock_rh_energy_rel": float(abs(energy) / energy_scale),
    }


class StartupError(RuntimeError):
    """启动求解器的输入、状态或数值过程不合法。"""


def integrate_positions(
    radii: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """一步的时间积分：显式速度 + 梯形位置。

    更新式：

        u^{n+1} = u^n + dt * a^n

        r^{n+1} = r^n + dt * (u^n + u^{n+1}) / 2

    恒加速度下这对公式对任意 dt 都给出精确的抛物线解；但 a^n 是旧时刻的
    加速度，没有预测—校正，所以**不能**据此把整个格式称为二阶。
    """
    velocity_new = velocity + dt * acceleration
    radii_new = radii + dt * 0.5 * (velocity + velocity_new)
    return radii_new, velocity_new

@dataclass(frozen=True)
class ShockLocation:
    """一次局部激波提取的结果。``valid=False`` 时只有 ``reason`` 有意义。"""

    valid: bool
    reason: str
    radius: float
    front_cell: int
    shock_cell: int
    plateau_cell: int


def locate_shock(
    centres: np.ndarray,
    pressure: np.ndarray,
    air_mask: np.ndarray,
    ambient_pressure: float,
    *,
    excess_tol: float = _SHOCK_EXCESS_TOL,
    ramp_tol: float = _RAMP_TOL,
    max_ramp_cells: int = _MAX_RAMP_CELLS,
    boundary_cells: int = _SHOCK_BOUNDARY_CELLS,
    plateau_tol: float = 0.01,
) -> ShockLocation:
    """从压力剖面局部提取激波位置与波后平台。

    规则与依据
    ----------
    1. **前沿**：空气区最外侧一个压力高于 ``p_amb (1 + excess_tol)`` 的单元。
    2. **局部峰值**：在 ``[前沿-12, 前沿]`` 窗口里取压力最大值；峰值落在窗口
       边缘时判为不可靠（真正的峰值可能更里面）。
    3. **平台**：从前沿向内第一个达到局部峰值 99% 的单元。展宽层很薄时它与
       峰值单元重合；展宽层较厚时它给出紧邻激波的波后状态。
    4. **阈值**：``target = (p_reference + p_amb)/2``，只依赖**局部**峰值。
       以前用空气区全局最大压力，接触面附近的高压峰值一变，阈值就跟着变，
       激波位置会被错误移动——这里不再使用全局最大值。
    5. **位置**：``target`` 在平台与前沿之间的单元中心之间线性插值。

    适用范围
    --------
    完整展宽层 ``[窗口内边缘, 前沿]`` 必须分两段单调：内边缘到峰值向外不减，
    峰值到前沿向外不增（均允许 ``ramp_tol`` 以内的起伏）。不适用于多个激波、
    激波与外边界相交、或展宽层内出现超过 ``ramp_tol`` 的数值过冲/凹陷——这些
    情况返回 ``valid=False`` 并给出原因，不静默给一个位置。

    失败条件
    --------
    空气区太少、没有扰动单元、激波贴外边界、局部峰值在窗口边缘、平台落在
    空气区内边缘、展宽层非单调、半跳变落在边界外或括号单元之外。
    """
    air = np.flatnonzero(air_mask)
    if air.size < 4:
        return ShockLocation(False, "too few air cells", float("nan"), -1, -1, -1)
    disturbed = np.flatnonzero(pressure[air] > ambient_pressure * (1.0 + excess_tol))
    if disturbed.size == 0:
        return ShockLocation(False, "no disturbed air cell", float("nan"), -1, -1, -1)
    front = int(air[disturbed[-1]])
    if front + boundary_cells >= int(air[-1]):
        return ShockLocation(
            False, "shock too close to the outer boundary", float("nan"), front, -1, -1
        )
    # 局部参考压力：只在这个窗口里找峰值，不看空气区全局最大值。
    window_low = int(max(air[0], front - max_ramp_cells))
    if window_low >= front:
        return ShockLocation(
            False, "search window truncated", float("nan"), front, -1, -1
        )
    window = np.arange(window_low, front + 1)
    reference = float(np.max(pressure[window]))
    # 取**最外侧**达到最大值的单元：平台可以一直延伸向内，紧邻展宽层的那一格
    # 才是峰值；用 argmax 的第一次出现会把平台内边缘误当成峰值。
    peak = int(window[np.flatnonzero(pressure[window] >= reference)[-1]])
    peak_offset = front - peak
    if peak_offset >= max_ramp_cells:
        # 峰值落在窗口边缘：真正的峰值可能在更里面，这个窗口给不出可靠参考。
        return ShockLocation(
            False, "local peak at the search-window edge", float("nan"), front, -1, -1
        )
    if reference <= ambient_pressure * (1.0 + excess_tol):
        return ShockLocation(False, "peak not above ambient", float("nan"), front, -1, -1)

    # 平台：从前沿向内第一个达到局部峰值 99% 的单元。它紧邻展宽层，而不是
    # 窗口里压力最高的那一格本身——两者在展宽层很薄时重合。
    above_reference = np.flatnonzero(pressure[window] >= reference * (1.0 - plateau_tol))
    plateau = int(window[above_reference[-1]])
    if plateau <= int(air[0]):
        return ShockLocation(
            False,
            "plateau at the inner edge of the air region",
            float("nan"),
            front,
            -1,
            plateau,
        )
    # 展宽层分成两段检查，合起来覆盖 [窗口内边缘, 前沿] 整段：
    #   1) 内边缘 -> 峰值：向外应单调不减（凹陷/过冲会破坏它）；
    #   2) 峰值 -> 前沿：向外应单调不增（峰值之后的回升同样是不可靠剖面）。
    approach = pressure[window_low : peak + 1]
    if np.any(approach[1:] < approach[:-1] * (1.0 - ramp_tol)):
        return ShockLocation(
            False, "non-monotone shock ramp", float("nan"), front, -1, plateau
        )
    outer_ramp = pressure[peak : front + 1]
    if np.any(outer_ramp[1:] > outer_ramp[:-1] * (1.0 + ramp_tol)):
        return ShockLocation(
            False, "non-monotone shock ramp", float("nan"), front, -1, plateau
        )

    target = 0.5 * (reference + ambient_pressure)
    above = air[pressure[air] >= target]
    if above.size == 0:  # pragma: no cover - target <= p_plateau
        return ShockLocation(
            False, "half-jump target not crossed", float("nan"), front, -1, plateau
        )
    index = int(above[-1])
    if index >= int(air[-1]) or not air_mask[index + 1]:
        return ShockLocation(
            False, "half-jump at the outer boundary", float("nan"), front, index, plateau
        )
    r_left, r_right = float(centres[index]), float(centres[index + 1])
    p_left, p_right = float(pressure[index]), float(pressure[index + 1])
    if p_left == p_right:  # pragma: no cover
        return ShockLocation(
            False, "degenerate half-jump", float("nan"), front, index, plateau
        )
    weight = (p_left - target) / (p_left - p_right)
    if not (0.0 <= weight <= 1.0):
        return ShockLocation(
            False,
            "half-jump outside the bracketing cells",
            float("nan"),
            front,
            index,
            plateau,
        )
    return ShockLocation(
        True, "ok", r_left + weight * (r_right - r_left), front, index, plateau
    )


# --------------------------------------------------------------------------
# 配置与网格
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StartupConfig:
    """启动推进的离散参数。

    默认值与旧前置求解器一致，便于对照；但本模块的推进格式不同。
    """

    #: 装药内均匀单元数；它同时决定 ``R_0`` 处的单元宽度 ``R_0 / cells_inside``。
    cells_inside: int = 2000
    #: 空气区单元宽度的固定增长比。
    outside_growth: float = 1.005
    #: 外边界半径 / ``R_0``。
    outer_radius_factor: float = 40.0
    #: CFL 数。
    cfl: float = 0.2
    #: von Neumann–Richtmyer 人工粘性系数。
    q1: float = 1.5
    q2: float = 0.06
    #: 判为零压缩的速度差阈值，m/s。
    compression_floor: float = 1.0e-8
    #: 步数保护。
    max_steps: int = 2_000_000

    def validate(self) -> None:
        if self.cells_inside < 8:
            raise StartupError(f"cells_inside must be >= 8, got {self.cells_inside}")
        if not math.isfinite(self.outside_growth) or self.outside_growth <= 1.0:
            raise StartupError(
                f"outside_growth must be finite and > 1, got {self.outside_growth!r}"
            )
        if not math.isfinite(self.outer_radius_factor) or self.outer_radius_factor <= 1.0:
            raise StartupError(
                f"outer_radius_factor must be finite and > 1, got {self.outer_radius_factor!r}"
            )
        if not (0.0 < self.cfl < 1.0):
            raise StartupError(f"cfl must lie in (0, 1), got {self.cfl!r}")
        if self.q1 < 0.0 or self.q2 < 0.0:
            raise StartupError("artificial viscosity coefficients must be non-negative")
        if self.max_steps < 1:
            raise StartupError(f"max_steps must be positive, got {self.max_steps}")


@dataclass
class StartupMesh:
    """球对称拉格朗日网格：节点半径、单元质量、材料标签。"""

    radii: np.ndarray
    mass: np.ndarray
    material: np.ndarray
    charge_radius: float

    def __post_init__(self) -> None:
        if self.radii.size != self.mass.size + 1:
            raise StartupError("radii must have one more entry than mass")
        if self.material.size != self.mass.size:
            raise StartupError("material must have one entry per cell")
        if np.any(np.diff(self.radii) <= 0.0):
            raise StartupError("node radii must be strictly increasing")
        if np.any(self.mass <= 0.0):
            raise StartupError("cell masses must be positive")

    @property
    def n_cells(self) -> int:
        return int(self.mass.size)

    @property
    def centres(self) -> np.ndarray:
        return (0.5 * (self.radii[1:] ** 3 + self.radii[:-1] ** 3)) ** (1.0 / 3.0)

    @property
    def n_inside(self) -> int:
        return int(np.count_nonzero(self.material == PRODUCT))

    def describe(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_product_cells": self.n_inside,
            "charge_radius_m": self.charge_radius,
            "outer_radius_m": float(self.radii[-1]),
            "min_width_m": float(np.diff(self.radii).min()),
            "max_width_m": float(np.diff(self.radii).max()),
        }


def build_startup_mesh(
    field: TwoMediumInitialField,
    config: StartupConfig | None = None,
) -> StartupMesh:
    """按初值场的物性建一套**足够分辨**的启动网格。

    初值在两种材料内部都是常数，所以换网格不需要任何重采样：直接按材料标签把
    ``(rho, e)`` 铺上去即可。``R_0`` 处保证恰好有一个节点。

    不能直接拿 ``TwoMediumInitialField.cell_edges`` 当求解网格：那是给双网络
    使用的存储网格，默认第一个空气单元宽约 ``0.32 R_0``，对启动推进太粗。
    """
    if not isinstance(field, TwoMediumInitialField):
        raise StartupError(
            "field must be a TwoMediumInitialField built by "
            "build_two_medium_initial_field"
        )
    cfg = config or StartupConfig()
    cfg.validate()

    r0 = float(field.charge_radius)
    outer = float(cfg.outer_radius_factor) * r0
    inner = np.linspace(0.0, r0, cfg.cells_inside + 1, dtype=np.float64)
    inner_width = r0 / cfg.cells_inside

    widths = []
    width = inner_width
    total = r0
    while total < outer:
        width *= cfg.outside_growth
        widths.append(width)
        total += width
    radii = np.concatenate([inner, r0 + np.cumsum(np.asarray(widths, dtype=np.float64))])

    # 外边界按最后一次累加的结果，保证节点严格递增且覆盖到 outer 以外。
    if radii[-1] < outer:
        raise StartupError("internal error: the generated grid does not reach outer_radius")
    if np.any(np.diff(radii) <= 0.0):
        raise StartupError("the generated grid is not strictly increasing")

    volume = (radii[1:] ** 3 - radii[:-1] ** 3) / 3.0
    material = np.concatenate(
        [
            np.full(cfg.cells_inside, PRODUCT, dtype=np.int64),
            np.full(radii.size - 1 - cfg.cells_inside, AIR, dtype=np.int64),
        ]
    )
    rho_product = float(field.product_state["density"])
    rho_air = float(field.air_state["density"])
    density = np.where(material == PRODUCT, rho_product, rho_air)
    mass = density * volume
    return StartupMesh(
        radii=radii, mass=mass, material=material, charge_radius=r0
    )


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------


@dataclass
class StartupState:
    """一个物理时刻的完整两介质状态（单位立体角）。"""

    time: float
    radii: np.ndarray
    velocity: np.ndarray
    density: np.ndarray
    specific_energy: np.ndarray
    pressure: np.ndarray
    sound_speed: np.ndarray
    material: np.ndarray
    mass: np.ndarray
    cell_volume: np.ndarray
    contact_radius: float
    shock_radius: float
    contact_speed: float
    shock_speed: float
    ambient_pressure: float
    ambient_density: float
    internal_energy: float
    kinetic_energy: float
    boundary_work: float
    energy_drift: float
    #: 最后一步的步长；全部状态量都在 ``time`` 这一整数时刻。
    last_dt: float
    #: 本步「几何体积增量」与「功中体积增量」的最大相对差，及全程最大值。
    volume_work_mismatch: float
    max_volume_work_mismatch: float
    #: 单次运行内部检查是否通过；训练资格还要外部证据（在交接对象上判定）。
    internal_checks_passed: bool
    qualification: dict[str, Any]
    diagnostics: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def centres(self) -> np.ndarray:
        return (0.5 * (self.radii[1:] ** 3 + self.radii[:-1] ** 3)) ** (1.0 / 3.0)

    @property
    def n_inside(self) -> int:
        return int(np.count_nonzero(self.material == PRODUCT))

    @property
    def total_energy(self) -> float:
        return self.internal_energy + self.kinetic_energy + self.boundary_work

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "n_cells": int(self.mass.size),
            "n_product_cells": self.n_inside,
            "contact_radius": self.contact_radius,
            "shock_radius": self.shock_radius,
            "contact_speed": self.contact_speed,
            "shock_speed": self.shock_speed,
            "last_dt": self.last_dt,
            "internal_energy": self.internal_energy,
            "kinetic_energy": self.kinetic_energy,
            "boundary_work": self.boundary_work,
            "energy_drift": self.energy_drift,
            "volume_work_mismatch": self.volume_work_mismatch,
            "max_volume_work_mismatch": self.max_volume_work_mismatch,
            "internal_checks_passed": self.internal_checks_passed,
            "qualification": self.qualification,
            **self.diagnostics,
        }


# --------------------------------------------------------------------------
# 求解器
# --------------------------------------------------------------------------


class TwoMediumStartup:
    """把分段常数两介质初值推进到给定物理时刻。"""

    def __init__(
        self,
        field: TwoMediumInitialField,
        *,
        config: StartupConfig | None = None,
    ) -> None:
        if not isinstance(field, TwoMediumInitialField):
            raise StartupError(
                "field must be a TwoMediumInitialField built by "
                "build_two_medium_initial_field"
            )
        self.field = field
        self.config = config or StartupConfig()
        self.config.validate()

        self.product_eos: JWLParameters = field.product_eos
        self.air_gamma = float(field.air_gamma)
        self.ambient_pressure = float(field.air_state["pressure"])
        self.ambient_density = float(field.air_state["density"])
        rho_product = float(field.product_state["density"])
        e_product = float(field.product_state["specific_energy"])
        rho_air = float(field.air_state["density"])
        e_air = float(field.air_state["specific_energy"])
        for name, value in (
            ("product density", rho_product),
            ("product specific energy", e_product),
            ("air density", rho_air),
            ("air specific energy", e_air),
            ("ambient pressure", self.ambient_pressure),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise StartupError(f"{name} must be finite and positive, got {value!r}")

        self.mesh = build_startup_mesh(field, self.config)
        self.radii = self.mesh.radii.copy()
        self.radii_initial = self.mesh.radii.copy()
        self.velocity = np.zeros(self.radii.size, dtype=np.float64)
        self.mass = self.mesh.mass.copy()
        self.material = self.mesh.material.copy()
        self.specific_energy = np.where(
            self.material == PRODUCT, e_product, e_air
        ).astype(np.float64)
        self.time = 0.0
        self.steps = 0
        self._boundary_work = 0.0
        self._last_dt = 0.0
        self._last_volume_mismatch = 0.0
        self._max_volume_mismatch = 0.0
        #: 最近一步实际使用的人工粘性（诊断用；不计入状态更新）。
        self.artificial_viscosity = np.zeros(self.mass.size, dtype=np.float64)
        self._contact_cell = int(np.flatnonzero(self.material == PRODUCT)[-1])
        #: 最近若干步的 (t, Rs)，用于轨迹差分速度；与 RH 反算是两个独立来源。
        self._shock_samples: deque[tuple[float, float]] = deque(maxlen=8192)

        volume = self._volume(self.radii)
        self.cell_volume = volume
        self.density = self.mass / volume
        self.pressure, self.sound_speed = self._thermodynamics(
            self.density, self.specific_energy
        )
        self._initial_total_energy = self._total_energy(self.specific_energy, self.velocity)

    # -- 热力学 -----------------------------------------------------------
    def _pressure(
        self, density: np.ndarray, specific_energy: np.ndarray
    ) -> np.ndarray:
        product = self.material == PRODUCT
        pressure = np.empty_like(density)
        if np.any(product):
            pressure[product] = jwl_pressure_numpy(
                density[product], specific_energy[product], self.product_eos
            )
        if np.any(~product):
            pressure[~product] = (self.air_gamma - 1.0) * (
                density[~product] * specific_energy[~product]
            )
        return pressure

    def _sound_speed_squared(
        self, density: np.ndarray, specific_energy: np.ndarray
    ) -> np.ndarray:
        product = self.material == PRODUCT
        c2 = np.empty_like(density)
        if np.any(product):
            # 用**未夹正**的 c²：负值必须被检测出来，而不是被 sqrt 变成 0。
            c2[product] = jwl_sound_speed_squared_numpy(
                density[product], specific_energy[product], self.product_eos
            )
        if np.any(~product):
            p_air = (self.air_gamma - 1.0) * (
                density[~product] * specific_energy[~product]
            )
            c2[~product] = self.air_gamma * p_air / density[~product]
        return c2

    def _thermodynamics(
        self, density: np.ndarray, specific_energy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(p, c)``，任何非有限或非正的结果都直接失败。"""
        if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
            raise StartupError("density became non-finite or non-positive")
        if np.any(~np.isfinite(specific_energy)) or np.any(specific_energy <= 0.0):
            raise StartupError(
                "specific energy became non-finite or non-positive; "
                "the timestep or the mesh is no longer admissible"
            )
        pressure = self._pressure(density, specific_energy)
        c2 = self._sound_speed_squared(density, specific_energy)
        if np.any(~np.isfinite(pressure)):
            raise StartupError("pressure became non-finite")
        if np.any(pressure <= 0.0):
            index = int(np.argmin(pressure))
            raise StartupError(
                "pressure became non-positive in cell "
                f"{index} (material {int(self.material[index])}): {float(pressure[index])!r}"
            )
        if np.any(~np.isfinite(c2)):
            raise StartupError("sound speed squared became non-finite")
        if np.any(c2 <= 0.0):
            index = int(np.argmin(c2))
            raise StartupError(
                "sound speed squared became non-positive in cell "
                f"{index} (material {int(self.material[index])}): {float(c2[index])!r}; "
                "refusing to clamp it to zero"
            )
        return pressure, np.sqrt(c2)

    # -- 几何 -------------------------------------------------------------
    @staticmethod
    def _volume(radii: np.ndarray) -> np.ndarray:
        return (radii[1:] ** 3 - radii[:-1] ** 3) / 3.0

    def _node_mass(self) -> np.ndarray:
        node = np.empty(self.mass.size + 1, dtype=np.float64)
        node[0] = 0.5 * self.mass[0]
        node[1:-1] = 0.5 * (self.mass[:-1] + self.mass[1:])
        node[-1] = 0.5 * self.mass[-1]
        return node

    # -- 人工粘性 ---------------------------------------------------------
    def _artificial_viscosity(
        self, velocity: np.ndarray, density: np.ndarray, sound_speed: np.ndarray
    ) -> np.ndarray:
        du = np.diff(velocity)
        compression = np.where(du < 0.0, du, 0.0)
        q = (
            self.config.q1 * density * sound_speed * np.abs(compression)
            + self.config.q2 * density * compression**2
        )
        return np.where(compression < -self.config.compression_floor, q, 0.0)

    # -- 时间步 -----------------------------------------------------------
    def _timestep(self) -> float:
        width = np.diff(self.radii)
        cell_velocity = 0.5 * (self.velocity[:-1] + self.velocity[1:])
        speed = self.sound_speed + np.abs(cell_velocity)
        dt = self.config.cfl * float(np.min(width / np.maximum(speed, 1e-30)))
        if not math.isfinite(dt) or dt <= 0.0:
            raise StartupError(f"non-positive timestep {dt!r}")
        return dt

    # -- 一步 -------------------------------------------------------------
    def _advance(self, dt: float) -> None:
        radii = self.radii
        velocity = self.velocity
        specific_energy = self.specific_energy

        volume = self._volume(radii)
        density = self.mass / volume
        pressure, sound_speed = self._thermodynamics(density, specific_energy)
        q = self._artificial_viscosity(velocity, density, sound_speed)
        self.artificial_viscosity = q.copy()
        total_pressure = pressure + q
        if np.any(total_pressure <= 0.0):
            index = int(np.argmin(total_pressure))
            raise StartupError(
                "p + q became non-positive in cell "
                f"{index}: {float(total_pressure[index])!r}"
            )

        # 动量：与能量使用同一组 (p+q) 和同一个 A = r^2。
        area = radii**2
        node_mass = self._node_mass()
        jump = np.zeros(radii.size, dtype=np.float64)
        jump[1:-1] = total_pressure[1:] - total_pressure[:-1]
        jump[-1] = self.ambient_pressure - total_pressure[-1]
        acceleration = -(area * jump) / node_mass
        acceleration[0] = 0.0  # 球心对称

        radii_new, velocity_new = integrate_positions(radii, velocity, acceleration, dt)
        u_bar = 0.5 * (velocity + velocity_new)

        # 相容能量：力—功配对 + 同一组新旧速度的平均。
        flux = area[1:] * u_bar[1:] - area[:-1] * u_bar[:-1]
        energy_new = specific_energy - (dt / self.mass) * total_pressure * flux
        if np.any(~np.isfinite(energy_new)) or np.any(energy_new <= 0.0):
            index = int(np.argmin(energy_new))
            raise StartupError(
                "specific energy became non-finite or non-positive in cell "
                f"{index}: {float(energy_new[index])!r}"
            )

        if np.any(radii_new < 0.0) or np.any(np.diff(radii_new) <= 0.0):
            raise StartupError(
                "the Lagrangian mesh tangled or crossed the axis; reduce the timestep"
            )

        # 几何体积增量与功中采用的体积增量是**两个不同的量**：
        #   几何：V(r^{n+1}) - V(r^n)，节点按梯形位置更新移动；
        #   功中：dt * (A_{i+1} ubar_{i+1} - A_i ubar_i)，A = (r^n)^2。
        # 二者只在 O(dt) 内一致；能量恒等式成立**不代表**它们相同，必须单独报告。
        volume_new = self._volume(radii_new)
        delta_volume_geometric = volume_new - volume
        delta_volume_work = dt * flux
        # 用全局尺度归一化：某些单元的 dV 可以接近 0（例如外边界附近速度很小），
        # 逐单元相除会把舍入噪声放大成 1e279 这种无意义的数。
        volume_scale = max(float(np.max(np.abs(delta_volume_geometric))), 1.0e-300)
        self._last_volume_mismatch = float(
            np.max(np.abs(delta_volume_work - delta_volume_geometric)) / volume_scale
        )
        self._max_volume_mismatch = max(
            self._max_volume_mismatch, self._last_volume_mismatch
        )

        # 实际边界功：外边界环境压力 × 面积 × 时间中心速度。
        self._boundary_work += dt * self.ambient_pressure * area[-1] * u_bar[-1]

        self.radii = radii_new
        self.velocity = velocity_new
        self.specific_energy = energy_new
        self.time += dt
        self.steps += 1
        self._last_dt = dt

        self.cell_volume = volume_new
        self.density = self.mass / self.cell_volume
        self.pressure, self.sound_speed = self._thermodynamics(
            self.density, self.specific_energy
        )

        shock_radius, formed, _ = self._shock_radius_only()
        if formed:
            self._shock_samples.append((self.time, shock_radius))

        if self._energy_drift() > _ENERGY_DRIFT_TOL:
            raise StartupError(
                "the discrete energy identity drifted by "
                f"{self._energy_drift():.3e} (tolerance {_ENERGY_DRIFT_TOL:.1e}); "
                "the force-work pairing is broken"
            )

    # -- 能量账 -----------------------------------------------------------
    def _kinetic_energy(self, velocity: np.ndarray) -> float:
        return float(0.5 * np.sum(self._node_mass() * velocity**2))

    def _total_energy(self, specific_energy: np.ndarray, velocity: np.ndarray) -> float:
        return (
            float(np.sum(self.mass * specific_energy))
            + self._kinetic_energy(velocity)
            + self._boundary_work
        )

    def _energy_drift(self) -> float:
        current = self._total_energy(self.specific_energy, self.velocity)
        scale = max(abs(self._initial_total_energy), 1.0e-300)
        return abs(current - self._initial_total_energy) / scale

    # -- 前沿 -------------------------------------------------------------
    def _contact_radius(self) -> float:
        return float(self.radii[self._contact_cell + 1])

    def _centres(self) -> np.ndarray:
        return (0.5 * (self.radii[1:] ** 3 + self.radii[:-1] ** 3)) ** (1.0 / 3.0)

    def _shock_location(self) -> ShockLocation:
        return locate_shock(
            self._centres(),
            self.pressure,
            self.material == AIR,
            self.ambient_pressure,
        )

    def _shock_radius_only(self) -> tuple[float, bool, int]:
        """返回 ``(Rs, 是否可可靠定位, 半跳变单元下标)``。"""
        location = self._shock_location()
        if location.valid:
            return location.radius, True, location.shock_cell
        # 定位不可靠时退化为「最外侧扰动单元的外侧节点」，只用于导出诊断快照；
        # 该位置不进入 RH、也不会让资格审查通过。
        if location.front_cell >= 0:
            index = min(location.front_cell + 1, self.radii.size - 1)
            return float(self.radii[index]), False, -1
        return self._contact_radius(), False, -1

    def _trajectory_shock_speed(
        self, *, window_cells: float = 10.0, min_cells: float = 3.0
    ) -> float:
        """由激波轨迹差分得到的速度（局部窗口），与 RH 反算互相独立。

        窗口规则：从**最近**样本往回找，取第一个至少落后 ``window_cells`` 个当地
        单元宽度的样本。这样得到的是满足分辨率要求的**最短**窗口，而不是从最早
        样本开始的整段历史——后者在长历史下会把早期慢速段平均进来。

        历史不足 ``min_cells`` 个单元宽度时返回 ``nan``（诊断无效），不拿一个
        量化噪声当速度。
        """
        if len(self._shock_samples) < 2:
            return float("nan")
        t1, r1 = self._shock_samples[-1]
        index = int(np.searchsorted(self.radii, r1, side="right")) - 1
        index = min(max(index, 0), self.radii.size - 2)
        width = float(self.radii[index + 1] - self.radii[index])
        chosen = self._shock_samples[0]
        for sample in reversed(self._shock_samples):
            chosen = sample
            if r1 - sample[1] >= window_cells * width:
                break
        t0, r0 = chosen
        if t1 <= t0 or (r1 - r0) < min_cells * width:
            return float("nan")
        window = [sample for sample in self._shock_samples if sample[0] >= t0]
        if len(window) >= 3:
            # 窗口内做二次拟合，取端点导数：把「窗口平均」带来的减速偏差降一阶。
            times = np.array([sample[0] - t1 for sample in window])
            radii = np.array([sample[1] - r1 for sample in window])
            if np.unique(times).size >= 3:
                return float(np.polyfit(times, radii, 2)[1])
        return float((r1 - r0) / (t1 - t0))

    def _shock(self) -> tuple[float, float, dict[str, Any]]:
        """返回 ``(Rs, shock_speed, 诊断)``。

        对外 ``shock_speed`` 的定义是**质量 RH 反算速度**：由波后平台态
        ``(rho_b, u_b)`` 与解析环境态 ``rho_0`` 给出，和 PINN 激波条件同一口径。
        轨迹差分速度单独保留在 ``shock_speed_trajectory``，只用于独立核对；两者
        不在同一处互相赋值。
        """
        location = self._shock_location()
        if location.valid:
            shock_radius = location.radius
        elif location.front_cell >= 0:
            shock_radius = float(
                self.radii[min(location.front_cell + 1, self.radii.size - 1)]
            )
        else:
            shock_radius = self._contact_radius()
        diagnostics: dict[str, Any] = {
            "shock_location_reason": location.reason,
            "shock_front_cell": float(location.front_cell),
            "shock_diagnostics_valid": False,
            "shock_speed_rh": float("nan"),
            "shock_speed_trajectory": float("nan"),
            "shock_speed_deviation": float("nan"),
            "shock_rh_mass": float("nan"),
            "shock_rh_momentum": float("nan"),
            "shock_rh_energy": float("nan"),
            "shock_rh_mass_rel": float("nan"),
            "shock_rh_momentum_rel": float("nan"),
            "shock_rh_energy_rel": float("nan"),
            "shock_plateau_cell": float(location.plateau_cell),
            "shock_shock_cell": float(location.shock_cell),
            "shock_plateau_offset": (
                float(location.shock_cell - location.plateau_cell)
                if location.valid
                else float("nan")
            ),
            "shock_pressure_peak": float(
                np.max(self.pressure[self.material == AIR])
            ),
        }
        if not location.valid:
            return shock_radius, 0.0, diagnostics

        plateau = location.plateau_cell
        offset = location.shock_cell - plateau
        rho_b = float(self.density[plateau])
        u_b = 0.5 * float(self.velocity[plateau] + self.velocity[plateau + 1])
        p_b = float(self.pressure[plateau])
        e_b = float(self.specific_energy[plateau])
        rho_0 = self.ambient_density
        p_0 = self.ambient_pressure
        e_0 = float(self.field.air_state["specific_energy"])

        denominator = rho_b - rho_0
        shock_speed_rh = rho_b * u_b / denominator if denominator > 0.0 else float("nan")
        shock_speed_trajectory = self._trajectory_shock_speed()

        residuals = shock_rh_residuals(
            post_density=rho_b,
            post_velocity=u_b,
            post_pressure=p_b,
            post_specific_energy=e_b,
            ambient_density=rho_0,
            ambient_pressure=p_0,
            ambient_specific_energy=e_0,
            shock_speed=shock_speed_trajectory,
        )
        scale = max(abs(shock_speed_rh), abs(shock_speed_trajectory), 1.0e-30)
        deviation = (
            abs(shock_speed_rh - shock_speed_trajectory) / scale
            if math.isfinite(shock_speed_rh) and math.isfinite(shock_speed_trajectory)
            else float("nan")
        )
        valid = bool(
            math.isfinite(shock_speed_rh)
            and math.isfinite(shock_speed_trajectory)
            and rho_b > rho_0
            and 0 <= offset <= _MAX_RAMP_CELLS
            and p_b > p_0 * (1.0 + _SHOCK_EXCESS_TOL)
            and residuals["shock_rh_mass_rel"] <= _RH_REL_TOL
            and residuals["shock_rh_momentum_rel"] <= _RH_REL_TOL
            and residuals["shock_rh_energy_rel"] <= _RH_REL_TOL
            and deviation <= _FRONT_SPEED_REL_TOL
        )
        diagnostics.update(
            {
                "shock_diagnostics_valid": valid,
                "shock_speed_rh": float(shock_speed_rh),
                "shock_speed_trajectory": float(shock_speed_trajectory),
                "shock_speed_deviation": float(deviation),
                "shock_plateau_cell": float(plateau),
                "shock_plateau_offset": float(offset),
                "shock_pressure_peak": p_b,
                "shock_post_density": rho_b,
                "shock_post_velocity": u_b,
                **residuals,
            }
        )
        return shock_radius, float(shock_speed_rh), diagnostics

    # -- 快照 -------------------------------------------------------------
    def state(self) -> StartupState:
        contact = self._contact_radius()
        shock, shock_speed, diagnostics = self._shock()
        internal = float(np.sum(self.mass * self.specific_energy))
        kinetic = self._kinetic_energy(self.velocity)
        drift = self._energy_drift()

        index = int(np.searchsorted(self.radii, shock, side="right")) - 1
        index = min(max(index, 0), self.radii.size - 2)
        local_width = float(self.radii[index + 1] - self.radii[index])
        gap_cells = (
            (shock - contact) / local_width if local_width > 0.0 else 0.0
        )
        checks = {
            "shock_diagnostics_valid": bool(
                diagnostics.get("shock_diagnostics_valid", False)
            ),
            "energy_identity": bool(drift <= _QUALITY_ENERGY_DRIFT),
            "volume_work_consistency": bool(
                self._max_volume_mismatch <= _QUALITY_VOLUME_MISMATCH
            ),
            "front_order": bool(contact < shock),
            "fronts_inside_domain": bool(
                0.0 < contact and shock < float(self.radii[-1])
            ),
            "air_gap_resolved": bool(gap_cells >= _QUALITY_AIR_GAP_CELLS),
        }
        qualification = {
            "checks": checks,
            "failed": [name for name, ok in checks.items() if not ok],
            "air_gap_cells": float(gap_cells),
        }
        return StartupState(
            time=self.time,
            radii=self.radii.copy(),
            velocity=self.velocity.copy(),
            density=self.density.copy(),
            specific_energy=self.specific_energy.copy(),
            pressure=self.pressure.copy(),
            sound_speed=self.sound_speed.copy(),
            material=self.material.copy(),
            mass=self.mass.copy(),
            cell_volume=self.cell_volume.copy(),
            contact_radius=contact,
            shock_radius=shock,
            contact_speed=float(self.velocity[self._contact_cell + 1]),
            shock_speed=shock_speed,
            ambient_pressure=self.ambient_pressure,
            ambient_density=self.ambient_density,
            internal_energy=internal,
            kinetic_energy=kinetic,
            boundary_work=float(self._boundary_work),
            energy_drift=drift,
            last_dt=self._last_dt,
            volume_work_mismatch=self._last_volume_mismatch,
            max_volume_work_mismatch=self._max_volume_mismatch,
            internal_checks_passed=all(checks.values()),
            qualification=qualification,
            diagnostics={"steps": float(self.steps), **diagnostics},
        )

    # -- 主循环 -----------------------------------------------------------
    def run_to(
        self,
        end_time: float,
        *,
        sample_times: Sequence[float] | None = None,
        on_sample: Callable[[StartupState], None] | None = None,
    ) -> StartupState:
        if not math.isfinite(end_time) or end_time <= 0.0:
            raise StartupError(f"end_time must be finite and positive, got {end_time!r}")
        targets = sorted(
            float(t) for t in (sample_times or ()) if 0.0 < float(t) <= end_time
        )
        pending = list(targets)
        while self.time < end_time:
            dt = self._timestep()
            remaining = end_time - self.time
            if dt >= remaining:
                dt = remaining
            before = self.time
            self._advance(dt)
            if self.time <= before:  # pragma: no cover - defensive
                raise StartupError(
                    "the timestep is below the floating-point resolution of the clock"
                )
            while pending and self.time >= pending[0] - 1.0e-18:
                if on_sample is not None:
                    on_sample(self.state())
                pending.pop(0)
            if self.steps > self.config.max_steps:
                raise StartupError(
                    f"exceeded max_steps = {self.config.max_steps} before t = {end_time}"
                )
        return self.state()


# --------------------------------------------------------------------------
# 交接对象
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoMediumHandoff:
    """双网络传播阶段的启动场：同一物理时刻的完整两介质状态与两个前沿。

    这不是旧的 ``ReferenceField``：它不携带旧参考场的验收状态，不能被旧闸门当作
    已验收场；插值接口按材料分开，从结构上禁止跨界面混合。
    """

    model_id: str
    method_version: str
    time: float
    charge_radius: float
    radii: np.ndarray
    velocity: np.ndarray
    density: np.ndarray
    specific_energy: np.ndarray
    pressure: np.ndarray
    sound_speed: np.ndarray
    material: np.ndarray
    mass: np.ndarray
    cell_volume: np.ndarray
    contact_radius: float
    shock_radius: float
    contact_speed: float
    shock_speed: float
    product_eos: JWLParameters
    air_gamma: float
    #: 单次运行的内部检查；训练资格另看外部证据。
    internal_checks_passed: bool
    external_validation: dict[str, Any]
    training_qualified: bool
    qualification: dict[str, Any]
    provenance: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        cells = int(self.material.size)
        for name in (
            "density",
            "specific_energy",
            "pressure",
            "sound_speed",
            "mass",
            "cell_volume",
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, array)
            if array.shape != (cells,):
                raise StartupError(f"{name} must have shape {(cells,)}, got {array.shape}")
            if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
                raise StartupError(f"{name} must be finite and positive")
        for name in ("radii", "velocity"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, array)
            if array.shape != (cells + 1,):
                raise StartupError(
                    f"{name} must have shape {(cells + 1,)}, got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise StartupError(f"{name} must be finite")
        material = np.asarray(self.material, dtype=np.int64)
        object.__setattr__(self, "material", material)
        if not np.all(np.isin(material, (PRODUCT, AIR))):
            raise StartupError("material tags must be product or air")
        product = material == PRODUCT
        if not np.any(product) or np.any(material[np.count_nonzero(product) :] == PRODUCT):
            raise StartupError("material must be sorted: product cells first, then air")
        if np.any(np.diff(self.radii) <= 0.0):
            raise StartupError("handoff radii must be strictly increasing")
        if not (0.0 < self.time and math.isfinite(self.time)):
            raise StartupError(f"handoff time must be finite and positive, got {self.time!r}")
        if not (self.contact_radius < self.shock_radius):
            raise StartupError(
                "the handoff must satisfy Rc < Rs; got "
                f"Rc={self.contact_radius!r}, Rs={self.shock_radius!r}"
            )
        if not all(
            math.isfinite(v)
            for v in (
                self.contact_radius,
                self.shock_radius,
                self.contact_speed,
                self.shock_speed,
            )
        ):
            raise StartupError("front positions and speeds must be finite")

    # -- 查询 -------------------------------------------------------------
    @property
    def centres(self) -> np.ndarray:
        return (0.5 * (self.radii[1:] ** 3 + self.radii[:-1] ** 3)) ** (1.0 / 3.0)

    @property
    def n_inside(self) -> int:
        return int(np.count_nonzero(self.material == PRODUCT))

    def _query(self, radius) -> np.ndarray:
        query = np.asarray(radius, dtype=np.float64)
        if np.any(~np.isfinite(query)):
            raise StartupError("query radii must be finite")
        return query

    def sample_product(self, radius) -> dict[str, np.ndarray]:
        """在产物侧物理域 ``[0, R_c]`` 内查询。

        ``rho`` 与 ``e`` 是单元量，在单元中心之间插值；``u`` 是**节点量**，在节点
        半径与节点速度之间插值——不再先转成单元平均速度再延拓。``p`` 与 ``c``
        由 ``product_eos`` 从重构出的 ``(rho, e)`` 重新计算，不插值。

        ``r = 0`` 处取到的是节点 0 的速度（构造成 ``u = 0``）；``r = R_c`` 处取到的
        是共享接触节点的速度，与 ``contact_speed`` 逐位相同。单元中心与物理边界
        之间（半格）的 ``rho, e`` 用端点值作**显式**常值延拓；这是边界取值的定义，
        不是精度声明。越过 ``R_c`` 直接报错，不做跨材料平均。
        """
        query = self._query(radius)
        if np.any(query < 0.0) or np.any(query > self.contact_radius):
            raise StartupError(
                "radius outside the product physical domain "
                f"[0, R_c = {self.contact_radius:.6g}] m"
            )
        mask = self.material == PRODUCT
        centres = self.centres[mask]
        rho = np.interp(query, centres, self.density[mask])
        energy = np.interp(query, centres, self.specific_energy[mask])
        nodes = self.n_inside + 1
        velocity = np.interp(query, self.radii[:nodes], self.velocity[:nodes])
        pressure = jwl_pressure_numpy(rho, energy, self.product_eos)
        c2 = jwl_sound_speed_squared_numpy(rho, energy, self.product_eos)
        if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise StartupError("interpolated product state gives a non-positive pressure")
        if not np.all(np.isfinite(c2)) or np.any(c2 <= 0.0):
            raise StartupError(
                "interpolated product state gives a non-positive sound speed squared"
            )
        return {
            "density": rho,
            "velocity": velocity,
            "specific_energy": energy,
            "pressure": pressure,
            "sound_speed": np.sqrt(c2),
        }

    def sample_air(self, radius) -> dict[str, np.ndarray]:
        """在空气侧物理域 ``[R_c, R_outer]`` 内查询。

        ``rho`` 与 ``e`` 在空气单元中心之间插值；``u`` 在**节点**半径与节点速度
        之间插值（节点 ``n_inside`` 就是接触节点）。``p`` 与 ``c`` 由 ``air_gamma``
        从重构出的 ``(rho, e)`` 重新计算。

        ``r > R_s``：环境态是**已知的精确常量**（静止、``rho_0, p_0, e_0``），
        直接返回并标记 ``undisturbed``；这不是用插值掩盖缺失数据。

        ``r = R_s`` 处的插值给出数值剖面的单侧极限，**不是**波后平台态；RH 使用的
        波后状态来自 ``shock_*`` 诊断里的平台单元值，不能拿这里的中间状态顶替。
        """
        query = self._query(radius)
        outer = float(self.radii[-1])
        if np.any(query < self.contact_radius) or np.any(query > outer):
            raise StartupError(
                "radius outside the air physical domain "
                f"[R_c = {self.contact_radius:.6g}, {outer:.6g}] m"
            )
        mask = self.material == AIR
        centres = self.centres[mask]
        rho = np.empty_like(query)
        energy = np.empty_like(query)
        velocity = np.empty_like(query)
        undisturbed = query > self.shock_radius
        inside = ~undisturbed
        if np.any(inside):
            rho[inside] = np.interp(query[inside], centres, self.density[mask])
            energy[inside] = np.interp(
                query[inside], centres, self.specific_energy[mask]
            )
            velocity[inside] = np.interp(
                query[inside],
                self.radii[self.n_inside :],
                self.velocity[self.n_inside :],
            )
        if np.any(undisturbed):
            ambient_energy = float(self.provenance["initial_state"]["air_specific_energy"])
            rho[undisturbed] = float(self.provenance["initial_state"]["air_density"])
            energy[undisturbed] = ambient_energy
            velocity[undisturbed] = 0.0
        pressure = (self.air_gamma - 1.0) * rho * energy
        c2 = self.air_gamma * pressure / rho
        if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise StartupError("interpolated air state gives a non-positive pressure")
        if not np.all(np.isfinite(c2)) or np.any(c2 <= 0.0):
            raise StartupError(
                "interpolated air state gives a non-positive sound speed squared"
            )
        return {
            "density": rho,
            "velocity": velocity,
            "specific_energy": energy,
            "pressure": pressure,
            "sound_speed": np.sqrt(c2),
            "undisturbed": undisturbed,
        }

    def _face_state(self, side: int) -> dict[str, float]:
        """把一侧最近两个单元中心线性外推到接触面 ``R_c``。

        单元中心之间的插值在 ``R_c`` 处只能给端点值；接触面本身在半个单元之外，
        压力梯度会混进「接触面压力差」里。这里用单侧线性外推读**面上**的值。
        """
        mask = self.material == side
        centres = self.centres[mask]
        density = self.density[mask]
        energy = self.specific_energy[mask]
        if side == PRODUCT:
            c0, c1 = float(centres[-2]), float(centres[-1])
            rho0, rho1 = float(density[-2]), float(density[-1])
            e0, e1 = float(energy[-2]), float(energy[-1])
        else:
            c0, c1 = float(centres[0]), float(centres[1])
            rho0, rho1 = float(density[0]), float(density[1])
            e0, e1 = float(energy[0]), float(energy[1])
        face = float(self.contact_radius)
        slope_rho = (rho1 - rho0) / (c1 - c0)
        slope_e = (e1 - e0) / (c1 - c0)
        rho = rho1 + slope_rho * (face - c1)
        specific = e1 + slope_e * (face - c1)
        if side == PRODUCT:
            pressure = float(
                jwl_pressure_numpy(np.array([rho]), np.array([specific]), self.product_eos)[0]
            )
        else:
            pressure = float((self.air_gamma - 1.0) * rho * specific)
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise StartupError(
                f"contact-face extrapolation gave a non-positive {side} pressure"
            )
        return {
            "density": rho,
            "specific_energy": specific,
            "pressure": pressure,
        }

    def contact_states(self) -> dict[str, Any]:
        """接触面一致性：相邻单元压力差 + 共享节点速度 + 线性外推诊断。

        ``pressure_relative`` 用接触面**两侧相邻单元中心**的压力。它是面上压力差的
        离散上界，包含了半个单元宽度内的梯度；接触面附近的产物稀疏波很陡时，
        这个值会明显偏大，但它是解里真实存在的离散跳变，不能靠外推把它抹掉。

        ``face_linear`` 是单侧线性外推到 ``R_c`` 的值，仅作诊断：最后两个单元
        跨越陡峭的稀疏波时线性外推并不可靠，因此不用于放行判定。
        """
        product_cell = int(self.n_inside - 1)
        air_cell = int(self.n_inside)
        product_pressure = float(self.pressure[product_cell])
        air_pressure = float(self.pressure[air_cell])
        pressure_relative = abs(product_pressure - air_pressure) / max(
            abs(product_pressure), abs(air_pressure), 1.0e-30
        )
        node_velocity = float(self.velocity[self.n_inside])
        linear_product = self._face_state(PRODUCT)
        linear_air = self._face_state(AIR)
        return {
            "product": {
                "pressure": product_pressure,
                "cell_index": product_cell,
                "radius_m": float(self.centres[product_cell]),
            },
            "air": {
                "pressure": air_pressure,
                "cell_index": air_cell,
                "radius_m": float(self.centres[air_cell]),
            },
            "pressure_relative": float(pressure_relative),
            "node_velocity": node_velocity,
            "contact_speed": self.contact_speed,
            # 两侧共享同一个节点，速度差按构造为零；如实报告，不当成验证。
            "velocity_relative": float(
                abs(node_velocity - self.contact_speed)
                / max(abs(self.contact_speed), 1.0e-30)
            ),
            "face_linear": {
                "product_pressure": linear_product["pressure"],
                "air_pressure": linear_air["pressure"],
                "pressure_relative": abs(
                    linear_product["pressure"] - linear_air["pressure"]
                )
                / max(
                    abs(linear_product["pressure"]),
                    abs(linear_air["pressure"]),
                    1.0e-30,
                ),
                "reliable": False,
                "reason": (
                    "linear extrapolation across the last cell is unreliable when the "
                    "product rarefaction is steep; diagnostic only"
                ),
            },
        }

    def fronts_at(self, time: float) -> tuple[float, float, float, float]:
        """本交接对象只表示一个时刻；传入其他时刻直接拒绝。"""
        if abs(float(time) - self.time) > 1.0e-18:
            raise StartupError(
                f"this handoff represents t = {self.time!r} s only, got {time!r}"
            )
        return (
            self.contact_radius,
            self.shock_radius,
            self.contact_speed,
            self.shock_speed,
        )

    def cache_key(self) -> str:
        """缓存标识绑定初值、EOS、网格、方法版本与交接时间。"""
        payload = {
            "model_id": self.model_id,
            "method_version": self.method_version,
            "time": self.time,
            "charge_radius": self.charge_radius,
            "jwl": {
                name: float(getattr(self.product_eos, name))
                for name in ("A", "B", "R1", "R2", "omega", "rho0", "energy_shift")
            },
            "air_gamma": self.air_gamma,
            "n_cells": int(self.material.size),
            "n_product_cells": self.n_inside,
            "contact_radius": self.contact_radius,
            "shock_radius": self.shock_radius,
            # 初值参数也必须进缓存标识，不能只靠演化后的场反推。
            "initial_state": self.provenance.get("initial_state", {}),
            "mesh": self.provenance.get("mesh", {}),
            "config": self.provenance.get("config", {}),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "method_version": self.method_version,
            "physical_time_s": self.time,
            "charge_radius_m": self.charge_radius,
            "n_cells": int(self.material.size),
            "n_product_cells": self.n_inside,
            "contact_radius_m": self.contact_radius,
            "shock_radius_m": self.shock_radius,
            "contact_speed_m_s": self.contact_speed,
            "shock_speed_m_s": self.shock_speed,
            "internal_checks_passed": self.internal_checks_passed,
            "external_validation_accepted": self.external_validation.get("accepted", False),
            "training_qualified": self.training_qualified,
            "qualification": self.qualification,
            "air_gamma": self.air_gamma,
            "product_eos": self.product_eos.as_dict(),
            "cache_key": self.cache_key(),
            **self.provenance,
        }


def build_handoff(
    solver: TwoMediumStartup,
    *,
    validation_evidence: Any = None,
) -> TwoMediumHandoff:
    """把求解器的当前状态封装成交接对象。"""
    state = solver.state()
    if state.time <= 0.0:
        raise StartupError(
            "the handoff must be taken at t > 0; the physical initial time t = 0 "
            "is not a startup field"
        )
    binding = {
        "method_version": METHOD_VERSION,
        "charge_radius": float(solver.field.charge_radius),
        "handoff_time": state.time,
        "initial_state": {
            "product_density": float(solver.field.product_state["density"]),
            "product_specific_energy": float(
                solver.field.product_state["specific_energy"]
            ),
            "air_density": float(solver.field.air_state["density"]),
            "air_specific_energy": float(solver.field.air_state["specific_energy"]),
            "air_gamma": float(solver.field.air_gamma),
            "charge_radius": float(solver.field.charge_radius),
        },
        "eos": {
            key: float(value) for key, value in solver.product_eos.as_dict().items()
        },
        "artificial_viscosity": {
            "q1": float(solver.config.q1),
            "q2": float(solver.config.q2),
        },
        "handoff_config": {
            "cells_inside": int(solver.config.cells_inside),
            "outside_growth": float(solver.config.outside_growth),
            "outer_radius_factor": float(solver.config.outer_radius_factor),
            "cfl": float(solver.config.cfl),
        },
    }
    external = assess_validation_evidence(validation_evidence, binding=binding)

    provenance = {
        "construction": "分段常数初值 + 球对称拉格朗日相容能量推进",
        "initial_model_id": solver.field.metadata.get("model_id"),
        "initial_physical_time_s": solver.field.time,
        "handoff_physical_time_s": state.time,
        "time_origin": "t_phys = 0 at the piecewise-constant initial field",
        "pin_model_time": "tau = t_phys - t_h",
        "units": "SI; mass and energy per unit solid angle",
        "initial_state": {
            "product_density": float(solver.field.product_state["density"]),
            "product_specific_energy": float(
                solver.field.product_state["specific_energy"]
            ),
            "air_density": float(solver.field.air_state["density"]),
            "air_specific_energy": float(solver.field.air_state["specific_energy"]),
            "air_gamma": float(solver.field.air_gamma),
            "charge_radius": float(solver.field.charge_radius),
        },
        "mesh": solver.mesh.describe(),
        "config": {
            "cells_inside": solver.config.cells_inside,
            "outside_growth": solver.config.outside_growth,
            "outer_radius_factor": solver.config.outer_radius_factor,
            "cfl": solver.config.cfl,
            "q1": solver.config.q1,
            "q2": solver.config.q2,
        },
        "energy": {
            "internal": state.internal_energy,
            "kinetic": state.kinetic_energy,
            "boundary_work": state.boundary_work,
            "drift": state.energy_drift,
        },
        "time_levels": {
            "all_state": "t^n = state.time (integer time levels; no half-step velocity)",
            "update": "u^{n+1} = u^n + dt a^n; r^{n+1} = r^n + dt (u^n+u^{n+1})/2",
            "order": (
                "trapezoidal position (exact for constant acceleration), but the "
                "force and area are evaluated at t^n with no predictor-corrector: "
                "the velocity is first order and the scheme is not full Verlet"
            ),
            "last_dt_s": state.last_dt,
            "variable_timestep": (
                "each step uses its own dt; the last step is clipped to land on t_h"
            ),
        },
        "volume_work": {
            "last_relative_mismatch": state.volume_work_mismatch,
            "max_relative_mismatch": state.max_volume_work_mismatch,
            "definition": (
                "|dV_work - dV_geometric| / |dV_geometric|, "
                "dV_geometric = V(r^{n+1}) - V(r^n), "
                "dV_work = dt * (A_{i+1} ubar_{i+1} - A_i ubar_i), A = (r^n)^2"
            ),
        },
        "shock_speed_definition": (
            "shock_speed = mass-RH speed from the post-shock plateau; "
            "shock_speed_trajectory is an independent local-window estimate and is "
            "never substituted for it"
        ),
        "internal_checks": state.qualification,
        "external_validation": external,
        "front_diagnostics": {
            key: value
            for key, value in state.diagnostics.items()
            if key.startswith("shock_")
        },
    }
    qualification = {
        "internal_checks": state.qualification,
        "external_validation": external,
        # 没有外部证据时训练资格必须保持 false；内部检查通过不构成训练资格。
        "training_qualified": bool(
            state.internal_checks_passed and external["accepted"]
        ),
    }
    return TwoMediumHandoff(
        model_id=STARTUP_MODEL_ID,
        method_version=METHOD_VERSION,
        time=state.time,
        charge_radius=float(solver.field.charge_radius),
        radii=state.radii,
        velocity=state.velocity,
        density=state.density,
        specific_energy=state.specific_energy,
        pressure=state.pressure,
        sound_speed=state.sound_speed,
        material=state.material,
        mass=state.mass,
        cell_volume=state.cell_volume,
        contact_radius=state.contact_radius,
        shock_radius=state.shock_radius,
        contact_speed=state.contact_speed,
        shock_speed=state.shock_speed,
        product_eos=solver.product_eos,
        air_gamma=solver.air_gamma,
        internal_checks_passed=state.internal_checks_passed,
        external_validation=external,
        training_qualified=bool(
            state.internal_checks_passed and external["accepted"]
        ),
        qualification=qualification,
        provenance=provenance,
    )


__all__ = [
    "METHOD_VERSION",
    "STARTUP_MODEL_ID",
    "StartupConfig",
    "StartupError",
    "StartupMesh",
    "StartupState",
    "TwoMediumHandoff",
    "TwoMediumStartup",
    "build_handoff",
    "build_startup_mesh",
]
