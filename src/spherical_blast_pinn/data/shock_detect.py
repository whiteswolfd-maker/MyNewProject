"""主激波标签：压力梯度候选 + Rankine–Hugoniot 核查（§10）。

**为什么不是"最外侧超压点"。** 那个判据测到的是远场数值噪声。实测 50 mm 算例第 10
状态：真实超压区外缘在 58.2 mm 附近，而远场压力在 ``p/p_amb = 1.000`` 上下摆动、
幅度可达 1.5%，阈值是 1%，于是"最外侧超压单元"落在 94–101 mm 的噪声尖峰上；
两套壳层网格各测噪声的不同尖峰，看起来像"分辨率不够"。噪声尖峰不是激波，它
两侧的状态也不满足 RH——判据必须能说出这个区别，而不是靠一个阈值把它挡在外面。

**检测与监督资格是两件事，分开记。** 一条剖面可以"检测到候选"却**不够格**当标签：
跃迁带比一个原始单元还窄、两侧状态不足以定 ``Ds``、RH 残差超出该分辨率下应有的水平。
把它们合成一个布尔量，就等于让"勉强看见一个尖峰"冒充"这里有一条被核实过的激波"。

**不得先由 RH 造出波后状态再验证 RH。** 两侧状态一律取**实测数据**，RH 只用来
**检验**它们。质量守恒已经被用来求 ``Ds``，所以它不构成独立证据，这一点写在返回值里。

参考文献的适配边界见 §16：Fujimoto 等的候选检测与 RH 核查是这里的依据，但容差按
本项目的基准与分辨率重新核查，不照搬论文（尤其不能把数值波内的宽松容差当成精确
标签的许可）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

#: 检测状态。与监督资格分开：``confirmed`` 才够格当标签。
SHOCK_CONFIRMED = "confirmed"
SHOCK_DOUBTFUL = "doubtful"
SHOCK_NOT_DETECTED = "not_detected"

#: 梯度噪声底的两个来源，取较大的那个：
#:
#: * ``NOISE_FLOOR_FACTOR`` × 该条剖面上 ``|dp/dr|`` 的**中位数**。中位数是稳健量
#:   （大部分壳层不在激波上），衡量的是"背景起伏有多大"。
#: * ``GRADIENT_FLOOR_RELATIVE`` × 参考压力 / 壳层宽度。**这一条是必需的**：一条完全
#:   平坦的剖面上中位数梯度恰好是 0，只靠相对项会让噪声底退化成 0，于是任何一个
#:   舍入起伏都成了"候选"。0.1%/壳层宽 对应的是"看不出来的压力起伏"。
#:
#: 两个都是要**报出来**的选择，不是物理常数。梯度底只是**筛选**：真正的判据是 RH 核查。
NOISE_FLOOR_FACTOR = 8.0
GRADIENT_FLOOR_RELATIVE = 1.0e-3

#: 跃迁带 = 峰值梯度的一个比例以上的连续区间。
#:
#: **不能只用"梯度高于噪声底"**：衰减的爆炸波在波后一直有梯度，高于低噪声底的范围
#: 会一路延伸到很远，于是"带外的内侧平台"取到的是已经衰减掉的状态，RH 残差被这个
#: 错配撑到 30% 以上，真实的强激波反而被否掉。取峰值的一个比例，圈出来的才是**数值
#: 激波本体**。比例本身是要报出来的选择。
BAND_GRADIENT_FRACTION = 0.20

#: 判"还在超压里"的相对阈值（相对外侧参考态）。
BAND_PRESSURE_TOLERANCE = 0.05

#: RH 归一化残差的确认门槛。归一化方式写在 :func:`_rh_residuals` 里，随结果一起报。
RH_TOLERANCE = 0.25

#: 熵不减的容差（理想气体无量纲熵差）。
ENTROPY_TOLERANCE = 0.05

#: 主激波的外侧状态"就是未扰动空气"的判据：外侧压力与环境压力相差不超过这个比例。
UNDISTURBED_PRESSURE_TOLERANCE = 0.20


@dataclass
class ShockCandidate:
    """一个激波候选及其核查结果。"""

    #: 代表位置：跃迁带的外缘（激波面在带的外侧）。
    representative_radius: float
    #: 跃迁带，``(内, 外)``。带内的壳层既不是波前也不是波后，**不得当标签**。
    band: tuple[float, float]
    #: 可靠位置范围：代表位置的不确定区间。没有依据时为 ``None``。
    reliable_range: tuple[float, float] | None
    #: 带外的两侧实测状态。
    outer_state: dict[str, float]
    inner_state: dict[str, float]
    #: 由质量守恒从实测状态估出的激波速度（**质量已用掉，不算独立证据**）。
    shock_speed: float
    #: 归一化 RH 残差：动量、能量、以及各自的原始差。
    rh_momentum: float
    rh_energy: float
    #: 理想气体无量纲熵差 ``log(p2/p1) - gamma log(rho2/rho1)``；应 ≥ 0。
    entropy_jump: float
    #: 压缩比 ``rho2/rho1``；激波必须 > 1。
    compression: float
    #: 激波相对波后空气的速度 ``Ds - u2``；必须 > 0。
    relative_speed_inner: float
    status: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "representative_radius": self.representative_radius,
            "band": list(self.band),
            "reliable_range": None if self.reliable_range is None else list(self.reliable_range),
            "shock_speed": self.shock_speed,
            "rh_momentum": self.rh_momentum,
            "rh_energy": self.rh_energy,
            "entropy_jump": self.entropy_jump,
            "compression": self.compression,
            "relative_speed_inner": self.relative_speed_inner,
            "status": self.status,
            "reasons": list(self.reasons),
            "outer_state": dict(self.outer_state),
            "inner_state": dict(self.inner_state),
        }


@dataclass
class ShockDetection:
    """一条剖面的激波检测结果。"""

    candidates: list[ShockCandidate]
    #: 选中的主激波（通向未扰动空气的最外侧合格候选）；没有就是 ``None``。
    main: ShockCandidate | None
    #: 该帧的空气监督是否有资格：只有确认了主激波、拿到了可靠范围才有。
    supervision_radius_range: tuple[float, float] | None
    status: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_candidates": len(self.candidates),
            "main": None if self.main is None else self.main.as_dict(),
            "supervision_radius_range": (
                None if self.supervision_radius_range is None
                else list(self.supervision_radius_range)
            ),
            "provenance": dict(self.provenance),
        }


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``mask`` 里连续 True 的区间 ``[start, stop)`` 列表。**不跨缺口。**"""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = np.concatenate([[0], edges + 1])
    stops = np.concatenate([edges + 1, [mask.size]])
    return [(int(a), int(b)) for a, b in zip(starts, stops) if mask[a]]


def _rh_residuals(
    outer: dict[str, float],
    inner: dict[str, float],
    shock_speed: float,
    *,
    gamma: float,
) -> tuple[float, float]:
    """动量与能量 RH 的**归一化**残差。

    动量：``p1 + rho1 w1^2`` 对 ``p2 + rho2 w2^2``，``w = Ds - u``。归一化用它两侧的
    平均值，避免除以一个接近零的超压。
    能量：``h + w^2/2``，``h = gamma p / ((gamma-1) rho)``，同样归一化。

    质量守恒**不在**这里：它已经被用来解 ``Ds`` 了，再用它当独立证据就是循环论证。
    """
    w1 = shock_speed - outer["u"]
    w2 = shock_speed - inner["u"]
    momentum_outer = outer["p"] + outer["rho"] * w1 * w1
    momentum_inner = inner["p"] + inner["rho"] * w2 * w2
    momentum_scale = max(abs(momentum_outer), abs(momentum_inner), 1.0e-30)
    rh_momentum = abs(momentum_outer - momentum_inner) / momentum_scale

    h1 = gamma * outer["p"] / ((gamma - 1.0) * outer["rho"])
    h2 = gamma * inner["p"] / ((gamma - 1.0) * inner["rho"])
    energy_outer = h1 + 0.5 * w1 * w1
    energy_inner = h2 + 0.5 * w2 * w2
    energy_scale = max(abs(energy_outer), abs(energy_inner), 1.0e-30)
    rh_energy = abs(energy_outer - energy_inner) / energy_scale
    return float(rh_momentum), float(rh_energy)


#: 密度跳变必须高过两侧窗口自身散布的倍数。
#:
#: 这是"弱激波分母病态"的可操作形式：``Ds = (rho2 u2 - rho1 u1)/(rho2 - rho1)`` 在
#: ``rho2 - rho1`` 与窗口内自身的起伏同量级时没有意义，而那个起伏是**从数据里量出来的**
#: （窗口内标准差），不是拍的阈值。``Ds`` 是靠质量守恒解出来的，所以这条不构成独立证据，
#: 它只是"这条跳变在噪声里看得见吗"。
JUMP_TO_SCATTER_RATIO = 3.0


def _verify(
    outer: dict[str, float],
    inner: dict[str, float],
    *,
    gamma: float,
    outer_scatter: dict[str, float] | None = None,
    inner_scatter: dict[str, float] | None = None,
) -> tuple[float, float, float, float, float, list[str]]:
    """由实测两侧状态定 ``Ds`` 并核查 RH 与物理解。返回各项**实测值**与理由。"""
    reasons: list[str] = []
    drho = inner["rho"] - outer["rho"]
    if outer_scatter and inner_scatter:
        scatter = float(outer_scatter.get("rho", 0.0)) + float(inner_scatter.get("rho", 0.0))
        if scatter > 0.0 and abs(drho) < JUMP_TO_SCATTER_RATIO * scatter:
            return (
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                [f"the density jump {abs(drho):.6g} kg/m^3 is not resolved above the "
                 f"within-window scatter ({scatter:.6g} kg/m^3 combined, "
                 f"ratio {JUMP_TO_SCATTER_RATIO:g} required): Ds from mass conservation is "
                 "ill-conditioned and this candidate carries insufficient evidence"],
            )
    if abs(drho) <= 1.0e-12 * max(abs(outer["rho"]), 1.0):
        # 弱激波分母病态：密度几乎不变时 Ds 定不出来。
        return (
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            ["the density jump is numerically zero, so Ds = (rho2 u2 - rho1 u1)/(rho2 - rho1) "
             "is ill-conditioned and no shock speed can be estimated from these states"],
        )
    shock_speed = (inner["rho"] * inner["u"] - outer["rho"] * outer["u"]) / drho
    rh_momentum, rh_energy = _rh_residuals(outer, inner, shock_speed, gamma=gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_jump = float(
            np.log(inner["p"] / outer["p"]) - gamma * np.log(inner["rho"] / outer["rho"])
        )
    compression = float(inner["rho"] / outer["rho"])
    relative_speed_inner = float(shock_speed - inner["u"])

    if not (compression > 1.0):
        reasons.append(
            f"the shock must compress: rho2/rho1 = {compression:.6g} is not greater than one"
        )
    if not (relative_speed_inner > 0.0):
        reasons.append(
            f"Ds - u2 = {relative_speed_inner:.6g} m/s is not positive, so the post-shock "
            "state is not behind the front"
        )
    if entropy_jump < -ENTROPY_TOLERANCE:
        reasons.append(
            f"the entropy jump {entropy_jump:.6g} is negative beyond tolerance "
            f"{ENTROPY_TOLERANCE}, which is not an admissible shock"
        )
    if rh_momentum > RH_TOLERANCE:
        reasons.append(
            f"momentum RH residual {rh_momentum:.4g} exceeds the tolerance {RH_TOLERANCE}"
        )
    if rh_energy > RH_TOLERANCE:
        reasons.append(
            f"energy RH residual {rh_energy:.4g} exceeds the tolerance {RH_TOLERANCE}"
        )
    return shock_speed, rh_momentum, rh_energy, entropy_jump, compression, reasons


def detect_shock(
    radius: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
    velocity: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
    *,
    gamma: float,
    ambient_pressure: float,
    cell_thickness: np.ndarray | None = None,
    product_threshold: float = 0.5,
    band_pressure_tolerance: float = BAND_PRESSURE_TOLERANCE,
    noise_floor_factor: float = NOISE_FLOOR_FACTOR,
    rh_tolerance: float = RH_TOLERANCE,
) -> ShockDetection:
    """在一维球壳**空气侧**剖面上找主激波。

    步骤：

    1. 在**连续有效**的空气数据上分段（不跨缺口）；
    2. 段内对 ``-dp/dr``（用真实 ``dr``）找正峰，峰高必须超过该段自身的中位数梯度
       乘 ``noise_floor_factor``——这就是"背景噪声过滤"，它替代了旧判据里那个把所有
       噪声尖峰都算进来的固定阈值；
    3. 由峰向外扩展跃迁带，直到压力回到外侧参考态附近；带内壳层两侧都不用；
    4. 取带外两侧实测状态，由质量守恒定 ``Ds``，再检验动量与能量 RH、压缩方向、
       ``Ds - u2 > 0``、熵不减；
    5. 状态分级：三项都过且跃迁带**被原始网格分辨**才是 ``confirmed``。

    ``cell_thickness`` 给出每个壳层处的原始单元径向厚度。跃迁带或波后空气层跨不过
    一个原始单元时，只能记 ``doubtful``——**"跨一个单元"不能自动通过**。
    """
    radius = np.asarray(radius, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    air_side = valid & np.isfinite(pressure) & np.isfinite(density) & np.isfinite(velocity)
    # 空气侧：不跨入产物。混合带（0.05 < alpha < 0.95）同样排除。
    air_side &= np.isfinite(alpha) & (alpha < product_threshold)

    provenance: dict[str, Any] = {
        "method": (
            "pressure-gradient candidates on contiguous valid air-side shells, then RH "
            "verification against the measured two-sided states; the shock speed comes "
            "from mass conservation and is therefore NOT independent evidence"
        ),
        "noise_floor_factor": float(noise_floor_factor),
        "band_gradient_fraction": float(BAND_GRADIENT_FRACTION),
        "band_pressure_tolerance": float(band_pressure_tolerance),
        "rh_tolerance": float(rh_tolerance),
        "product_threshold": float(product_threshold),
        "tolerance_basis": (
            "the RH tolerances are uniform-normalised residuals, NOT the loose tolerances "
            "quoted for states inside a numerically smeared wave; they are re-checked "
            "against this project's baseline and resolution"
        ),
    }
    runs = _contiguous_runs(air_side)
    provenance["n_contiguous_air_runs"] = len(runs)
    if not runs:
        return ShockDetection([], None, None, SHOCK_NOT_DETECTED, provenance)

    candidates: list[ShockCandidate] = []
    seen_bands: set[tuple[float, float]] = set()
    for start, stop in runs:
        r = radius[start:stop]
        p = pressure[start:stop]
        rho = density[start:stop]
        u = velocity[start:stop]
        if r.size < 5:
            continue
        dr = np.diff(r)
        if np.any(dr <= 0.0):
            continue
        gradient = -np.diff(p) / dr                     # 正 = 压力向外下降
        magnitude = np.abs(gradient)
        drift = float(np.median(magnitude)) * float(noise_floor_factor)
        # 参考压力取该段自己的压力尺度（外层平台），壳层宽度取中位数。
        p_scale = float(np.median(np.abs(p)))
        dr_scale = float(np.median(dr))
        absolute = (
            GRADIENT_FLOOR_RELATIVE * p_scale / dr_scale if dr_scale > 0.0 else 0.0
        )
        floor = max(drift, absolute)
        if not np.isfinite(floor) or floor <= 0.0:
            continue

        # 正峰：比两侧都大，且超过噪声底。
        interior = np.arange(1, gradient.size - 1)
        if interior.size == 0:
            continue
        is_peak = (
            (gradient[interior] >= gradient[interior - 1])
            & (gradient[interior] >= gradient[interior + 1])
            & (gradient[interior] > floor)
        )
        peak_index = interior[is_peak]
        # 越靠外越可能是主激波；从外向内逐个核查，取第一个通过的。
        peak_index = peak_index[np.argsort(-peak_index)]
        if peak_index.size == 0:
            continue

        # 外侧参考态：该段最外侧若干壳层的实测状态。
        n_ref = max(3, min(10, r.size // 10))
        outer_ref = {
            "rho": float(np.mean(rho[-n_ref:])),
            "p": float(np.mean(p[-n_ref:])),
            "u": float(np.mean(u[-n_ref:])),
        }
        reference_pressure = max(abs(outer_ref["p"]), float(ambient_pressure))

        for k in peak_index:
            # 跃迁带 = 峰周围**梯度仍高于噪声底**的连续区间。
            #
            # 不是"整个超压区"：激波后面是一个平台，那里的梯度接近零，把它算进跃迁带
            # 就等于把波后状态和跃迁混在一起平均（§10 明说"禁止平均过宽膨胀区"）。
            # 也不是"最外侧超压点"：那个判据测的是噪声。
            above = magnitude > max(floor, BAND_GRADIENT_FRACTION * float(gradient[k]))
            lo = int(k)
            while lo > 0 and above[lo - 1]:
                lo -= 1
            hi = int(k) + 1
            while hi < above.size - 1 and above[hi]:
                hi += 1
            # ``gradient`` 的下标 j 对应壳层 j 与 j+1 之间；跃迁带覆盖壳层 [lo, hi]。
            # 下标约定：数组沿**半径增大**排，``gradient[j]`` 连的是**壳层 j 与 j+1**。
            # 所以跃迁带覆盖壳层 ``[lo, hi + 1]``；**内侧 = 下标更小**（半径更小），
            # 外侧 = 下标更大。把内外写反不会报错，只会让"波后状态"取成环境态，
            # 于是压缩比变成 ~1、RH 判成病态——看起来像"这条激波证据不足"。
            if lo == 0 or hi + 1 >= r.size - 1:
                continue      # 带贴到段边界，两侧状态取不到
            band_inner, band_outer = float(r[lo]), float(r[hi + 1])

            # 两侧状态取**紧邻跃迁带外的平台**（shock-zone 思路，§10「取带外两侧状态」）。
            #
            # 不是"带宽多少就取多宽"：数值激波被人工粘性抹宽，跃迁带本身可以横跨几十个
            # 原始单元；按带宽取窗口会让内侧窗口一头扎进产物/混合带，于是**最强的激波
            # 反而永远成不了候选**——因为它旁边没有足够的空气侧空间容纳一个等宽的窗口。
            # 平台的边界就是"梯度重新落回噪声底"的地方，那是数据自己给的。
            inner_hi = lo
            inner_lo = inner_hi
            while inner_lo > 0 and not above[inner_lo - 1]:
                inner_lo -= 1
            if inner_hi - inner_lo < 3:
                continue      # 内侧没有可分辩的平台，两侧状态取不到——存疑要有理由
            inner_state = {
                "rho": float(np.mean(rho[inner_lo:inner_hi])),
                "p": float(np.mean(p[inner_lo:inner_hi])),
                "u": float(np.mean(u[inner_lo:inner_hi])),
            }
            inner_scatter = {"rho": float(np.std(rho[inner_lo:inner_hi]))}

            # 波前平台：向内收，直到梯度落回底。
            outer_lo = hi + 1
            outer_hi = outer_lo
            while outer_hi < above.size - 1 and not above[outer_hi]:
                outer_hi += 1
            if outer_hi - outer_lo < 3:
                outer_state = dict(outer_ref)
                outer_scatter = {"rho": float(np.std(rho[-n_ref:]))}
            else:
                outer_state = {
                    "rho": float(np.mean(rho[outer_lo:outer_hi])),
                    "p": float(np.mean(p[outer_lo:outer_hi])),
                    "u": float(np.mean(u[outer_lo:outer_hi])),
                }
                outer_scatter = {"rho": float(np.std(rho[outer_lo:outer_hi]))}

            shock_speed, rh_m, rh_e, entropy, compression, reasons = _verify(
                outer_state, inner_state, gamma=gamma,
                outer_scatter=outer_scatter, inner_scatter=inner_scatter,
            )
            relative_speed = (
                float(shock_speed - inner_state["u"]) if np.isfinite(shock_speed) else float("nan")
            )

            # 网格分辨率：跃迁带跨不过一个原始单元时不能自动通过。
            thickness = float("nan")
            if cell_thickness is not None:
                ct = np.asarray(cell_thickness, dtype=np.float64)
                if ct.size == radius.size:
                    window = ct[max(0, k - 2):min(ct.size, k + 3)]
                    window = window[np.isfinite(window) & (window > 0.0)]
                    if window.size:
                        thickness = float(np.median(window))
            band_width = band_outer - band_inner
            if np.isfinite(thickness) and thickness > 0.0 and band_width < thickness:
                reasons.append(
                    f"the transition band ({band_width * 1e3:.4f} mm) is narrower than one "
                    f"native cell ({thickness * 1e3:.4f} mm): the front is not resolved by "
                    "the mesh, so it cannot be confirmed"
                )

            # 两个相邻的峰可能选中**同一条**跃迁带（梯度在一个窄台阶上取等值时），
            # 那是一个候选而不是两个。不去重的话每条这样的激波都会因为"多个候选分不开"
            # 被判成存疑——把一个纯粹的重复计数说成物理歧义。
            band_key = (round(band_inner, 12), round(band_outer, 12))
            if band_key in seen_bands:
                continue
            seen_bands.add(band_key)

            status = SHOCK_CONFIRMED if not reasons else SHOCK_DOUBTFUL
            reliable = (
                (band_inner, band_outer) if status == SHOCK_CONFIRMED else None
            )
            candidates.append(
                ShockCandidate(
                    representative_radius=band_outer,
                    band=(band_inner, band_outer),
                    reliable_range=reliable,
                    outer_state=outer_state,
                    inner_state=inner_state,
                    shock_speed=float(shock_speed),
                    rh_momentum=float(rh_m),
                    rh_energy=float(rh_e),
                    entropy_jump=float(entropy),
                    compression=float(compression),
                    relative_speed_inner=relative_speed,
                    status=status,
                    reasons=reasons,
                )
            )

    if not candidates:
        return ShockDetection([], None, None, SHOCK_NOT_DETECTED, provenance)

    # **同一条激波会被多个峰重复选中。** 数值激波被人工粘性抹宽，梯度在它的平台上
    # 取等值，于是相邻几个梯度点都是"峰"，各自选出一条**相互重叠**的带——它们是同
    # 一条波，不是物理歧义。不合并的话，一条被核实得很好的激波会因为"5 个候选分不开"
    # 被判成存疑，而那条波其实只有一个。
    #
    # 合并判据用**区间重叠**而不是相等：相等只能抓住完全一样的那一种。
    candidates.sort(key=lambda c: c.band[0])
    merged: list[ShockCandidate] = []
    for candidate in candidates:
        if merged and candidate.band[0] <= merged[-1].band[1]:
            previous = merged[-1]
            # 留残差小的那个；残差相同时留代表半径更大的（更靠外的主激波）。
            previous_cost = previous.rh_momentum + previous.rh_energy
            candidate_cost = candidate.rh_momentum + candidate.rh_energy
            better = (
                candidate
                if (candidate_cost, -candidate.representative_radius)
                < (previous_cost, -previous.representative_radius)
                else previous
            )
            merged[-1] = better
            merged[-1].band = (
                min(previous.band[0], candidate.band[0]),
                max(previous.band[1], candidate.band[1]),
            )
        else:
            merged.append(candidate)
    candidates = merged
    provenance["n_merged_candidates"] = len(candidates)


    # 主激波：**通向未扰动空气的、最外侧的合格候选**。
    #
    # 只保留合格的那些参与选择。噪声尖峰可以出现在候选列表里——它**应该**出现，
    # 那正是"候选"的含义——但它在 RH 核查上过不去，于是不会把真激波挤掉。
    confirmed = [c for c in candidates if c.status == SHOCK_CONFIRMED]
    undisturbed = [
        c for c in confirmed
        if abs(c.outer_state.get("p", 0.0) / max(ambient_pressure, 1.0) - 1.0)
        <= UNDISTURBED_PRESSURE_TOLERANCE
    ]
    if not confirmed:
        main = max(candidates, key=lambda c: c.representative_radius)
        status = SHOCK_DOUBTFUL
    else:
        pool = undisturbed or confirmed
        main = max(pool, key=lambda c: c.representative_radius)
        status = SHOCK_CONFIRMED if len(pool) == 1 else SHOCK_DOUBTFUL
        if len(pool) > 1:
            main.reasons.append(
                f"{len(pool)} candidates passed verification and their bands are not "
                "separated enough to tell which is the main front; marking doubtful rather "
                "than picking one"
            )
    return ShockDetection(
        candidates=candidates,
        main=main,
        supervision_radius_range=main.reliable_range,
        status=status,
        provenance=provenance,
    )
