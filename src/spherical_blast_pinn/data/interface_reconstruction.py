"""接触面的逐帧几何重构（提示词 §9）。

借鉴 isoAdvector 的体积分数相容几何思想，做一个**可验证的球对称 PLIC**：在已知
单元拓扑、节点位置、邻域与产物体积分数的前提下，用局部界面切分再现该单元的相
体积，再由重构面形成球对称代表位置。

**只做读取后的逐帧重构，不引入界面时间推进**——这里没有一步是把界面从上一帧
推进到下一帧的。

球对称 PLIC 的切分是闭式的。单元是球壳 ``[a, b]``，产物在内侧，占体积分数
``alpha``。设界面在半径 ``r`` 处，则

.. math::
    \\frac{r^3 - a^3}{b^3 - a^3} = \\alpha
    \\quad\\Longrightarrow\\quad
    r = \\left(a^3 + \\alpha\\,(b^3 - a^3)\\right)^{1/3}

所以在这个几何里，"局部界面切分在声明容差内再现该单元相体积"是可以**逐单元核对**
的：把 ``r`` 代回去算出的体积分数与声明的 ``alpha`` 之差就是重构误差，它随结果
一起报出来。

**这个数在球对称闭式解下必然是浮点尾数量级**，这一点必须说清楚，不能让它读起来
像一项物理校验：``r`` 是由 ``alpha`` 反解出来的，代回去当然等于 ``alpha``。它守住
的是**公式实现**（求积写错、指数写错、量纲写错），不是"数据与网格是否自洽"——
后者从 ``alpha`` 一个量里根本看不出来，因为 ``alpha`` 就是输入本身。数据是否自洽
要靠 §15 的数据契约与 §10 的检测资格去管，不能在这里假装查过。

真正能在这里查出来的是**结构**问题，它们各自走"未定位 + 原因"这条路：混合带不
连续、混合带贴着边界没有邻域、邻域的相不是预期的相、以及拓扑与体积分数长度不
一致（这一条直接抛错，因为那些数根本不能放在一起用）。

用法边界，逐条对应 §9：

* **不给精确半径的场合绝不给**。定位不了就返回区间 ``(band_lo, band_hi)`` 与
  原因；区间是"界面在这个混合带里的某处"，不是"界面在带中心"。
* ``alpha = 0.5`` 的壳平均穿越只作**辅助对照**（``shell_crossing_radius``），
  不再充当最终几何重构。
* 体积等效半径 ``volume_equivalent_radius`` 用当前相体积与对称份额折算，八分之一
  模型要乘 8——它只是交叉检查，**不自动替代**几何前沿。
* 重构出来的是**亚单元**位置时，``sub_cell_position`` 会置位并写进报告：它没有
  让原始数据多出任何分辨率，原始分辨率仍是 ``native_resolution``。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: 定位状态。
STATUS_RECONSTRUCTED = "reconstructed"      # 在混合带里定位到界面
STATUS_ON_SHARED_FACE = "on_shared_face"    # 相邻纯相单元的公共面上
STATUS_UNRESOLVED = "unresolved"            # 定位不了，只给区间与原因

#: 默认的纯相阈值。与 :mod:`shell_average` 的 ``PURE_*_THRESHOLD`` 同源。
PURE_PRODUCT_THRESHOLD = 0.99
PURE_AIR_THRESHOLD = 0.01

#: 重构求积默认的相对容差。球对称闭式解应当精确到浮点尾数。
DEFAULT_VOLUME_TOLERANCE = 1.0e-9


class ReconstructionError(ValueError):
    """输入不成立（不是"定位不了"，是"这些数根本不能拿来重构"）。"""


@dataclass(frozen=True)
class InterfaceReconstruction:
    """一帧的接触面几何重构结果。

    ``radius`` 为 ``nan`` 时看 ``status`` 与 ``reason``；``interval`` **任何状态
    下都有效**，它是不依赖定位是否成功的外括号。
    """

    status: str
    radius: float
    interval: tuple[float, float]
    reason: str
    #: 混合带（``0 < alpha < 1`` 的单元）的下标与该带的径向范围。
    mixing_cells: tuple[int, ...]
    mixing_band: tuple[float, float]
    #: 原始数据的分辨率：单元宽度的中位数。重构不会让它变小。
    native_resolution: float
    #: 位置不确定性。混合带定位时是带宽的一半（界面只保证在带内）；公共面定位
    #: 时是 0——那是数据本身的一个面，不是插出来的亚单元位置。
    position_uncertainty: float
    #: 逐单元把重构半径代回去得到的最大体积分数误差。球对称闭式解下这是**浮点
    #: 尾数**量级；它守的是公式实现，不是"数据与网格自洽"（见模块文档）。
    max_volume_fraction_error: float
    #: 代表位置是否落在单元内部（亚单元）。真时**不得**据此宣称获得了额外分辨率。
    sub_cell_position: bool
    #: 混合带两侧是否都有邻域、且邻域的相是预期的相。
    neighbourhood_complete: bool
    #: 方向离散。径向平均过的数据没有方向信息，此时是 ``nan``。
    direction_spread: float
    direction_note: str
    symmetry_multiplier: float
    #: 由相体积与对称份额折出的等效球半径。交叉检查用。
    volume_equivalent_radius: float
    #: ``alpha = 0.5`` 壳平均穿越。辅助对照用。
    shell_crossing_radius: float
    #: 几何位置与壳平均穿越的相对差。两者差得多时说明该帧的界面并不干净。
    cross_check_relative_difference: float
    #: §9 要求分别输出的三项结论，这里作为独立的键给出，不混进 ``radius``。
    detection: str = STATUS_UNRESOLVED
    supervision_eligible: bool = False
    supervision_reason: str = ""
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "radius_m": self.radius,
            "interval_m": list(self.interval),
            "reason": self.reason,
            "mixing_cells": list(self.mixing_cells),
            "mixing_band_m": list(self.mixing_band),
            "native_resolution_m": self.native_resolution,
            "position_uncertainty_m": self.position_uncertainty,
            "max_volume_fraction_error": self.max_volume_fraction_error,
            "sub_cell_position": bool(self.sub_cell_position),
            "neighbourhood_complete": bool(self.neighbourhood_complete),
            "direction_spread": self.direction_spread,
            "direction_note": self.direction_note,
            "symmetry_multiplier": self.symmetry_multiplier,
            "volume_equivalent_radius_m": self.volume_equivalent_radius,
            "shell_crossing_radius_m": self.shell_crossing_radius,
            "cross_check_relative_difference": self.cross_check_relative_difference,
            "detection": self.detection,
            "supervision_eligible": bool(self.supervision_eligible),
            "supervision_reason": self.supervision_reason,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# 几何
# --------------------------------------------------------------------------


def shell_volume(inner: float, outer: float) -> float:
    """球壳体积（整球口径）。"""
    return (4.0 / 3.0) * math.pi * (float(outer) ** 3 - float(inner) ** 3)


def _cell_volumes(edges: np.ndarray) -> np.ndarray:
    return (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def plic_radius(inner: float, outer: float, alpha: float) -> float:
    """球对称 PLIC 的闭式切分：产物占内侧 ``alpha`` 时的界面半径。"""
    a3 = float(inner) ** 3
    b3 = float(outer) ** 3
    return float((a3 + float(alpha) * (b3 - a3)) ** (1.0 / 3.0))


def fraction_from_radius(inner: float, outer: float, radius: float) -> float:
    """``plic_radius`` 的逆：把半径代回去算体积分数。核对接得上用。"""
    a3 = float(inner) ** 3
    b3 = float(outer) ** 3
    if b3 <= a3:
        return float("nan")
    return float((float(radius) ** 3 - a3) / (b3 - a3))


def volume_equivalent_radius(
    edges: Sequence[float], product_fraction: Sequence[float], *,
    symmetry_multiplier: float = 1.0,
) -> float:
    """由相体积与对称份额折出的等效球半径（§9 的辅助量）。

    ``symmetry_multiplier`` 是**整球相对被建模那一块**的倍数：八分之一模型取 8，
    完整球取 1。少了它，八分之一模型的产物体积会被当成整球的，等效半径小
    ``8^{1/3} ≈ 2`` 倍——而那个数看起来仍然像一个半径。
    """
    edges_array = np.asarray(edges, dtype=np.float64)
    alpha = np.asarray(product_fraction, dtype=np.float64)
    if alpha.size != edges_array.size - 1:
        raise ReconstructionError(
            f"{alpha.size} fractions for {edges_array.size - 1} cells"
        )
    multiplier = float(symmetry_multiplier)
    if not (multiplier > 0.0):
        raise ReconstructionError(f"symmetry multiplier must be positive, got {multiplier}")
    total = float(np.sum(np.clip(alpha, 0.0, 1.0) * _cell_volumes(edges_array)))
    total *= multiplier
    if total <= 0.0:
        return 0.0
    return float((3.0 * total / (4.0 * math.pi)) ** (1.0 / 3.0))


def shell_crossing_radius(
    edges: Sequence[float], product_fraction: Sequence[float], *, level: float = 0.5
) -> float:
    """``alpha`` 剖面上第一次降到 ``level`` 以下的位置，线性内插。

    **辅助对照**（§9）：它是"壳平均穿越"，不是几何重构。用它当最终位置会系统性地
    偏，因为壳平均已经把界面抹平了一个单元那么宽。
    """
    edges_array = np.asarray(edges, dtype=np.float64)
    alpha = np.asarray(product_fraction, dtype=np.float64)
    if alpha.size != edges_array.size - 1 or alpha.size == 0:
        return float("nan")
    centres = 0.5 * (edges_array[:-1] + edges_array[1:])
    descending = alpha >= float(level)
    for index in range(alpha.size - 1):
        if descending[index] and not descending[index + 1]:
            lo, hi = alpha[index], alpha[index + 1]
            if hi == lo:
                return float(centres[index])
            weight = (float(level) - hi) / (lo - hi)
            return float(centres[index] + weight * (centres[index + 1] - centres[index]))
    return float("nan")


# --------------------------------------------------------------------------
# 重构
# --------------------------------------------------------------------------


def reconstruct_interface(
    edges: Sequence[float],
    product_fraction: Sequence[float],
    *,
    symmetry_multiplier: float = 1.0,
    pure_product_threshold: float = PURE_PRODUCT_THRESHOLD,
    pure_air_threshold: float = PURE_AIR_THRESHOLD,
    volume_tolerance: float = DEFAULT_VOLUME_TOLERANCE,
    direction_radii: Sequence[float] | None = None,
) -> InterfaceReconstruction:
    """从单元边与产物体积分数重构一帧的接触面位置。

    Parameters
    ----------
    edges:
        单元边（节点位置），长度 ``n + 1``，严格递增。
    product_fraction:
        每个单元的产物体积分数，长度 ``n``，取值在 ``[0, 1]``。
    symmetry_multiplier:
        整球相对被建模那一块的倍数（八分之一模型取 8）。
    direction_radii:
        若调用方另有按方向的估计（例如从三维体积分数场按方向切出来的半径），
        传入即可得到方向离散；径向平均过的数据没有这个信息，留 ``None``。
    """
    edges_array = np.asarray(edges, dtype=np.float64).reshape(-1)
    alpha = np.asarray(product_fraction, dtype=np.float64).reshape(-1)

    if edges_array.size < 2:
        raise ReconstructionError("need at least two cell edges")
    if alpha.size != edges_array.size - 1:
        raise ReconstructionError(
            f"got {alpha.size} volume fractions for {edges_array.size - 1} cells; "
            "the fractions and the topology do not belong together"
        )
    if not np.all(np.isfinite(edges_array)):
        raise ReconstructionError("cell edges contain non-finite values")
    if np.any(np.diff(edges_array) <= 0.0):
        raise ReconstructionError("cell edges must be strictly increasing")
    if np.any(alpha[np.isfinite(alpha)] < -1.0e-12) or np.any(
        alpha[np.isfinite(alpha)] > 1.0 + 1.0e-12
    ):
        raise ReconstructionError("volume fractions must lie in [0, 1]")

    widths = np.diff(edges_array)
    native = float(np.median(widths))
    notes: list[str] = []

    # -- 空壳层（非有限的体积分数）----------------------------------------
    # 真实数据里这不是异常输入：壳层里没有单元质心时，``mixture_fraction_profile``
    # 返回 ``nan``——那一层根本没有相信息。它既不是产物也不是空气，所以它**不是**
    # 一个可以参与重构的单元；把它当 0 处理会凭空造出一个"空气侧"，把它当 1 则
    # 凭空造出产物。
    #
    # 处理方式：只在**有限**的一段上重构，并且要求这一段的内部没有空洞。空洞如果
    # 夹在产物到空气的过渡里，界面就可能藏在它后面，此时返回"未定位"并说明原因。
    finite = np.isfinite(alpha)
    if not np.any(finite):
        return _unresolved(
            edges_array, alpha, np.zeros(0, dtype=np.int64),
            (float("nan"), float("nan")), native, 0.0,
            "no cell in this frame carries a finite product volume fraction",
            notes, symmetry_multiplier, direction_radii,
        )

    first_finite = int(np.flatnonzero(finite)[0])
    last_finite = int(np.flatnonzero(finite)[-1])
    holes = int(np.count_nonzero(~finite[first_finite:last_finite + 1]))
    if holes:
        return _unresolved(
            edges_array, alpha, np.zeros(0, dtype=np.int64),
            (float(edges_array[first_finite]), float(edges_array[last_finite + 1])),
            native, 0.0,
            f"{holes} cells inside the product-to-air span carry no phase fraction "
            "(empty shells), so the mixing band cannot be bracketed; the interface may "
            "be behind them",
            notes, symmetry_multiplier, direction_radii,
        )

    # 只在有限的一段上重构，下标最后再平移回原网格。
    offset = first_finite
    alpha = np.clip(alpha[first_finite:last_finite + 1], 0.0, 1.0)
    edges_array = edges_array[first_finite:last_finite + 2]

    # -- 逐单元核对：重构半径代回去能不能再现该单元的相体积 ---------------
    cut = np.flatnonzero((alpha > pure_air_threshold) & (alpha < pure_product_threshold))
    max_error = 0.0
    for index in cut:
        radius = plic_radius(edges_array[index], edges_array[index + 1], alpha[index])
        reproduced = fraction_from_radius(
            edges_array[index], edges_array[index + 1], radius
        )
        max_error = max(max_error, abs(reproduced - float(alpha[index])))

    if cut.size and max_error > volume_tolerance:
        notes.append(
            "the closed-form cut does not reproduce the declared phase volume to "
            f"round-off ({max_error:.3e} > {volume_tolerance:.3e}); this is a numerical "
            "consistency failure in the reconstruction itself, not a physical verdict "
            "on the data, so no position is reported"
        )

    mixing_band = (
        (float(edges_array[cut[0]]), float(edges_array[cut[-1] + 1]))
        if cut.size else (float("nan"), float("nan"))
    )

    if cut.size and max_error > volume_tolerance:
        return _unresolved(
            edges_array, alpha, cut, mixing_band, native, max_error,
            "the reconstruction does not reproduce the phase volume", notes,
            symmetry_multiplier, direction_radii, index_offset=offset,
        )

    # -- 没有混合带：相邻纯相单元，界面恰在公共面上 -----------------------
    if cut.size == 0:
        transition = np.flatnonzero(
            (alpha >= pure_product_threshold)
        )
        transition = transition[transition + 1 < alpha.size]
        face = None
        for index in transition:
            if alpha[index + 1] <= pure_air_threshold:
                face = float(edges_array[index + 1])
                break
        if face is None:
            return _unresolved(
                edges_array, alpha, cut, mixing_band, native, max_error,
                "no cell carries an intermediate fraction and no product/air pair "
                "shares a face, so the interface is not located by this frame",
                notes, symmetry_multiplier, direction_radii, index_offset=offset,
            )
        notes.append(
            "the interface sits on the shared face of two pure cells; that is a "
            "genuine data feature, not a sub-cell interpolation"
        )
        return _finish(
            status=STATUS_ON_SHARED_FACE,
            radius=face,
            interval=(face, face),
            reason="product and air are adjacent pure cells; the interface is their shared face",
            edges_array=edges_array, alpha=alpha, cut=cut, mixing_band=(face, face),
            native=native, max_error=max_error, sub_cell=False,
            neighbourhood_complete=True, notes=notes,
            symmetry_multiplier=symmetry_multiplier, direction_radii=direction_radii,
            index_offset=offset,
        )

    # -- 有混合带：要求带是连续的一段，且两侧邻域是预期的相 ---------------
    contiguous = bool(np.all(np.diff(cut) == 1))
    first, last = int(cut[0]), int(cut[-1])
    has_inner_neighbour = first - 1 >= 0
    has_outer_neighbour = last + 1 <= alpha.size - 1
    inner_is_product = has_inner_neighbour and alpha[first - 1] >= pure_product_threshold
    outer_is_air = has_outer_neighbour and alpha[last + 1] <= pure_air_threshold
    neighbourhood_ok = (
        contiguous and has_inner_neighbour and has_outer_neighbour
        and inner_is_product and outer_is_air
    )
    if not neighbourhood_ok:
        problems = []
        if not contiguous:
            problems.append("the intermediate cells are not one contiguous run")
        if not has_inner_neighbour:
            problems.append("the mixing band has no cell on its inner side")
        if not has_outer_neighbour:
            problems.append("the mixing band has no cell on its outer side")
        if has_inner_neighbour and not inner_is_product:
            problems.append("the cell inside the band is not pure product")
        if has_outer_neighbour and not outer_is_air:
            problems.append("the cell outside the band is not pure air")
        return _unresolved(
            edges_array, alpha, cut, mixing_band, native, max_error,
            "; ".join(problems) + " (section 9: report an interval and a reason rather "
            "than forcing an exact radius)",
            notes, symmetry_multiplier, direction_radii,
            neighbourhood_complete=False, index_offset=offset,
        )

    # 代表位置：一整段混合带里，用一个球面再现带内的产物总体积。
    band_lo, band_hi = mixing_band
    band_product = float(np.sum(alpha[first:last + 1] * _cell_volumes(edges_array)[first:last + 1]))
    band_total = shell_volume(band_lo, band_hi)
    fraction_in_band = min(max(band_product / band_total, 0.0), 1.0) if band_total > 0 else 0.0
    representative = plic_radius(band_lo, band_hi, fraction_in_band)
    sub_cell = not any(
        math.isclose(representative, float(edge), rel_tol=0.0, abs_tol=1.0e-12)
        for edge in edges_array
    )
    if sub_cell:
        notes.append(
            "the representative position is inside a cell; it does not give the raw data "
            "any resolution beyond the native spacing"
        )

    return _finish(
        status=STATUS_RECONSTRUCTED,
        radius=representative,
        interval=(band_lo, band_hi),
        reason="the interface lies inside the mixing band",
        edges_array=edges_array, alpha=alpha, cut=cut, mixing_band=mixing_band,
        native=native, max_error=max_error, sub_cell=sub_cell,
        neighbourhood_complete=True, notes=notes,
        symmetry_multiplier=symmetry_multiplier, direction_radii=direction_radii,
        index_offset=offset,
    )


# --------------------------------------------------------------------------
# 组装结果
# --------------------------------------------------------------------------


def _direction_spread(direction_radii: Sequence[float] | None) -> tuple[float, str]:
    if direction_radii is None:
        return float("nan"), (
            "the extraction is radially averaged, so it carries no directional "
            "information; no direction spread is reported rather than a made-up zero"
        )
    values = np.asarray(direction_radii, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), "fewer than two usable directions"
    mean = float(np.mean(values))
    if mean == 0.0:
        return float("nan"), "the mean direction radius is zero"
    return float((values.max() - values.min()) / mean), (
        "relative spread of the per-direction reconstructed radii"
    )


def _finish(
    *,
    status: str,
    radius: float,
    interval: tuple[float, float],
    reason: str,
    edges_array: np.ndarray,
    alpha: np.ndarray,
    cut: np.ndarray,
    mixing_band: tuple[float, float],
    native: float,
    max_error: float,
    sub_cell: bool,
    neighbourhood_complete: bool,
    notes: list[str],
    symmetry_multiplier: float,
    direction_radii: Sequence[float] | None,
    index_offset: int = 0,
) -> InterfaceReconstruction:
    band_width = (
        float(mixing_band[1] - mixing_band[0]) if np.isfinite(mixing_band[0]) else 0.0
    )
    # 公共面上的定位是精确的（那是数据里的一个面）；混合带里的定位只知道"在带内"。
    uncertainty = 0.0 if status == STATUS_ON_SHARED_FACE else 0.5 * band_width
    spread, spread_note = _direction_spread(direction_radii)

    equivalent = volume_equivalent_radius(
        edges_array, alpha, symmetry_multiplier=symmetry_multiplier
    )
    crossing = shell_crossing_radius(edges_array, alpha)
    if math.isfinite(crossing) and crossing > 0.0 and math.isfinite(radius):
        cross_difference = abs(radius - crossing) / max(radius, 1.0e-30)
    else:
        cross_difference = float("nan")

    eligible = status == STATUS_RECONSTRUCTED and neighbourhood_complete
    eligibility_reason = (
        "located inside a mixing band with both neighbours confirmed"
        if eligible else reason
    )
    if status == STATUS_ON_SHARED_FACE:
        eligible = True
        eligibility_reason = "located exactly on a shared face of two pure cells"

    return InterfaceReconstruction(
        status=status,
        radius=float(radius),
        interval=(float(interval[0]), float(interval[1])),
        reason=reason,
        mixing_cells=tuple(int(index) + int(index_offset) for index in cut),
        mixing_band=(float(mixing_band[0]), float(mixing_band[1])),
        native_resolution=native,
        position_uncertainty=float(uncertainty),
        max_volume_fraction_error=float(max_error),
        sub_cell_position=bool(sub_cell),
        neighbourhood_complete=bool(neighbourhood_complete),
        direction_spread=spread,
        direction_note=spread_note,
        symmetry_multiplier=float(symmetry_multiplier),
        volume_equivalent_radius=float(equivalent),
        shell_crossing_radius=float(crossing),
        cross_check_relative_difference=float(cross_difference),
        detection=status,
        supervision_eligible=bool(eligible),
        supervision_reason=eligibility_reason,
        notes=tuple(notes),
    )


def _unresolved(
    edges_array: np.ndarray,
    alpha: np.ndarray,
    cut: np.ndarray,
    mixing_band: tuple[float, float],
    native: float,
    max_error: float,
    reason: str,
    notes: list[str],
    symmetry_multiplier: float,
    direction_radii: Sequence[float] | None,
    *,
    neighbourhood_complete: bool = False,
    index_offset: int = 0,
) -> InterfaceReconstruction:
    """定位不了：给区间与原因，**不给**半径。"""
    if not np.isfinite(mixing_band[0]):
        # 连混合带都没有，外括号退化为整个网格
        band = (float(edges_array[0]), float(edges_array[-1]))
    else:
        band = mixing_band
    return _finish(
        status=STATUS_UNRESOLVED,
        radius=float("nan"),
        interval=band,
        reason=reason,
        edges_array=edges_array, alpha=alpha, cut=cut, mixing_band=band,
        native=native, max_error=max_error, sub_cell=False,
        neighbourhood_complete=neighbourhood_complete, notes=list(notes),
        symmetry_multiplier=symmetry_multiplier, direction_radii=direction_radii,
        index_offset=index_offset,
    )


__all__ = [
    "DEFAULT_VOLUME_TOLERANCE",
    "PURE_AIR_THRESHOLD",
    "PURE_PRODUCT_THRESHOLD",
    "STATUS_ON_SHARED_FACE",
    "STATUS_RECONSTRUCTED",
    "STATUS_UNRESOLVED",
    "InterfaceReconstruction",
    "ReconstructionError",
    "fraction_from_radius",
    "plic_radius",
    "reconstruct_interface",
    "shell_crossing_radius",
    "shell_volume",
    "volume_equivalent_radius",
]
