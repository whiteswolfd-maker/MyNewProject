"""Three-dimensional to one-dimensional spherical shell averaging.

球壳是**后处理统计区域**，不重划 DYNA 网格（§8）。一个 3D ALE 结果被约化成径向剖面：

.. math::

    \\bar q_m(r_k) = \\frac{\\sum_{j} \\alpha_{m,j} V_{jk}\\, q_j}
                          {\\sum_{j} \\alpha_{m,j} V_{jk}},

其中 ``V_{jk}`` 是单元 ``j`` 与壳层 ``k`` 的**重叠体积**（不再是"质心落在壳层 k 里的
单元的整格体积"）。

三条不能混的规则：

* **加权按字段分**：``rho``、``p`` 用体积权重；``u``、``e`` 用**质量权重**。能量与速度
  是强度量，按体积平均会被稀释，按质量平均才是它们的守恒平均。
* **重叠体积不增加分辨率。** 它只把同一个单元的信息在壳层之间重新分配。单元内部的
  梯度仍然看不见，"壳层 0.12 mm" 不等于"分辨率 0.12 mm"。
* **重构假设必须写明**：单元内部的体积在 ``[r_in, r_out]`` 上按圆台（``dV/dr ~ r^2``）
  分布；混合单元按体积分数线性分摊，隐含单元内均匀混合。两条都写在
  ``assignment_note`` 里随数据一起走。

纯介质阈值固定，从不调整：

* ``alpha_p >= 0.95`` —— 纯产物，可作产物监督；
* ``alpha_p <= 0.05`` —— 纯空气，可作空气监督；
* ``0.05 < alpha_p < 0.95`` —— ALE 混合单元，用来定位界面带，**两种介质的场量标签
  都不用**。

模块同时给出球对称性审计：各场量的角向离散度、径向/切向速度比、前沿半径的方向散布。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .raw_reader import RawFrame

#: Fixed pure-medium thresholds (task specification, section 九.2).
PURE_PRODUCT_THRESHOLD = 0.95
PURE_AIR_THRESHOLD = 0.05

PRODUCT = "product"
AIR = "air"


class ShellAverageError(ValueError):
    """Raised when shells cannot be built or averaged."""


# --------------------------------------------------------------------------
# Shell grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellGrid:
    """A radial binning, fine near the charge and coarse in the far field."""

    edges: np.ndarray
    #: 计算域相对整球的几何份额的**倒数**：八分之一球时是 8，完整球是 1。
    #:
    #: 壳层的**几何体积**必须按**被建模的那一块**算，不能按整球算：把整球体积当分母，
    #: 覆盖率会系统性地小 8 倍（实测八分之一球模型上正好是 0.125），而"覆盖率 0.125"
    #: 读起来像"大部分没覆盖"，其实那一层是满的。
    symmetry_multiplier: float = 1.0

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ShellAverageError("shell edges must be a 1-D array of at least two values")
        if np.any(np.diff(edges) <= 0.0):
            raise ShellAverageError("shell edges must be strictly increasing")
        if not (float(self.symmetry_multiplier) > 0.0):
            raise ShellAverageError("the symmetry multiplier must be positive")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "symmetry_multiplier", float(self.symmetry_multiplier))

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def n_shells(self) -> int:
        return int(self.edges.size - 1)

    @property
    def inner_radius(self) -> float:
        return float(self.edges[0])

    @property
    def outer_radius(self) -> float:
        return float(self.edges[-1])

    @classmethod
    def linear(cls, inner: float, outer: float, n_shells: int) -> "ShellGrid":
        if outer <= inner:
            raise ShellAverageError(f"outer radius {outer} must exceed inner radius {inner}")
        return cls(np.linspace(inner, outer, int(n_shells) + 1))

    @classmethod
    def graded(
        cls,
        inner: float,
        outer: float,
        n_shells: int,
        *,
        anchor: float,
        near_fraction: float = 0.5,
        near_span: float = 3.0,
        max_growth: float = 1.25,
    ) -> "ShellGrid":
        """Two-piece grid: uniform near the charge, geometric in the far field.

        Blast meshes are graded this way for a reason — the interface, the
        shock and the whole positive phase live within a few charge radii,
        while the far field only needs to carry the decaying tail.

        Parameters
        ----------
        anchor:
            The charge radius.  The near region spans ``[inner, near_span *
            anchor]`` and gets ``near_fraction`` of the shells.
        near_fraction:
            Fraction of shells spent in the near region.
        near_span:
            Near-region extent, in charge radii.
        max_growth:
            Upper bound on the ratio between consecutive far-field shells.
        """
        inner = float(inner)
        outer = float(outer)
        n_shells = int(n_shells)
        if outer <= inner or n_shells < 8:
            raise ShellAverageError("need outer > inner and at least eight shells")
        if anchor <= 0.0:
            raise ShellAverageError(f"anchor (charge radius) must be positive, got {anchor}")
        if not (0.0 < near_fraction < 1.0):
            raise ShellAverageError("near_fraction must lie strictly between 0 and 1")

        transition = min(inner + float(near_span) * anchor, inner + 0.5 * (outer - inner))
        n_near = max(2, int(round(near_fraction * n_shells)))
        n_far = n_shells - n_near

        near_edges = np.linspace(inner, transition, n_near + 1)
        if n_far <= 0:
            return cls(near_edges)

        near_width = float(near_edges[-1] - near_edges[-2])
        far_span = outer - transition

        # Linear width growth ``w_k = w_0 (1 + g k)``.  The sum is
        # ``w_0 (n + g n (n-1) / 2)``, which is solved for ``g`` directly.  This
        # always has a positive solution when the far span is large enough, and
        # never collapses the way a fixed-ratio geometric series does.
        index = np.arange(n_far, dtype=np.float64)
        if n_far > 1 and far_span > near_width * n_far:
            growth = (far_span / near_width - n_far) * 2.0 / (n_far * (n_far - 1))
        else:
            growth = 0.0
        growth = float(np.clip(growth, 0.0, max(float(max_growth), 1.0) - 1.0))
        far_widths = near_width * (1.0 + growth * index)
        # Renormalise so the far widths tile the span exactly, whatever the cap
        # did to the growth factor.
        far_widths *= far_span / far_widths.sum()
        far_edges = transition + np.concatenate([[0.0], np.cumsum(far_widths)])
        far_edges[-1] = outer
        edges = np.concatenate([near_edges[:-1], far_edges])
        edges = np.maximum.accumulate(edges)
        # Enforce strict monotonicity against floating-point collapse.
        minimum_width = 1.0e-12 * max(outer - inner, 1.0)
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1] + minimum_width:
                edges[i] = edges[i - 1] + minimum_width
        return cls(edges)

    @classmethod
    def from_charge(
        cls,
        charge_radius: float,
        outer_radius: float,
        n_shells: int,
        *,
        mode: str = "graded",
        near_fraction: float = 0.5,
        near_span: float = 3.0,
        max_growth: float = 1.25,
    ) -> "ShellGrid":
        """Grid spanning the whole domain, anchored on the charge radius."""
        if mode == "linear":
            return cls.linear(0.0, outer_radius, n_shells)
        if mode == "graded":
            return cls.graded(
                0.0,
                outer_radius,
                n_shells,
                anchor=charge_radius,
                near_fraction=near_fraction,
                near_span=near_span,
                max_growth=max_growth,
            )
        raise ShellAverageError(f"unknown shell grid mode {mode!r}")

    def indices(self, radius: np.ndarray) -> np.ndarray:
        """Shell index per cell; ``-1`` for cells outside the grid."""
        radius = np.asarray(radius, dtype=np.float64)
        index = np.digitize(radius, self.edges) - 1
        return np.where(
            (index >= 0) & (index < self.n_shells) & (radius >= self.edges[0]) & (radius < self.edges[-1]),
            index,
            -1,
        )

    def bin_index(self, value: np.ndarray, *, inclusive_upper: bool = True) -> np.ndarray:
        """Cell index for a monotone quantity (used for the front radius)."""
        value = np.asarray(value, dtype=np.float64)
        index = np.digitize(value, self.edges)
        if inclusive_upper:
            index = np.clip(index - 1, -1, self.n_shells - 1)
        else:
            index = np.clip(index, -1, self.n_shells - 1)
        return index

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_shells": self.n_shells,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "min_width": float(self.widths.min()),
            "max_width": float(self.widths.max()),
        }


# --------------------------------------------------------------------------
# Shell averaging
# --------------------------------------------------------------------------


@dataclass
class ShellProfile:
    """Radially averaged fields for one medium at one time."""

    medium: str
    time: float
    grid: ShellGrid
    radius: np.ndarray
    weight: np.ndarray
    count: np.ndarray
    density: np.ndarray
    radial_velocity: np.ndarray
    specific_internal_energy: np.ndarray
    pressure: np.ndarray
    volume_fraction: np.ndarray
    validity: np.ndarray
    dispersion: dict[str, np.ndarray] = field(default_factory=dict)
    #: 该壳层里这一侧介质真正占据的体积，以及壳层自身的几何体积（§8 要求保存）。
    covered_volume: np.ndarray | None = None
    shell_volume: np.ndarray | None = None
    #: ``covered_volume / shell_volume``。壳层平均只在覆盖到的部分上成立，
    #: 覆盖率说明它成立在多大范围上。
    coverage: np.ndarray | None = None
    #: 分配方法（重叠体积 / 质心分箱）与它的说明。**两种方法不能互相冒充**：
    #: 质心分箱下壳层之间不守恒，必须写得出来。
    assignment_method: str = ""
    assignment_note: str = ""

    def support_fraction(self) -> float:
        """被覆盖的壳层体积占网格内总可供体积的比例。"""
        if self.covered_volume is None or self.shell_volume is None:
            return float("nan")
        total = float(np.sum(self.shell_volume))
        if total <= 0.0:
            return float("nan")
        return float(np.sum(np.clip(self.covered_volume, 0.0, None)) / total)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.validity))

    def valid_arrays(self) -> dict[str, np.ndarray]:
        mask = self.validity
        return {
            "radius": self.radius[mask],
            "density": self.density[mask],
            "radial_velocity": self.radial_velocity[mask],
            "specific_internal_energy": self.specific_internal_energy[mask],
            "pressure": self.pressure[mask],
            "volume_fraction": self.volume_fraction[mask],
            "weight": self.weight[mask],
            "count": self.count[mask],
        }

    def as_dict(self) -> dict[str, Any]:
        return {"medium": self.medium, "time": self.time, "grid": self.grid.as_dict(),
                "n_valid": self.n_valid}


def medium_mask(frame: RawFrame, medium: str) -> np.ndarray:
    """Pure-medium mask using the fixed thresholds."""
    alpha = np.asarray(frame.tnt_volume_fraction, dtype=np.float64)
    if medium == PRODUCT:
        return alpha >= PURE_PRODUCT_THRESHOLD
    if medium == AIR:
        return alpha <= PURE_AIR_THRESHOLD
    raise ShellAverageError(f"unknown medium {medium!r}")


def mixed_cell_mask(frame: RawFrame) -> np.ndarray:
    """ALE mixed cells: the interface band, never a field label."""
    alpha = np.asarray(frame.tnt_volume_fraction, dtype=np.float64)
    return (alpha > PURE_AIR_THRESHOLD) & (alpha < PURE_PRODUCT_THRESHOLD)


def _shell_dispersion(
    index: np.ndarray,
    values: np.ndarray,
    n_shells: int,
) -> np.ndarray:
    """Unweighted relative standard deviation of ``values`` within each shell.

    This is the angular dispersion proxy: within a thin radial shell the only
    remaining variation is directional (plus mesh noise), so a small value is
    direct evidence of spherical symmetry.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = (index >= 0) & np.isfinite(values)
    idx = index[valid]
    v = values[valid]
    counts = np.bincount(idx, minlength=n_shells).astype(np.float64)
    total = np.bincount(idx, weights=v, minlength=n_shells)
    total_sq = np.bincount(idx, weights=v * v, minlength=n_shells)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, total / np.maximum(counts, 1.0), np.nan)
        variance = np.where(
            counts > 1, total_sq / np.maximum(counts, 1.0) - mean * mean, np.nan
        )
        std = np.sqrt(np.maximum(variance, 0.0))
        relative = std / np.maximum(np.abs(mean), np.finfo(np.float64).tiny)
    return relative


# --------------------------------------------------------------------------
# 单元 x 球壳的重叠体积分配（§8）
# --------------------------------------------------------------------------
#
# 球壳是**后处理统计区域**，不重划 DYNA 网格。旧实现按质心半径把整个单元塞进一个
# 壳层，于是：
#
# * 壳层比单元细时（近场 0.12 mm 壳层对 0.85 mm 单元）一个单元只落进一个壳层，
#   剖面是阶梯；
# * 跨壳层边界的单元被整个记在一边，质量与体积在壳层之间**不守恒**。
#
# 重叠体积分配解决这两条：一个单元贡献给它真正跨越的每一个壳层，贡献多少按体积算。
#
# **重构假设必须写明。** 单元的节点半径给出它在径向上覆盖的 ``[r_in, r_out]``，
# 但单元内部的体积如何分布在这个区间上是**看不见的**。这里取**圆台**假设：
# ``dV/dr ∝ r^2``，也就是把单元当成固定立体角的一段。对球对称分区的六面体这是
# 精确的；对畸变单元是重构。它天然守恒体积（分数在 [r_in, r_out] 上积分为 1）。
#
# **重叠体积不会增加原始分辨率。** 它只把同一个单元的信息在壳层之间重新分配，
# 单元内部的梯度仍然不可见。"壳层加密到 0.12 mm" 不等于"分辨率到 0.12 mm"。

#: 分配方法的标识，会随数据一起写出。
ASSIGN_OVERLAP = "frustum_overlap_volume"
ASSIGN_CENTROID = "centroid_binning"


@dataclass
class OverlapAssignment:
    """一个帧的单元→壳层重叠体积表。

    ``cell_index`` / ``shell_index`` / ``volume`` 三个等长数组构成稀疏三元组：
    ``volume[k]`` 是 ``cell_index[k]`` 落在 ``shell_index[k]`` 里的体积。

    **它是壳层统计的聚合算子。** 把三元组看成矩阵 ``A``（行=壳层、列=单元、
    ``A[k, j]`` = 壳层 k 与单元 j 的重叠体积），那么"按体积加权求和"就是
    ``A @ x``，"按质量加权"就是 ``A @ (rho * x)``。稀疏矩阵乘一次算完所有壳层，
    比逐场量做 2000 万配对的 ``bincount`` 快一个量级；而 ``A`` 只依赖几何，
    欧拉网格上整个算例建一次就够。
    """

    cell_index: np.ndarray
    shell_index: np.ndarray
    volume: np.ndarray
    method: str
    note: str = ""
    _matrix: Any = field(default=None, repr=False, compare=False)
    _matrix_shape: tuple[int, int] | None = field(default=None, repr=False, compare=False)
    _binary: Any = field(default=None, repr=False, compare=False)

    @property
    def n_pairs(self) -> int:
        return int(self.shell_index.size)

    def matrix(self, n_cells: int, n_shells: int):
        """缓存的 ``(n_shells, n_cells)`` CSR 聚合矩阵。"""
        if self._matrix is None or self._matrix_shape != (int(n_shells), int(n_cells)):
            from scipy.sparse import csr_matrix

            self._matrix = csr_matrix(
                (self.volume, (self.shell_index, self.cell_index)),
                shape=(int(n_shells), int(n_cells)),
            )
            self._matrix_shape = (int(n_shells), int(n_cells))
        return self._matrix

    def structure(self, n_cells: int, n_shells: int):
        """同一稀疏结构的 **0/1** 矩阵：``(structure @ x)[k]`` 是壳层 k 里被计入的
        **单元个数**（按 ``x`` 加权）。

        "贡献了几个单元"必须由结构决定，不能由 :meth:`matrix` 决定——后者的非零元是
        **重叠体积**，拿它当计数会得到 1e-5 这样的数，于是 ``count >= 1`` 永远不成立，
        整个壳层的 ``validity`` 全变成 ``False``，两种介质的监督被静默关掉。这个错误
        不会报错，只会让"有效壳层比例"变成 0。
        """
        if self._binary is None or self._matrix_shape != (int(n_shells), int(n_cells)):
            from scipy.sparse import csr_matrix

            self._binary = csr_matrix(
                (np.ones(self.volume.size, dtype=np.float64),
                 (self.shell_index, self.cell_index)),
                shape=(int(n_shells), int(n_cells)),
            )
        return self._binary

    def accumulate(
        self,
        n_cells: int,
        n_shells: int,
        cell_weight: np.ndarray,
        values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``A @ (weight * values)`` 与 ``A @ weight``，以及每个壳层的无效计数。

        **非有限值不进分子也不进分母**：它们被置零并单独计数，所以分母反映的是
        "真正被有效样本覆盖了多少"，而不是把没覆盖的部分当成零值拉低均值。
        """
        matrix = self.matrix(n_cells, n_shells)
        w = np.asarray(cell_weight, dtype=np.float64)
        positive = np.isfinite(w) & (w > 0.0)
        if values is None:
            v = np.ones(n_cells, dtype=np.float64)
            good = positive
        else:
            v = np.asarray(values, dtype=np.float64)
            good = positive & np.isfinite(v)
        numerator = matrix @ np.where(good, w * np.where(good, v, 0.0), 0.0)
        denominator = matrix @ np.where(good, w, 0.0)
        invalid = matrix @ (positive & ~good).astype(np.float64)
        return (
            np.asarray(numerator).ravel(),
            np.asarray(denominator).ravel(),
            np.asarray(invalid).ravel(),
        )

    def mean(
        self, n_cells: int, n_shells: int, cell_weight, values
    ) -> tuple[np.ndarray, np.ndarray]:
        """加权均值与它的分母；分母为零处返回 ``nan``。"""
        numerator, denominator, _ = self.accumulate(n_cells, n_shells, cell_weight, values)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(
                denominator > 0.0,
                numerator / np.where(denominator > 0.0, denominator, 1.0),
                np.nan,
            )
        return mean, denominator


def _overlap_pairs(
    r_in: np.ndarray, r_out: np.ndarray, edges: np.ndarray, n_shells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每个单元与它跨越的每个壳层的重叠**体积分数**，返回三元组索引。

    圆台假设：``frac = (b^3 - a^3) / (r_out^3 - r_in^3)``，其中 ``[a, b]`` 是壳层与
    ``[r_in, r_out]`` 的交。退化单元（``r_out <= r_in``）整个落在包含 ``r_in`` 的
    壳层里，分数为 1——否则 ``denom = 0`` 会让它的体积**凭空消失**，而消失的体积
    不会报错，只会让壳层的质量守恒检查差一点点。
    """
    r_in = np.asarray(r_in, dtype=np.float64)
    r_out = np.asarray(r_out, dtype=np.float64)
    n_cells = r_in.size
    if n_cells == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)

    # 退化：给一个极小的径向厚度，分母不为零，且整个体积只落在包含它的那个壳层。
    degenerate = ~(r_out > r_in)
    if np.any(degenerate):
        scale = np.maximum(np.abs(r_in[degenerate]), 1.0e-30)
        r_out = np.where(degenerate, r_in + scale * 1.0e-12, r_out)

    # 与网格完全不相交的单元直接排除（不是"落到第一个/最后一个壳层"）。
    inside = (r_out > edges[0]) & (r_in < edges[-1])
    if not np.any(inside):
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)
    cell_ids = np.flatnonzero(inside)
    lo_all = np.searchsorted(edges, r_in[cell_ids], side="right") - 1
    hi_all = np.searchsorted(edges, r_out[cell_ids], side="right") - 1
    lo_all = np.clip(lo_all, 0, n_shells - 1)
    hi_all = np.clip(hi_all, 0, n_shells - 1)
    hi_all = np.maximum(hi_all, lo_all)

    counts = (hi_all - lo_all + 1).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)

    cell_index = np.repeat(cell_ids, counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    shell_index = np.repeat(lo_all, counts) + (np.arange(total, dtype=np.int64) - starts)

    r0 = r_in[cell_index]
    r1 = r_out[cell_index]
    a = np.maximum(edges[shell_index], r0)
    b = np.minimum(edges[shell_index + 1], r1)
    denom = r1 * r1 * r1 - r0 * r0 * r0
    numer = np.maximum(b, a) ** 3 - a ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom > 0.0, numer / np.where(denom > 0.0, denom, 1.0), 0.0)
    frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)
    return cell_index, shell_index, frac


def overlap_assignment_from_arrays(
    r_in, r_out, cell_volume, grid: ShellGrid
) -> OverlapAssignment:
    """由**固定**的单元几何建重叠表。

    ALE 网格固定在空间中，单元的 ``[r_in, r_out]`` 和体积都是模型的常量，所以这张
    表整个算例只需要建一次——实测建一次约 1.8 s（1.2e6 个单元 × 37 个壳层，
    2040 万条配对），逐帧重建成 779 次就是纯浪费。网格不固定时才退回逐帧建。
    """
    r_in = np.asarray(r_in, dtype=np.float64)
    r_out = np.asarray(r_out, dtype=np.float64)
    volume = np.abs(np.asarray(cell_volume, dtype=np.float64))
    cell_index, shell_index, frac = _overlap_pairs(r_in, r_out, grid.edges, grid.n_shells)
    return OverlapAssignment(
        cell_index=cell_index,
        shell_index=shell_index,
        volume=frac * volume[cell_index],
        method=ASSIGN_OVERLAP,
        note=(
            "each cell contributes to every shell it spans, in proportion to the "
            "overlapped volume under a conical-frustum reconstruction (dV/dr ~ r^2, "
            "exact for constant solid angle). The reconstruction does NOT add "
            "resolution: gradients inside a cell remain invisible. Built once from the "
            "fixed mesh geometry and reused for every state"
        ),
    )


def overlap_assignment(frame: RawFrame, grid: ShellGrid) -> OverlapAssignment:
    """单元→壳层重叠体积表；缺少径向覆盖区间时退回质心分箱并**说明**。"""
    r_in = getattr(frame, "cell_radius_inner", None)
    r_out = getattr(frame, "cell_radius_outer", None)
    volume = np.abs(np.asarray(frame.cell_volume, dtype=np.float64))
    if r_in is None or r_out is None:
        index = grid.indices(np.asarray(frame.radius, dtype=np.float64))
        keep = index >= 0
        return OverlapAssignment(
            cell_index=np.flatnonzero(keep),
            shell_index=np.asarray(index[keep], dtype=np.int64),
            volume=volume[keep],
            method=ASSIGN_CENTROID,
            note=(
                "the frame carries no [r_in, r_out] per cell, so each cell was placed "
                "wholly into the shell containing its centroid: mass and volume are then "
                "NOT conserved across shells, and a shell finer than a cell comes out as "
                "a staircase"
            ),
        )
    return overlap_assignment_from_arrays(r_in, r_out, volume, grid)




def _shell_volumes(grid: ShellGrid) -> np.ndarray:
    """每个壳层**被建模的那一块**的体积。

    完整球是 ``(4/3)pi(r_o^3 - r_i^3)``；八分之一球模型只建了其中的 ``1/m``，
    所以除以 ``symmetry_multiplier``。用整球体积当分母会让覆盖率小 m 倍。
    """
    e = np.asarray(grid.edges, dtype=np.float64)
    sphere = 4.0 / 3.0 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    return sphere / float(grid.symmetry_multiplier)


def shell_average(
    frame: RawFrame,
    grid: ShellGrid,
    medium: str,
    *,
    pure_only: bool = True,
    minimum_cells: int = 1,
    assignment: OverlapAssignment | None = None,
) -> ShellProfile:
    """Average one frame onto ``grid`` for one medium.

    ``pure_only=True`` (the default, and what the specification requires for
    field supervision) restricts the average to pure cells of that medium.
    Setting it to ``False`` includes the mixed band and is used **only** for
    diagnostics, never for training data.

    A shell is *valid* when it contains at least ``minimum_cells`` cells of the
    medium and all the averaged fields are finite.  Validity deliberately does
    **not** depend on the shell's weight relative to other shells: a graded mesh
    makes the outermost shells orders of magnitude heavier than the near-field
    ones, and judging by a global weight ceiling would discard the entire near
    field — exactly the region the model cares about.

    **加权方式按字段分（§8）**：``rho``、``p`` 体积加权；``u``、``e`` **质量加权**。
    前者是"每单位体积有多少"，后者是"每单位质量有多少"——能量和速度是强度量，
    按体积平均会被高密度区以外的样本稀释，按质量平均才是它们的守恒平均。
    """
    alpha = np.asarray(frame.tnt_volume_fraction, dtype=np.float64)

    if medium == PRODUCT:
        alpha_medium = alpha
    elif medium == AIR:
        alpha_medium = 1.0 - alpha
    else:
        raise ShellAverageError(f"unknown medium {medium!r}")

    select = medium_mask(frame, medium) if pure_only else np.ones_like(alpha, dtype=bool)
    # 介质份额：混合单元按体积分数线性分摊。**这是一个声明过的重构假设**
    # （单元内均匀混合），不是实测细节；§9 的几何重构落地之前它一直成立。
    medium_fraction = np.where(select, alpha_medium, 0.0)

    # 同一帧的重叠表被产物、空气、混合分数、逐场量剖面对用；它只依赖几何，
    # 不依赖介质，所以由调用方建一次传进来。重建一次的代价是 1.2e6 个单元 × 37 个
    # 壳层的配对（实测约 1.8 s/帧），重复四遍就是整条提取里最大的一笔浪费。
    if assignment is None:
        assignment = overlap_assignment(frame, grid)
    n_shells = grid.n_shells
    n_cells = int(frame.n_cells)
    # 聚合就是矩阵乘：A @ (cell_weight * value)。体积加权用 cell_weight = 介质份额，
    # 质量加权再乘一个 rho。这样每个场量只做一次稀疏乘，而不是两千万配对的 bincount。
    volume_weight_cell = medium_fraction
    # Prefer the per-material density m_m/(alpha_m V) when the reader supplied
    # it.  The mixture density is only a fallback for readers that do not carry
    # the ALE group masses; for a pure cell the two coincide, but for a cell at
    # the 0.95 purity threshold they do not.
    density_values = getattr(frame, f"density_{medium}", None)
    if density_values is not None:
        density_values = np.asarray(density_values, dtype=np.float64)
        if density_values.size != n_cells:
            raise ShellAverageError(
                f"density_{medium} has {density_values.size} entries, expected {n_cells}"
            )
    else:
        density_values = np.asarray(frame.density, dtype=np.float64)
    rho_cell = density_values
    mass_weight_cell = volume_weight_cell * rho_cell

    density, _ = assignment.mean(n_cells, n_shells, volume_weight_cell, density_values)
    pressure, total_weight = assignment.mean(
        n_cells, n_shells, volume_weight_cell, np.asarray(frame.pressure, dtype=np.float64)
    )

    if frame.radial_velocity is None:
        raise ShellAverageError(
            "frame has no radial velocity; call RawFrame.with_radial_velocity() first"
        )
    velocity, _ = assignment.mean(
        n_cells, n_shells, mass_weight_cell, np.asarray(frame.radial_velocity, dtype=np.float64)
    )

    if frame.specific_internal_energy is None:
        energy = np.full(n_shells, np.nan)
    else:
        energy, _ = assignment.mean(
            n_cells, n_shells, mass_weight_cell,
            np.asarray(frame.specific_internal_energy, dtype=np.float64),
        )

    fraction, _ = assignment.mean(n_cells, n_shells, volume_weight_cell, alpha)

    # 覆盖：这一侧真正被介质占据了多少体积，占壳层几何体积的多少。壳层平均只在
    # 覆盖到的部分上成立；覆盖率是"它成立在多大范围上"的度量。**分母是壳层自己的
    # 几何体积**，不是"贡献过来的体积之和"——后者恒等于 1，量不出任何东西。
    shell_volume = _shell_volumes(grid)
    covered, _, _ = assignment.accumulate(n_cells, n_shells, volume_weight_cell, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = covered / np.where(shell_volume > 0.0, shell_volume, 1.0)

    # 参与统计的**单元个数**：用 0/1 结构矩阵数，不能用重叠体积矩阵（那是体积不是个数）。
    counts = np.asarray(
        assignment.structure(n_cells, n_shells)
        @ (volume_weight_cell > 0.0).astype(np.float64)
    ).ravel()

    validity = (counts >= max(1, int(minimum_cells))) & (total_weight > 0.0)
    validity &= np.isfinite(density) & np.isfinite(pressure) & np.isfinite(velocity)

    index = grid.indices(np.asarray(frame.radius, dtype=np.float64))
    dispersion = {
        "density": _shell_dispersion(index[select], density_values[select], n_shells),
        "pressure": _shell_dispersion(index[select], frame.pressure[select], n_shells),
        "radial_velocity": _shell_dispersion(
            index[select], np.asarray(frame.radial_velocity)[select], n_shells
        ),
    }

    return ShellProfile(
        medium=medium,
        time=float(frame.time),
        grid=grid,
        radius=grid.centres.copy(),
        weight=total_weight,
        count=counts,
        density=density,
        radial_velocity=velocity,
        specific_internal_energy=energy,
        pressure=pressure,
        volume_fraction=fraction,
        validity=validity,
        dispersion=dispersion,
        covered_volume=covered,
        shell_volume=shell_volume,
        coverage=np.clip(coverage, 0.0, 1.0),
        assignment_method=assignment.method,
        assignment_note=assignment.note,
    )


@dataclass
class MixtureProfile:
    """Volume-weighted mean explosive volume fraction per shell.

    This is the plain mixture fraction ``sum(V a) / sum(V)`` over **all** cells,
    pure and mixed alike.  It is the profile the interface is located on: the
    ``alpha``-weighted average used for field supervision can never cross
    ``0.5``, because restricting the average to pure product cells forces it to
    ``~1`` in every shell it visits.

    It is also the profile every **classification** must use.  ``alpha`` averaged
    over pure product cells is ``nan`` — not ``0`` — in a shell that holds no
    product at all, so a classifier built on it sees neither ``>= 0.95`` nor
    ``<= 0.05`` and silently drops every air shell into the "unknown" branch.
    """

    time: float
    radius: np.ndarray
    alpha: np.ndarray
    weight: np.ndarray
    count: np.ndarray
    validity: np.ndarray
    #: 分箱方法（重叠体积 / 质心分箱）及说明。混合单元按体积分数线性分摊隐含
    #: "单元内均匀混合"，这条写在 ``assignment_note`` 里。
    assignment_method: str = ""
    assignment_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "n_valid": int(np.count_nonzero(self.validity)),
            "alpha_min": float(np.nanmin(self.alpha)) if self.alpha.size else float("nan"),
            "alpha_max": float(np.nanmax(self.alpha)) if self.alpha.size else float("nan"),
        }


def volume_weighted_profile(
    frame: RawFrame,
    grid: ShellGrid,
    values,
    *,
    minimum_cells: int = 1,
    assignment: OverlapAssignment | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sum(V q) / sum(V)`` per shell over **all** cells.  Returns ``(mean,
    weight, count)``.

    Every cell contributes, whatever its material.  That is what makes the
    result defined in every shell the domain covers: the denominator is the
    shell's own volume, so it vanishes only for a genuinely empty shell.  An
    average weighted by ``alpha`` (or by the pure-medium mask) does *not* have
    that property — its denominator is zero wherever that medium is absent, and
    the shell comes back as ``nan``.

   分箱用**重叠体积**（§8）。这里分子分母都按重叠体积算，所以壳层之间的体积与质量
    守恒；用质心分箱时同一个量会在壳层边界上少一块、在另一侧多一块。
    """
    if assignment is None:
        assignment = overlap_assignment(frame, grid)
    n_shells = grid.n_shells
    n_cells = int(frame.n_cells)
    ones = np.ones(n_cells, dtype=np.float64)
    mean, denominator = assignment.mean(
        n_cells, n_shells, ones, np.asarray(values, dtype=np.float64)
    )
    counts = np.asarray(assignment.structure(n_cells, n_shells) @ ones).ravel()
    return mean, denominator, counts


def mixture_fraction_profile(
    frame: RawFrame,
    grid: ShellGrid,
    *,
    minimum_cells: int = 1,
    assignment: OverlapAssignment | None = None,
) -> MixtureProfile:
    """Volume-weighted explosive volume fraction on ``grid``, over all cells.

    **由所有贡献单元统计**，不是只对纯产物样本求平均（§8）。后者在没有产物的
    壳层上是 ``nan`` 而不是 0，既过不了 ``>= 0.95`` 也过不了 ``<= 0.05``；而且它在
    产物侧恒等于 ~1，永远不会穿过 0.5。

    混合单元按体积分数线性分摊隐含**单元内均匀混合**这个假设；§9 的几何重构落地之前
    必须带着这个说明读这组数。
    """
    if assignment is None:
        assignment = overlap_assignment(frame, grid)
    alpha_mean, total_weight, counts = volume_weighted_profile(
        frame, grid, frame.tnt_volume_fraction, assignment=assignment
    )
    validity = (counts >= max(1, int(minimum_cells))) & (total_weight > 0.0) & np.isfinite(alpha_mean)
    return MixtureProfile(
        time=float(frame.time),
        radius=grid.centres.copy(),
        alpha=alpha_mean,
        weight=total_weight,
        count=counts,
        validity=validity,
        assignment_method=assignment.method,
        assignment_note=assignment.note,
    )


# --------------------------------------------------------------------------
# Sphericity audit
# --------------------------------------------------------------------------


@dataclass
class SphericityAudit:
    """Angular-dispersion evidence that the 3D result really is spherical."""

    time: float
    pressure_dispersion_p95: float
    density_dispersion_p95: float
    velocity_dispersion_p95: float
    tangential_energy_fraction: float
    radial_to_tangential_ratio: float
    n_cells: int
    per_shell: dict[str, np.ndarray] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "pressure_dispersion_p95": self.pressure_dispersion_p95,
            "density_dispersion_p95": self.density_dispersion_p95,
            "velocity_dispersion_p95": self.velocity_dispersion_p95,
            "tangential_energy_fraction": self.tangential_energy_fraction,
            "radial_to_tangential_ratio": self.radial_to_tangential_ratio,
            "n_cells": self.n_cells,
        }


def _percentile_from_shells(values: np.ndarray, grid: ShellGrid, percentile: float = 95.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    del grid
    return float(np.percentile(finite, percentile))


def tangential_velocity_fraction(frame: RawFrame) -> tuple[float, float]:
    """Fraction of kinetic energy in the tangential direction, and ``|u_r|/|u_t|``.

    A perfectly spherical flow has no tangential velocity at all, so this is
    the sharpest single-number test of the symmetry assumption.
    """
    if frame.cell_centroid is None:
        raise ShellAverageError("frame has no cell centroids; cannot separate radial and tangential velocity")
    centroid = np.asarray(frame.cell_centroid, dtype=np.float64)
    norm = np.linalg.norm(centroid, axis=1)
    safe = np.where(norm > 0.0, norm, 1.0)
    unit = centroid / safe[:, None]
    v = np.stack(
        [
            np.asarray(frame.velocity_x, dtype=np.float64),
            np.asarray(frame.velocity_y, dtype=np.float64),
            np.asarray(frame.velocity_z, dtype=np.float64),
        ],
        axis=1,
    )
    radial = np.einsum("ij,ij->i", v, unit)
    tangential = v - radial[:, None] * unit
    radial_energy = np.sum(radial**2)
    tangential_energy = np.sum(np.sum(tangential**2, axis=1))
    total = radial_energy + tangential_energy
    fraction = float(tangential_energy / total) if total > 0.0 else 0.0
    radial_rms = float(np.sqrt(np.mean(radial**2)))
    tangential_rms = float(np.sqrt(np.mean(np.sum(tangential**2, axis=1))))
    ratio = radial_rms / tangential_rms if tangential_rms > 0.0 else float("inf")
    return fraction, ratio


def sphericity_audit(
    frame: RawFrame,
    grid: ShellGrid,
    *,
    percentile: float = 95.0,
) -> SphericityAudit:
    """Full sphericity audit for one frame."""
    product = shell_average(frame, grid, PRODUCT)
    air = shell_average(frame, grid, AIR)
    pressure_dispersion = np.concatenate([product.dispersion["pressure"], air.dispersion["pressure"]])
    density_dispersion = np.concatenate([product.dispersion["density"], air.dispersion["density"]])
    velocity_dispersion = np.concatenate(
        [product.dispersion["radial_velocity"], air.dispersion["radial_velocity"]]
    )
    tangential_fraction, ratio = tangential_velocity_fraction(frame)

    return SphericityAudit(
        time=float(frame.time),
        pressure_dispersion_p95=_percentile_from_shells(pressure_dispersion, grid, percentile),
        density_dispersion_p95=_percentile_from_shells(density_dispersion, grid, percentile),
        velocity_dispersion_p95=_percentile_from_shells(velocity_dispersion, grid, percentile),
        tangential_energy_fraction=tangential_fraction,
        radial_to_tangential_ratio=ratio,
        n_cells=frame.n_cells,
        per_shell={
            "pressure_dispersion": pressure_dispersion,
            "density_dispersion": density_dispersion,
            "velocity_dispersion": velocity_dispersion,
        },
    )


def directional_front_radii(
    frame: RawFrame,
    *,
    quantity: Any,
    level: float,
    n_theta: int = 6,
    n_phi: int = 12,
    radial_bins: int = 400,
) -> dict[str, Any]:
    """Front radius measured independently in many directions.

    The sphere is split into ``n_theta x n_phi`` solid-angle bins; inside each
    bin a radial profile of ``quantity`` is built and interpolated to ``level``.
    The spread of the resulting radii is the directional front-spread metric
    the audit reports (2 % by default).
    """
    if frame.cell_centroid is None:
        raise ShellAverageError("directional extraction needs cell centroids")
    centroid = np.asarray(frame.cell_centroid, dtype=np.float64)
    r = np.asarray(frame.radius, dtype=np.float64)
    values = np.asarray(quantity, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.where(r > 0, centroid[:, 2] / np.maximum(r, 1.0e-300), 1.0)
        phi = np.arctan2(centroid[:, 1], centroid[:, 0])
    theta_bin = np.clip(((np.arccos(np.clip(cos_theta, -1.0, 1.0)) / np.pi) * n_theta).astype(int), 0, n_theta - 1)
    phi_bin = np.clip((((phi + np.pi) / (2.0 * np.pi)) * n_phi).astype(int), 0, n_phi - 1)
    bin_id = theta_bin * n_phi + phi_bin

    edges = np.linspace(0.0, float(r.max()) if r.size else 1.0, radial_bins + 1)
    radii: dict[int, float] = {}
    for identifier in np.unique(bin_id):
        select = bin_id == identifier
        if np.count_nonzero(select) < 8:
            continue
        profile_r = r[select]
        profile_v = values[select]
        order = np.argsort(profile_r)
        profile_r, profile_v = profile_r[order], profile_v[order]
        counts, _ = np.histogram(profile_r, bins=edges)
        sums, _ = np.histogram(profile_r, bins=edges, weights=profile_v)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        centres = 0.5 * (edges[:-1] + edges[1:])
        finite = np.isfinite(mean)
        if np.count_nonzero(finite) < 2:
            continue
        front = _outermost_crossing(centres[finite], mean[finite], level)
        if front is not None:
            radii[int(identifier)] = front

    if not radii:
        return {"radii": np.array([]), "relative_spread": float("nan"), "count": 0}
    array = np.array(sorted(radii.values()), dtype=np.float64)
    reference = float(np.median(array))
    spread = float((array.max() - array.min()) / reference) if reference > 0.0 else float("nan")
    del radial_bins
    return {
        "radii": array,
        "median": reference,
        "min": float(array.min()),
        "max": float(array.max()),
        "relative_spread": spread,
        "count": int(array.size),
    }


def _outermost_crossing(radius: np.ndarray, values: np.ndarray, level: float) -> float | None:
    """Outermost radius at which ``values`` crosses ``level``, by linear interpolation."""
    shifted = values - level
    sign_change = np.where(np.diff(np.sign(shifted)) != 0)[0]
    if sign_change.size == 0:
        return None
    index = int(sign_change[-1])
    v0, v1 = shifted[index], shifted[index + 1]
    if v1 == v0:
        return float(radius[index])
    fraction = -v0 / (v1 - v0)
    return float(radius[index] + fraction * (radius[index + 1] - radius[index]))


def directional_front_metrics(
    frame: RawFrame,
    *,
    contact_radius: float,
    n_theta: int = 6,
    n_phi: int = 12,
    radial_bins: int = 200,
    minimum_cells: int = 4,
) -> dict[str, float]:
    """Measure ``R_c`` and ``R_s`` independently in each direction.

    The sphere is split into ``n_theta x n_phi`` solid-angle bins.  Within each
    bin the explosive volume fraction is interpolated to ``0.5`` for the contact
    and the pressure gradient is minimised for the shock.  The spread of the
    resulting radii is the metric that actually tests the spherical-symmetry
    assumption: measuring the radius over *time* would only show that the blast
    expands.

    The angular resolution is reduced automatically until enough bins hold
    enough cells.  A coarse direction set would otherwise leave every bin empty
    and the metric would report ``nan`` — which is honest but useless when the
    data is simply sparse.
    """
    if frame.cell_centroid is None:
        return _empty_directional_metrics()

    for theta_bins, phi_bins in _resolution_ladder(n_theta, n_phi):
        result = _directional_front_metrics_at(
            frame,
            contact_radius=contact_radius,
            n_theta=theta_bins,
            n_phi=phi_bins,
            radial_bins=radial_bins,
            minimum_cells=minimum_cells,
        )
        if result["n_contact_directions"] >= 3 and result["n_shock_directions"] >= 3:
            return result
    return result


def _empty_directional_metrics() -> dict[str, float]:
    return {
        "contact_spread": float("nan"),
        "shock_spread": float("nan"),
        "n_contact_directions": 0.0,
        "n_shock_directions": 0.0,
    }


def _resolution_ladder(n_theta: int, n_phi: int) -> list[tuple[int, int]]:
    """Angular resolutions to try, from the requested one downwards."""
    ladder: list[tuple[int, int]] = []
    theta, phi = max(1, int(n_theta)), max(1, int(n_phi))
    while True:
        ladder.append((theta, phi))
        if theta <= 1 and phi <= 2:
            break
        theta = max(1, theta - 1)
        phi = max(2, phi // 2)
    return ladder


def _directional_front_metrics_at(
    frame: RawFrame,
    *,
    contact_radius: float,
    n_theta: int,
    n_phi: int,
    radial_bins: int,
    minimum_cells: int,
) -> dict[str, float]:
    centroid = np.asarray(frame.cell_centroid, dtype=np.float64)
    radius = np.asarray(frame.radius, dtype=np.float64)
    alpha = np.asarray(frame.tnt_volume_fraction, dtype=np.float64)
    pressure = np.asarray(frame.pressure, dtype=np.float64)
    if radius.size == 0:
        return _empty_directional_metrics()

    safe = np.where(radius > 0.0, radius, 1.0)
    cos_theta = np.where(radius > 0.0, centroid[:, 2] / safe, 1.0)
    phi = np.arctan2(centroid[:, 1], centroid[:, 0])
    theta_bin = np.clip(
        ((np.arccos(np.clip(cos_theta, -1.0, 1.0)) / np.pi) * n_theta).astype(int),
        0,
        n_theta - 1,
    )
    phi_bin = np.clip(
        (((phi + np.pi) / (2.0 * np.pi)) * n_phi).astype(int), 0, n_phi - 1
    )
    bin_id = theta_bin * n_phi + phi_bin

    edges = np.linspace(0.0, float(radius.max()) if radius.size else 1.0, int(radial_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    contact_radii: list[float] = []
    shock_radii: list[float] = []
    for identifier in np.unique(bin_id):
        select = bin_id == identifier
        if np.count_nonzero(select) < minimum_cells:
            continue
        profile_r = radius[select]
        order = np.argsort(profile_r)
        alpha_profile = _binned_mean(profile_r[order], alpha[select][order], edges)
        pressure_profile = _binned_mean(profile_r[order], pressure[select][order], edges)
        finite = np.isfinite(alpha_profile)
        if np.count_nonzero(finite) >= 2:
            crossing = _outermost_crossing(centres[finite], alpha_profile[finite], 0.5)
            if crossing is not None:
                contact_radii.append(crossing)
        usable = np.isfinite(pressure_profile) & (centres > contact_radius)
        if np.count_nonzero(usable) >= 5:
            sub_r = centres[usable]
            sub_p = pressure_profile[usable]
            gradient = np.gradient(sub_p, sub_r)
            index = int(np.argmin(np.where(np.isfinite(gradient), gradient, 0.0)))
            if gradient[index] < 0.0:
                shock_radii.append(float(sub_r[index]))

    return {
        "contact_spread": _relative_spread(contact_radii),
        "shock_spread": _relative_spread(shock_radii),
        "n_contact_directions": float(len(contact_radii)),
        "n_shock_directions": float(len(shock_radii)),
    }


def _binned_mean(radius: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(radius, bins=edges)
    sums, _ = np.histogram(radius, bins=edges, weights=values)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _relative_spread(values: Sequence[float]) -> float:
    """``(max - min) / median`` of a set of directional front radii."""
    if len(values) < 3:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    reference = float(np.median(array))
    if reference <= 0.0:
        return float("nan")
    return float((array.max() - array.min()) / reference)


def shell_average_sequence(
    frames: Iterable[RawFrame],
    grid: ShellGrid,
    media: Sequence[str] = (PRODUCT, AIR),
    *,
    pure_only: bool = True,
) -> dict[str, list[ShellProfile]]:
    """Average a sequence of frames, grouped by medium."""
    result: dict[str, list[ShellProfile]] = {medium: [] for medium in media}
    for frame in frames:
        for medium in media:
            result[medium].append(shell_average(frame, grid, medium, pure_only=pure_only))
    return result


__all__ = [
    "AIR",
    "PRODUCT",
    "PURE_AIR_THRESHOLD",
    "PURE_PRODUCT_THRESHOLD",
    "ShellAverageError",
    "ShellGrid",
    "MixtureProfile",
    "ShellProfile",
    "SphericityAudit",
    "directional_front_metrics",
    "directional_front_radii",
    "medium_mask",
    "mixture_fraction_profile",
    "mixed_cell_mask",
    "shell_average",
    "shell_average_sequence",
    "sphericity_audit",
    "tangential_velocity_fraction",
]
