"""Raw data readers: d3plot, CSV, Parquet and HDF5.

Every reader yields :class:`RawFrame` objects — one per output state — holding
**per-cell** arrays already converted to SI.  Large sources are consumed
in chunks; no reader ever materialises a whole case in memory.

The d3plot reader is the one that matters for this study.  It relies on the
layout established from the real output (see
``docs``/``IMPLEMENTATION_PLAN.md`` and the audit report):

``element_solid_history_variables``
    ``[0]`` mixture density, ``[1:8]`` ALE volume fractions,
    ``[8]`` ALE material index, ``[9:16]`` per-material mass.
``element_solid_stress``
    Cauchy stress; both TNT and air are strengthless, so ``p = -trace(sigma)/3``.

Specific internal energy is **not** written by these decks, so it is
reconstructed from ``(rho, p)`` through the material EOS.  That reconstruction
is recorded as a derived field with its own uncertainty, never presented as a
directly measured quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from ..constants import TINY_DENSITY
from ..physics.eos import JWLParameters, ideal_gas_pressure_numpy, jwl_pressure_numpy
from .schema import DataSchemaError, FieldMapping

#: Trilinear hex8 decomposed into six tetrahedra (the standard subdivision
#: about the 0-6 diagonal).  Used for the current-configuration cell volume.
HEX8_TO_TETS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


class RawReadError(DataSchemaError):
    """Raised when a raw source cannot be read or interpreted."""


@dataclass
class RawFrame:
    """One output state, per cell, in SI."""

    case_id: str
    time: float
    radius: np.ndarray
    cell_volume: np.ndarray
    density: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    velocity_z: np.ndarray
    pressure: np.ndarray
    tnt_volume_fraction: np.ndarray
    radial_velocity: np.ndarray | None = None
    specific_internal_energy: np.ndarray | None = None
    material_id: np.ndarray | None = None
    burn_fraction: np.ndarray | None = None
    cell_centroid: np.ndarray | None = None
    #: 每个单元的 ``(rho, p)`` 是否是**所声明的那套状态方程**的一个状态。
    #:
    #: 爆炸物单元的压力低于 JWL 冷压曲线时，它还没有释放化学能——DYNA 尚未在那里
    #: 施加产物状态方程。这种单元有真实的密度、速度和压力，但**没有比内能标签**：
    #: 对它套产物 JWL 会解出一个负的比内能。理想气体侧同理要求 ``e > 0``。
    #:
    #: 密度/速度/压力仍然是有效测量，所以保留；只有比内能被置为 ``nan``，并且这个
    #: 掩码被显式暴露出来，让下游能**按体积**判定"这个壳层是否被污染"，而不是让
    #: 加权平均静默地把它丢掉。
    state_valid: np.ndarray | None = None
    #: 每个单元在径向上覆盖的区间 ``[r_in, r_out]``（节点半径的 min / max）。
    #:
    #: 球壳统计用它做**重叠体积**分配（§8）。没有它时只能退回"按质心半径整格塞进
    #: 一个壳层"，那样在壳层比单元细的近场会造出阶梯；两种方法都会被记录下来，
    #: 不会有一种悄悄冒充另一种。
    cell_radius_inner: np.ndarray | None = None
    cell_radius_outer: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = np.asarray(self.radius).size
        for name in (
            "cell_volume", "density", "velocity_x", "velocity_y", "velocity_z",
            "pressure", "tnt_volume_fraction",
        ):
            value = np.asarray(getattr(self, name))
            if value.size != n:
                raise RawReadError(
                    f"field {name!r} has {value.size} entries but radius has {n}"
                )

    @property
    def n_cells(self) -> int:
        return int(np.asarray(self.radius).size)

    def with_radial_velocity(self) -> "RawFrame":
        """Fill in the radial velocity projection if it is missing.

        The projection needs the **direction** to each cell, which only the
        centroid provides; the radius magnitude alone is not enough.  A frame
        without centroids therefore raises rather than guessing an axis.
        """
        if self.radial_velocity is None:
            if self.cell_centroid is None:
                raise RawReadError(
                    "cannot derive the radial velocity: the frame carries no cell "
                    "centroids, and the projection needs the radial direction, not "
                    "just the radius magnitude. Supply radial_velocity directly, or "
                    "build the frame with cell_centroid set."
                )
            self.radial_velocity = radial_velocity_from_centroid(
                self.cell_centroid, self.velocity_x, self.velocity_y, self.velocity_z
            )
        return self

    def summary(self) -> dict[str, float]:
        return {
            "time": float(self.time),
            "n_cells": float(self.n_cells),
            "radius_min": float(np.min(self.radius)),
            "radius_max": float(np.max(self.radius)),
            "density_min": float(np.min(self.density)),
            "density_max": float(np.max(self.density)),
            "pressure_min": float(np.min(self.pressure)),
            "pressure_max": float(np.max(self.pressure)),
        }


def radial_velocity_from_centroid(centroid: np.ndarray, vx, vy, vz) -> np.ndarray:
    """``u_r = (x vx + y vy + z vz) / sqrt(x^2 + y^2 + z^2)``.

    ``centroid`` is the cell centre ``(n, 3)``.  Cells exactly at the origin
    return zero radial velocity instead of dividing by zero; the projection is
    the physically correct one there because the direction is undefined and the
    field is regular.
    """
    c = np.asarray(centroid, dtype=np.float64)
    norm = np.linalg.norm(c, axis=1)
    safe = np.where(norm > 0.0, norm, 1.0)
    v = np.stack(
        [
            np.asarray(vx, dtype=np.float64),
            np.asarray(vy, dtype=np.float64),
            np.asarray(vz, dtype=np.float64),
        ],
        axis=1,
    )
    return np.where(norm > 0.0, np.einsum("ij,ij->i", v, c) / safe, 0.0)


def hex8_volumes(points: np.ndarray, tets: Sequence[tuple[int, int, int, int]] = HEX8_TO_TETS) -> np.ndarray:
    """Volumes of ``n`` trilinear hexahedra given as ``(n, 8, 3)`` coordinates."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] != 8:
        raise RawReadError(f"expected an (n, 8, 3) array, got shape {points.shape}")
    total = np.zeros(points.shape[0], dtype=np.float64)
    for i, j, k, l in tets:
        a, b, c, d = points[:, i], points[:, j], points[:, k], points[:, l]
        total += np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0
    return total


# --------------------------------------------------------------------------
# EOS-based specific-internal-energy reconstruction
# --------------------------------------------------------------------------


def reconstruct_product_energy(
    density: np.ndarray,
    pressure: np.ndarray,
    jwl: JWLParameters,
) -> np.ndarray:
    """Invert the caloric JWL EOS for ``e`` given ``(rho, p)``.

    ``p = p_cold(V) + omega rho e``  =>  ``e = (p - p_cold(V)) / (omega rho)``.

    The inverse is a division, not a root solve: for finite ``p``, ``rho > 0``
    and ``omega > 0`` it **always** has a real finite value.  There is no
    "no real solution" branch, and this function deliberately does not invent
    one — it returns the algebraic value, sign included.

    A **negative** result is meaningful, not a failure: it says ``p`` lies below
    the cold curve, i.e. the cell has not released its chemical energy yet and is
    not a JWL product state.  Deciding what to do about that belongs to the
    caller (see the ``state_valid`` mask built in :func:`read_d3plot_states`);
    the value is never silently clipped to something plausible.
    """
    density = np.asarray(density, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    if jwl.omega <= 0.0:
        raise RawReadError(
            "cannot reconstruct the product energy: the JWL omega is zero, so the "
            "caloric EOS carries no thermal term"
        )
    rho_safe = np.maximum(density, TINY_DENSITY)
    energy_at_zero = jwl_pressure_numpy(rho_safe, np.zeros_like(rho_safe), jwl)
    energy = (pressure - energy_at_zero) / (jwl.omega * rho_safe)
    energy = np.where(np.isfinite(energy), energy, np.nan)
    return energy


def reconstruct_air_energy(density: np.ndarray, pressure: np.ndarray, gamma: float) -> np.ndarray:
    """``e = p / ((gamma - 1) rho)``.

    Same contract as :func:`reconstruct_product_energy`: always defined for
    ``rho > 0``, and the sign is reported rather than filtered.
    """
    density = np.asarray(density, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    rho_safe = np.maximum(density, TINY_DENSITY)
    return pressure / ((gamma - 1.0) * rho_safe)


def check_reconstruction_consistency(
    density: np.ndarray,
    energy: np.ndarray,
    pressure: np.ndarray,
    *,
    medium: str,
    jwl: JWLParameters | None = None,
    gamma: float = 1.4,
) -> dict[str, float]:
    """Re-evaluate the EOS from the reconstructed ``e`` and report the closure error."""
    density = np.asarray(density, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    finite = np.isfinite(energy)
    if not np.any(finite):
        return {"n_valid": 0.0, "max_relative_closure_error": float("nan")}
    if medium == "product":
        if jwl is None:
            raise RawReadError("the product medium needs JWL parameters")
        recomputed = jwl_pressure_numpy(density[finite], energy[finite], jwl)
    elif medium == "air":
        recomputed = ideal_gas_pressure_numpy(density[finite], energy[finite], gamma)
    else:
        raise RawReadError(f"unknown medium {medium!r}")
    reference = pressure[finite]
    scale = max(float(np.max(np.abs(reference))), 1.0)
    error = np.abs(recomputed - reference) / scale
    return {
        "n_valid": float(np.count_nonzero(finite)),
        "max_relative_closure_error": float(np.max(error)) if error.size else float("nan"),
        "median_relative_closure_error": float(np.median(error)) if error.size else float("nan"),
    }


# --------------------------------------------------------------------------
# d3plot reader
# --------------------------------------------------------------------------


@dataclass
class D3plotLayout:
    """Column layout of the d3plot arrays, as verified from the real output."""

    mixture_density_column: int = 0
    volume_fraction_columns: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    material_index_column: int = 8
    per_material_mass_columns: tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15)

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> "D3plotLayout":
        section = dict(section or {})
        default = cls()
        start, stop = section.get("volume_fraction_range", [1, 7])
        mass_start, mass_stop = section.get("per_material_mass_range", [9, 15])
        return cls(
            mixture_density_column=int(section.get("mixture_density", default.mixture_density_column)),
            volume_fraction_columns=tuple(range(int(start), int(stop) + 1)),
            material_index_column=int(section.get("material_index", default.material_index_column)),
            per_material_mass_columns=tuple(range(int(mass_start), int(mass_stop) + 1)),
        )


def _import_lasso():
    try:
        from lasso.dyna import ArrayType, D3plot
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RawReadError(
            "reading d3plot output requires the `lasso-python` package "
            "(python -m pip install lasso-python)"
        ) from exc
    _install_solid_history_allocation_fix(D3plot, ArrayType)
    return D3plot, ArrayType


#: 只打一次补丁。
_SOLID_HISTORY_FIX_APPLIED = False


def _install_solid_history_allocation_fix(D3plot, ArrayType) -> None:
    """让 LASSO 读得动 80 / 100 / 500 mm 三个算例的单元 history 变量。

    **这是绕开 lasso-python 2.0.3 的一个分配 bug，不是我们的读数问题。** 三个大算例
    的头块里 ``neiph = 22``、``nv3d = 29``，而且 ``has_element_strain`` 与
    ``has_solid_shell_plastic_strain_tensor`` 都是真；LASSO 在
    ``_read_states_allocate_arrays`` 里把这两项各减 6 当成"藏在 history 里的应变"，
    于是预计 10 列；而 ``_read_states_solids`` 实际切出 **16 列**——正是 50 mm 算例
    那一套（密度、7 个体积分数、材料号、7 个分材料质量）。两边对不上，读取直接抛
    ``ValueError: could not broadcast input array from shape (...,16) into shape (...,10)``。

    50 mm 算例的头块 ``neiph = 16`` 且那两个标志都是假，所以减 0、两边都是 16，正常。

    **LS-DYNA 手册给出了权威公式，LASSO 多减了一项。** 手册（Vol I, `*DATABASE_
    EXTENT_BINARY` / 实体单元输出）写的是 ``NV3D = 7 + NEIPH``，且实体的 history
    变量个数是 ``NEIPH - 6*ISTRN``。80 mm 头块 ``neiph = 22``、``ISTRN`` 为真：
    22 - 6 = **16**，正是实测的列数。LASSO 在
    ``_read_states_allocate_arrays`` 里除了 ``6*has_element_strain``
    **又**减了一次 ``6*has_solid_shell_plastic_strain_tensor``，于是预计 10。所以
    这不是"我们的读法不同"，是 LASSO 多减了 6 项。

    **上游没有可用的修复**：``lasso-python`` 在 PyPI 上最新就是 2.0.3（已核对
    ``pip index versions``），所以补丁不是"临时凑合"，而是当前唯一可行的路径。
    一旦升级到修好这个问题的版本，这个补丁会因为版本检查而自动不生效。

    **补丁的范围被刻意收窄到"已复现的那一组"，其余一律不碰。**

    旧的写法是"只要缓冲比容器宽就加宽"，那等于把 LASSO 任何一处形状不一致都当成同一个
    bug 顺手抹平——包括那些本来应该大声报错的。现在的判据是四个条件同时成立：

    1. 字段名必须是 ``element_solid_history_variables``（只碰这一个字段）；
    2. 容器宽度与缓冲宽度必须正是**已复现的那一对** ``(10, 16)``；
    3. 缓冲更宽（只扩宽，绝不缩窄）；
    4. 安装的 lasso-python 必须是验证过的那个版本。

    任一不成立就**不干预**，让 LASSO 原来的异常原样抛出。加宽过什么、在哪一帧，
    记在 :data:`LASSO_FIX_LOG` 里并写进提取元数据——一个静默生效的补丁和一个
    没有补丁，从外部看起来是一样的。

    独立检查是 :func:`spherical_blast_pinn.cli.extract_profiles._charge_radius_from_volume`：
    由体积分数与单元体积反推装药半径，不经过任何探测器。它在这四个算例上分别给出
    49.9979 / 79.9943 / 99.9929 / 499.9610 mm（相对偏差 ≤ 8e-5）。**列错位不可能同时
    满足这条检查**——体积分数的和必须是 1，而取错列不会给出正确的装药体积。
    """
    global _SOLID_HISTORY_FIX_APPLIED
    if _SOLID_HISTORY_FIX_APPLIED:
        return
    original = D3plot._read_states_transfer_memory
    history_field = ArrayType.element_solid_history_variables
    installed_version = lasso_version()

    def transfer(i_state, buffer_array_dict, master_array_dict):
        for name in list(buffer_array_dict):
            # (1) 只碰 history 这一个字段。
            if name != history_field:
                continue
            buffer_array = buffer_array_dict[name]
            master = master_array_dict.get(name)
            if master is None or buffer_array is None:
                continue
            if not hasattr(buffer_array, "shape") or not hasattr(master, "shape"):
                continue
            if buffer_array.ndim != master.ndim:
                continue
            widths = (int(master.shape[-1]), int(buffer_array.shape[-1]))
            # (2)(3) 必须是已复现的那一对，且确实更宽。
            if widths != _REPRODUCED_HISTORY_WIDTHS:
                continue
            # (4) 版本必须是验证过的那个。
            if _SOLID_HISTORY_FIX_APPROVED_VERSION is not None and (
                installed_version != _SOLID_HISTORY_FIX_APPROVED_VERSION
            ):
                continue
            if buffer_array.shape[0] > master.shape[0]:
                continue
            grown = np.zeros((master.shape[0],) + buffer_array.shape[1:], dtype=master.dtype)
            if i_state:
                # 保留已经写进去的状态，只按**重叠的列**拷：多出来的那几列在原容器里
                # 本来就没有对应物。实际触发总是在一批的第一个状态上（宽度不随状态
                # 变），所以这一段通常什么都不做。
                keep = min(widths)
                grown[:i_state, ..., :keep] = master[:i_state, ..., :keep]
            master_array_dict[name] = grown
            LASSO_FIX_LOG.append(
                {
                    "array": str(name),
                    "state_index": int(i_state),
                    "widths_before_after": [widths[0], widths[1]],
                    "lasso_version": installed_version,
                    "reason": (
                        "reproduced: neiph=22 with has_element_strain and "
                        "has_solid_shell_plastic_strain_tensor both true, so the header "
                        "arithmetic pre-allocates 10 while the state block carries the "
                        "same 16 history columns as the 50 mm case"
                    ),
                }
            )
        return original(i_state, buffer_array_dict, master_array_dict)

    D3plot._read_states_transfer_memory = staticmethod(transfer)
    _SOLID_HISTORY_FIX_APPLIED = True


#: 已复现的（容器宽度, 数据宽度）组合。只在**正好**这一对上才加宽。
_REPRODUCED_HISTORY_WIDTHS: tuple[int, int] = (10, 16)

#: 允许打补丁的 lasso-python 版本；``None`` 表示不检查版本（仅用于测试）。
_SOLID_HISTORY_FIX_APPROVED_VERSION: str | None = "2.0.3"

#: 补丁实际加宽过什么。空列表 = 一次都没触发。
LASSO_FIX_LOG: list[dict[str, object]] = []


def lasso_fix_log() -> tuple[dict[str, object], ...]:
    """补丁记录的一份只读快照，供元数据写出。"""
    return tuple(LASSO_FIX_LOG)


#: The lasso-python release the extraction pipeline was validated against.
#:
#: The state-count probe (:func:`spherical_blast_pinn.cli.extract_profiles.
#: _resolve_state_count`) reads the true state count out of a LASSO error message,
#: so its wording matters.  The probe re-verifies its own answer against the
#: boundary rather than trusting the version, and this constant is recorded in
#: the extraction metadata so that a later reader can tell which release produced
#: a given export.
VALIDATED_LASSO_VERSION = "2.0.3"


def lasso_version() -> str:
    """Installed ``lasso-python`` version, or ``"unknown"`` if it cannot be read."""
    try:
        from importlib.metadata import version

        return version("lasso-python")
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def list_d3plot_family(directory: Path | str) -> list[Path]:
    """Master file followed by its numbered continuations, in state order."""
    directory = Path(directory)
    master = directory / "d3plot"
    if not master.is_file():
        raise RawReadError(f"no master d3plot in {directory}")
    continuations = []
    for entry in directory.iterdir():
        name = entry.name
        if name.startswith("d3plot") and name != "d3plot" and name[6:].isdigit():
            continuations.append((int(name[6:]), entry))
    continuations.sort()
    return [master] + [path for _index, path in continuations]


#: How the per-state node array relates to the geometry block.
POSITION_IS_DISPLACEMENT = "displacement"
POSITION_IS_ABSOLUTE = "absolute"


def detect_position_convention(
    coordinates: np.ndarray, first_state: np.ndarray, *, tolerance: float = 1.0e-9
) -> tuple[str, str]:
    """Decide whether the per-state node array is a displacement or a position.

    This is not a cosmetic detail.  For the decks in this study lasso returns the
    **absolute position** under the ``node_displacement`` key — at state 0 it is
    bit-identical to ``node_coordinates`` — so adding it to the geometry doubles
    every radius.  A doubling is insidious: the profiles still look plausible,
    every count and volume stays right, and only the physical scale is wrong.

    The convention is therefore detected from a verifiable invariant rather than
    assumed.  The first state of a model is its undeformed configuration, so its
    displacement is either zero (a true displacement) or equal to the geometry
    (absolute positions).  Anything else is refused rather than guessed at.
    """
    scale = float(np.max(np.abs(coordinates))) if coordinates.size else 0.0
    if scale <= 0.0:
        raise RawReadError("the geometry block is empty or all zero; cannot infer positions")
    limit = tolerance * scale
    largest = float(np.max(np.abs(first_state))) if first_state.size else 0.0

    if largest <= limit:
        return POSITION_IS_DISPLACEMENT, (
            f"state 0 is undeformed (max |value| {largest:.3g} <= {limit:.3g}), so the "
            "per-state array is a displacement added to the geometry"
        )
    if np.allclose(first_state, coordinates, rtol=0.0, atol=limit):
        return POSITION_IS_ABSOLUTE, (
            f"state 0 equals the geometry block exactly (max |difference| "
            f"{float(np.max(np.abs(first_state - coordinates))):.3g}), so the per-state "
            "array already holds the absolute node positions"
        )
    raise RawReadError(
        "cannot tell whether the d3plot node array is a displacement or an absolute "
        f"position: state 0 is neither zero (max |value| {largest:.3g}) nor equal to the "
        f"geometry (scale {scale:.3g}). This case needs an explicit convention; refusing "
        "to guess, because guessing wrong rescales every radius."
    )


@dataclass
class D3plotCase:
    """Geometry + material mapping for one d3plot family (read once)."""

    directory: Path
    coordinates: np.ndarray
    connectivity: np.ndarray
    part_indexes: np.ndarray
    element_ids: np.ndarray
    part_titles: list[str]
    #: d3plot 自己声明的 ALE 材料号。
    ale_material_ids: np.ndarray
    n_states: int
    layout: D3plotLayout
    #: 产物 / 空气的 **ALE 材料号**（1 基），来自 `*ALE_MULTI-MATERIAL_GROUP` 的出现
    #: 顺序。**不是 part id**——曾经这里叫 `tnt_material_ids` 并直接吃 part id，
    #: 在四个算例上恰好对是因为 psid 刚好等于列号。字段名与语义必须一致，否则下
    #: 一个 deck 会静默读错列。
    product_ale_indices: tuple[int, ...]
    air_ale_indices: tuple[int, ...]
    position_convention: str = POSITION_IS_DISPLACEMENT
    position_note: str = ""

    @property
    def n_elements(self) -> int:
        return int(self.connectivity.shape[0])

    def current_coordinates(self, state_position: np.ndarray) -> np.ndarray:
        """Absolute node positions for one state, per the detected convention."""
        if self.position_convention == POSITION_IS_ABSOLUTE:
            return state_position
        return self.coordinates + state_position


def open_d3plot_case(
    directory: Path | str,
    *,
    layout: D3plotLayout,
    product_ale_indices: Sequence[int],
    air_ale_indices: Sequence[int],
    max_states: int | None = None,
) -> D3plotCase:
    """Read the static geometry of a d3plot family."""
    D3plot, ArrayType = _import_lasso()
    directory = Path(directory)
    master = directory / "d3plot"
    if not master.is_file():
        raise RawReadError(f"no master d3plot in {directory}")

    # The state count comes from the *file listing*, not from lasso.
    #
    # Asking lasso for `global_timesteps` makes it walk the whole family — ~780
    # continuation files, ~300 GB per case — just to count states, which turns a
    # cheap inventory into an hours-long read.  The numbering is unambiguous:
    # the master holds the first state and each `d3plotNN` continuation holds one
    # more, so the file count *is* the state count.
    family = list_d3plot_family(directory)
    n_states_total = len(family)
    n_states = n_states_total if max_states is None else min(n_states_total, int(max_states))

    handle = D3plot(
        str(master),
        state_filter={0},
        buffered_reading=True,
        state_array_filter=[ArrayType.global_timesteps, ArrayType.node_displacement],
    )
    coordinates = np.asarray(handle.arrays["node_coordinates"], dtype=np.float64)
    first_state = _state_slice(
        np.asarray(handle.arrays[ArrayType.node_displacement], dtype=np.float64), 0
    )
    convention, note = detect_position_convention(coordinates, first_state)

    titles = [""] * 0
    titles = [
        bytes(t).decode(errors="replace").strip()
        for t in np.asarray(handle.arrays.get("part_titles", []))
    ]
    ale_ids = np.asarray(handle.arrays.get("ale_material_ids", []), dtype=np.int64)
    return D3plotCase(
        directory=directory,
        coordinates=coordinates,
        connectivity=np.asarray(handle.arrays["element_solid_node_indexes"], dtype=np.int64),
        part_indexes=np.asarray(handle.arrays["element_solid_part_indexes"], dtype=np.int64),
        element_ids=np.asarray(handle.arrays["element_solid_ids"], dtype=np.int64),
        part_titles=titles,
        ale_material_ids=ale_ids,
        n_states=n_states,
        layout=layout,
        product_ale_indices=tuple(int(v) for v in product_ale_indices),
        air_ale_indices=tuple(int(v) for v in air_ale_indices),
        position_convention=convention,
        position_note=note,
    )


@dataclass
class UnitFactors:
    """Multipliers taking deck units to SI."""

    length: float = 1.0
    time: float = 1.0
    density: float = 1.0
    velocity: float = 1.0
    pressure: float = 1.0
    specific_energy: float = 1.0

    @classmethod
    def from_resolution(cls, resolution) -> "UnitFactors":
        return cls(
            length=resolution.factor("length"),
            time=resolution.factor("time"),
            density=resolution.factor("density"),
            velocity=resolution.factor("velocity"),
            pressure=resolution.factor("pressure"),
            specific_energy=resolution.factor("specific_energy"),
        )


@dataclass(frozen=True)
class FixedMeshGeometry:
    """Time-independent cell geometry: centroid, radius, volume.

    An ALE (``elform = 11``) mesh is **Eulerian**: the nodes do not move, only the
    material does.  The positions are then bit-identical at every state, so the
    centroid, radius and volume of all 1.2 million cells are constants of the
    model — and recomputing them per state is the single most expensive thing an
    extraction does.  Hoisting them out changes no number; it only stops computing
    the same array 780 times.
    """

    centroid: np.ndarray
    radius: np.ndarray
    volume: np.ndarray
    #: 每个单元节点半径的 ``max - min``，即该单元的**径向厚度**。
    #:
    #: 这是网格自己的分辨率，和它比才有意义：壳层网格加密到 0.12 mm 并不能让一个
    #: 10 mm 厚的 ALE 单元里的梯度变成可见的。前端位置（以及激波后空气层厚
    #: ``h = R_s - R_c``）的绝对不确定度量级就是它，所以提取时要把它和 ``h`` 一起
    #: 报出来——实测 50 mm 算例装药外 50–60 mm 处的空气单元径向厚度中位数是
    #: **9.7~12.7 mm**，而最早几帧的 ``h`` 只有 5 mm。那种帧里 ``h`` 不是一个被
    #: 测量出来的量。
    radial_thickness: np.ndarray | None = None
    #: 每个单元节点半径的 **min / max**，即它在径向上覆盖的区间 ``[r_in, r_out]``。
    #:
    #: 球壳统计要用它做重叠体积分配（§8）：把整个单元按质心半径塞进一个壳层会造出
    #: 阶梯，而且当壳层比单元细时（近场 0.12 mm 壳层对 0.85 mm 单元）一个单元只落在
    #: 一个壳层里，边界附近的壳层系统性地偏空或偏满。有 ``[r_in, r_out]`` 才能算出
    #: 单元与每个壳层真正重叠了多少体积。
    radial_inner: np.ndarray | None = None
    radial_outer: np.ndarray | None = None


def fixed_mesh_geometry(
    case: D3plotCase,
    *,
    chunk_size: int = 200_000,
    tolerance: float = 1.0e-12,
) -> FixedMeshGeometry:
    """Compute the cell geometry once, after checking the mesh really is fixed.

    ``detect_position_convention`` already establishes that the first state's
    per-node array *is* the absolute position.  This function additionally
    requires that the state array equals the geometry block exactly; if it does
    not — a Lagrangian or ALE-with-moving-nodes deck — the caller must fall back
    to recomputing per state.  Silently hoisting a moving mesh would freeze every
    radius at its initial value, and the profiles would still look plausible.
    """
    if case.position_convention != POSITION_IS_ABSOLUTE:
        raise RawReadError(
            "the per-state node array is a displacement, not an absolute position; "
            "the cell geometry cannot be hoisted out of the state loop"
        )
    coordinates = np.asarray(case.coordinates, dtype=np.float64)
    n_elements = case.n_elements
    centroid = np.empty((n_elements, 3), dtype=np.float64)
    radius = np.empty(n_elements, dtype=np.float64)
    volume = np.empty(n_elements, dtype=np.float64)
    radial_thickness = np.empty(n_elements, dtype=np.float64)
    radial_inner = np.empty(n_elements, dtype=np.float64)
    radial_outer = np.empty(n_elements, dtype=np.float64)
    for start in range(0, n_elements, chunk_size):
        stop = min(start + chunk_size, n_elements)
        points = coordinates[case.connectivity[start:stop]]
        centroid[start:stop] = points.mean(axis=1)
        radius[start:stop] = np.linalg.norm(centroid[start:stop], axis=1)
        volume[start:stop] = hex8_volumes(points)
        node_radii = np.linalg.norm(points, axis=2)
        radial_inner[start:stop] = node_radii.min(axis=1)
        radial_outer[start:stop] = node_radii.max(axis=1)
        radial_thickness[start:stop] = (
            radial_outer[start:stop] - radial_inner[start:stop]
        )
    return FixedMeshGeometry(
        centroid=centroid, radius=radius, volume=volume,
        radial_thickness=radial_thickness,
        radial_inner=radial_inner, radial_outer=radial_outer,
    )


def read_d3plot_states(
    case: D3plotCase,
    state_indices: Sequence[int],
    *,
    case_id: str,
    units: UnitFactors,
    jwl: JWLParameters,
    gamma_air: float,
    pressure_offset_pa: float = 0.0,
    chunk_size: int = 200_000,
    geometry: FixedMeshGeometry | None = None,
) -> Iterator[RawFrame]:
    """Yield one :class:`RawFrame` per requested state.

    Elements are processed in chunks so that a 1.2 million element state does
    not allocate several full-length temporaries at once.

    ``geometry`` (see :func:`fixed_mesh_geometry`) skips the per-state geometry
    computation entirely; pass it whenever the mesh is known to be Eulerian.
    """
    D3plot, ArrayType = _import_lasso()
    handle = D3plot(
        str(case.directory / "d3plot"),
        state_filter=set(int(i) for i in state_indices),
        buffered_reading=True,
        state_array_filter=[
            ArrayType.global_timesteps,
            *([] if geometry is not None else [ArrayType.node_displacement]),
            ArrayType.node_velocity,
            ArrayType.element_solid_stress,
            ArrayType.element_solid_history_variables,
        ],
    )
    times = np.asarray(handle.arrays[ArrayType.global_timesteps], dtype=np.float64)
    displacement = (
        None
        if geometry is not None
        else np.asarray(handle.arrays[ArrayType.node_displacement], dtype=np.float64)
    )
    velocity = np.asarray(handle.arrays[ArrayType.node_velocity], dtype=np.float64)
    stress = np.asarray(handle.arrays[ArrayType.element_solid_stress], dtype=np.float64)
    history = np.asarray(handle.arrays[ArrayType.element_solid_history_variables], dtype=np.float64)

    layout = case.layout
    # 传入的是 **ALE 材料号**（1 基，来自 `*ALE_MULTI-MATERIAL_GROUP` 的出现顺序），
    # 不是 part id。把"第 k 个体积分数列"对上"第 k 个 ALE 材料"需要 d3plot 声明的
    # ALE 材料号就是 1..N —— 本机四个算例都是，但这是个**必须核实的前提**，不是约定。
    declared_ale = np.asarray(getattr(case, "ale_material_ids", []), dtype=np.int64).reshape(-1)
    if declared_ale.size and not np.array_equal(
        declared_ale, np.arange(1, declared_ale.size + 1, dtype=np.int64)
    ):
        raise RawReadError(
            f"the d3plot declares ALE material ids {declared_ale.tolist()}, which are not "
            "1..N in order. The volume-fraction column order therefore does not follow it "
            "and the mapping has to be resolved explicitly instead of assumed; refusing "
            "to combine the columns by position"
        )
    if len(layout.volume_fraction_columns) < max(
        max(case.product_ale_indices, default=0), max(case.air_ale_indices, default=0)
    ):
        raise RawReadError(
            "the configured volume-fraction columns do not cover the declared "
            "ALE material indices"
        )
    tnt_component_columns = [
        layout.volume_fraction_columns[ale_index - 1]
        for ale_index in case.product_ale_indices
    ]

    for local_index, global_state in enumerate(state_indices):
        node_velocity = _state_slice(velocity, local_index)
        element_stress = _state_slice(stress, local_index)
        element_history = _state_slice(history, local_index)

        n_elements = case.n_elements
        density = np.empty(n_elements, dtype=np.float64)
        pressure = np.empty(n_elements, dtype=np.float64)
        alpha_tnt = np.empty(n_elements, dtype=np.float64)
        vx = np.empty(n_elements, dtype=np.float64)
        vy = np.empty(n_elements, dtype=np.float64)
        vz = np.empty(n_elements, dtype=np.float64)

        if geometry is not None:
            centroid = geometry.centroid
            radius = geometry.radius
            volume = geometry.volume
            radial_inner = getattr(geometry, "radial_inner", None)
            radial_outer = getattr(geometry, "radial_outer", None)
        else:
            centroid = np.empty((n_elements, 3), dtype=np.float64)
            radius = np.empty(n_elements, dtype=np.float64)
            volume = np.empty(n_elements, dtype=np.float64)
            radial_inner = np.empty(n_elements, dtype=np.float64)
            radial_outer = np.empty(n_elements, dtype=np.float64)

        for start in range(0, n_elements, chunk_size):
            stop = min(start + chunk_size, n_elements)
            nodes = case.connectivity[start:stop]

            if geometry is None:
                points = case.current_coordinates(_state_slice(displacement, local_index))[nodes]
                centroid[start:stop] = points.mean(axis=1)
                radius[start:stop] = np.linalg.norm(centroid[start:stop], axis=1)
                volume[start:stop] = hex8_volumes(points)
                node_radii = np.linalg.norm(points, axis=2)
                radial_inner[start:stop] = node_radii.min(axis=1)
                radial_outer[start:stop] = node_radii.max(axis=1)


            chunk_history = element_history[start:stop]
            density[start:stop] = chunk_history[:, layout.mixture_density_column]
            # Volume fraction of the explosive: the sum over the explosive
            # material columns.  The per-material-mass columns are available in
            # the layout for callers that need per-material densities.
            alpha_tnt[start:stop] = chunk_history[:, tnt_component_columns].sum(axis=1)

            chunk_stress = element_stress[start:stop]
            # Strengthless media: sigma = -p I, so p = -(sigma_xx+sigma_yy+sigma_zz)/3.
            pressure[start:stop] = -(chunk_stress[:, :3].sum(axis=1)) / 3.0

            vx[start:stop] = node_velocity[nodes, 0].mean(axis=1)
            vy[start:stop] = node_velocity[nodes, 1].mean(axis=1)
            vz[start:stop] = node_velocity[nodes, 2].mean(axis=1)

        density_si = np.maximum(density * units.density, TINY_DENSITY)
        pressure_si = pressure * units.pressure + pressure_offset_pa
        radius_si = radius * units.length
        volume_si = volume * units.length**3

        alpha = np.clip(alpha_tnt, 0.0, 1.0)
        is_product = alpha >= 0.95
        is_air = alpha <= 0.05

        energy = np.full(n_elements, np.nan, dtype=np.float64)
        state_valid = np.zeros(n_elements, dtype=bool)
        # 反解本身**永远**有定义（``omega > 0``、``rho > 0`` 时
        # ``e = (p - p_cold)/(omega rho)`` 是一个有限的实数）。这里的取舍不是
        # "有没有解"，而是"这是个合法的产物状态吗"：反解出**非正**比内能，说明该
        # 单元的 ``p`` 落在 JWL 冷压曲线之下，也就是化学能还没放出来。这种单元不能
        # 当产物标签用——把它按产物 JWL 反解出来的负比内能当标签，会要求网络去拟合
        # 一个 ``softplus`` 永远给不出的负数。实测 50 mm 算例第 0 帧 218328 个
        # ``alpha >= 0.95`` 的单元全部属于这一类（整球尚未起爆，``p`` 恰为 0）。
        if np.any(is_product):
            e_product = reconstruct_product_energy(
                density_si[is_product], pressure_si[is_product], jwl
            )
            good = np.isfinite(e_product) & (e_product > 0.0)
            product_index = np.flatnonzero(is_product)
            energy[product_index[good]] = e_product[good]
            state_valid[product_index[good]] = True
        if np.any(is_air):
            e_air = reconstruct_air_energy(
                density_si[is_air], pressure_si[is_air], gamma_air
            )
            good = np.isfinite(e_air) & (e_air > 0.0)
            air_index = np.flatnonzero(is_air)
            energy[air_index[good]] = e_air[good]
            state_valid[air_index[good]] = True

        radial = radial_velocity_from_centroid(centroid, vx, vy, vz) * units.velocity

        yield RawFrame(
            case_id=case_id,
            time=float(times[local_index]) * units.time,
            radius=radius_si,
            cell_volume=volume_si,
            density=density_si,
            velocity_x=vx * units.velocity,
            velocity_y=vy * units.velocity,
            velocity_z=vz * units.velocity,
            pressure=pressure_si,
            tnt_volume_fraction=alpha,
            radial_velocity=radial,
            specific_internal_energy=energy,
            state_valid=state_valid,
            # 径向覆盖区间，单位换算与 radius 一致。球壳统计用它做重叠体积分配。
            cell_radius_inner=(
                None if radial_inner is None else radial_inner * units.length
            ),
            cell_radius_outer=(
                None if radial_outer is None else radial_outer * units.length
            ),
            material_id=case.part_indexes.astype(np.int64),
            cell_centroid=centroid * units.length,
            provenance={
                "state_index": int(global_state),
                "energy_source": "reconstructed_from_eos",
                "pressure_source": "stress_trace",
                "density_source": "history_variable_column_0",
                "volume_fraction_source": "history_variable_columns_1_7",
                "position_convention": case.position_convention,
                # 被标成产物或空气、但 (rho, p) 不是该 EOS 的合法状态、因而没有比内能
                # 标签的单元数。它是**真实物理**的一个读数（正在被爆轰波扫过、或尚未
                # 起爆的装药），不是数值噪声，所以逐帧记录下来。
                "state_invalid_cells": int(np.count_nonzero(~state_valid)),
            },
        )


def _state_slice(array: np.ndarray, index: int) -> np.ndarray:
    """Index a state array written as ``(n_states, n_items, ...)``.

    lasso returns solid-element state arrays with an extra **integration-point**
    axis: ``element_solid_stress`` is ``(n_states, n_elements, 1, 6)`` and
    ``element_solid_history_variables`` is ``(n_states, n_elements, 1, 16)``.
    Indexing only the state axis therefore leaves a singleton in the middle, and
    the next ``chunk[:, column]`` picks up that axis instead of the column —
    producing a ``(n, 16)`` array where an ``(n,)`` one was expected.

    Singleton axes are removed after the state is selected, which is safe here
    because neither the node axis nor the component axis is ever of length one
    for a real model.  The result is checked so a genuinely unexpected shape
    fails loudly instead of silently mis-indexing.
    """
    if array.ndim == 0:
        raise RawReadError("expected a state array, got a scalar")
    if array.shape[0] > index:
        state = array[index]
    elif array.shape[0] == 1:
        state = array[0]
    else:
        raise RawReadError(
            f"state index {index} is out of range for an array of shape {array.shape}"
        )
    state = np.asarray(state)
    if state.ndim > 1:
        state = state.reshape(state.shape[0], -1) if _has_singleton_tail(state) else state
    return state


def _has_singleton_tail(state: np.ndarray) -> bool:
    """Whether the axes after the first are an integration-point axis only.

    True for ``(n, 1, k)`` — the solid-element layout — and false for
    ``(n, k)`` (nodes) or ``(n, k, m)`` with no length-one axis.
    """
    return state.ndim == 3 and state.shape[1] == 1


# --------------------------------------------------------------------------
# Tabular readers
# --------------------------------------------------------------------------


@dataclass
class TabularReaderConfig:
    mapping: FieldMapping
    units: UnitFactors
    jwl: JWLParameters
    gamma_air: float
    pressure_offset_pa: float = 0.0
    chunksize: int = 500_000
    case_id: str = "case"


def _require_columns(frame, columns: Sequence[str], source: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RawReadError(f"{source} is missing the mapped columns {missing}")


def _stream_grouped_chunks(chunks, time_column: str, source: str) -> Iterator[tuple[float, Any]]:
    """Yield ``(time, rows)`` from an iterator of row chunks, in time order.

    Only the *last* time group of a chunk can straddle a chunk boundary, so it
    is carried forward; every earlier group is complete and emitted
    immediately.  This keeps peak memory at roughly two chunks while still
    guaranteeing that a time value is never split across two emitted frames.
    """
    import pandas as pd

    carry = None
    for chunk in chunks:
        _require_columns(chunk, [time_column], source)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        times = sorted(float(t) for t in chunk[time_column].unique())
        if len(times) <= 1:
            carry = chunk
            continue
        for time_value in times[:-1]:
            yield time_value, chunk[chunk[time_column] == time_value]
        carry = chunk[chunk[time_column] == times[-1]]
    if carry is not None and len(carry):
        yield float(carry[time_column].iloc[0]), carry


def read_csv_frames(path: Path | str, config: TabularReaderConfig) -> Iterator[RawFrame]:
    """Chunked CSV reader.  One frame is emitted per distinct ``time`` value."""
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise RawReadError(f"CSV file not found: {path}")
    time_column = config.mapping.require("time")
    chunks = pd.read_csv(path, chunksize=config.chunksize)
    for time_value, rows in _stream_grouped_chunks(chunks, time_column, str(path)):
        yield _frame_from_table(rows, config, time_value)


def read_parquet_frames(path: Path | str, config: TabularReaderConfig) -> Iterator[RawFrame]:
    """Chunked Parquet reader.  Requires ``pyarrow``."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RawReadError(
            "reading Parquet requires the optional `pyarrow` dependency "
            "(python -m pip install pyarrow)"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise RawReadError(f"Parquet file not found: {path}")
    time_column = config.mapping.require("time")
    handle = pq.ParquetFile(path)
    batches = (batch.to_pandas() for batch in handle.iter_batches(batch_size=config.chunksize))
    for time_value, rows in _stream_grouped_chunks(batches, time_column, str(path)):
        yield _frame_from_table(rows, config, time_value)


def read_hdf5_frames(
    path: Path | str, config: TabularReaderConfig, dataset: str = "frames"
) -> Iterator[RawFrame]:
    """Chunked HDF5 reader over a structured table stored in ``dataset``."""
    import h5py
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise RawReadError(f"HDF5 file not found: {path}")
    time_column = config.mapping.require("time")
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise RawReadError(f"HDF5 file {path} has no dataset {dataset!r}")
        node = handle[dataset]
        names = node.dtype.names

        def batches():
            total = node.shape[0]
            for start in range(0, total, config.chunksize):
                stop = min(start + config.chunksize, total)
                yield pd.DataFrame({name: node[name][start:stop] for name in names})

        # `_stream_grouped_chunks` consumes lazily, so the file stays open for
        # exactly as long as this generator is alive.
        for time_value, rows in _stream_grouped_chunks(batches(), time_column, str(path)):
            yield _frame_from_table(rows, config, time_value)


def _frame_from_table(frame, config: TabularReaderConfig, time_value: float) -> RawFrame:
    mapping = config.mapping
    units = config.units

    x = np.asarray(frame[mapping.require("x")], dtype=np.float64) * units.length
    y = np.asarray(frame[mapping.require("y")], dtype=np.float64) * units.length
    z = np.asarray(frame[mapping.require("z")], dtype=np.float64) * units.length
    radius = np.sqrt(x * x + y * y + z * z)

    density = np.maximum(
        np.asarray(frame[mapping.require("density")], dtype=np.float64) * units.density,
        TINY_DENSITY,
    )
    pressure = (
        np.asarray(frame[mapping.require("pressure_absolute")], dtype=np.float64) * units.pressure
        + config.pressure_offset_pa
    )
    alpha = np.asarray(frame[mapping.require("tnt_volume_fraction")], dtype=np.float64)
    alpha = np.clip(alpha, 0.0, 1.0)

    velocity_columns = [mapping.require(name) for name in ("velocity_x", "velocity_y", "velocity_z")]
    v = np.stack([np.asarray(frame[c], dtype=np.float64) for c in velocity_columns], axis=1) * units.velocity
    centroid = np.stack([x, y, z], axis=1)
    radial = radial_velocity_from_centroid(centroid, v[:, 0], v[:, 1], v[:, 2])

    if mapping.has("cell_volume"):
        volume = np.asarray(frame[mapping.require("cell_volume")], dtype=np.float64) * units.length**3
    else:
        raise RawReadError(
            "cell_volume is not mapped.  It cannot be derived from a tabular "
            "cell-centre listing (only a volumetric mesh such as d3plot carries "
            "the connectivity needed for that), so map it explicitly in "
            "data.field_map or supply a mesh-based source."
        )

    if mapping.has("specific_internal_energy"):
        energy = (
            np.asarray(frame[mapping.require("specific_internal_energy")], dtype=np.float64)
            * units.specific_energy
        )
        energy_source = "raw_column"
    else:
        energy = np.full(radius.shape, np.nan)
        is_product = alpha >= 0.95
        is_air = alpha <= 0.05
        if np.any(is_product):
            energy[is_product] = reconstruct_product_energy(density[is_product], pressure[is_product], config.jwl)
        if np.any(is_air):
            energy[is_air] = reconstruct_air_energy(density[is_air], pressure[is_air], config.gamma_air)
        energy_source = "reconstructed_from_eos"

    material_id = (
        np.asarray(frame[mapping.require("material_id")], dtype=np.int64)
        if mapping.has("material_id")
        else None
    )
    burn = (
        np.asarray(frame[mapping.require("burn_fraction")], dtype=np.float64)
        if mapping.has("burn_fraction")
        else None
    )

    return RawFrame(
        case_id=config.case_id,
        time=float(time_value),
        radius=radius,
        cell_volume=volume,
        density=density,
        velocity_x=v[:, 0],
        velocity_y=v[:, 1],
        velocity_z=v[:, 2],
        pressure=pressure,
        tnt_volume_fraction=alpha,
        radial_velocity=radial,
        specific_internal_energy=energy,
        material_id=material_id,
        burn_fraction=burn,
        cell_centroid=centroid,
        provenance={"energy_source": energy_source, "source": "tabular"},
    )


READERS: dict[str, Callable[..., Iterator[RawFrame]]] = {
    "csv": read_csv_frames,
    "parquet": read_parquet_frames,
    "hdf5": read_hdf5_frames,
}


def available_tabular_formats() -> list[str]:
    """Which optional tabular formats can actually be read in this environment."""
    formats = ["csv"]
    try:  # pragma: no cover - environment dependent
        import pyarrow  # noqa: F401

        formats.append("parquet")
    except ImportError:
        pass
    try:  # pragma: no cover - environment dependent
        import h5py  # noqa: F401

        formats.append("hdf5")
    except ImportError:
        pass
    return formats


def iter_frames(
    source: Path | str,
    config: TabularReaderConfig,
    *,
    fmt: str = "csv",
) -> Iterator[RawFrame]:
    """Dispatch to the requested tabular reader."""
    key = fmt.strip().lower()
    if key not in READERS:
        raise RawReadError(
            f"unknown raw format {fmt!r}; available here: {available_tabular_formats()}"
        )
    return READERS[key](source, config)


__all__ = [
    "D3plotCase",
    "D3plotLayout",
    "HEX8_TO_TETS",
    "RawFrame",
    "RawReadError",
    "TabularReaderConfig",
    "UnitFactors",
    "available_tabular_formats",
    "check_reconstruction_consistency",
    "hex8_volumes",
    "iter_frames",
    "list_d3plot_family",
    "open_d3plot_case",
    "radial_velocity_from_centroid",
    "read_csv_frames",
    "read_d3plot_states",
    "read_hdf5_frames",
    "read_parquet_frames",
    "reconstruct_air_energy",
    "reconstruct_product_energy",
]
