"""从 d3plot 重新提取时空剖面，**并保留材料分区**。

为什么要重新提取
----------------

已有的 ``raw_line.npz`` 只写了 ``t/x/rho/u/P``。它下游的一切——接触面位置、材料
归属、哪些点可以当标签——都只能靠**密度最陡下降**这个启发式判据重新猜。实测结果
是 50 mm 的 101 个"有效"帧里有 86 个给出的 ``R_c < R_0``，而 ``R_c < R_0`` 在物理
上不可能。这不是数据的问题：**算例本来就是 ALE 多物质，每个单元都带材料体积分数**，
只是提取时被丢掉了。

本命令把这些体积分数读出来并保存成一份配套文件。分区因此从"启发式"变成"读数"。

算例的结构（四份卡片一致）
--------------------------

* ``*SECTION_SOLID`` 的 ``elform = 11``：一点 ALE 多物质实体单元，**网格固定在空间
  中**，材料在网格之间流动；
* 7 个 part，其中 4 个是 ``*MAT_HIGH_EXPLOSIVE_BURN`` + ``*EOS_JWL``（同一套参数，
  是建模工具把装药切成几块的结果），3 个是 ``*MAT_NULL`` + ``*EOS_LINEAR_POLYNOMIAL``
  的空气；
* ``*ALE_MULTI-MATERIAL_GROUP`` 把这 7 个 part 都登记为 ALE 物质；
* 计算域是一个**八分之一球**（3 组 ``*BOUNDARY_SPC_SET`` 各约束一个方向的平动）。

d3plot 的每个实体单元的 history variable 有 16 个槽位，实测布局是
``[混合密度, 7 个体积分数, 1 个索引, 7 个分物质质量]``：体积分数在**每个单元上
精确求和为 1**，初始时刻没有混合单元。
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..constants import UNIT_SYSTEMS
from ..data import compatibility as compat
from ..data.shock_detect import detect_shock
from ..data.extracted_case import FieldMapping
from ..data.interface_reconstruction import reconstruct_interface
from ..data.kfile_parser import parse_deck
from ..data.raw_reader import (
    VALIDATED_LASSO_VERSION,
    D3plotLayout,
    RawFrame,
    RawReadError,
    UnitFactors,
    fixed_mesh_geometry,
    lasso_fix_log,
    lasso_version,
    list_d3plot_family,
    open_d3plot_case,
    read_d3plot_states,
)
from ..data.shell_average import (
    AIR,
    PRODUCT,
    PURE_AIR_THRESHOLD,
    PURE_PRODUCT_THRESHOLD,
    ShellGrid,
    ShellProfile,
    mixture_fraction_profile,
    overlap_assignment,
    overlap_assignment_from_arrays,
    shell_average,
    volume_weighted_profile,
)
from ..physics import material as mat
from ._common import add_common_arguments, configure_stdout, load_config_from_args, write_json


class ExtractionError(RuntimeError):
    """提取无法继续。"""


@dataclass
class DeckContext:
    """从卡片读到的、提取需要知道的全部事实。"""

    deck_path: Path
    unit_system: str
    length_to_m: float
    time_to_s: float
    density_to_si: float
    velocity_to_si: float
    pressure_to_si: float
    product_part_ids: tuple[int, ...]
    air_part_ids: tuple[int, ...]
    #: 产物 / 空气在 **ALE 材料号**（1 基）下的编号，来自 `*ALE_MULTI-MATERIAL_GROUP`
    #: 的**出现顺序**。history 体积分数的第 k 列对应 ALE 材料号 k。
    #:
    #: 这是材料映射的唯一来源。**不要拿 part id 当列号**：四个算例上两者恰好相等
    #: （psid = 1..7 且列也按 1..7 排），但那是卡片刚好这么写的，不是格式约定。
    product_ale_indices: tuple[int, ...]
    air_ale_indices: tuple[int, ...]
    detonation_velocity_m_s: float
    initial_detonation: tuple[float, float, float] | None
    termination_time: float | None
    source: dict[str, Any] = field(default_factory=dict)

    def part_ids_for_material(self, material_id: int) -> bool:
        return material_id in self.product_part_ids


def _unit_system_from_deck(summary) -> tuple[str, Any]:
    """由炸药密度判定单位制；判不出就停。"""
    explosive = summary.explosive_materials()
    if not explosive:
        raise ExtractionError("the deck has no *MAT_HIGH_EXPLOSIVE_BURN; cannot continue")
    card = summary.require_explosive()
    raw_density = getattr(card, "density", None)
    if raw_density is None or raw_density <= 0.0:
        raise ExtractionError("the deck's explosive card has no usable density")
    for name in ("mm_s_t", "si", "cm_us_g", "mm_ms_kg"):
        system = UNIT_SYSTEMS.get(name)
        if system is None:
            continue
        converted = float(raw_density) * system.density_to_si
        if abs(converted - mat.PRODUCT_REFERENCE_DENSITY) / mat.PRODUCT_REFERENCE_DENSITY < 1.0e-3:
            return name, system
    raise ExtractionError(
        f"could not identify the unit system from the explosive density {raw_density!r}; "
        "guessing units would scale every extracted field by an unknown factor"
    )


def read_deck_context(deck_path: Path | str) -> DeckContext:
    """读卡片，返回提取所需的单位、part 归属与几何/时间参数。"""
    deck_path = Path(deck_path)
    if not deck_path.is_file():
        raise ExtractionError(f"{deck_path} is not a file")
    summary = parse_deck(deck_path, geometry=False)
    name, system = _unit_system_from_deck(summary)

    explosive_ids = set(summary.explosive_material_ids())
    air_ids = set(summary.air_material_ids())
    product_parts: list[int] = []
    air_parts: list[int] = []
    for part_id, part in sorted(summary.parts.items()):
        if part.material_id in explosive_ids:
            product_parts.append(int(part_id))
        elif part.material_id in air_ids:
            air_parts.append(int(part_id))
    if not product_parts:
        raise ExtractionError("no part uses an explosive material; cannot identify the charge")
    if not air_parts:
        raise ExtractionError("no part uses an air material; cannot identify the ambient")

    card = summary.require_explosive()
    raw_velocity = getattr(card, "detonation_velocity", None) or 0.0

    # ALE 材料号由 `*ALE_MULTI-MATERIAL_GROUP` 的顺序决定，不由 part id 推。
    # 三种介质各自登记：产物、空气、以及既不是产物也不是空气的（本算例没有，
    # 但出现时必须报出来，而不是悄悄算进空气里）。
    product_ale = tuple(summary.ale_material_indices_for_parts(product_parts))
    air_ale = tuple(summary.ale_material_indices_for_parts(air_parts))
    if not product_ale:
        raise ExtractionError(
            "no *ALE_MULTI-MATERIAL_GROUP entry maps to an explosive part; the "
            "volume-fraction column order cannot be established"
        )
    if not air_ale:
        raise ExtractionError(
            "no *ALE_MULTI-MATERIAL_GROUP entry maps to an air part; the air side of "
            "the volume fractions cannot be established"
        )
    n_ale = len(summary.ale_groups)
    other_ale = [
        index for index in range(1, n_ale + 1)
        if index not in product_ale and index not in air_ale
    ]
    ale_kinds = {index: summary.ale_material_kind(index) for index in range(1, n_ale + 1)}

    return DeckContext(
        deck_path=deck_path,
        unit_system=name,
        length_to_m=system.length_to_si,
        time_to_s=system.time_to_si,
        density_to_si=system.density_to_si,
        velocity_to_si=system.velocity_to_si,
        pressure_to_si=system.pressure_to_si,
        product_part_ids=tuple(product_parts),
        air_part_ids=tuple(air_parts),
        product_ale_indices=product_ale,
        air_ale_indices=air_ale,
        detonation_velocity_m_s=float(raw_velocity) * system.velocity_to_si,
        initial_detonation=summary.initial_detonation,
        termination_time=summary.termination_time,
        source={
            "deck": str(deck_path),
            "unit_system": name,
            "product_part_ids": product_parts,
            "air_part_ids": air_parts,
            "ale_material_mapping": {
                "source": "*ALE_MULTI-MATERIAL_GROUP, in card order",
                "n_groups": len(summary.ale_groups),
                "resolved_as": [g.resolved_as for g in summary.ale_groups],
                "product_ale_indices": list(product_ale),
                "air_ale_indices": list(air_ale),
                "other_ale_indices": other_ale,
                "kind_by_ale_index": {str(k): v for k, v in ale_kinds.items()},
                "note": (
                    "the ALE material index is the card order, NOT part_id - 1; the "
                    "volume-fraction column for ALE material k is column k"
                ),
            },
            "initial_detonation": list(summary.initial_detonation or ()),
            "initial_detonation_time": summary.initial_detonation_time,
            "termination_time": summary.termination_time,
            "d3plot_interval": summary.d3plot_interval,
            "control_ale": summary.control_ale,
        },
    )


# --------------------------------------------------------------------------
# 提取
# --------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """一次提取的结果摘要。"""

    case_id: str
    output_dir: Path
    n_states: int
    n_shells: int
    times: np.ndarray
    charge_radius_m: float
    implied_charge_radius_m: float
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "output_dir": str(self.output_dir),
            "n_states": int(self.n_states),
            "n_shells": int(self.n_shells),
            "time_first_s": float(self.times[0]),
            "time_last_s": float(self.times[-1]),
            "charge_radius_m": self.charge_radius_m,
            "implied_charge_radius_m": self.implied_charge_radius_m,
            **self.metadata,
        }


def _part_index_to_deck_id(summary) -> dict[int, int]:
    """``element_solid_part_indexes`` 是 0 基的；映射回卡片的 part id。

    映射按卡片里 part 的出现顺序，而不是 ``index + 1``。这里两者的结果相同，
    但当卡片不是从 1 连续编号时，只有按顺序才对。
    """
    ordered = sorted(summary.parts)
    return {index: int(part_id) for index, part_id in enumerate(ordered)}


def _charge_radius_from_volume(
    volume: np.ndarray, alpha_product: np.ndarray, *, symmetry_multiplier: float
) -> float:
    """由产物体积反推装药半径。

    ``V_full = m * sum(alpha_j V_j)``，其中 ``m`` 是计算域相对整球的几何倍数
    （八分之一球时是 8）。这是对提取结果的一条独立检查：它不经过任何探测器，
    只用体积分数和单元体积。
    """
    wedge_volume = float(np.sum(alpha_product * volume))
    if wedge_volume <= 0.0:
        raise ExtractionError("the product volume came out non-positive")
    return float((3.0 * symmetry_multiplier * wedge_volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def _symmetry_multiplier(centroids: np.ndarray) -> float:
    """计算域相对整球的几何倍数。

    八分之一球（三个对称面）、四分之一（两个）、二分之一（一个）或整球。判据是
    节点坐标的符号组合数：ALE 楔形网格的节点在某一个坐标象限上全部同号，说明
    那个方向被对称面切掉了一半。
    """
    signs = np.sign(centroids)
    active = []
    for axis in range(3):
        column = signs[:, axis]
        nonzero = column[column != 0.0]
        if nonzero.size == 0:
            continue
        # 只出现一种符号，说明这个方向是对称面切出来的半个域。
        active.append(1 if np.all(nonzero == nonzero[0]) else 0)
    return float(2 ** sum(active))


#: 一个不可能存在的状态号。查询它一定失败，而失败信息里带着真实的状态总数。
_IMPOSSIBLE_STATE = 100_000_000


def _resolve_state_count(d3plot_dir: Path, *, file_count: int) -> tuple[int, str]:
    """状态总数。**文件数不是状态数**，而且这里不再有回退到文件数的分支。

    实测：50 mm 有 780 个 d3plot 文件而只有 **779** 个状态，另外三个算例是 783 个
    文件 / **782** 个状态——每个都差一个。按文件数去请求最后一个状态会直接报错；
    更糟的是，如果某个 continuation 文件装了两个状态，按文件数取会**静默错位**，
    而错位的时间戳不会报错，只会把每条曲线平移。

    拿权威值的办法是请求一个不可能存在的状态号，让 LASSO 在报错信息里给出真正
    的上限，然后**独立复核**这个上限：``n-1`` 必须能读出恰好一个时间戳，``n`` 必须
    读不出来。探针依赖 LASSO 报错的措辞，复核不依赖——它直接验证结论本身，所以
    LASSO 换版本、改措辞时会在这里停下来，而不是安静地退回文件数。
    """
    from ..data.raw_reader import _import_lasso, lasso_version

    D3plot, ArrayType = _import_lasso()
    version = lasso_version()

    def _read(state_filter) -> int:
        handle = D3plot(
            str(Path(d3plot_dir) / "d3plot"),
            state_filter=state_filter,
            buffered_reading=True,
            state_array_filter=[ArrayType.global_timesteps],
        )
        return int(np.asarray(handle.arrays[ArrayType.global_timesteps]).reshape(-1).size)

    if _IMPOSSIBLE_STATE <= int(file_count):
        raise ExtractionError(
            f"the d3plot family has {file_count} files, which already reaches the "
            f"impossible-state sentinel {_IMPOSSIBLE_STATE}; the state-count probe "
            "cannot be formed. Refusing to fall back to the file count."
        )

    probed = False
    try:
        _read({_IMPOSSIBLE_STATE})
    except ValueError as exc:
        match = re.search(r"maximum number of states of '(\d+)'", str(exc))
        if not match:
            raise ExtractionError(
                "the state-count probe failed: LASSO raised a ValueError for an "
                f"impossible state index but the message did not carry a state count "
                f"(lasso-python {version}): {exc!r}. The extraction stops here on "
                "purpose — the only alternative is to trust the file count, and the "
                "file count is wrong by one on every case on this machine."
            ) from exc
        n_states = int(match.group(1))
        probed = True
    except Exception as exc:  # pragma: no cover - defensive
        raise ExtractionError(
            f"the state-count probe failed ({type(exc).__name__}: {exc}); refusing to "
            "fall back to the d3plot file count"
        ) from exc
    if not probed:
        raise ExtractionError(
            f"requesting state {_IMPOSSIBLE_STATE} did not fail, so the d3plot family "
            f"claims at least that many states; the probe is not measuring what it "
            f"thinks it is (lasso-python {version})"
        )

    # 独立复核：不看报错措辞，直接验证边界两侧的行为。
    if n_states < 1:
        raise ExtractionError(f"the state-count probe returned {n_states}")
    try:
        last = _read({n_states - 1})
    except Exception as exc:  # pragma: no cover - defensive
        raise ExtractionError(
            f"the state-count probe returned {n_states}, but state {n_states - 1} could "
            f"not be read ({type(exc).__name__}: {exc})"
        ) from exc
    if last != 1:
        raise ExtractionError(
            f"the state-count probe returned {n_states}, but state {n_states - 1} "
            f"yielded {last} timesteps instead of exactly one"
        )
    try:
        beyond = _read({n_states})
    except ValueError:
        beyond = None
    except Exception as exc:  # pragma: no cover - defensive
        raise ExtractionError(
            f"the state-count probe returned {n_states}, but reading state {n_states} "
            f"failed with {type(exc).__name__} instead of the expected out-of-range "
            f"ValueError: {exc}"
        ) from exc
    if beyond is not None:
        raise ExtractionError(
            f"the state-count probe returned {n_states}, but state {n_states} is "
            f"readable ({beyond} timesteps); the probe is off by at least one"
        )

    note = f"lasso state-count probe + boundary verification (lasso-python {version})"
    if n_states != int(file_count):
        note += f"; file count {file_count} differs by {int(file_count) - n_states}"
    return n_states, note


def _iter_frames(
    case,
    case_id: str,
    states: Sequence[int],
    units,
    jwl,
    gamma_air: float,
    batch: int,
    geometry=None,
):
    """按**受控小批次**读取状态，逐帧产出。

    ``buffered_reading=True`` 不是"只保留一帧"的同义词：LASSO 仍然会为**每一个
    被选中的状态**分配结果数组。把 780 个状态一次交给它，峰值内存就是 780 份
    实体单元 history/stress/node 数组——50 mm 算例每个状态约 0.3 GB，那是两百多
    GB，进程会先被系统杀掉。分批之后峰值是 ``batch`` 份，读满一批就释放。

    ``geometry`` 是 :func:`raw_reader.fixed_mesh_geometry` 预先算好的单元几何。
    ALE 网格固定不动，所以单元中心、半径和体积都是常量；每帧重算一遍 122 万个
    六面体的体积是整条提取里最贵的一步，而且算的是同一个数组。
    """
    states = list(states)
    batch = max(1, int(batch))
    for start in range(0, len(states), batch):
        chunk = [int(s) for s in states[start:start + batch]]
        for frame in read_d3plot_states(
            case, chunk, case_id=case_id, units=units, jwl=jwl, gamma_air=gamma_air,
            geometry=geometry,
        ):
            yield frame


def _crossing_radius(centres: np.ndarray, values: np.ndarray, level: float) -> float:
    """``values`` 从上方穿过 ``level`` 的半径，线性插值；没有穿过时返回 ``nan``。"""
    radius, _, _ = _crossing_radius_and_gap(centres, values, level)
    return radius


def _crossing_radius_and_gap(
    centres: np.ndarray, values: np.ndarray, level: float
) -> tuple[float, float, int]:
    """同上，另外返回插值**跨过的距离**与跨过的**空壳层个数**。

    这两个数是判据的一部分，不是诊断的装饰。插值在跨越处没有数据时，会在两个相距很远
    的壳层之间画一条直线，得出一个位于"中间某处"的半径——它看起来是个正经的数，其实
    那一带根本没有单元。实测 80 / 100 / 500 mm 算例：装药外第一个空气单元的径向厚度
    分别是 **208–361 / 260–452 / 1303–2258 mm**，体积分数从 1 直接跳到 0，0.5 穿越被
    插值到 131.97 / 164.96 / 824.80 mm，而装药半径是 80 / 100 / 500 mm。

    判据用的是**空壳层个数**而不是跨距与单元厚度之比：单个巨型单元内部处处是空的，跨距
    （104 mm）反而**小于**单元厚度（128 mm），比大小会把这一档漏掉。空壳层个数直接说
    的是"这个穿越是跨着一段没有数据的地方插出来的"。
    """
    values = np.asarray(values, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan"), float("nan"), 0
    finite_centres, finite_values = centres[finite], values[finite]
    below = np.flatnonzero(finite_values < level)
    if below.size == 0 or below[0] == 0:
        return float("nan"), float("nan"), 0
    index = int(below[0])
    v0, v1 = float(finite_values[index - 1]), float(finite_values[index])
    r0, r1 = float(finite_centres[index - 1]), float(finite_centres[index])
    gap = abs(r1 - r0)
    lo, hi = min(r0, r1), max(r0, r1)
    inside = (centres > lo) & (centres < hi)
    empty_inside = int(np.count_nonzero(inside & ~finite))
    if v1 == v0:
        return r0, gap, empty_inside
    return r0 + (r1 - r0) * (level - v0) / (v1 - v0), gap, empty_inside


#: 激波位置的三种状态。用整数而不是布尔量：**"没测到"和"测到了但分辨不出"是
#: 两个不同的事实**，一个布尔量会把它们合并成"激波不可用"，而它们的处理方法不同。
_SHOCK_NOT_DETECTED = 0
_SHOCK_SUBCELL = 1
_SHOCK_RESOLVED = 2


def _cell_thickness_table(radii, thickness, *, n_bins: int = 512):
    """把"ALE 单元径向厚度 vs 半径"压成一张小表，只建一次。

    单元厚度随半径单调而光滑（实测：装药内 0.85 mm，装药外 9.7 mm，90–120 mm 处
    31 mm，3 m 以上 105 mm），所以按半径分箱取均值再插值就够了。逐帧去 122 万个单元
    里找最近的那个是最直观的写法，但那是 779 × 122 万次比较——为了一个光滑的一维
    函数付三十分钟。

    入参必须是**同一套单位**（SI）。返回 ``(半径, 厚度)`` 或 ``None``。
    """
    if radii is None or thickness is None:
        return None
    radii = np.asarray(radii, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    finite = np.isfinite(radii) & np.isfinite(thickness) & (thickness > 0.0)
    if not np.any(finite):
        return None
    r, t = radii[finite], thickness[finite]
    order = np.argsort(r)
    r, t = r[order], t[order]
    edges = np.linspace(float(r[0]), float(r[-1]), int(n_bins) + 1)
    which = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, int(n_bins) - 1)
    totals = np.bincount(which, weights=t, minlength=int(n_bins))
    counts = np.bincount(which, minlength=int(n_bins)).astype(np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    occupied = counts > 0.0
    if not np.any(occupied):
        return None
    return centres[occupied], totals[occupied] / counts[occupied]


def _mesh_cell_thickness(table, radius: float) -> float:
    """半径 ``radius`` 处的 ALE 单元径向厚度；``table`` 为 ``None`` 时返回 ``nan``。"""
    if table is None or not np.isfinite(radius):
        return float("nan")
    centres, values = table
    return float(np.interp(float(radius), centres, values))


def _shock_radius(
    centres: np.ndarray,
    pressure: np.ndarray,
    alpha: np.ndarray,
    *,
    ambient_pressure: float,
    relative_tolerance: float = 1.0e-2,
    min_below: int = 3,
) -> tuple[float, dict[str, float]]:
    """主激波半径：与接触面相连的那一段**连续超压区**的外缘。

    三个刻意的选择：

    * 阈值相对**环境压力**，不用全域最大压力。全域最大值落在产物中心（~1e10 Pa），
      拿它的千分之一当阈值等于把判据交给产物区，而激波在空气侧的过压低了六七个
      数量级。
    * 只在 ``alpha < 0.5`` 的**空气侧**搜索。产物区内压力始终很高，把产物算进来
      只会让判据变成"产物外缘"，而那正是接触面。
    * **取连续超压区的外缘，不取"最外侧的超压壳层"。** 这条是实测出来的：激波以外
      的远场压力是数值噪声，实测在 ``p/p_amb = 1.000`` 上下摆动、幅度可达 1.5%，
      而阈值是 1%。于是"最外侧超压单元"落在噪声尖峰上——50 mm 算例第 10 状态测到
      95.2 mm（2500 层）/ 101.0 mm（5000 层），而真实的超压区外缘在 58.2 mm 附近，
      两者差了 40 mm。也就是说旧判据测的是噪声，不是激波，而且两套网格各测噪声的
      不同尖峰，看起来像是"分辨率不够"，其实是不稳健。同一帧按连通段取外缘，两套
      网格只差 0.1 mm。

    返回 ``(半径, 诊断)``。求不到时半径是 ``nan``，而诊断里的峰值过压说明**为什么**
    求不到——早期空气层比 ALE 单元还薄的时候，过压被壳层平均摊薄到阈值以下，这是
    真实的测量下限，不是可以糊过去的噪声。
    """
    threshold = ambient_pressure * (1.0 + abs(float(relative_tolerance)))
    finite = np.isfinite(pressure) & np.isfinite(alpha)
    air = np.flatnonzero(finite & (alpha < 0.5))
    if air.size == 0:
        return float("nan"), {"air_side_shells": 0.0, "peak_overpressure": float("nan"),
                              "run_outer_radius": float("nan"), "stopped_at_drop": 0.0}
    p_air = np.asarray(pressure, dtype=np.float64)[air]
    over = p_air > threshold
    peak_ratio = float(np.max(p_air) / ambient_pressure) if p_air.size else float("nan")
    diagnostics = {"air_side_shells": float(air.size), "peak_overpressure": peak_ratio,
                   "run_outer_radius": float("nan"), "stopped_at_drop": 0.0}
    if not over.any():
        return float("nan"), diagnostics

    # 从最内侧的超压壳层（紧挨接触面，压力最高处）向外走，直到压力**持续**低于阈值。
    start = int(np.flatnonzero(over)[0])
    end = start
    below = 0
    for k in range(start + 1, air.size):
        if over[k]:
            end = k
            below = 0
        else:
            below += 1
            if below >= max(1, int(min_below)):
                break
    diagnostics["run_outer_radius"] = float(centres[air[end]])

    # 外缘：在最后一个超压壳层与它之后第一个低于阈值的壳层之间线性插值。
    edge = float(centres[air[end]])
    for k in range(end + 1, air.size):
        if not over[k]:
            r0, r1 = float(centres[air[end]]), float(centres[air[k]])
            p0, p1 = float(p_air[end]), float(p_air[k])
            if r1 > r0 and p1 != p0:
                edge = r0 + (r1 - r0) * (threshold - p0) / (p1 - p0)
                diagnostics["stopped_at_drop"] = 1.0
            break
    return float(edge), diagnostics


def extract_case(
    *,
    case_id: str,
    d3plot_dir: Path | str,
    deck: DeckContext,
    output_dir: Path | str,
    charge_radius_m: float,
    grid: ShellGrid,
    layout: D3plotLayout | None = None,
    states: Sequence[int] | None = None,
    stride: int = 1,
    max_states: int | None = None,
    gamma_air: float = mat.AIR_GAMMA,
    batch_states: int = 16,
    n_states: int | None = None,
    assume_fixed_geometry: bool = True,
    progress_every: int = 0,
) -> ExtractionResult:
    """把一个算例的 d3plot 家族提取成剖面 + 材料分区。"""
    d3plot_dir = Path(d3plot_dir)
    layout = layout or D3plotLayout()
    summary = parse_deck(deck.deck_path, geometry=False)
    part_map = _part_index_to_deck_id(summary)

    family = list_d3plot_family(d3plot_dir)
    n_files = len(family)
    # **文件数不等于状态数。** 一个 continuation 文件可以装多个状态（``*DATABASE_
    # BINARY_D3PLOT`` 的 ``N`` 参数），所以"每个文件一帧"只是一个特定算例的巧合。
    # 这里用 LASSO 自己数出来的状态数作为权威值，并把两者都记进元数据；不一致时
    # 以 LASSO 为准，同时报告差异。
    if n_states is not None:
        n_available, count_source = int(n_states), "explicit --n-states override"
    else:
        n_available, count_source = _resolve_state_count(d3plot_dir, file_count=n_files)
    if states is None:
        chosen = list(range(0, n_available, max(1, int(stride))))
    else:
        chosen = sorted({int(s) for s in states if 0 <= int(s) < n_available})
    if max_states is not None:
        chosen = chosen[: int(max_states)]
    if not chosen:
        raise ExtractionError(f"no d3plot state selected out of {n_available}")

    case = open_d3plot_case(
        d3plot_dir,
        layout=layout,
        product_ale_indices=deck.product_ale_indices,
        air_ale_indices=deck.air_ale_indices,
        max_states=n_available,
    )
    # ALE 材料号必须落在 d3plot 声明的 ALE 材料表里。数量对不上就说明卡片与
    # d3plot 不是同一套模型，此时继续读只会把体积分数按错的列组合起来。
    declared_ale = np.asarray(getattr(case, "ale_material_ids", []), dtype=np.int64).reshape(-1)
    if declared_ale.size:
        highest = int(max(deck.product_ale_indices + deck.air_ale_indices))
        if highest > declared_ale.size:
            raise ExtractionError(
                f"the deck declares ALE material {highest} but the d3plot carries "
                f"{declared_ale.size} ALE material(s); the mapping cannot be trusted"
            )

    # ``element_solid_part_indexes`` 是 0 基的下标；卡片的 part id 可能是任意整数。
    deck_ids = np.asarray(
        [part_map.get(int(index), -1) for index in np.unique(case.part_indexes)], dtype=np.int64
    )
    lookup = {int(index): int(part_map.get(int(index), -1)) for index in np.unique(case.part_indexes)}
    element_part_ids = np.asarray([lookup[int(v)] for v in case.part_indexes], dtype=np.int64)
    del deck_ids

    units = UnitFactors(
        length=deck.length_to_m,
        time=deck.time_to_s,
        density=deck.density_to_si,
        velocity=deck.velocity_to_si,
        pressure=deck.pressure_to_si,
        specific_energy=deck.velocity_to_si**2,
    )
    jwl = mat.product_jwl()

    n_shells = grid.n_shells
    centres = grid.centres
    times: list[float] = []
    surfaces = {
        PRODUCT: {
            "density": [], "velocity": [], "specific_energy": [],
            "pressure": [], "volume_fraction": [], "count": [], "coverage": [],
        },
        AIR: {
            "density": [], "velocity": [], "specific_energy": [],
            "pressure": [], "volume_fraction": [], "count": [], "coverage": [],
        },
    }
    shell_alpha: list[np.ndarray] = []
    shell_pressure: list[np.ndarray] = []
    shell_count: list[np.ndarray] = []
    shell_empty: list[np.ndarray] = []
    shell_contaminated: list[np.ndarray] = []
    contact_radius: list[float] = []
    shock_radius: list[float] = []
    #: 检测状态（§10）：confirmed / doubtful / not_detected。**与监督资格分开。**
    shock_status: list[str] = []
    #: 该帧是否有资格给出空气监督范围。只有 confirmed 才有。
    shock_supervision: list[bool] = []
    shock_reliable: list[tuple[float, float]] = []
    shock_ds: list[float] = []
    shock_rh_momentum: list[float] = []
    shock_rh_energy: list[float] = []
    shock_compression: list[float] = []
    shock_entropy_jump: list[float] = []
    shock_n_candidates: list[int] = []
    shock_cell_thickness: list[float] = []
    shock_h_over_cell: list[float] = []
    contact_cell_thickness: list[float] = []
    contact_gap_list: list[float] = []
    contact_holes_list: list[int] = []
    contact_state: list[int] = []
    # §9：接触面由**逐帧几何重构**给出，壳平均穿越只作辅助对照。这几列把重构
    # 自己的证据一并带出去：定位状态、位置不确定性、混合带、原始分辨率、是否
    # 亚单元位置、以及辅助穿越值。
    contact_status: list[str] = []
    contact_uncertainty: list[float] = []
    contact_band_lo: list[float] = []
    contact_band_hi: list[float] = []
    contact_native_resolution: list[float] = []
    contact_sub_cell: list[bool] = []
    contact_mixing_cells: list[int] = []
    contact_crossing: list[float] = []
    contact_supervision: list[bool] = []
    contact_reconstruction_reason: list[str] = []
    mixed_fraction: list[float] = []
    unreacted_cells: list[int] = []
    state_indices: list[int] = []

    shock_provenance: dict[str, Any] = {}
    implied_radius = float("nan")
    symmetry = float("nan")

    # ALE 网格固定：单元几何是模型的常量，先算一次。
    cell_thickness_table = None
    try:
        if not assume_fixed_geometry:
            raise RawReadError("--no-fixed-geometry: recompute the cell geometry every state")
        geometry = fixed_mesh_geometry(case)
    except RawReadError:
        geometry = None
        mesh_note = "recomputed per state (the mesh is not Eulerian-fixed)"
    else:
        mesh_note = "computed once (Eulerian mesh: node positions are state-independent)"
        # 单元径向厚度转成 SI 再建表：``geometry`` 里的长度还是网格单位。
        if geometry.radial_thickness is not None:
            cell_thickness_table = _cell_thickness_table(
                geometry.radius * units.length, geometry.radial_thickness * units.length
            )

    # ALE 网格固定时，单元→壳层的重叠表是**模型的常量**：整表只建一次。
    # 实测建一次约 1.8 s（1.2e6 个单元 × 37 个壳层，2040 万条配对），逐帧重建成
    # 779 次就是二十多分钟的纯浪费。网格不是 Eulerian 固定时只能逐帧建。
    fixed_assignment = None
    if geometry is not None and getattr(geometry, "radial_inner", None) is not None:
        fixed_assignment = overlap_assignment_from_arrays(
            geometry.radial_inner * units.length,
            geometry.radial_outer * units.length,
            np.abs(geometry.volume) * units.length**3,
            grid,
        )

    for frame in _iter_frames(
        case, case_id, chosen, units, jwl, gamma_air, batch_states, geometry=geometry
    ):
        if not np.isfinite(implied_radius):
            # 独立检查：由体积分数与单元体积反推装药半径。它不经过任何探测器。
            symmetry = _symmetry_multiplier(np.asarray(frame.cell_centroid, dtype=np.float64))
            implied_radius = _charge_radius_from_volume(
                np.abs(np.asarray(frame.cell_volume, dtype=np.float64)),
                np.asarray(frame.tnt_volume_fraction, dtype=np.float64),
                symmetry_multiplier=symmetry,
            )
            # 壳层的**几何体积**要按被建模的那一块算，不能按整球算。八分之一球模型
            # 上用整球体积当分母会让覆盖率恰好小 8 倍，而 0.125 读起来像"大部分没
            # 覆盖"。对称份额就是上面那条独立检查已经算出来的那个数。
            grid = replace(grid, symmetry_multiplier=float(symmetry))

        # 重叠体积配对只依赖几何、不依赖介质，**每帧建一次**给四个消费者共用。
        # ALE 网格固定时整张表是模型的常量，所以整个算例只建一次（见下面
        # ``fixed_assignment``）；网格不固定时只能逐帧建。
        assignment = (
            fixed_assignment if fixed_assignment is not None
            else overlap_assignment(frame, grid)
        )

        profiles: dict[str, ShellProfile] = {
            medium: shell_average(frame, grid, medium, assignment=assignment)
            for medium in (PRODUCT, AIR)
        }
        times.append(float(frame.time))
        state_indices.append(int(frame.provenance["state_index"]))
        if progress_every and len(times) % int(progress_every) == 0:
            print(
                f"[extract] {case_id}: {len(times)}/{len(chosen)} states, "
                f"t = {float(frame.time):.6g} s",
                flush=True,
            )
        for medium in (PRODUCT, AIR):
            profile = profiles[medium]
            surfaces[medium]["density"].append(profile.density)
            surfaces[medium]["velocity"].append(profile.radial_velocity)
            surfaces[medium]["specific_energy"].append(profile.specific_internal_energy)
            surfaces[medium]["pressure"].append(profile.pressure)
            surfaces[medium]["volume_fraction"].append(profile.volume_fraction)
            surfaces[medium]["count"].append(profile.count)
            surfaces[medium]["coverage"].append(profile.coverage)

        # 分类与前沿定位一律用**全部单元**的体积加权量。用只对纯产物单元求平均的
        # 那个 alpha 会有两个后果：壳层里没有产物单元时它返回 ``nan``（不是 0），
        # 于是那些壳层既不是产物也不是空气；而且它在产物侧恒等于 ~1，永远不会
        # 穿过 0.5，接触面根本定不出来。
        mixture = mixture_fraction_profile(frame, grid, assignment=assignment)
        alpha_profile = np.asarray(mixture.alpha, dtype=np.float64)
        pressure_profile, _, _ = volume_weighted_profile(
            frame, grid, np.asarray(frame.pressure, dtype=np.float64), assignment=assignment
        )
        shell_alpha.append(alpha_profile)
        shell_pressure.append(pressure_profile)
        shell_count.append(np.asarray(mixture.count, dtype=np.int64))
        shell_empty.append(~np.asarray(mixture.validity, dtype=bool))
        # §9：接触面用**逐帧几何重构**定位（球对称 PLIC，由体积分数闭式切分），
        # 不再用 ``alpha = 0.5`` 的壳平均穿越——穿越值只作辅助对照。
        #
        # 定位不了时 ``reconstruction.radius`` 是 ``nan`` 而 ``interval`` 仍然有效；
        # 这一帧因此拿不到接触面标签，而不是拿到一个看起来正常的数。
        reconstruction = reconstruct_interface(
            grid.edges,
            alpha_profile,
            symmetry_multiplier=float(grid.symmetry_multiplier),
        )
        crossing_r, contact_gap, contact_holes = _crossing_radius_and_gap(
            centres, alpha_profile, 0.5
        )
        contact_r = float(reconstruction.radius)
        contact_radius.append(contact_r)
        contact_gap_list.append(contact_gap)
        contact_holes_list.append(contact_holes)
        contact_status.append(reconstruction.status)
        contact_uncertainty.append(float(reconstruction.position_uncertainty))
        contact_band_lo.append(float(reconstruction.mixing_band[0]))
        contact_band_hi.append(float(reconstruction.mixing_band[1]))
        contact_native_resolution.append(float(reconstruction.native_resolution))
        contact_sub_cell.append(bool(reconstruction.sub_cell_position))
        # 混合带跨了几个**壳层**。壳层网格远细于 ALE 单元时这个数会很大——那说明
        # 混合带反映的是"壳层在重采样一个粗单元"，不是界面本身被抹开了。带内
        # 平均壳层宽度 = 带宽 / 这个数，读的人据此判断不确定性的来源。
        contact_mixing_cells.append(int(len(reconstruction.mixing_cells)))
        contact_crossing.append(float(crossing_r))
        contact_reconstruction_reason.append(str(reconstruction.reason))
        # 混合带里的壳层若本身是空的（跨着一段没有数据的地方），这个定位就不是
        # 被网格分辨出来的，即便拓扑上看着连续。
        _band = reconstruction.mixing_cells
        _band_clear = all(not bool(shell_empty[-1][i]) for i in _band if i < shell_empty[-1].size)
        contact_supervision.append(
            bool(reconstruction.supervision_eligible and _band_clear)
        )

        # **被"没有状态"的单元污染的壳层不能算有效。** 单元级的比内能是在
        # ``raw_reader`` 里反解的，反解出非正值的单元已经被置为 ``nan``；而壳层平均
        # 会**静默地跳过** ``nan``，于是同一个壳层里剩下的有效单元会被平均成一个
        # 看起来很正常的数，掩盖掉"这个壳层里有一部分根本不是产物状态"这件事。
        # 所以这里按**体积**单独算一遍被污染的比例，任何非零比例都让整个壳层失效。
        state_valid = getattr(frame, "state_valid", None)
        if state_valid is None:
            contaminated_profile = np.zeros_like(alpha_profile)
            unreacted_cells.append(0)
        else:
            invalid_cells = ~np.asarray(state_valid, dtype=bool)
            contaminated_profile, _, _ = volume_weighted_profile(
                frame, grid, invalid_cells.astype(np.float64), assignment=assignment
            )
            contaminated_profile = np.asarray(contaminated_profile, dtype=np.float64)
            unreacted_cells.append(int(np.count_nonzero(invalid_cells)))
        shell_contaminated.append(contaminated_profile)

        # 主激波标签：**压力梯度候选 + RH 核查**（§10），不再用"最外侧超压点"。
        #
        # 空气侧用纯空气剖面（``profiles[AIR]``）：混合带和产物侧都不参与，两侧状态
        # 因此一定落在"材料身份可靠"的连续空气数据上。``alpha_profile`` 仍传进去，
        # 好让检测器自己也能排除混合带。
        air_profile = profiles[AIR]
        air_thickness = np.full(n_shells, np.nan)
        if cell_thickness_table is not None:
            air_thickness = np.array(
                [_mesh_cell_thickness(cell_thickness_table, float(c)) for c in centres]
            )
        detection = detect_shock(
            centres,
            np.asarray(air_profile.pressure, dtype=np.float64),
            np.asarray(air_profile.density, dtype=np.float64),
            np.asarray(air_profile.radial_velocity, dtype=np.float64),
            alpha_profile,
            np.asarray(air_profile.validity, dtype=bool),
            gamma=gamma_air,
            ambient_pressure=mat.AIR_PRESSURE,
            cell_thickness=air_thickness,
        )
        main = detection.main
        shock_r = (
            float(main.representative_radius) if main is not None else float("nan")
        )
        shock_radius.append(shock_r)
        # **检测状态与监督资格分开记。** 只有 confirmed 才有可靠位置范围，才够格当标签。
        shock_status.append(detection.status)
        shock_supervision.append(detection.supervision_radius_range is not None)
        shock_reliable.append(
            detection.supervision_radius_range
            if detection.supervision_radius_range is not None
            else (float("nan"), float("nan"))
        )
        shock_ds.append(float(main.shock_speed) if main is not None else float("nan"))
        shock_rh_momentum.append(float(main.rh_momentum) if main is not None else float("nan"))
        shock_rh_energy.append(float(main.rh_energy) if main is not None else float("nan"))
        shock_compression.append(float(main.compression) if main is not None else float("nan"))
        shock_entropy_jump.append(
            float(main.entropy_jump) if main is not None else float("nan")
        )
        shock_n_candidates.append(int(len(detection.candidates)))
        if not shock_provenance:
            shock_provenance.update(detection.provenance)
        # 激波后空气层厚 h = R_s - R_c 与**网格自己的**径向单元厚度比（不是壳层宽度）。
        cell_thickness = _mesh_cell_thickness(cell_thickness_table, shock_r)
        shock_cell_thickness.append(cell_thickness)
        contact_r = float(contact_radius[-1])
        contact_thickness = _mesh_cell_thickness(cell_thickness_table, contact_r)
        contact_cell_thickness.append(contact_thickness)
        # 接触面也走同一套判据：0.5 穿越是**跨着一段没有数据的地方**插出来的，就说明那一带
        # 没有被网格分辨，报出来的半径是"两个壳层中间某处"。80/100/500 mm 三个算例装药外
        # 第一个空气单元的径向厚度分别是 208–361 / 260–452 / 1303–2258 mm，正是这一档。
        # §9：状态由**重构自己的**资格给出（定位成功 + 混合带连续 + 两侧邻域相正确
        # + 带内没有空壳层）。以前这里是"0.5 穿越有没有跨过空洞"——那是在给一个
        # 本来就不该当几何位置用的数做体检。
        if not contact_supervision[-1]:
            contact_state.append(
                _SHOCK_NOT_DETECTED if not np.isfinite(contact_r) else _SHOCK_SUBCELL
            )
        else:
            contact_state.append(_SHOCK_RESOLVED)
        h_now = (
            float(shock_r) - contact_r
            if np.isfinite(shock_r) and np.isfinite(contact_r)
            else float("nan")
        )
        # h 与**网格**的关系单独记：它说明"这个厚度是不是被网格分辨"，与"检测器给出
        # 了什么状态"是两件事，两者都要留下。
        if not np.isfinite(h_now) or not np.isfinite(cell_thickness) or cell_thickness <= 0:
            shock_h_over_cell.append(float("nan"))
        else:
            shock_h_over_cell.append(float(h_now / cell_thickness))
        alpha_cells = np.asarray(frame.tnt_volume_fraction, dtype=np.float64)
        mixed_fraction.append(
            float(
                np.mean(
                    (alpha_cells > PURE_AIR_THRESHOLD) & (alpha_cells < PURE_PRODUCT_THRESHOLD)
                )
            )
        )

    # -- 组装 --------------------------------------------------------------
    time_array = np.asarray(times, dtype=np.float64)
    order = np.argsort(time_array)
    time_array = time_array[order]

    def stack(medium: str, key: str) -> np.ndarray:
        return np.asarray(surfaces[medium][key], dtype=np.float64)[order]

    alpha_product = np.asarray(shell_alpha, dtype=np.float64)[order]

    # 每个壳层的材料归属：按**全部单元**的体积加权体积分数判定，不按密度。
    contact_array = np.asarray(contact_radius, dtype=np.float64)[order]
    shock_array = np.asarray(shock_radius, dtype=np.float64)[order]
    shock_status_array = np.asarray(shock_status, dtype="<U16")[order]
    shock_supervision_array = np.asarray(shock_supervision, dtype=bool)[order]
    shock_status_str = np.asarray(shock_status_array, dtype=object)
    contact_status_array = np.asarray(contact_status, dtype="<U16")[order]
    contact_sup_arrays = np.asarray(contact_supervision, dtype=bool)[order]
    contact_sub_cell_arrays = np.asarray(contact_sub_cell, dtype=bool)[order]
    empty = np.asarray(shell_empty, dtype=bool)[order]

    material = np.full(alpha_product.shape, 2, dtype=np.int64)
    material[alpha_product >= PURE_PRODUCT_THRESHOLD] = 0
    material[alpha_product <= PURE_AIR_THRESHOLD] = 1
    # 混合带（0.05 < alpha < 0.95）没有自己的标签：按多数侧给一个标签便于阅读，
    # 但下面会被标成无效，永远不进监督。
    mixed = (alpha_product > PURE_AIR_THRESHOLD) & (alpha_product < PURE_PRODUCT_THRESHOLD)
    material[mixed] = 2

    band_width = 2.0 * float(np.median(np.diff(grid.edges)))
    valid = (material != 2) & ~empty
    # **没有可信激波就没有被监督的空气域。** 规范里被监督的空气是 ``R_c < r < R_s``；
    # ``R_s`` 没有**核实过的**位置时，这条带子没有边界，整个空气侧无法与环境态区分——
    # 而环境态按规范是**无标签**的。此时把空气侧整体撤标，而不是把两米以内的均匀大气
    # 当成空气监督送进训练。
    #
    # 判据是**监督资格**而不是"检测到没有"：一个 doubtful 的候选（跃迁带跨不过一个原始
    # 单元、或两侧状态不足以定 Ds）不能给出可靠范围，因此同样不够格。
    no_shock = ~shock_supervision_array
    if np.any(no_shock):
        air_labelled = no_shock[:, None] & (material == 1)
        material[air_labelled] = 2
        valid[air_labelled] = False
    # 被"没有状态"的单元污染的壳层整层作废（见上面 contaminated_profile 的说明）。
    contaminated = np.asarray(shell_contaminated, dtype=np.float64)[order]
    contaminated_shells = np.isfinite(contaminated) & (contaminated > 0.0)
    valid = valid & ~contaminated_shells
    for i in range(material.shape[0]):
        if np.isfinite(shock_array[i]):
            outside = centres >= shock_array[i]
            material[i, outside] = 2
            valid[i, outside] = False
        # 前沿附近各留一个保护带，与 §11"剔除展宽区"一致。
        for front in (contact_array[i], shock_array[i]):
            if np.isfinite(front):
                valid[i, np.abs(centres - front) <= band_width] = False

    # 剖面：两介质各自填充，界面处各取单侧。
    #
    # **填充条件必须与分类条件逐字相同。** 之前这里用的是"纯产物单元加权"的
    # alpha（没有产物单元的壳层上是 ``nan``），而分类用的是补过之后的标签，两者
    # 不一致，于是空气壳层被标成有效、场量却整片是 ``nan``——一个"识别出了 N 个
    # 空气壳层"的假象。下面的断言就是为了让这种不一致不可能再发生。
    coverage_product = np.asarray(surfaces[PRODUCT]["coverage"], dtype=np.float64)[order]
    coverage_air = np.asarray(surfaces[AIR]["coverage"], dtype=np.float64)[order]

    rho = np.full((time_array.size, n_shells), np.nan)
    vel = np.full_like(rho, np.nan)
    energy = np.full_like(rho, np.nan)
    pressure = np.full_like(rho, np.nan)
    product_here = material == 0
    air_here = material == 1
    for key, target in (
        ("density", rho), ("velocity", vel), ("specific_energy", energy), ("pressure", pressure)
    ):
        target[product_here] = stack(PRODUCT, key)[product_here]
        target[air_here] = stack(AIR, key)[air_here]

    finite = np.isfinite(rho) & np.isfinite(vel) & np.isfinite(energy) & np.isfinite(pressure)
    # **有效必须蕴含"有状态"。** 一个标着产物或空气、却四项场量全空的壳层不是监督，
    # 它只会被采样器逐个丢掉——数据损失里的空气组于是悄无声息地消失。所以有效性
    # 直接由有限性收口，而不是由标签收口。
    labelled_without_state = (material != 2) & ~finite
    # 逐字段有效性（§11.1）：先把"这个**位置**能不能用"记下来，再各自与场量的
    # 有限性相交。分开之前必须先留一份，否则 ``valid`` 一旦并进四项的联合有限性，
    # 三个逐字段掩码会与它逐字相同——那就只是把同一张掩码抄了三遍。
    #
    # 真实差别是有的：产物膨胀到低密度、压力转负之后 EOS 反解没有实数解，``e`` 与
    # ``p`` 会在密度、速度仍然可靠的壳层上变成 ``nan``。联合掩码会把那些壳层整层
    # 丢掉，而它们的位置信息本来是好用的。
    location_valid = valid.copy()
    valid = valid & finite
    empty_shells = np.asarray(shell_empty, dtype=bool)[order] & (material == 2)
    labelled_without_state_fraction = (
        float(np.count_nonzero(labelled_without_state)) / max(1, int(np.count_nonzero(material != 2)))
    )

    # 逐字段有效性。数值上每项 = 位置可用 且 该字段有限。
    field_valid_density = location_valid & np.isfinite(rho)
    field_valid_velocity = location_valid & np.isfinite(vel)
    field_valid_energy = location_valid & np.isfinite(energy)
    field_valid_pressure = location_valid & np.isfinite(pressure)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # ``e`` 一并写出。它是**重建量，不是 DYNA 的输出量**：这套 d3plot 里实体单元
    # 根本没有比内能结果（LASSO 的 ``element_solid_*`` 系列里只有应力、应变、history
    # 变量，没有 internal energy），所以纸上只有 ``(rho, p)``，比内能只能按各材料的
    # EOS 反解。反解本身永远有定义——``e = (p - p_cold)/(omega rho)`` 是除法，不是求根
    # ——所以问题从来不是"解不出来"，而是**解出来的非正值不能当标签用**（见下）。
    #
    # 这里写的是壳层平均的**单元级**反解能量（每个单元先用自己那份 ``(rho, p)`` 反解，
    # 再按体积加权），比拿壳层平均后的 ``(rho, p)`` 再解一次准：后者先把非线性关系
    # 平均掉了，而且会把一个非产物单元混进平均里再反解。
    np.savez_compressed(
        output_dir / "raw_line.npz",
        t=time_array, x=centres, rho=rho, u=vel, P=pressure, e=energy,
    )
    np.savez_compressed(
        output_dir / "material.npz",
        alpha_product=alpha_product,
        material=material,
        valid=valid,
        # 逐字段有效性（§11.1）。``valid`` 是"四项联合可用"，这三个是"这一项自己
        # 可用"。下游按字段各自取点，不再因为 e 缺失而丢掉同一位置的 rho 与 u。
        field_valid_density=field_valid_density,
        field_valid_velocity=field_valid_velocity,
        field_valid_energy=field_valid_energy,
        field_valid_pressure=field_valid_pressure,
        shell_count=np.asarray(shell_count, dtype=np.int64)[order],
        # §9：接触面由**逐帧几何重构**给出（球对称 PLIC，由产物体积分数闭式切分）。
        # 下面这一组是重构自己的证据：定位状态、位置不确定性、混合带、原始分辨率、
        # 是否亚单元位置。少了它们，下游只能看到一个半径，分不清"定位到了混合带
        # 里的界面"和"跨着空洞插出来的一个数"。
        contact_radius=np.asarray(contact_radius, dtype=np.float64)[order],
        contact_reconstruction_status=np.asarray(contact_status, dtype="<U16")[order],
        contact_position_uncertainty_m=np.asarray(contact_uncertainty, dtype=np.float64)[order],
        contact_mixing_band_inner_m=np.asarray(contact_band_lo, dtype=np.float64)[order],
        contact_mixing_band_outer_m=np.asarray(contact_band_hi, dtype=np.float64)[order],
        contact_native_resolution_m=np.asarray(contact_native_resolution, dtype=np.float64)[order],
        contact_mixing_cell_count=np.asarray(contact_mixing_cells, dtype=np.int64)[order],
        contact_is_sub_cell=np.asarray(contact_sub_cell, dtype=bool)[order],
        contact_has_supervision=np.asarray(contact_supervision, dtype=bool)[order],
        contact_reconstruction_reason=np.asarray(contact_reconstruction_reason, dtype="<U200")[order],
        # 辅助对照：``alpha = 0.5`` 的壳平均穿越。§9 明确它**不能**继续充当最终
        # 几何重构，所以它与几何位置分开存放，谁也不能冒充谁。
        contact_shell_crossing_radius_m=np.asarray(contact_crossing, dtype=np.float64)[order],
        # 接触面同样受网格限制。一个不用任何模型的直接读数：t = 0 时接触面按定义
        # 就在 R_0，实测偏差见 metadata 的 contact_initial_offset_m。
        contact_cell_thickness_m=np.asarray(contact_cell_thickness, dtype=np.float64)[order],
        contact_interpolation_gap_m=np.asarray(contact_gap_list, dtype=np.float64)[order],
        contact_empty_shells_in_gap=np.asarray(contact_holes_list, dtype=np.int64)[order],
        contact_state=np.asarray(contact_state, dtype=np.int64)[order],
        # 每帧、每介质、每壳层的**覆盖率**：该壳层里这一侧介质真正占据的体积占
        # 壳层几何体积的比例。壳层平均只在覆盖到的部分上成立，覆盖率说明它成立在
        # 多大范围上；不看它就无法区分"这一层平均下来是 3 kg/m^3"和"这一层只有
        # 3% 被覆盖，剩下的是外推"。
        coverage_product=coverage_product,
        coverage_air=coverage_air,
        shock_radius=shock_array,
        # 激波"找到了吗"和"分辨得出吗"是两件事，所以三态码、峰值过压、以及
        # h 与单元厚度之比都写出来。早期空气层比 ALE 单元还薄时，壳层平均会把过压摊
        # 到阈值以下（那时 R_s 是 nan，peak 值说明这是测量下限而不是漏检）；即便测到了，
        # h 小于一个单元厚度时它也不是一个被测量出来的量。
        # 激波：状态字符串、监督资格、可靠位置范围、以及 RH 核查的实测值。
        # 没有这些，下游只能看到一个半径，分不清"核实过的激波"和"一个勉强看见的尖峰"。
        shock_status=np.asarray(shock_status, dtype="<U16")[order],
        shock_has_supervision=np.asarray(shock_supervision, dtype=bool)[order],
        shock_reliable_inner_m=np.asarray([v[0] for v in shock_reliable], dtype=np.float64)[order],
        shock_reliable_outer_m=np.asarray([v[1] for v in shock_reliable], dtype=np.float64)[order],
        shock_speed_m_s=np.asarray(shock_ds, dtype=np.float64)[order],
        shock_rh_momentum=np.asarray(shock_rh_momentum, dtype=np.float64)[order],
        shock_rh_energy=np.asarray(shock_rh_energy, dtype=np.float64)[order],
        shock_compression=np.asarray(shock_compression, dtype=np.float64)[order],
        shock_entropy_jump=np.asarray(shock_entropy_jump, dtype=np.float64)[order],
        shock_n_candidates=np.asarray(shock_n_candidates, dtype=np.int64)[order],
        shock_cell_thickness_m=np.asarray(shock_cell_thickness, dtype=np.float64)[order],
        shock_h_over_cell=np.asarray(shock_h_over_cell, dtype=np.float64)[order],
        mixed_cell_fraction=np.asarray(mixed_fraction, dtype=np.float64)[order],
        # 每帧被"尚未具备产物状态"的单元按体积污染的比例。非零即整层作废。
        contaminated_shell_fraction=contaminated,
        state_invalid_cells=np.asarray(unreacted_cells, dtype=np.int64)[order],
        state_index=np.asarray(state_indices, dtype=np.int64)[order],
    )

    implied = float(implied_radius)
    shock_status_array = np.asarray(shock_status, dtype="<U16")[order]
    shock_supervision_array = np.asarray(shock_supervision, dtype=bool)[order]
    shock_rh_m_array = np.asarray(shock_rh_momentum, dtype=np.float64)[order]
    shock_rh_e_array = np.asarray(shock_rh_energy, dtype=np.float64)[order]
    shock_compression_array = np.asarray(shock_compression, dtype=np.float64)[order]
    thickness_array = np.asarray(shock_cell_thickness, dtype=np.float64)[order]
    h_over_cell_array = np.asarray(shock_h_over_cell, dtype=np.float64)[order]
    contact_thickness_array = np.asarray(contact_cell_thickness, dtype=np.float64)[order]
    contact_state_array = np.asarray(contact_state, dtype=np.int64)[order]
    contact_gap_array = np.asarray(contact_gap_list, dtype=np.float64)[order]
    # t = 0 时接触面按定义在 R_0，所以这个偏差是网格给出的**直接**误差读数，
    # 不经过任何探测器、任何模型。
    contact_offset = None
    # 同一个读数也给辅助的 0.5 穿越算一遍。两个数并排放在元数据里，"几何重构比
    # 壳平均穿越好在哪"就不再是一句主张，而是一个可以自己复算的对比。
    contact_crossing_offset = None
    state_array = np.asarray(state_indices, dtype=np.int64)[order]
    first_contact = np.asarray(contact_radius, dtype=np.float64)[order]
    first_crossing = np.asarray(contact_crossing, dtype=np.float64)[order]
    if state_array.size and state_array[0] == 0 and np.isfinite(first_contact[0]):
        contact_offset = float(first_contact[0]) - float(charge_radius_m)
        if np.isfinite(first_crossing[0]):
            contact_crossing_offset = float(first_crossing[0]) - float(charge_radius_m)

    metadata = {
        "case_id": case_id,
        "R_0": float(charge_radius_m),
        "t_0": float(time_array[0]),
        "t_end": float(time_array[-1]),
        "n_x": int(n_shells),
        "n_t": int(time_array.size),
        "x_end": float(grid.outer_radius),
        "x_min": float(grid.inner_radius),
        # §8 / §15：球对称份额。体积等效半径、覆盖率、以及"八分之一模型展开到整球"
        # 都靠它做分母；不写出来，下游只能猜一个 8 或者干脆不折算。
        "symmetry_factor": float(grid.symmetry_multiplier),
        "symmetry_share": float(1.0 / float(grid.symmetry_multiplier)),
        "symmetry_factor_note": (
            "full sphere / modelled wedge: 8 for an octant model, 1 for a full sphere. "
            "Written by the extractor from the shell grid it actually used"
        ),
        "rho_TNT": mat.PRODUCT_REFERENCE_DENSITY,
        "field_map": FieldMapping().mapping,
        # 每个字段的来源。`e` 必须在这里说清楚是反解量：它不是测量量，而且它是否
        # 存在取决于该单元是否处于所声明 EOS 的合法状态。
        "field_provenance": {
            "t": "d3plot global timesteps, converted by the deck's time unit",
            "x": "cell centroid radius; ALE mesh, computed once and state-independent",
            "rho": "d3plot history variable column 0 (mixture density)",
            "u": "node velocity averaged per cell, projected on the radial direction",
            "P": "negative trace of the solid element stress (strengthless: sigma = -p I)",
            "e": (
                "RECONSTRUCTED, not an LS-DYNA output: this d3plot carries no "
                "solid-element internal energy, so e = (p - p_cold(V))/(omega rho) "
                "for products and e = p/((gamma-1) rho) for air, inverted per CELL "
                "and then volume-weighted onto the shells. The inverse always exists; "
                "cells where it comes out non-positive are not in a valid state of "
                "that EOS (p below the JWL cold curve, i.e. the explosive has not "
                "released its chemical energy yet) and get no energy label"
            ),
        },
        "material_partition_source": (
            "ALE material volume fraction read from the d3plot history variables "
            "(columns 1..7), summed over the explosive parts; pure product alpha >= 0.95, "
            "pure air alpha <= 0.05, the mixed band is excluded from every label"
        ),
        "material_partition_is_volume_fraction_based": True,
        "state_validity_source": (
            "per cell: (rho, p) must be a state of the declared EOS with e > 0. A shell "
            "is invalid if ANY cell contributing to it failed this, by volume, and the "
            "fraction is recorded in material.npz as contaminated_shell_fraction. "
            "Shell averaging skips nan silently, so contamination is measured "
            "separately instead of being averaged away"
        ),
        "lasso_version": lasso_version(),
        "validated_lasso_version": VALIDATED_LASSO_VERSION,
        # 补丁实际动过什么。空列表 = 一次都没触发（50 mm 就是这种）。一个静默生效的
        # 补丁和一个没打补丁，从外部看起来一模一样，所以必须写出来。
        "lasso_history_allocation_fix_log": [dict(entry) for entry in lasso_fix_log()],
        "deck": deck.source,
        "d3plot_directory": str(d3plot_dir),
        "n_states_available": int(n_available),
        "n_states_source": count_source,
        "n_d3plot_files": int(n_files),
        "state_invalid_cells_per_frame": [
            int(v) for v in np.asarray(unreacted_cells, dtype=np.int64)[order]
        ],
        "state_indices": [int(s) for s in np.asarray(state_indices, dtype=np.int64)[order]],
        "stride": int(stride),
        "batch_states": int(batch_states),
        "cell_geometry": mesh_note,
        "labelled_shells": {
            "product": int(np.count_nonzero(material == 0)),
            "air": int(np.count_nonzero(material == 1)),
            "ambient": int(np.count_nonzero(material == 2)),
        },
        "finite_shells": {
            "product": int(np.count_nonzero((material == 0) & finite)),
            "air": int(np.count_nonzero((material == 1) & finite)),
        },
        "valid_shell_fraction": float(np.mean(valid)),
        "labelled_without_state_fraction": float(labelled_without_state_fraction),
        "empty_shell_fraction": float(np.mean(empty_shells)),
        "contaminated_shell_fraction": float(np.mean(contaminated_shells)),
        # 用 ``nanmean``：没有单元中心的空壳层其污染比例是 ``nan``（不是 0），用
        # ``mean`` 会让**每一帧**都变成 ``nan``——一列全是 ``nan`` 的"逐帧诊断"读起来
        # 像"没有数据"，实际是"有一个空壳层"。
        "contaminated_shell_fraction_by_frame": [
            float(v) for v in np.nanmean(contaminated, axis=1)
        ],
        "resolution_note": (
            "shells finer than the ALE cell spacing near the charge are empty by "
            "construction: a shell with no cell centroid in it has no state to average. "
            "empty_shell_fraction measures that mismatch; those shells are marked invalid"
        ),
        "alpha_range": [float(np.nanmin(alpha_product)), float(np.nanmax(alpha_product))],
        "front_detection": {
            "contact": (
                "per-frame geometric reconstruction (section 9): a spherically symmetric "
                "PLIC cut of the all-cell volume-weighted product fraction. The interface "
                "radius in a shell [a, b] with fraction alpha solves "
                "(r^3 - a^3)/(b^3 - a^3) = alpha, so the cut reproduces the cell phase "
                "volume by construction and the round-off is reported per frame. The "
                "position is the outer bound of the mixing band when the band spans more "
                "than one shell. It is NOT the alpha = 0.5 shell crossing any more: that "
                "crossing is kept separately as contact_shell_crossing_radius_m, purely "
                "as an auxiliary cross-check, because the shell average smears the "
                "interface over about one shell. When the reconstruction cannot locate "
                "the interface it returns no radius at all and the reason is recorded in "
                "contact_reconstruction_reason; a bracket is always reported."
            ),
            "contact_status_counts": {
                name: int(np.count_nonzero(contact_status_array == name))
                for name in ("reconstructed", "on_shared_face", "unresolved")
            },
            "contact_has_supervision_frames": int(np.count_nonzero(contact_sup_arrays)),
            "contact_sub_cell_frames": int(np.count_nonzero(contact_sub_cell_arrays)),
            "shock": (
                "OUTER EDGE OF THE CONTIGUOUS OVER-PRESSURE RUN attached to the contact, "
                "air side only (alpha < 0.5), threshold = ambient x (1 + 1e-2), edge "
                "linearly interpolated. NOT 'the outermost over-threshold shell': the "
                "far-field pressure is numerical noise around p/p_amb = 1.000 with "
                "excursions up to ~1.5%, so the outermost rule locks onto a noise spike "
                "tens of millimetres beyond the shock and moves by 5-8 mm between 2500 "
                "and 5000 shells. See shock_status_counts and shock_rh_* for why a front "
                "when it is"
            ),
            "shock_at_grid_edge": bool(
                np.any(np.isfinite(shock_array) & (shock_array >= centres[-1] - band_width))
            ),
            **shock_provenance,
            # 检测状态与监督资格分开报：前者说"看见了什么"，后者说"够不够格当标签"。
            "shock_status_counts": {
                name: int(np.count_nonzero(shock_status_str == name))
                for name in ("confirmed", "doubtful", "not_detected")
            },
            "shock_frames_with_supervision": int(np.count_nonzero(shock_supervision_array)),
            "shock_rh_momentum_range": (
                [float(np.nanmin(shock_rh_m_array)), float(np.nanmax(shock_rh_m_array))]
                if np.any(np.isfinite(shock_rh_m_array)) else None
            ),
            "shock_rh_energy_range": (
                [float(np.nanmin(shock_rh_e_array)), float(np.nanmax(shock_rh_e_array))]
                if np.any(np.isfinite(shock_rh_e_array)) else None
            ),
            "shock_compression_range": (
                [float(np.nanmin(shock_compression_array)),
                 float(np.nanmax(shock_compression_array))]
                if np.any(np.isfinite(shock_compression_array)) else None
            ),
            "shock_h_over_cell_range": (
                [
                    float(np.nanmin(h_over_cell_array)),
                    float(np.nanmax(h_over_cell_array)),
                ]
                if np.any(np.isfinite(h_over_cell_array))
                else None
            ),
            "shock_cell_thickness_range_m": (
                [float(np.nanmin(thickness_array)), float(np.nanmax(thickness_array))]
                if np.any(np.isfinite(thickness_array))
                else None
            ),
        },
        # §10 / §15：**检测状态与监督资格分开记**，两条前沿各一份。检测说"看见了
        # 什么"，资格说"够不够格当标签"。合成一个字段的话，一个 doubtful 的前沿
        # 会以"已检测"的身份进训练集，而报表上只看得到一个半径。
        "front_supervision_eligibility": {
            "contact": {
                "status_array_key": "contact_reconstruction_status",
                "status_counts": {
                    name: int(np.count_nonzero(contact_status_array == name))
                    for name in ("reconstructed", "on_shared_face", "unresolved")
                },
                "eligibility_array_key": "contact_has_supervision",
                "n_frames_eligible": int(np.count_nonzero(contact_sup_arrays)),
                "rule": (
                    "eligible only when the geometric reconstruction located the "
                    "interface inside a mixing band bracketed by the expected phases, "
                    "or exactly on a shared face, and the band spans no empty shell"
                ),
            },
            "shock": {
                "status_array_key": "shock_status",
                "status_counts": {
                    name: int(np.count_nonzero(shock_status_str == name))
                    for name in ("confirmed", "doubtful", "not_detected")
                },
                "eligibility_array_key": "shock_has_supervision",
                "n_frames_eligible": int(np.count_nonzero(shock_supervision_array)),
                "rule": (
                    "eligible only for confirmed candidates, which is where a reliable "
                    "radius range exists"
                ),
            },
            "note": (
                "the two fronts are independent: an ineligible shock never removes a "
                "usable contact label, and vice versa"
            ),
        },
        "contact_initial_offset_m": contact_offset,
        "contact_initial_offset_shell_crossing_m": contact_crossing_offset,
        "contact_initial_offset_note": (
            "at t = 0 the contact surface is at R_0 by definition, so both numbers are "
            "direct, model-free error readings of the two locators on the same frame. "
            "The geometric reconstruction is the one carried in contact_radius; the "
            "shell crossing is kept only as the auxiliary cross-check (section 9)"
        ),
        "contact_resolution": {
            "0": "NOT DETECTED: the geometric reconstruction could not locate the "
                 "interface, so contact_radius is nan. The bracket is still valid and "
                 "the reason is in contact_reconstruction_reason",
            "1": "SUB-CELL: a reconstruction exists but is not eligible as a label — the "
                 "mixing band is not bracketed by the expected phases, or it spans an "
                 "empty shell. contact_empty_shells_in_gap counts the latter",
            "2": "resolved: the interface was located with both neighbours confirmed",
            "resolved_frames": int(np.count_nonzero(contact_state_array == _SHOCK_RESOLVED)),
            "subcell_frames": int(np.count_nonzero(contact_state_array == _SHOCK_SUBCELL)),
            "not_found_frames": int(np.count_nonzero(contact_state_array == _SHOCK_NOT_DETECTED)),
            "interpolation_gap_range_m": (
                [float(np.nanmin(contact_gap_array)), float(np.nanmax(contact_gap_array))]
                if np.any(np.isfinite(contact_gap_array))
                else None
            ),
        },
        "contact_cell_thickness_m_range": (
            [float(np.nanmin(contact_thickness_array)), float(np.nanmax(contact_thickness_array))]
            if np.any(np.isfinite(contact_thickness_array))
            else None
        ),
        "mesh_resolution_note": (
            "The ALE element radial thickness is the resolution floor for every front "
            "position and for h = R_s - R_c. Measured on the 50 mm case it is ~0.85 mm "
            "inside the charge but 9.7-12.7 mm in the 50-60 mm air, 31 mm at 90-120 mm "
            "and 105 mm beyond 3 m. Refining the SHELL grid does not change it: the shell "
            "grid only resamples a coarse cell. Two quantities are therefore reported "
            "separately: the shell-grid convergence (2500 vs 5000 shells, differences "
            "relative to R_0 / R_s / h) and the mesh floor from shock_cell_thickness_m."
        ),
        "initial_detonation_check": (
            "The contact radius at t = 0 is R_0 by construction, so its measured value is "
            "a free calibration of the whole front-location chain. See "
            "contact_initial_offset_m; on the 50 mm case the measured value is ~2 mm "
            "high, which is about a third of the first air cell's 10 mm radial thickness."
        ),
        "units": {"source": deck.unit_system, "density_to_si": deck.density_to_si},
        "detonation_velocity_m_s": deck.detonation_velocity_m_s,
        "initial_detonation": (
            list(deck.initial_detonation) if deck.initial_detonation else None
        ),
        "implied_charge_radius_m": float(implied_radius),
        "symmetry_multiplier": float(symmetry),
        "charge_radius_relative_deviation": (
            abs(float(implied_radius) - float(charge_radius_m)) / float(charge_radius_m)
            if np.isfinite(implied_radius)
            else None
        ),
        "grid": grid.as_dict() if hasattr(grid, "as_dict") else {"edges": grid.edges.tolist()},
        "time_origin": "t is the raw LS-DYNA physical time; t_0 is the first extracted state",
    }
    write_json(output_dir / "metadata.json", metadata)
    return ExtractionResult(
        case_id=case_id,
        output_dir=output_dir,
        n_states=int(time_array.size),
        n_shells=int(n_shells),
        times=time_array,
        charge_radius_m=float(charge_radius_m),
        implied_charge_radius_m=implied,
        metadata=metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--case", required=True, help="算例 id")
    parser.add_argument("--deck", required=True, help="该算例的 input.k")
    parser.add_argument("--d3plot-dir", required=True, help="含 d3plot 家族的目录")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--charge-radius", type=float, required=True, help="R_0，m")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--states", type=int, nargs="*", default=None)
    parser.add_argument(
        "--batch-states", type=int, default=16,
        help=(
            "每次交给 d3plot 读取器的状态数。每建一个 handle 都要重新解析一遍几何，"
            "实测每次调用约 6.5 s 的固定开销加约 1.1 s/状态：batch=1 是 7.6 s/状态，"
            "batch=4 是 2.4，batch=16 是 1.3。峰值内存与它成正比（约 0.28 GB/状态）"
        ),
    )
    parser.add_argument(
        "--n-states", type=int, default=None,
        help=(
            "直接用给定值，跳过状态数探针与它的边界复核。**只在你能独立给出这个数**"
            "时使用：文件数不是状态数（本机四个算例每个都差一），而回退到文件数会"
            "静默错位。默认走探针，探针解析失败或复核不过一律停下"
        ),
    )
    parser.add_argument(
        "--no-fixed-geometry", action="store_true",
        help="每帧重算单元几何（用于对照；ALE 网格固定时两者必须逐位相同）",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--shells", type=int, default=2500)
    # 外半径必须覆盖整段求解时间内的膨胀：50 mm 算例在 0.4 ms 时产物外缘已到
    # 0.64 m，用 12 R_0 = 0.6 m 会让接触面在后期跑到网格之外。默认与前置求解器
    # 的域因子一致（40 R_0），两张场的覆盖范围因此可比。
    parser.add_argument("--outer-radius-factor", type=float, default=40.0)
    args = parser.parse_args(argv)
    configure_stdout()
    load_config_from_args(args)

    deck = read_deck_context(args.deck)
    grid = ShellGrid.graded(
        inner=args.charge_radius / 500.0,
        outer=args.charge_radius * float(args.outer_radius_factor),
        n_shells=int(args.shells),
        anchor=args.charge_radius,
    )
    result = extract_case(
        case_id=args.case,
        d3plot_dir=args.d3plot_dir,
        deck=deck,
        output_dir=args.out,
        charge_radius_m=args.charge_radius,
        grid=grid,
        states=args.states,
        stride=args.stride,
        max_states=args.max_states,
        batch_states=int(args.batch_states),
        n_states=args.n_states,
        assume_fixed_geometry=not args.no_fixed_geometry,
        progress_every=int(args.progress_every),
    )
    print(f"[extract] {result.case_id}: {result.n_states} states x {result.n_shells} shells")
    print(f"[extract] t = {result.times[0]:.6g} .. {result.times[-1]:.6g} s")
    print(f"[extract] unit system = {deck.unit_system}; product parts = {deck.product_part_ids}")
    print(f"[written] {result.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
