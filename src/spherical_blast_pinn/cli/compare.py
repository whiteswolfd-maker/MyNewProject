"""预测与验证仿真的对照（提示词 §14）。

.. code-block:: bash

    python -m spherical_blast_pinn.cli.compare --config configs/base.yaml \
        --prediction outputs/predict/predict_R0080mm.npz \
        --case tnt_80mm --output outputs/compare/80mm

先核对单位、球心、半径和时间零点，再比较。三件事**不做**：

* **不做时间对齐寻优。** 时间零点由物理定义给出（``t_PINN = t_phys - T_h``），
  不通过事后平移压力曲线把到达时间误差抹掉。两支曲线的到达时间差就是要报告的量
  本身，不是要消除的残差。
* **不把界面附近算成一个数。** 前沿位置误差与间断附近的场误差分开显示；保护带
  依据参考数据的分辨率**预先**设定，不因为结果不好而事后扩大。
* **不只在共同区域算分。** 两个模型前沿不一致时，材料不匹配的区域单独报告。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.extracted_case import load_extracted_case
from ..evaluation.waveform import compare_gauge
from ..physics import material as mat
from ._common import (
    SetupError,
    add_common_arguments,
    configure_stdout,
    load_config_from_args,
    write_json,
)


#: 预测网格上的时间插值容差。超出这个范围就**不插值**，返回 ``nan``——
#: 静默外推会把"这里没有预测"读成一个具体的数。
_TIME_TOLERANCE_FRACTION = 1.0e-9

#: 预测文件里**被实际消费**的键。缺任何一个都要报出文件与字段名——用 ``.get``
#: 或默认常数补上的"预测"和真实预测长得一模一样，只是里面没有模型。
REQUIRED_PREDICTION_KEYS = (
    "radius",
    "pinn_time",
    "physical_time",
    "start_time",
    "charge_radius",
    "density",
    "velocity",
    "specific_energy",
    "pressure",
    "region",
    "contact_radius",
    "shock_radius",
    # 环境压力是被**消费**的：测点的超压、峰值与冲量全都相对它定义。以前它走
    # ``prediction.get("ambient_pressure", mat.AIR_PRESSURE)``——一个默认常数。
    # 用一个仓库里的常数顶替预测自己的环境态，得到的是一份"看起来正常"的超压
    # 时程，而它与模型根本不一致。
    "ambient_pressure",
    "scale_product_density",
    "scale_air_density",
    "scale_velocity",
    "scale_specific_energy",
)


class PredictionError(SetupError):
    """预测文件的格式或语义不合法。"""


def load_prediction(path: Any) -> dict[str, Any]:
    """读预测 npz —— **命令行这一层的唯一入口**。

    以前这里写成两行：先 ``with np.load(...)`` 把内容取成一份 dict，紧接着对
    那份 dict 取 ``.files``。``data`` 已经是字典，``dict`` 没有 ``files`` 属性，
    于是 ``compare`` 在任何输入下都抛 ``AttributeError: 'dict' object has no
    attribute 'files'``；而已有的单元测试只覆盖 ``compare_fields()``，正好绕开了
    命令行这一层，所以它一直没被发现。

    句柄与字典在这里一次处理清楚，下游只认字典。
    """
    path = Path(path)
    if not path.exists():
        raise PredictionError(f"prediction file {path} does not exist")
    with np.load(path, allow_pickle=True) as handle:
        data = {key: handle[key] for key in handle.files}
    missing = [key for key in REQUIRED_PREDICTION_KEYS if key not in data]
    if missing:
        raise PredictionError(
            f"{path} is missing {missing}; the comparison consumes these fields, so a "
            "missing one cannot be replaced by a default. Present keys: "
            f"{sorted(data)}"
        )
    return data


def validate_prediction(prediction: Mapping[str, Any], *, path: Any = "<prediction>") -> None:
    """轴序、形状与时间语义的核对。不合法就**不产出对照结论**。

    三条都在"看起来正常"的失败模式里：轴反向会让所有基于顺序的切片取反；
    场量帧数与时间轴对不上会让查询在很久之后抛 ``IndexError``；``physical_time``
    与 ``pinn_time + start_time`` 不一致说明这份文件的两条时间轴不是同一个约定，
    而两个数组各自都单调递增、各自都"看起来没问题"。
    """
    times = np.asarray(prediction["pinn_time"], dtype=np.float64)
    physical = np.asarray(prediction["physical_time"], dtype=np.float64)
    grid = np.asarray(prediction["radius"], dtype=np.float64)

    def axis(name, values, *, minimum):
        if values.ndim != 1:
            raise PredictionError(f"{path}: {name} must be 1-D, got shape {values.shape}")
        if values.size < minimum:
            raise PredictionError(
                f"{path}: {name} has {values.size} entries but at least {minimum} are "
                "needed; this command interpolates, it does not extrapolate a constant"
            )
        if not np.all(np.isfinite(values)):
            raise PredictionError(f"{path}: {name} contains non-finite values")
        if not np.all(np.diff(values) > 0.0):
            raise PredictionError(f"{path}: {name} must be strictly increasing")

    axis("radius", grid, minimum=2)
    axis("pinn_time", times, minimum=2)
    axis("physical_time", physical, minimum=2)
    if physical.size != times.size:
        raise PredictionError(
            f"{path}: 'physical_time' has {physical.size} entries but 'pinn_time' has "
            f"{times.size}; they describe the same frames"
        )

    start_time = float(prediction["start_time"])
    if not np.isfinite(start_time):
        raise PredictionError(f"{path}: 'start_time' is {start_time!r}")
    # 一次换算，只在这里核对：predict 写出的是 t_PINN 与物理时间两条轴，
    # 它们之间必须只差一个 T_h。对不上说明这份文件混用了两套时间约定。
    residual = np.max(np.abs(physical - (times + start_time)))
    tolerance = _TIME_TOLERANCE_FRACTION * max(
        abs(float(times[-1] - times[0])), abs(start_time), 1.0e-300
    )
    if residual > tolerance:
        raise PredictionError(
            f"{path}: 'physical_time' is not 'pinn_time + start_time' (max deviation "
            f"{residual:.3e} s). The two time axes are not the same convention; the "
            "comparison would silently shift every arrival time"
        )

    expected = (times.size, grid.size)
    for name in ("density", "velocity", "specific_energy", "pressure", "region"):
        array = np.asarray(prediction[name])
        if array.shape != expected:
            raise PredictionError(
                f"{path}: {name} has shape {array.shape}, expected {expected} "
                "(frames, radial points)"
            )
    region = np.asarray(prediction["region"])
    if not np.issubdtype(region.dtype, np.integer):
        raise PredictionError(
            f"{path}: 'region' has dtype {region.dtype}, not an integer type"
        )
    unknown = sorted(set(np.unique(region).tolist()) - {-1, 0, 1, 2})
    if unknown:
        raise PredictionError(
            f"{path}: 'region' contains {unknown}, which this comparison does not "
            "define (known: -1 unavailable, 0 product, 1 air, 2 ambient)"
        )
    for name in ("contact_radius", "shock_radius"):
        array = np.asarray(prediction[name], dtype=np.float64)
        if array.shape != (times.size,):
            raise PredictionError(
                f"{path}: {name} has shape {array.shape}, expected {(times.size,)}"
            )

    for name in ("charge_radius", "scale_product_density", "scale_air_density",
                 "scale_velocity", "scale_specific_energy"):
        value = float(prediction[name])
        if not np.isfinite(value) or value <= 0.0:
            raise PredictionError(
                f"{path}: {name} is {value!r}; it is used as a fixed physical scale and "
                "must be finite and positive"
            )

    # **环境压力是一条真的单位/约定核对。** 本项目的环境压来自卡片
    # ``*EOS_LINEAR_POLYNOMIAL`` 反解出来的 101320 Pa，不是标准大气压 101325 Pa；
    # 两者差 5 Pa，而超压、峰值、冲量全都相对环境压定义。混用不会报错，只会让
    # 每条曲线的基线整体挪 5 Pa。
    ambient = float(prediction["ambient_pressure"])
    if not np.isfinite(ambient) or ambient <= 0.0:
        raise PredictionError(
            f"{path}: 'ambient_pressure' is {ambient!r}; overpressure is measured "
            "relative to it, so it must be finite and positive"
        )
    tolerance = 1.0e-6 * float(mat.AIR_PRESSURE)
    if abs(ambient - float(mat.AIR_PRESSURE)) > tolerance:
        raise PredictionError(
            f"{path}: 'ambient_pressure' is {ambient} Pa but this project's ambient is "
            f"{float(mat.AIR_PRESSURE)} Pa. The deck's *EOS_LINEAR_POLYNOMIAL gives "
            "101320, not the standard atmosphere 101325; mixing the two shifts every "
            "overpressure by ~5 Pa"
        )
    coverage = prediction["scaled_time_coverage"]
    # ``np.savez`` 把标量存成 **0 维数组**，读回来是 ``array(0.0003)`` 而不是
    # ``float``。用 ``isinstance(..., float)`` 判会把它当成"不是数"——而它明明是。
    if np.ndim(coverage) != 0:
        raise PredictionError(
            f"{path}: 'scaled_time_coverage' has shape {np.shape(coverage)}, not a scalar"
        )


# --------------------------------------------------------------------------
# 逐点采样：**一个核心规则**，场量与测点压力共用
# --------------------------------------------------------------------------

def _time_bracket(times: np.ndarray, pinn_time: float) -> tuple[int, float] | None:
    """``(下限下标, 权重)``；超出网格覆盖时返回 ``None``。

    ``np.searchsorted`` 之后把下标钳到 ``[0, n-2]`` 会让权重跑出 ``[0, 1]``：
    查询时间在网格末端之外时，权重是大于 1 的数，插值结果是一个**外推值**，
    却和一个真正的插值长得一模一样。所以这里先判定覆盖，再决定权重。
    """
    if times.size < 2:
        return None
    span = float(times[-1] - times[0])
    tolerance = _TIME_TOLERANCE_FRACTION * max(abs(span), 1.0e-300)
    if pinn_time < times[0] - tolerance or pinn_time > times[-1] + tolerance:
        return None
    index = int(np.clip(np.searchsorted(times, pinn_time) - 1, 0, times.size - 2))
    t0, t1 = float(times[index]), float(times[index + 1])
    weight = (pinn_time - t0) / (t1 - t0) if t1 > t0 else 0.0
    return index, float(min(max(weight, 0.0), 1.0))


#: ``sample_field`` 的逐点结果码。**互斥**：每个查询点恰好落入一个。
SAMPLE_OK = "ok"
SAMPLE_TIME_OUTSIDE = "time_outside_prediction_coverage"
SAMPLE_RADIUS_OUTSIDE = "radius_outside_prediction_grid"
SAMPLE_FRONT_UNAVAILABLE = "front_record_unavailable_at_query_time"
SAMPLE_CROSS_DISCONTINUITY = "cross_discontinuity_interpolation"
SAMPLE_NON_FINITE = "prediction_non_finite_at_support"

#: 哪些结果码属于"这个点根本不在可比较范围内"，哪些属于"模型在支撑点上无值"。
SAMPLE_OUT_OF_RANGE = (SAMPLE_TIME_OUTSIDE, SAMPLE_RADIUS_OUTSIDE, SAMPLE_FRONT_UNAVAILABLE)
SAMPLE_INTERPOLATION_REFUSED = (SAMPLE_CROSS_DISCONTINUITY,)


def _region_from_fronts(radii, contact: float, shock: float) -> np.ndarray:
    """由**某一时刻**的 ``(R_c, R_s)`` 给出半径上的区域标签，不可用时全 ``-1``。"""
    radii = np.asarray(radii, dtype=np.float64)
    if not (np.isfinite(contact) and np.isfinite(shock)) or shock < contact:
        # 前沿缺测或排序不合法：返回"不可用"，不拿上一帧顶替，也不外推。
        return np.full(radii.shape, -1, dtype=np.int64)
    return np.where(radii < contact, 0, np.where(radii < shock, 1, 2)).astype(np.int64)


def _fronts_at(prediction: Mapping[str, Any], pinn_time: float) -> tuple[float, float]:
    """``(R_c, R_s)`` 在 ``pinn_time`` 处的值，**时间线性插值**；不可用则为 ``nan``。

    帧间插值方法是明确写出来的：两条前沿各自按时间线性插值。以前 ``_region_at``
    直接取时间区间的**左端**那一帧的前沿，于是"查询最后一帧"时用的是倒数第二帧
    的前沿——查询时刻越靠近区间右端，错得越多，而结果看不出任何异常。

    **只有权重非零的那一帧才需要有效。** 查询精确落在第 ``i`` 帧上时权重是 0，
    另一帧整个不参与计算；无条件检查两个端点会让"第 i+1 帧前沿缺测"把一次完全
    可用的查询判成不可用——场量插值早就按这个规则改了（零权重支撑点不参与），
    前沿这一侧当时漏掉了。
    """
    times = np.asarray(prediction["pinn_time"], dtype=np.float64)
    bracket = _time_bracket(times, float(pinn_time))
    if bracket is None:
        return float("nan"), float("nan")
    i, weight = bracket
    j = min(i + 1, times.size - 1)
    contact = np.asarray(prediction["contact_radius"], dtype=np.float64)
    shock = np.asarray(prediction["shock_radius"], dtype=np.float64)

    contact_total = 0.0
    shock_total = 0.0
    weight_total = 0.0
    for index, share in ((i, 1.0 - weight), (j, weight)):
        if share <= 0.0:
            continue          # 零权重的帧不参与，它的缺测无从污染结果
        c = float(contact[index])
        s = float(shock[index])
        if not (np.isfinite(c) and np.isfinite(s)):
            return float("nan"), float("nan")
        contact_total += share * c
        shock_total += share * s
        weight_total += share
    if weight_total <= 0.0:  # pragma: no cover - 两个权重不可能同时为 0
        return float("nan"), float("nan")
    return contact_total / weight_total, shock_total / weight_total


def sample_field(
    prediction: Mapping[str, Any], name: str, radii, physical_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """在预测网格上取 ``(r, t_phys)`` 处的场量，并给出**逐点**结果码。

    这是场量与测点压力**共用**的核心规则（§7.3）。以前这里有三分实现：
    ``_field_at``（数组、按行的区域检查）、``_field_corner_average``（标量、
    没有跨时间层的区域检查）、以及 ``_region_at`` 自己的前沿取值——三条规则各不
    相同，于是同一个查询在不同的消费路径上给出不同的答案。

    规则：

    * 时间与半径都只在网格覆盖内取值，域外**不外推**；
    * 精确落在节点上时直接取该节点的值：零权重的支撑点**不参与**，所以邻点是
      ``nan`` 也不会污染它（``nan * 0`` 是 ``nan``，不是 0）；
    * 参与插值的每一个支撑点（两个时间层 × 两个径向节点，权重非零的那些）必须
      落在**同一个区域**里，且与查询时刻的区域一致。接触面或激波在两帧之间扫过
      查询点时，两侧支撑点分属不同区域，此时**不插值**——线性插值会造出一个
      两侧都不属于的中间态，再拿它去和真实状态比；
    * 支撑点上出现非有限值 → 该点不可用，原因单独记，不当成"模型发散"以外的
      任何东西，也不静默删掉。
    """
    times = np.asarray(prediction["pinn_time"], dtype=np.float64)
    grid = np.asarray(prediction["radius"], dtype=np.float64)
    field = np.asarray(prediction[name])
    radii = np.asarray(radii, dtype=np.float64)
    values = np.full(radii.shape, np.nan, dtype=np.float64)
    reasons = np.full(radii.shape, SAMPLE_TIME_OUTSIDE, dtype=object)

    bracket = _time_bracket(
        times, float(physical_time) - float(prediction["start_time"])
    )
    if bracket is None:
        return values, reasons
    i, weight_t = bracket
    j = min(i + 1, times.size - 1)
    time_weights = (1.0 - weight_t, weight_t)

    outside = (radii < grid[0]) | (radii > grid[-1])
    reasons[outside] = SAMPLE_RADIUS_OUTSIDE
    positions = np.flatnonzero(~outside)
    if positions.size == 0:
        return values, reasons

    query = radii[positions]
    index = np.clip(
        np.searchsorted(grid, query, side="right") - 1, 0, grid.size - 2
    )
    left, right = grid[index], grid[index + 1]
    span = right - left
    weight_r = np.where(
        span > 0.0, (query - left) / np.where(span > 0.0, span, 1.0), 0.0
    )
    radial_weights = (1.0 - weight_r, weight_r)
    nodes = (index, index + 1)

    # 查询时刻自身的区域：用**插值后的前沿**。这一刻的区域才是"这个点属于谁"。
    contact, shock = _fronts_at(prediction, float(physical_time) - float(prediction["start_time"]))
    query_region = _region_from_fronts(query, contact, shock)

    # 每一帧的整网格区域标签，用来核对参与插值的支撑点。
    rows = (i, j)
    row_regions = [
        _region_from_fronts(
            grid,
            float(prediction["contact_radius"][row]),
            float(prediction["shock_radius"][row]),
        )
        for row in rows
    ]

    size = positions.size
    seen = np.zeros(size, dtype=bool)
    missing = np.zeros(size, dtype=bool)
    conflict = np.zeros(size, dtype=bool)
    common = np.full(size, -1, dtype=np.int64)
    total = np.zeros(size, dtype=np.float64)
    finite_ok = np.ones(size, dtype=bool)

    for row_index in (0, 1):
        if time_weights[row_index] <= 0.0:
            continue          # 零权重的时间层整个不参与
        for side_index in (0, 1):
            weight = time_weights[row_index] * radial_weights[side_index]
            active = weight > 0.0
            if not np.any(active):
                continue      # 零权重的邻点不参与，它的 nan 也就无从污染
            node = nodes[side_index]
            region_here = row_regions[row_index][node]
            sample = np.asarray(field[rows[row_index]], dtype=np.float64)[node]
            missing |= active & (region_here < 0)
            conflict |= active & seen & (region_here != common)
            common = np.where(active & ~seen, region_here, common)
            seen |= active
            finite_ok &= ~(active & ~np.isfinite(sample))
            total += np.where(
                active, np.where(np.isfinite(sample), sample * weight, 0.0), 0.0
            )

    no_front = query_region < 0
    mismatch = seen & (~missing) & (~conflict) & (query_region != common) & (~no_front)
    usable = seen & ~missing & ~conflict & ~no_front & ~mismatch

    local = np.full(size, SAMPLE_OK, dtype=object)
    local[no_front] = SAMPLE_FRONT_UNAVAILABLE
    local[~no_front & (missing | conflict | mismatch)] = SAMPLE_CROSS_DISCONTINUITY
    local[usable & ~finite_ok] = SAMPLE_NON_FINITE
    local[usable & finite_ok] = SAMPLE_OK

    # 只有**结果码为 OK** 的点拿到值；其余保持 ``nan``，原因写在 ``reasons`` 里。
    produced = local == SAMPLE_OK
    values[positions[produced]] = total[produced]
    reasons[positions] = local
    return values, reasons


def sample_field_values(prediction, name, radii, physical_time) -> np.ndarray:
    """:func:`sample_field` 的取值一侧。需要原因时用前者。"""
    return sample_field(prediction, name, radii, physical_time)[0]


def _pressure_at(prediction: dict[str, Any], radius: float, physical_time: float) -> float:
    """测点压力。**与场量走同一个核心**（§7.3），不再单独实现一套插值。"""
    values = sample_field_values(
        prediction, "pressure", np.asarray([radius], dtype=np.float64), physical_time
    )
    return float(values[0])


def _reference_at(case, radius: float, physical_time: float) -> float:
    """参考压力；**只读物理时间在数据覆盖内的点**，域外返回 ``nan``。

    用最近帧是静默外推：请求到数据末端之后，它会给出最后一帧的值，读起来和
    一个真实采样一样。
    """
    times = case.times
    if physical_time < times[0] or physical_time > times[-1]:
        return float("nan")
    frame = case.frame_index(physical_time)
    value = float(case.sample_at(frame, np.array([radius]))["pressure"][0])
    return value if np.isfinite(value) else float("nan")


def compare_trajectories(prediction: dict[str, Any], case, *, ambient: float) -> dict[str, Any]:
    """``R_c``/``R_s`` 轨迹相对 RMS 误差，**按统一物理时间加权**。

    逐点等权会让输出帧密的那一段主导结论；这里把两条轨迹都插值到同一个均匀
    物理时间网格上再比较，帧间隔的不均匀就不再改变结论。
    """
    # ``physical_time`` 已经从 predict 里出来就是物理秒，``epsilon(R_0)`` 在那边
    # 加过一次。这里再加一次会让两条轨迹整体错开一个 ``epsilon``——R_0 = 50 mm
    # 时那是 1 µs，在 3000/4000 m/s 的前沿速度下就是 3/4 mm 的假误差。
    pred_phys = np.asarray(prediction["physical_time"], dtype=np.float64)
    grid = np.linspace(float(pred_phys[0]), float(pred_phys[-1]), 400)

    pred_contact = np.interp(grid, pred_phys, prediction["contact_radius"])
    pred_shock = np.interp(grid, pred_phys, prediction["shock_radius"])

    usable = case.supervisable_frames()
    frames = np.flatnonzero(usable)
    if frames.size < 4:
        return {"available": False, "reason": "参考数据没有足够的有效帧"}
    ref_times = case.times[frames]
    keep = (grid >= ref_times[0]) & (grid <= ref_times[-1])
    if np.count_nonzero(keep) < 4:
        return {"available": False, "reason": "预测与参考在物理时间上没有重叠区间"}

    ref_contact = np.interp(grid[keep], ref_times, case.contact_radius[frames])
    ref_shock = np.interp(grid[keep], ref_times, case.shock_radius[frames])
    scale = float(case.charge_radius)

    def rms(model, reference):
        return float(np.sqrt(np.mean(((model - reference) / scale) ** 2)))

    # 到达时间：两条轨迹第一次越过某个固定 ``r/R_0`` 的时刻之差，由轨迹本身
    # 给出，不做任何时间对齐寻优。
    arrival = {}
    for factor in (2.0, 5.0, 10.0):
        target = factor * scale
        arrival[f"r_over_R0_{factor:g}"] = {
            "model_s": _first_crossing(pred_phys, prediction["shock_radius"], target),
            "reference_s": _first_crossing(ref_times, case.shock_radius[frames], target),
        }
    for entry in arrival.values():
        model_time, reference_time = entry["model_s"], entry["reference_s"]
        if model_time is None or reference_time is None:
            entry["absolute_error_s"] = None
            entry["relative_error"] = None
        else:
            entry["absolute_error_s"] = model_time - reference_time
            entry["relative_error"] = (
                (model_time - reference_time) / reference_time if reference_time > 0.0 else None
            )

    return {
        "available": True,
        "normalisation": "charge radius R_0 (fixed, from the prediction's own scales)",
        "n_overlap_points": int(np.count_nonzero(keep)),
        "overlap_physical_time_s": [float(grid[keep][0]), float(grid[keep][-1])],
        "contact_relative_rms": rms(pred_contact[keep], ref_contact),
        "shock_relative_rms": rms(pred_shock[keep], ref_shock),
        "shock_arrival": arrival,
    }


def _first_crossing(times, radii, target: float) -> float | None:
    """``radii`` 第一次达到 ``target`` 的**线性插值**时刻；从未达到时 ``None``。"""
    times = np.asarray(times, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    finite = np.isfinite(radii) & np.isfinite(times)
    if not np.any(finite):
        return None
    times, radii = times[finite], radii[finite]
    above = np.flatnonzero(radii >= target)
    if above.size == 0 or above[0] == 0:
        return None
    index = int(above[0])
    r0, r1 = float(radii[index - 1]), float(radii[index])
    t0, t1 = float(times[index - 1]), float(times[index])
    if r1 == r0:
        return t0
    return t0 + (t1 - t0) * (target - r0) / (r1 - r0)


def _frozen_scales(prediction: dict[str, Any]) -> dict[str, float]:
    """每个场量的**固定物理尺度**，全部来自检查点，不来自被比较的数据。

    用参考数据的局部最大值当分母，分母会随"这次比了哪些点"变化，于是两次
    报告里的"相对误差"不是同一个量。这里用的是乘积里就有的尺度：
    ``rho_{*,k}``、``U_*``、``U_*^2``。
    """
    return {
        "density_product": float(prediction["scale_product_density"]),
        "density_air": float(prediction["scale_air_density"]),
        "velocity": float(prediction["scale_velocity"]),
        "specific_energy": float(prediction["scale_specific_energy"]),
    }


def compare_fields(prediction: dict[str, Any], case, *, buffer_cells: int = 3) -> dict[str, Any]:
    """两材料光滑区域内的场量相对 L2，**按球壳体积加权**，**分材料报告**。

    四条规则，每一条都对应一个会改变结论的失误：

    1. **保护带按参考数据的分辨率预先设置**（``buffer_cells`` 个单元），不因为
       结果不好而事后扩大。前沿附近的结果单独统计，不并入光滑区。
    2. **只比较材料身份一致的单元。** 参考说产物、预测说空气的点，既不属于
       产物误差也不属于空气误差，必须单独计数。把它们混进"共同区域"会让一个
       前沿错位的模型得到一个不错的光滑区误差。
    3. **未受扰动标签（2）不当作空气。** 数据里 ``material == 2`` 是主激波之外
       的静止空气；把它记为"空气"会让一个完全正确的环境态预测被判为材料错配。
    4. **分介质报告**，并保留错配统计与界面被排除的统计。
    """
    physical_time_grid = np.asarray(prediction["physical_time"], dtype=np.float64)
    usable = case.supervisable_frames()
    frames = np.flatnonzero(usable)
    if frames.size == 0:
        return {"available": False, "reason": "参考数据没有有效帧"}

    cell = float(np.median(np.diff(case.radius)))
    scales = _frozen_scales(prediction)
    field_names = ("density", "velocity", "specific_energy")
    medium_scale = {
        "density": {"product": scales["density_product"], "air": scales["density_air"]},
        "velocity": {"product": scales["velocity"], "air": scales["velocity"]},
        "specific_energy": {"product": scales["specific_energy"], "air": scales["specific_energy"]},
    }

    totals = {
        f"{medium}_{name}": {"num": 0.0, "den": 0.0, "n": 0, "max_abs": 0.0}
        for medium in ("product", "air")
        for name in field_names
    }
    #: 每个字段的**全程计数**。每一类都单独记，且这些类**互斥**：候选点 =
    #: 参考缺测 + 域外 + 保护区 + 区域错配 + 插值不可用 + 模型非有限 + 实际比较。
    #: 记这几个数不是装饰——"三个候选点里一个预测是 NaN"这件事只有在这一列里
    #: 才看得见，否则它就从分母里消失了。
    counts = {
        f"{medium}_{name}": dict.fromkeys(
            ("candidates", "reference_missing", "model_non_finite",
             "outside_domain", "buffer_zone", "region_mismatch",
             "interpolation_unavailable", "unclassified", "compared"),
            0,
        )
        for medium in ("product", "air")
        for name in field_names
    }
    #: 逐点不可用原因的**分项计数**。它们与上面那组互斥类别是同一批点的两种视图：
    #: 类别回答"这个点算不算误差"，原因回答"为什么不算"。§7 要求在报告里能把
    #: "跨间断不能插"、"支撑点非有限"、"域外"分开，而不是一律记成"模型非有限"。
    reason_totals: dict[str, int] = {
        code: 0 for code in (
            SAMPLE_OK, SAMPLE_TIME_OUTSIDE, SAMPLE_RADIUS_OUTSIDE,
            SAMPLE_FRONT_UNAVAILABLE, SAMPLE_CROSS_DISCONTINUITY, SAMPLE_NON_FINITE,
        )
    }
    mismatch = {medium: 0 for medium in ("product", "air", "ambient")}
    excluded_inside_prediction_domain = 0
    compared_cells = 0
    #: 参考帧落在预测时间覆盖之外的数目。它们**不在候选范围内**（不是"不可比"，
    #: 是"这段时间预测根本没给"），所以单独报告覆盖范围与排除数量。
    frames_outside_prediction_coverage = 0
    # 参考数据用的是逐字段有效性还是那张联合掩码——退化要写出来。
    validity_source = (
        "per-field validity (case.field_validity)"
        if getattr(case, "field_validity", None) is not None
        else "joint `case.valid` mask (per-field validity not recorded on this case)"
    )

    for frame in frames:
        physical_time = float(case.times[frame])
        if physical_time < physical_time_grid[0] or physical_time > physical_time_grid[-1]:
            frames_outside_prediction_coverage += 1
            continue
        # 预测自己的分区标签，用于判定"材料身份是否一致"。
        region = _region_at(prediction, case.radius, physical_time)
        if not np.any(region >= 0):
            continue
        for label, medium in ((0, "product"), (1, "air"), (2, "ambient")):
            reference_here = case.material[frame] == label
            if not np.any(reference_here):
                continue
            # 错配：参考说这是 `label`，而预测的前沿把它划进了别的区域。
            mismatched = reference_here & (region >= 0) & (region != label)
            mismatch[medium] += int(np.count_nonzero(mismatched))
            compared_cells += int(np.count_nonzero(reference_here & (region >= 0)))
            if label == 2:
                # 环境态不参与场误差：那里是几千个常数单元，拿它评分对所有模型
                # 一样地"好看"，衡量不出任何东西（见 README 第八节）。
                continue

            contact = case.contact_radius[frame]
            shock = case.shock_radius[frame]
            near = (np.abs(case.radius - contact) <= buffer_cells * cell) | (
                np.abs(case.radius - shock) <= buffer_cells * cell
            )
            indices = np.flatnonzero(reference_here)
            if indices.size == 0:
                continue
            radii_sample = case.radius[indices]
            values = case.sample_at(int(frame), radii_sample)
            weights = 4.0 * np.pi * radii_sample**2 * cell
            inside = (region[indices] >= 0)
            in_region = region[indices] == label
            in_buffer = near[indices]

            for name in field_names:
                key = f"{medium}_{name}"
                entry = counts[key]
                entry["candidates"] += int(indices.size)
                # **逐字段有效性**（§11.1）：一个字段缺失不该丢掉同一位置其他
                # 可靠的字段。以前这里用的是 ``case.valid``——那是"四项联合可用"
                # 的掩码，于是独立有效的字段在评估里被一起丢掉，而训练采样早就
                # 改成逐字段了。训练与评估用两套有效性，评估出来的误差不是同一个
                # 量。
                field_valid = _field_validity_at(case, frame, name, indices)
                outside = ~inside
                entry["outside_domain"] += int(np.count_nonzero(outside))
                entry["region_mismatch"] += int(
                    np.count_nonzero(~outside & ~in_region)
                )
                entry["buffer_zone"] += int(
                    np.count_nonzero(~outside & in_region & in_buffer)
                )
                position_ok = (~outside) & in_region & (~in_buffer)
                # ``reference_missing`` 只记**位置本来可用、但这个字段自己没有效**
                # 的点。写成 ``count_nonzero(~keep)`` 会把域外、错配、保护区一起算
                # 进来——那三类已经在上面各自记过，再加一遍就是重复计数。
                keep = position_ok & field_valid
                entry["reference_missing"] += int(
                    np.count_nonzero(position_ok & ~field_valid)
                )
                if not np.any(keep):
                    continue
                reference = values[name][keep]
                # **同一个核心**给出值与逐点原因（§7.3）。以前这里用的是另一条
                # 实现（``_field_at``），它只在**单个时间层内**比较左右节点的区域，
                # 不比较两个时间层之间——于是接触面在两帧之间扫过查询点时，产物与
                # 空气被线性插成一个两侧都不属于的中间值。
                model, reasons = sample_field(
                    prediction, name, radii_sample[keep], physical_time
                )
                codes = np.asarray(reasons, dtype=object)
                for code in reason_totals:
                    reason_totals[code] += int(np.count_nonzero(codes == code))
                # 逐点原因 → 互斥计数。**不可插值**与**支撑点非有限**分开记：
                # 前者是"这里不该插"（跨间断），后者才是"模型在这里没有值"。
                #
                # ``keep`` 已经要求位置落在预测范围内（区域标签 >= 0），所以这里
                # **不可能**再出现"时间/半径域外""前沿缺测"这几种码——它们与
                # ``outside_domain`` 是同一批点，重复计一次就是重复计数。残留的
                # 域外码如果真出现，说明两条规则不一致，折进
                # ``interpolation_unavailable`` 而不是丢掉：宁可少比一个点，
                # 不能让一个点从分类里消失。
                unavailable = np.isin(
                    codes, SAMPLE_INTERPOLATION_REFUSED
                ) | np.isin(codes, SAMPLE_OUT_OF_RANGE)
                model_bad = codes == SAMPLE_NON_FINITE
                entry["interpolation_unavailable"] += int(np.count_nonzero(unavailable))
                entry["model_non_finite"] += int(np.count_nonzero(model_bad))
                usable = ~unavailable & ~model_bad
                good = usable & np.isfinite(reference)
                entry["compared"] += int(np.count_nonzero(good))
                entry["unclassified"] += int(
                    np.count_nonzero(usable & ~np.isfinite(reference))
                )
                if not np.any(good):
                    continue
                w = weights[keep][good]
                difference = model[good] - reference[good]
                scale = medium_scale[name][medium]
                totals[key]["num"] += float(np.sum(w * (difference / scale) ** 2))
                totals[key]["den"] += float(np.sum(w))
                totals[key]["n"] += int(np.count_nonzero(good))
                totals[key]["max_abs"] = max(
                    totals[key]["max_abs"], float(np.max(np.abs(difference)))
                )
        # 预测域外的参考点单独统计，不当作误差。
        excluded_inside_prediction_domain += int(np.count_nonzero(region < 0))

    result: dict[str, Any] = {
        "available": True,
        "metric": (
            "sqrt( sum_shell w (model - reference)^2 / sum_shell w ) / scale, "
            "w = 4 pi r^2 dr (spherical shell volume), scale = fixed physical scale "
            "from the checkpoint"
        ),
        "buffer_cells": int(buffer_cells),
        "buffer_width_m": float(buffer_cells * cell),
        "buffer_basis": "reference data radial resolution, fixed before the comparison",
        "volume_weighted": True,
        "normalisation_scales": scales,
        "reference_label_semantics": "0=product, 1=air, 2=undisturbed ambient",
        "ambient_cells_excluded": True,
        "compared_cells": int(compared_cells),
        "region_mismatch_cells": mismatch,
        "region_mismatch_total": int(sum(mismatch.values())),
        "cells_outside_prediction_grid": int(excluded_inside_prediction_domain),
        "validity_source": validity_source,
        "counts": counts,
        "counts_meaning": (
            "per medium and field, mutually exclusive: candidates = "
            "reference_missing + outside_domain + region_mismatch + buffer_zone + "
            "interpolation_unavailable + compared, and within the kept subset every "
            "point is either compared or counted in model_non_finite / unclassified"
        ),
        # **逐点不可用原因**（§7.6）。它与上面那组互斥类别是同一批点的两种视图：
        # 类别回答"这个点算不算误差"，原因回答"为什么不算"。以前这里只有
        # "model_non_finite" 一项，于是"跨间断不该插"和"模型在这里是 nan"被记成
        # 同一件事——前者是评估规则，后者是模型行为，混在一起等于把评估的保守
        # 当成模型的失败。
        "sample_reason_counts": reason_totals,
        "sample_reason_meaning": {
            SAMPLE_OK: "a finite value was produced from that point's own support nodes",
            SAMPLE_TIME_OUTSIDE: "query time outside the prediction's time coverage",
            SAMPLE_RADIUS_OUTSIDE: "query radius outside the prediction's radial grid",
            SAMPLE_FRONT_UNAVAILABLE: "the prediction's front record is unusable at that time",
            SAMPLE_CROSS_DISCONTINUITY: (
                "the support points straddle a contact or shock, so interpolating would "
                "produce a state belonging to neither side; no value is fabricated"
            ),
            SAMPLE_NON_FINITE: "a support node carrying non-zero weight is not finite",
        },
        "prediction_time_coverage_s": [
            float(physical_time_grid[0]), float(physical_time_grid[-1])
        ],
        "reference_frames_outside_prediction_coverage": int(
            frames_outside_prediction_coverage
        ),
        "reference_frames_used": int(frames.size),
        "coverage_note": (
            "frames whose physical time falls outside the prediction's coverage are not "
            "candidates at all: there is no prediction there to compare with. They are "
            "counted separately above rather than being folded into 'not comparable'"
        ),
    }
    incomplete: list[str] = []
    for key, entry in totals.items():
        if entry["den"] > 0.0:
            result[f"{key}_relative_l2"] = float(np.sqrt(entry["num"] / entry["den"]))
            result[f"{key}_n_points"] = entry["n"]
            result[f"{key}_max_abs_error"] = entry["max_abs"]
        else:
            # **零样本不是零误差。** 以前这一项直接不出现在结果里，读的人看到的
            # 是一张少了行的表，而不是"这一项没法评价"。
            result[f"{key}_available"] = False
            result[f"{key}_relative_l2"] = float("nan")
            # 配套的点数与最大绝对误差也一起给出：只写 `_relative_l2` 而漏掉它们，
            # 会让任何按"有 relative_l2 就一定有 n_points"读取的消费方拿到
            # ``KeyError``——渲染那一步就是这么挂掉的。
            result[f"{key}_n_points"] = 0
            result[f"{key}_max_abs_error"] = float("nan")
            incomplete.append(key)
        if counts[key]["model_non_finite"] > 0:
            incomplete.append(f"{key} (model non-finite: {counts[key]['model_non_finite']})")
    result["unavailable_fields"] = sorted(set(incomplete))
    result["complete"] = not incomplete
    # **顶层可用性与逐字段可用性不能互相矛盾。** 一个可比点都没有的时候，顶层
    # 还写着 ``available: True`` 会让读的人以为"对照做完了，只是某些字段空"——
    # 而实际情况是这次对照什么都没比出来。两者必须由同一个事实决定。
    comparable = int(sum(entry["n"] for entry in totals.values()))
    result["comparable_points"] = comparable
    if comparable == 0:
        result["available"] = False
        result["reason"] = (
            "no comparable point existed in any field: every candidate was outside the "
            "prediction's coverage, inside the buffer zone, a material mismatch, or "
            "could not be interpolated without crossing a discontinuity. The per-field "
            "counts and sample_reason_counts below say which"
        )
    result["note_on_incomplete"] = (
        "fields listed in unavailable_fields did not contribute a usable error: either no "
        "comparable point existed, or the model returned non-finite values there. A "
        "finite-subset error is still reported for the rest, and it does not certify the "
        "whole evaluation"
    )
    for medium in ("product", "air"):
        keys = [f"{medium}_{name}_relative_l2" for name in field_names]
        present = [result[k] for k in keys if k in result]
        result[f"{medium}_relative_l2_over_fields"] = (
            float(np.mean(present)) if present else float("nan")
        )
    return result


def _field_validity_at(case, frame: int, name: str, indices: np.ndarray) -> np.ndarray:
    """某个字段在第 ``frame`` 帧上、``indices`` 这些位置处的可用性。

    有逐字段记录就用它；没有就退回联合掩码，并把退化写进 ``validity_source``。
    训练采样早就按字段各自取点了，评估这一侧以前还在用联合掩码——两边不是同一个
    有效性，算出来的误差也就不是同一个量。
    """
    validity = getattr(case, "field_validity", None)
    if validity is not None and name in validity:
        return np.asarray(validity[name][frame], dtype=bool)[indices]
    return np.asarray(case.valid[frame], dtype=bool)[indices]


def _region_at(prediction, radii, physical_time: float) -> np.ndarray:
    """预测在 ``(r, t_phys)`` 处的区域标签：0 产物、1 空气、2 环境、``-1`` 不可用。

    **不可用**有三种，都返回 ``-1``：查询时间在预测时间网格之外、半径在预测径向
    网格之外、前沿记录在该时刻缺测或排序不合法。以前只有第一种返回 ``-1``：半径
    在网格外的点照样被贴上一个"产物/空气/环境"的标签，于是它们被记进"材料错配"
    或"模型非有限"，而不是"根本不在预测范围内"。

    区域按**查询时刻**的前沿判定。以前直接取时间区间左端那一帧的前沿，查询最后
    一帧时用的其实是倒数第二帧的前沿——越靠近区间右端错得越多，而报告上看不出
    任何异常。
    """
    radii = np.asarray(radii, dtype=np.float64)
    grid = np.asarray(prediction["radius"], dtype=np.float64)
    out = np.full(radii.shape, -1, dtype=np.int64)
    inside = (radii >= grid[0]) & (radii <= grid[-1])
    if not np.any(inside):
        return out
    contact, shock = _fronts_at(
        prediction, float(physical_time) - float(prediction["start_time"])
    )
    out[inside] = _region_from_fronts(radii[inside], contact, shock)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--prediction", required=True, help="predict 命令写出的 npz")
    parser.add_argument("--case", required=True, help="验证算例的 case_id")
    parser.add_argument("--gauge-factors", type=float, nargs="*", default=[2.0, 5.0, 10.0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    configure_stdout()
    config = load_config_from_args(args)

    section = config.section("data")
    specification = next(
        (s for s in (section.get("cases") or []) if str(s.get("case_id")) == args.case), None
    )
    if specification is None:
        raise SetupError(f"case {args.case!r} is not declared under data.cases")
    case = load_extracted_case(Path(str(specification["directory"])), case_id=args.case)

    # **一个入口读预测，紧接着核对形状与时间语义。** 不合法就不产出对照结论——
    # 一份轴反向或帧数对不上的预测照样能算出一张漂亮的表，而表里的数没有意义。
    prediction = load_prediction(args.prediction)
    validate_prediction(prediction, path=args.prediction)

    radius = float(prediction["charge_radius"])
    if abs(radius - case.charge_radius) > 1.0e-9:
        raise SetupError(
            f"预测的装药半径 {radius} m 与参考算例的 {case.charge_radius} m 不一致；"
            "半径、单位和时间零点必须先核对再比较"
        )

    ambient = float(prediction.get("ambient_pressure", mat.AIR_PRESSURE))
    report: dict[str, Any] = {
        "case_id": args.case,
        "charge_radius_m": radius,
        "checks": {
            "units": "两侧均为 SI",
            "charge_radius_match": True,
            "time_origin_s": float(prediction["start_time"]),
            "reference_time_origin_s": float(case.start_time),
            "time_origin_match": abs(
                float(prediction["start_time"]) - float(case.start_time)
            )
            < 1.0e-15,
            "time_origin_rule": (
                "t_PINN = t_phys - T_h; the prediction's physical_time already "
                "carries the shift, so nothing is shifted again here"
            ),
            "time_alignment_optimised": False,
            "scaled_time_coverage_s_max": float(
                prediction.get("scaled_time_coverage", float("nan"))
            ),
        },
        "trajectories": compare_trajectories(prediction, case, ambient=ambient),
        "fields": compare_fields(prediction, case),
    }

    # -- 测点压力时程 -----------------------------------------------------
    #
    # 每个测点**独立**判定可用性：某个测点的时程里出现 NaN（跨间断、域外或模型
    # 非有限）时，只把那一个测点记为不可用并写清原因与计数——不让整条 compare
    # 崩掉，也不把 NaN 当成 0 去积分出一个"看起来正常"的冲量。
    gauges = []
    for factor in args.gauge_factors:
        gauge_radius = float(factor) * radius
        entry: dict[str, Any] = {
            "radius_m": gauge_radius,
            "radius_over_charge_radius": float(factor),
        }
        if gauge_radius >= float(prediction["radius"][-1]):
            entry["available"] = False
            entry["reason"] = (
                "the gauge radius lies outside the prediction's radial grid; there is "
                "nothing to sample, and extrapolating would invent a waveform"
            )
            gauges.append(entry)
            continue
        physical = np.asarray(prediction["physical_time"], dtype=np.float64)
        model_pressure = np.array(
            [_pressure_at(prediction, gauge_radius, float(t)) for t in physical]
        )
        reference_pressure = np.array(
            [_reference_at(case, gauge_radius, float(t)) for t in physical]
        )
        model_finite = np.isfinite(model_pressure)
        if not np.all(model_finite):
            entry["available"] = False
            entry["model_non_finite_samples"] = int(np.count_nonzero(~model_finite))
            entry["samples"] = int(model_pressure.size)
            entry["reason"] = (
                "the model's pressure history at this gauge has non-finite samples "
                f"({entry['model_non_finite_samples']} of {entry['samples']}); a "
                "waveform with gaps is not integrated as one record"
            )
            gauges.append(entry)
            continue

        peak = float(np.max(model_pressure - ambient))
        pressure_scale = max(abs(peak), ambient)
        time_scale = max(float(physical[-1] - physical[0]), 1.0e-12)
        entry["frozen_scales"] = {
            "ambient_pa": float(ambient),
            "peak_overpressure_pa": peak,
            "pressure_scale_pa": pressure_scale,
            "time_scale_s": time_scale,
            "impulse_scale_pa_s": pressure_scale * time_scale,
        }
        reference_usable = bool(np.all(np.isfinite(reference_pressure)))
        comparison = compare_gauge(
            times=physical,
            model_pressure=model_pressure,
            reference_pressure=reference_pressure if reference_usable else None,
            ambient=ambient,
            pressure_scale=pressure_scale,
            time_scale=time_scale,
            impulse_scale=pressure_scale * time_scale,
            radius=gauge_radius,
        )
        entry.update(comparison.as_dict())
        entry["available"] = True
        if not reference_usable:
            entry.setdefault("notes", []).append(
                "参考时程含缺测样本，只报告模型指标，不做误差比较"
            )
        gauges.append(entry)
    report["gauges"] = gauges

    out_dir = Path(args.output) if args.output else config.path_of(
        "training.output_dir", default=Path("outputs/run")
    ) / "compare"
    write_json(out_dir / f"compare_{args.case}.json", report)
    print(_render(report))
    print(f"\nwritten: {out_dir / f'compare_{args.case}.json'}")
    return 0


def _render(report: dict[str, Any]) -> str:
    lines = ["# 预测—验证对照", "", f"算例：{report['case_id']}", ""]
    checks = report["checks"]
    lines.append("## 单位、半径与时间零点")
    lines.append("")
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    traj = report["trajectories"]
    lines.append("## 前沿轨迹")
    lines.append("")
    if traj.get("available"):
        lines.append(f"- 归一化：{traj['normalisation']}")
        lines.append(f"- R_c 相对 RMS：{traj['contact_relative_rms']:.4e}")
        lines.append(f"- R_s 相对 RMS：{traj['shock_relative_rms']:.4e}")
        lines.append(f"- 重叠物理时间：{traj['overlap_physical_time_s']}")
        lines.append("")
        lines.append("| 测点 r/R_0 | 主激波到达（模型） | 主激波到达（参考） | 相对误差 |")
        lines.append("|---|---:|---:|---:|")
        for key, entry in traj.get("shock_arrival", {}).items():
            model_time = entry["model_s"]
            reference_time = entry["reference_s"]
            relative = entry["relative_error"]
            lines.append(
                f"| {key.replace('r_over_R0_', '')} | "
                f"{'—' if model_time is None else f'{model_time:.4e}'} | "
                f"{'—' if reference_time is None else f'{reference_time:.4e}'} | "
                f"{'—' if relative is None else f'{relative:.3e}'} |"
            )
    else:
        lines.append(f"- 不可评价：{traj.get('reason')}")
    fields = report["fields"]
    lines.append("")
    lines.append("## 场量（球壳体积加权相对 L2，分介质）")
    lines.append("")
    if fields.get("available"):
        lines.append(f"- 保护带：{fields['buffer_cells']} 个单元"
                     f"（{fields['buffer_width_m']:.4g} m，{fields['buffer_basis']}）")
        lines.append(f"- 归一化：{fields['metric']}")
        lines.append(f"- 固定尺度：{fields['normalisation_scales']}")
        lines.append("")
        lines.append("| 介质 | 场量 | 相对 L2 | 点数 | 最大绝对误差 |")
        lines.append("|---|---|---:|---:|---:|")
        for medium in ("product", "air"):
            for name in ("density", "velocity", "specific_energy"):
                key = f"{medium}_{name}_relative_l2"
                value = fields.get(key)
                # 判"不可评价"要看**值**，不能只看键在不在：零样本时这个键是存在的
                # （值是 ``nan``），只查键会让渲染去取配套的 ``_n_points``，而那一项
                # 在零样本路径上根本没有——整条命令就在最后一步崩掉。
                if key not in fields or value is None or not np.isfinite(value):
                    reason = "无可比点" if fields.get(f"{medium}_{name}_available") is False \
                        else "模型在支撑点上非有限"
                    lines.append(f"| {medium} | {name} | 不可评价（{reason}） | 0 | — |")
                    continue
                lines.append(
                    f"| {medium} | {name} | {value:.4e} | "
                    f"{fields[f'{medium}_{name}_n_points']} | "
                    f"{fields[f'{medium}_{name}_max_abs_error']:.4e} |"
                )
        lines.append("")
        lines.append(f"- 材料不匹配单元数：{fields['region_mismatch_cells']}"
                     "（分介质单独报告，不并入共同区域）")
        lines.append(f"- 环境态单元数（不参与场误差）：{fields['ambient_cells_excluded']} 标记启用")
    else:
        lines.append(f"- 不可评价：{fields.get('reason')}")
    lines.append("")
    lines.append("## 测点压力时程")
    lines.append("")
    lines.append("| r/R_0 | 峰值超压误差 | 到达时间误差 | 正压时长误差 | 正相冲量误差 | 正压完整 |")
    lines.append("|---|---|---|---|---|---|")
    for gauge in report["gauges"]:
        ratio = gauge["radius_m"] / report["charge_radius_m"]
        if not gauge.get("available", True):
            # 不可用的测点要**说出原因**，而不是在表里留一行空白让人猜。
            lines.append(f"| {ratio:.3g} | 不可用 | 不可用 | 不可用 | 不可用 | "
                         f"{gauge.get('reason', '')} |")
            continue
        errors = gauge.get("relative_errors", {})

        def fmt(key):
            value = errors.get(key)
            return "—" if value is None else f"{value:.3e}"

        reference = gauge.get("reference") or {}
        if reference and not reference.get("arrival_observed", True):
            arrival_note = "（到达在起点之前，只是上界）"
        else:
            arrival_note = ""
        lines.append(
            f"| {ratio:.3g} | "
            f"{fmt('peak_overpressure_relative')} | "
            f"{fmt('arrival_time_relative')}{arrival_note} | "
            f"{fmt('positive_duration_relative')} | {fmt('positive_impulse_relative')} | "
            f"{'是' if reference.get('positive_phase_complete') else '否'} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
