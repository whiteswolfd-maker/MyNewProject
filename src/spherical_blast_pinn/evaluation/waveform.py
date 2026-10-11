"""压力时程指标（提示词 §14）。

在固定的测点上比较模型与参考的压力时程：主激波到达时间、峰值超压、正压
持续时间、正相冲量，以及整条曲线。

两条规范要求直接体现在实现里：

* **判据预先固定**。"到达"定义为零上穿，"正压阶段结束"定义为首次由正变负
  的过零；两者的阈值与噪声处理写在这里，不在看到结果之后再改。
* **截断要如实标注**。曲线在正压阶段结束之前就被终止时间截断时，正相冲量
  标为不完整，**不虚构一个完整的冲量**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


#: ``numpy >= 2.0`` 把 ``trapz`` 改名为 ``trapezoid``，旧版本只有前者。
#: 两个名字各在一半的环境里可用，所以这里选一次而不是在每个调用点判断。
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class WaveformError(ValueError):
    """压力时程不合法。"""


@dataclass
class GaugeMetrics:
    """一个测点的全部指标。"""

    radius: float
    peak_overpressure: float = float("nan")
    arrival_time: float = float("nan")
    #: **相位起点是否被观测到。** 第一个采样点就已经严格高于阈值时，真实到达在
    #: 记录**之前**，``arrival_time`` 只是上界——即使后面观测到了过零，也不能
    #: 据此声称完整正相已覆盖。``False`` 的判据是"首点严格高于阈值"，不是
    #: "``arrival > times[0]``"：阈值为 0 时上穿点正好落在首点那个零上，
    #: 那一刻**是**被观测到的起点，与"首点已经为正"是两回事。
    arrival_observed: bool = False
    #: **相位终点是否被观测到**（曲线确实回到了零/负，不是被终止时间切断）。
    phase_end_observed: bool = False
    positive_duration: float = float("nan")
    positive_impulse: float = float("nan")
    negative_impulse: float = float("nan")
    #: **整段正相是否都被记录覆盖** = 起点观测到 **且** 终点观测到。
    #:
    #: 以前它只表示"终点观测到"。于是"首点已经为正、末端正常过零"的曲线会被
    #: 判成完整正相，正相时长与冲量照常参与判分——而左端缺失意味着测到的不是
    #: 完整相位。两个端点各自的事实保留在 ``arrival_observed`` 与
    #: ``phase_end_observed`` 里，细节留在 ``positive_phase_coverage``。
    positive_phase_complete: bool = False
    #: 覆盖情况的**分类**：``full`` / ``start_missing`` / ``end_truncated`` /
    #: ``start_missing_and_end_truncated`` / ``no_positive_phase`` /
    #: ``unavailable``。三种缺失各有各的原因，不合并成一句"被终止时间截断"。
    positive_phase_coverage: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_m": self.radius,
            "peak_overpressure_pa": self.peak_overpressure,
            "arrival_time_s": self.arrival_time,
            "arrival_observed": self.arrival_observed,
            "phase_end_observed": self.phase_end_observed,
            "positive_duration_s": self.positive_duration,
            "positive_duration_covers": (
                "the complete positive phase" if self.positive_phase_complete
                else "only the covered part of the positive phase"
            ),
            "positive_impulse_pa_s": self.positive_impulse,
            "positive_impulse_is_partial": not self.positive_phase_complete,
            "negative_impulse_pa_s": self.negative_impulse,
            "positive_phase_complete": self.positive_phase_complete,
            "positive_phase_coverage": self.positive_phase_coverage,
            "notes": list(self.notes),
        }


def overpressure(pressure: np.ndarray, ambient: float) -> np.ndarray:
    return np.asarray(pressure, dtype=np.float64) - float(ambient)


def first_crossing(times: np.ndarray, values: np.ndarray, level: float = 0.0) -> float | None:
    """首次由下向上穿过 ``level`` 的线性插值时刻。"""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    above = values > float(level)
    index = int(np.argmax(above)) if np.any(above) else -1
    if index <= 0:
        return float(times[0]) if index == 0 and above[0] else None
    t0, t1 = float(times[index - 1]), float(times[index])
    v0, v1 = float(values[index - 1]), float(values[index])
    if v1 == v0:
        return t1
    return t0 + (float(level) - v0) * (t1 - t0) / (v1 - v0)


def gauge_metrics(
    times: Sequence[float],
    pressure: Sequence[float],
    *,
    ambient: float,
    minimum_overpressure_ratio: float = 1.0e-4,
) -> GaugeMetrics:
    """一个测点的指标。

    ``minimum_overpressure_ratio`` 是一条**预先固定**的噪声门槛：低于
    ``ratio × 峰值超压`` 的过压算作零，以免数值噪声把到达时间提前到第一个点。
    """
    times = np.asarray(times, dtype=np.float64)
    values = overpressure(np.asarray(pressure, dtype=np.float64), ambient)
    if times.size != values.size:
        raise WaveformError("times and pressure must have the same length")
    if times.size < 3:
        raise WaveformError("a waveform needs at least three samples")
    if not np.all(np.isfinite(times)):
        raise WaveformError("times contains non-finite values")
    if np.any(np.diff(times) <= 0.0):
        raise WaveformError("times must be strictly increasing")
    # **缺测段不跨着积分。** 一段中间带 NaN 的时程可以把面积凭空造出来或抹掉，
    # 而结果看上去只是一个数。要处理缺口就把缺口切出去，分段评估——
    # 这里不替调用方决定缺口属于哪一段。
    if not np.all(np.isfinite(values)):
        raise WaveformError(
            "pressure contains non-finite values; a waveform with gaps cannot be "
            "integrated as one record — split it at the gaps and evaluate each piece"
        )
    # 阈值本身要合法。**零阈值是合法的**——噪声为零的解析曲线就该能用 0——
    # 所以这里只拒绝负数与非有限值；上一版的零阈值错误由相位起点的处理修掉，
    # 不是靠"禁止 0"盖过去。
    ratio = float(minimum_overpressure_ratio)
    if not np.isfinite(ratio) or ratio < 0.0:
        raise WaveformError(
            f"minimum_overpressure_ratio must be finite and non-negative, got "
            f"{minimum_overpressure_ratio!r}"
        )

    metrics = GaugeMetrics(radius=float("nan"))
    peak = float(values.max())
    metrics.peak_overpressure = peak
    if peak <= 0.0:
        metrics.notes.append("整条时程都没有正超压")
        metrics.positive_phase_coverage = "no_positive_phase"
        return metrics

    threshold = minimum_overpressure_ratio * peak
    arrival = first_crossing(times, values, threshold)
    if arrival is None:
        metrics.notes.append("找不到正超压的上穿点")
        metrics.positive_phase_coverage = "no_positive_phase"
        return metrics
    metrics.arrival_time = arrival
    # **相位起点是否被观测到：看首点相对阈值的位置，不看 arrival 落在哪。**
    #
    # 阈值为 0 时，上穿点正好落在首点那个零上——那一刻**是**观测到的起点，
    # 只是它恰好为零。而"首点已经严格高于阈值"才说明真实起始在记录之前。
    # 用 ``arrival > times[0]`` 当唯一判据会把这两种情形混成一种。
    metrics.arrival_observed = bool(float(values[0]) <= threshold)
    if not metrics.arrival_observed:
        metrics.notes.append(
            "第一个采样点已经严格高于阈值：正相在记录开始之前就已经开始，"
            "arrival_time 只是上界，起点未观测到"
        )

    # **积分节点必须严格递增，起点必须是真正的到达时刻。**
    #
    # 这里以前有两处会把面积算错，而两处的结果都"看起来合理"：
    #
    # ① 把首个负压采样点留在节点里，再把插出来的过零时刻**追加到数组末尾**——
    #    节点序列出现倒序（…, 2, 1.5）。``_trapezoid`` 只是逐段求和，不排序、
    #    也不报错，于是 [0,1,2,3]/[2,2,-2,-2] 的 2.5 被算成 3.0。
    # ② ``after = times >= arrival`` 把到达时刻到第一个采样点之间那一段整个丢掉，
    #    到达落在两点之间时首段面积凭空消失。
    if arrival <= float(times[0]):
        # 到达就在第一个采样点上：那一点的超压是**实测值**，不是阈值。
        arrival_value = float(values[0])
    else:
        # 到达落在两点之间，上穿点处的超压按定义就是阈值。
        arrival_value = float(threshold)
    phase_times = np.concatenate(([float(arrival)], times[times > arrival]))
    phase_values = np.concatenate(([arrival_value], values[times > arrival]))

    # **相位起点与相位终点是两件事。** 阈值取 0 时，上穿点就落在零点上，于是
    # ``phase_values[0]`` 本身就是 0——它是**正相的起点**。从下标 0 开始找第一个
    # ``<= 0`` 会立刻选中这个起点，把整段正相判成"退化成一个点"，面积 0、时长 0，
    # 还报 ``positive_phase_complete=True``。
    #
    # 正确的顺序是：先定位**严格正**的区间，再从那里往后找第一次下穿（或到零）。
    strictly_positive = np.flatnonzero(phase_values > 0.0)
    if strictly_positive.size == 0:
        # 到达之后一次正压都没有：正相区间退化成一个点。
        metrics.positive_duration = 0.0
        metrics.positive_impulse = 0.0
        metrics.phase_end_observed = True
        metrics.positive_phase_complete = metrics.arrival_observed
        metrics.positive_phase_coverage = (
            "full" if metrics.arrival_observed else "start_missing"
        )
        metrics.notes.append(
            "到达时刻之后没有任何正超压：正相区间退化成一个点，冲量为 0"
        )
        return metrics

    first_positive = int(strictly_positive[0])
    tail = np.flatnonzero(phase_values[first_positive:] <= 0.0)
    if tail.size == 0:
        # **右端截断**：曲线在正相结束之前就被终止时间切断。
        metrics.phase_end_observed = False
        metrics.positive_phase_complete = False
        metrics.positive_phase_coverage = (
            "start_missing_and_end_truncated" if not metrics.arrival_observed
            else "end_truncated"
        )
        metrics.notes.append(
            "曲线在正压阶段结束之前被终止时间截断（右端未观测到）：正相冲量标为"
            "不完整，不作为完整的正相冲量使用"
        )
        end_time = float(phase_times[-1])
        metrics.positive_duration = end_time - arrival
        metrics.positive_impulse = float(_trapezoid(np.maximum(phase_values, 0.0), phase_times))
        return metrics

    stop = first_positive + int(tail[0])
    t0, t1 = float(phase_times[stop - 1]), float(phase_times[stop])
    v0, v1 = float(phase_values[stop - 1]), float(phase_values[stop])
    zero_time = t0 + (0.0 - v0) * (t1 - t0) / (v1 - v0) if v1 != v0 else t1
    # 节点 = 过零之前的**全部**采样点（包括起点那个 0），末端接**真正的过零点**。
    # 首个非正点不留在节点里：它属于负相。起点那个 0 留在节点里，它是相位的左端。
    nodes_t = phase_times[:stop]
    nodes_v = phase_values[:stop]
    # 过零恰好落在节点上时 ``zero_time`` 就是最后一个节点，不重复追加。
    if zero_time > float(nodes_t[-1]):
        nodes_t = np.append(nodes_t, zero_time)
        nodes_v = np.append(nodes_v, 0.0)

    metrics.positive_duration = float(nodes_t[-1]) - arrival
    metrics.positive_impulse = float(_trapezoid(np.maximum(nodes_v, 0.0), nodes_t))
    # **两个端点都观测到，才叫完整正相。** 右端过零在这里是观测到的；左端是否
    # 观测到已经在上面判过（首点严格高于阈值 = 左端缺失）。
    metrics.phase_end_observed = True
    metrics.positive_phase_complete = bool(metrics.arrival_observed)
    metrics.positive_phase_coverage = (
        "full" if metrics.positive_phase_complete else "start_missing"
    )
    if not metrics.arrival_observed:
        metrics.notes.append(
            "左端未观测到：正相在记录开始之前就已经开始，已覆盖区间上的积分值"
            "保留，但它不是完整正相冲量"
        )

    # 负相冲量：从**同一个过零边界**起，节点同样严格递增。
    tail_times = phase_times[stop:]
    tail_values = phase_values[stop:]
    if zero_time < float(tail_times[0]):
        tail_times = np.concatenate(([zero_time], tail_times))
        tail_values = np.concatenate(([0.0], tail_values))
    if tail_times.size >= 2:
        metrics.negative_impulse = float(
            _trapezoid(np.minimum(tail_values, 0.0), tail_times)
        )
    return metrics


def relative_error(model: float, reference: float, *, scale: float) -> float:
    """相对误差，分母用**事先固定**的物理尺度，不是局部均值。

    近零分母下的相对误差没有意义；用一个固定的物理尺度，并同时报告绝对误差，
    才能看出误差是"相对大"还是"绝对小"。
    """
    reference = float(reference)
    denominator = max(abs(reference), abs(float(scale)), 1.0e-30)
    return abs(float(model) - reference) / denominator


@dataclass
class WaveformComparison:
    """一个测点上模型与参考的并排比较。"""

    radius: float
    model: GaugeMetrics
    reference: GaugeMetrics | None
    errors: dict[str, float] = field(default_factory=dict)
    absolute_errors: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_m": self.radius,
            "model": self.model.as_dict(),
            "reference": self.reference.as_dict() if self.reference else None,
            "relative_errors": self.errors,
            "absolute_errors": self.absolute_errors,
            "notes": self.notes,
        }


def compare_gauge(
    *,
    times: Sequence[float],
    model_pressure: Sequence[float],
    reference_pressure: Sequence[float] | None,
    ambient: float,
    pressure_scale: float,
    time_scale: float,
    impulse_scale: float,
    radius: float,
) -> WaveformComparison:
    """比较一个测点。

    ``pressure_scale``、``time_scale``、``impulse_scale`` 都是**事先固定**的参考
    尺度（通常来自参考解本身），不是每个测点各自的局部量。
    """
    model = gauge_metrics(times, model_pressure, ambient=ambient)
    model.radius = float(radius)
    if reference_pressure is None:
        empty = GaugeMetrics(radius=float(radius))
        return WaveformComparison(
            radius=float(radius), model=model, reference=None,
            notes=["没有参考时程，只报告模型指标"],
        )

    reference = gauge_metrics(times, reference_pressure, ambient=ambient)
    reference.radius = float(radius)

    comparison = WaveformComparison(radius=float(radius), model=model, reference=reference)
    # 峰值超压两边都有定义，无条件比较。
    comparison.errors = {
        "peak_overpressure_relative": relative_error(
            model.peak_overpressure, reference.peak_overpressure, scale=pressure_scale
        ),
    }
    comparison.absolute_errors = {
        "peak_overpressure_pa": model.peak_overpressure - reference.peak_overpressure,
    }

    # **到达时间误差：两边的到达都必须是被观测到的。**
    #
    # ``arrival_observed=False`` 表示曲线起点就已经高于阈值，真实到达在记录之外,
    # 给出的只是**上界**。拿两个上界相减会得到 0——一个"两边完全一致"的假象，
    # 而它衡量的是"两条曲线都从正压开始"，不是到达时间。
    if model.arrival_observed and reference.arrival_observed:
        comparison.errors["arrival_time_relative"] = relative_error(
            model.arrival_time, reference.arrival_time, scale=time_scale
        )
        comparison.absolute_errors["arrival_time_s"] = (
            model.arrival_time - reference.arrival_time
        )
    else:
        comparison.notes.append(
            "到达时间不可比较："
            + ("模型" if not model.arrival_observed else "")
            + ("与" if not model.arrival_observed and not reference.arrival_observed else "")
            + ("参考" if not reference.arrival_observed else "")
            + "曲线的起点已经处于正相，真实到达在记录之外，给出的只是上界"
        )

    # **正相时长与冲量：两边的完整正相都要被覆盖。**
    #
    # 以前只检查了参考一侧的**右端**。三种缺失现在各有各的说法：
    #
    # * 左端缺失：曲线起点已经在正相里，记录之前那一段没观测到；
    # * 右端截断：曲线在回到零之前就被终止时间切断；
    # * 没有正相：整条曲线不含正超压。
    #
    # 任意一侧缺任意一端，两边算的就不是同一个量：一边是"已覆盖区间上的部分
    # 积分"，另一边是完整冲量，相减得到的误差既不是模型误差也不是数据误差。
    # 已覆盖区间上的积分值仍然保留在各自的 ``positive_impulse`` 里，只是不当成
    # 可比指标——``positive_impulse_is_partial`` 会标出来。
    _COVERAGE_WORDS = {
        "full": "两端都观测到",
        "start_missing": "左端缺失（起点已在正相里，记录之前没观测到）",
        "end_truncated": "右端截断（回到零之前被终止时间切断）",
        "start_missing_and_end_truncated": "左端缺失且右端截断",
        "no_positive_phase": "整条曲线没有正超压",
        "unavailable": "指标不可用",
    }
    unusable = [
        (name, _COVERAGE_WORDS.get(entry.positive_phase_coverage, entry.positive_phase_coverage))
        for name, entry in (("模型", model), ("参考", reference))
        if not entry.positive_phase_complete
    ]
    if unusable and len(unusable) == 2:
        comparison.notes.append(
            "两条曲线的正相都不完整（"
            + "；".join(f"{name}：{why}" for name, why in unusable)
            + "）：正相持续时间与冲量不参与判分，各自的已覆盖区间积分值仍保留"
        )
    elif unusable:
        name, why = unusable[0]
        comparison.notes.append(
            f"{name}曲线的正相不完整（{why}）：正相持续时间与冲量不参与判分，"
            "已覆盖区间上的积分值仍保留在各自指标里"
        )

    if model.positive_phase_complete and reference.positive_phase_complete:
        comparison.errors["positive_duration_relative"] = relative_error(
            model.positive_duration, reference.positive_duration, scale=time_scale
        )
        comparison.errors["positive_impulse_relative"] = relative_error(
            model.positive_impulse, reference.positive_impulse, scale=impulse_scale
        )
        comparison.absolute_errors["positive_duration_s"] = (
            model.positive_duration - reference.positive_duration
        )
        comparison.absolute_errors["positive_impulse_pa_s"] = (
            model.positive_impulse - reference.positive_impulse
        )
    return comparison


__all__ = [
    "GaugeMetrics",
    "WaveformComparison",
    "WaveformError",
    "compare_gauge",
    "first_crossing",
    "gauge_metrics",
    "overpressure",
    "relative_error",
]
