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
    #: 到达时刻是**观测到的**还是**只被夹住的**。第一个采样点就已经高于阈值时，
    #: 真实到达在它之前，`arrival_time` 只是上界。把上界当观测值报出去，会让
    #: "曲线开头就正压"读成一个漂亮的到达时间。
    arrival_observed: bool = False
    positive_duration: float = float("nan")
    positive_impulse: float = float("nan")
    negative_impulse: float = float("nan")
    positive_phase_complete: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_m": self.radius,
            "peak_overpressure_pa": self.peak_overpressure,
            "arrival_time_s": self.arrival_time,
            "arrival_observed": self.arrival_observed,
            "positive_duration_s": self.positive_duration,
            "positive_impulse_pa_s": self.positive_impulse,
            "negative_impulse_pa_s": self.negative_impulse,
            "positive_phase_complete": self.positive_phase_complete,
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

    metrics = GaugeMetrics(radius=float("nan"))
    peak = float(values.max())
    metrics.peak_overpressure = peak
    if peak <= 0.0:
        metrics.notes.append("整条时程都没有正超压")
        return metrics

    threshold = minimum_overpressure_ratio * peak
    arrival = first_crossing(times, values, threshold)
    if arrival is None:
        metrics.notes.append("找不到正超压的上穿点")
        return metrics
    metrics.arrival_time = arrival
    # 第一个采样点就已经高于阈值时，真实到达在它**之前**，这里给出的只是上界。
    metrics.arrival_observed = bool(arrival > float(times[0]))
    if not metrics.arrival_observed:
        metrics.notes.append(
            "第一个采样点已经高于阈值：真实到达早于曲线起点，arrival_time 只是上界，"
            "到达时间误差因此不可用"
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

    negative_index = np.flatnonzero(phase_values <= 0.0)
    if negative_index.size == 0:
        metrics.positive_phase_complete = False
        metrics.notes.append(
            "曲线在正压阶段结束之前被终止时间截断：正相冲量标为不完整，不作为完整的正相冲量使用"
        )
        end_time = float(phase_times[-1])
        metrics.positive_duration = end_time - arrival
        metrics.positive_impulse = float(_trapezoid(np.maximum(phase_values, 0.0), phase_times))
        return metrics

    stop = int(negative_index[0])
    if stop == 0:
        # 到达时刻本身就不在正压段里（阈值取 0、且上穿点正好落在零点）：正相区间
        # 退化成一个点，冲量为 0——不是拿后面的负值去凑一个面积。
        nodes_t = phase_times[:1]
        nodes_v = np.zeros(1)
        zero_time = float(phase_times[0])
    else:
        t0, t1 = float(phase_times[stop - 1]), float(phase_times[stop])
        v0, v1 = float(phase_values[stop - 1]), float(phase_values[stop])
        zero_time = t0 + (0.0 - v0) * (t1 - t0) / (v1 - v0) if v1 != v0 else t1
        # 节点 = 过零之前的**全部非负采样点**，末端接**真正的过零点**。
        # 首个负压点不留在节点里：它属于负相。
        nodes_t = phase_times[:stop]
        nodes_v = phase_values[:stop]
        # 过零恰好落在节点上时 ``zero_time`` 就是最后一个节点，不重复追加。
        if zero_time > float(nodes_t[-1]):
            nodes_t = np.append(nodes_t, zero_time)
            nodes_v = np.append(nodes_v, 0.0)

    metrics.positive_duration = float(nodes_t[-1]) - arrival
    metrics.positive_impulse = float(_trapezoid(np.maximum(nodes_v, 0.0), nodes_t))
    metrics.positive_phase_complete = True

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
    comparison.errors = {
        "peak_overpressure_relative": relative_error(
            model.peak_overpressure, reference.peak_overpressure, scale=pressure_scale
        ),
        "arrival_time_relative": relative_error(
            model.arrival_time, reference.arrival_time, scale=time_scale
        ),
    }
    comparison.absolute_errors = {
        "peak_overpressure_pa": model.peak_overpressure - reference.peak_overpressure,
        "arrival_time_s": model.arrival_time - reference.arrival_time,
    }

    if reference.positive_phase_complete:
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
    else:
        comparison.notes.append(
            "参考曲线被终止时间截断，正压阶段不完整：正相持续时间与冲量不参与判分"
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
