"""正相冲量的过零积分（提示词 §8）。

复核给的反例：``times = [0, 1, 2, 3]``、``pressure = [102, 102, 98, 98]``、
``ambient = 100``。超压是 ``[2, 2, -2, -2]``，零点在 ``t = 1.5``。在**已覆盖区间**
``[0, 1.5]`` 上，正超压折线的面积是 ``2.5``。

修前的实现给出 ``3.0``，因为它做了两件事：

* 把首个负压采样点（``t = 2``）留在积分节点里，再把插出来的过零时刻
  （``t = 1.5``）**追加到数组末尾**——节点序列 ``[0, 1, 2, 1.5]`` 出现倒序。
  ``trapezoid`` 只是逐段求和，既不排序也不报错；
* ``times >= arrival`` 把到达时刻到第一个采样点之间的那一段整个丢掉。

预期值在每一个用例里都由**独立的解析面积**给出，不从实现公式反推。
"""
from __future__ import annotations

import numpy as np
import pytest

from spherical_blast_pinn.evaluation.waveform import WaveformError, gauge_metrics


def test_the_review_counterexample_is_2_point_5():
    """2 + 0.5 = 2.5：矩形段 2×1，三角段 (2+0)×0.5/2。"""
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    assert metrics.positive_impulse == pytest.approx(2.5, abs=1e-12)
    assert metrics.positive_duration == pytest.approx(1.5, abs=1e-12)
    assert metrics.negative_impulse == pytest.approx(-2.5, abs=1e-12)
    assert metrics.positive_phase_complete is True


def test_the_integration_nodes_are_strictly_increasing_on_the_covered_interval():
    """面积对不对是结果，节点顺序对不对是原因。

    单独把这一点钉住：即使某天面积碰巧对了，节点里再出现 ``(…, 2, 1.5)`` 这样的
    倒序仍然是错的——梯形法不会告诉你。
    """
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    # duration 必须等于"最后一个节点 − 到达时刻"，而最后一个节点是过零点，
    # 不是那个负压采样点。用错节点会得到 2.0 而不是 1.5。
    assert metrics.positive_duration == pytest.approx(1.5)
    assert metrics.positive_duration != pytest.approx(2.0)


def test_a_triangle_has_half_base_times_height():
    """峰值 4、底宽 2 的三角形，面积 = 4。解析值，与实现无关。"""
    times = np.linspace(0.0, 2.0, 20001)
    overpressure = np.where(times <= 1.0, 4.0 * times, 4.0 * (2.0 - times))
    metrics = gauge_metrics(times, 100.0 + overpressure, ambient=100.0)
    # 峰值 4、底宽 2 的三角形：面积 = 4。
    assert metrics.positive_impulse == pytest.approx(4.0, rel=1e-3)
    # 到达落在两点之间（阈值上穿），所以时长比 2.0 少一个 arrival。
    assert metrics.positive_duration == pytest.approx(2.0, rel=1e-3)


def test_a_trapezoid_has_the_analytic_area():
    """下底 2 的对边、上底 1、高 1 的梯形：面积 = (2 + 1) / 2 × 1 = 1.5。"""
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    overpressure = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    metrics = gauge_metrics(times, 100.0 + overpressure, ambient=100.0)
    # 到达是阈值上穿（≈ 5e-5），所以比精确的 1.5 少掉那一小段。
    assert metrics.positive_impulse == pytest.approx(1.5, abs=1e-6)


def test_a_zero_crossing_exactly_on_a_node_is_not_duplicated():
    """过零恰好落在节点上：那一**就是**相位终点，不再从 0 到 2 补一段。

    超压 ``[2, 2, 0, -2]`` 的面积 = 矩形 2×1 + 三角形 (2+0)/2×1 = 3。
    若实现把过零点**追加**到节点末尾（``[0, 1, 2, 2.0]``），面积不变但节点序列
    出现重复；若换成从第一个非正点起算，则会多算或漏算。
    """
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 100, 98], ambient=100)
    assert metrics.positive_impulse == pytest.approx(3.0, abs=1e-12)
    assert metrics.positive_duration == pytest.approx(2.0, abs=1e-12)


def test_arrival_between_two_nodes_keeps_the_first_segment():
    """到达落在两点之间时，首段面积不能丢。

    构造：``t = 0`` 起线性升到 ``t = 1`` 的峰值 1，再线性降到 ``t = 2`` 的 0。
    阈值取极小，所以到达 ≈ 0，面积 = 1。
    若实现从"第一个 ≥ arrival 的采样点"起算，会丢掉 ``[arrival, t_1]`` 这一小段。
    """
    times = np.linspace(0.0, 2.0, 2001)
    overpressure = np.where(times <= 1.0, times, 2.0 - times)
    metrics = gauge_metrics(times, 100.0 + overpressure, ambient=100.0)
    assert metrics.positive_impulse == pytest.approx(1.0, rel=1e-2)
    # 阈值上穿发生在 0 之后一点点，所以到达时刻是一个很小的正数。
    assert metrics.arrival_time == pytest.approx(0.0, abs=1e-3)
    assert metrics.arrival_observed is True


def test_a_waveform_that_starts_positive_does_not_claim_an_observed_arrival():
    """首个采样点已经正压时，真实到达在曲线起点**之前**。

    ``arrival_time`` 只是上界，不能当成观测到的到达时间报出去。冲量仍然是**已覆盖
    区间**上的正确积分——这不代表完整爆炸时程已经覆盖。
    """
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    assert metrics.arrival_observed is False
    assert any("上界" in note for note in metrics.notes)


def test_a_waveform_still_positive_at_the_end_is_marked_incomplete():
    """末端仍正压：不是完整的正相冲量，标为不完整。"""
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 101, 101], ambient=100)
    assert metrics.positive_phase_complete is False
    assert metrics.positive_impulse == pytest.approx(4.5, abs=1e-12)
    assert any("截断" in note for note in metrics.notes)


def test_a_waveform_without_any_positive_overpressure_reports_that():
    metrics = gauge_metrics([0, 1, 2], [98, 98, 98], ambient=100)
    assert metrics.peak_overpressure == pytest.approx(-2.0)
    assert not np.isfinite(metrics.positive_impulse)
    assert any("没有正超压" in note for note in metrics.notes)


def test_a_consecutive_zero_stretch_is_an_exact_zero_crossing():
    """连续零值：相位在**第一个**零点结束，后面的零不延长正相。

    超压 ``[2, 2, 0, 0, -2]`` 的面积与上一例相同（3），只是零值延续了两格——
    若实现把"最后一个非正值"当成终点，会把相位拉到 4。
    """
    metrics = gauge_metrics([0, 1, 2, 3, 4], [102, 102, 100, 100, 98], ambient=100)
    assert metrics.positive_impulse == pytest.approx(3.0, abs=1e-12)
    assert metrics.positive_duration == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize(
    "times, pressure, needle",
    [
        ([0, 1, 2, 3], [102, np.nan, 98, 98], "non-finite"),
        ([0, 1, 2, 3], [102, np.inf, 98, 98], "non-finite"),
        ([0, np.nan, 2, 3], [102, 102, 98, 98], "non-finite"),
        ([0, 2, 1, 3], [102, 102, 98, 98], "strictly increasing"),
        ([0, 1, 1, 3], [102, 102, 98, 98], "strictly increasing"),
    ],
)
def test_gaps_and_invalid_time_axes_are_refused(times, pressure, needle):
    """**不跨缺测段编造冲量。** 一段中间带 NaN 的时程可以把面积凭空造出来或抹掉，
    而结果看上去只是一个数。非法时间轴同样要在积分之前拒绝。"""
    with pytest.raises(WaveformError, match=needle):
        gauge_metrics(times, pressure, ambient=100)


def test_too_few_samples_is_refused():
    with pytest.raises(WaveformError, match="at least three"):
        gauge_metrics([0, 1], [102, 98], ambient=100)


def test_mismatched_lengths_are_refused():
    with pytest.raises(WaveformError, match="same length"):
        gauge_metrics([0, 1, 2], [102, 98], ambient=100)
