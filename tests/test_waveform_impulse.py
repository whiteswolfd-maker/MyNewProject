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
    """2 + 0.5 = 2.5：矩形段 2×1，三角段 (2+0)×0.5/2。

    **面积不变，完整性断言改了。** 这条曲线首点超压是 2、严格高于阈值，说明
    正相在记录开始**之前**就已经开始——左端未观测到。以前 ``positive_phase_complete``
    只表示"右端观测到"，于是它被断言为 ``True``；按已经明确的完整性要求
    （"整段正相都被记录覆盖"= 左端 **且** 右端），这一条必须是 ``False``。
    解析面积 2.5 **不改**：它是已覆盖区间上的正确积分，保留在
    ``positive_impulse`` 里，并由 ``positive_impulse_is_partial`` 标明是部分积分。
    """
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    assert metrics.positive_impulse == pytest.approx(2.5, abs=1e-12)
    assert metrics.positive_duration == pytest.approx(1.5, abs=1e-12)
    assert metrics.negative_impulse == pytest.approx(-2.5, abs=1e-12)
    assert metrics.arrival_observed is False
    assert metrics.phase_end_observed is True
    assert metrics.positive_phase_complete is False
    assert metrics.positive_phase_coverage == "start_missing"


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


# --------------------------------------------------------------------------
# 第七轮 §6 指标的可用性标志必须控制最终判分
# --------------------------------------------------------------------------


def _gauge(times, model_pressure, reference_pressure):
    from spherical_blast_pinn.evaluation.waveform import compare_gauge

    return compare_gauge(
        times=times,
        model_pressure=model_pressure,
        reference_pressure=reference_pressure,
        ambient=100.0,
        pressure_scale=100.0,
        time_scale=3.0,
        impulse_scale=300.0,
        radius=0.1,
    )


def test_an_incomplete_model_phase_removes_the_phase_errors():
    """**复核的原反例。** 模型正相尚未结束、参考已经结束：修前仍然输出正相时长
    与冲量误差——而那两边算的根本不是同一个量。

    这里要让两边的**左端都观测到**（首点不高于阈值），否则两侧都会因为左端缺失
    而不完整，测不出"只有右端截断"这一条。
    """
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    comparison = _gauge(
        times,
        # 首点 0（观测到起点），末端仍正压 → 右端截断
        model_pressure=np.array([100.0, 102.0, 104.0, 103.0, 102.0, 101.0, 101.0]),
        # 首点 0（观测到起点），1.5 s 之后回到零 → 两端都观测到
        reference_pressure=np.array([100.0, 102.0, 104.0, 101.0, 99.0, 99.0, 99.0]),
    )
    assert comparison.model.arrival_observed is True
    assert comparison.reference.arrival_observed is True
    assert comparison.model.phase_end_observed is False
    assert comparison.reference.phase_end_observed is True
    assert comparison.model.positive_phase_complete is False
    assert comparison.reference.positive_phase_complete is True
    assert "positive_duration_relative" not in comparison.errors
    assert "positive_impulse_relative" not in comparison.errors
    # 已覆盖区间上的积分值仍然保留在各自的指标里。
    assert np.isfinite(comparison.model.positive_impulse)
    assert np.isfinite(comparison.reference.positive_impulse)
    assert any("右端截断" in note for note in comparison.notes)


def test_both_truncated_is_reported_as_such():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    comparison = _gauge(
        times,
        model_pressure=np.array([102.0, 104.0, 103.0, 103.0]),
        reference_pressure=np.array([102.0, 104.0, 103.5, 103.5]),
    )
    assert "positive_impulse_relative" not in comparison.errors
    assert any("两条曲线" in note for note in comparison.notes)


def test_an_unobserved_arrival_removes_the_arrival_error():
    """**复核的原反例。** ``arrival_observed=False`` 时修前仍然输出到达时间误差，
    甚至显示为零——那衡量的是"两条曲线都从正压开始"，不是到达时间。"""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    comparison = _gauge(
        times,
        model_pressure=np.array([103.0, 104.0, 99.0, 99.0]),
        reference_pressure=np.array([101.0, 104.0, 99.0, 99.0]),
    )
    assert comparison.model.arrival_observed is False
    assert comparison.reference.arrival_observed is False
    assert "arrival_time_relative" not in comparison.errors
    assert any("到达" in note for note in comparison.notes)


def test_a_fully_observed_pair_still_gets_all_the_errors():
    """对照组：两边都观测到、都完整时，四项误差照常给出。"""
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    model = np.array([100.0, 102.0, 104.0, 102.0, 99.0, 99.0])
    reference = np.array([100.0, 101.5, 103.0, 101.0, 99.0, 99.0])
    comparison = _gauge(times, model, reference)
    assert comparison.model.arrival_observed is True
    assert comparison.reference.arrival_observed is True
    assert comparison.model.positive_phase_complete is True
    assert comparison.reference.positive_phase_complete is True
    for key in (
        "peak_overpressure_relative", "arrival_time_relative",
        "positive_duration_relative", "positive_impulse_relative",
    ):
        assert key in comparison.errors, key


# --------------------------------------------------------------------------
# 第七轮 §7 零阈值下，上穿零点不是正相终点
# --------------------------------------------------------------------------


def test_a_zero_threshold_does_not_confuse_the_start_with_the_end():
    """**复核的原反例。** 超压 ``[0, 2, 0, -1]``、阈值 0：正确面积 2、时长 2。

    修前：到达插值点恰好为零，找"第一个 ``<= 0``"时直接选中了这个**起点**，
    于是面积 0、时长 0，还报 ``positive_phase_complete=True``。
    """
    metrics = gauge_metrics(
        [0, 1, 2, 3], [100, 102, 100, 99], ambient=100, minimum_overpressure_ratio=0.0
    )
    assert metrics.positive_impulse == pytest.approx(2.0, abs=1e-12)
    assert metrics.positive_duration == pytest.approx(2.0, abs=1e-12)
    assert metrics.positive_phase_complete is True


def test_the_zero_threshold_case_still_reports_the_original_regression():
    """同一次改动不能把原来的 2.5 反例弄坏。"""
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    assert metrics.positive_impulse == pytest.approx(2.5, abs=1e-12)


def test_a_zero_threshold_with_a_leading_zero_plateau():
    """连续零起点：正相从第一个严格正值开始，终点仍是第一次下穿。"""
    metrics = gauge_metrics(
        [0, 1, 2, 3, 4], [100, 100, 102, 100, 99], ambient=100,
        minimum_overpressure_ratio=0.0,
    )
    assert metrics.positive_impulse == pytest.approx(2.0, abs=1e-12)
    assert metrics.positive_duration == pytest.approx(2.0, abs=1e-12)


def test_a_zero_threshold_that_never_goes_positive():
    """峰值恰好为 0：整条曲线没有正超压。

    规定下来的行为是"不可用"（``nan`` + 说明），**不是** 0——0 是一个"量出来的
    冲量"，而这里根本没有正压段可以积分。这条与"峰值小于 0"走同一条路径。
    """
    metrics = gauge_metrics(
        [0, 1, 2, 3], [100, 100, 100, 99], ambient=100, minimum_overpressure_ratio=0.0
    )
    assert metrics.peak_overpressure == pytest.approx(0.0)
    assert not np.isfinite(metrics.positive_impulse)
    assert any("没有正超压" in note for note in metrics.notes)


@pytest.mark.parametrize("ratio", [-1.0, float("nan"), float("inf")])
def test_an_invalid_threshold_is_refused(ratio):
    """零阈值是**合法**的；负值与非有限值不是。"""
    with pytest.raises(WaveformError, match="minimum_overpressure_ratio"):
        gauge_metrics(
            [0, 1, 2, 3], [102, 102, 98, 98], ambient=100,
            minimum_overpressure_ratio=ratio,
        )


# --------------------------------------------------------------------------
# 第八轮 §4 左端缺失的时程不能判为完整正相
# --------------------------------------------------------------------------

_TIMES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

#: 起点在环境压上（超压 0 ≤ 阈值）→ 左端观测到；末端回到负压 → 右端观测到。
_FULL = np.array([100.0, 102.0, 104.0, 101.0, 99.0, 99.0, 99.0])
#: 首点已经严格高于阈值 → 左端缺失；末端回到负压。
_LEFT_MISSING = np.array([103.0, 104.0, 103.0, 101.0, 99.0, 99.0, 99.0])
#: 左端观测到，末端仍正压 → 右端截断。
_RIGHT_TRUNCATED = np.array([100.0, 102.0, 104.0, 103.0, 102.0, 101.0, 101.0])
#: 首点已为正，末端仍正压 → 两端都缺。
_BOTH_MISSING = np.array([103.0, 104.0, 103.0, 102.0, 101.0, 101.0, 101.0])


@pytest.mark.parametrize(
    "model_pressure, reference_pressure, model_coverage, reference_coverage, judged",
    [
        (_LEFT_MISSING, _FULL, "start_missing", "full", False),
        (_FULL, _LEFT_MISSING, "full", "start_missing", False),
        (_LEFT_MISSING, _LEFT_MISSING, "start_missing", "start_missing", False),
        (_RIGHT_TRUNCATED, _FULL, "end_truncated", "full", False),
        (_FULL, _RIGHT_TRUNCATED, "full", "end_truncated", False),
        (_RIGHT_TRUNCATED, _RIGHT_TRUNCATED, "end_truncated", "end_truncated", False),
        (_BOTH_MISSING, _FULL, "start_missing_and_end_truncated", "full", False),
        (_FULL, _FULL, "full", "full", True),
    ],
)
def test_the_positive_phase_coverage_matrix(
    model_pressure, reference_pressure, model_coverage, reference_coverage, judged
):
    """左端/右端 × 模型/参考 的组合，逐格核对判分与原因。"""
    comparison = _gauge(_TIMES, model_pressure, reference_pressure)
    assert comparison.model.positive_phase_coverage == model_coverage
    assert comparison.reference.positive_phase_coverage == reference_coverage
    assert comparison.model.positive_phase_complete == (model_coverage == "full")
    assert comparison.reference.positive_phase_complete == (reference_coverage == "full")

    phase_keys = {"positive_duration_relative", "positive_impulse_relative"}
    if judged:
        assert phase_keys <= set(comparison.errors)
    else:
        assert not (phase_keys & set(comparison.errors)), (
            "任意一侧缺左端或右端时，完整正相指标不得判分"
        )
        assert comparison.notes, "不可比时必须给出具体原因"
    # 峰值超压两端都有定义，永远比较。
    assert "peak_overpressure_relative" in comparison.errors


def test_the_reasons_name_the_missing_side_and_edge():
    """三种缺失各有各的说法，不统一写成"被终止时间截断"。"""
    left = _gauge(_TIMES, _LEFT_MISSING, _FULL)
    assert any("左端缺失" in note for note in left.notes)
    assert not any("右端截断" in note for note in left.notes)

    right = _gauge(_TIMES, _RIGHT_TRUNCATED, _FULL)
    assert any("右端截断" in note for note in right.notes)
    assert not any("左端缺失" in note for note in right.notes)

    none = gauge_metrics([0, 1, 2], [98, 98, 98], ambient=100)
    assert none.positive_phase_coverage == "no_positive_phase"
    assert any("没有正超压" in note for note in none.notes)


def test_partial_integrals_are_kept_and_labelled():
    """已覆盖区间上的积分值保留，并明确标成部分积分；不可用误差不填 0。"""
    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    assert metrics.positive_impulse == pytest.approx(2.5, abs=1e-12)
    payload = metrics.as_dict()
    assert payload["positive_impulse_is_partial"] is True
    assert "covered part" in payload["positive_duration_covers"]
    assert payload["positive_phase_coverage"] == "start_missing"

    comparison = _gauge(_TIMES, _LEFT_MISSING, _FULL)
    assert "positive_impulse_relative" not in comparison.errors
    assert "positive_impulse_relative" not in comparison.absolute_errors


def test_a_start_exactly_on_the_zero_threshold_is_observed():
    """**首点恰在零上穿**：那一刻**是**观测到的起点，与"首点已经为正"不同。"""
    metrics = gauge_metrics(
        [0, 1, 2, 3], [100, 102, 100, 99], ambient=100, minimum_overpressure_ratio=0.0
    )
    assert metrics.arrival_observed is True
    assert metrics.positive_phase_complete is True
    assert metrics.positive_phase_coverage == "full"
    assert metrics.positive_impulse == pytest.approx(2.0, abs=1e-12)


def test_a_leading_zero_plateau_is_an_observed_start():
    metrics = gauge_metrics(
        [0, 1, 2, 3, 4], [100, 100, 102, 100, 99], ambient=100,
        minimum_overpressure_ratio=0.0,
    )
    assert metrics.arrival_observed is True
    assert metrics.positive_phase_complete is True
    assert metrics.positive_impulse == pytest.approx(2.0, abs=1e-12)


def test_a_first_sample_strictly_positive_is_a_missing_start():
    metrics = gauge_metrics([0, 1, 2], [103, 102, 98], ambient=100, minimum_overpressure_ratio=0.0)
    assert metrics.arrival_observed is False
    assert metrics.positive_phase_complete is False
    assert metrics.positive_phase_coverage == "start_missing"
    # 已覆盖区间上的面积仍然算得出来，只是它是部分积分。
    assert metrics.positive_impulse > 0.0
