"""§14 三阶段训练机的定向回归。

覆盖三件事，每件单独都能静默出错：

* **时间窗口只扩大**，早期采样点一直留在池子里；
* **阶段推进由预先声明的独立检查决定**，判据不通过就明确停止，不静默跳阶段；
* **有界梯度平衡**不拿近零梯度做除数、不给零样本组建统计、不接管 ``lambda_PDE``。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import torch

from spherical_blast_pinn.losses.objective import BASE_WEIGHTS
from spherical_blast_pinn.sampling.batch import SampleBatch, restrict_to_time_window
from spherical_blast_pinn.training.balancing import (
    BalanceConfig,
    BoundedGradientBalancer,
    live_groups,
)
from spherical_blast_pinn.training.schedule import (
    SCHEDULE,
    STAGE_CRITERIA,
    ScheduleError,
    Stage,
    StageCriterion,
    criteria_for,
    describe_schedule,
    time_window_fraction_at,
)

from test_review_regressions import (  # noqa: E402  (tests/ 由 pytest 放进 sys.path)
    TIME_SCALE,
    _scales,
    _tiny_reference,
    _write_case,
)


# ==========================================================================
# 时间窗口（§14.2）
# ==========================================================================


def test_the_declared_schedule_expands_the_window_and_never_narrows_it():
    """§14.2 的"逐步扩大"必须是日程本身的性质，不能只写在文档里。"""
    windows = [stage.time_window_fraction for stage in SCHEDULE]
    assert windows == sorted(windows), "窗口必须单调不减"
    assert windows[0] < windows[-1], "窗口必须真的扩大，而不是三个阶段一样大"
    # 三个阶段必须各有自己的窗口；三个相同的数字等于没有这个机制。
    assert len(set(windows)) == len(windows)


def test_the_time_window_is_a_function_of_the_step():
    assert time_window_fraction_at(0) == pytest.approx(SCHEDULE[0].time_window_fraction)
    # 阶段内恒定，阶段边界跳变
    last_of_first = SCHEDULE[0].updates - 1
    assert time_window_fraction_at(last_of_first) == pytest.approx(
        SCHEDULE[0].time_window_fraction
    )
    assert time_window_fraction_at(SCHEDULE[0].updates) == pytest.approx(
        SCHEDULE[1].time_window_fraction
    )


def test_a_schedule_that_narrows_the_window_is_refused():
    """把收窄做成"照做但记一笔"是不行的：报表上只会看到一个更小的百分比。"""
    from spherical_blast_pinn.training.schedule import _assert_window_is_non_decreasing

    narrowed = (
        Stage("initialise", 1, 0.1, 0.1, 1e-3, 1e-3, time_window_fraction=0.8),
        Stage("strengthen_pde", 1, 0.1, 1.0, 1e-3, 1e-4, time_window_fraction=0.3),
    )
    with pytest.raises(ScheduleError, match="narrows"):
        _assert_window_is_non_decreasing(narrowed)


def test_the_sampler_expands_its_window_and_keeps_the_early_frames():
    """扩大窗口时早期帧必须留在池子里（§14.2"保留早期采样"）。"""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        case = _write_case(tmp)
        scales = _scales(case.charge_radius)
        from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig

        sampler = RadiusSampler(
            case=case, scales=scales, reference=_tiny_reference(),
            config=SamplingConfig(
                n_pde_product=4, n_pde_air=4, n_ic_product=2, n_ic_air=2,
                n_contact=2, n_shock=2, n_data_product=4, n_data_air=4, n_data_front=3,
            ),
            seed=0,
        )
        full = sampler.full_pinn_time_max
        assert full > 0.0

        sampler.set_time_window(0.25)
        first = set(int(f) for f in sampler.window_frames)
        first_max = sampler.window_pinn_time_max
        assert first_max == pytest.approx(0.25 * full)
        assert first, "25 % 窗口里应当有可监督帧，否则这条测试没有意义"

        sampler.set_time_window(0.65)
        second = set(int(f) for f in sampler.window_frames)
        assert first <= second, (
            "扩窗后早期帧被丢掉了：窗口虽然变大，但早期采样点退出了池子"
        )
        assert sampler.window_pinn_time_max == pytest.approx(0.65 * full)

        sampler.set_time_window(1.0)
        third = set(int(f) for f in sampler.window_frames)
        assert second <= third
        assert sampler.window_pinn_time_max == pytest.approx(full)
        # 窗口恒以 0 为左端：不是"整段平移"，所以早期帧一直在里面。
        assert sampler.pinn_time_min == 0.0


def test_the_sampler_refuses_to_narrow_its_time_window():
    """收窄会让已经用过的时段重新退出采样，而表面上只是百分比变小。"""
    import tempfile

    from spherical_blast_pinn.sampling.sampler import (
        RadiusSampler,
        SamplingConfig,
        SamplingError,
    )

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        case = _write_case(tmp)
        scales = _scales(case.charge_radius)
        sampler = RadiusSampler(
            case=case, scales=scales, reference=_tiny_reference(),
            config=SamplingConfig(
                n_pde_product=4, n_pde_air=4, n_ic_product=2, n_ic_air=2,
                n_contact=2, n_shock=2, n_data_product=4, n_data_air=4, n_data_front=2,
            ),
            seed=0,
        )
        # 第一次是"设定"，可以往任一方向——采样器刚建好时不属于任何阶段。
        sampler.set_time_window(0.25)
        sampler.set_time_window(0.65)
        with pytest.raises(SamplingError, match="shrink"):
            sampler.set_time_window(0.25)
        # 拒绝之后状态没有被改坏
        assert sampler.window_fraction == pytest.approx(0.65)


def _handmade_batch() -> SampleBatch:
    """一个手搭的 batch：``s_product`` 与 ``s_air`` 长度相同、取值不同。

    这正是"按长度猜时间坐标"会出错的地方：两者行数一样，猜错不会报错，只会把
    一个介质的点按另一个介质的时间坐标裁掉。
    """
    def col(values):
        return torch.as_tensor(
            np.asarray(values, dtype=np.float64).reshape(-1, 1)
        )

    scales = _scales(0.05)
    batch = SampleBatch(
        case_id="handmade",
        charge_radius=0.05,
        detonation_velocity=6930.0,
        scales=scales,
        pinn_time_min=0.0,
        pinn_time_max=10.0,
    )
    batch.s_product = col([1.0, 5.0, 9.0])
    batch.xi_product = col([0.1, 0.2, 0.3])
    batch.s_air = col([9.0, 5.0, 1.0])          # 与 s_product 同长、顺序相反
    batch.eta_air = col([0.4, 0.5, 0.6])
    batch.s_data_product = col([2.0, 8.0])
    batch.rho_data_product = col([1.0, 2.0])
    batch.lo_data_product = col([0.01, 0.02])
    batch.hi_data_product = col([0.02, 0.03])
    batch.xi_ic_product = col([0.1, 0.5, 0.9])
    batch.rho_ic_product = col([1.0, 1.0, 1.0])
    batch.geometry_delta_contact = 1.0e-3
    batch.geometry_delta_shock = 2.0e-3
    return batch


def test_the_window_clip_uses_each_fields_own_time_coordinate():
    batch = _handmade_batch()
    clipped = restrict_to_time_window(batch, 6.0)

    # 产物侧：留下 s <= 6 的两点（1 和 5），丢掉 9
    assert clipped.s_product.reshape(-1).tolist() == [1.0, 5.0]
    assert clipped.xi_product.reshape(-1).tolist() == [0.1, 0.2]
    # 空气侧：坐标顺序相反，留下的应当是 1 和 5，而不是"前两行"
    assert clipped.s_air.reshape(-1).tolist() == [5.0, 1.0]
    assert clipped.eta_air.reshape(-1).tolist() == [0.5, 0.6]
    # 数据侧：只留 s <= 6 的那一点
    assert clipped.s_data_product.reshape(-1).tolist() == [2.0]
    assert clipped.hi_data_product.reshape(-1).tolist() == [0.02]

    # 原 batch 不被就地修改
    assert batch.s_product.reshape(-1).tolist() == [1.0, 5.0, 9.0]


def test_the_window_clip_keeps_the_initial_condition_points():
    """初始条件在 ``t_PINN = 0`` 上评价，与时间窗口无关，必须全留。

    按长度一起裁的话，"初值预训练"会在扩窗那一步把自己的一半初值点削掉——而
    初值点本来就该是全时段都成立的。
    """
    batch = _handmade_batch()
    clipped = restrict_to_time_window(batch, 0.5)
    assert clipped.xi_ic_product.reshape(-1).tolist() == [0.1, 0.5, 0.9]
    assert clipped.rho_ic_product.numel() == 3
    # 时间相关的照常被裁干净（这个窗口里一个 PDE 点都不在）
    assert clipped.s_product.numel() == 0
    assert clipped.s_air.numel() == 0


def test_the_window_clip_records_the_window_it_used():
    clipped = restrict_to_time_window(_handmade_batch(), 6.0)
    assert clipped.diagnostics["time_window_s_max"] == pytest.approx(6.0)
    assert clipped.pinn_time_max == pytest.approx(6.0)


def test_the_initial_condition_targets_never_come_from_the_model_itself():
    """§14.2：不能把前一阶段的预测冻结为新的"真初值"。

    三层检查，缺一层就有一种静默错误的活路：

    1. 在同一批点上、换一个时间窗口，初值标签**一个字节都不变**——窗口不参与
       初值标签的生成；
    2. 它们确实等于参考 B 初场在启动时刻的取值（独立重算，不调用被测代码）；
    3. 它们**不等于**网络自己在那些点上的输出——若把前一阶段预测冻结成"真初值"，
       这一条会立刻失败，而损失曲线仍然一切正常。
    """
    import tempfile

    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet
    from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        case = _write_case(tmp)
        scales = _scales(case.charge_radius)
        reference = _tiny_reference()
        sampler = RadiusSampler(
            case=case, scales=scales, reference=reference,
            config=SamplingConfig(
                n_pde_product=4, n_pde_air=4, n_ic_product=5, n_ic_air=5,
                n_contact=2, n_shock=2, n_data_product=4, n_data_air=4, n_data_front=2,
            ),
            seed=0,
        )

        sampler.set_time_window(0.25)
        state = sampler.rng_state()
        first = sampler.batch()
        # 回到同一处随机状态，于是两次抽到的是**同一批**初值点；这样"标签相同"
        # 才是窗口无关的证据，而不是"点不同所以值不同"的巧合。
        sampler.set_time_window(1.0)
        sampler.set_rng_state(state)
        second = sampler.batch()

        for name in ("xi_ic_product", "rho_ic_product", "u_ic_product", "e_ic_product",
                     "eta_ic_air", "rho_ic_air", "u_ic_air", "e_ic_air"):
            assert torch.equal(getattr(first, name), getattr(second, name)), (
                f"{name} 随窗口变了；初值标签必须与时间窗口无关"
            )
        for name in ("contact_ic", "shock_ic"):
            assert getattr(first, name) == pytest.approx(getattr(second, name))

        # 第 2 层：独立重算参考场在启动时刻的取值
        start = scales.start_time()
        r_product = first.xi_ic_product.reshape(-1).numpy() * first.contact_ic
        rho, u, e = reference.sample(r_product, np.full_like(r_product, start))
        assert np.allclose(first.rho_ic_product.reshape(-1).numpy(), rho, rtol=1.0e-12)
        assert np.allclose(first.u_ic_product.reshape(-1).numpy(), u, rtol=1.0e-12)
        assert np.allclose(first.e_ic_product.reshape(-1).numpy(), e, rtol=1.0e-12)

        # 第 3 层：标签不是网络自己的输出
        model = CoupledModel(
            product=ProductNet(scales=scales, width=8, depth=1, seed=0),
            air=AirNet(scales=scales, width=8, depth=1, seed=1),
            front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.1, seed=2),
            scales=scales,
        )
        prediction = model.product_fields(
            torch.zeros_like(first.xi_ic_product), first.xi_ic_product
        )
        assert not np.allclose(
            first.rho_ic_product.reshape(-1).numpy(),
            prediction.density.detach().reshape(-1).numpy(),
            rtol=1.0e-6,
        ), "初值标签看起来等于网络自己的输出——那正是「把预测当成真值」的形态"


# ==========================================================================
# 阶段推进判据（§14）
# ==========================================================================


def test_every_declared_stage_has_predeclared_criteria():
    for stage in SCHEDULE:
        criteria = criteria_for(stage.name)
        assert criteria, f"阶段 {stage.name!r} 没有任何预先声明的判据"
    assert set(STAGE_CRITERIA) == {stage.name for stage in SCHEDULE}


def test_an_undeclared_stage_is_refused():
    """没有声明判据的阶段不能靠步数蒙混过关。"""
    with pytest.raises(ScheduleError, match="no declared advance criteria"):
        criteria_for("some_new_stage")


def test_stage_criteria_use_the_same_tolerance_table_as_acceptance():
    """判据的限额必须引用 §16 的容差表，不能另立第二套数。"""
    pde = [c for c in criteria_for("joint") if c.metric == "pde_rms_max"][0]
    assert pde.limit({"pde_rms": 0.5}) == pytest.approx(0.5)
    assert pde.limit({}) == pytest.approx(1.0e-2)  # 表里没有时退回声明的默认值


def test_stage_criterion_evaluation_fails_closed():
    """指标缺失或非有限一律判为**未通过**，不能当成"没有超出"。"""
    criterion = StageCriterion("pde_rms_max", "max", fallback=1.0e-2)
    assert criterion.evaluate({"pde_rms_max": 0.0})[0] is True
    assert criterion.evaluate({"pde_rms_max": 0.02})[0] is False
    assert criterion.evaluate({})[0] is False, "指标缺失不能算通过"
    assert criterion.evaluate({"pde_rms_max": float("nan")})[0] is False

    lower = StageCriterion("coverage", "min", fallback=0.9)
    assert lower.evaluate({"coverage": 0.95})[0] is True
    assert lower.evaluate({"coverage": 0.5})[0] is False


def test_a_criterion_needs_a_limit():
    with pytest.raises(ScheduleError, match="needs a tolerance_key or a fallback"):
        StageCriterion("x", "max")


def test_the_schedule_report_lists_the_window_and_the_criteria():
    """运行清单里必须能逐阶段核对这些东西，否则事后无从复现。"""
    rows = describe_schedule()
    assert [row["time_window_fraction"] for row in rows] == [
        stage.time_window_fraction for stage in SCHEDULE
    ]
    for row in rows:
        assert row["criteria"], f"阶段 {row['name']!r} 的清单里没有判据"


def test_the_trainer_refuses_to_advance_a_stage_whose_criteria_fail(tmp_path):
    """预算耗尽而判据未通过 ⇒ 明确停止，不静默跳阶段（§14）。"""
    import test_review_regressions as base

    from spherical_blast_pinn.training.trainer import StageNotPassed, TrainingError

    tiny = (
        Stage("initialise", 2, 0.1, 0.1, 1.0e-3, 1.0e-3, time_window_fraction=0.25),
        Stage("strengthen_pde", 2, 0.1, 1.0, 1.0e-3, 1.0e-4, time_window_fraction=0.65),
        Stage("joint", 2, 1.0, 1.0, 1.0e-4, 1.0e-5, time_window_fraction=1.0),
    )
    trainer = base._tiny_trainer(tmp_path / "w", tmp_path / "out")
    trainer.schedule = tiny
    # 把初值判据的限额压到 0：随机初始化的网络不可能让初值误差恰好为零，于是
    # 这条判据**必然**不通过，测试不依赖"大概会失败"。
    trainer.tolerances = {
        "initial_relative_error": 0.0,
        "initial_front_relative_error": 0.0,
    }

    with pytest.raises(StageNotPassed) as error:
        trainer.run(verbose=False)

    message = str(error.value)
    assert "initialise" in message
    assert error.value.stage == "initialise"
    assert error.value.failures, "必须逐条列出未通过的判据"
    # 停止发生在**进入下一阶段之前**，而不是先跳过去再报告
    assert "joint" not in [row.get("stage") for row in trainer.stage_history if row.get("event") == "enter"]
    assert trainer.step == tiny[0].updates, (
        "停止位置应当是第一阶段预算耗尽的那一步"
    )


def test_a_small_budget_run_traverses_all_three_stages(tmp_path):
    """小预算的阶段机验证（§14）：三个阶段的窗口、采样与平衡都真的动起来。

    这条**不**评价阶段判据——判据是物理闸门，未训练的模型本来就不该通过（那由
    上一条测试守着）。这里验证的是机器本身：窗口逐阶段扩大、每一步的采样区间跟着
    窗口走、有界梯度平衡在预热之后真的产出了权重。

    判据用 ``previous=None`` 跳过，是因为这条测试问的是"机器会不会走"，不是
    "该不该走"。把判据放宽来让它走，就等于把闸门拆了。
    """
    import test_review_regressions as base

    from spherical_blast_pinn.training.schedule import SCHEDULE as DECLARED

    tiny = tuple(
        Stage(
            stage.name, 3,
            stage.pde_weight_start, stage.pde_weight_end,
            stage.learning_rate_start, stage.learning_rate_end,
            time_window_fraction=stage.time_window_fraction,
        )
        for stage in DECLARED
    )
    trainer = base._tiny_trainer(tmp_path / "w", tmp_path / "out")
    trainer.schedule = tiny
    # 让平衡器在前两步就能介入，不必等 200 步预热。
    trainer.balance_config = BalanceConfig(every=1, warmup_steps=0)
    trainer.balancer.config = trainer.balance_config

    seen_windows = []
    seen_batch_windows = []
    for stage in tiny:
        trainer._enter_stage(stage, previous=None, verbose=False)
        seen_windows.append(trainer.sampler.samplers[0].window_fraction)
        for _ in range(stage.updates):
            trainer._update()
            trainer.step += 1
        # 单独采一次，核对 batch 自己报出来的时间上端就是本阶段窗口
        batch = trainer.sampler.samplers[0].batch()
        seen_batch_windows.append(batch.pinn_time_max)

    fractions = [stage.time_window_fraction for stage in tiny]
    assert seen_windows == pytest.approx(fractions)
    assert seen_windows == sorted(seen_windows), "窗口必须逐阶段扩大"

    sampler = trainer.sampler.samplers[0]
    full = sampler.full_pinn_time_max
    for fraction, reported in zip(fractions, seen_batch_windows):
        assert reported == pytest.approx(fraction * full), (
            "batch 报出的时间上端与本阶段窗口不一致；采样器与损失对时间的理解分家了"
        )

    # 平衡器确实介入过，而且留下了可审查的输出
    assert trainer.balancer.updates > 0
    monitors = trainer.balancer.monitors()
    assert monitors["adaptive_weights"], "平衡器没有产出任何权重"
    assert "pde" not in monitors["adaptive_weights"], "PDE 权重不该由平衡器接管"
    assert trainer.last_balance is not None
    assert trainer.last_balance["weights"], "最近一次平衡记录是空的"


def test_entering_a_stage_sets_the_window_on_every_radius(tmp_path):
    """§14.2："同一套两侧网络及前沿同步扩展"——窗口要一起动。"""
    import test_review_regressions as base

    trainer = base._tiny_trainer(tmp_path / "w", tmp_path / "out")
    trainer._enter_stage(SCHEDULE[0], previous=None, verbose=False)
    assert trainer.sampler.samplers[0].window_fraction == pytest.approx(
        SCHEDULE[0].time_window_fraction
    )
    trainer._enter_stage(SCHEDULE[1], previous=None, verbose=False)
    assert trainer.sampler.samplers[0].window_fraction == pytest.approx(
        SCHEDULE[1].time_window_fraction
    )
    entered = [row for row in trainer.stage_history if row["event"] == "enter"]
    assert [row["time_window_fraction"] for row in entered] == [
        SCHEDULE[0].time_window_fraction,
        SCHEDULE[1].time_window_fraction,
    ]
    assert all(row["criteria"] for row in entered), "进场记录里要带上本阶段的判据"


# ==========================================================================
# 有界梯度平衡（§14）
# ==========================================================================


def _balancer(**overrides) -> BoundedGradientBalancer:
    config = BalanceConfig(every=1, warmup_steps=0, **overrides)
    return BoundedGradientBalancer(BASE_WEIGHTS, config)


def test_the_balancer_never_takes_over_the_pde_weight():
    """``lambda_PDE`` 由 §13 日程给。两个机制写同一个数，结果取决于更新顺序。"""
    balancer = _balancer()
    assert "pde" not in balancer.adaptive
    weights = balancer.weights()
    assert "pde" in weights, "仍然要有这个键，只是不参与自适应"


def test_the_balancer_does_not_divide_by_a_negligible_gradient():
    """近零梯度不得变成巨大权重（§14）。

    除以 1e-13 会得到 1e13 量级的目标值，即使随后夹到上限，也等于把一个**没有
    信息**的项长期钉在上限上——而报表里只会看到一个合法的权重数字。
    """
    balancer = _balancer()
    balancer.update("0", {})  # 占位，确保第一次 update 走的是 step 参数
    record = balancer.update(
        0,
        {"data": 1.0e-3, "geometry": 1.0e-13, "admissibility": 1.0e-3},
    )
    assert record.held.get("geometry") == "negligible_gradient"
    assert record.weights["geometry"] == pytest.approx(BASE_WEIGHTS["geometry"]), (
        "证据不足的组必须保持原权重，不能被推到上限"
    )
    assert balancer.bound_hits["geometry"]["upper"] == 0


def test_the_balancer_holds_a_group_with_no_samples_and_does_not_fake_a_zero():
    """缺标签/零样本的组：不参与，且**不更新**它的统计（§14）。"""
    balancer = _balancer()
    balancer.update(0, {"data": 1.0, "shock": 1.0, "geometry": 1.0e-2})
    ema_before = dict(balancer.ema)

    # 这一轮 data 的**相对**范数变了（1.0 → 0.5，参考值由 shock 顶着），所以它
    # 应当被更新；geometry 记为不存在，统计必须原样不动。
    record = balancer.update(
        1,
        {"data": 0.5, "shock": 1.0, "geometry": 0.0},
        live={"data": True, "shock": True, "geometry": False},
    )
    assert record.held.get("geometry") == "no_samples"
    assert balancer.ema["geometry"] == pytest.approx(ema_before["geometry"]), (
        "零样本组把 0 当成一次真实测量写进了移动平均"
    )
    assert balancer.ema["data"] != ema_before["data"], (
        "有样本的组没有更新，说明整轮都被跳过了"
    )


def test_the_balancer_clips_to_its_bounds_and_records_which_bound():
    """触到上下限要能被看见：长期贴在上限的项等于被钉成了"最大权重"。"""
    # 参考值 1.0（data），geometry 范数 5e-4 ⇒ 目标权重 2000 > 上限 1e3
    aggressive = _balancer(min_weight=1.0e-3, max_weight=1.0e3)
    record = aggressive.update(0, {"data": 1.0, "geometry": 5.0e-4})
    assert record.at_bound.get("geometry") == "upper"
    assert record.weights["geometry"] == pytest.approx(1.0e3)
    assert aggressive.bound_hits["geometry"]["upper"] == 1

    # 目标权重是"最大梯度范数 / 本组范数"，恒 **>= 1**，所以下限只有在配置把它
    # 抬到 1 以上时才会真正起作用。这里就用那种配置验证下限分支是活的——否则
    # 它是一段永远不会执行的代码，而"监测权重上下限"就成了空话。
    floor_binding = _balancer(min_weight=5.0, max_weight=1.0e3)
    record = floor_binding.update(0, {"data": 1.0e6, "geometry": 1.0e6})
    # 两组范数相同 ⇒ 目标权重都是 1.0 < 下限 5.0
    assert record.at_bound.get("data") == "lower"
    assert record.weights["data"] == pytest.approx(5.0)
    assert floor_binding.bound_hits["data"]["lower"] == 1


def test_the_balancer_monitors_unweighted_residuals():
    """§14 要求监测"未加权残差"，不是只看加权后的总损失。"""
    balancer = _balancer()
    record = balancer.update(
        0,
        {"data": 1.0, "geometry": 0.5},
        unweighted={"data": 0.25, "geometry": 3.0},
    )
    assert record.unweighted == {"data": 0.25, "geometry": 3.0}
    monitors = balancer.monitors()
    assert monitors["last_unweighted_residuals"]["geometry"] == pytest.approx(3.0)
    assert "adaptive_weights" in monitors


def test_the_balancer_state_round_trips():
    """检查点恢复必须带上自适应统计，否则"从第 N 步继续"不是同一次训练。"""
    balancer = _balancer()
    balancer.update(0, {"data": 1.0, "geometry": 1.0e-2})
    state = balancer.state_dict()

    restored = _balancer()
    restored.load_state_dict(state)
    assert restored.weights() == balancer.weights()
    assert restored.ema == balancer.ema
    assert restored.bound_hits == balancer.bound_hits


def test_gradient_measurement_skips_terms_without_samples():
    """没有样本的组返回的是不带梯度的常数——那不是一个测量结果。"""
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    live_term = (parameter**2)
    dead_term = torch.zeros(())
    assert live_groups({"live": live_term, "dead": dead_term}) == {
        "live": True, "dead": False,
    }


def test_measuring_gradients_does_not_pollute_the_backward_pass():
    """量梯度用 ``autograd.grad``，它不写 ``.grad``。

    若改用 ``backward`` 累加进 ``.grad``，随后真正的反传会拿到一个"测量 + 目标"
    的混合梯度——数值看起来正常，训练却悄悄错了。
    """
    from spherical_blast_pinn.training.balancing import measure_gradient_norms

    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    group = (parameter**3).sum()
    total = group + (parameter**2).sum()

    norms = measure_gradient_norms({"group": group}, [parameter], retain_graph=True)
    assert norms["group"] == pytest.approx(12.0)  # d(p^3)/dp = 3p^2 = 12
    assert parameter.grad is None, "测量污染了 .grad"

    total.backward()
    # d(total)/dp = 3p^2 + 2p = 12 + 4 = 16，且只有这一次反传
    assert float(parameter.grad) == pytest.approx(16.0)
