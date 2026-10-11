"""有界的**合成**联检（修改提示词 §10.2）。

它走的是一整条真实控制流：阶段边界 → 保存 → 加载 → 继续执行 → 检查 → 精修 →
落盘 → 对照，用的却是一个通用的合成算例与一个小网络，**不碰爆炸物理、不碰正式
数据、不碰正式入口的闸门**。

预算写死在这里，不读环境也不读配置：

* 累计 Adam 更新 ≤ 18；
* L-BFGS ≤ 5 次请求迭代，并记录**实际 closure 次数**；
* 固定 CPU、固定种子、小网络、小样本。**不会**自动切到正式训练配置。

每一项检查**单独记录**结果、实际更新数、closure 数、耗时与输出路径。
一个总的 ``ok = true`` **不能**替代逐项结果——那正是这份脚本存在的理由。

用法::

    PYTHONPATH=src python scripts/bounded_snake_check.py [输出目录]

退出码 0 = 全部检查通过；非 0 = 至少一项不成立。
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

#: 预算上限。写死的常量，不从环境读——它们是一份声明，不是旋钮。
MAX_UPDATES = 18
MAX_REFINE_ITERATIONS = 5

#: 这份脚本的版本。改了检查项就升它，否则"上次联检通过"指的不是同一件事。
SNAKE_CHECK_VERSION = 4


# --------------------------------------------------------------------------
# 记录
# --------------------------------------------------------------------------


class Recorder:
    """逐项记录。每一项都有名字、结论与它自己的证据。"""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def check(self, name: str, ok: bool, **detail) -> bool:
        self.entries.append({"check": name, "ok": bool(ok), **detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(entry["ok"] for entry in self.entries)


# --------------------------------------------------------------------------
# 合成算例与替身
# --------------------------------------------------------------------------


def _synthetic_case(directory: Path):
    """一个通用的小算例：两介质、单调时间、解析剖面。**不是爆炸仿真。**"""
    from spherical_blast_pinn.data.extracted_case import load_extracted_case
    from spherical_blast_pinn.physics import material as mat

    directory.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.002, 0.30, 300)
    times = np.linspace(1.0e-5, 3.0e-4, 12)
    density = np.empty((times.size, grid.size))
    pressure = np.empty_like(density)
    velocity = np.zeros_like(density)
    #: 剖面的**解析**前沿。启发式检测器在这么粗的合成网格上给出的 `R_c` 会
    #: 落到 `R_s` 之外（前沿顺序不合法），对照因此判"区域不可用"——那是**对的**，
    #: 但对这条联检来说噪声太大。所以把构造时用的真值一起带出来，供需要一份
    #: 合法预测的那一步使用。
    analytic_contact = np.empty(times.size)
    analytic_shock = np.empty(times.size)
    for index in range(times.size):
        contact = 0.05 * (1.0 + 0.02 * index)
        shock = contact * 2.0
        analytic_contact[index] = contact
        analytic_shock[index] = shock
        inside = grid < contact
        air = (grid >= contact) & (grid < shock)
        density[index] = np.where(
            inside, 1600.0, np.where(air, 6.0 * mat.AIR_DENSITY, mat.AIR_DENSITY)
        )
        pressure[index] = np.where(
            inside, 1.0e10, np.where(air, 25.0 * mat.AIR_PRESSURE, mat.AIR_PRESSURE)
        )
    np.savez(
        directory / "raw_line.npz",
        t=times, x=grid, rho=density, u=velocity, P=pressure,
    )
    (directory / "metadata.json").write_text(
        json.dumps({
            "R_0": 0.05,
            "x_end": float(grid[-1]),
            "detonation_velocity_m_s": 6930.0,
            "symmetry_factor": 8.0,
            "symmetry_share": 0.125,
        }),
        encoding="utf-8",
    )
    case = load_extracted_case(directory, case_id="synthetic")
    return case, analytic_contact, analytic_shock


class _Reference:
    """一个解析的参考场替身：只提供采样器真正用到的两个方法。

    **它是测试替身，只在这条脚本里存在**，不进任何正式入口。
    """

    def scale_to(self, charge_radius):
        """采样器会对参考场做一次缩放视图；替身里它是恒等映射。"""
        return self

    def as_dict(self):
        return {"note": "analytic stand-in used only by scripts/bounded_snake_check.py"}

    def contact_and_shock_at(self, time):
        return np.asarray([0.05]), np.asarray([0.10])

    def contact_pressure_at(self, time):
        """接触面压力尺度。一个正的常数就够——这条联检不看它的数值。"""
        return 6.0e9

    def sample(self, radius, time):
        radius = np.asarray(radius, dtype=np.float64)
        from spherical_blast_pinn.physics import material as mat

        return (
            np.where(radius < 0.05, 1600.0, 6.0 * mat.AIR_DENSITY),
            np.zeros_like(radius),
            np.where(radius < 0.05, 1.0e7, 2.0e5),
        )


def _trainer(case, output: Path):
    from spherical_blast_pinn.data import compatibility as compat
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet
    from spherical_blast_pinn.physics.similitude import SimilitudeScales
    from spherical_blast_pinn.sampling.sampler import SamplingConfig
    from spherical_blast_pinn.training.schedule import SCHEDULE, Stage
    from spherical_blast_pinn.training.trainer import Trainer

    radius = float(case.charge_radius)
    scales = SimilitudeScales.for_radius(
        radius, time_scale=2.0, detonation_velocity=case.detonation_velocity
    )
    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.1, seed=2),
        scales=scales,
    )

    # 三阶段各 2 步：预算 6 步就走完整个日程，阶段边界一定被踩到。
    tiny = tuple(
        Stage(
            stage.name, 2,
            stage.pde_weight_start, stage.pde_weight_end,
            stage.learning_rate_start, stage.learning_rate_end,
            time_window_fraction=stage.time_window_fraction,
        )
        for stage in SCHEDULE
    )
    assert sum(stage.updates for stage in tiny) <= MAX_UPDATES

    class _Config:
        def section(self, name):
            if name == "sampling":
                return {
                    "n_pde_product": 8, "n_pde_air": 8, "n_ic_product": 4,
                    "n_ic_air": 4, "n_contact": 2, "n_shock": 2,
                    "n_data_product": 8, "n_data_air": 8, "n_data_front": 4,
                }
            return {}

        def get(self, key, default=None):
            return default

        def path_of(self, key, default=None):
            return default

    trainer = Trainer(
        config=_Config(),
        cases=[case],
        reference=_Reference(),
        model=model,
        scales_by_radius={radius: scales},
        time_scale=2.0,
        schedule=tiny,
        sampling=SamplingConfig(
            n_pde_product=8, n_pde_air=8, n_ic_product=4, n_ic_air=4,
            n_contact=2, n_shock=2, n_data_product=8, n_data_air=8, n_data_front=4,
        ),
        seed=3,
        output_dir=output,
        log_every=1,
        checkpoint_every=1,
        check_every=1,
        compatibility={case.case_id: compat.CaseCompatibility(
            case.case_id, status=compat.INCOMPATIBLE
        )},
        allow_unverified_data=True,
    )
    return trainer, tiny


# --------------------------------------------------------------------------
# 逐项检查
# --------------------------------------------------------------------------


def _check_full_schedule(rec, case, out_dir):
    """① 连续跑满缩短后的整条日程，并真正跨过每一个阶段边界。"""
    trainer, tiny = _trainer(case, out_dir / "continuous")
    total = sum(stage.updates for stage in tiny)
    started = time.time()
    result = trainer.run(max_updates=total, verbose=False)
    stages = [row["stage"] for row in trainer.stage_history if row["event"] == "enter"]
    checkpoint = trainer.output_dir / "last.pt"

    rec.check(
        "continuous_run_reaches_the_end_of_the_schedule",
        result.updates == total and result.status == "completed",
        updates=result.updates, status=result.status, elapsed_s=round(time.time() - started, 2),
        checkpoint=str(checkpoint), checkpoint_exists=checkpoint.exists(),
    )
    rec.check(
        "every_shortened_stage_boundary_is_crossed",
        stages == [stage.name for stage in tiny],
        stages_entered=stages, expected=[stage.name for stage in tiny],
    )
    return trainer, tiny, total, result


def _check_resume(rec, case, out_dir, continuous, tiny, total):
    """② 断在**阶段边界**上，保存独立检查点，换实例恢复后继续。

    阶段边界是默认配置下必然踩到的点（默认每 500 步保存一次，而 2000/5000 恰好
    就是两个边界），所以这里专门挑它。
    """
    interrupted, _ = _trainer(case, out_dir / "interrupted")
    interrupted.run(max_updates=tiny[0].updates, verbose=False)
    boundary = interrupted.output_dir / "boundary.pt"
    interrupted._save(boundary)

    resumed, _ = _trainer(case, out_dir / "resumed")
    resumed.load(boundary)
    checkpoint_step = int(resumed.step)
    resumed.run(max_updates=total, verbose=False)

    identical = all(
        bool((value == resumed.model.state_dict()[name]).all())
        for name, value in continuous.model.state_dict().items()
    )
    rec.check(
        "resume_at_a_stage_boundary_matches_a_continuous_run",
        identical
        and resumed.sampler.rng_state() == continuous.sampler.rng_state()
        and resumed.step == continuous.step,
        resumed_at_step=checkpoint_step,
        weights_bitwise_identical=identical,
        sampler_rng_identical=resumed.sampler.rng_state() == continuous.sampler.rng_state(),
        window=resumed._stage_window,
        independent_checkpoint=str(boundary),
    )
    rec.check(
        "a_resumed_run_reports_the_measured_loss_of_its_last_update",
        bool(np.isfinite(resumed._last_update_loss))
        and resumed._last_update_loss_step == total,
        last_update_loss=float(resumed._last_update_loss),
        last_update_loss_step=int(resumed._last_update_loss_step),
    )


def _check_blocked_stage(rec, case, out_dir):
    """③ 判据未过的阶段恢复之后**仍然**被闸门拦下。"""
    from spherical_blast_pinn.training import trainer as module
    from spherical_blast_pinn.training.schedule import StageCriterion

    trainer, tiny = _trainer(case, out_dir / "blocked")
    # 判据的替换要罩住**两个**阶段：创建阻塞状态、以及恢复之后的重新评价。
    # 只在创建时替换，恢复那一步用的是空判据集合——那样一跑就过，"仍然被拦下"
    # 测的其实是"判据被我撤掉了"，不是"闸门还在"。
    original = module.criteria_for
    module.criteria_for = lambda name: (
        StageCriterion("pde_rms_max", "max", fallback=-1.0),   # 必然不过
    )
    try:
        try:
            trainer.run(max_updates=tiny[0].updates, verbose=False)
            blocked = False
        except Exception:
            blocked = True
        path = trainer.output_dir / "blocked.pt"
        trainer._save(path)

        fresh, _ = _trainer(case, out_dir / "blocked_resumed")
        fresh.load(path)
        loaded_pending = fresh._pending_stage
        loaded_blocked = fresh._stage_blocked
        try:
            fresh.run(max_updates=sum(stage.updates for stage in tiny), verbose=False)
            still_blocked = False
        except Exception:
            still_blocked = True
    finally:
        module.criteria_for = original

    rec.check(
        "a_failed_stage_stays_blocked_after_a_resume",
        blocked and loaded_blocked and loaded_pending == tiny[0].name and still_blocked,
        stopped_on_failure=blocked,
        blocked_flag_survived_the_load=loaded_blocked,
        pending_stage_recorded=loaded_pending,
        still_blocked_after_resume=still_blocked,
        checkpoint=str(path),
    )


def _check_refinement(rec, case, out_dir, tiny, total):
    """④ 截断精修与完成后的精修：复检指纹、终态与所有产物一致。"""
    trainer, _ = _trainer(case, out_dir / "refine_full")
    trainer.run(max_updates=total, verbose=False)
    checks_before = len(trainer.check_history)
    started = time.time()
    result = trainer.refine(iterations=MAX_REFINE_ITERATIONS, verbose=False)
    record = trainer.refine_history[-1]

    manifest = json.loads(
        (trainer.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    import torch

    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    current = trainer.parameter_fingerprint()

    rec.check(
        "a_completed_refinement_reports_the_same_identity_everywhere",
        result.status == "diagnostic"
        and manifest["status"] == "diagnostic"
        and manifest["diagnostic"] is True
        and manifest["refinement_count"] == 1
        and checkpoint["run_status"] == "diagnostic"
        and len(checkpoint["refinement"]) == 1,
        status=result.status, manifest_status=manifest["status"],
        manifest_diagnostic=manifest["diagnostic"],
        checkpoint_run_status=checkpoint["run_status"],
        checkpoint_refinement_records=len(checkpoint["refinement"]),
    )
    new_records = trainer.check_history[checks_before:]
    full_on_current = [
        record for record in new_records
        if record.get("scope") == "full"
        and record.get("parameter_fingerprint") == current
    ]
    # 要求的是"**存在**一次落在当前参数上的完整检查"，不是"最后一条必须是完整的"：
    # 收尾的阶段判据会在它之后再追加一条窗口检查，那一条本来就该是 windowed。
    rec.check(
        "a_refinement_rechecks_the_parameters_it_changed",
        len(new_records) > 0 and len(full_on_current) == 1,
        checks_before=checks_before, checks_after=len(trainer.check_history),
        full_checks_on_current_parameters=len(full_on_current),
        scopes=[record.get("scope") for record in new_records],
    )
    rec.check(
        "the_refinement_records_what_it_did",
        record["iterations_requested"] == MAX_REFINE_ITERATIONS
        and 0 < record["closures"] <= MAX_REFINE_ITERATIONS * 8
        and record["parameters_changed"] is True
        and record["fingerprint_before"] != record["fingerprint_after"]
        and np.isfinite(record["loss_on_accepted_parameters"])
        and record["resume_supported"] is False,
        iterations_requested=record["iterations_requested"],
        closures=record["closures"], closure_budget=MAX_REFINE_ITERATIONS * 8,
        parameters_changed=record["parameters_changed"],
        loss_on_accepted_parameters=record["loss_on_accepted_parameters"],
        loss_at_last_closure_evaluation=record["loss_at_last_closure_evaluation"],
        elapsed_s=round(time.time() - started, 2),
    )

    # 截断之后精修：同样必须是诊断性的，且复检落在当前参数上。
    truncated, _ = _trainer(case, out_dir / "refine_truncated")
    truncated.run(max_updates=1, verbose=False)
    truncated_result = truncated.refine(iterations=1, verbose=False)
    rec.check(
        "a_refinement_after_a_truncated_run_is_still_diagnostic",
        truncated_result.status == "diagnostic"
        and truncated.check_history[-1]["parameter_fingerprint"]
        == truncated.parameter_fingerprint(),
        status=truncated_result.status,
        updates=int(truncated.step),
    )


def _check_refinement_failure(rec, case, out_dir):
    """⑤ 注入一次精修失败，确认参数与有效检查点受保护。"""
    import torch

    trainer, _ = _trainer(case, out_dir / "refine_failure")
    trainer.run(max_updates=1, verbose=False)
    before = trainer.parameter_fingerprint()
    checkpoint_before = (trainer.output_dir / "last.pt").read_bytes()

    parameter = next(p for p in trainer.model.parameters() if p.requires_grad)
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    raised = None
    try:
        trainer.refine(iterations=1, verbose=False)
    except Exception as exc:  # noqa: BLE001 - 这里就是要看它怎么失败
        raised = exc
    finally:
        handle.remove()

    diagnostics = trainer.output_dir / "refinement_failures.json"
    record = json.loads(diagnostics.read_text(encoding="utf-8"))[-1] if diagnostics.exists() else {}
    rec.check(
        "an_injected_refinement_failure_leaves_the_state_alone",
        raised is not None
        and trainer.parameter_fingerprint() == before
        and (trainer.output_dir / "last.pt").read_bytes() == checkpoint_before
        and record.get("parameters_restored") is True
        and record.get("fingerprint_restored") is True,
        error=f"{type(raised).__name__}: {raised}" if raised else None,
        parameters_restored=record.get("parameters_restored"),
        fingerprint_restored=record.get("fingerprint_restored"),
        checkpoint_untouched=(trainer.output_dir / "last.pt").read_bytes() == checkpoint_before,
        diagnostics_file=str(diagnostics), diagnostics_exists=diagnostics.exists(),
    )


def _check_impulse(rec):
    """⑥a 正相冲量的解析反例（提示词 §8）。"""
    from spherical_blast_pinn.evaluation.waveform import gauge_metrics

    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    rec.check(
        "the_impulse_counterexample_is_2_point_5",
        abs(metrics.positive_impulse - 2.5) < 1e-12
        and abs(metrics.positive_duration - 1.5) < 1e-12,
        impulse=metrics.positive_impulse, expected_impulse=2.5,
        duration=metrics.positive_duration, expected_duration=1.5,
    )


def _check_sample_reasons(rec):
    """⑥b 跨间断与域外的**原因码**。"""
    from spherical_blast_pinn.cli.compare import (
        SAMPLE_CROSS_DISCONTINUITY,
        SAMPLE_RADIUS_OUTSIDE,
        SAMPLE_TIME_OUTSIDE,
        sample_field,
    )

    crossing = {
        "pinn_time": np.array([0.0, 1.0]),
        "physical_time": np.array([0.0, 1.0]),
        "start_time": 0.0,
        "radius": np.array([0.10, 0.11]),
        "density": np.array([[2.0, 2.0], [100.0, 100.0]]),
        "contact_radius": np.array([0.09, 0.12]),
        "shock_radius": np.array([0.20, 0.20]),
    }
    value, why = sample_field(crossing, "density", np.array([0.105]), 0.5)
    rec.check(
        "a_contact_sweeping_between_frames_is_not_interpolated_across",
        why[0] == SAMPLE_CROSS_DISCONTINUITY and not np.isfinite(value[0]),
        reason=why[0], value=float(value[0]),
    )

    smooth = {
        "pinn_time": np.array([0.0, 1.0]),
        "physical_time": np.array([0.0, 1.0]),
        "start_time": 0.0,
        "radius": np.array([0.10, 0.11, 0.12]),
        "density": np.ones((2, 3)),
        "contact_radius": np.full(2, 0.05),
        "shock_radius": np.full(2, 0.30),
    }
    _, outside = sample_field(smooth, "density", np.array([0.5]), 0.5)
    _, late = sample_field(smooth, "density", np.array([0.11]), 99.0)
    rec.check(
        "outside_the_prediction_grid_or_time_coverage_says_why",
        outside[0] == SAMPLE_RADIUS_OUTSIDE and late[0] == SAMPLE_TIME_OUTSIDE,
        radius_reason=outside[0], time_reason=late[0],
    )


def _compare_case(directory: Path):
    """一条**专门给对照**用的合成导出。

    为什么不复用 ``_synthetic_case``：那一份是给训练那条链用的，网格粗、介质带窄，
    启发式前沿检测器在上面给出的 ``R_c`` 会落到 ``R_s`` 之外（实测两者几乎相等），
    于是材料分区退化成一整片"未受扰动"——对照于是**什么也比不了**。那是检测器在
    这份数据上的真实表现，不是 bug，但拿它当对照的输入就测不出任何东西。

    这里走的另外一条路径：写一份 ``material.npz`` 配套文件，材料分区是**读数**
    而不是启发式判据。前沿顺序合法、介质带足够宽、产物与空气各有上百个单元。
    """
    import json as _json

    from spherical_blast_pinn.data.extracted_case import load_extracted_case
    from spherical_blast_pinn.physics import material as mat

    directory.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.004, 0.30, 200)
    times = np.linspace(0.0, 3.0e-4, 8)
    n_t = times.size
    contact = 0.05 * (1.0 + 0.03 * np.arange(n_t))
    shock = contact * 2.0

    density = np.empty((n_t, grid.size))
    pressure = np.empty_like(density)
    velocity = np.zeros_like(density)
    energy = np.empty_like(density)
    material = np.empty((n_t, grid.size), dtype=np.int64)
    for index in range(n_t):
        product = grid < contact[index]
        air = (grid >= contact[index]) & (grid < shock[index])
        ambient = ~(product | air)
        material[index] = np.where(product, 0, np.where(air, 1, 2))
        density[index] = np.where(
            product, 1600.0, np.where(air, 6.0 * mat.AIR_DENSITY, mat.AIR_DENSITY)
        )
        pressure[index] = np.where(
            product, 1.0e10, np.where(air, 25.0 * mat.AIR_PRESSURE, mat.AIR_PRESSURE)
        )
        energy[index] = np.where(
            product, 6.0e6, np.where(air, 2.0e5, mat.AIR_SPECIFIC_ENERGY)
        )
    np.savez(
        directory / "raw_line.npz",
        t=times, x=grid, rho=density, u=velocity, P=pressure, e=energy,
    )
    valid = material != 2
    np.savez(
        directory / "material.npz",
        material=material,
        valid=valid,
        contact_radius=contact,
        shock_radius=shock,
        contact_state=np.full(n_t, 2, dtype=np.int64),
        shock_state=np.full(n_t, 2, dtype=np.int64),
        contact_has_supervision=np.ones(n_t, dtype=bool),
        shock_has_supervision=np.ones(n_t, dtype=bool),
        field_valid_density=valid, field_valid_velocity=valid,
        field_valid_energy=valid, field_valid_pressure=valid,
    )
    (directory / "metadata.json").write_text(
        _json.dumps({
            "R_0": 0.05,
            "x_end": float(grid[-1]),
            "detonation_velocity_m_s": 6930.0,
            "material_partition_is_volume_fraction_based": True,
            "material_partition_source": {
                "method": "ale_volume_fraction",
                "producer": "scripts.bounded_snake_check",
                "producer_version": 1,
                "fields": ["tnt_volume_fraction"],
                "input_identifier": "sha256:bounded-snake-check",
                "detail": {"note": "synthetic companion written by the snake check"},
            },
            "profile_layout": {"version": 1, "layout": "frames_by_radius"},
            "symmetry_factor": 8.0,
            "symmetry_share": 0.125,
        }),
        encoding="utf-8",
    )
    return load_extracted_case(directory, case_id="synthetic-compare"), contact, shock


def _check_compare_cli(rec, out_dir):
    """⑥c 真实 compare CLI：报告要真被写出来，而且**真的比过东西**。"""
    from spherical_blast_pinn.cli import compare as compare_module
    from spherical_blast_pinn.physics import material as mat

    root = out_dir / "compare_cli"
    root.mkdir(parents=True, exist_ok=True)
    case, contact, shock = _compare_case(root / "case")
    radii = np.asarray(case.radius, dtype=np.float64)
    n_t = case.times.size
    region = np.where(
        radii[None, :] < contact[:, None], 0,
        np.where(radii[None, :] < shock[:, None], 1, 2),
    ).astype(np.int64)
    prediction = {
        "start_time": float(case.start_time),
        "pinn_time": np.asarray(case.pinn_times, dtype=np.float64),
        "physical_time": np.asarray(case.times, dtype=np.float64),
        "radius": radii,
        "charge_radius": float(case.charge_radius),
        "region": region,
        "contact_radius": contact,
        "shock_radius": shock,
        "density": np.asarray(case.density, dtype=np.float64),
        "velocity": np.asarray(case.velocity, dtype=np.float64),
        "specific_energy": np.asarray(case.specific_energy, dtype=np.float64),
        "pressure": np.asarray(case.pressure, dtype=np.float64),
        "scale_product_density": 1630.0,
        "scale_air_density": 1.225,
        "scale_velocity": mat.REFERENCE_VELOCITY,
        "scale_specific_energy": mat.PRODUCT_SPECIFIC_ENERGY,
        # 这两项是被**消费**的：环境压定义全部超压，缩放时间覆盖写进报告。
        # ``compare`` 入口把它们当必需键，缺一个就拒绝——所以这里也要给出来。
        "ambient_pressure": float(mat.AIR_PRESSURE),
        "scaled_time_coverage": float(case.times[-1] - case.start_time),
    }
    prediction_path = root / "predict.npz"
    np.savez(prediction_path, **prediction)

    config_path = root / "config.yaml"
    config_path.write_text(
        "\n".join([
            f"inherits: {(REPO / 'configs' / 'base.yaml').as_posix()}",
            "",
            "data:",
            "  cases:",
            f"    - case_id: {case.case_id}",
            "      charge_radius_m: 0.05",
            f"      directory: {Path(case.source['directory']).as_posix()}",
            "      split: train",
        ]) + "\n",
        encoding="utf-8",
    )
    output = root / "out"
    started = time.time()
    code = compare_module.main([
        "--config", str(config_path),
        "--prediction", str(prediction_path),
        "--case", case.case_id,
        "--output", str(output),
    ])
    report_path = output / f"compare_{case.case_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    rec.check(
        "compare_cli_runs_end_to_end_and_writes_a_readable_report",
        code == 0 and report_path.exists() and report.get("case_id") == case.case_id
        and "fields" in report and "gauges" in report
        # 报告要真的**比过东西**：只写出一个"什么都不可评价"的文件不算跑通。
        and report["fields"].get("comparable_points", 0) > 0,
        exit_code=code, report=str(report_path), report_exists=report_path.exists(),
        fields_available=report.get("fields", {}).get("available"),
        comparable_points=report.get("fields", {}).get("comparable_points"),
        elapsed_s=round(time.time() - started, 2),
    )



def _check_refinement_transaction(rec, case, out_dir):
    """⑦ 精修在**任意阶段**失败都不能留下半改状态。"""
    import torch

    from spherical_blast_pinn.training import trainer as module

    trainer, _ = _trainer(case, out_dir / "refine_transaction")
    trainer.run(max_updates=1, verbose=False)

    def snapshot():
        best = trainer.output_dir / "best.pt"
        return {
            "parameters": trainer.parameter_fingerprint(),
            "rng": [s.rng_state() for s in trainer.sampler.samplers],
            "refine_history": len(trainer.refine_history),
            "last_pt": (trainer.output_dir / "last.pt").read_bytes(),
            "best_pt": best.read_bytes() if best.exists() else None,
        }

    before = snapshot()
    original_verify = trainer.verify

    def failing_verify(*_args, **_kwargs):
        raise RuntimeError("injected: the post-refinement re-check fails")

    trainer.verify = failing_verify
    raised = None
    try:
        trainer.refine(iterations=1, verbose=False)
    except Exception as exc:  # noqa: BLE001 - 这里就是要看它怎么失败
        raised = exc
    finally:
        trainer.verify = original_verify

    after = snapshot()
    diagnostics = trainer.output_dir / "refinement_failures.json"
    record = (
        json.loads(diagnostics.read_text(encoding="utf-8"))[-1]
        if diagnostics.exists() else {}
    )
    rec.check(
        "a_refinement_failure_after_the_update_leaves_no_half_changed_state",
        raised is not None
        and after == before
        and record.get("parameters_restored") is True
        and record.get("artifacts_restored") is True,
        error=f"{type(raised).__name__}: {raised}" if raised else None,
        parameters_unchanged=after["parameters"] == before["parameters"],
        sampler_rng_unchanged=after["rng"] == before["rng"],
        refine_history_unchanged=after["refine_history"] == before["refine_history"],
        checkpoints_unchanged=(
            after["last_pt"] == before["last_pt"] and after["best_pt"] == before["best_pt"]
        ),
        diagnostics_stage=record.get("stage"),
    )


def _check_best_and_origin(rec, case, out_dir):
    """⑧ best 的终态与显式诊断身份要跟着运行走。"""
    import torch

    trainer, tiny = _trainer(case, out_dir / "best_origin")
    total = sum(stage.updates for stage in tiny)
    trainer.run(max_updates=total, verbose=False, diagnostic=True)

    checkpoint = torch.load(
        trainer.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    fresh, _ = _trainer(case, out_dir / "best_origin_reloaded")
    fresh.load(trainer.output_dir / "last.pt")
    reloaded = fresh.run(max_updates=fresh.step, verbose=False)

    rec.check(
        "an_explicit_diagnostic_identity_survives_a_reload",
        checkpoint["run_origin"]["explicit_diagnostic"] is True
        and fresh._run_origin["explicit_diagnostic"] is True
        and reloaded.status == "diagnostic"
        and reloaded.formally_accepted is False,
        checkpoint_origin=checkpoint.get("run_origin"),
        status_after_reload=reloaded.status,
        formally_accepted=reloaded.formally_accepted,
    )

    # 让精修后的复检成为新的最佳，检查 best.pt 与 last.pt 的终态一致。
    refined, _ = _trainer(case, out_dir / "best_refined")
    refined.run(max_updates=total, verbose=False)
    refined.best = None
    refined.refine(iterations=1, verbose=False)
    best = torch.load(
        refined.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    last = torch.load(
        refined.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    rec.check(
        "best_pt_carries_the_status_of_the_run_that_selected_it",
        best["run_status"] == "diagnostic"
        and last["run_status"] == "diagnostic"
        and best["run_origin"]["refined"] is True,
        best_run_status=best.get("run_status"),
        last_run_status=last.get("run_status"),
        best_origin=best.get("run_origin"),
    )


def _check_data_identity(rec, case, out_dir):
    """⑨ 数据身份要覆盖真正参与训练的场量。"""
    from spherical_blast_pinn.training.trainer import TrainingError

    trainer, tiny = _trainer(case, out_dir / "data_identity")
    trainer.run(max_updates=sum(stage.updates for stage in tiny), verbose=False)

    def own_copy():
        """**每个变体都要一份自己的算例对象。**

        ``_trainer`` 把传进来的算例直接放进 ``Trainer.cases``（只复制了列表），
        所以多个 trainer 共享同一份 case。就地改一个变体的场量会连带改掉原来
        那一份——"对照组"于是拿一份已经被改过的数据去比，测出来的是"我把自己
        改坏了"，不是数据身份。这里按来源目录重新读一份。
        """
        from spherical_blast_pinn.data.extracted_case import load_extracted_case

        return load_extracted_case(
            Path(case.source["directory"]), case_id=case.case_id
        )

    refused = {}
    for label, mutate in (
        ("density", lambda c: setattr(c, "density", np.asarray(c.density) * 2.0)),
        ("velocity", lambda c: setattr(c, "velocity", np.asarray(c.velocity) + 123.0)),
        # 整体轮转 0→1→2→0：每个单元都变，而且仍在合法取值域里。写成
        # "把空气改成环境"在这份合成算例上是**空操作**（检测器在这里给出的
        # material 全是 2），那测的就不是数据身份了。
        ("material", lambda c: setattr(
            c, "material", (np.asarray(c.material) + 1) % 3
        )),
        ("valid", lambda c: setattr(c, "valid", ~np.asarray(c.valid))),
    ):
        fresh, _ = _trainer(own_copy(), out_dir / f"data_identity_{label}")
        mutate(fresh.cases[0])
        try:
            fresh.load(trainer.output_dir / "last.pt")
            refused[label] = False
        except TrainingError:
            refused[label] = True

    # 对照组：什么都不改时必须能加载（否则上面那几条"拒绝"可能只是碰巧）。
    control, _ = _trainer(own_copy(), out_dir / "data_identity_control")
    try:
        control.load(trainer.output_dir / "last.pt")
        control_ok = True
    except Exception:  # noqa: BLE001
        control_ok = False

    rec.check(
        "a_changed_field_is_refused_by_the_data_identity",
        all(refused.values()) and control_ok,
        refused=refused, unmodified_case_loads=control_ok,
    )


def _check_evaluation_boundaries(rec):
    """⑩ 评估边界：零阈值三角波、零权重邻帧、不可用指标不判分。"""
    from spherical_blast_pinn.cli.compare import (
        SAMPLE_FRONT_UNAVAILABLE,
        SAMPLE_OK,
        _fronts_at,
        sample_field,
    )
    from spherical_blast_pinn.evaluation.waveform import compare_gauge, gauge_metrics

    metrics = gauge_metrics(
        [0, 1, 2, 3], [100, 102, 100, 99], ambient=100, minimum_overpressure_ratio=0.0
    )
    rec.check(
        "a_zero_threshold_triangle_reports_area_2_and_duration_2",
        abs(metrics.positive_impulse - 2.0) < 1e-12
        and abs(metrics.positive_duration - 2.0) < 1e-12,
        impulse=metrics.positive_impulse, duration=metrics.positive_duration,
    )

    prediction = {
        "pinn_time": np.array([0.0, 1.0, 2.0]),
        "physical_time": np.array([0.0, 1.0, 2.0]),
        "start_time": 0.0,
        "radius": np.array([0.10, 0.11, 0.12]),
        "density": np.full((3, 3), 1000.0),
        "contact_radius": np.array([0.05, np.nan, 0.05]),
        "shock_radius": np.full(3, 0.30),
    }
    values, why = sample_field(prediction, "density", np.array([0.10]), 0.0)
    between, why_between = sample_field(prediction, "density", np.array([0.10]), 0.5)
    contact, _ = _fronts_at(prediction, 0.0)
    rec.check(
        "an_exact_frame_ignores_a_missing_neighbour_frame",
        why[0] == SAMPLE_OK and values[0] == 1000.0
        and np.isfinite(contact) and abs(contact - 0.05) < 1e-12
        and why_between[0] == SAMPLE_FRONT_UNAVAILABLE,
        exact_hit_reason=why[0], exact_hit_value=float(values[0]),
        between_frames_reason=why_between[0],
    )

    comparison = compare_gauge(
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        model_pressure=np.array([102.0, 104.0, 103.0, 103.0]),
        reference_pressure=np.array([102.0, 104.0, 99.0, 99.0]),
        ambient=100.0, pressure_scale=100.0, time_scale=3.0, impulse_scale=300.0,
        radius=0.1,
    )
    rec.check(
        "an_incomplete_phase_removes_the_phase_and_arrival_errors",
        "positive_duration_relative" not in comparison.errors
        and "positive_impulse_relative" not in comparison.errors
        and "arrival_time_relative" not in comparison.errors
        and np.isfinite(comparison.model.positive_impulse),
        errors=sorted(comparison.errors),
        model_phase_complete=comparison.model.positive_phase_complete,
        arrival_observed=comparison.model.arrival_observed,
    )


def _check_best_candidate(rec, case, out_dir):
    """⑪ 中途选出的最佳必须连带它**入选那一刻**的参数一起发布。"""
    import hashlib

    import torch

    from spherical_blast_pinn.evaluation.checks import CheckReport

    trainer, tiny = _trainer(case, out_dir / "best_candidate")
    total = sum(stage.updates for stage in tiny)

    def fingerprint(state):
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
        return digest.hexdigest()

    per_step: dict[int, str] = {}
    original_update = trainer._update

    def recording_update():
        breakdown = original_update()
        per_step[int(trainer.step) + 1] = trainer.parameter_fingerprint()
        return breakdown

    trainer._update = recording_update
    original_checks = module_run_checks()

    def score_at_first(model, check_set, **kwargs):
        recorded = int(kwargs.get("steps", 0))
        value = 0.0 if recorded == 1 else float(recorded)
        return CheckReport(step=recorded, metrics={
            "nonphysical_violation_count": 0.0,
            "physics_violation_score": value,
            "development_prediction_error": value,
            "total_loss": value,
        })

    set_module_run_checks(score_at_first)
    try:
        trainer.run(max_updates=total, verbose=False)
    finally:
        set_module_run_checks(original_checks)

    best = torch.load(trainer.output_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(trainer.output_dir / "last.pt", map_location="cpu", weights_only=False)
    best_actual = fingerprint(best["model"])
    last_actual = fingerprint(last["model"])

    # 中途最佳必须能从 best.pt 继续，而且与那一刻独立存的检查点一致。
    at_candidate = out_dir / "best_candidate" / "independent_at_candidate.pt"
    resumed_from_best, _ = _trainer(case, out_dir / "best_resumed")
    resumed_from_best.load(trainer.output_dir / "best.pt")

    rec.check(
        "a_mid_run_best_is_published_with_its_own_parameters",
        trainer.best["step"] == 1
        and best["step"] == 1
        and best_actual == per_step.get(1)
        and best_actual == trainer.best["parameter_fingerprint"]
        and last["step"] == total
        and last_actual == per_step.get(total)
        and best_actual != last_actual,
        best_check_step=trainer.best["step"],
        best_pt_step=best["step"],
        best_pt_matches_step_1=best_actual == per_step.get(1),
        fingerprint_matches_file=best_actual == trainer.best["parameter_fingerprint"],
        last_pt_step=last["step"],
        best_differs_from_last=best_actual != last_actual,
        candidate_available=at_candidate.exists(),
    )
    rec.check(
        "best_pt_can_be_resumed",
        resumed_from_best.step == best["step"],
        resumed_step=resumed_from_best.step,
        checkpoint_step=best["step"],
    )


def _check_late_publish_failure(rec, case, out_dir):
    """⑫ 前几份产物已经真的写出之后才失败：整套都要回滚。"""
    import pathlib as _pathlib
    from pathlib import Path as _Path

    trainer, tiny = _trainer(case, out_dir / "late_failure")
    trainer.run(max_updates=1, verbose=False)
    names = ("last.pt", "best.pt", "checks.json", "run_manifest.json")
    before = {
        name: (trainer.output_dir / name).read_bytes()
        if (trainer.output_dir / name).exists() else None
        for name in names
    }
    before_state = trainer._final_check_state
    before_fingerprint = trainer.parameter_fingerprint()

    real_write = _pathlib.Path.write_text
    target = str((trainer.output_dir / "run_manifest.json").resolve())

    def guarded(self, *args, **kwargs):
        if str(_Path(self).resolve()) == target:
            raise OSError("injected: run_manifest.json cannot be written")
        return real_write(self, *args, **kwargs)

    _pathlib.Path.write_text = guarded
    raised = None
    try:
        trainer.refine(iterations=1, verbose=False)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    finally:
        _pathlib.Path.write_text = real_write

    after = {
        name: (trainer.output_dir / name).read_bytes()
        if (trainer.output_dir / name).exists() else None
        for name in names
    }
    diagnostics = trainer.output_dir / "refinement_failures.json"
    record = (
        json.loads(diagnostics.read_text(encoding="utf-8"))[-1]
        if diagnostics.exists() else {}
    )
    rec.check(
        "a_failure_after_earlier_artifacts_were_published_rolls_all_of_them_back",
        raised is not None
        and all(after[name] == before[name] for name in names)
        and trainer._final_check_state == before_state
        and trainer.parameter_fingerprint() == before_fingerprint
        and record.get("artifacts_restored") is True,
        error=f"{type(raised).__name__}" if raised else None,
        artifacts_unchanged=[name for name in names if after[name] == before[name]],
        check_state_restored=trainer._final_check_state == before_state,
        artifact_rollback_keys=sorted(record.get("artifact_rollback") or {}),
        marker_removed=not (trainer.output_dir / ".publish_in_progress.json").exists(),
    )


def _check_independent_restore(rec, case, out_dir):
    """⑬ 内存恢复与磁盘恢复各自独立尝试；不可恢复的产物不能被加载。"""
    import torch

    from spherical_blast_pinn.training.trainer import TrainingError

    # ① 内存恢复失败，磁盘恢复仍要执行。
    trainer, _ = _trainer(case, out_dir / "restore_memory")
    trainer.run(max_updates=1, verbose=False)
    good_last = (trainer.output_dir / "last.pt").read_bytes()

    real_write = pathlib.Path.write_text

    def guard_manifest(owner):
        """让**这个** trainer 的 ``run_manifest.json`` 写出失败。

        每个 trainer 有各自的输出目录；用同一个闭包会让第二段注入打空炮，
        于是第二段根本没有触发失败，测出来的是"没失败所以没拒绝"。
        """
        wanted = str((owner.output_dir / "run_manifest.json").resolve())

        def guarded(self, *args, **kwargs):
            if str(pathlib.Path(self).resolve()) == wanted:
                raise OSError("injected: run_manifest.json cannot be written")
            return real_write(self, *args, **kwargs)

        return guarded

    real_restore = trainer._restore_state

    def broken_restore(_snapshot):
        raise RuntimeError("injected: the in-memory restore fails")

    trainer._restore_state = broken_restore
    pathlib.Path.write_text = guard_manifest(trainer)
    raised = None
    try:
        trainer.refine(iterations=1, verbose=False)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    finally:
        trainer._restore_state = real_restore
        pathlib.Path.write_text = real_write

    disk_restored = (trainer.output_dir / "last.pt").read_bytes() == good_last
    loaded = torch.load(trainer.output_dir / "last.pt", map_location="cpu", weights_only=False)
    rec.check(
        "a_memory_restore_failure_still_restores_the_disk",
        raised is not None
        and trainer._unusable is not None
        and disk_restored
        and len(loaded.get("refinement") or []) == 0,
        error=f"{type(raised).__name__}" if raised else None,
        instance_marked_unusable=trainer._unusable is not None,
        disk_restored_to_previous_bytes=disk_restored,
        refinement_records_on_disk=len(loaded.get("refinement") or []),
    )

    # ② 磁盘无法恢复：未提交产物必须被拒绝加载。
    trainer2, _ = _trainer(case, out_dir / "restore_unrecoverable")
    trainer2.run(max_updates=1, verbose=False)
    real_restore_artifacts = trainer2._restore_artifacts

    def failing_artifacts(captured):
        results = real_restore_artifacts(captured)
        results["last.pt"] = "FAILED: injected: cannot restore last.pt"
        return results

    trainer2._restore_artifacts = failing_artifacts
    pathlib.Path.write_text = guard_manifest(trainer2)
    try:
        trainer2.refine(iterations=1, verbose=False)
    except Exception:  # noqa: BLE001
        pass
    finally:
        trainer2._restore_artifacts = real_restore_artifacts
        pathlib.Path.write_text = real_write

    fresh, _ = _trainer(case, out_dir / "restore_probe")
    try:
        fresh.load(trainer2.output_dir / "last.pt")
        refused = False
    except TrainingError:
        refused = True
    rec.check(
        "an_unrestorable_publish_cannot_be_loaded",
        refused and (trainer2.output_dir / ".publish_in_progress.json").exists(),
        refused_by_a_new_instance=refused,
        marker_present=(trainer2.output_dir / ".publish_in_progress.json").exists(),
    )


def _check_phase_coverage(rec):
    """⑭ 左端缺失的时程不能判为完整正相。"""
    from spherical_blast_pinn.evaluation.waveform import compare_gauge, gauge_metrics

    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    full = np.array([100.0, 102.0, 104.0, 101.0, 99.0, 99.0, 99.0])
    left_missing = np.array([103.0, 104.0, 103.0, 101.0, 99.0, 99.0, 99.0])

    metrics = gauge_metrics([0, 1, 2, 3], [102, 102, 98, 98], ambient=100)
    comparison = compare_gauge(
        times=times, model_pressure=left_missing, reference_pressure=full,
        ambient=100.0, pressure_scale=100.0, time_scale=3.0, impulse_scale=300.0,
        radius=0.1,
    )
    rec.check(
        "a_missing_left_edge_is_not_a_complete_positive_phase",
        metrics.arrival_observed is False
        and metrics.positive_phase_complete is False
        and metrics.positive_phase_coverage == "start_missing"
        and abs(metrics.positive_impulse - 2.5) < 1e-12
        and comparison.model.positive_phase_coverage == "start_missing"
        and "positive_impulse_relative" not in comparison.errors,
        area_kept=metrics.positive_impulse,
        coverage=metrics.positive_phase_coverage,
        model_coverage=comparison.model.positive_phase_coverage,
        judged=sorted(comparison.errors),
    )


def module_run_checks():
    from spherical_blast_pinn.training import trainer as module

    return module.run_checks


def set_module_run_checks(function):
    from spherical_blast_pinn.training import trainer as module

    module.run_checks = function

# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    from spherical_blast_pinn.training import trainer as trainer_module

    out_dir = Path(argv[1]) if len(argv) > 1 else Path(tempfile.mkdtemp(prefix="snake_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder()
    started = time.time()

    # 判据换成空集：这条联检问的是**控制流**，不是物理闸门；闸门本身由
    # tests/test_stage_machine.py、tests/test_refine_and_checkpoint.py 与
    # tests/test_run_status_and_guards.py 守着。换掉之后**不可能**证明任何物理
    # 判据通过，正式入口也不借用它。
    original_criteria = trainer_module.criteria_for
    trainer_module.criteria_for = lambda name: ()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            case, _analytic_contact, _analytic_shock = _synthetic_case(Path(tmp) / "case")
            continuous, tiny, total, _ = _check_full_schedule(recorder, case, out_dir)
            _check_resume(recorder, case, out_dir, continuous, tiny, total)
            _check_blocked_stage(recorder, case, out_dir)
            _check_refinement(recorder, case, out_dir, tiny, total)
            _check_refinement_failure(recorder, case, out_dir)
            _check_impulse(recorder)
            _check_sample_reasons(recorder)
            _check_compare_cli(recorder, out_dir)
            _check_refinement_transaction(recorder, case, out_dir)
            _check_best_and_origin(recorder, case, out_dir)
            _check_data_identity(recorder, case, out_dir)
            _check_evaluation_boundaries(recorder)
            _check_best_candidate(recorder, case, out_dir)
            _check_late_publish_failure(recorder, case, out_dir)
            _check_independent_restore(recorder, case, out_dir)
            _check_phase_coverage(recorder)
    finally:
        trainer_module.criteria_for = original_criteria

    report = {
        "script": "scripts/bounded_snake_check.py",
        "version": SNAKE_CHECK_VERSION,
        "output_dir": str(out_dir),
        "budget": {
            "max_adam_updates": MAX_UPDATES,
            "max_lbfgs_iterations_requested": MAX_REFINE_ITERATIONS,
            "note": (
                "written here as constants; this script never switches to the formal "
                "training configuration"
            ),
        },
        "device": "cpu",
        "duration_s": round(time.time() - started, 2),
        "checks": recorder.entries,
        "ok": recorder.ok,
    }
    sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "snake_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if recorder.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
