"""训练循环。

三条规范要求的性质直接体现在循环结构里：

* **每次参数更新，三个半径都参与且等权**。每一步都重新为三个半径各采一个
  batch，算出一个联合损失，做**一次**优化器更新。不是"顺序做三次更新"——
  那三次更新会互相覆盖，与一次等权联合更新不是同一个算法。
* **Adam 每次更新重采训练点**。采样器每步产出新的点，没有固定训练集。
  用于诊断的物理检查点另行固定，不参与梯度。
* **诊断不改变目标函数**。质量与能量平衡只记录，不进梯度。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..data import compatibility as compat
from ..data.extracted_case import ExtractedCase
from ..evaluation.checks import CheckReport, CheckSet, build_check_set, run_checks
from ..losses.objective import LossBreakdown, contact_pressure_scale, total_loss
from ..models.coupled_model import CoupledModel
from ..physics.precursor import ReferenceField
from ..physics.similitude import SimilitudeScales
from ..sampling.sampler import GroupedSampler, SamplingConfig
from .balancing import (
    BalanceConfig,
    BoundedGradientBalancer,
    live_groups,
    measure_gradient_norms,
)
from .schedule import (
    MAX_LBFGS_ITERATIONS,
    SCHEDULE,
    Stage,
    criteria_for,
    describe_schedule,
    learning_rate_at,
    pde_weight_at,
    time_window_fraction_at,
)

#: 检查点格式标识与版本。加载时两者都必须匹配，否则拒绝。
CHECKPOINT_FORMAT = "spherical_blast_pinn.checkpoint"
#: 第 3 版起检查点必须带**采样器自己的** RNG 状态（第 2 版只存全局 NumPy 状态，
#: 恢复出来的不是一个连续的采样轨迹），并且带固定检查集与兼容性闸门的结论。
CHECKPOINT_VERSION = 3


class TrainingError(RuntimeError):
    """训练无法继续。"""


class StageNotPassed(TrainingError):
    """阶段预算耗尽，且 :data:`~...training.schedule.STAGE_CRITERIA` 的判据未通过。

    §14 要求这种情况下**明确停止并报告**，不许静默跳到下一阶段。跳过去的话，
    模型带着"初值没做对"的状态进入扩窗阶段，错误会被后续的 PDE 项掩盖成看起来
    还行的波形——从头到尾没有任何一处会报错。
    """

    def __init__(self, stage: str, failures: list[dict[str, Any]]) -> None:
        self.stage = stage
        self.failures = failures
        lines = [f"stage {stage!r} budget is exhausted and its declared criteria did not pass:"]
        for row in failures:
            lines.append(
                f"  - {row['metric']} = {row['value']:.6g}  "
                f"(needs {row['sense']} {row['limit']:.6g})"
            )
        lines.append(
            "Refusing to advance silently. Fix the stage or change the declared "
            "criteria deliberately, in the open."
        )
        super().__init__("\n".join(lines))


@dataclass
class TrainingResult:
    """一次训练的结果摘要。"""

    output_dir: Path
    updates: int
    final_loss: float
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.updates > 0 and np.isfinite(self.final_loss)

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "updates": self.updates,
            "final_loss": self.final_loss,
            "checkpoints": self.checkpoints,
            "metadata": self.metadata,
            "history_length": len(self.history),
        }


class Trainer:
    """把模型、采样器与日程接在一起。"""

    def __init__(
        self,
        *,
        config,
        cases: Sequence[ExtractedCase],
        reference: ReferenceField,
        model: CoupledModel,
        scales_by_radius: Mapping[float, SimilitudeScales],
        time_scale: float,
        schedule: Sequence[Stage] = SCHEDULE,
        sampling: SamplingConfig | None = None,
        seed: int = 0,
        device=None,
        dtype=None,
        output_dir: Path | str | None = None,
        log_every: int = 50,
        checkpoint_every: int = 500,
        check_every: int = 500,
        gradient_clip: float = 1.0,
        r_min_fraction: float = 1.0e-2,
        compatibility: Mapping[str, compat.CaseCompatibility] | None = None,
        compatibility_report: str | None = None,
        allow_unverified_data: bool | None = None,
        synthetic_smoke: bool | None = None,
        tolerances: Mapping[str, float] | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        if not cases:
            raise TrainingError("training needs at least one radius")
        self.config = config
        self.cases = list(cases)
        self.reference = reference
        self.model = model
        self.scales_by_radius = dict(scales_by_radius)
        self.time_scale = float(time_scale)
        self.schedule = tuple(schedule)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype or torch.float64
        self.output_dir = Path(output_dir or "outputs/run")
        self.log_every = int(log_every)
        self.checkpoint_every = int(checkpoint_every)
        self.check_every = int(check_every)
        self.gradient_clip = float(gradient_clip)
        self.tolerances = dict(tolerances or {})

        # -- 数据兼容性闸门（§3、§11） ------------------------------------
        # 审查报告发现"这份数据与方案 A 不一致"与训练照常开始，不能同时成立。
        if allow_unverified_data is None:
            allow_unverified_data = bool(
                config.get("data.allow_unverified_cases", False)
            )
        if synthetic_smoke is None:
            synthetic_smoke = bool(config.get("data.synthetic_smoke", False))
        if compatibility is None:
            from ..cli._common import resolve_case_compatibility

            compatibility, resolved_path = resolve_case_compatibility(config, self.cases)
            compatibility_report = compatibility_report or (
                str(resolved_path) if resolved_path is not None else None
            )
        self.compatibility = dict(compatibility)
        self.compatibility_report = compatibility_report
        self.gate = compat.enforce_training_gate(
            [case.case_id for case in self.cases],
            self.compatibility,
            allow_unverified=bool(allow_unverified_data),
            require_compatible=not bool(synthetic_smoke),
        )

        self.sampler = GroupedSampler.build(
            cases=self.cases,
            scales_by_radius=self.scales_by_radius,
            reference=self.reference,
            config=sampling,
            base_seed=self.seed,
            dtype=self.dtype,
            device=self.device,
            r_min_fraction=r_min_fraction,
            use_development_frames=False,
            dev_holdout_fraction=float(
                config.get("data.development_holdout_fraction", 0.1)
            ),
            split_seed=int(config.get("data.split_seed", self.seed)),
        )
        # 固定检查集：独立种子、开发帧、一次采样、全程复用、不进梯度（§12、§13）。
        self.check_set = build_check_set(
            cases=self.cases,
            scales_by_radius=self.scales_by_radius,
            reference=self.reference,
            config=sampling,
            seed=self.seed + 7919,
            dtype=self.dtype,
            device=self.device,
            r_min_fraction=r_min_fraction,
            dev_holdout_fraction=float(
                config.get("data.development_holdout_fraction", 0.1)
            ),
            split_seed=int(config.get("data.split_seed", self.seed)),
        )
        self.contact_pressure = contact_pressure_scale(
            self.reference.scale_to(self.cases[0].charge_radius),
            charge_radius=self.cases[0].charge_radius,
            detonation_velocity=self.cases[0].detonation_velocity,
        )

        self.check_history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        self.step = 0
        self.history: list[dict[str, Any]] = []

        # -- 有界梯度平衡（§14） -------------------------------------------
        self.balance_config = BalanceConfig.from_config(config)
        from ..losses.objective import BASE_WEIGHTS

        self.balancer = BoundedGradientBalancer(BASE_WEIGHTS, self.balance_config)

        #: 最近一次平衡更新的审计记录。
        self.last_balance: dict[str, Any] | None = None

        # -- 阶段机（§14） --------------------------------------------------
        self.stage_history: list[dict[str, Any]] = []
        self._current_stage: str | None = None
        self._stage_window = float(self.schedule[0].time_window_fraction)
        # 训练采样器从**第一阶段**的窗口开始。构造时它默认是完整时段，若不在
        # 这里对齐，第一阶段就会在完整时段上采样，"初值预训练"名存实亡。
        self.sampler.set_time_window(float(self.schedule[0].time_window_fraction))
        self.provenance = self._provenance()

    # -- 清单 -------------------------------------------------------------
    def _provenance(self) -> dict[str, Any]:
        return {
            "charge_radii": [float(c.charge_radius) for c in self.cases],
            "case_ids": [c.case_id for c in self.cases],
            "time_scale": self.time_scale,
            "start_times": {
                c.case_id: float(c.start_time) for c in self.cases
            },
            "valid_pinn_time_max": {
                c.case_id: float(c.last_valid_pinn_time()) for c in self.cases
            },
            "contact_pressure_scale_pa": float(self.contact_pressure),
            "schedule": describe_schedule(self.schedule),
            "reference_field": self.reference.as_dict(),
            "material": {
                "jwl": self.model.product_net.jwl.as_dict(),
                "gamma_air": self.model.gamma_air,
                "ambient": self.model.undisturbed.as_dict(),
            },
            "sampler": self.sampler.as_dict(),
            "model": self.model.describe(),
            "data_compatibility": {
                "gate": self.gate.as_dict(),
                "report": self.compatibility_report,
                "cases": {k: v.as_dict() for k, v in self.compatibility.items()},
            },
            "check_set": self.check_set.as_dict(),
            "check_every": int(self.check_every),
            "tolerances": dict(self.tolerances),
            "balancing": self.balance_config.as_dict(),
            "time_window": self.sampler.time_window_state(),
        }

    # -- 一步 -------------------------------------------------------------
    def _update(self) -> LossBreakdown:
        batches = self.sampler.batch()
        pde_weight = pde_weight_at(self.step, self.schedule)
        breakdown = total_loss(
            batches,
            self.model,
            contact_pressure=self.contact_pressure,
            pde_weight=pde_weight,
            # 平衡器接管的六组；``pde`` 由日程给出，``total_loss`` 会覆盖掉
            # 传进来的那个值（见 `SCHEDULE_OWNED`）。
            weights=self.balancer.weights(),
        )

        # 先量梯度、再反传。顺序反过来的话图已经被 ``backward`` 释放，
        # ``measure_gradient_norms`` 会因为拿不到计算图而失败。
        # 量到的权重从**下一步**开始生效——本步的权重已经用在上面那次求值里了。
        if self.balancer.should_update(self.step):
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            norms = measure_gradient_norms(
                breakdown.group_tensors, parameters, retain_graph=True
            )
            record = self.balancer.update(
                self.step,
                norms,
                unweighted=breakdown.terms,
                live=live_groups(breakdown.group_tensors),
            )
            self.last_balance = record.as_dict()

        self.optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()
        if self.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=self.gradient_clip,
            )
        self.optimizer.step()
        return breakdown

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    # -- 独立检查 ---------------------------------------------------------
    def verify(
        self,
        *,
        verbose: bool = False,
        total_loss_value: float | None = None,
        window_fraction: float | None = None,
        update_best: bool = True,
    ) -> CheckReport:
        """在**固定**检查集上跑一次完整物理检查，并据此更新最佳检查点。

        ``window_fraction`` 给出时，检查集被裁到该尺度化时间窗口（§14.2）。裁的
        是同一批固定点，不重采。阶段判据用它，因为拿窗口外的点去评判一个还没训到
        那些时段的阶段，判据永远不可能通过。

        ``update_best=False`` 用于阶段判据：选模型必须始终在**完整**的固定检查集
        上做（§13），不能因为某个阶段只在窗口内评价就把那个窗口的结果写成 best。

        "每 500 步保存一个检查点"不是"每 500 步完成一次物理验收"。训练损失是在
        当步新采的样本上算的，随采样抖动；判断"方程是否已经满足"要看这一套固定
        点上的逐方程残差。

        选模型用的是 :data:`~...evaluation.checks.SELECTION_CRITERIA` 里那条固定
        的偏序：先看有没有非物理状态，再看物理与界面残差超出 §16 容差多少，最后
        才比较开发集误差。标准在查看最终测试结果之前就固定在这里，不按加权训练
        总损失选模型。
        """
        check_set = (
            self.check_set if window_fraction is None
            else self.check_set.windowed(float(window_fraction))
        )
        report = run_checks(
            self.model,
            check_set,
            pde_weight=pde_weight_at(self.step, self.schedule),
            total_loss=total_loss_value,
            tolerances=self.tolerances,
            steps=self.step,
        )
        record = report.as_dict()
        if window_fraction is not None:
            record["window_fraction"] = float(window_fraction)
        self.check_history.append(record)

        if not update_best:
            return report

        if self.best is None or report.score < tuple(self.best["score"]):
            self.best = record
            self._save(self.output_dir / "best.pt")
            if verbose:
                print(
                    f"  [check] step {self.step}: new best  "
                    f"nonphysical={report.metrics['nonphysical_violation_count']:.3g}  "
                    f"pde_rms_max={report.metrics['pde_rms_max']:.3e}  "
                    f"dev={report.metrics['development_prediction_error']:.3e}"
                )
        elif verbose:
            print(
                f"  [check] step {self.step}: "
                f"nonphysical={report.metrics['nonphysical_violation_count']:.3g}  "
                f"pde_rms_max={report.metrics['pde_rms_max']:.3e}  "
                f"dev={report.metrics['development_prediction_error']:.3e}  (kept best)"
            )
        return report

    # -- 阶段机（§14） -----------------------------------------------------
    def _enter_stage(self, stage: Stage, *, previous: str | None, verbose: bool) -> None:
        """进入一个阶段：先**验收上一阶段**，再设置本阶段的采样时间窗口。

        顺序不能反。先扩窗再审上一阶段的话，上一阶段的判据会在一个更大的时段上
        评价——而模型还没训过那些时段，判据几乎必然不通过，于是每次扩窗都会报一次
        假失败。
        """
        if previous is not None:
            self._evaluate_stage_criteria(
                previous, window_fraction=self._stage_window, fatal=True, verbose=verbose
            )

        self.sampler.set_time_window(float(stage.time_window_fraction))
        self._stage_window = float(stage.time_window_fraction)
        self._current_stage = stage.name
        self.stage_history.append({
            "event": "enter",
            "stage": stage.name,
            "step": int(self.step),
            "time_window_fraction": float(stage.time_window_fraction),
            "criteria": [c.as_dict(self.tolerances) for c in criteria_for(stage.name)],
        })
        if verbose:
            print(
                f"[stage] {stage.name} at step {self.step}  "
                f"window={stage.time_window_fraction:.0%}"
            )

    def _evaluate_stage_criteria(
        self,
        stage_name: str,
        *,
        window_fraction: float,
        fatal: bool,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """按**预先声明**的判据验收一个阶段（§14）。

        ``fatal=True``：未通过时抛 :class:`StageNotPassed`——阶段预算耗尽仍不通过
        就必须明确停止，不许静默跳到下一阶段。
        """
        criteria = criteria_for(stage_name)
        report = self.verify(
            verbose=False, window_fraction=window_fraction, update_best=False
        )
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for criterion in criteria:
            passed, value, limit = criterion.evaluate(report.metrics, self.tolerances)
            row = {
                "metric": criterion.metric,
                "sense": criterion.sense,
                "value": value,
                "limit": limit,
                "passed": bool(passed),
                "description": criterion.description,
            }
            rows.append(row)
            if not passed:
                failures.append(row)

        record = {
            "event": "criteria",
            "stage": stage_name,
            "step": int(self.step),
            "time_window_fraction": float(window_fraction),
            "passed": not failures,
            "criteria": rows,
        }
        self.stage_history.append(record)
        if verbose:
            verdict = "passed" if not failures else "NOT PASSED"
            print(f"[stage] {stage_name} criteria: {verdict}")

        if failures and fatal:
            raise StageNotPassed(stage_name, failures)
        if failures:
            print(
                f"\n** 阶段 {stage_name} 的判据未通过（训练已结束，无下一阶段可跳）**"
            )
            for row in failures:
                print(
                    f"  - {row['metric']} = {row['value']:.6g}  "
                    f"(needs {row['sense']} {row['limit']:.6g})"
                )
        return record

    # -- 主循环 -----------------------------------------------------------
    def run(self, *, max_updates: int | None = None, verbose: bool = False) -> TrainingResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 完整日程的长度。收尾判断必须用它，**不能**用截断后的 ``total``。
        schedule_total = sum(stage.updates for stage in self.schedule)
        total = schedule_total
        if max_updates is not None:
            total = min(total, int(max_updates))

        started = time.time()
        # **从实例状态起手，而不是从空字符串起手。**
        #
        # ``stage_name`` 初值写成 ``""`` 时，任何一次"重新调用 run()"都会被当成
        # 第一次进循环：``current.name != ""`` 成立，而 ``previous`` 是 ``None``，
        # 于是**阶段判据不会被重新评价**。如果上一次 run() 正是在阶段边界上被判据
        # 拦下的，第二次调用就会直接跳过那道闸门继续训练——失败状态没有被保留。
        stage_name = self._current_stage or ""
        # 每个阶段**跑完**时应当落在哪一步。只有真的落在那里才推进。
        #
        # 用步数区间来判断"该不该切阶段"是不行的：``--max-steps`` 截断的联检停在
        # 半路，那时按区间会算出"已经进入下一阶段"，于是拿没训完的模型去审上一阶段
        # 的判据——把"没跑完"报成"没通过"。
        stage_ends: dict[str, int] = {}
        cursor = 0
        for stage in self.schedule:
            cursor += stage.updates
            stage_ends[stage.name] = cursor

        while self.step < total:
            stage_index, _ = _stage_index(self.step, self.schedule)
            current = self.schedule[stage_index]
            if current.name != stage_name:
                self._enter_stage(current, previous=stage_name or None, verbose=verbose)
                stage_name = current.name
            self._set_learning_rate(learning_rate_at(self.step, self.schedule))
            breakdown = self._update()
            self.step += 1

            # **步数一增加就切阶段**，不留到下一轮循环的开头。
            #
            # 留到下一轮的话，"第 2000 步的检查点"里存的是阶段 1 的窗口 0.25，而
            # 按步数推算第 2000 步属于阶段 2（0.65）——恢复函数会拒绝自己刚写的
            # 检查点。默认每 500 步保存一次，2000 与 5000 恰好就是两个阶段边界，
            # 所以这不是一个罕见角落，是默认配置下的必然。
            if stage_ends.get(stage_name) == self.step:
                position = next(
                    (index for index, stage in enumerate(self.schedule)
                     if stage.name == stage_name),
                    None,
                )
                if position is not None and position + 1 < len(self.schedule):
                    following = self.schedule[position + 1]
                    self._enter_stage(following, previous=stage_name, verbose=verbose)
                    stage_name = following.name

            if self.step % self.log_every == 0 or self.step == 1:
                record = {
                    "step": self.step,
                    "stage": stage_name,
                    "lr": learning_rate_at(self.step, self.schedule),
                    "pde_weight": pde_weight_at(self.step, self.schedule),
                    "total": float(breakdown.total.detach()),
                    "per_case": dict(breakdown.per_case),
                    "terms": {k: float(v) for k, v in breakdown.terms.items()},
                }
                self.history.append(record)
                if verbose:
                    print(
                        f"  step {self.step:6d}  total={record['total']:.6e}  "
                        f"lr={record['lr']:.2e}  lambda_PDE={record['pde_weight']:.3f}"
                    )
            if self.check_every > 0 and self.step % self.check_every == 0:
                self.verify(
                    verbose=verbose, total_loss_value=float(breakdown.total.detach())
                )
            if self.step % self.checkpoint_every == 0:
                self._save(self.output_dir / "last.pt")

        self._save(self.output_dir / "last.pt")
        final = float(self.history[-1]["total"]) if self.history else float("nan")
        # 收尾检查：即使还没到 check_every，也要让最后一次迭代有物理验收结论。
        if self.check_every > 0 and (
            not self.check_history or self.check_history[-1]["step"] != self.step
        ):
            self.verify(verbose=verbose, total_loss_value=final)

        # 最后一个阶段的判据：只有在它的预算**真的跑完**时才评价。
        #
        # 判据是 ``self.step >= schedule_total``，**不是** ``>= total``。后者是
        # 截断后的长度：三段各两步、``--max-steps 4`` 时，前两段刚好跑完、第三段
        # **一步没跑**，而 ``self.step (4) >= total (4)`` 成立，于是会拿一个完全
        # 没训练过的模型去评价第三段的最终判据。用完整日程的长度就不会。
        stage_verdict: dict[str, Any] | None = None
        if self._current_stage is not None and self.step >= schedule_total:
            last_stage = self.schedule[-1]
            if last_stage.name == self._current_stage:
                stage_verdict = self._evaluate_stage_criteria(
                    self._current_stage,
                    window_fraction=self._stage_window,
                    fatal=False,
                    verbose=verbose,
                )

        result = TrainingResult(
            output_dir=self.output_dir,
            updates=self.step,
            final_loss=final,
            history=list(self.history),
            checkpoints=self._checkpoint_paths(),
            metadata={
                **self.provenance,
                "wall_seconds": time.time() - started,
                "checks": self.check_history,
                "best_check": self.best,
                "data_compatibility": "unverified" if self.gate.unverified else "verified",
                "stages": self.stage_history,
                "final_stage_verdict": stage_verdict,
                "balancing": {
                    "config": self.balance_config.as_dict(),
                    "monitors": self.balancer.monitors(),
                },
                "time_window": self.sampler.time_window_state(),
            },
        )
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(result.as_dict(), indent=2, default=_jsonable), encoding="utf-8"
        )
        (self.output_dir / "checks.json").write_text(
            json.dumps(self.check_history, indent=2, default=_jsonable), encoding="utf-8"
        )
        return result

    def _checkpoint_paths(self) -> dict[str, str]:
        paths = {"last": str(self.output_dir / "last.pt")}
        best = self.output_dir / "best.pt"
        if best.exists():
            paths["best"] = str(best)
        return paths

    # -- L-BFGS 精修 ------------------------------------------------------
    def refine(
        self,
        *,
        iterations: int = MAX_LBFGS_ITERATIONS,
        verbose: bool = False,
    ) -> TrainingResult:
        """可选的 L-BFGS 精修。

        **采样点、权重与目标函数全部固定**：一次采样，反复求值。规范明确禁止
        在每次 closure 求值时重采样——那会让"同一次线搜索"里的目标函数不是
        同一个函数，线搜索的前提就不成立了。
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        iterations = min(int(iterations), MAX_LBFGS_ITERATIONS)
        if iterations <= 0:
            return self.run(max_updates=self.step)

        batches = self.sampler.batch()
        pde_weight = pde_weight_at(self.step, self.schedule)
        optimizer = torch.optim.LBFGS(
            [p for p in self.model.parameters() if p.requires_grad],
            max_iter=iterations,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        last: dict[str, Any] = {}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            breakdown = total_loss(
                batches,
                self.model,
                contact_pressure=self.contact_pressure,
                pde_weight=pde_weight,
            )
            breakdown.total.backward()
            last["total"] = float(breakdown.total.detach())
            return breakdown.total

        optimizer.step(closure)
        self.history.append(
            {"step": self.step, "stage": "lbfgs", "total": last.get("total", float("nan")),
             "lr": 0.0, "pde_weight": pde_weight, "per_case": {}, "terms": {}}
        )
        if verbose:
            print(f"[lbfgs] fixed-sample refinement -> {last.get('total')}")
        return self.run(max_updates=self.step)

    # -- 检查点 -----------------------------------------------------------
    def _save(self, path: Path) -> Path:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rates": [g["lr"] for g in self.optimizer.param_groups],
            "rng": _rng_state(self.seed),
            # 采样器自己的发生器**不在** ``np.random.get_state()`` 里，必须单独
            # 存：只恢复全局状态会让采样序列从别的地方继续，"中断后恢复到第 N 步"
            # 于是不等于"连续跑 N 步"。
            "sampler_rng": self.sampler.rng_state(),
            "history": self.history,
            "checks": self.check_history,
            "best_check": self.best,
            "check_set": self.check_set.as_dict(),
            "data_compatibility": self.gate.as_dict(),
            # §14：恢复必须连阶段、时间窗口与自适应权重统计一起恢复。少了它们，
            # "从第 N 步继续"会换掉采样区间和损失权重，与连续跑 N 步不是同一次
            # 训练——而模型文件本身完全正常，看不出来。
            "balancing": self.balancer.state_dict(),
            "stage": {
                "current": self._current_stage,
                "window_fraction": float(self._stage_window),
                "history": self.stage_history,
            },
            "provenance": self.provenance,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "case_ids": [c.case_id for c in self.cases],
            "split": {c.case_id: c.source.get("split") for c in self.cases},
            "time_scale": self.time_scale,
            "contact_pressure_scale_pa": self.contact_pressure,
            "scales": {
                c.case_id: self.scales_by_radius[float(c.charge_radius)].as_dict()
                for c in self.cases
            },
            "code_version": _code_version(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        return target

    def load(self, path: Path | str) -> None:
        """恢复训练。

        恢复的是**优化状态**：模型、优化器、步数、随机数发生器。恢复后继续
        跑完剩余日程；已经走过的更新次数不会重复。
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise TrainingError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
            raise TrainingError(
                f"checkpoint version {payload.get('version')} does not match "
                f"{CHECKPOINT_VERSION}"
            )
        if abs(float(payload.get("time_scale", 0.0)) - self.time_scale) > 1.0e-12:
            raise TrainingError(
                "the checkpoint was trained with a different hat T_*; resuming would "
                "silently change the network's time input"
            )
        if "sampler_rng" not in payload:
            raise TrainingError(
                f"{path} has no sampler RNG state. Restoring training from it would "
                "continue the data-sampling sequence from somewhere else, so 'resume to "
                "step N' would not reproduce 'run N steps'. Retrain from a checkpoint "
                "written by this version."
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])
        self.history = list(payload.get("history", []))
        self.check_history = list(payload.get("checks", []))
        self.best = payload.get("best_check")
        _restore_rng(payload.get("rng"))
        self.sampler.set_rng_state(payload.get("sampler_rng"))

        # 阶段、时间窗口、自适应权重统计（§14）。窗口按**检查点记的步数**应当
        # 是多少来核对：若两者不一致，说明检查点是在一份不同的日程下跑的，继续
        # 恢复会让采样区间与步数对不上，而损失曲线看起来完全正常。
        stage_payload = payload.get("stage") or {}
        expected_window = time_window_fraction_at(self.step, self.schedule)
        stored_window = stage_payload.get("window_fraction")
        if stored_window is not None and abs(
            float(stored_window) - expected_window
        ) > 1.0e-12:
            raise TrainingError(
                f"the checkpoint was written in a stage whose time window is "
                f"{stored_window}, but step {self.step} of this schedule needs "
                f"{expected_window}; the two schedules are not the same"
            )
        self.sampler.set_time_window(float(expected_window))
        self._stage_window = float(expected_window)
        self._current_stage = stage_payload.get("current")
        self.stage_history = list(stage_payload.get("history", []))
        self.balancer.load_state_dict(payload.get("balancing"))


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------


def _stage_index(step: int, schedule: Sequence[Stage]) -> tuple[int, float]:
    remaining = int(step)
    for index, stage in enumerate(schedule):
        if remaining < stage.updates:
            return index, (remaining / stage.updates if stage.updates else 0.0)
        remaining -= stage.updates
    return len(schedule) - 1, 1.0


def _rng_state(seed: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "seed": int(seed),
        "python": None,
        "numpy": None,
        "torch": None,
    }
    try:
        import random

        state["python"] = random.getstate()
    except Exception:  # pragma: no cover - defensive
        pass
    state["numpy"] = np.random.get_state()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
    return state


def _restore_rng(payload: Mapping[str, Any] | None) -> None:
    if not payload:
        return
    try:
        import random

        if payload.get("python") is not None:
            random.setstate(payload["python"])
    except Exception:  # pragma: no cover - defensive
        pass
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if torch is not None and payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])


def _code_version() -> dict[str, Any]:
    from .. import __version__

    return {"package": __version__}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "detach"):
        return float(value.detach())
    return str(value)


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "StageNotPassed",
    "Trainer",
    "TrainingError",
    "TrainingResult",
]
