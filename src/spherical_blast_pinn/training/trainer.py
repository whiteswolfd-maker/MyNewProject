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

# --------------------------------------------------------------------------
# 一次运行的**终态**（提示词 §1）
# --------------------------------------------------------------------------
# 四档必须分开。"跑完了"与"通过了"不是一回事，而以前 ``ok`` 只看
# ``updates > 0 and isfinite(final_loss)``——一个最终判据失败的运行和一次成功的
# 正式训练拿到**同一个**布尔值，CLI 于是返回 0。
RUN_COMPLETED = "completed"              # 预算跑满，最终阶段判据通过
RUN_CRITERIA_FAILED = "criteria_failed"  # 预算跑满，最终阶段判据未通过
RUN_INCOMPLETE = "incomplete"            # 预算被截断，完整日程没跑完
RUN_DIAGNOSTIC = "diagnostic"            # 显式诊断运行：不主张任何验收结论

#: 每个终态的含义，写进运行清单，免得读的人各自理解。
RUN_STATUS_MEANING: dict[str, str] = {
    RUN_COMPLETED: (
        "the full schedule ran and the final stage's pre-declared criteria passed — "
        "this is a numerically completed run, which is still not a physics acceptance"
    ),
    RUN_CRITERIA_FAILED: (
        "the full schedule ran but the final stage's pre-declared criteria did not "
        "pass; the run must not be presented as a formal success"
    ),
    RUN_INCOMPLETE: (
        "the budget was truncated before the full schedule completed; no claim is made "
        "about stages that never ran"
    ),
    RUN_DIAGNOSTIC: (
        "an explicitly diagnostic run: it makes no acceptance claim at all"
    ),
}
#: 第 3 版起检查点必须带**采样器自己的** RNG 状态（第 2 版只存全局 NumPy 状态，
#: 恢复出来的不是一个连续的采样轨迹），并且带固定检查集与兼容性闸门的结论。
#:
#: 第 4 版起 ``stage`` 块必须同时带 ``pending`` 与 ``blocked``，且加载端**要求
#: 它们存在**（缺失即拒绝）。这两项记录的是"更新跑完了、判据没过、尚不允许推进"
#: 这个中间状态；旧写法没有它们，读出来会被当成"没有待决判据"——那正是这两个
#: 字段要区分的那件事。所以旧检查点按版本号直接拒绝，而不是"缺字段就默认通过"。
CHECKPOINT_VERSION = 4


class TrainingError(RuntimeError):
    """训练无法继续。"""


class NonFiniteUpdate(TrainingError):
    """一次参数更新在**应用之前**被拦下：损失或梯度不是有限值（提示词 §8）。

    这是一个硬失败，不是可以跳过去继续的事件：跳过之后**下一步大概率还会发散**，
    而真正的问题——哪一步、哪个参数——会被埋在一串 nan 里。这里把步骤与参数名
    直接报出来，并保证参数没有被改（拦截发生在 ``optimizer.step()`` 之前），
    因而最近一个已知有效的检查点没有被覆盖。
    """

    def __init__(
        self,
        *,
        step: int,
        loss: float,
        bad_parameters: Sequence[str],
        detail: str,
    ) -> None:
        self.step = int(step)
        self.loss = float(loss)
        self.bad_parameters = list(bad_parameters)
        lines = [f"update {self.step} was blocked: {detail}"]
        if bad_parameters:
            lines.append(f"  non-finite gradient in: {list(bad_parameters)[:8]}")
        lines.append(
            "  the parameters were NOT updated, so the last good checkpoint is intact; "
            "no NaN was papered over with nan_to_num or a gradient filter"
        )
        super().__init__("\n".join(lines))


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
    #: 终态（提示词 §1）。见 :data:`RUN_STATUS_MEANING`。
    status: str = RUN_DIAGNOSTIC
    #: ``final_loss`` 是哪一步实测的。以前它取的是**最后一条日志**，一旦日志间隔
    #: 不是 1，它就来自更早的模型——却和最终结论写在同一个字段里。
    final_loss_step: int = 0
    #: 最终阶段的判据结论（原始记录，含每条判据的实际值与限额）。
    final_stage_verdict: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """**只有**"预算跑满且最终判据通过"才算成功。

        截断是 ``incomplete``，判据失败是 ``criteria_failed``，诊断运行是
        ``diagnostic``——三者都不是成功，而它们各自的原因不同，不能压成一个
        ``False``。
        """
        return self.status == RUN_COMPLETED

    @property
    def formally_accepted(self) -> bool:
        """正式验收。与 :attr:`ok` 同义，单独给一个名字免得被读成"物理验收通过"。

        这一档只说明**数值运行**按要求跑完并通过了**数值**判据；物理验收另有
        单独的闸门（参考场验收、物理依据），两者不能互相替代。
        """
        return self.status == RUN_COMPLETED

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "updates": self.updates,
            "status": self.status,
            "status_meaning": RUN_STATUS_MEANING.get(self.status, "unknown status"),
            "formally_accepted": self.formally_accepted,
            "final_loss": self.final_loss,
            "final_loss_step": int(self.final_loss_step),
            "final_stage_verdict": self.final_stage_verdict,
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
        #: 最后一次参数更新的实测损失与它发生在哪一步。
        self._last_update_loss = float("nan")
        self._last_update_loss_step = 0
        #: 参数指纹与那个指纹下的检查结论（提示词 §7）。
        #:
        #: **只用 step 判断"这一步检查过了"是不够的**：L-BFGS 精修改参数却不改
        #: step，于是"同一步已检查"会把精修前的结论留下来，报告里的误差、best/last
        #: 与最终结论都对不上当前参数。指纹取参数张量的内容哈希。
        self._parameter_fingerprint: str | None = None
        self._check_fingerprint: str | None = None
        #: 精修记录。精修**恢复不了**：L-BFGS 的优化器状态不进检查点，所以
        #: "从检查点继续一次精修"是不支持的，这里如实记下来而不是假装支持。
        self.refine_history: list[dict[str, Any]] = []
        self._lbfgs_closures = 0

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
        #: 上一阶段"更新跑完了、判据没过、因此不允许推进"的中间状态。见
        #: :meth:`_evaluate_stage_criteria` 与 :meth:`load`。
        self._pending_stage: str | None = None
        self._stage_blocked = False
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
        # **非有限损失在反传之前就拦下。** 让它传下去只会得到一堆 NaN 梯度，
        # 然后被后面的有限性检查再拦一次——但那时已经白算了一遍。
        if not bool(torch.isfinite(breakdown.total)):
            raise NonFiniteUpdate(
                step=int(self.step) + 1, loss=float("nan"), bad_parameters=[],
                detail="the total loss is not finite",
            )
        breakdown.total.backward()

        # **参数更新之前识别非有限梯度**（提示词 §8）。这一条缺失时，一次 NaN
        # 梯度会把整个模型污染成 NaN，而训练"照常"跑完、损失显示为 nan、检查点覆盖
        # 掉上一个已知有效的快照——事后什么也追不回来。
        #
        # 这里**不**用 ``nan_to_num``、不裁剪、不跳过该组：那三种做法都把异常变成
        # "看起来还行"，而这一段存在的意义正好相反。定位到具体的参数名之后直接停。
        bad_parameters = [
            name for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if bad_parameters:
            self.optimizer.zero_grad(set_to_none=True)
            raise NonFiniteUpdate(
                step=int(self.step) + 1,
                loss=float(breakdown.total.detach()),
                bad_parameters=bad_parameters,
                detail=(
                    "the gradient is not finite for these parameters; the update was "
                    "blocked before it could overwrite the last good checkpoint"
                ),
            )

        if self.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=self.gradient_clip,
            )
        self.optimizer.step()
        # 最后一次更新的**实测**损失。收尾结论必须用它，不能用最后一条日志——
        # 日志按 ``log_every`` 打，间隔不是 1 时它来自更早的模型。
        self._last_update_loss = float(breakdown.total.detach())
        return breakdown

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    # -- 参数指纹（提示词 §7）----------------------------------------------
    def parameter_fingerprint(self) -> str:
        """当前模型参数的**内容**指纹。

        ``step`` 不变但参数变过的情况是真实存在的：L-BFGS 精修就是。用 step 当
        "检查过没有"的键，会让精修后的报告沿用精修前的结论。
        """
        import hashlib

        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
        return digest.hexdigest()

    def _check_stale(self) -> bool:
        """当前参数下还没有检查结论吗（从来没查过 / 参数变了 / 步数变了）。"""
        if not self.check_history:
            return True
        last = self.check_history[-1]
        if int(last.get("step", -1)) != int(self.step):
            return True
        return last.get("parameter_fingerprint") != self.parameter_fingerprint()

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
        record["parameter_fingerprint"] = self.parameter_fingerprint()
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
        # 切换成功：上一阶段没有悬而未决的判据
        self._stage_blocked = False
        self._pending_stage = None
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
            # **把"已完成更新、但判据未过、尚不允许推进"这件事记在实例上。**
            # 只抛异常的话，状态只活在调用栈里：同一个实例再跑一次能拦住（因为
            # ``_current_stage`` 还停在失败的阶段），但**换个实例从这个检查点恢复
            # 就拦不住了**——恢复只能从步数推算阶段，推算出来的正是"已经进入下一
            # 阶段"。所以这个中间状态必须被显式保存。
            self._stage_blocked = True
            self._pending_stage = stage_name
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
    def run(
        self,
        *,
        max_updates: int | None = None,
        verbose: bool = False,
        diagnostic: bool = False,
    ) -> TrainingResult:
        """跑训练循环。

        ``max_updates`` 截断的运行是 ``incomplete``（提示词 §1）：不拿没跑过的
        阶段判失败，也不宣称正式训练完成。``diagnostic=True`` 的运行不主张任何
        验收结论，无论它跑到哪一步。
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 完整日程的长度。收尾判断必须用它，**不能**用截断后的 ``total``。
        schedule_total = sum(stage.updates for stage in self.schedule)
        total = schedule_total
        if max_updates is not None:
            total = min(total, int(max_updates))

        # 悬而未决的阶段判据：**先重新评价，再谈继续训练**。
        #
        # 这一条覆盖"从这样一个检查点恢复"的情形：上一次运行跑完了某阶段的更新、
        # 判据没过、于是不允许推进。恢复出来的 ``self.step`` 正好在那个阶段的结束
        # 步上，若只是照常进循环，步数已经等于 total 时循环体一次都不执行，那道
        # 闸门就永远不会被重新评价。
        if self._stage_blocked and self._pending_stage is not None:
            self._evaluate_stage_criteria(
                self._pending_stage,
                window_fraction=self._stage_window,
                fatal=True,
                verbose=verbose,
            )

        started = time.time()
        start_step = int(self.step)
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
            self._last_update_loss_step = self.step

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

        # **收尾结论必须在写最终检查点之前算出来。**
        #
        # 以前 ``last.pt`` 先写、判据后评，于是最终检查点里没有那个结论——从它
        # 恢复的人看到的是一份"跑完了"的快照，而这次运行其实是被判失败的。
        final = float(self._last_update_loss)
        final_step = int(self._last_update_loss_step)

        # 收尾检查：即使还没到 check_every，也要让最后一次迭代有物理验收结论。
        # 一次更新都没发生时跳过——拿一个没动过的模型去跑完整检查，只会往
        # ``best.pt`` 里写一份"从未训练过"的快照。
        if self.check_every > 0 and self.step > start_step and self._check_stale():
            self.verify(verbose=verbose, total_loss_value=final)

        # 最后一个阶段的判据：只有在它的预算**真的跑完**时才评价。
        #
        # 判据是 ``self.step >= schedule_total``，**不是** ``>= total``。后者是
        # 截断后的长度：三段各两步、``--max-steps 4`` 时，前两段刚好跑完、第三段
        # **一步没跑**，而 ``self.step (4) >= total (4)`` 成立，于是会拿一个完全
        # 没训练过的模型去评价第三段的最终判据。用完整日程的长度就不会。
        stage_verdict: dict[str, Any] | None = None
        truncated = self.step < schedule_total
        if truncated:
            # 截断的运行**不评价任何阶段**：没跑到的阶段既不算通过也不算失败。
            stage_verdict = None
        elif self.check_every <= 0:
            # 一次检查都没跑，就没有任何验收结论可言。**缺检查不是通过。**
            stage_verdict = {
                "event": "criteria",
                "stage": self._current_stage,
                "step": int(self.step),
                "passed": False,
                "criteria": [],
                "reason": (
                    "check_every <= 0, so no acceptance check was executed; a run with "
                    "no checks cannot be reported as a formal success"
                ),
            }
        elif self._current_stage is not None and (
            self.schedule[-1].name == self._current_stage
        ):
            stage_verdict = self._evaluate_stage_criteria(
                self._current_stage,
                window_fraction=self._stage_window,
                fatal=False,
                verbose=verbose,
            )
        else:
            # 预算跑满了，却停在**非最后一个**阶段——日程与循环不一致，不能算通过。
            stage_verdict = {
                "event": "criteria",
                "stage": self._current_stage,
                "step": int(self.step),
                "passed": False,
                "criteria": [],
                "reason": (
                    "the full schedule ran out but the run is not in its last stage; the "
                    "final acceptance criteria were never evaluated"
                ),
            }

        if diagnostic:
            status = RUN_DIAGNOSTIC
        elif truncated:
            status = RUN_INCOMPLETE
        elif stage_verdict is not None and stage_verdict.get("passed") is True:
            status = RUN_COMPLETED
        else:
            status = RUN_CRITERIA_FAILED

        # 现在才写最终检查点：它带着上面那个结论。
        self._final_status = status
        self._final_stage_verdict = stage_verdict
        self._save(self.output_dir / "last.pt")

        result = TrainingResult(
            output_dir=self.output_dir,
            updates=self.step,
            final_loss=final,
            status=status,
            final_loss_step=final_step,
            final_stage_verdict=stage_verdict,
            history=list(self.history),
            checkpoints=self._checkpoint_paths(),
            metadata={
                **self.provenance,
                "wall_seconds": time.time() - started,
                "checks": self.check_history,
                "best_check": self.best,
                "data_compatibility": "unverified" if self.gate.unverified else "verified",
                "stages": self.stage_history,
                "status": status,
                "status_meaning": RUN_STATUS_MEANING.get(status, "unknown status"),
                "schedule_total_updates": int(schedule_total),
                "truncated": bool(truncated),
                "diagnostic": bool(diagnostic),
                "final_stage_verdict": stage_verdict,
                "balancing": {
                    "config": self.balance_config.as_dict(),
                    "monitors": self.balancer.monitors(),
                },
                "time_window": self.sampler.time_window_state(),
            },
        )
        # 落盘顺序：结论已经写进检查点（在 `_save` 里），再写清单与检查历史。
        # 反过来写的话，中途崩溃会留下一份**有清单、没有结论**的目录，而清单看起来
        # 是完整的。
        (self.output_dir / "checks.json").write_text(
            json.dumps(self.check_history, indent=2, default=_jsonable), encoding="utf-8"
        )
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(result.as_dict(), indent=2, default=_jsonable), encoding="utf-8"
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
        if self._stage_blocked:
            # 精修会改参数。在一个"判据没过、不允许推进"的状态上精修，等于绕过
            # 那道闸门——先把判据跑过再说。
            raise TrainingError(
                f"refusing to refine while stage {self._pending_stage!r} is blocked on a "
                "failed acceptance check; resolve the stage first"
            )
        iterations = min(int(iterations), MAX_LBFGS_ITERATIONS)
        if iterations <= 0:
            return self.run(max_updates=self.step)

        batches = self.sampler.batch()
        pde_weight = pde_weight_at(self.step, self.schedule)
        # **权重必须与训练时一致，而且要记下来。** 以前这里调用 ``total_loss`` 时
        # 没传 ``weights``，于是精修悄悄退回了基础权重——自适应平衡器算出来的那套
        # 权重被丢掉了，目标函数与训练时不是同一个，而精修的损失看起来完全正常。
        weights = self.balancer.weights()
        self._refine_weights = dict(weights)
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
                weights=weights,
                contact_pressure=self.contact_pressure,
                pde_weight=pde_weight,
            )
            breakdown.total.backward()
            last["total"] = float(breakdown.total.detach())
            return breakdown.total

        self._lbfgs_closures = 0

        def counted_closure():
            self._lbfgs_closures += 1
            return closure()

        optimizer.step(counted_closure)
        self.refine_history.append({
            "step": self.step,
            "iterations_requested": int(iterations),
            "closures": int(self._lbfgs_closures),
            "loss_after": last.get("total", float("nan")),
            "weights": dict(weights),
            "pde_weight": float(pde_weight),
            "parameters_changed": True,
            "resume_supported": False,
        })
        if verbose:
            print(
                f"[lbfgs] fixed-sample refinement -> {last.get('total')} "
                f"({self._lbfgs_closures} closures)"
            )

        # **精修改了参数，必须重新评价。** 精修不动 ``step``，所以按"这一步查过了"
        # 判断会把精修前的结论留在报告里——误差、best/last 与最终结论都对不上当前
        # 参数。这里显式作废当前指纹下的检查结论，再让收尾流程重查一遍。
        self._check_fingerprint = None
        result = self.run(max_updates=self.step, verbose=verbose)
        # 精修过之后状态一律不是"正式成功"：目标函数中途换过一次，这次运行不再是
        # 一条单一优化轨迹。要得到正式结论，就重跑一次不含精修的训练。
        result.status = RUN_DIAGNOSTIC
        result.metadata["refinement"] = list(self.refine_history)
        result.metadata["status"] = result.status
        result.metadata["status_meaning"] = RUN_STATUS_MEANING[RUN_DIAGNOSTIC]
        result.metadata["refinement_note"] = (
            "L-BFGS refinement changed the parameters without advancing the step, so "
            "this run is reported as diagnostic; its objective also changed mid-run"
        )
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(result.as_dict(), indent=2, default=_jsonable), encoding="utf-8"
        )
        return result

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
            "run_status": getattr(self, "_final_status", None),
            "final_stage_verdict": getattr(self, "_final_stage_verdict", None),
            "stage": {
                "current": self._current_stage,
                "window_fraction": float(self._stage_window),
                # "更新跑完了、判据没过"是一个**合法的中间状态**，必须能存下来：
                # 不存的话，恢复只能从步数推算阶段，推算出来的正是"已经进入下一
                # 阶段"，于是同一个检查点会被自己的恢复函数判成日程不一致。
                "pending": self._pending_stage,
                "blocked": bool(self._stage_blocked),
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

        # -- 第一阶段：只读检查。**一个字节都不改** --------------------------
        #
        # 上一轮只把"窗口检查"挪到了前面，于是窗口之外的失败点仍然会半加载：
        # 非法采样器 RNG、缺 ``window_fraction``、模型/优化器键不匹配，全都在
        # 模型、优化器、步数与全局 RNG 已经改掉之后才报错。下面把**全部**前置
        # 条件一次查完，任何一条不成立都在提交之前退出。
        if not isinstance(payload["step"], (int, np.integer)):
            raise TrainingError(
                f"the checkpoint's step is {payload['step']!r}, not an integer"
            )
        if not isinstance(payload.get("model"), Mapping) or not isinstance(
            payload.get("optimizer"), Mapping
        ):
            raise TrainingError(
                "the checkpoint must carry 'model' and 'optimizer' state dictionaries"
            )
        # 模型状态必须与本模型的键、形状完全对得上——先比，别等 load_state_dict
        # 装到一半才发现。
        current = self.model.state_dict()
        stored_model = payload["model"]
        missing = [key for key in current if key not in stored_model]
        unexpected = [key for key in stored_model if key not in current]
        if missing or unexpected:
            raise TrainingError(
                f"the checkpoint's model state does not match this model: missing "
                f"{missing[:4]}, unexpected {unexpected[:4]}"
            )
        mismatched = [
            key for key, value in current.items()
            if tuple(stored_model[key].shape) != tuple(value.shape)
        ]
        if mismatched:
            raise TrainingError(
                f"the checkpoint's model state has the wrong shapes for {mismatched[:4]}"
            )
        if not isinstance(payload.get("balancing"), Mapping):
            raise TrainingError(
                "the checkpoint carries no adaptive-weight state; restoring without it "
                "would silently fall back to the base weights"
            )
        # 采样器随机状态：结构、版本、每个算例的键一次查完。采样器抛的是
        # ``SamplingError``；恢复的调用方要的是 ``TrainingError``，所以在这里
        # 转一次，并保留它给出的**实际原因**（不把所有不兼容都说成"日程错误"）。
        try:
            self.sampler.validate_rng_state(payload["sampler_rng"])
        except Exception as exc:
            raise TrainingError(f"the checkpoint's sampler RNG state is not usable: {exc}") from exc

        stage_payload = payload.get("stage")
        if not isinstance(stage_payload, Mapping):
            raise TrainingError(
                "the checkpoint carries no 'stage' block; the training stage, its time "
                "window and any pending stage verdict cannot be reconstructed"
            )
        for key in ("current", "window_fraction", "pending", "blocked"):
            if key not in stage_payload:
                raise TrainingError(
                    f"the checkpoint's 'stage' block has no {key!r}. A missing record is "
                    "not a pass: an older writer that omitted 'blocked' would be read as "
                    "'no pending verdict', which is exactly the state this field exists "
                    "to distinguish"
                )
        stored_step = int(payload["step"])
        stored_window = stage_payload["window_fraction"]
        if not isinstance(stored_window, (int, float, np.floating)) or not np.isfinite(
            float(stored_window)
        ):
            raise TrainingError(
                f"the checkpoint's stage window is {stored_window!r}, not a finite number"
            )
        recorded_stage = stage_payload["current"]
        if recorded_stage is not None and recorded_stage not in {
            stage.name for stage in self.schedule
        }:
            raise TrainingError(
                f"the checkpoint was written in stage {recorded_stage!r}, which this "
                "schedule does not define; refusing rather than guessing a stage"
            )
        # 允许的窗口**不是**"按步数算出的、或者任何一个记录过的阶段窗口"——那太松
        # 了：任何阶段窗口都会被接受一次，等于把"窗口对不上"这件事放过去。真正的
        # 一致性条件按**阻塞与否**分成两条：
        #
        # * 未阻塞：步数已经推进到下一阶段，窗口必须就是该步数所属阶段的窗口；
        # * 已阻塞：步数停在边界上、阶段没切，窗口必须是**那个未切走的阶段**的窗口，
        #   而且被阻塞的阶段必须正是它。
        #
        # 两条都是精确等值，不是"属于某个集合"。
        blocked = bool(stage_payload["blocked"])
        pending = stage_payload["pending"]
        stage_window_of = {
            stage.name: float(stage.time_window_fraction) for stage in self.schedule
        }
        if blocked:
            if pending not in stage_window_of:
                raise TrainingError(
                    f"the checkpoint is marked blocked on stage {pending!r}, which this "
                    "schedule does not define"
                )
            if recorded_stage != pending:
                raise TrainingError(
                    f"the checkpoint is blocked on {pending!r} but records its current "
                    f"stage as {recorded_stage!r}; a blocked checkpoint must still be "
                    "sitting in the stage whose criteria failed"
                )
            expected_window = stage_window_of[recorded_stage]
        else:
            if pending is not None:
                raise TrainingError(
                    f"the checkpoint records a pending stage {pending!r} but is not "
                    "marked blocked; the two fields must agree"
                )
            expected_window = time_window_fraction_at(stored_step, self.schedule)
        if abs(float(stored_window) - float(expected_window)) > 1.0e-12:
            raise TrainingError(
                f"the checkpoint's stage window is {stored_window}, but "
                f"{'the blocked stage' if blocked else 'step'} "
                f"{recorded_stage if blocked else stored_step} of this schedule requires "
                f"{expected_window}; this is not the same schedule"
            )

        # ---- 第二阶段：真正提交。任何一步失败都要把对象还原回去 --------------
        #
        # "先把几个 if 挪到前面"不等于事务性：模型/优化器的 ``load_state_dict``
        # 本身就可能在中途抛错（键不匹配、形状不符、优化器状态里引用了别的参数
        # 组），那时前面的检查全过了也没用。所以这里对**所有**受影响的状态做快照，
        # 提交过程中任何异常都完整回滚——包括全局随机状态，它同样是被改过的。
        snapshot = self._snapshot_state()
        try:
            self._commit_checkpoint(payload, stored_window, recorded_stage, stage_payload)
        except Exception:
            self._restore_state(snapshot)
            raise

    # -- 恢复的提交与回滚（提示词 §3）--------------------------------------

    def _snapshot_state(self) -> dict[str, Any]:
        """把所有会被 ``load`` 改到的状态拷一份。

        覆盖：模型、优化器、步数、历史、best、阶段/窗口/待决/阻塞、平衡器、
        各采样器（含窗口与 RNG）、全局随机状态。
        """
        import copy as _copy

        return {
            "model": {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            },
            "optimizer": _copy.deepcopy(self.optimizer.state_dict()),
            "step": int(self.step),
            "history": _copy.deepcopy(self.history),
            "check_history": _copy.deepcopy(self.check_history),
            "best": _copy.deepcopy(self.best),
            "current_stage": self._current_stage,
            "stage_window": float(self._stage_window),
            "pending_stage": self._pending_stage,
            "stage_blocked": bool(self._stage_blocked),
            "stage_history": _copy.deepcopy(self.stage_history),
            "balancer": _copy.deepcopy(self.balancer.state_dict()),
            "rng": _snapshot_global_rng(),
            "samplers": [
                {
                    "sampler": sampler,
                    "window": float(sampler.window_fraction),
                    "rng": _copy.deepcopy(sampler.rng_state()),
                }
                for sampler in self.sampler.samplers
            ],
            "last_update_loss": float(self._last_update_loss),
            "last_update_loss_step": int(self._last_update_loss_step),
            "final_status": getattr(self, "_final_status", None),
            "final_verdict": getattr(self, "_final_stage_verdict", None),
        }

    def _restore_state(self, snapshot: Mapping[str, Any]) -> None:
        self.model.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.step = int(snapshot["step"])
        self.history = list(snapshot["history"])
        self.check_history = list(snapshot["check_history"])
        self.best = snapshot["best"]
        self._current_stage = snapshot["current_stage"]
        self._stage_window = float(snapshot["stage_window"])
        self._pending_stage = snapshot["pending_stage"]
        self._stage_blocked = bool(snapshot["stage_blocked"])
        self.stage_history = list(snapshot["stage_history"])
        self.balancer.load_state_dict(snapshot["balancer"])
        _restore_global_rng(snapshot["rng"])
        for entry in snapshot["samplers"]:
            sampler = entry["sampler"]
            # 回滚要能**收窄**窗口，所以走 restore_ 而不是 set_。
            sampler.restore_time_window(entry["window"])
            sampler.set_rng_state(entry["rng"])
        self._last_update_loss = float(snapshot["last_update_loss"])
        self._last_update_loss_step = int(snapshot["last_update_loss_step"])
        self._final_status = snapshot["final_status"]
        self._final_stage_verdict = snapshot["final_verdict"]

    def _commit_checkpoint(
        self,
        payload: Mapping[str, Any],
        stored_window: float,
        recorded_stage: str | None,
        stage_payload: Mapping[str, Any],
    ) -> None:
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])
        self.history = list(payload.get("history", []))
        self.check_history = list(payload.get("checks", []))
        self.best = payload.get("best_check")
        _restore_rng(payload.get("rng"))
        self.sampler.set_rng_state(payload["sampler_rng"])

        # 恢复**当时实际在用**的窗口，而不是按步数另算一个。两者在正常路径上
        # 相等（上面的核对保证了这一点），在"判据未过、尚未推进"那一档上前者
        # 才是对的：那时步数已在边界上，实际用的还是上一阶段的窗口。
        self.sampler.set_time_window(float(stored_window))
        self._stage_window = float(stored_window)
        self._current_stage = recorded_stage
        self._pending_stage = stage_payload["pending"]
        self._stage_blocked = bool(stage_payload["blocked"])
        self._final_status = payload.get("run_status")
        self._final_stage_verdict = payload.get("final_stage_verdict")
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


def _snapshot_global_rng() -> dict[str, Any]:
    """当前全局随机状态的快照，供回滚用。

    与 :func:`_rng_state` 的区别是它**不绑定 seed 语义**：回滚要还原的是"现在
    这一刻"的状态，不是"某一次运行开始时的"。
    """
    import random

    state: dict[str, Any] = {"python": None, "numpy": None, "torch": None}
    state["python"] = random.getstate()
    state["numpy"] = np.random.get_state()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
    return state


def _restore_global_rng(payload: Mapping[str, Any] | None) -> None:
    """还原 :func:`_snapshot_global_rng` 的快照。"""
    if not payload:  # pragma: no cover - defensive
        return
    import random

    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if torch is not None and payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])


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
