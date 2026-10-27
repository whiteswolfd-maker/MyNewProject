"""训练循环（新启动场，单半径与多半径共用同一套逻辑）。

* **每次参数更新，所有半径都参与且等权**：每步为每个半径各采一个 batch，算出
  一个联合损失，只做**一次**反向传播，再更新**一个共享优化器**。
* **共享一套模型参数**：ProductNet、AirNet 与前沿网络只有一份；每个半径的物理
  输出用该 case 的 ``batch.scales`` 换算。
* **RAR** 只在触发步更新每个 case/材料自己的 PDE 点池，不调用优化器、不写梯度；
  调用顺序是 RAR → 采样 → 联合损失 → 一次优化器更新。
* **开发集**：训练监督只采训练帧；固定检查集另建开发帧数据批次，只用于
  data_loss 和既有选模偏序。
* **恢复先核对再修改**：检查点绑定完整 case 集合、采样/窗口/RNG、连续帧组划分、
  RAR 点池与配置、阶段、平衡器、选模记录；任何一项不同都拒绝。
* **诊断运行**（``diagnostic=True``）仍记录诊断身份和步数上限，但它不再被用来
  替代已取消的科研资格门槛。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..evaluation.checks import CheckReport, CheckSet, build_check_set, run_checks
from ..losses.objective import (
    BASE_WEIGHTS,
    LossBreakdown,
    total_loss_for_cases,
)
from ..sampling.sampler import SamplingConfig
from .balancing import (
    BalanceConfig,
    BoundedGradientBalancer,
    live_groups,
    measure_gradient_norms,
)
from .checkpoint import (
    TrainingError,
    collect_rng_state,
    load_training_checkpoint,
    load_startup_checkpoint,
    parameter_fingerprint_of,
    save_training_checkpoint,
)
from .schedule import (
    MAX_LBFGS_ITERATIONS,
    SCHEDULE,
    Stage,
    criteria_for,
    describe_schedule,
    learning_rate_at,
    pde_weight_at,
)
from .startup_cases import StartupCase

RUN_COMPLETED = "completed"
RUN_CRITERIA_FAILED = "criteria_failed"
RUN_INCOMPLETE = "incomplete"
RUN_DIAGNOSTIC = "diagnostic"

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

#: 诊断运行的默认步数上限。命令行可以调低，不能调高到"用诊断模式跑正式日程"。
DEFAULT_DIAGNOSTIC_MAX_UPDATES = 100


class NonFiniteUpdate(TrainingError):
    """一次参数更新在**应用之前**被拦下：损失或梯度不是有限值。"""

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
        self.detail = str(detail)
        lines = [f"update {self.step} was blocked: {detail}"]
        if bad_parameters:
            lines.append(f"  non-finite gradient in: {list(bad_parameters)[:8]}")
        lines.append(
            "  the parameters were NOT updated, so the last good checkpoint is intact; "
            "no NaN was papered over with nan_to_num or a gradient filter"
        )
        super().__init__("\n".join(lines))


class StageNotPassed(TrainingError):
    """阶段预算耗尽，且预先声明的判据未通过。"""

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
    output_dir: Path
    updates: int
    final_loss: float
    status: str = RUN_DIAGNOSTIC
    final_loss_step: int = 0
    final_stage_verdict: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == RUN_COMPLETED

    @property
    def formally_accepted(self) -> bool:
        return self.status == RUN_COMPLETED

    def as_dict(self) -> dict[str, Any]:
        metadata = self.metadata or {}
        return {
            "output_dir": str(self.output_dir),
            "updates": self.updates,
            "status": self.status,
            "status_meaning": RUN_STATUS_MEANING.get(self.status, "unknown status"),
            "formally_accepted": self.formally_accepted,
            "final_stage_verdict_passed": bool(
                (self.final_stage_verdict or {}).get("passed") is True
            ),
            "final_loss": self.final_loss,
            "final_loss_step": int(self.final_loss_step),
            "final_stage_verdict": self.final_stage_verdict,
            "diagnostic": bool(metadata.get("diagnostic", False)),
            "checkpoints": self.checkpoints,
            "history": list(self.history),
            "history_length": len(self.history),
            "metadata": metadata,
        }


class Trainer:
    """把一个或多个 :class:`StartupCase` 接到日程、检查与检查点上。"""

    def __init__(
        self,
        *,
        cases: Sequence[StartupCase],
        model: Any,
        schedule: Sequence[Stage] = SCHEDULE,
        seed: int = 0,
        output_dir: Path | str | None = None,
        log_every: int = 50,
        checkpoint_every: int = 500,
        check_every: int = 500,
        gradient_clip: float = 1.0,
        tolerances: Mapping[str, float] | None = None,
        diagnostic: bool = False,
        diagnostic_max_updates: int = DEFAULT_DIAGNOSTIC_MAX_UPDATES,
        config: Any = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        if not cases:
            raise TrainingError("training needs at least one startup case")
        ids = [case.case_id for case in cases]
        if len(set(ids)) != len(ids):
            raise TrainingError(f"duplicate case ids: {ids}")
        for case in cases:
            if not getattr(case, "field", None).checks.get("passed") is True:
                raise TrainingError(
                    f"case {case.case_id!r}: the startup field did not pass its checks"
                )
        schedule = tuple(schedule)
        if not schedule:
            raise TrainingError("the training schedule is empty")
        self.cases = list(cases)
        if model is None:
            raise TrainingError("shared training needs one CoupledModel")
        self.model = model
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate_at(0, schedule),
        )
        self.schedule = schedule
        self.seed = int(seed)
        self.output_dir = Path(output_dir or "outputs/run")
        self.log_every = max(1, int(log_every))
        self.checkpoint_every = max(0, int(checkpoint_every))
        self.check_every = max(0, int(check_every))
        self.gradient_clip = float(gradient_clip)
        self.tolerances = dict(tolerances or {})
        self.config = config
        self.diagnostic = bool(diagnostic)
        self.diagnostic_max_updates = int(diagnostic_max_updates)
        if self.diagnostic_max_updates <= 0:
            raise TrainingError("diagnostic_max_updates must be positive")

        self.step = 0
        self.history: list[dict[str, Any]] = []
        self.stage_history: list[dict[str, Any]] = []
        self.check_history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        #: 入选那一刻的**共享参数**。参数只有一份，因此最佳权重也只有一份。
        self._best_model: dict[str, Any] | None = None
        self._final_status: str | None = None
        self._final_stage_verdict: dict[str, Any] | None = None
        self._last_update_loss = float("nan")
        self._last_update_loss_step = 0
        self._force_final_check = False
        self._run_origin = {
            "explicit_diagnostic": bool(diagnostic),
            "refined": False,
        }
        self._current_stage: str | None = None
        self._stage_window = float(self.schedule[0].time_window_fraction)
        self._stage_blocked = False
        self._pending_stage: str | None = None
        self._check_sets: dict[str, CheckSet] | None = None
        self._check_set_identity: str | None = None

        self.balance_config = (
            BalanceConfig.from_config(config) if config is not None else BalanceConfig()
        )
        self.balancer = BoundedGradientBalancer(BASE_WEIGHTS, self.balance_config)
        self.last_balance: dict[str, Any] | None = None

        for case in self.cases:
            case.sampler.set_time_window(float(self.schedule[0].time_window_fraction))
        self._set_learning_rate(learning_rate_at(0, self.schedule))

    # ------------------------------------------------------------------
    # 基本状态
    # ------------------------------------------------------------------
    @property
    def total_schedule_updates(self) -> int:
        return sum(stage.updates for stage in self.schedule)

    def _all_parameters(self) -> list[Any]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    def parameter_fingerprint(self) -> str:
        return parameter_fingerprint_of(self.model.state_dict())

    def _diagnostic_payload(self) -> dict[str, Any]:
        origin = dict(self._run_origin)
        enabled = bool(
            self.diagnostic
            or origin["explicit_diagnostic"]
            or origin["refined"]
        )
        reason = None
        if enabled:
            reasons = []
            if origin["explicit_diagnostic"] or self.diagnostic:
                reasons.append("explicit diagnostic run")
            if origin["refined"]:
                reasons.append("parameters come from an L-BFGS refinement")
            reason = "; ".join(reasons)
        return {
            "enabled": enabled,
            "max_updates": int(self.diagnostic_max_updates) if enabled else None,
            "reason": reason,
        }

    def _training_settings(self) -> dict[str, Any]:
        return {
            "gradient_clip": float(self.gradient_clip),
            "tolerances": dict(self.tolerances),
            "balancing_config": self.balance_config.as_dict(),
            "diagnostic_max_updates": int(self.diagnostic_max_updates),
        }

    def training_state(self) -> dict[str, Any]:
        """影响后续训练的动态状态；随每个检查点一起保存。"""
        return {
            "step": int(self.step),
            "stage": {
                "current": self._current_stage,
                "window_fraction": float(self._stage_window),
                "blocked": bool(self._stage_blocked),
                "pending": self._pending_stage,
            },
            "schedule": [stage.as_dict() for stage in self.schedule],
            "balancing": {
                "config": self.balance_config.as_dict(),
                "state": self.balancer.state_dict(),
            },
            "rng": collect_rng_state(),
            "history": list(self.history),
            "stage_history": list(self.stage_history),
            "check_history": list(self.check_history),
            "best": (
                None if self.best is None else {
                    **dict(self.best),
                    **(
                        {
                            "model_state": {
                                name: value.detach().cpu().clone()
                                for name, value in self._best_model.items()
                            }
                        }
                        if self._best_model else {}
                    ),
                }
            ),
            "run_origin": dict(self._run_origin),
            "diagnostic": self._diagnostic_payload(),
            "settings": self._training_settings(),
            "resumable": True,
        }

    # ------------------------------------------------------------------
    # 一步
    # ------------------------------------------------------------------
    def _update(self) -> LossBreakdown:
        # RAR 只更新内部 PDE 点池；它不调用优化器、不写参数梯度。调用顺序固定为
        # “RAR（若到触发步）→ 每 case 采样 → 联合损失 → 一次优化器更新”。检查点
        # 在更新完成后保存，所以 last_update_step 足以防止恢复时重复/遗漏一次更新。
        for case in self.cases:
            case.sampler.maybe_update_rar(self.model, self.step)
        batches = [case.sampler.batch() for case in self.cases]
        contact_pressures = [case.contact_pressure for case in self.cases]
        pde_weight = pde_weight_at(self.step, self.schedule)
        breakdown = total_loss_for_cases(
            batches,
            self.model,
            contact_pressures=contact_pressures,
            pde_weight=pde_weight,
            weights=self.balancer.weights(),
        )
        parameters = self._all_parameters()
        if self.balancer.should_update(self.step):
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
        if not bool(torch.isfinite(breakdown.total)):
            raise NonFiniteUpdate(
                step=self.step + 1,
                loss=float("nan"),
                bad_parameters=[],
                detail="the total loss is not finite",
            )
        breakdown.total.backward()

        bad_parameters = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if bad_parameters:
            self.optimizer.zero_grad(set_to_none=True)
            raise NonFiniteUpdate(
                step=self.step + 1,
                loss=float(breakdown.total.detach()),
                bad_parameters=bad_parameters,
                detail="the gradient is not finite for these parameters",
            )

        if self.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.gradient_clip)
        # **只有一次**优化器更新：所有半径的损失已在同一个联合目标里，共享参数
        # 在 optimizer 中只出现一次。
        self.optimizer.step()
        self._last_update_loss = float(breakdown.total.detach())
        return breakdown

    # ------------------------------------------------------------------
    # 固定检查集
    # ------------------------------------------------------------------
    def _ensure_check_sets(self) -> dict[str, CheckSet]:
        if self._check_sets is not None:
            return self._check_sets
        check_sets: dict[str, CheckSet] = {}
        for index, case in enumerate(self.cases):
            base = case.sampler.config
            # 物理检查批次不含数据标签；开发标签另建一个固定批次，只用于
            # data_loss 和既有选模偏序，不进入 PDE 或 RAR。
            config = replace(
                base,
                n_data_product=0,
                n_data_air=0,
                n_data_contact=0,
                n_data_shock=0,
            )
            check_sets[case.case_id] = build_check_set(
                field=case.field,
                scales=case.scales,
                tau_end=case.tau_end,
                config=config,
                seed=self.seed + 50000 + index,
                dtype=self.model.product_net.dtype_,
                device=next(self.model.parameters()).device,
                r_min_fraction=float(
                    getattr(case.sampler, "r_min_fraction", 1.0e-2)
                ),
                case_id=case.case_id,
                label=f"check:{case.case_id}",
                development_case=case.data,
                development_config=base,
            )
        self._check_sets = check_sets
        import hashlib

        payload = json.dumps(
            {cid: cs.as_dict() for cid, cs in check_sets.items()},
            sort_keys=True,
            default=str,
        )
        self._check_set_identity = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:16]
        return check_sets

    def _run_checks(
        self, *, window_fraction: float | None = None, total_loss_value: float | None = None
    ) -> tuple[CheckReport, dict[str, Any]]:
        check_sets = self._ensure_check_sets()
        pde_weight = pde_weight_at(self.step, self.schedule)
        per_case: dict[str, dict[str, float]] = {}
        metrics: dict[str, float] = {}
        conservation: dict[str, Any] = {}
        nonphysical = 0.0
        physics_excess = 0.0
        for case in self.cases:
            check_set = check_sets[case.case_id]
            if window_fraction is not None:
                check_set = check_set.windowed(float(window_fraction))
            report = run_checks(
                self.model,
                check_set,
                pde_weight=pde_weight,
                total_loss=total_loss_value,
                tolerances=self.tolerances,
                steps=self.step,
            )
            entry = report.per_radius.get(case.case_id, {})
            per_case[case.case_id] = entry
            for key, value in entry.items():
                metrics[f"{case.case_id}.{key}"] = value
            nonphysical += float(report.metrics.get("nonphysical_violation_count", 0.0))
            physics_excess += float(report.metrics.get("physics_violation_score", 0.0))
            conservation[case.case_id] = report.conservation

        pde_values = [
            value
            for entry in per_case.values()
            for key, value in entry.items()
            if key.startswith("pde_") and key.endswith("_rms")
        ]
        metrics["pde_rms_max"] = max(pde_values, default=float("nan"))
        for aggregate, source in (
            ("ic_fields", "ic_fields"),
            ("ic_gap_relative_error", "ic_gap_relative_error"),
        ):
            values = [
                float(entry[source])
                for entry in per_case.values()
                if source in entry and math.isfinite(float(entry[source]))
            ]
            metrics[aggregate] = float(np.mean(values)) if values else float("nan")
        development = [
            float(entry["data_development_rms"])
            for entry in per_case.values()
            if "data_development_rms" in entry
            and math.isfinite(float(entry["data_development_rms"]))
        ]
        metrics["development_prediction_error"] = (
            float(np.mean(development)) if development else float("nan")
        )
        metrics["nonphysical_violation_count"] = float(nonphysical)
        metrics["physics_violation_score"] = float(physics_excess)
        metrics["total_loss"] = float(
            total_loss_value if total_loss_value is not None else float("nan")
        )
        metrics["pde_weight"] = float(pde_weight)
        report = CheckReport(
            step=int(self.step),
            metrics=metrics,
            per_radius=per_case,
            conservation=conservation,
        )
        record = report.as_dict()
        if window_fraction is not None:
            record["window_fraction"] = float(window_fraction)
        record["parameter_fingerprint"] = self.parameter_fingerprint()
        record["scope"] = "full" if window_fraction is None else "windowed"
        record["check_set_identity"] = self._ensure_check_sets_identity()
        return report, record

    def _ensure_check_sets_identity(self) -> str:
        if self._check_set_identity is None:
            self._ensure_check_sets()
        return str(self._check_set_identity)

    def _check_stale(self) -> bool:
        if not self.check_history:
            return True
        last = self.check_history[-1]
        if int(last.get("step", -1)) != int(self.step):
            return True
        if last.get("parameter_fingerprint") != self.parameter_fingerprint():
            return True
        if last.get("scope") != "full":
            return True
        return last.get("check_set_identity") != self._ensure_check_sets_identity()

    def verify(
        self,
        *,
        verbose: bool = False,
        total_loss_value: float | None = None,
        window_fraction: float | None = None,
        update_best: bool = True,
    ) -> CheckReport:
        report, record = self._run_checks(
            window_fraction=window_fraction, total_loss_value=total_loss_value
        )
        self.check_history.append(record)
        if not update_best:
            return report
        score = tuple(float(value) for value in report.score)
        if self.best is None or score < tuple(self.best["score"]):
            self.best = {
                "step": int(self.step),
                "score": list(score),
                "parameter_fingerprint": self.parameter_fingerprint(),
                "check_set_identity": self._ensure_check_sets_identity(),
                "admissible": bool(report.admissible),
            }
            self._best_model = {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            }
            if verbose:
                print(
                    f"  [check] step {self.step}: new best  "
                    f"nonphysical={report.metrics['nonphysical_violation_count']:.3g}  "
                    f"pde_rms_max={report.metrics['pde_rms_max']:.3e}"
                )
        elif verbose:
            print(f"  [check] step {self.step}: kept best")
        return report

    # ------------------------------------------------------------------
    # 阶段机
    # ------------------------------------------------------------------
    def _enter_stage(
        self, stage: Stage, *, previous: str | None, verbose: bool, evaluate: bool
    ) -> None:
        if previous is not None and evaluate:
            self._evaluate_stage_criteria(
                previous,
                window_fraction=self._stage_window,
                fatal=True,
                verbose=verbose,
            )
        for case in self.cases:
            case.sampler.set_time_window(float(stage.time_window_fraction))
        self._stage_window = float(stage.time_window_fraction)
        self._current_stage = stage.name
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
        if self.check_every <= 0:
            record = {
                "event": "criteria",
                "stage": stage_name,
                "step": int(self.step),
                "passed": False,
                "criteria": [],
                "reason": (
                    "check_every <= 0: no acceptance check was executed, so a stage "
                    "criterion cannot be evaluated"
                ),
            }
            self.stage_history.append(record)
            if fatal:
                self._stage_blocked = True
                self._pending_stage = stage_name
                raise StageNotPassed(
                    stage_name,
                    [{
                        "metric": "checks", "sense": "min", "value": float("nan"),
                        "limit": 1.0, "passed": False,
                        "description": "no acceptance checks were enabled",
                    }],
                )
            return record
        report = self.verify(
            verbose=False, window_fraction=window_fraction, update_best=False
        )
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for criterion in criteria_for(stage_name):
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
            print(
                f"[stage] {stage_name} criteria: "
                f"{'passed' if not failures else 'NOT PASSED'}"
            )
        if failures and fatal:
            self._stage_blocked = True
            self._pending_stage = stage_name
            raise StageNotPassed(stage_name, failures)
        return record

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _stage_index(self, step: int) -> int:
        remaining = int(step)
        for index, stage in enumerate(self.schedule):
            if remaining < stage.updates:
                return index
            remaining -= stage.updates
        return len(self.schedule) - 1

    def run(
        self,
        *,
        max_updates: int | None = None,
        verbose: bool = False,
        diagnostic: bool = False,
    ) -> TrainingResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if diagnostic:
            self._run_origin["explicit_diagnostic"] = True
            self.diagnostic = True
        origin_diagnostic = bool(
            self._run_origin["explicit_diagnostic"] or self._run_origin["refined"]
        )
        schedule_total = self.total_schedule_updates
        if self.diagnostic:
            cap = int(self.diagnostic_max_updates)
            if max_updates is not None and int(max_updates) > cap:
                raise TrainingError(
                    f"diagnostic runs are capped at {cap} updates; got "
                    f"max_updates={max_updates}. Use formal training for longer runs."
                )
            budget = cap if max_updates is None else int(max_updates)
        else:
            budget = schedule_total if max_updates is None else int(max_updates)
        if int(budget) < 0:
            raise TrainingError(f"max_updates must be non-negative, got {budget!r}")
        total = min(int(budget), schedule_total)
        if origin_diagnostic:
            total = min(total, int(self.diagnostic_max_updates))

        if self._stage_blocked and self._pending_stage is not None:
            self._evaluate_stage_criteria(
                self._pending_stage,
                window_fraction=self._stage_window,
                fatal=True,
                verbose=verbose,
            )

        parameters_at_entry = self.parameter_fingerprint()
        force_final_check = bool(self._force_final_check)
        self._force_final_check = False
        stage_name = self._current_stage or ""
        stage_ends: dict[str, int] = {}
        cursor = 0
        for stage in self.schedule:
            cursor += stage.updates
            stage_ends[stage.name] = cursor

        while self.step < total:
            index = self._stage_index(self.step)
            current = self.schedule[index]
            if current.name != stage_name:
                self._enter_stage(
                    current,
                    previous=stage_name or None,
                    verbose=verbose,
                    evaluate=not origin_diagnostic,
                )
                stage_name = current.name
            self._set_learning_rate(learning_rate_at(self.step, self.schedule))
            breakdown = self._update()
            self.step += 1
            self._last_update_loss_step = self.step

            if stage_ends.get(stage_name) == self.step:
                position = next(
                    (i for i, stage in enumerate(self.schedule) if stage.name == stage_name),
                    None,
                )
                if position is not None and position + 1 < len(self.schedule):
                    following = self.schedule[position + 1]
                    self._enter_stage(
                        following,
                        previous=stage_name,
                        verbose=verbose,
                        evaluate=not origin_diagnostic,
                    )
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
            if self.checkpoint_every > 0 and self.step % self.checkpoint_every == 0:
                self.save(self.output_dir / "last.pt")

        final = float(self._last_update_loss)
        final_step = int(self._last_update_loss_step)
        parameters_moved = (
            self.parameter_fingerprint() != parameters_at_entry or force_final_check
        )
        if (
            self.check_every > 0
            and parameters_moved
            and self._check_stale()
        ):
            self.verify(verbose=verbose, total_loss_value=final)

        truncated = self.step < schedule_total
        if origin_diagnostic:
            status = RUN_DIAGNOSTIC
            verdict = None
        elif truncated:
            status = RUN_INCOMPLETE
            verdict = None
        elif self.check_every <= 0:
            status = RUN_CRITERIA_FAILED
            verdict = {
                "event": "criteria",
                "stage": self._current_stage,
                "step": int(self.step),
                "passed": False,
                "criteria": [],
                "reason": "check_every <= 0, so no acceptance check was executed",
            }
        elif self._current_stage is not None and self._current_stage == self.schedule[-1].name:
            verdict = self._evaluate_stage_criteria(
                self._current_stage,
                window_fraction=self._stage_window,
                fatal=False,
                verbose=verbose,
            )
            status = RUN_COMPLETED if verdict.get("passed") else RUN_CRITERIA_FAILED
        else:
            verdict = {
                "event": "criteria",
                "stage": self._current_stage,
                "step": int(self.step),
                "passed": False,
                "criteria": [],
                "reason": "the schedule ran out but the run is not in its last stage",
            }
            status = RUN_CRITERIA_FAILED

        self._final_status = status
        self._final_stage_verdict = verdict
        # 最终检查点带着结论；中途检查点没有结论（_final_status=None）。
        self.save(self.output_dir / "last.pt")
        (self.output_dir / "checks.json").write_text(
            json.dumps(self.check_history, indent=2, default=_jsonable),
            encoding="utf-8",
        )
        result = TrainingResult(
            output_dir=self.output_dir,
            updates=self.step,
            final_loss=final,
            status=status,
            final_loss_step=final_step,
            final_stage_verdict=verdict,
            history=list(self.history),
            checkpoints={"last": str(self.output_dir / "last.pt")},
            metadata={
                "status": status,
                "status_meaning": RUN_STATUS_MEANING.get(status, ""),
                "diagnostic": origin_diagnostic,
                "run_origin": dict(self._run_origin),
                "diagnostic_payload": self._diagnostic_payload(),
                "schedule_total_updates": int(schedule_total),
                "truncated": bool(truncated),
                "cases": [case.describe() for case in self.cases],
                "schedule": describe_schedule(self.schedule),
                "sampling": {
                    name: getattr(case.sampler.config, name)
                    for case in self.cases[:1]
                    for name in SamplingConfig.__dataclass_fields__
                },
                "balancing": {
                    "config": self.balance_config.as_dict(),
                    "monitors": self.balancer.monitors(),
                },
                "time_windows": {
                    case.case_id: case.sampler.window_state() for case in self.cases
                },
                "checks": self.check_history,
                "best_check": self.best,
                "checks_executed": bool(self.check_history),
                "rar": {
                    "enabled": bool(
                        self.cases[0].sampler.rar_config.enabled
                        if self.cases
                        else False
                    ),
                    "config": (
                        self.cases[0].sampler.rar_config.as_dict()
                        if self.cases
                        else None
                    ),
                    "last_update_step": {
                        case.case_id: case.sampler._rar_last_update_step
                        for case in self.cases
                    },
                    "pool_sizes": {
                        case.case_id: {
                            medium: len(case.sampler.rar_pools[medium])
                            for medium in ("product", "air")
                        }
                        for case in self.cases
                    },
                },
                "wall_seconds": time.time() - started,
            },
        )
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(result.as_dict(), indent=2, default=_jsonable),
            encoding="utf-8",
        )
        return result

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        return save_training_checkpoint(
            path,
            cases=self.cases,
            model=self.model,
            optimizer=self.optimizer,
            training_state=self.training_state(),
            extra={"run_status_at_save": self._final_status},
        )

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        """**续训**：恢复共享模型、优化器、采样器窗口/RNG、阶段与训练状态。"""
        info = load_training_checkpoint(
            path,
            cases=self.cases,
            model=self.model,
            optimizer=self.optimizer,
            schedule=self.schedule,
            expected_training_settings=self._training_settings(),
        )
        state = info["training"]
        self.step = int(state["step"])
        stage = state["stage"]
        self._current_stage = stage["current"]
        self._stage_window = float(stage["window_fraction"])
        self._stage_blocked = bool(stage["blocked"])
        self._pending_stage = stage["pending"]
        self.history = list(state["history"])
        self.stage_history = list(state["stage_history"])
        self.check_history = list(state["check_history"])
        self.best = None if state["best"] is None else dict(state["best"])
        best_state = None if state["best"] is None else state["best"].get("model_state")
        self._best_model = (
            None if not best_state else {
                name: value.detach().cpu().clone()
                for name, value in dict(best_state).items()
            }
        )
        self.balancer.load_state_dict(state["balancing"]["state"])
        self._run_origin = {
            "explicit_diagnostic": bool(state["run_origin"]["explicit_diagnostic"]),
            "refined": bool(state["run_origin"]["refined"]),
        }
        self.diagnostic = bool(
            self._run_origin["explicit_diagnostic"]
            or self._run_origin["refined"]
            or state["diagnostic"].get("enabled")
        )
        cap = state["diagnostic"].get("max_updates")
        if cap is not None:
            self.diagnostic_max_updates = int(cap)
        self._final_status = None
        self._final_stage_verdict = None
        self._last_update_loss = float("nan")
        self._last_update_loss_step = int(self.step)
        self._set_learning_rate(learning_rate_at(self.step, self.schedule))
        return info

    def load_weights(self, path: Path | str) -> dict[str, Any]:
        """**只加载参数**：不恢复优化器、采样窗口、RNG 或训练步号。"""
        return load_startup_checkpoint(
            path,
            cases=self.cases,
            model=self.model,
            schedule=None,
            restore_rng=False,
            weights_only=True,
        )

    # ------------------------------------------------------------------
    # L-BFGS 精修
    # ------------------------------------------------------------------
    def resume_refinement(self, *_: Any, **__: Any) -> TrainingResult:
        raise TrainingError(
            "resuming an interrupted L-BFGS refinement is not supported: L-BFGS "
            "optimiser state and the line-search state are not checkpointed. Load the "
            "refined weights and start a new optimisation stage, or start a new "
            "refinement with refine()"
        )

    def refine(
        self, *, iterations: int = MAX_LBFGS_ITERATIONS, verbose: bool = False
    ) -> TrainingResult:
        """固定采样点的**联合** L-BFGS 精修。

        一次采样（每个半径一个 batch），一个优化器、一组共享参数；线搜索的
        每次求值都评价联合目标。**不**逐 case 精修覆盖——那会让最后一个半径
        覆盖前面半径的结果。
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        if self._stage_blocked:
            raise TrainingError(
                f"refusing to refine while stage {self._pending_stage!r} is blocked"
            )
        iterations = min(int(iterations), MAX_LBFGS_ITERATIONS)
        if iterations <= 0:
            return self.run(max_updates=self.step)
        model_backup = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        window_backup = {
            case.case_id: case.sampler.window_state() for case in self.cases
        }
        sampler_rng_backup = {
            case.case_id: case.sampler.rng_state() for case in self.cases
        }
        global_backup = collect_rng_state()
        history_backup = list(self.history)
        stage_history_backup = list(self.stage_history)
        check_backup = list(self.check_history)
        best_backup = None if self.best is None else dict(self.best)
        try:
            batches = [case.sampler.batch() for case in self.cases]
            contact_pressures = [case.contact_pressure for case in self.cases]
            weights = self.balancer.weights()
            pde_weight = pde_weight_at(self.step, self.schedule)
            optimizer = torch.optim.LBFGS(
                self._all_parameters(),
                max_iter=iterations,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            last: dict[str, float] = {"total": float("nan")}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                breakdown = total_loss_for_cases(
                    batches,
                    self.model,
                    contact_pressures=contact_pressures,
                    pde_weight=pde_weight,
                    weights=weights,
                )
                if not bool(torch.isfinite(breakdown.total)):
                    raise NonFiniteUpdate(
                        step=self.step,
                        loss=float("nan"),
                        bad_parameters=[],
                        detail="the joint refinement loss is not finite",
                    )
                breakdown.total.backward()
                bad = [
                    name
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                if bad:
                    raise NonFiniteUpdate(
                        step=self.step,
                        loss=float(breakdown.total.detach()),
                        bad_parameters=bad,
                        detail="the joint refinement gradient is not finite",
                    )
                last["total"] = float(breakdown.total.detach())
                return breakdown.total

            optimizer.step(closure)
            with torch.enable_grad():
                final = total_loss_for_cases(
                    batches,
                    self.model,
                    contact_pressures=contact_pressures,
                    pde_weight=pde_weight,
                    weights=weights,
                )
            accepted = float(final.total.detach())
            if not math.isfinite(accepted):
                raise NonFiniteUpdate(
                    step=self.step,
                    loss=accepted,
                    bad_parameters=[],
                    detail="the refined joint objective is not finite",
                )
            record = {
                "iterations_requested": int(iterations),
                "loss_on_accepted_parameters": accepted,
                "loss_at_last_closure": last["total"],
                "case_ids": [case.case_id for case in self.cases],
                "contact_pressures_pa": [float(p) for p in contact_pressures],
                "note": "one joint objective on one fixed batch per radius",
            }
            self._run_origin["refined"] = True
            self._final_status = None
            # 精修改了参数但不改步号；收尾必须重新评价当前参数，否则 best/清单
            # 描述的是精修前的模型。
            self._force_final_check = True
            result = self.run(max_updates=self.step, verbose=verbose)
            result.metadata["refinement"] = [record]
            result.metadata["refinement_note"] = (
                "L-BFGS changed the shared parameters without advancing the step; the "
                "run is diagnostic and its objective used one fixed batch per radius "
                "with a single joint update"
            )
            (self.output_dir / "run_manifest.json").write_text(
                json.dumps(result.as_dict(), indent=2, default=_jsonable),
                encoding="utf-8",
            )
            return result
        except BaseException:
            self.model.load_state_dict(model_backup, strict=True)
            for case in self.cases:
                case.sampler.set_window_state(window_backup[case.case_id])
                case.sampler.set_rng_state(sampler_rng_backup[case.case_id])
            import random

            random.setstate(global_backup["python"])
            np.random.set_state(global_backup["numpy"])
            torch.random.set_rng_state(global_backup["torch"])
            self.history = history_backup
            self.stage_history = stage_history_backup
            self.check_history = check_backup
            self.best = best_backup
            raise


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
    "DEFAULT_DIAGNOSTIC_MAX_UPDATES",
    "NonFiniteUpdate",
    "RUN_COMPLETED",
    "RUN_CRITERIA_FAILED",
    "RUN_DIAGNOSTIC",
    "RUN_INCOMPLETE",
    "RUN_STATUS_MEANING",
    "StageNotPassed",
    "Trainer",
    "TrainingError",
    "TrainingResult",
]
