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
import math
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
from ..evaluation.checks import CheckReport, build_check_set, run_checks
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
from .checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    DETONATION_VELOCITY_FIELD,
    PUBLISH_MARKER,
    PUBLISH_OWNER_KEY,
    TrainingError,
    publish_marker_path,
    read_publish_marker,
    refuse_if_uncommitted,
    require_best_candidate_consistency,
    require_detonation_velocity,
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

#: 检查点格式标识、版本号与读取规则现在定义在 :mod:`.checkpoint`——保存端与
#: 读取端（``cli.predict``）共用同一份定义，这里只是重新导出。
#:
#: 放在那儿的原因见该模块的说明：两边各维护一份清单时，"保存端写了什么"与
#: "读取端要求什么"会悄悄脱节。

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
#: 第 8 版起顶层带 ``detonation_velocity``（见 :mod:`.checkpoint`）。


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
        # ``detail`` 也留成属性：精修的失败记录要把"第几次精修、第几次 closure"
        # 结构化地写进诊断文件，而不是让调用方去正则匹配异常消息。
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
        metadata = self.metadata or {}
        refinement = metadata.get("refinement") or []
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
            # **关键身份提到顶层。** 它们本来只在 ``metadata`` 里，读清单的人要往下
            # 翻一层才看得到"这是一次诊断运行、它精修过、当前参数检查过没有"。
            # 交付物之间的一致性正是靠这几个字段核对的，藏起来只会让人漏看。
            "diagnostic": bool(metadata.get("diagnostic", False)),
            "refined": bool(refinement),
            "refinement_count": len(refinement),
            "check_state": metadata.get("check_state"),
            "checkpoints": self.checkpoints,
            "metadata": metadata,
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
        # **"这一步检查过了没有"由参数指纹判定**（``_check_stale`` 每次现算），
        # 不缓存成实例属性：缓存下来的指纹在精修改参数之后会悄悄过期，而
        # ``_check_fingerprint = None`` 这种"作废"手法一旦漏掉一处就再也发现不了。
        # 早先的两个字段（``_parameter_fingerprint`` / ``_check_fingerprint``）只写
        # 不读，已删除——它们让事务审计看起来漏了两项状态。
        #: 精修记录。精修**恢复不了**：L-BFGS 的优化器状态不进检查点，所以
        #: "从检查点继续一次精修"是不支持的，这里如实记下来而不是假装支持。
        self.refine_history: list[dict[str, Any]] = []
        self._lbfgs_closures = 0
        #: 一次回滚失败的记录。非 ``None`` 时这个对象处于未知状态，任何会读取或
        #: 写出训练状态的入口都拒绝工作——继续用它是把这个混合状态当成正常结果。
        self._unusable: str | None = None
        #: 收尾时对"当前参数检查过没有"的判定，随检查点一起落盘。
        self._final_check_state: str | None = None
        #: 模型**来源**：这条参数是怎么来的。"本次调用是不是诊断"是一个问题，
        #: "这份参数是不是一条单一 Adam 轨迹的产物"是另一个问题——后者要跟着
        #: 参数走，随检查点一起存、一起恢复。
        #:
        #: 以前只把它当成本次调用的参数：``diagnostic=True`` 跑完存盘，新实例加载后
        #: ``_final_status`` 是 ``diagnostic``，但一次零更新的 ``run()`` 会把状态
        #: 重新算成 ``completed``——精修身份被继承了，显式诊断身份没有。
        #:
        #: 刻意**不**持久化 ``_final_status`` 本身：阶段失败、预算截断这些状态必须
        #: 在恢复之后**重新评价**，把旧结论搬过来会让"重新跑一遍"变成"复述上次的
        #: 结论"。这里只记**为什么**它是诊断性的。
        self._run_origin: dict[str, bool] = {
            "explicit_diagnostic": False,
            "refined": False,
        }
        #: ``best`` 被更新但还没落盘。选出最佳候选与写出最佳检查点是两件事：
        #: 验证函数在跑的过程中并不知道本次运行的终态，提前写盘会让新参数配上
        #: 上一次的状态（复核实测：``best.pt`` 写 ``completed``，``last.pt`` 写
        #: ``diagnostic``，两份检查点的参数却相同）。
        self._best_dirty = False
        #: **入选那一刻的状态快照**（模型、优化器、步号、RNG、采样器、平衡器、
        #: 阶段……）。只记 ``self.best`` 这个记录是不够的：它只有分数与指纹，
        #: 而发布 ``best.pt`` 时通用 ``_save()`` 读的是**当前**模型——继续训练到
        #: 第 6 步之后，第 1 步选出的最佳会被写成一个第 6 步的检查点，且
        #: ``best_check.parameter_fingerprint`` 与文件里的参数对不上（复核实测）。
        self._best_candidate: dict[str, Any] | None = None
        #: 本次**发布事务**的归属标识，由 :meth:`_begin_publish` 铸造。
        #: :meth:`_end_publish` 只清理属于它的那一个标记。刻意**不进**
        #: ``_snapshot_state``：它是事务机制，不是训练状态——回滚时若把它一并还原，
        #: 回滚方就再也认不出"刚才是自己写的标记"，于是要么删错、要么删不掉。
        self._publish_token: str | None = None
        #: 本次事务开始时目录里已经存在的未完成发布（若有）。``run()`` 会重写全部
        #: 产物，覆盖它是合法的，但这件事要留下痕迹，不能假装没发生过。
        self._superseded_publish: dict[str, Any] | None = None

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

    def _assert_usable(self) -> None:
        """一次回滚失败之后，这个对象不能再被使用。"""
        if self._unusable is not None:
            raise TrainingError(
                "this trainer instance is in an unknown state and cannot be used: "
                f"{self._unusable}"
            )

    def _frozen_detonation_velocity(self) -> float:
        """本次训练**实际使用**的卡片爆速 ``D``，从冻结的尺度集合里取。

        取尺度集合、不取 ``cases[0]``：尺度集合是构造时就冻结下来的那份，
        它才是网络与时间轴真正用的东西，而 ``cases`` 的元数据只是来源记录。

        **四个算例必须给出同一个 ``D``。** 它们共用同一套尺度化网络，只有一个
        ``D`` 时"预测端拿一个 ``D`` 重建 ``T_h``"才成立；不一致时这里硬失败，
        而不是悄悄挑一个——挑错的后果是整条预测轨迹平移一个爆轰穿越时间，而
        两个 ``D`` 都有 m/s 的量纲，错了不会报错。
        """
        values = {float(s.detonation_velocity) for s in self.scales_by_radius.values()}
        if len(values) != 1:
            raise TrainingError(
                "the training scales carry more than one detonation velocity "
                f"({sorted(values)}); a single checkpoint cannot describe what time "
                "origin to rebuild, and picking one silently would shift the predicted "
                "trajectory by R_0/D"
            )
        velocity = values.pop()
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise TrainingError(
                "the training scales carry detonation velocity "
                f"{velocity!r}; it must be finite and positive because the scheme-B "
                "time origin is T_h = R_0/D + epsilon"
            )
        return velocity

    # -- 清单 -------------------------------------------------------------
    def _data_identity(self) -> dict[str, Any]:
        """这批数据的**内容**身份，用来回答"是不是同一批数据"。

        刻意不包含目录字符串：把输出目录搬到别处是正常操作，不该让检查点作废；
        而把同一路径下的数据换掉必须让它作废。所以按时间轴、径向网格、两条前沿
        与三种材料的计数取哈希——它们是这个训练真正依赖的东西，且都是小数组。
        """
        import hashlib

        identity: dict[str, Any] = {}
        for case in self.cases:
            digest = hashlib.sha256()
            # **真正参与训练的东西都要进去。** 以前只哈希了时间轴、径向坐标、两条
            # 前沿，以及材料码的**计数**。把密度乘 2、速度加 123，计数一字不变，
            # 哈希也就不变——旧检查点照常加载（复核实测）。计数相同也证明不了逐点
            # 材料分布相同：把产物与空气的单元整体对调，计数完全一样。
            for name in (
                "times", "radius",
                "density", "velocity", "pressure", "specific_energy",
                "contact_radius", "shock_radius",
                "material", "valid", "contact_valid", "shock_valid",
            ):
                _hash_field(digest, name, getattr(case, name, None))

            validity = getattr(case, "field_validity", None)
            if validity is None:
                digest.update(b"field_validity|missing")   # 与"空数组"分开
            else:
                for key in sorted(validity):
                    _hash_field(digest, f"field_validity[{key}]", validity[key])

            block = getattr(case, "front_supervision_eligibility", None)
            if block is None:
                digest.update(b"front_supervision_eligibility|missing")
            else:
                for key in sorted(block):
                    value = block[key]
                    if isinstance(value, (np.ndarray, list, tuple)) or value is None:
                        _hash_field(digest, f"front_supervision_eligibility[{key}]", value)
                    else:
                        digest.update(
                            f"front_supervision_eligibility[{key}]|{value!r}".encode("utf-8")
                        )

            identity[case.case_id] = {
                "charge_radius_m": float(case.charge_radius),
                "frames": int(np.asarray(case.times).size),
                "radial_points": int(np.asarray(case.radius).size),
                "content_sha256": digest.hexdigest(),
                # 时间原点所需的量也在身份里：少了爆速，同一条时间轴会被读成
                # 另一套时间约定。
                "detonation_velocity_m_s": case.source.get("detonation_velocity_m_s"),
                "symmetry_factor": case.source.get("symmetry_factor"),
                "profile_layout": (
                    case.source.get("profile_layout")
                    if isinstance(case.source.get("profile_layout"), Mapping)
                    else "unrecorded"
                ),
            }
        return identity

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

    def _check_set_identity(self) -> str:
        """固定检查集的**内容**身份，短哈希。

        检查记录要能回答"这条结论是在哪一套点上得到的"。检查集定义变了（采样
        配置、开发帧、数据）而参数没变时，旧结论不能给新定义背书。
        """
        import hashlib

        payload = json.dumps(self.check_set.as_dict(), sort_keys=True, default=_jsonable)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _check_stale(self) -> bool:
        """当前参数下还没有**完整**检查结论吗。

        条件有三条，缺一条就要重查：

        * 从来没查过；
        * 步数变了（``step`` 是报告的一部分）；
        * 参数指纹变了——精修改参数不改步数，只用步数判断会把精修前的结论留下；
        * 上一次是**窗口内**检查、或用的是另一套检查集。窗口检查与全时段检查
          回答的不是同一个问题，不能因为步数和指纹相同就互相顶替。
        """
        if not self.check_history:
            return True
        last = self.check_history[-1]
        if int(last.get("step", -1)) != int(self.step):
            return True
        if last.get("parameter_fingerprint") != self.parameter_fingerprint():
            return True
        if last.get("scope") != "full":
            return True
        return last.get("check_set_identity") != self._check_set_identity()

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
        # **这次检查是在什么范围、什么检查集上做的**，要随记录一起留下。
        # 只记步数与指纹的话，"窗口内的阶段检查"与"全时段的固定检查"看起来是
        # 两条同样的记录，可以互相顶替——而它们回答的不是同一个问题。
        record["scope"] = "full" if window_fraction is None else "windowed"
        record["check_set_identity"] = self._check_set_identity()
        self.check_history.append(record)

        if not update_best:
            return report

        if self.best is None or report.score < tuple(self.best["score"]):
            # **只登记候选，不写盘。** 验证函数在训练过程中被调用，那时本次运行的
            # 终态还没算出来；在这里 ``_save`` 会把新参数配上**上一次**的
            # ``run_status``。
            #
            # 而且**必须把那一刻的状态整个快照下来**：只留记录的话，发布的时候
            # 通用保存路径读的是当前模型，中途选出的最佳会被写成末步的参数。
            # 快照里每一项都是深拷贝／克隆，后续的原地更新改不动它。
            record["run_origin"] = dict(self._run_origin)
            record["refinement_count"] = len(self.refine_history)
            record["check_set_identity"] = self._check_set_identity()
            self.best = record
            self._best_candidate = self._checkpoint_state()
            self._best_dirty = True
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
        refinement: Sequence[Mapping[str, Any]] | None = None,
    ) -> TrainingResult:
        """跑训练循环。

        ``max_updates`` 截断的运行是 ``incomplete``（提示词 §1）：不拿没跑过的
        阶段判失败，也不宣称正式训练完成。``diagnostic=True`` 的运行不主张任何
        验收结论，无论它跑到哪一步。

        ``refinement`` 由 :meth:`refine` 传入：这一批精修记录会成为本次运行的
        **来源身份**。它必须在这里（写任何产物之前）参与终态的判定，否则会出现
        "返回值说 diagnostic、``last.pt`` 里写着 completed"这种自相矛盾的交付——
        以前 ``refine`` 就是把 ``run()`` 的结果拿来之后再改字段，只重写了清单，
        检查点里留在原处的是 ``completed``。
        """
        self._assert_usable()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # **一次正在进行的运行没有终态。** 上一次调用留下的结论如果不清掉，
        # 循环里周期写出的检查点会带着一个旧状态（"跑到一半，文件说完成"）。
        self._final_status = None
        self._final_stage_verdict = None
        self._final_check_state = None
        # 完整日程的长度。收尾判断必须用它，**不能**用截断后的 ``total``。
        schedule_total = sum(stage.updates for stage in self.schedule)
        total = schedule_total
        if max_updates is not None:
            total = min(total, int(max_updates))
        # 模型**来源**里带着精修记录时，这次运行同样是诊断性的：参数不是一条
        # 单一 Adam 轨迹的产物。去掉这一条，"加载精修过的检查点 → 再 run 一次"
        # 就会把精修身份洗掉。
        refinement_records = list(refinement) if refinement is not None else list(
            self.refine_history
        )
        # **来源合并，而不是覆盖。** 本次调用的 ``diagnostic=False`` 是"这次没有
        # 显式要求诊断"，不是"把这份参数的诊断身份清掉"——后者会让"加载显式诊断
        # 运行写出的检查点，再跑一次"变成一个正式完成的结果（复核实测）。
        origin = dict(self._run_origin)
        if refinement_records:
            origin["refined"] = True
        if diagnostic:
            origin["explicit_diagnostic"] = True
        self._run_origin = origin
        origin_diagnostic = bool(origin["explicit_diagnostic"] or origin["refined"])
        refined = bool(origin["refined"])
        if origin_diagnostic:
            reason = [
                name for name, flag in (
                    ("explicitly requested", origin["explicit_diagnostic"]),
                    ("the parameters come from a refinement", origin["refined"]),
                ) if flag
            ]
            diagnostic_reason = "; ".join(reason)
        else:
            diagnostic_reason = None

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
        #: 进入循环那一刻的参数指纹。收尾时用它回答"这次运行**改动过参数**吗"——
        #: 这跟"步数涨了吗"不是同一个问题，精修就是那个反例。
        parameters_at_entry = self.parameter_fingerprint()
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
        #
        # **"这一步检查过没有"不能用"步数有没有增加"代替。** 以前这里写的是
        # ``self.step > start_step``，它把两种完全不同的情形混成了一类：
        #
        # * 零预算的调用（``step == start_step``）：确实什么都不该查——拿一个没
        #   动过的模型去跑完整检查，只会往 ``best.pt`` 里写一份"从未训练过"的快照；
        # * 精修之后的调用（``step`` 一点没动、参数却全变了）：**必须**重查，
        #   否则报告里的误差、best/last 与最终结论对的是一个已经不存在的模型。
        #
        # 正确的判据是"这次运行确实改动过参数，且当前参数下还没有结论"。
        # 指纹比较覆盖第一种（没动过 → 指纹相同 → 不查）；第二种由**调用方**告诉
        # 我们——``refine`` 在进这个方法**之前**就把参数改完了，所以进入时的指纹
        # 已经是精修后的，只比指纹会把精修看成"什么都没发生"。
        parameters_moved = (
            self.parameter_fingerprint() != parameters_at_entry
            or refinement is not None
        )
        checked_this_call = False
        if self.check_every > 0 and parameters_moved and self._check_stale():
            self.verify(verbose=verbose, total_loss_value=final)
            checked_this_call = True
        # 没有在这里重查时，**如实记下是哪一种"没查"**：关掉了检查、这次调用没动
        # 参数、还是当前参数下已经有结论。三者混成一句"没有检查"会让一个"关掉了
        # 检查"的运行看起来与"参数没变、结论仍有效"一样。
        if self.check_every <= 0:
            check_state = "checks_disabled"
        elif checked_this_call:
            check_state = "checked_on_current_parameters"
        elif parameters_moved:
            check_state = "current_parameters_already_carry_a_conclusion"
        else:
            check_state = "no_parameter_change_in_this_call"
        check_state_note = {
            "checks_disabled": (
                "check_every <= 0: no acceptance check was executed, so no conclusion "
                "about these parameters exists"
            ),
            "checked_on_current_parameters": (
                "the full fixed check set was evaluated on the current parameters"
            ),
            "current_parameters_already_carry_a_conclusion": (
                "the parameters changed during this call and already carry a check "
                "conclusion from this call"
            ),
            "no_parameter_change_in_this_call": (
                "this call performed no parameter update; the current parameters were "
                "not re-checked, and any earlier conclusion applies only if its "
                "parameter fingerprint matches the current one"
            ),
        }[check_state]

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

        if origin_diagnostic:
            # **诊断身份由来源决定，不由本次调用决定。** 显式诊断、或参数来自精修，
            # 都让"这条轨迹能不能当正式结果"这个问题的答案是"不能"；而它跟着参数
            # 走——加载之后再跑一次也不会变回正式结果。这一条必须在**写任何产物
            # 之前**生效，返回值、清单、检查点与 CLI 的 ``train_result.json`` 才能
            # 引用同一个结论。
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
        self._final_check_state = check_state
        # **最佳候选推迟到这里才落盘。** ``verify()`` 在循环里就可能选出新的最佳，
        # 但那时本次运行的终态还没算出来——提前写盘会让新参数配上**上一次**的状态
        # （复核实测：``best.pt`` 是 ``completed``、``last.pt`` 是 ``diagnostic``，
        # 而两份的参数完全相同）。保留旧最佳时**不重写** ``best.pt``：
        # 它自己的来源属于选出它的那次运行，不该被这次的状态改写。
        # 多份产物**不是**一次原子操作：先写下未提交标记，发布完再删。进程在
        # 两次写入之间被杀掉时，``load`` 能看到标记并拒绝把半套产物当成正常检查点。
        self._begin_publish(["best.pt", "last.pt", "checks.json", "run_manifest.json"])
        if self._best_dirty and self._best_candidate is not None:
            # **用候选快照发布，不让通用保存路径去读当前模型。** 这份检查点属于
            # 中途的某一步，它的参数、步号、优化器、RNG、采样器与平衡器都必须是
            # 那一刻的。``best_check``（候选自己的检查记录）与它一一对应。
            payload = self._build_payload(
                self._best_candidate,
                run_status=status,
                # 判据是整个**运行**在它自己最后那组参数上评出来的，不是这个中途
                # 候选的。挂在这里会把末步的结论套在中途模型身上。
                final_stage_verdict=None,
                check_state="candidate_selected_from_this_run",
                extra={
                    # ``best.pt`` **就是**已发布的候选，所以它自己不持有"待发布
                    # 候选"——否则同一份状态在文件里存两遍（体积翻倍），而且
                    # 加载它之后会被当成"还有候选没发布"。
                    "pending_best_candidate": None,
                    "best_dirty": False,
                    "candidate": {
                        "step": int(self._best_candidate["step"]),
                        "parameter_fingerprint": self.best.get("parameter_fingerprint"),
                        "selected_in_run_with_status": status,
                        "note": (
                            "this checkpoint holds the best candidate selected during "
                            "the run; the run's acceptance verdict was evaluated on the "
                            "run's final parameters, not on these"
                        ),
                    }
                },
            )
            self._write_payload(payload, self.output_dir / "best.pt")
            self._best_dirty = False
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
                "diagnostic": origin_diagnostic,
                "refined": refined,
                "run_origin": dict(origin),
                "diagnostic_reason": diagnostic_reason,
                "diagnostic_reason_note": (
                    "the model's origin, not just this call: an explicitly diagnostic "
                    "run or a refinement keeps that identity across save, load and "
                    "further runs"
                ),
                # **判据通过 != 正式接受。** 精修过的运行可以是"最终判据确实通过"
                # 同时"整次运行不能当正式结果"，两者都如实保留。
                "final_stage_verdict": stage_verdict,
                "final_stage_verdict_passed": bool(
                    stage_verdict is not None and stage_verdict.get("passed") is True
                ),
                "formally_accepted": status == RUN_COMPLETED,
                "refinement": refinement_records,
                "refinement_note": (
                    "L-BFGS refinement changed the parameters without advancing the "
                    "step, so this run is reported as diagnostic; its objective also "
                    "changed mid-run (one fixed sample batch, adaptive weights and the "
                    "PDE weight frozen at the step where refinement started)"
                    if refined else None
                ),
                "check_state": check_state,
                "check_state_meaning": check_state_note,
                # **本次发布覆盖掉的未完成发布**（若有）。``run()`` 会重写全部四份
                # 产物，所以覆盖它是合法的——但那是"修复了一个不一致的目录"，不是
                # "什么都没发生过"，清单里要看得见。
                "superseded_uncommitted_publish": self._superseded_publish,
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
        # 四份产物都写完了，撤掉未提交标记。放在最后一步：中途崩溃时标记还在，
        # 加载端因此会把这一套产物认成"未提交"。
        self._end_publish()
        return result

    def _checkpoint_paths(self) -> dict[str, str]:
        paths = {"last": str(self.output_dir / "last.pt")}
        best = self.output_dir / "best.pt"
        if best.exists():
            paths["best"] = str(best)
        return paths

    # -- L-BFGS 精修 ------------------------------------------------------
    def resume_refinement(self, *_: Any, **__: Any) -> TrainingResult:
        """**不支持，而且明确拒绝——不要假装支持。**

        三种操作必须分开，它们不是同一件事：

        ================================================  ==========================
        操作                                              支持情况
        ================================================  ==========================
        用精修后的权重做**推理**                          支持（直接加载模型即可）
        从精修后的权重**显式开始一次新的优化阶段**        支持（``run()``；来源身份保留）
        继续一次**中断的** L-BFGS 精修                    **不支持**
        ================================================  ==========================

        L-BFGS 的优化器状态（历史方向与步长、Hessian 的对角近似、上一次的平坦梯度）
        不进检查点，而更强的一点是：**线搜索本身不能在半途停下**。它的每一次求值都
        假设"上一次试探的参数还在原处"，中途接管会得到一个既不是原来那条轨迹、
        也不是一条新轨迹的东西——而损失曲线看起来完全正常。

        所以这里放一个会抛错的方法，而不是只在报告里写一行
        ``resume_supported: false``：后者要靠人去读，前者在调用时就拦住。
        """
        raise TrainingError(
            "resuming an interrupted L-BFGS refinement is not supported: L-BFGS "
            "optimiser state and the line-search state are not checkpointed, and a "
            "line search cannot be stopped halfway and taken over. Load the refined "
            "weights and start a new optimisation stage, or start a new refinement "
            "with refine()"
        )

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

        精修是**事务性**的：动参数之前先取全状态快照，闭包、线搜索或收尾核对
        任何一步失败都把对象还原回精修前的那一刻。L-BFGS 的线搜索会临时改参数，
        所以只在闭包里抛错不足以保证模型还停在原处——必须由调用方负责还原。
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        self._assert_usable()
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

        # **目录里留着未完成的发布时不许开始精修。**
        #
        # 精修的事务语义建立在"开始那一刻的盘上状态"上：它先留底、失败再还原。
        # 若开始时就有一套"一部分新、一部分旧"的产物，那它留的底本身就是不一致的
        # 状态，还原回去只是把不一致**原样恢复**——而上一版随后还会把标记删掉
        # （``_end_publish`` 当时是无条件删除），于是那套从未提交成功的产物变成
        # "加载成功"（复核实测：四份文件字节未变、旧标记被删除、新实例由拒绝变成功）。
        #
        # 恢复路径是 ``run()``：它会重写全部四份产物，覆盖一次未完成的发布是合法的
        # ——``_begin_publish`` 会把被覆盖的那一份记进 ``supersedes``。
        # 零预算精修走的就是 ``run()``，所以它在上面先返回，不经过这道检查。
        unfinished = self.unfinished_publish()
        if unfinished is not None:
            raise TrainingError(
                f"{self.output_dir} holds an unfinished publish (see "
                f"{self._publish_marker_path()}); its artifacts may be a mixture of new "
                "and old. Refusing to refine on top of it, because the refinement "
                "transaction would snapshot that inconsistent state and restore it "
                "verbatim. Run the trainer (which rewrites every artifact) or resolve "
                "the marker first"
            )

        refine_index = len(self.refine_history) + 1
        fingerprint_before = self.parameter_fingerprint()
        # **快照要在取批次之前。** ``self.sampler.batch()`` 会推进采样器自己的
        # 发生器；晚一步取快照，回滚之后采样序列已经前进了一次——参数还原了，
        # 采样轨迹没有，而"恢复到精修前"的语义正好包括后者。
        snapshot = self._snapshot_state()
        # 有效检查点的**字节**也留一份底。精修成功之后收尾的 ``run()`` 会重写
        # ``last.pt``/``best.pt``；失败若发生在写盘之后，"内存回滚了、盘上是新的"
        # 是一种新的不一致，比两边都不回滚更难查。
        artifacts = self._capture_artifacts()
        closures = {"count": 0}
        last: dict[str, Any] = {"total": float("nan")}
        objective_definition: dict[str, Any] = {
            "kind": "fixed-sample L-BFGS (strong Wolfe line search)",
            "optimizer_state_checkpointed": False,
            "resume_of_an_interrupted_refine": "not supported (see resume_refinement)",
        }
        stage = "snapshot"
        # **整个精修是一条受保护流程。** 以前 ``try`` 只罩住"线搜索 + 最终求值"：
        # 前面的采样、后面的历史追加、复检与写盘全在外面，于是"收尾的 verify 抛错"
        # 会留下一个改了参数、多了历史、却没有诊断文件的中间状态（复核实测：参数没
        # 恢复、``refine_history`` 从 0 变 1、没有失败诊断文件）。
        try:
            stage = "sampling"
            batches = self.sampler.batch()
            pde_weight = pde_weight_at(self.step, self.schedule)
            # **权重必须与训练时一致，而且要记下来。** 以前这里调用 ``total_loss``
            # 时没传 ``weights``，于是精修悄悄退回了基础权重——自适应平衡器算出来的
            # 那套权重被丢掉了，目标函数与训练时不是同一个，而精修的损失看起来完全
            # 正常。
            weights = self.balancer.weights()
            # 下面记录用的是局部变量 ``weights`` 与 ``objective_definition``；
            # 早先那个 ``self._refine_weights`` 只写不读，已删除。
            objective_definition.update({
                "adaptive_weights": {key: float(value) for key, value in weights.items()},
                "pde_weight": float(pde_weight),
                "pde_weight_note": (
                    "frozen at the step where refinement started; the schedule's ramp "
                    "is not advanced by refinement"
                ),
                "samples": (
                    "one batch drawn once before the first closure; every closure "
                    "evaluates the same samples — resampling inside a line search would "
                    "make the objective a different function at each evaluation"
                ),
                "samples_fingerprint": _batch_fingerprint(batches),
            })

            optimizer = torch.optim.LBFGS(
                [p for p in self.model.parameters() if p.requires_grad],
                max_iter=iterations,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                closures["count"] += 1
                optimizer.zero_grad(set_to_none=True)
                breakdown = total_loss(
                    batches,
                    self.model,
                    weights=weights,
                    contact_pressure=self.contact_pressure,
                    pde_weight=pde_weight,
                )
                # **非有限损失在反传之前拦下。** 与 Adam 的 ``_update`` 同一条
                # 规则，以前这一层只覆盖 Adam，精修那条路完全没查。
                if not bool(torch.isfinite(breakdown.total)):
                    raise NonFiniteUpdate(
                        step=int(self.step),
                        loss=float("nan"),
                        bad_parameters=[],
                        detail=(
                            f"refinement #{refine_index}, closure #{closures['count']}: "
                            "the total loss is not finite; the line search is stopped "
                            "before backward so no parameter is moved from it"
                        ),
                    )
                breakdown.total.backward()
                # **非有限梯度在更新或接受试探之前拦下。** 不裁剪、不跳过、不换成
                # 零：那三种做法都把"这里有异常"变成"看起来还行"。
                bad_parameters = [
                    name for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                    and parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                if bad_parameters:
                    raise NonFiniteUpdate(
                        step=int(self.step),
                        loss=float(breakdown.total.detach()),
                        bad_parameters=bad_parameters,
                        detail=(
                            f"refinement #{refine_index}, closure #{closures['count']}: "
                            "the gradient is not finite for these parameters; the trial "
                            "step was not offered to the line search"
                        ),
                    )
                last["total"] = float(breakdown.total.detach())
                return breakdown.total

            stage = "optimisation"
            optimizer.step(closure)

            # 最终目标值必须在**被接受的最终参数**上求。线搜索最后一次求值的那个
            # 试探点可能已经被否决（`strong_wolfe` 会回退），而 Adam 的旧损失更是
            # 来自另一组参数——两者都不是精修的结果。
            #
            # **不能用 ``torch.no_grad()``**：目标函数内部要用 ``autograd.grad``
            # 取前沿网络对坐标的导数（移动坐标的链式法则），关掉建图会让它在
            # 求值阶段就抛错。这里只是**不调用 backward**——不反传就不会动参数。
            stage = "final evaluation"
            with torch.enable_grad():
                final_breakdown = total_loss(
                    batches,
                    self.model,
                    weights=weights,
                    contact_pressure=self.contact_pressure,
                    pde_weight=pde_weight,
                )
                accepted_loss = float(final_breakdown.total.detach())
            if not math.isfinite(accepted_loss):
                raise NonFiniteUpdate(
                    step=int(self.step),
                    loss=accepted_loss,
                    bad_parameters=[],
                    detail=(
                        f"refinement #{refine_index}: the objective evaluated on the "
                        "accepted parameters is not finite, so the refined model cannot "
                        "be reported as a usable result"
                    ),
                )
            stage = "fingerprint"
            fingerprint_after = self.parameter_fingerprint()
            parameters_changed = fingerprint_after != fingerprint_before

            record = {
                "index": refine_index,
                "step": int(self.step),
                "iterations_requested": int(iterations),
                "closures": int(closures["count"]),
                # 线搜索最后一次试探值**单独**留着做诊断，不当成精修结果。
                "loss_at_last_closure_evaluation": float(last.get("total", float("nan"))),
                "loss_on_accepted_parameters": accepted_loss,
                "loss_source": (
                    "evaluated on the parameters the line search actually accepted, on "
                    "the same fixed batch and weights as every closure"
                ),
                "weights": {key: float(value) for key, value in weights.items()},
                "pde_weight": float(pde_weight),
                "objective": objective_definition,
                "fingerprint_before": fingerprint_before,
                "fingerprint_after": fingerprint_after,
                # **由指纹比较产生，不是写死的 True。** 一次一步都没走成的精修
                # （例如强 Wolfe 立刻判满足）不该报成"参数变了"。
                "parameters_changed": bool(parameters_changed),
                "resume_supported": False,
                "resume_note": (
                    "L-BFGS optimiser state and line-search state are not checkpointed; "
                    "use resume_refinement() to be told so explicitly"
                ),
            }

            # **历史提交、复检与写盘也在事务里。** 它们在 ``try`` 外面时，任何一步
            # 抛错都会把"参数改了、历史多了、产物可能写了一半"留在原地。
            stage = "history commit"
            self.refine_history.append(record)

            # **精修改了参数，必须重新评价。** 收尾流程按"参数指纹变了没有"
            # 判断要不要重查（见 ``run`` 的 ``parameters_moved``），不需要在这里
            # 作废什么——指纹是现算的。精修记录作为**来源身份**传给 ``run``，
            # 让终态、检查点与清单在写出来之前就引用同一个结论。
            stage = "recheck and publish"
            result = self.run(
                max_updates=self.step,
                verbose=verbose,
                refinement=list(self.refine_history),
            )
        except BaseException as exc:
            # **内存与磁盘分别受保护、独立尝试。**
            #
            # 以前是一个 `try` 罩住两步：内存恢复一失败，第二步根本不执行，
            # 于是本次失败精修写出的 ``last.pt`` 留在盘上——**另一个实例照样能
            # 把它加载成功**（复核实测：`last.pt` 未恢复旧字节、仍带 1 条精修记录、
            # 新实例加载成功）。一个对象不可用并不能阻止别的对象读盘。
            memory_error: BaseException | None = None
            try:
                self._restore_state(snapshot)
            except BaseException as rollback:  # noqa: BLE001 - 回滚失败要如实报
                memory_error = rollback
            # 内存恢复失败也要继续还原磁盘：两件事各有各的失败方式。
            artifact_results = self._restore_artifacts(artifacts)
            file_failures = {
                name: result for name, result in artifact_results.items()
                if str(result).startswith("FAILED")
            }
            restored_everything = memory_error is None and not file_failures
            if not file_failures:
                # **标记只描述盘上的一致性。** 内存恢复失败不影响目录里是什么，
                # 所以只要每一份文件都还原了（或按原样删掉了），就要把标记撤掉
                # ——否则一个恢复得好好的目录会被永久拒绝加载。内存那一路的失败
                # 仍然由 ``_unusable`` 表达，那是**实例**的状态，不是盘的状态。
                try:
                    self._end_publish()
                except Exception:  # noqa: BLE001 - 撤标记失败不影响结论
                    pass
            if not restored_everything:
                self._unusable = (
                    "a failed refinement could not be fully rolled back: "
                    + "; ".join(
                        ([f"memory: {memory_error}"] if memory_error is not None else [])
                        + [f"{name}: {why}" for name, why in sorted(file_failures.items())]
                    )
                )
            failure = {
                "refinement_index": refine_index,
                "stage": stage,
                "step": int(self.step),
                "closures": int(closures["count"]),
                "iterations_requested": int(iterations),
                "error": f"{exc.__class__.__name__}: {exc}",
                # 原始异常与**每一项**恢复异常都留下；不用一个总布尔值代替事实。
                "memory_rollback_error": (
                    None if memory_error is None
                    else f"{memory_error.__class__.__name__}: {memory_error}"
                ),
                "artifact_rollback": dict(artifact_results),
                "artifacts_restored": not file_failures,
                "bad_parameters": list(getattr(exc, "bad_parameters", [])),
                "objective": objective_definition,
                "parameters_restored": memory_error is None,
                "fingerprint_restored": (
                    memory_error is None
                    and self.parameter_fingerprint() == fingerprint_before
                ),
                "note": (
                    "the parameters, the sampling sequence and the last valid "
                    "checkpoints were left untouched; this record is written to a "
                    "separate diagnostics file, never into the resumable checkpoint"
                ),
            }
            diagnostic_error: BaseException | None = None
            try:
                self._write_refinement_failure(failure)
            except BaseException as log_error:  # noqa: BLE001 - 诊断写不进去也不能吞
                diagnostic_error = log_error
            if not restored_everything or diagnostic_error is not None:
                raise TrainingError(
                    f"refinement #{refine_index} failed ({exc.__class__.__name__}: {exc})."
                    + (
                        f" Rolling back did not restore everything: "
                        + "; ".join(
                            ([f"memory: {memory_error}"] if memory_error is not None else [])
                            + [f"{name}: {why}" for name, why in sorted(file_failures.items())]
                        )
                        + ". This trainer instance is in an unknown state and must not "
                          "be reused."
                        if not restored_everything else ""
                    )
                    + (
                        f" The failure diagnostics file could not be written either "
                        f"({diagnostic_error.__class__.__name__}: {diagnostic_error}); "
                        "the original failure above is the one to act on."
                        if diagnostic_error is not None else ""
                    )
                ) from (memory_error or diagnostic_error or exc)
            raise

        if verbose:
            print(
                f"[lbfgs] fixed-sample refinement #{refine_index} -> "
                f"{record['loss_on_accepted_parameters']:.6e} on accepted parameters "
                f"({record['closures']} closures)"
            )
        return result

    #: 一次事务要保护的**盘上**产物。
    #:
    #: 按``run()`` 的**实际调用链**列：它先写 ``best.pt``（若有候选）、再写
    #: ``last.pt``，然后写 ``checks.json``，最后写 ``run_manifest.json``。
    #: 以前这里只有两份检查点，于是"``checks.json`` 已经发布、``run_manifest.json``
    #: 才失败"这一档只回滚了一半（复核实测：``checks.json`` 未恢复、
    #: ``_final_check_state`` 也未恢复）。
    #:
    #: ``cli/train.py`` 另外写的 ``train_result.json`` 不在这里：它在 ``run()``
    #: **返回之后**由外层写下，责任属于外层。一次失败的运行不会走到那里。
    _PROTECTED_ARTIFACTS = ("last.pt", "best.pt", "checks.json", "run_manifest.json")

    #: 未提交标记。发布多份产物**不是**原子的：进程在两次写入之间被杀掉，目录里
    #: 会留下"一部分新、一部分旧"的状态，而每一份单看都是合法文件。启动发布前写
    #: 下这个标记，成功后删掉；``load`` 看到自己的名字在里面就拒绝加载。
    #:
    #: 文件名、字段名与读取规则都定义在 :mod:`.checkpoint`，训练与预测共用。
    _PUBLISH_MARKER = PUBLISH_MARKER

    def _capture_artifacts(self) -> dict[str, bytes | None]:
        """把可恢复检查点的**字节**留底。``None`` 表示当时不存在。"""
        captured: dict[str, bytes | None] = {}
        for name in self._PROTECTED_ARTIFACTS:
            path = self.output_dir / name
            captured[name] = path.read_bytes() if path.exists() else None
        return captured

    def _restore_artifacts(self, captured: Mapping[str, bytes | None]) -> dict[str, str]:
        """还原 :meth:`_capture_artifacts` 的快照，**逐个文件独立尝试**。

        当时不存在的文件要**删掉**：留下一个"这次事务新写出来的"检查点，就等于
        把一个从未成功过的状态发布出去了。

        一个文件恢复失败不能让其余文件放弃：剩下的清单一并还原之后，盘上至少
        回到"除了一个文件之外都是旧的"，而不是"一个都没还原"。返回每个文件的
        实际结果，调用方据此写进失败诊断——**不以一个总布尔值代替事实**。
        """
        results: dict[str, str] = {}
        for name, payload in captured.items():
            path = self.output_dir / name
            try:
                if payload is None:
                    if path.exists():
                        path.unlink()
                    results[name] = "removed (did not exist before)"
                else:
                    path.write_bytes(payload)
                    results[name] = "restored to the previous bytes"
            except Exception as exc:  # noqa: BLE001 - 逐个兜错，最后一起报
                results[name] = f"FAILED: {exc.__class__.__name__}: {exc}"
        return results

    def _write_refinement_failure(self, failure: Mapping[str, Any]) -> Path:
        """把一次失败的精修写到**独立**的诊断文件。

        刻意不写进 ``last.pt``/``best.pt``：那两个是可恢复检查点，往里面塞一条
        "这次精修崩了"的记录，会让人以为恢复出来的状态包含那次失败。目录里多一个
        ``refinement_failures.json``，失败可查，可恢复状态的语义不变。
        """
        path = self.output_dir / "refinement_failures.json"
        history: list[dict[str, Any]] = []
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - 诊断文件损坏不该盖掉真正的失败
                history = []
        # **不要把整份 dict 交给 ``_jsonable``**：它对不认识的对象回落到
        # ``str(value)``，整份记录会被压成一行字符串，读回来既不是 dict 也不是
        # JSON 结构。这里只把 JSON 无法直接序列化的**叶子**交给它。
        history.append(dict(failure))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(history, indent=2, default=_jsonable), encoding="utf-8"
        )
        return path

    # -- 检查点 -----------------------------------------------------------
    def _checkpoint_state(self) -> dict[str, Any]:
        """构造一份检查点载荷所需要的那**一刻**的状态，一次取齐。

        **每一项都是独立的副本**：模型张量克隆、优化器/平衡器/历史深拷贝。
        只存 ``state_dict()`` 里仍然引用当前 tensor 的浅拷贝是不够的——后续的
        原地更新会把它一起改掉，"候选快照"就退化成了"当前状态"。

        这是"候选"与"当前"能被分开的前提，也是把保存拆成"取状态 / 拼载荷 /
        原子写出"三层的中间那一层。
        """
        import copy as _copy

        return {
            "step": int(self.step),
            "model": {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            },
            "optimizer": _copy.deepcopy(self.optimizer.state_dict()),
            "learning_rates": [g["lr"] for g in self.optimizer.param_groups],
            "last_update_loss": float(self._last_update_loss),
            "last_update_loss_step": int(self._last_update_loss_step),
            "refinement": _copy.deepcopy(self.refine_history),
            "rng": _rng_state(self.seed),
            "sampler_rng": self.sampler.rng_state(),
            "history": _copy.deepcopy(self.history),
            "checks": _copy.deepcopy(self.check_history),
            "best_check": _copy.deepcopy(self.best),
            "balancing": _copy.deepcopy(self.balancer.state_dict()),
            "run_origin": dict(self._run_origin),
            "stage": {
                "current": self._current_stage,
                "window_fraction": float(self._stage_window),
                "pending": self._pending_stage,
                "blocked": bool(self._stage_blocked),
                "history": _copy.deepcopy(self.stage_history),
            },
        }

    def _build_payload(
        self,
        state: Mapping[str, Any],
        *,
        run_status: str | None,
        final_stage_verdict: Mapping[str, Any] | None,
        check_state: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """把"某一刻的状态"拼成一份检查点载荷。

        与 :meth:`_checkpoint_state` 分开，是为了让 ``best.pt`` 能用**候选那一刻**
        的状态拼载荷，而不是让通用保存路径去读当前模型。
        """
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "schedule": _schedule_signature(self.schedule),
            "check_set": self.check_set.as_dict(),
            "data_identity": self._data_identity(),
            "data_compatibility": self.gate.as_dict(),
            "provenance": self.provenance,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "case_ids": [c.case_id for c in self.cases],
            "split": {c.case_id: c.source.get("split") for c in self.cases},
            "time_scale": self.time_scale,
            # **冻结爆速必须随检查点走。** 预测入口要在**新半径**上重建时间原点
            # ``T_h = R_0/D + epsilon(R_0)``，而它读不到训练时的任何对象——只能读
            # 这个文件。以前保存端从不写它，于是预测入口把每一份方案 B 的训练
            # 检查点都当成"方案 A 的旧文件"拒掉（复核实测）。
            DETONATION_VELOCITY_FIELD: self._frozen_detonation_velocity(),
            "contact_pressure_scale_pa": self.contact_pressure,
            "scales": {
                c.case_id: self.scales_by_radius[float(c.charge_radius)].as_dict()
                for c in self.cases
            },
            "code_version": _code_version(),
            "run_status": run_status,
            "final_stage_verdict": final_stage_verdict,
            "check_state": check_state,
            # 尚未发布的最佳候选也一起留着：一次中断之后，恢复出来的实例不该
            # 只剩一个分数、找不到对应的参数。
            "pending_best_candidate": self._best_candidate,
            "best_dirty": bool(self._best_dirty),
        }
        payload.update(state)
        payload.update(dict(extra or {}))
        return payload

    def _write_payload(self, payload: Mapping[str, Any], path: Path) -> Path:
        """原子写出：先写临时文件，再替换。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(dict(payload), temporary)
        temporary.replace(target)
        return target

    def _save(self, path: Path) -> Path:
        """按**当前**状态写一份检查点。"""
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        self._assert_usable()
        return self._write_payload(
            self._build_payload(
                self._checkpoint_state(),
                run_status=getattr(self, "_final_status", None),
                final_stage_verdict=getattr(self, "_final_stage_verdict", None),
                check_state=getattr(self, "_final_check_state", None),
            ),
            Path(path),
        )

    # -- 多文件发布的事务标记 ---------------------------------------------

    def _publish_marker_path(self) -> Path:
        return publish_marker_path(self.output_dir)

    def unfinished_publish(self) -> dict[str, Any] | None:
        """本目录里有没有**上一次没做完**的发布。没有则 ``None``。"""
        return read_publish_marker(self.output_dir)

    def _begin_publish(self, names) -> Path:
        """写下"这些产物正在发布"。成功后由 :meth:`_end_publish` 删除。

        单文件原子替换**不等于**多文件事务：``run()`` 要写四份产物，进程在第
        二份和第三份之间被杀掉时，目录里留下的是一套互相矛盾的组合，而每一份
        单看都合法。标记让加载端能把"未提交"认出来。

        **标记记下自己的归属**（``owner``）。没有归属时，后来的一次失败操作只能
        *无条件*删除标记——包括删除**上一次**中断留下的那一个，于是一套从未提交
        成功的产物变成"看起来已提交"（复核实测）。有了归属，
        :meth:`_end_publish` 就只清理自己写的那一个。

        标记已经存在时，说明上一次发布没有完成。这里把它的内容记进
        ``supersedes`` 并留在实例上，**不假装它不存在**：``run()`` 随后会重写
        全部四份产物，覆盖它是合法的，但"覆盖掉了一次未完成的发布"这件事要
        留下痕迹。
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._publish_marker_path()
        previous = read_publish_marker(self.output_dir)
        self._superseded_publish = previous
        self._publish_token = f"{id(self):x}-{time.time_ns():x}"
        body: dict[str, Any] = {
            "artifacts": sorted(names),
            PUBLISH_OWNER_KEY: self._publish_token,
        }
        if previous is not None:
            body["supersedes"] = previous
        path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        return path

    def _end_publish(self) -> bool:
        """撤掉**本次事务自己**的标记；不是自己的标记一律留着。

        返回是否撤掉了。上一版是无条件 ``unlink()``：一次失败的 ``refine()``
        会把上一次失败的 ``run()`` 留下的标记一起删掉，而它回滚到的是那次的
        半套产物——目录由"拒绝加载"变成"加载成功"（复核实测）。
        """
        if self._publish_token is None:
            # 这次事务根本没开始过发布。此时存在的标记一定是**别人的**——包括
            # 更早的写入端留下的、没有 ``owner`` 字段的老标记——一律不动。
            return False
        path = self._publish_marker_path()
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 读不懂就不动它，保守留下
            return False
        if not isinstance(payload, dict):
            return False
        if payload.get(PUBLISH_OWNER_KEY) != self._publish_token:
            return False
        path.unlink()
        return True

    def _refuse_if_uncommitted(self, path: Path) -> None:
        """加载端必须**落实**拒绝规则，而不是只在日志里提一句。

        标记在**被加载的那份检查点旁边**找，不是在自己的 ``output_dir`` 里找：
        读它的人往往是一个新建的实例，输出目录与写出它的那次运行不同。
        预测入口调用的是同一个函数（:func:`..training.checkpoint.refuse_if_uncommitted`）。
        """
        refuse_if_uncommitted(Path(path))

    def load(self, path: Path | str) -> None:
        """恢复训练。

        恢复的是**优化状态**：模型、优化器、步数、随机数发生器。恢复后继续
        跑完剩余日程；已经走过的更新次数不会重复。
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required")
        self._assert_usable()
        # **未提交的产物不能当正常检查点加载。** 规则要在这里**落实**，不是在
        # 日志里提一句——另一个实例读到的就是这一份盘上文件。
        self._refuse_if_uncommitted(Path(path))
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise TrainingError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
            raise TrainingError(
                f"{path} has checkpoint version {payload.get('version')}, but this "
                f"trainer writes and reads version {CHECKPOINT_VERSION}. There is no "
                "migration path: an older writer did not record the last update loss, "
                "the schedule, the fixed check set or the data identity, and those "
                "cannot be reconstructed from the file. Retrain, or import the weights "
                "explicitly as an initialisation rather than as a resume."
            )
        # 全局随机状态是**严格续跑**的一部分：少了它，"恢复到第 N 步"与"连跑
        # N 步"的采样序列不同。以前缺这一项时 ``_restore_rng(None)`` 静默返回，
        # 于是恢复出来的是一条不同的轨迹，而损失曲线看不出任何异常。
        rng_payload = payload.get("rng")
        if not isinstance(rng_payload, Mapping):
            raise TrainingError(
                f"{path} carries no global RNG state; resuming from it would continue "
                "the global random sequence from somewhere else, so 'resume to step N' "
                "would not reproduce 'run N steps'"
            )
        missing_rng = [
            key for key in ("numpy", "torch")
            if rng_payload.get(key) is None and torch is not None
        ]
        if missing_rng:
            raise TrainingError(
                f"{path}: the global RNG record has no {missing_rng}; a partial record "
                "cannot restore the sequence it claims to"
            )
        if abs(float(payload.get("time_scale", 0.0)) - self.time_scale) > 1.0e-12:
            raise TrainingError(
                "the checkpoint was trained with a different hat T_*; resuming would "
                "silently change the network's time input"
            )
        # **冻结爆速也要核。** 与 ``time_scale`` 同一个道理，但以前**完全没查**：
        # 恢复端从不读顶层 ``detonation_velocity``，于是把它删掉或改成 ``-1``
        # 都能照常恢复（复核实测）；而顶层与逐算例 ``scales`` 记录互相矛盾时，
        # 恢复端用**接收实例的** D、预测端用**文件里的** D，同一份检查点被解释成
        # 两条时间轴（50 mm 上实测 8.21500721501 µs 对 41.5186385737 µs）。
        #
        # ``expected`` 传的是本实例**实际使用**的尺度（``_frozen_detonation_velocity``
        # 读的是冻结的 ``scales_by_radius``，不是 ``cases`` 的元数据）：只查"文件
        # 内部自洽"拦不住一份自洽的检查点被另一个爆速的实例恢复。
        require_detonation_velocity(
            payload, path, expected=self._frozen_detonation_velocity()
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
        if int(payload["step"]) < 0:
            raise TrainingError(f"the checkpoint's step is negative: {payload['step']!r}")

        # -- 日程、检查集与数据身份：**必须是同一件事** ----------------------
        #
        # 只比"当前窗口"远远不够。同名阶段把预算从 2 改成 5、学习率与 PDE 权重
        # 一起改掉，只要那一刻的窗口碰巧相同，以前就照常恢复——于是"从第 N 步继续"
        # 换掉的目标函数和学习率，而检查点里的 best/last 结论是在另一套日程下得到的。
        stored_schedule = payload.get("schedule")
        if not isinstance(stored_schedule, list):
            raise TrainingError(
                f"{path} carries no schedule definition; resuming against a different "
                "schedule would silently change the objective and the learning rate"
            )
        current_schedule = _schedule_signature(self.schedule)
        if _signatures_differ(stored_schedule, current_schedule):
            raise TrainingError(
                f"{path} was trained on a different schedule. Stored "
                f"{_describe_signature(stored_schedule)}; this trainer has "
                f"{_describe_signature(current_schedule)}. Comparing the time window "
                "alone is not enough to show it is the same training run"
            )
        stored_check_set = payload.get("check_set")
        if not isinstance(stored_check_set, Mapping):
            raise TrainingError(
                f"{path} carries no fixed check-set definition; the stored best/last "
                "verdicts would then be matched against a different evaluation set"
            )
        if _jsonable(stored_check_set) != _jsonable(self.check_set.as_dict()):
            raise TrainingError(
                f"{path} was evaluated on a different fixed check set; the best/last "
                "conclusions in it do not belong to this trainer's evaluation definition"
            )
        stored_data = payload.get("data_identity")
        if not isinstance(stored_data, Mapping):
            raise TrainingError(
                f"{path} carries no data identity; matching on the output directory "
                "string would accept a different data set at the same path"
            )
        current_data = self._data_identity()
        if _jsonable(stored_data) != _jsonable(current_data):
            raise TrainingError(
                f"{path} was trained on different data: stored "
                f"{sorted(stored_data)}, this trainer has {sorted(current_data)}, or the "
                "contents behind those ids differ. Moving an output directory is fine; "
                "changing the data behind it is not"
            )

        stored_origin = payload.get("run_origin")
        if not isinstance(stored_origin, Mapping):
            raise TrainingError(
                f"{path} carries no 'run_origin'. Without it, a checkpoint written by an "
                "explicit diagnostic run or by a refinement would come back looking like "
                "a plain completed run"
            )
        for key in ("explicit_diagnostic", "refined"):
            if not isinstance(stored_origin.get(key), (bool, np.bool_)):
                raise TrainingError(
                    f"the checkpoint's run_origin[{key!r}] is "
                    f"{stored_origin.get(key)!r}, not a boolean"
                )
        # 候选与"待发布"标志必须**在位**：一份中途写下的检查点如果丢了这两项，
        # 恢复出来的实例就只剩 ``self.best`` 那张分数表，找不到对应的参数。
        if "pending_best_candidate" not in payload:
            raise TrainingError(
                f"{path} carries no 'pending_best_candidate' key; a checkpoint written "
                "mid-run may hold an unpublished best candidate, and without this field "
                "it cannot be reconstructed"
            )
        if "best_dirty" not in payload or not isinstance(
            payload["best_dirty"], (bool, np.bool_)
        ):
            raise TrainingError(
                f"the checkpoint's 'best_dirty' flag is "
                f"{payload.get('best_dirty')!r}, not a boolean"
            )
        # **``best_dirty=True`` 必须对应一份完整的候选。** 少了这一步，一份
        # "标志为真、候选为空"的检查点会**加载成功**：标志被悄悄清成 ``False``，
        # 最佳记录的分数还在、参数却没有，再次运行也不再产出 ``best.pt``
        # （复核实测）。这是**输入自相矛盾**，按本仓库的约定直接拒绝。
        require_best_candidate_consistency(payload, path)
        for key in ("last_update_loss", "last_update_loss_step"):
            if key not in payload:
                raise TrainingError(
                    f"{path} carries no {key!r}. Without it a resumed run that performs "
                    "no update would report a fabricated NaN loss at step 0"
                )
        stored_loss = payload["last_update_loss"]
        if not isinstance(stored_loss, (int, float, np.floating)):
            raise TrainingError(
                f"the checkpoint's last update loss is {stored_loss!r}, not a number"
            )
        if not isinstance(payload["last_update_loss_step"], (int, np.integer)):
            raise TrainingError(
                "the checkpoint's last update loss step is "
                f"{payload['last_update_loss_step']!r}, not an integer"
            )
        if int(payload["last_update_loss_step"]) > int(payload["step"]):
            raise TrainingError(
                f"the checkpoint records its last measured loss at step "
                f"{payload['last_update_loss_step']}, beyond its own step "
                f"{payload['step']}"
            )
        if not isinstance(payload.get("refinement", []), list):
            raise TrainingError(
                f"the checkpoint's 'refinement' record is "
                f"{type(payload.get('refinement')).__name__}, not a list"
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
        blocked = stage_payload["blocked"]
        pending = stage_payload["pending"]
        # 类型先核：``bool("False")`` 是 ``True``，一个字符串 ``"false"`` 会被读成
        # "已阻塞"，而它想说的正好相反。
        if not isinstance(blocked, (bool, np.bool_)):
            raise TrainingError(
                f"the checkpoint's stage 'blocked' flag is {blocked!r}, which is not a "
                "boolean; a truthy string would be read as 'blocked'"
            )
        blocked = bool(blocked)
        if pending is not None and not isinstance(pending, str):
            raise TrainingError(
                f"the checkpoint's pending stage is {pending!r}, which is neither None "
                "nor a stage name"
            )
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
            try:
                self._restore_state(snapshot)
            except Exception as rollback_error:
                # **回滚也失败**：对象现在处于"一半新、一半旧"的未知状态。继续用它
                # 训练或保存会把那个混合状态当成正常结果写出去，所以把它标成不可用
                # 并明确报出来——而不是吞掉回滚异常、把对象当成还能用。
                self._unusable = (
                    "a failed load could not be rolled back: "
                    f"{rollback_error.__class__.__name__}: {rollback_error}"
                )
                raise TrainingError(
                    f"restoring {path} failed, and rolling the object back failed too "
                    f"({rollback_error.__class__.__name__}: {rollback_error}). This "
                    "trainer instance is in an unknown state and must not be reused; "
                    "build a new one."
                ) from rollback_error
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
            # 最近一次平衡记录是**可观测**的（报表与测试都读它），也要进事务。
            "last_balance": _copy.deepcopy(self.last_balance),
            "rng": _snapshot_global_rng(),
            "samplers": [
                {
                    "sampler": sampler,
                    "window": float(sampler.window_fraction),
                    # **窗口历史也要快照。** 只还原窗口值的话，回滚会在
                    # ``_window_history`` 里留下一条"窗口曾经变过"的记录，而那次
                    # 变化从未真正生效——历史是事实记录，不是尝试记录。
                    "window_history": sampler.window_history,
                    "rng": _copy.deepcopy(sampler.rng_state()),
                }
                for sampler in self.sampler.samplers
            ],
            "last_update_loss": float(self._last_update_loss),
            "last_update_loss_step": int(self._last_update_loss_step),
            "refine_history": _copy.deepcopy(self.refine_history),
            "run_origin": dict(self._run_origin),
            "best_dirty": bool(self._best_dirty),
            # 候选快照也要进事务：回滚之后如果只剩 ``self.best`` 那张分数表，
            # 下一次发布就会拿**当前**模型去补 ``best.pt``。
            "best_candidate": _copy.deepcopy(self._best_candidate),
            "final_status": getattr(self, "_final_status", None),
            "final_stage_verdict": _copy.deepcopy(
                getattr(self, "_final_stage_verdict", None)
            ),
            "final_check_state": getattr(self, "_final_check_state", None),
            "final_status": getattr(self, "_final_status", None),
            "final_verdict": getattr(self, "_final_stage_verdict", None),
        }

    def _restore_state(self, snapshot: Mapping[str, Any]) -> None:
        """把对象还原到快照那一刻。

        顺序上**先做模型与优化器**（它们最可能抛错），但真正要保证的是：**任何一步
        抛错都不能把对象留成"一半旧、一半新"**。所以调用方（``load``）在回滚本身
        失败时会继续往上抛，并且不把对象当成可继续使用的正常对象——见
        :meth:`load` 里 ``except`` 的处理。
        """
        import copy as _copy

        # 先在**局部**把要用的东西取齐，避免还原到一半才发现快照缺字段。
        restored_samplers = [
            (entry["sampler"], float(entry["window"]),
             [float(v) for v in entry["window_history"]],
             _copy.deepcopy(entry["rng"]))
            for entry in snapshot["samplers"]
        ]
        model_state = snapshot["model"]
        optimizer_state = snapshot["optimizer"]
        balancer_state = snapshot["balancer"]
        rng_state = snapshot["rng"]
        history = list(snapshot["history"])
        check_history = list(snapshot["check_history"])
        stage_history = list(snapshot["stage_history"])
        refine_history = list(snapshot["refine_history"])
        best = snapshot["best"]

        # **每一步单独兜错，失败不中断其余步骤。** 回滚的每一个环节都可能抛错
        # （优化器状态引用别的参数组、平衡器的记录对不上……），而"回滚到一半"
        # 正是最坏的结果：对象既不是旧的也不是新的，却还能被继续使用。所以能还原
        # 的一律还原，把失败收集起来最后一起报——并且由调用方把对象标成不可用。
        failures: list[str] = []

        def attempt(label: str, action) -> None:
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - 回滚要尽力而为，最后统一报
                failures.append(f"{label}: {exc.__class__.__name__}: {exc}")

        attempt("model", lambda: self.model.load_state_dict(model_state))
        attempt("optimizer", lambda: self.optimizer.load_state_dict(optimizer_state))
        attempt("balancer", lambda: self.balancer.load_state_dict(balancer_state))
        attempt(
            "last balance record",
            lambda: setattr(self, "last_balance", _copy.deepcopy(snapshot["last_balance"])),
        )
        attempt("global rng", lambda: _restore_global_rng(rng_state))
        for sampler, window, window_history, rng in restored_samplers:
            # 回滚要能**收窄**窗口，所以走 restore_ 而不是 set_；历史一起还原，
            # 否则一次失败的恢复会留下一个从未生效过的窗口事件。
            label = getattr(getattr(sampler, "case", None), "case_id", "?")
            attempt(
                f"sampler {label} window",
                lambda s=sampler, w=window, h=window_history: s.restore_time_window(
                    w, history=h
                ),
            )
            attempt(
                f"sampler {label} rng",
                lambda s=sampler, r=rng: s.set_rng_state(r),
            )

        self.step = int(snapshot["step"])
        self.history = history
        self.check_history = check_history
        self.best = best
        self._current_stage = snapshot["current_stage"]
        self._stage_window = float(snapshot["stage_window"])
        self._pending_stage = snapshot["pending_stage"]
        self._stage_blocked = bool(snapshot["stage_blocked"])
        self.stage_history = stage_history
        self._last_update_loss = float(snapshot["last_update_loss"])
        self._last_update_loss_step = int(snapshot["last_update_loss_step"])
        self.refine_history = refine_history
        self._run_origin = dict(snapshot["run_origin"])
        self._best_dirty = bool(snapshot["best_dirty"])
        self._best_candidate = _copy.deepcopy(snapshot["best_candidate"])
        self._final_status = snapshot["final_status"]
        self._final_stage_verdict = _copy.deepcopy(snapshot["final_stage_verdict"])
        self._final_check_state = snapshot["final_check_state"]
        self._final_status = snapshot["final_status"]
        self._final_stage_verdict = snapshot["final_verdict"]
        if failures:
            raise TrainingError(
                "rolling back did not restore everything: " + "; ".join(failures)
            )

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
        stored_origin = payload["run_origin"]
        self._run_origin = {
            "explicit_diagnostic": bool(stored_origin.get("explicit_diagnostic", False)),
            "refined": bool(stored_origin.get("refined", False)),
        }
        # 尚未发布的最佳候选随检查点一起回来：中断之后恢复出来的实例还能把
        # ``best.pt`` 补出来，而不是只剩一个分数。
        stored_candidate = payload.get("pending_best_candidate")
        self._best_candidate = (
            dict(stored_candidate) if isinstance(stored_candidate, Mapping) else None
        )
        # **标志照抄，不在这里兜底。** 旧写法是
        # ``bool(...) and self._best_candidate is not None``——"候选缺失"被悄悄
        # 折成 ``False``，自相矛盾的输入于是变成一份看不出异常的实例。
        # 一致性由 :func:`require_best_candidate_consistency` 负责，``load`` 的
        # 只读阶段已经查过；这里再查一次，任何绕过 ``load`` 的调用路径也拦得住。
        require_best_candidate_consistency(
            payload, "the checkpoint about to be committed"
        )
        self._best_dirty = bool(payload.get("best_dirty"))
        self.stage_history = list(stage_payload.get("history", []))
        # **最后一次实测损失与精修历史一起恢复。** 少了前者，零更新的恢复运行会
        # 报 ``final_loss=nan, final_loss_step=0``；少了后者，一个精修过的模型
        # 加载回来会显示成"从没精修过"。
        self._last_update_loss = float(payload["last_update_loss"])
        self._last_update_loss_step = int(payload["last_update_loss_step"])
        self.refine_history = [dict(entry) for entry in payload.get("refinement", [])]
        self.balancer.load_state_dict(payload.get("balancing"))


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------


#: 日程签名里参与比较的字段。``description`` **不**参与：改一句说明文字不该让
#: 一个检查点作废，而改预算、学习率、PDE 权重或窗口必须让它作废。
_SCHEDULE_SIGNATURE_FIELDS = (
    "name",
    "updates",
    "pde_weight_start",
    "pde_weight_end",
    "learning_rate_start",
    "learning_rate_end",
    "time_window_fraction",
)


def _hash_field(digest, name: str, value: Any) -> None:
    """把一个字段按 ``名字 | 形状 | dtype | 字节`` 混进身份哈希。

    带上名字与形状，是为了避免"不同的结构拼出同一串字节"：一个 ``(2, 3)`` 的
    数组和一个 ``(6,)`` 的数组内容相同，但它们不是同一份数据。``None`` 写成显式
    的 ``missing``，**不能**与空数组或全零混为一谈——"这份数据没有逐字段有效性"
    和"逐字段有效性全是 False"是两件完全不同的事。
    """
    if value is None:
        digest.update(f"{name}|missing".encode("utf-8"))
        return
    array = np.asarray(value)
    digest.update(f"{name}|{array.shape}|{array.dtype.str}|".encode("utf-8"))
    if array.dtype.kind in ("U", "S", "O"):
        # 字符串与对象数组没有稳定的内存布局，按元素逐个写成 UTF-8。
        for element in np.asarray(array, dtype=object).ravel():
            digest.update(f"{element!r}\x1f".encode("utf-8"))
        return
    digest.update(np.ascontiguousarray(array).tobytes())


def _batch_fingerprint(batches: Any) -> str:
    """固定样本批次的**内容**指纹，写进精修的目标函数定义里。

    "同一次精修用的是同一批样本"是线搜索成立的前提，所以要有一个可以核对的
    值，而不是一句说明文字。
    """
    import hashlib

    digest = hashlib.sha256()
    try:
        entries = batches.as_dict() if hasattr(batches, "as_dict") else str(batches)
    except Exception:  # noqa: BLE001 - 指纹失败不该阻止精修，但要如实说
        return "unavailable: the batch could not be serialised"
    digest.update(json.dumps(entries, sort_keys=True, default=_jsonable).encode("utf-8"))
    return digest.hexdigest()


def _schedule_signature(schedule: Sequence[Stage]) -> list[dict[str, Any]]:
    """日程的**数值**签名，用于"这是不是同一次训练"的核对。"""
    return [
        {field: getattr(stage, field) for field in _SCHEDULE_SIGNATURE_FIELDS}
        for stage in schedule
    ]


def _signatures_differ(stored: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]) -> bool:
    """逐阶段、逐字段比较，浮点按相对 1e-12 判定相等。"""
    if len(stored) != len(current):
        return True
    for left, right in zip(stored, current):
        for field in _SCHEDULE_SIGNATURE_FIELDS:
            a, b = left.get(field), right.get(field)
            if isinstance(a, str) or isinstance(b, str):
                if a != b:
                    return True
                continue
            if a is None or b is None:
                if a != b:
                    return True
                continue
            scale = max(abs(float(a)), abs(float(b)), 1.0e-300)
            if abs(float(a) - float(b)) > 1.0e-12 * scale:
                return True
    return False


def _describe_signature(signature: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        f"{entry.get('name')}: {entry.get('updates')} updates, lr "
        f"{entry.get('learning_rate_start')}->{entry.get('learning_rate_end')}, "
        f"lambda_PDE {entry.get('pde_weight_start')}->{entry.get('pde_weight_end')}, "
        f"window {entry.get('time_window_fraction')}"
        for entry in signature
    )


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
