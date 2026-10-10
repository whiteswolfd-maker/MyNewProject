"""训练日程（提示词 §13、§14）。

.. list-table::
   :header-rows: 1

   * - 阶段
     - Adam 更新次数
     - ``lambda_PDE``
     - 学习率
     - 时间窗口
   * - 建立初值、基本场和前沿
     - 2000
     - 0.1
     - 1e-3
     - 25 %
   * - 增强 PDE、逐步扩大时间范围
     - 3000
     - 0.1 → 1（平滑）
     - 1e-3 → 1e-4（平滑）
     - 65 %
   * - 完整联合训练
     - 15000
     - 1
     - 1e-4 → 1e-5（平滑）
     - 100 %

§14 明确要求三个阶段各自有**实际行为**，不能只有三个名字加一条 PDE 权重爬坡。
这里把三个行为落成可执行的东西：

1. **初值预训练**（``initialise``）：在**前 25 %** 的尺度化时间上拟合 B 两侧的
   ``rho/u/e`` 与初始前沿。窗口小是有意的——初值项要在起步区间里说了算，模型
   才有机会把起点做对，而不是一上来就被后期波形牵走。
2. **时间范围逐步扩大**（``strengthen_pde``）：窗口推到 65 %，**同一套**两侧网络
   与前沿网络。区间恒为 ``[0, W]``，扩大只把右端往外推、**不动左端**，所以早期
   采样点一直留在池子里——这就是"保留早期采样"的可执行含义。全程使用参考 B 初场
   作为初值目标，**不**把前一阶段的预测冻结成新的"真初值"：那样等于让模型自己
   给自己出题，误差会被固化下来，而且表面上完全看不出来。
3. **全目标时段联合优化**（``joint``）：窗口给满，两侧物理、界面、RH、可用数据与
   几何约束共同优化。

三条不可动摇的性质：

* **所有物理方程从一开始就保留**。阶段改变的是 ``lambda_PDE``、学习率、采样时间
  窗口，不是"先关掉能量方程"之类的物理开关。
* **不冻结点络**。三个模块（产物、空气、前沿）全程联合优化。
* **阶段不是物理阶段**。把优化阶段解释成"先算产物、再算空气"是对这张表的误读。

阶段推进由 :data:`STAGE_CRITERIA` 里**预先声明**的独立检查决定，不只看步数。
判据的限额直接引用 §16 的容差表，不另立一套数——否则"验收标准"就会随阶段多出
第二个版本。若预算耗尽而判据仍未通过，训练**明确停止并报告**（见
:class:`~spherical_blast_pinn.training.trainer.StageNotPassed`），不静默跳阶段。

20000 次更新只是基线预算，**不构成收敛保证**。可选 L-BFGS 精修最多 500 次
迭代，且必须固定采样点、权重与目标函数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ScheduleError(ValueError):
    """日程本身不成立。"""


@dataclass(frozen=True)
class StageCriterion:
    """阶段推进判据的一条。

    ``tolerance_key`` 指向 §16 容差表里的键；取不到时才退回 ``fallback``。这样
    判据与验收用的是**同一个数**，不会出现"阶段判据说通过、验收说不过"。
    """

    metric: str
    sense: str  # "max"（metric <= limit）或 "min"（metric >= limit）
    tolerance_key: str | None = None
    fallback: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.sense not in ("max", "min"):
            raise ScheduleError(f"criterion {self.metric!r}: sense must be 'max' or 'min'")
        if self.tolerance_key is None and self.fallback is None:
            raise ScheduleError(
                f"criterion {self.metric!r} needs a tolerance_key or a fallback limit"
            )

    def limit(self, tolerances: Mapping[str, float] | None = None) -> float:
        if self.tolerance_key and tolerances:
            value = tolerances.get(self.tolerance_key)
            if value is not None:
                return float(value)
        if self.fallback is None:  # pragma: no cover - guarded in __post_init__
            raise ScheduleError(f"criterion {self.metric!r} has no usable limit")
        return float(self.fallback)

    def evaluate(self, metrics: Mapping[str, float], tolerances: Mapping[str, float] | None = None):
        """返回 ``(通过?, 实际值, 限额)``。指标缺失或非有限一律判为**未通过**。"""
        limit = self.limit(tolerances)
        raw = metrics.get(self.metric)
        value = float(raw) if raw is not None else float("nan")
        if not math.isfinite(value):
            return False, value, limit
        passed = value <= limit if self.sense == "max" else value >= limit
        return bool(passed), value, limit

    def as_dict(self, tolerances: Mapping[str, float] | None = None) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "sense": self.sense,
            "limit": self.limit(tolerances),
            "tolerance_key": self.tolerance_key,
            "description": self.description,
        }


@dataclass(frozen=True)
class Stage:
    """一个优化阶段。"""

    name: str
    updates: int
    pde_weight_start: float
    pde_weight_end: float
    learning_rate_start: float
    learning_rate_end: float
    #: 本阶段使用的尺度化时间窗口占完整有效时段的比例（§14.2）。
    time_window_fraction: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.updates < 0:
            raise ValueError(f"stage {self.name!r} has negative updates")
        for name in ("pde_weight_start", "pde_weight_end",
                     "learning_rate_start", "learning_rate_end"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"stage {self.name!r}: {name} must be finite and non-negative")
        window = float(self.time_window_fraction)
        if not math.isfinite(window) or not (0.0 < window <= 1.0):
            raise ValueError(
                f"stage {self.name!r}: time_window_fraction must lie in (0, 1], got "
                f"{self.time_window_fraction}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "updates": self.updates,
            "pde_weight_start": self.pde_weight_start,
            "pde_weight_end": self.pde_weight_end,
            "learning_rate_start": self.learning_rate_start,
            "learning_rate_end": self.learning_rate_end,
            "time_window_fraction": float(self.time_window_fraction),
            "description": self.description,
        }


#: 提示词 §13 的默认日程，逐行对应；§14 的时间窗口逐列对应。
SCHEDULE: tuple[Stage, ...] = (
    Stage(
        name="initialise",
        updates=2000,
        pde_weight_start=0.1,
        pde_weight_end=0.1,
        learning_rate_start=1.0e-3,
        learning_rate_end=1.0e-3,
        time_window_fraction=0.25,
        description="建立初值、基本场和前沿（B 两侧 rho/u/e 与初始前沿）",
    ),
    Stage(
        name="strengthen_pde",
        updates=3000,
        pde_weight_start=0.1,
        pde_weight_end=1.0,
        learning_rate_start=1.0e-3,
        learning_rate_end=1.0e-4,
        time_window_fraction=0.65,
        description="PDE 权重线性增强，时间范围逐步扩大到 65 %，保留早期采样",
    ),
    Stage(
        name="joint",
        updates=15000,
        pde_weight_start=1.0,
        pde_weight_end=1.0,
        learning_rate_start=1.0e-4,
        learning_rate_end=1.0e-5,
        time_window_fraction=1.0,
        description="全目标时段联合优化（物理、界面、RH、数据、几何）",
    ),
)


def _assert_window_is_non_decreasing(schedule: Sequence[Stage]) -> None:
    """窗口必须**单调不减**。

    若某个阶段把窗口收窄，左端仍从 0 开始，看起来"还在扩大"，实际上是把已经
    用过的时段重新排除在外——早期采样被丢掉，而报表上只会看到一个更小的百分比。
    """
    previous = 0.0
    for stage in schedule:
        current = float(stage.time_window_fraction)
        if current < previous:
            raise ScheduleError(
                f"stage {stage.name!r} narrows the time window from {previous} to "
                f"{current}; §14 requires the window to expand and keep early samples"
            )
        previous = current


_assert_window_is_non_decreasing(SCHEDULE)


#: 阶段推进的**预先声明**判据。在训练开始之前就固定，不按结果回头改。
STAGE_CRITERIA: dict[str, tuple[StageCriterion, ...]] = {
    "initialise": (
        StageCriterion(
            "ic_fields", "max", tolerance_key="initial_relative_l2", fallback=1.0e-2,
            description="B 两侧初值场（rho/u/e）的相对平方误差",
        ),
        StageCriterion(
            "ic_gap_relative_error", "max",
            tolerance_key="initial_gap_relative", fallback=5.0e-2,
            description="起点空气层厚度（Rs - Rc）的相对误差",
        ),
        StageCriterion(
            "nonphysical_violation_count", "max", fallback=0.0,
            description="非物理状态计数必须为零",
        ),
    ),
    "strengthen_pde": (
        StageCriterion(
            "pde_rms_max", "max", tolerance_key="pde_rms", fallback=1.0e-2,
            description="逐材料逐方程 PDE 残差的最大 RMS（窗口内）",
        ),
        StageCriterion(
            "nonphysical_violation_count", "max", fallback=0.0,
            description="非物理状态计数必须为零",
        ),
    ),
    "joint": (
        StageCriterion(
            "pde_rms_max", "max", tolerance_key="pde_rms", fallback=1.0e-2,
            description="逐材料逐方程 PDE 残差的最大 RMS（全时段）",
        ),
        StageCriterion(
            "development_prediction_error", "max",
            tolerance_key="field_relative_l2", fallback=0.10,
            description="开发帧预测误差",
        ),
        StageCriterion(
            "nonphysical_violation_count", "max", fallback=0.0,
            description="非物理状态计数必须为零",
        ),
    ),
}


def criteria_for(stage_name: str) -> tuple[StageCriterion, ...]:
    """某个阶段的判据。未声明的阶段视为**没有**判据，会被拒绝。"""
    if stage_name not in STAGE_CRITERIA:
        raise ScheduleError(
            f"stage {stage_name!r} has no declared advance criteria; §14 forbids "
            "advancing a stage on step count alone"
        )
    return STAGE_CRITERIA[stage_name]


TOTAL_UPDATES = sum(stage.updates for stage in SCHEDULE)

#: L-BFGS 精修的上限（提示词 §13）。
MAX_LBFGS_ITERATIONS = 500


def _linear(start: float, end: float, fraction: float) -> float:
    return float(start) + (float(end) - float(start)) * float(fraction)


def cosine_learning_rate(start: float, end: float, fraction: float) -> float:
    """余弦下降：``end + (start - end) * (1 + cos(pi f)) / 2``。"""
    fraction = min(max(float(fraction), 0.0), 1.0)
    return float(end) + (float(start) - float(end)) * 0.5 * (
        1.0 + math.cos(math.pi * fraction)
    )


def stage_and_fraction(step: int, schedule: Sequence[Stage] = SCHEDULE):
    """把全局步数映射到 ``(阶段, 阶段内比例)``。超出日程时停在最后一个阶段。"""
    remaining = int(step)
    for stage in schedule:
        if remaining < stage.updates:
            return stage, (remaining / stage.updates if stage.updates else 0.0)
        remaining -= stage.updates
    last = schedule[-1]
    return last, 1.0


def pde_weight_at(step: int, schedule: Sequence[Stage] = SCHEDULE) -> float:
    """``lambda_PDE``：阶段内**线性**从起始值升到结束值。

    线性而不是余弦，是因为规范说"平滑增至"，而 PDE 权重在早期必须真正小到
    让初值项主导，余弦在前段太平缓，等于让 PDE 提前介入。
    """
    stage, fraction = stage_and_fraction(step, schedule)
    return _linear(stage.pde_weight_start, stage.pde_weight_end, fraction)


def learning_rate_at(step: int, schedule: Sequence[Stage] = SCHEDULE) -> float:
    """学习率：阶段内余弦下降。"""
    stage, fraction = stage_and_fraction(step, schedule)
    return cosine_learning_rate(stage.learning_rate_start, stage.learning_rate_end, fraction)


def time_window_fraction_at(step: int, schedule: Sequence[Stage] = SCHEDULE) -> float:
    """当前阶段的尺度化时间窗口比例（§14.2）。阶段内恒定，切阶段时跳变。"""
    stage, _ = stage_and_fraction(step, schedule)
    return float(stage.time_window_fraction)


def describe_schedule(schedule: Sequence[Stage] = SCHEDULE) -> list[dict[str, Any]]:
    """展开成逐步可核对的表，写进运行清单。"""
    rows = []
    cursor = 0
    for stage in schedule:
        rows.append({
            **stage.as_dict(),
            "first_step": cursor,
            "last_step": cursor + stage.updates - 1,
            "pde_weight_first": pde_weight_at(cursor, schedule),
            "pde_weight_last": pde_weight_at(cursor + max(stage.updates - 1, 0), schedule),
            "learning_rate_first": learning_rate_at(cursor, schedule),
            "learning_rate_last": learning_rate_at(cursor + max(stage.updates - 1, 0), schedule),
            "criteria": [c.as_dict() for c in criteria_for(stage.name)],
        })
        cursor += stage.updates
    return rows


__all__ = [
    "MAX_LBFGS_ITERATIONS",
    "SCHEDULE",
    "STAGE_CRITERIA",
    "TOTAL_UPDATES",
    "ScheduleError",
    "Stage",
    "StageCriterion",
    "cosine_learning_rate",
    "criteria_for",
    "describe_schedule",
    "learning_rate_at",
    "pde_weight_at",
    "stage_and_fraction",
    "time_window_fraction_at",
]
