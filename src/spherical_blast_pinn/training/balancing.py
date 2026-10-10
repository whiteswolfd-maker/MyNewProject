"""有界梯度平衡与移动平均（提示词 §14）。

PINN 的各项损失梯度量级相差很多：强激波上的 RH 残差与近似均匀产物里的轴条件
可以差好几个数量级。固定权重只能二选一——要么忽略小的那项，要么让大的那项
主导共享主干。这里按**原方案**（``_legacy_v1_backup/losses/adaptive_weights.py``
里的 scheme，本模块按当前接口重写，**不导入**那个备份）测量每一项对共享参数的
梯度范数，再令

.. math::  \\lambda_k \\leftarrow \\hat g / g_k

其中 :math:`\\hat g` 是活跃项里最大的梯度范数。权重做指数移动平均后再**夹在
上下限之间**，单次噪声测量不足以动摇训练。

与原方案相比，这里补上提示词 §14 明确要求、而原方案没有的三件事：

1. **不拿近零梯度做除数**。原方案的分母是 ``norm + 1e-12``：一个梯度范数为
   1e-13 的项会拿到约 5e11 的目标权重，虽然随后被 ``max_weight`` 截住，但
   "截住"意味着它长期贴在**上限**上，等于把一个没有信息的项钉成最大权重。
   这里改用**相对下限**：范数低于"最大范数的 ``floor_ratio`` 倍"时，判定为
   "本步梯度证据不足"，**保持上一轮权重**，不做除法、不更新 EMA。
2. **零样本组不参与、也不更新统计**。缺标签、零样本的组，其未加权张量是
   ``requires_grad=False`` 的常数；这里直接跳过它，**不**把 0 当成一次真实
   测量写进 EMA（提示词 §14：不以伪零更新统计）。
3. **监测权重上下限与未加权残差**。记录每项触到上/下限的次数、被判为证据不足
   的次数、以及每项的**未加权**残差，作为可审查的输出。

``lambda_PDE`` **不在这里**。它的 0.1 → 1 爬坡是 §13 日程规定的，若同时由平衡器
接管，同一个数会被两个机制写，冲突的结果取决于更新顺序。所以平衡器只管
初始值/接触面/激波/数据/几何/可容许性这六组，PDE 权重仍由日程给出。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class BalancingError(RuntimeError):
    """平衡配置或状态不成立。"""


#: 平衡器**不**接管的组：``lambda_PDE`` 由 §13 日程规定。
SCHEDULE_OWNED = ("pde",)


@dataclass
class BalanceConfig:
    """有界梯度平衡的配置。默认值取自原方案。"""

    enabled: bool = True
    #: 每隔多少次更新测一次梯度范数。测一次要多做一遍反向，不宜每步都做。
    every: int = 50
    #: 移动平均系数：``ema <- alpha * ema + (1 - alpha) * target``。
    alpha: float = 0.9
    min_weight: float = 1.0e-3
    max_weight: float = 1.0e3
    #: 预热步数：这之前不做测量（此时梯度还没成形，量出来的是初值噪声）。
    warmup_steps: int = 200
    #: 相对下限。范数小于 ``floor_ratio * reference`` 的组判为"梯度证据不足"，
    #: 保持上一轮权重。用它替代原方案的 ``1e-12`` 绝对分母。
    floor_ratio: float = 1.0e-8

    def validate(self) -> None:
        if not (0.0 <= float(self.alpha) < 1.0):
            raise BalancingError(f"alpha must lie in [0, 1), got {self.alpha}")
        if int(self.every) < 1:
            raise BalancingError("every must be at least 1")
        if not (0.0 < float(self.min_weight) <= float(self.max_weight)):
            raise BalancingError("weights must satisfy 0 < min_weight <= max_weight")
        if int(self.warmup_steps) < 0:
            raise BalancingError("warmup_steps must be non-negative")
        if not (0.0 <= float(self.floor_ratio) < 1.0):
            raise BalancingError(
                f"floor_ratio must lie in [0, 1), got {self.floor_ratio}"
            )

    @classmethod
    def from_config(cls, config) -> "BalanceConfig":
        section = config.section("losses.adaptive")
        return cls(
            enabled=bool(section.get("enabled", True)),
            every=int(section.get("every", 50)),
            alpha=float(section.get("alpha", 0.9)),
            min_weight=float(section.get("min_weight", 1.0e-3)),
            max_weight=float(section.get("max_weight", 1.0e3)),
            warmup_steps=int(section.get("warmup_steps", 200)),
            floor_ratio=float(section.get("floor_ratio", 1.0e-8)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "every": int(self.every),
            "alpha": float(self.alpha),
            "min_weight": float(self.min_weight),
            "max_weight": float(self.max_weight),
            "warmup_steps": int(self.warmup_steps),
            "floor_ratio": float(self.floor_ratio),
            "schedule_owned": list(SCHEDULE_OWNED),
        }


@dataclass
class BalanceRecord:
    """一次平衡更新的审计记录。"""

    step: int
    weights: dict[str, float]
    gradient_norms: dict[str, float]
    unweighted: dict[str, float]
    held: dict[str, str] = field(default_factory=dict)
    at_bound: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "weights": dict(self.weights),
            "gradient_norms": dict(self.gradient_norms),
            "unweighted": dict(self.unweighted),
            "held": dict(self.held),
            "at_bound": dict(self.at_bound),
        }


class BoundedGradientBalancer:
    """按组测量梯度范数、给出有界权重。"""

    def __init__(
        self,
        base_weights: Mapping[str, float],
        config: BalanceConfig | None = None,
    ) -> None:
        config = config or BalanceConfig()
        config.validate()
        self.config = config
        self.base = {name: float(value) for name, value in base_weights.items()}
        #: 平衡器接管的组：基础权重非零、且不由日程掌管。
        self.adaptive = [
            name
            for name, value in self.base.items()
            if value != 0.0 and name not in SCHEDULE_OWNED
        ]
        self.current: dict[str, float] = {
            name: float(value) for name, value in self.base.items()
        }
        self.ema: dict[str, float] = {}
        self.history: list[BalanceRecord] = []
        self.bound_hits: dict[str, dict[str, int]] = {
            name: {"lower": 0, "upper": 0, "held": 0} for name in self.adaptive
        }
        self.updates = 0

    # -- 查询 -------------------------------------------------------------
    def weights(self) -> dict[str, float]:
        return dict(self.current)

    def adaptive_weights(self) -> dict[str, float]:
        """只含平衡器接管的六组，供写进运行清单。"""
        return {name: self.current[name] for name in self.adaptive}

    def should_update(self, step: int) -> bool:
        cfg = self.config
        if not cfg.enabled or not self.adaptive:
            return False
        if int(step) < int(cfg.warmup_steps):
            return False
        return int(step) % int(cfg.every) == 0

    # -- 更新 -------------------------------------------------------------
    def update(
        self,
        step: int,
        gradient_norms: Mapping[str, float],
        *,
        unweighted: Mapping[str, float] | None = None,
        live: Mapping[str, bool] | None = None,
    ) -> BalanceRecord:
        """用本步测得的梯度范数更新权重。

        ``live[name] is False`` 表示该组本步**不存在**（缺标签、零样本）——它的
        梯度范数多半是 0，但那个 0 不是测量结果。这样的组保持原权重，EMA 也不动
        （提示词 §14：不能以伪零更新统计）。
        """
        cfg = self.config
        held: dict[str, str] = {}
        at_bound: dict[str, str] = {}
        measured: dict[str, float] = {}

        for name in self.adaptive:
            raw = float(gradient_norms.get(name, float("nan")))
            if live is not None and not bool(live.get(name, True)):
                held[name] = "no_samples"
                self.bound_hits[name]["held"] += 1
                continue
            if not math.isfinite(raw):
                held[name] = "non_finite"
                self.bound_hits[name]["held"] += 1
                continue
            measured[name] = raw

        if not measured:
            # 一个可测量的组都没有：本步不产生新的权重，也不写历史。
            return BalanceRecord(
                step=int(step), weights=self.weights(), gradient_norms=dict(measured),
                unweighted=dict(unweighted or {}), held=held, at_bound=at_bound,
            )

        reference = max(measured.values())
        floor = float(cfg.floor_ratio) * reference
        for name, norm in measured.items():
            if norm <= floor:
                # 梯度证据不足：保持上一轮权重。**不**做除法——除以一个近零的
                # 数会得到一个天文数字的目标值，即使随后夹到上限，也等于把一个
                # 没有信息的项长期钉在上限上。
                held[name] = "negligible_gradient"
                self.bound_hits[name]["held"] += 1
                continue
            target = reference / norm
            if name in self.ema:
                smoothed = cfg.alpha * self.ema[name] + (1.0 - cfg.alpha) * target
            else:
                # 第一次测量直接用目标值，不从 1.0 爬过去。
                smoothed = target
            self.ema[name] = float(smoothed)
            clipped = float(np.clip(smoothed, cfg.min_weight, cfg.max_weight))
            if smoothed < cfg.min_weight:
                at_bound[name] = "lower"
                self.bound_hits[name]["lower"] += 1
            elif smoothed > cfg.max_weight:
                at_bound[name] = "upper"
                self.bound_hits[name]["upper"] += 1
            self.current[name] = clipped

        self.updates += 1
        record = BalanceRecord(
            step=int(step),
            weights=self.weights(),
            gradient_norms=dict(measured),
            unweighted=dict(unweighted or {}),
            held=held,
            at_bound=at_bound,
        )
        self.history.append(record)
        return record

    # -- 状态 -------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "base": dict(self.base),
            "current": dict(self.current),
            "ema": dict(self.ema),
            "bound_hits": {k: dict(v) for k, v in self.bound_hits.items()},
            "updates": int(self.updates),
            "config": self.config.as_dict(),
        }

    def load_state_dict(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        for name, value in (payload.get("current") or {}).items():
            if name in self.current:
                self.current[name] = float(value)
        self.ema = {k: float(v) for k, v in (payload.get("ema") or {}).items()}
        for name, counts in (payload.get("bound_hits") or {}).items():
            if name in self.bound_hits:
                self.bound_hits[name] = {k: int(v) for k, v in dict(counts).items()}
        self.updates = int(payload.get("updates", 0))

    def monitors(self) -> dict[str, Any]:
        """可写进运行清单的监测输出（提示词 §14）。"""
        out: dict[str, Any] = {
            "adaptive_weights": self.adaptive_weights(),
            "ema_gradient_norms": dict(self.ema),
            "updates": int(self.updates),
        }
        if self.history:
            last = self.history[-1]
            out["last_gradient_norms"] = dict(last.gradient_norms)
            out["last_unweighted_residuals"] = dict(last.unweighted)
            out["last_held"] = dict(last.held)
            out["last_at_bound"] = dict(last.at_bound)
        for name, counts in self.bound_hits.items():
            out[f"bound_hits.{name}"] = dict(counts)
        return out


def measure_gradient_norms(
    terms: Mapping[str, Any],
    parameters: Sequence[Any],
    *,
    retain_graph: bool = True,
) -> dict[str, float]:
    """每一项对共享参数的梯度范数。

    逐项单独求导，所以量到的是 ``d(term)/d(theta)`` 的范数，而不是某个组合的。
    用 ``torch.autograd.grad`` 而不用 ``backward``：后者会把梯度**累加**进
    ``.grad``，把随后的真正反传污染掉。

    ``retain_graph=True`` 是必须的——量完还要用同一张图做总损失的反传；图被释放
    掉的话，第一次测量之后训练就崩了。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    active = [p for p in parameters if p.requires_grad]
    if not active:
        return {}
    norms: dict[str, float] = {}
    for name, term in terms.items():
        # 没有可用样本的组返回的是 ``requires_grad=False`` 的常数，跳过即可：
        # 这一组本步不存在，不该往它的统计里写一个伪零。
        if not isinstance(term, torch.Tensor) or not term.requires_grad:
            continue
        grads = torch.autograd.grad(
            term, active, retain_graph=retain_graph, allow_unused=True
        )
        total = 0.0
        for grad in grads:
            if grad is None:
                continue
            total += float(grad.detach().pow(2).sum().item())
        norms[name] = float(math.sqrt(total))
    return norms


def live_groups(terms: Mapping[str, Any]) -> dict[str, bool]:
    """哪些组本步真的有样本（张量带梯度）。"""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    return {
        name: bool(isinstance(term, torch.Tensor) and term.requires_grad)
        for name, term in terms.items()
    }


__all__ = [
    "SCHEDULE_OWNED",
    "BalanceConfig",
    "BalanceRecord",
    "BalancingError",
    "BoundedGradientBalancer",
    "live_groups",
    "measure_gradient_norms",
]
