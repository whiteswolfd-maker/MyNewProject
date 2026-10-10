"""验证隔离（提示词 §16）。

§16 规定 **500 mm 保持最终独立验证，不用于调参、初值标定、归一化极值、早停或
选模**。在这之前，这条规定只存在于配置里的一个字符串（``split:
extrapolation_test``）和一句 ``splits.strict_blind_test: true``——而后者**没有任何
代码读它**。也就是说，把那一行的 ``split`` 改成 ``train``，什么都不会报错，500 mm
就悄悄进了训练集，同时所有报表仍然显示"盲测半径已隔离"。

本模块把这句话变成可执行的检查。它检查的不是"意图"，而是四件**具体**的事：

1. 被声明为最终验证半径的算例，其 ``split`` 不能是 ``train``；
2. 训练算例的半径集合与最终验证半径集合不相交；
3. 参考 B 初场建立在**训练半径**上——初值标定也算"用过了"，500 mm 不能参与；
4. 声明了 ``strict_blind_test`` 就必须真的存在一个非训练的最终验证算例，
   否则这个声明是空的（"我们做了盲测"而根本没有盲测对象）。

归一化极值（``hat T_*``）与选模检查集在 :func:`~...cli._common.build_context`
里只由训练算例构成；这里把"它们确实只由训练算例构成"作为可核对的结论记下来。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: 默认的最终验证半径，mm。§16 逐字规定 500 mm。
DEFAULT_FINAL_VALIDATION_RADII_MM = (500.0,)

#: 允许进入训练的 split。
TRAINING_SPLIT = "train"


class IsolationError(RuntimeError):
    """验证隔离被破坏。"""


@dataclass
class ValidationIsolation:
    """验证隔离的判定结果。"""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": bool(self.ok), "reasons": list(self.reasons), **self.detail}

    def raise_if_broken(self) -> None:
        if not self.ok:
            raise IsolationError(
                "the final-validation radii are not isolated from training "
                "(section 16):\n  - " + "\n  - ".join(self.reasons)
            )


def final_validation_radii_m(config) -> tuple[float, ...]:
    """配置里声明的最终验证半径，单位 m。

    取自 ``splits.final_validation_radii_mm``；未声明时用 §16 的 500 mm。
    """
    declared = config.get("splits.final_validation_radii_mm", None)
    if declared is None:
        values = DEFAULT_FINAL_VALIDATION_RADII_MM
    elif isinstance(declared, (int, float)):
        values = (float(declared),)
    else:
        values = tuple(float(v) for v in declared)
    return tuple(value / 1000.0 for value in values)


def _case_specs(config) -> list[Mapping[str, Any]]:
    return list(config.get("data.cases", []) or [])


def evaluate_validation_isolation(
    config,
    training_cases: Sequence[Any],
    *,
    reference_radius_m: float | None = None,
    radius_tolerance_m: float = 1.0e-9,
) -> ValidationIsolation:
    """检查最终验证半径是否真的没有参与训练、初值标定与归一化。

    ``training_cases`` 是**实际进入训练**的算例（已经按 split 过滤过）。这里再
    独立核一遍半径，而不是信任调用方的过滤：过滤器写错时，两份信息里至少还有
    一份是对的。
    """
    reasons: list[str] = []
    protected = final_validation_radii_m(config)
    strict_blind = bool(config.get("splits.strict_blind_test", False))

    def same_radius(left: float, right: float) -> bool:
        return math.isclose(float(left), float(right), abs_tol=radius_tolerance_m)

    is_protected = lambda value: any(same_radius(value, r) for r in protected)  # noqa: E731

    # -- 1. 被保护的算例不得以 train 身份出现 ------------------------------
    specs = _case_specs(config)
    protected_specs = [
        spec for spec in specs
        if is_protected(float(spec.get("charge_radius_m", float("nan"))))
    ]
    for spec in protected_specs:
        split = str(spec.get("split", ""))
        if split == TRAINING_SPLIT:
            reasons.append(
                f"case {spec.get('case_id')!r} sits at a final-validation radius but is "
                f"declared split={split!r}; section 16 forbids it entering training, "
                "initial-field calibration, normalisation extremes, early stopping or "
                "model selection"
            )

    # -- 2. 训练集合与受保护半径不相交 -------------------------------------
    for case in training_cases:
        if is_protected(case.charge_radius):
            reasons.append(
                f"training case {case.case_id!r} has R_0 = {case.charge_radius * 1000:g} mm, "
                "which is a final-validation radius"
            )

    # -- 3. 初值标定用的半径必须是训练半径 ---------------------------------
    if reference_radius_m is not None:
        if is_protected(reference_radius_m):
            reasons.append(
                f"the reference initial field is built at R_0 = "
                f"{reference_radius_m * 1000:g} mm, a final-validation radius; "
                "initial-field calibration is one of the uses section 16 forbids"
            )

    # -- 4. 声明的盲测必须真的有对象 ---------------------------------------
    if strict_blind and protected and not any(
        str(spec.get("split", "")) != TRAINING_SPLIT for spec in protected_specs
    ):
        reasons.append(
            "splits.strict_blind_test is true but no case at a final-validation radius "
            "carries a non-training split; the declaration has no blind counterpart"
        )

    detail = {
        "final_validation_radii_mm": [round(value * 1000.0, 6) for value in protected],
        "strict_blind_test": strict_blind,
        "training_radii_mm": sorted(
            round(float(case.charge_radius) * 1000.0, 6) for case in training_cases
        ),
        "reference_radius_mm": (
            None if reference_radius_m is None else round(reference_radius_m * 1000.0, 6)
        ),
        "rule": (
            "section 16: the final-validation radius is never used for tuning, "
            "initial-field calibration, normalisation extremes, early stopping or "
            "model selection"
        ),
    }
    return ValidationIsolation(ok=not reasons, reasons=reasons, detail=detail)


__all__ = [
    "DEFAULT_FINAL_VALIDATION_RADII_MM",
    "TRAINING_SPLIT",
    "IsolationError",
    "ValidationIsolation",
    "evaluate_validation_isolation",
    "final_validation_radii_m",
]
