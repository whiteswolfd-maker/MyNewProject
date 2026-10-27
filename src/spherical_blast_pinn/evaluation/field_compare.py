"""材料侧场量对照（新链路与旧格式共用的一种实现）。

规则只有四条：

* **逐字段有效性**：一个字段在某点无效，只跳过该字段的那个点；
* **只在预测的同一材料块内插值**：产物点用产物块、空气点用空气块，绝不跨
  接触面/激波插值；落在块外的参考点不参与该字段的误差；
* **不外推**：参考半径超出预测块就丢弃，并单独计数；
* **缺失不是零**：没有可比较点时相对误差返回 ``None``，不是 0。

``counts`` 记录每个被丢弃的原因，保证"丢了多少、为什么丢"可核对。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def compare_field_blocks(
    *,
    reference_radius: np.ndarray,
    reference_values: np.ndarray,
    valid: np.ndarray,
    predicted_radius: np.ndarray,
    predicted_values: np.ndarray,
    low: float,
    high: float,
) -> dict[str, Any]:
    """在 ``[low, high]``（预测材料块的物理范围）内比较同一字段。"""
    reference_radius = np.asarray(reference_radius, dtype=np.float64).reshape(-1)
    reference_values = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    predicted_radius = np.asarray(predicted_radius, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted_values, dtype=np.float64).reshape(-1)
    if not (
        reference_radius.size == reference_values.size == valid.size
    ):
        raise ValueError("reference radius/value/validity sizes disagree")
    if predicted_radius.size != predicted_values.size:
        raise ValueError("predicted radius/value sizes disagree")

    n_reference = int(reference_radius.size)
    finite_reference = np.isfinite(reference_values)
    inside = (reference_radius >= float(low)) & (reference_radius <= float(high))
    invalid = ~valid
    nonfinite_reference = valid & ~finite_reference & inside
    outside = valid & finite_reference & ~inside
    usable = valid & finite_reference & inside

    counts = {
        "reference_points": n_reference,
        "invalid_field": int((invalid & inside).sum()),
        "non_finite_reference": int(nonfinite_reference.sum()),
        "outside_predicted_block": int(outside.sum()),
        "predicted_non_finite": 0,
        "compared": 0,
    }
    if predicted_radius.size < 2:
        return {
            "n_points": 0,
            "relative_l2": None,
            "rms": None,
            "counts": counts,
            "missing_reason": "prediction block has fewer than two radii",
        }
    if not bool(usable.any()):
        return {
            "n_points": 0,
            "relative_l2": None,
            "rms": None,
            "counts": counts,
            "missing_reason": (
                "no valid, finite reference point inside the predicted material block"
            ),
        }
    order = np.argsort(predicted_radius)
    predicted_radius = predicted_radius[order]
    predicted_values = predicted_values[order]
    interpolated = np.interp(
        reference_radius[usable], predicted_radius, predicted_values
    )
    finite_prediction = np.isfinite(interpolated)
    counts["predicted_non_finite"] = int((~finite_prediction).sum())
    if not bool(finite_prediction.any()):
        return {
            "n_points": 0,
            "relative_l2": None,
            "rms": None,
            "counts": counts,
            "missing_reason": "every interpolated prediction is non-finite",
        }
    target = reference_values[usable][finite_prediction]
    predicted = interpolated[finite_prediction]
    difference = predicted - target
    rms = float(np.sqrt(np.mean(difference**2)))
    norm = float(np.sqrt(np.mean(target**2)))
    relative = None if not math.isfinite(norm) or norm <= 0.0 else rms / norm
    counts["compared"] = int(target.size)
    return {
        "n_points": int(target.size),
        "relative_l2": relative,
        "rms": rms,
        "counts": counts,
    }


__all__ = ["compare_field_blocks"]
