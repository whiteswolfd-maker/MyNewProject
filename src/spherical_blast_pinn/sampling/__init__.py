"""采样层：按提示词 §12 为每个半径生成一个优化步所需的全部样本。"""
from __future__ import annotations

from .batch import SampleBatch
from .sampler import (
    AIR,
    PRODUCT,
    GroupedSampler,
    RadiusSampler,
    SamplingConfig,
    SamplingError,
)

__all__ = [
    "AIR",
    "PRODUCT",
    "GroupedSampler",
    "RadiusSampler",
    "SampleBatch",
    "SamplingConfig",
    "SamplingError",
]
