"""采样层：按提示词 §12 为每个半径生成一个优化步所需的全部样本。"""
from __future__ import annotations

from .batch import SampleBatch
from .planar_startup_initial import (
    PLANAR_STARTUP_IC_ADAPTER_VERSION,
    PlanarStartupBatchError,
    build_planar_startup_initial_batch,
    initial_radii_from_batch,
)
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
    "PLANAR_STARTUP_IC_ADAPTER_VERSION",
    "PlanarStartupBatchError",
    "RadiusSampler",
    "SampleBatch",
    "SamplingConfig",
    "SamplingError",
    "build_planar_startup_initial_batch",
    "initial_radii_from_batch",
]
