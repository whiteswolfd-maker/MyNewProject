"""训练层：提示词 §13 的三阶段日程与恢复训练。"""
from __future__ import annotations

from .schedule import (
    SCHEDULE,
    Stage,
    cosine_learning_rate,
    learning_rate_at,
    pde_weight_at,
)
from .trainer import Trainer, TrainingError, TrainingResult

__all__ = [
    "SCHEDULE",
    "Stage",
    "Trainer",
    "TrainingError",
    "TrainingResult",
    "cosine_learning_rate",
    "learning_rate_at",
    "pde_weight_at",
]
