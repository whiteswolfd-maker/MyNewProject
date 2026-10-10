"""评价层：只保留与模型无关的量与图。

具体到爆炸量（超压、到达时间、正压持续时间、正相冲量）的对照，以及
predict / compare 的流程，由 ``cli`` 下的两个命令驱动。
"""
from __future__ import annotations

from . import metrics, plots

__all__ = ["metrics", "plots"]
