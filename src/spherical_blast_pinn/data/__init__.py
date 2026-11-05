"""数据层。

两条来源：

* :mod:`extracted_case` —— 旧格式的 ``(t, r, rho, u, P)`` 提取剖面；只作为
  ``dyna_supervision`` 的输入与旧审计命令的输入；
* :mod:`raw_reader` / :mod:`shell_average` —— 直接从 d3plot 读取、按材料
  体积分数做球壳平均的完整流程（提示词 §11 要求的能力）。

新训练链只消费 :class:`startup_supervision.SupervisionDataset`：
:mod:`dyna_supervision` 负责把旧提取剖面和显式 ``T_origin`` 转成它。旧的
``data.contract`` 已删除；监督资格由 ``startup_supervision`` 自己判定。

``specific_internal_energy`` 在两条路径上都不是输出字段，而是由 ``(rho, p)``
按材料各自的状态方程反解出来的；反解式在 :mod:`extracted_case` 与
:mod:`raw_reader` 中各有一份，且都由 ``cli.validate_eos`` 交叉检查。
"""
from __future__ import annotations

from . import (
    dyna_supervision,
    extracted_case,
    interface_reconstruction,
    isolation,
    kfile_parser,
    raw_reader,
    schema,
    shell_average,
    startup_supervision,
    units,
)

__all__ = [
    "dyna_supervision",
    "extracted_case",
    "interface_reconstruction",
    "isolation",
    "kfile_parser",
    "raw_reader",
    "schema",
    "shell_average",
    "startup_supervision",
    "units",
]
