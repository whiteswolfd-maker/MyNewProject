"""数据层。

两条来源，共用一个逻辑字段契约：

* :mod:`extracted_case` —— 已提取的 ``(t, r, rho, u, P)`` 时空剖面；
* :mod:`raw_reader` / :mod:`shell_average` —— 直接从 d3plot 读取、按材料
  体积分数做球壳平均的完整流程（提示词 §11 要求的能力）。

``specific_internal_energy`` 在两条路径上都不是输出字段，而是由 ``(rho, p)``
按材料各自的状态方程反解出来的；反解式在 :mod:`extracted_case` 与
:mod:`raw_reader` 中各有一份，且都由 ``cli.validate_eos`` 交叉检查。
"""
from __future__ import annotations

from . import (
    contract,
    extracted_case,
    interface_reconstruction,
    isolation,
    kfile_parser,
    raw_reader,
    schema,
    shell_average,
    units,
)

__all__ = [
    "contract",
    "extracted_case",
    "interface_reconstruction",
    "isolation",
    "kfile_parser",
    "raw_reader",
    "schema",
    "shell_average",
    "units",
]
