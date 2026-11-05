"""命令行之间共享的装配逻辑。

本模块只保留训练/预测入口真正共用的部分：

* 配置与命令行参数；
* 从 ``startup.cases`` 装配新启动场运行上下文。

**初值来源只有一个**：``python -m spherical_blast_pinn.cli.startup`` 写出的
平面黎曼启动 ``handoff``。:func:`build_context` 把它接到训练上下文、采样器和
共享模型；不回退到已经删除的旧参考场/前置求解方案。

旧 ``ExtractedCase`` 剖面的读取与审计已移到
:mod:`spherical_blast_pinn.cli._extracted_audit`，只供 ``audit_data`` /
``validate_eos`` 使用，训练入口不导入它。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config, load_config


class SetupError(RuntimeError):
    """运行的前置条件不满足。"""


def configure_stdout() -> None:
    """把标准输出切成 UTF-8。

    报告里有中文和 ``µ``、``ρ`` 一类字符，而 Windows 控制台默认是 GBK；不切换
    的话，报告写文件没问题，但打印到终端会直接抛 ``UnicodeEncodeError``，把一次
    已经算完的运行变成一次崩溃。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive
                pass


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="配置文件路径（YAML）")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="覆盖配置项，形如 --set training.max_updates=10（可重复）",
    )
    parser.add_argument("--verbose", action="store_true")


def load_config_from_args(args) -> Config:
    return load_config(args.config, args.overrides)


# --------------------------------------------------------------------------
# 运行上下文
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    """一次运行的装配结果：新启动场算例 + 隔离判定。"""

    config: Config
    startup: Any
    validation_isolation: Any = None
    notes: list[str] = field(default_factory=list)

    @property
    def training_cases(self) -> list[Any]:
        return list(self.startup.cases)

    @property
    def training_radii(self) -> list[float]:
        return [float(case.charge_radius) for case in self.startup.cases]

    @property
    def model(self):
        return self.startup.model

    @property
    def time_scale(self) -> float:
        return float(self.startup.time_scale)

    @property
    def sampling(self):
        return self.startup.sampling

    def scales_for(self, charge_radius: float):
        for case in self.startup.cases:
            if abs(case.charge_radius - float(charge_radius)) <= 1.0e-12:
                return case.scales
        raise SetupError(f"no startup case at R_0 = {charge_radius!r} m")

    def describe(self) -> dict[str, Any]:
        return {
            **self.startup.describe(),
            "validation_isolation": (
                None if self.validation_isolation is None
                else self.validation_isolation.as_dict()
            ),
            "notes": list(self.notes) + list(self.startup.notes),
        }


def build_context(
    config: Config,
    *,
    verbose: bool = False,
    seed: int | None = None,
    diagnostic: bool = False,
    device: str | None = None,
) -> RunContext:
    """从配置装配新启动场训练上下文。

    启动场不再因 ``training_qualified=False`` 被拒绝；诊断模式只限制运行身份与
    步数，不构成科研资格判断。程序正确性检查仍在这里硬失败。
    """
    from ..data.isolation import evaluate_validation_isolation
    from ..training.startup_cases import build_startup_training_context

    startup = build_startup_training_context(
        config,
        seed=int(seed if seed is not None else 0),
        diagnostic=bool(diagnostic),
        device=device,
    )
    isolation = evaluate_validation_isolation(config, startup.cases)
    return RunContext(
        config=config,
        startup=startup,
        validation_isolation=isolation,
        notes=[],
    )



def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "detach"):
        return float(value.detach())
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=jsonable), encoding="utf-8")
    return path


__all__ = [
    "RunContext",
    "SetupError",
    "add_common_arguments",
    "build_context",
    "configure_stdout",
    "jsonable",
    "load_config_from_args",
    "write_json",
]
