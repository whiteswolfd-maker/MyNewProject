"""命令行之间共享的装配逻辑。

本模块只保留与初值方案无关的通用装配：

* 配置与命令行参数；
* 算例读取与数据兼容性判定；
* 尺度集合与 ``hat T_*`` 的冻结规则（仍由训练半径决定）。

**初值来源只有一个**：``python -m spherical_blast_pinn.cli.startup`` 写出的
平面黎曼启动 ``handoff``。:func:`build_context` 把它接到训练上下文、采样器和
共享模型；不回退到已经删除的旧参考场/前置求解方案。
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
from ..data.extracted_case import (
    DetectorSettings,
    ExtractedCase,
    FieldMapping,
    load_extracted_case,
)


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
# 算例
# --------------------------------------------------------------------------


def case_specs(config: Config, *, split: str | None = None) -> list[dict[str, Any]]:
    section = config.section("data")
    specs = list(section.get("cases", []) or [])
    if split is not None:
        specs = [s for s in specs if str(s.get("split")) == split]
    return specs


def load_cases(
    config: Config,
    *,
    split: str | None = None,
    strict: bool | None = None,
) -> list[ExtractedCase]:
    """读取算例；缺数据时的严格程度取决于它在这次运行里扮演什么角色。

    * ``split="train"``（以及显式 ``strict=True``）—— 缺一个训练算例是**致命**的。
      少一个半径意味着"三个半径等权联合更新"这句话不再成立，而那是最不该被
      静默降级的东西。
    * 其它情况（缺一个验证或盲测算例）—— 只是**那一次对照跑不了**。报告缺失，
      继续处理其余的。

    ``strict=None`` 时按 ``split == "train"`` 推断。
    """
    if strict is None:
        strict = split == "train"
    section = config.section("data")
    mapping = FieldMapping.from_mapping(section.get("field_map"))
    settings = DetectorSettings(
        **{
            key: value
            for key, value in (section.get("detector") or {}).items()
            if key in DetectorSettings.__dataclass_fields__
        }
    )
    cases: list[ExtractedCase] = []
    missing: list[str] = []
    for specification in case_specs(config, split=split):
        directory = Path(str(specification["directory"]))
        if not (directory / "raw_line.npz").exists():
            missing.append(f"{specification.get('case_id')} -> {directory}")
            continue
        case = load_extracted_case(
            directory,
            case_id=specification.get("case_id"),
            field_mapping=mapping,
            settings=settings,
            gamma_air=float(config.get("air.gamma", 1.4)),
        )
        case.source["split"] = specification.get("split")
        expected = float(specification.get("charge_radius_m", case.charge_radius))
        if abs(expected - case.charge_radius) > 1.0e-9:
            raise SetupError(
                f"case {case.case_id!r}: configuration says R_0 = {expected} m but the "
                f"data declares {case.charge_radius} m"
            )
        cases.append(case)
    if missing and strict:
        raise SetupError(
            "these configured cases have no readable dataset:\n  "
            + "\n  ".join(missing)
            + "\nMissing data is reported, never silently substituted."
        )
    if not cases:
        raise SetupError(f"no case matched split={split!r}")
    if missing:
        for case in cases:
            case.source.setdefault("notes", [])
        setattr(cases[0], "_missing_cases", list(missing))
    return cases


# --------------------------------------------------------------------------
# 数据兼容性报告
# --------------------------------------------------------------------------






# --------------------------------------------------------------------------
# 运行上下文
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    """一次运行的装配结果：新启动场算例 + 数据/隔离判定。"""

    config: Config
    startup: Any
    validation_isolation: Any = None
    #: 旧提取数据的契约结论。新链路从 ``startup.cases[].data`` 读取监督数据；
    #: 这里保留 ``None``，训练入口不会因此放行或阻塞。
    data_contract: Any = None
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

    @property
    def compatibility(self) -> dict[str, Any]:
        return {}

    @property
    def compatibility_report(self) -> None:
        return None

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
    with_model: bool = True,
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
        notes=[] if with_model else ["models were built regardless of with_model"],
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
    "case_specs",
    "jsonable",
    "load_cases",
    "load_config_from_args",
    "write_json",
]
