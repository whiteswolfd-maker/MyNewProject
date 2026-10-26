"""命令行之间共享的装配逻辑。

本模块只保留与初值方案无关的通用装配：

* 配置与命令行参数；
* 算例读取与数据兼容性判定；
* 尺度集合与 ``hat T_*`` 的冻结规则（仍由训练半径决定）。

**初值来源只有一个**：``python -m spherical_blast_pinn.cli.startup`` 写出的
平面黎曼启动 ``handoff``。把它接到训练、预测与检查点的接口**尚未实现**，
因此正式的装配入口 :func:`build_context` 在这里明确失败，而不是回退到已经
删除的旧参考场/前置求解方案。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config, load_config
from ..data import compatibility as compat
from ..data.extracted_case import (
    DetectorSettings,
    ExtractedCase,
    FieldMapping,
    load_extracted_case,
)
from ..models.coupled_model import CoupledModel
from ..physics.similitude import SimilitudeScales
from ..sampling.sampler import SamplingConfig


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


def compatibility_report_path(config: Config) -> Path | None:
    """兼容性报告的位置。

    优先用 ``data.compatibility_report``；没设时退回 ``output_dir/dyna_audit/``
    下由 ``audit_dyna`` 写出的那一份。两处都没有就是"没有证据"，而不是"通过"。
    """
    section = config.section("data")
    declared = section.get("compatibility_report")
    if declared:
        return Path(str(declared))
    audit_dir = config.path_of("training.output_dir", default=Path("outputs/run")) / "dyna_audit"
    candidate = audit_dir / compat.REPORT_FILENAME
    return candidate if candidate.exists() else None


def resolve_case_compatibility(
    config: Config, cases: Sequence[ExtractedCase]
) -> tuple[dict[str, compat.CaseCompatibility], Path | None]:
    """把每个训练算例的兼容性状态凑齐。

    报告里没有的算例不是"默认通过"，而是 ``unknown``；它自己记录的分区来源仍然
    会被读一遍——那一项与报告无关，是算例数据本身的性质。
    """
    path = compatibility_report_path(config)
    stored = compat.load_report(path) if path is not None else {}
    resolved: dict[str, compat.CaseCompatibility] = {}
    for case in cases:
        entry = stored.get(case.case_id)
        partition, evidence = compat.material_partition_verdict(case)
        if entry is None:
            missing = compat.compatibility_from_findings(
                case.case_id,
                (),
                source=str(path) if path is not None else "no compatibility report found",
                material_partition=partition,
                material_partition_evidence=evidence,
            )
            missing.reasons.insert(
                0,
                "没有读到这个算例的初始化审查结论（"
                + (f"{path} 里没有它的条目" if path is not None
                   else "没有找到 data_compatibility.json；先跑 audit_dyna")
                + "）",
            )
            resolved[case.case_id] = missing
            continue
        # 报告里的分区状态若比算例自身的更乐观，以算例自身的为准：报告是快照，
        # 数据是当下的事实。
        if partition == compat.INCOMPATIBLE and entry.checks.get("material_partition") != compat.INCOMPATIBLE:
            entry.checks["material_partition"] = partition
            entry.reasons.append("material_partition：算例记录显示分区不是来自体积分数")
            entry.status = compat.INCOMPATIBLE
        entry.evidence.setdefault("material_partition", evidence)
        resolved[case.case_id] = entry
    return resolved, path


# --------------------------------------------------------------------------
# 运行上下文
# --------------------------------------------------------------------------


class TrainingNotConnected(RuntimeError):
    """新启动场的训练接口尚未接入。

    旧参考场（爆轰产物 + 拉格朗日双材料前置）已经删除，而把
    ``PlanarStartupField`` / ``handoff`` 接到采样器、损失尺度、训练器与检查点
    的工作还没有做。这里必须硬失败：两条时间轴、两套初值来源都只有秒和场的
    量纲，自动挑一个不会报错，只会把整条轨迹放到另一个物理时刻上。
    """


#: 正式训练/预测命令的统一提示。测试与用户都按这句话判断"是尚未接入"还是
#: "程序出错"，所以它只在一处定义。
TRAINING_NOT_CONNECTED_MESSAGE = (
    "新启动场的训练接口尚未接入：旧的参考场（爆轰产物 + 拉格朗日前置）已经删除，"
    "而 PlanarStartupField/handoff 还没有接到采样器、损失尺度、训练器和检查点。"
    "当前可用的只有 python -m spherical_blast_pinn.cli.startup（写出 handoff）与 "
    "python -m spherical_blast_pinn.cli.validate_startup_planar（delta 诊断）。"
)


@dataclass
class RunContext:
    """一次运行需要的全部对象（当前只在接口接通后才会被构造）。"""

    config: Config
    training_cases: list[ExtractedCase]
    time_scale: float
    time_scale_provenance: dict[str, Any]
    scales: dict[float, SimilitudeScales]
    model: CoupledModel | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    notes: list[str] = field(default_factory=list)
    compatibility: dict[str, compat.CaseCompatibility] = field(default_factory=dict)
    compatibility_report: str | None = None
    #: 数据契约判定（§15）。训练入口按它阻塞；predict/compare 只记录不阻塞。
    data_contract: Any = None
    #: 验证隔离判定（§16）。同样只在训练入口阻塞。
    validation_isolation: Any = None

    @property
    def training_radii(self) -> list[float]:
        return [float(c.charge_radius) for c in self.training_cases]

    def scales_for(self, charge_radius: float) -> SimilitudeScales:
        return self.scales[float(charge_radius)]

    def describe(self) -> dict[str, Any]:
        return {
            "training_cases": [c.describe() for c in self.training_cases],
            "training_radii": self.training_radii,
            "time_scale": self.time_scale,
            "time_scale_provenance": self.time_scale_provenance,
            "scales": {str(k): v.as_dict() for k, v in self.scales.items()},
            "sampling": {
                name: getattr(self.sampling, name)
                for name in SamplingConfig.__dataclass_fields__
            },
            "data_compatibility": {
                "report": self.compatibility_report,
                "cases": {k: v.as_dict() for k, v in self.compatibility.items()},
            },
            "data_contract": (
                None if self.data_contract is None else self.data_contract.as_dict()
            ),
            "validation_isolation": (
                None if self.validation_isolation is None
                else self.validation_isolation.as_dict()
            ),
            "notes": self.notes,
        }


def build_context(
    config: Config,
    *,
    verbose: bool = False,
    with_model: bool = True,
    seed: int | None = None,
) -> RunContext:
    """正式装配入口：当前明确失败（新启动场的训练接口尚未接入）。"""
    raise TrainingNotConnected(TRAINING_NOT_CONNECTED_MESSAGE)



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
    "TRAINING_NOT_CONNECTED_MESSAGE",
    "TrainingNotConnected",
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
