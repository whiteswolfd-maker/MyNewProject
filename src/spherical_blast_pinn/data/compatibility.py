"""数据与**方案 B** 的兼容性状态，以及训练入口的强制检查（提示词 §3、§11）。

为什么需要这个模块
------------------

审查报告里写着"这份数据与方案 B 不一致"，而训练入口只看
"文件存在、半径对得上、有可监督帧"。于是"审查发现不兼容"与"训练照常开始"
可以同时成立——审查报告变成了一个漂亮的附件，而不是一道闸门。

这里把审查结论写成机器可读的**状态**，并让训练入口对状态负责：

* ``compatible`` —— 初始化、单位、EOS、时间定义、材料分区都审过，与方案 B 一致；
* ``incompatible`` —— 审过，并且**发现**与方案 B 冲突；
* ``unknown`` —— 没有审查证据，或者审查报告里该项是"缺证据"；
* ``synthetic`` —— 明确标注的合成数据，只能用于接口与数学关系检查。

正式训练只接受 ``compatible``。``unknown`` 与 ``incompatible`` 都会让运行停下
来；要越过它，必须在配置里显式写 ``data.allow_unverified_cases: true``，并且
结果与检查点会被打上 ``data_compatibility: unverified`` 的标记。

这不比"训练入口只看文件存不存在"更严格——它只是让审查结论能被代码读到。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

#: 兼容性状态。
COMPATIBLE = "compatible"
INCOMPATIBLE = "incompatible"
UNKNOWN = "unknown"
SYNTHETIC = "synthetic"

ALL_STATUSES = (COMPATIBLE, INCOMPATIBLE, UNKNOWN, SYNTHETIC)

#: 中文审查结论 -> 状态。``audit_dyna`` 的表格用的是人读的中文，机器读的是这里。
VERDICT_TO_STATUS = {
    "一致": COMPATIBLE,
    "不一致": INCOMPATIBLE,
    "缺证据": UNKNOWN,
}

#: 审查报告文件的格式标识与版本。
REPORT_FORMAT = "spherical_blast_pinn.data_compatibility"
REPORT_VERSION = 1
REPORT_FILENAME = "data_compatibility.json"

#: 判定"与方案 B 一致"必须全部通过的审查项关键字（``Finding.item`` 的前缀）。
#:
#: **方案 B 的一致性就是数据本身的一致**：中心起爆、有限爆速、
#: ``T_b = R_0/D`` 的时间零点、来自体积分数的材料分区。方案 A 把
#: "整球瞬时转化"当基准，那一版的判据会把合格的 B 算例判成不一致，
#: 而"不一致"会直接阻止训练——所以判据必须与初值方案同源。
#:
#: 物理初始化的判定（第 1、2 项）是二者的核心；单位制与 EOS 不通过时，后面
#: 每一个数值比较都失去意义；材料分区不通过时，监督标签本身是错的。
REQUIRED_CHECKS = {
    "detonation_model": "1. 整球瞬时初始化还是有限速度爆轰",
    "time_origin": "2. 时间零点代表什么",
    "unit_system": "5. 单位制",
    "jwl_parameters": "3a. JWL 系数与规范 §4 是否一致",
    "material_partition": "9. 材料分区是否来自体积分数",
}


class CompatibilityError(RuntimeError):
    """数据兼容性不满足训练的前置条件。"""


@dataclass
class CaseCompatibility:
    """一个算例的兼容性状态及其证据。"""

    case_id: str
    status: str = UNKNOWN
    checks: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if self.status not in ALL_STATUSES:
            raise CompatibilityError(
                f"unknown compatibility status {self.status!r}; expected one of {ALL_STATUSES}"
            )

    @property
    def trains_physical_model(self) -> bool:
        return self.status == COMPATIBLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "evidence": self.evidence,
            "source": self.source,
        }

    def summary_line(self) -> str:
        detail = f"（{'; '.join(self.reasons)}）" if self.reasons else ""
        return f"{self.case_id}: {self.status}{detail}"


def build_report(
    cases: Mapping[str, CaseCompatibility],
    *,
    generated_by: str = "",
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "generated_by": generated_by,
        "notes": list(notes),
        "cases": {case_id: entry.as_dict() for case_id, entry in cases.items()},
    }


def write_report(path: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return target


def load_report(path: Path | str) -> dict[str, CaseCompatibility]:
    """读兼容性报告。文件缺失时返回空字典——调用方据此判为 ``unknown``。"""
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if str(payload.get("format")) != REPORT_FORMAT:
        raise CompatibilityError(
            f"{target} is not a {REPORT_FORMAT} report (found {payload.get('format')!r})"
        )
    if int(payload.get("version", -1)) != REPORT_VERSION:
        raise CompatibilityError(
            f"{target} has report version {payload.get('version')}, expected {REPORT_VERSION}"
        )
    out: dict[str, CaseCompatibility] = {}
    for case_id, entry in (payload.get("cases") or {}).items():
        out[str(case_id)] = CaseCompatibility(
            case_id=str(case_id),
            status=str(entry.get("status", UNKNOWN)),
            checks={str(k): str(v) for k, v in (entry.get("checks") or {}).items()},
            reasons=[str(r) for r in (entry.get("reasons") or [])],
            evidence=dict(entry.get("evidence") or {}),
            source=str(entry.get("source", str(target))),
        )
    return out


# --------------------------------------------------------------------------
# 由审查表构造状态
# --------------------------------------------------------------------------


def compatibility_from_findings(
    case_id: str,
    findings: Sequence[Any],
    *,
    source: str = "",
    material_partition: str | None = None,
    material_partition_evidence: Mapping[str, Any] | None = None,
) -> CaseCompatibility:
    """把 ``audit_dyna`` 的 ``Finding`` 列表折成一个兼容性状态。

    ``findings`` 的每一项只要有 ``item`` 与 ``scheme_consistency`` 两个属性即可，
    因此这个函数不需要导入审查模块，也不会因为审查表的措辞变化而失效——它只在
    ``REQUIRED_CHECKS`` 里列出的那几项上做判定。

    **属性名是 ``scheme_consistency``**（方案 B 迁移时从 ``scheme_a`` 改来）。
    读取点若沿用旧名字，``getattr`` 会一路返回空串，于是**每一项都变成"缺证据"**
    ——闸门仍然拦住训练，但拦的理由是错的，而且报告里看不出真正的原因。所以这里
    显式区分"没有这一项"和"这一项的写法过时"。

    任何一项是"不一致"就是 ``incompatible``；任何一项"缺证据"就是 ``unknown``；
    全部"一致"才是 ``compatible``。这个偏序是刻意的：未知不等于通过。
    """
    checks: dict[str, str] = {}
    reasons: list[str] = []
    for key, prefix in REQUIRED_CHECKS.items():
        if key == "material_partition":
            continue
        verdict = None
        stale = False
        for finding in findings:
            if str(getattr(finding, "item", "")).startswith(prefix):
                if hasattr(finding, "scheme_consistency"):
                    verdict = str(getattr(finding, "scheme_consistency", ""))
                else:
                    # 方案 A 时代的审查表字段名。它的判据与方案 B 相反，不能拿来
                    # 沿用；报"过时的写法"而不是静默算成缺证据。
                    stale = True
                break
        if verdict is None:
            checks[key] = UNKNOWN
            reasons.append(
                f"{key}：审查表的字段名是方案 A 时代的写法（scheme_a），判据与方案 B "
                "相反，必须用当前版本的 audit_dyna 重新审查"
                if stale
                else f"{key}：审查表里没有这一项"
            )
            continue
        checks[key] = VERDICT_TO_STATUS.get(verdict, UNKNOWN)
        if checks[key] != COMPATIBLE:
            reasons.append(f"{key}：{verdict}")

    if material_partition is not None:
        checks["material_partition"] = material_partition
        if material_partition != COMPATIBLE:
            reasons.append(f"material_partition：{material_partition}")
    else:
        checks["material_partition"] = UNKNOWN
        reasons.append("material_partition：没有材料体积分区的审查证据")

    if INCOMPATIBLE in checks.values():
        status = INCOMPATIBLE
    elif all(value == COMPATIBLE for value in checks.values()):
        status = COMPATIBLE
    else:
        status = UNKNOWN

    evidence: dict[str, Any] = {}
    if material_partition_evidence:
        evidence["material_partition"] = dict(material_partition_evidence)
    return CaseCompatibility(
        case_id=case_id,
        status=status,
        checks=checks,
        reasons=reasons,
        evidence=evidence,
        source=source,
    )


#: 判定材料分区可用的下限。这些是**对提取结果**的检查门槛，不是对数据来源描述的
#: 检查门槛——一份"来源写着体积分数、结果里空气壳层全是 nan"的导出必须被拒。
#:
#: 分母取"主激波以内"的壳层，不取整张表：网格外半径是装药半径的几十倍，整张表
#: 大部分本来就是未受扰动环境态，拿它当分母会把一个完全正常的导出判成不合格。
MINIMUM_INTERIOR_FINITE_FRACTION = 0.9
MINIMUM_FINITE_LABELLED_FRACTION = 0.99


def material_partition_verdict(
    case,
    *,
    reload_from_disk: bool = True,
) -> tuple[str, dict[str, Any]]:
    """判定材料分区**是否真的可用**，而不只是"来源写着体积分数"。

    只看来源说明是不够的：一个把壳层标成空气、却没给空气填任何场量的导出，来源
    说明一样漂亮，而它的空气监督是空的——采样器会把那些 ``nan`` 标签逐个丢掉，
    于是数据损失里的空气组悄无声息地消失。所以这里检查的是结果本身：

    * ``alpha`` 有有限值、落在 ``[0, 1]``；
    * 材料标签与 ``alpha`` 阈值自洽（产物 ``>= 0.95``、空气 ``<= 0.05``）；
    * **每一个被标为有效的壳层都带有限的 ``rho/u/e/p``**；
    * 有限值与有效壳层的比例；
    * 从磁盘重新读一遍，与内存中的这张表逐元素一致。

    任何一条不满足就是 ``incompatible``。分区通过**也不**代表起爆方式与时间零点
    的检查通过——那是另外的项，各自独立判定。
    """
    source = getattr(case, "source", {}) or {}
    described = str(source.get("material_masks_from", ""))
    if "volume fraction" not in described.lower():
        return INCOMPATIBLE, {
            "material_masks_from": described,
            "has_volume_fraction_field": False,
            "material_companion": source.get("material_companion"),
            "rejected_frames_below_charge_radius": int(
                source.get("frames_rejected_contact_below_charge_radius", 0)
            ),
            "reason": (
                "the partition is a heuristic on the density profile, not a read-out of "
                "the ALE material volume fraction"
            ),
        }

    import numpy as np

    material = np.asarray(case.material)
    valid = np.asarray(case.valid)
    density = np.asarray(case.density)
    velocity = np.asarray(case.velocity)
    energy = np.asarray(case.specific_energy)
    pressure = np.asarray(case.pressure)
    companion = source.get("material_companion")
    alpha = None
    shell_count = None
    block = None
    if companion is not None:
        try:
            block = np.load(companion, allow_pickle=True)
            alpha = np.asarray(block["alpha_product"], dtype=np.float64)
            if "shell_count" in block.files:
                shell_count = np.asarray(block["shell_count"], dtype=np.int64)
        except Exception:  # pragma: no cover - defensive
            alpha = None

    finite = (
        np.isfinite(density) & np.isfinite(velocity) & np.isfinite(energy) & np.isfinite(pressure)
    )
    labelled = material != 2
    problems: list[str] = []
    evidence: dict[str, Any] = {
        "material_masks_from": described,
        "has_volume_fraction_field": True,
        "material_companion": source.get("material_companion"),
        "mixed_cell_fraction_max": source.get("mixed_cell_fraction_max"),
        "rejected_frames_below_charge_radius": int(
            source.get("frames_rejected_contact_below_charge_radius", 0)
        ),
        "shape": list(material.shape),
    }

    if alpha is None:
        problems.append("the companion volume-fraction array could not be read")
    else:
        # 空壳层（网格比单元还细，壳层里一个单元中心都没有）**本来就没有** alpha。
        # 那不是数据错误，是分辨率不匹配，必须与"域内出现了非有限值"分开算。
        occupied = np.ones_like(alpha, dtype=bool)
        if shell_count is not None and shell_count.shape == alpha.shape:
            occupied = shell_count > 0
            evidence["empty_shell_fraction"] = float(np.mean(~occupied))
        missing_inside = occupied & ~np.isfinite(alpha)
        evidence["alpha_missing_inside_domain"] = int(np.count_nonzero(missing_inside))
        finite_alpha = alpha[occupied]
        evidence["alpha_min"] = float(np.min(finite_alpha)) if finite_alpha.size else float("nan")
        evidence["alpha_max"] = float(np.max(finite_alpha)) if finite_alpha.size else float("nan")
        evidence["alpha_finite_fraction"] = (
            float(np.mean(np.isfinite(finite_alpha))) if finite_alpha.size else 0.0
        )
        if np.any(missing_inside):
            problems.append(
                f"{int(np.count_nonzero(missing_inside))} shell(s) contain cells but have "
                "no volume fraction"
            )
        if evidence["alpha_min"] < -1.0e-9 or evidence["alpha_max"] > 1.0 + 1.0e-9:
            problems.append(
                f"alpha left [0, 1]: min {evidence['alpha_min']:.6g}, "
                f"max {evidence['alpha_max']:.6g}"
            )
        elif alpha.shape == material.shape:
            inconsistent = ((material == 0) & (alpha < 0.95)) | ((material == 1) & (alpha > 0.05))
            evidence["label_alpha_inconsistencies"] = int(np.count_nonzero(inconsistent))
            if evidence["label_alpha_inconsistencies"]:
                problems.append(
                    f"{evidence['label_alpha_inconsistencies']} shells are labelled product "
                    "or air but their volume fraction does not match that label"
                )
        else:
            problems.append(
                f"alpha has shape {alpha.shape} but the material table is {material.shape}"
            )

    labelled_count = int(np.count_nonzero(labelled))
    labelled_finite = int(np.count_nonzero(labelled & finite))
    evidence["labelled_shells"] = labelled_count
    evidence["labelled_shells_with_finite_fields"] = labelled_finite
    evidence["finite_shell_fraction"] = float(np.mean(finite))
    for label, name in ((0, "product"), (1, "air")):
        mask = material == label
        evidence[f"{name}_shells"] = int(np.count_nonzero(mask))
        evidence[f"{name}_shells_with_finite_fields"] = int(np.count_nonzero(mask & finite))
    if labelled_count == 0:
        problems.append("no shell carries a product or air label")

    # 标了材料却没有状态的壳层，**必须被记录在案地解释掉**。
    #
    # 原来这里是一条"已标壳层里至少 99% 要有有限场量"的比例门槛。比例门槛抓得住
    # "空气整片是 ``nan``"那种成片漏洞，却抓不住更要紧的东西，也误伤真实的物理：
    #
    # * 真实的物理：装药里**尚未释放化学能**的单元没有产物状态（实测 50 mm 第 0 帧
    #   218328 个 ``alpha >= 0.95`` 的单元全部如此，``p`` 恰为 0）。这些壳层没有能量
    #   标签是**对的**，把它们算成"标签漏洞"等于要求提取器编一个数出来。
    # * 更要紧的东西：一个"99% 合格"的导出可以在剩下的 1% 里藏一整个静默消失的空气组。
    #
    # 所以换成精确判据：每个"标了材料却没有有限场量"的壳层必须落在提取器写出的
    # ``contaminated_shell_fraction > 0`` 里。**没法解释的漏洞必须是零个**——不是
    # "比例很低"。这样成片漏洞抓得住，而"这个单元本来就没有产物状态"不会被误判。
    contaminated = None
    if block is not None and "contaminated_shell_fraction" in block.files:
        contaminated = np.asarray(block["contaminated_shell_fraction"], dtype=np.float64)
    if contaminated is None or contaminated.shape != labelled.shape:
        contaminated = np.zeros_like(labelled, dtype=bool)
        evidence["contamination_is_recorded"] = False
    else:
        contaminated = np.isfinite(contaminated) & (contaminated > 0.0)
        evidence["contamination_is_recorded"] = True
    explained_missing = int(np.count_nonzero(labelled & ~finite & contaminated))
    unexplained = labelled & ~finite & ~contaminated
    unexplained_missing = int(np.count_nonzero(unexplained))
    evidence["labelled_shells_without_state_explained_by_contamination"] = explained_missing
    evidence["labelled_shells_without_state_unexplained"] = unexplained_missing
    evidence["contaminated_shell_fraction"] = float(np.mean(contaminated))
    if unexplained_missing:
        problems.append(
            f"{unexplained_missing} shell(s) are labelled but carry no finite rho/u/e/p and "
            "are not flagged as contaminated; a label without a state cannot supervise "
            "anything, and only a recorded reason makes the hole acceptable"
        )
    still_finite = int(np.count_nonzero(labelled & ~contaminated & finite))
    still_total = int(np.count_nonzero(labelled & ~contaminated))
    if still_total and still_finite < MINIMUM_FINITE_LABELLED_FRACTION * still_total:
        problems.append(
            f"only {still_finite} of {still_total} labelled, uncontaminated shells carry "
            "finite rho/u/e/p"
        )
    for name in ("product", "air"):
        if evidence[f"{name}_shells"] and not evidence[f"{name}_shells_with_finite_fields"]:
            problems.append(f"every {name} shell is labelled but carries no field values")

    # **有效必须蕴含"有状态"。** 这条是审查里那个"标成空气、场量却为空"的直接判据：
    # 一个 ``valid=True`` 却没有有限场量的壳层会被采样器逐个丢掉，数据损失里的那一组
    # 于是静默消失。必须为**零个**，不是"比例很低"。
    valid_without_state = int(np.count_nonzero(valid & ~finite))
    evidence["valid_shells_without_finite_fields"] = valid_without_state
    if valid_without_state:
        problems.append(
            f"{valid_without_state} shell(s) are marked valid but carry no finite rho/u/e/p"
        )

    # 主激波以内的有限值比例。混合带本来就该是空的，所以门槛留在 0.9。
    #
    # 分母要**排掉被记录在案的污染壳层**：装药里尚未释放化学能的单元没有产物状态，
    # 那是仿真的读数，不是提取的缺口。把分母算成"激波以内全部壳层"的话，一个早期帧
    # 只要还没起爆完就会被判不合格——而那时候"没有产物状态"恰恰是正确的结论。
    # 未记录的缺口仍然要满足 0.9，这一条没有放松。
    shock = np.asarray(case.shock_radius, dtype=np.float64).reshape(-1, 1)
    inside = np.isfinite(shock) & (np.asarray(case.radius)[None, :] < shock)
    interior_total = int(np.count_nonzero(inside))
    if interior_total:
        interior_finite = int(np.count_nonzero(inside & finite))
        accounted = inside & ~contaminated
        accounted_total = int(np.count_nonzero(accounted))
        accounted_finite = int(np.count_nonzero(accounted & finite))
        evidence["interior_shells"] = interior_total
        evidence["interior_shells_with_finite_fields"] = interior_finite
        evidence["interior_finite_fraction"] = interior_finite / interior_total
        evidence["interior_shells_excluding_contaminated"] = accounted_total
        evidence["interior_finite_fraction_excluding_contaminated"] = (
            accounted_finite / accounted_total if accounted_total else float("nan")
        )
        if (
            accounted_total
            and evidence["interior_finite_fraction_excluding_contaminated"]
            < MINIMUM_INTERIOR_FINITE_FRACTION
        ):
            problems.append(
                "only "
                f"{evidence['interior_finite_fraction_excluding_contaminated']:.3f} of the "
                "shells inside the shock that are NOT flagged as contaminated carry "
                "finite fields"
            )

    if reload_from_disk and not problems:
        directory = source.get("directory")
        if directory:
            try:
                from .extracted_case import load_extracted_case

                reloaded = load_extracted_case(directory, case_id=case.case_id)
            except Exception as exc:  # pragma: no cover - defensive
                problems.append(f"the saved extraction could not be read back: {exc}")
            else:
                same = (
                    np.array_equal(np.asarray(reloaded.material), material)
                    and np.array_equal(np.asarray(reloaded.valid), valid)
                    and np.allclose(
                        np.asarray(reloaded.density), density, rtol=0.0, atol=0.0, equal_nan=True
                    )
                )
                evidence["reload_matches_in_memory"] = bool(same)
                if not same:
                    problems.append(
                        "the table read back from disk differs from the one in memory"
                    )

    evidence["problems"] = problems
    return (COMPATIBLE if not problems else INCOMPATIBLE), evidence


def synthetic_case(case_id: str, *, source: str, reason: str = "") -> CaseCompatibility:
    """明确标注的合成样例。它**不能**用于正式监督训练。"""
    return CaseCompatibility(
        case_id=case_id,
        status=SYNTHETIC,
        checks={key: SYNTHETIC for key in REQUIRED_CHECKS},
        reasons=[reason or "explicitly flagged synthetic; not DYNA reference data"],
        source=source,
    )


# --------------------------------------------------------------------------
# 训练入口的闸门
# --------------------------------------------------------------------------


@dataclass
class GateDecision:
    """闸门的结论。``allowed`` 为假时 ``message`` 说明为什么。"""

    allowed: bool
    mode: str
    unverified: bool
    statuses: dict[str, str]
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "mode": self.mode,
            "data_compatibility": "unverified" if self.unverified else "verified",
            "statuses": self.statuses,
            "message": self.message,
        }


def enforce_training_gate(
    case_ids: Sequence[str],
    statuses: Mapping[str, CaseCompatibility],
    *,
    allow_unverified: bool = False,
    require_compatible: bool = True,
) -> GateDecision:
    """训练入口的强制检查。

    ``require_compatible=False`` 只用于明确标注的合成冒烟路径：那时数据本来就
    不是 DYNA 参考解，要求它"与方案 B 一致"没有意义，但**必须**标注成合成。

    返回 :class:`GateDecision`；不通过时抛 :class:`CompatibilityError`。返回一个
    对象而不是只抛异常，是为了让调用方能把 ``data_compatibility`` 写进检查点与
    运行清单——一次在未核实数据上跑出来的结果，必须在它自己的元数据里说得清楚。
    """
    resolved = {
        case_id: statuses.get(case_id, CaseCompatibility(case_id=case_id)) for case_id in case_ids
    }
    status_map = {case_id: entry.status for case_id, entry in resolved.items()}

    if not require_compatible:
        offenders = [entry for entry in resolved.values() if entry.status != SYNTHETIC]
        if offenders:
            raise CompatibilityError(
                "this run is configured as a synthetic smoke check, but these cases are "
                "not flagged synthetic: "
                + "; ".join(entry.summary_line() for entry in offenders)
                + ". Synthetic runs must use explicitly synthetic data; they may not "
                "quietly consume real cases."
            )
        return GateDecision(
            allowed=True,
            mode="synthetic_smoke",
            unverified=True,
            statuses=status_map,
            message="synthetic smoke check: results carry no physical claim",
        )

    blocking = [entry for entry in resolved.values() if entry.status != COMPATIBLE]
    if not blocking:
        return GateDecision(allowed=True, mode="production", unverified=False, statuses=status_map)

    detail = "\n  ".join(entry.summary_line() for entry in blocking)
    if allow_unverified:
        return GateDecision(
            allowed=True,
            mode="production_unverified",
            unverified=True,
            statuses=status_map,
            message=(
                "data compatibility is NOT verified for:\n  "
                + detail
                + "\nRunning anyway because data.allow_unverified_cases = true. The "
                "checkpoint and run manifest are stamped data_compatibility = unverified, "
                "and no result from this run may be reported as a physical validation."
            ),
        )
    raise CompatibilityError(
        "training refused: these cases are not cleared for scheme A supervision:\n  "
        + detail
        + "\nRun `python -m spherical_blast_pinn.cli.audit_dyna` to produce a "
        "compatibility report, then set data.compatibility_report to it. To train "
        "anyway on unaudited data, set data.allow_unverified_cases: true explicitly — "
        "the result will be stamped unverified."
    )


__all__ = [
    "ALL_STATUSES",
    "COMPATIBLE",
    "CompatibilityError",
    "CaseCompatibility",
    "GateDecision",
    "INCOMPATIBLE",
    "REPORT_FILENAME",
    "REPORT_FORMAT",
    "REPORT_VERSION",
    "REQUIRED_CHECKS",
    "SYNTHETIC",
    "UNKNOWN",
    "VERDICT_TO_STATUS",
    "build_report",
    "compatibility_from_findings",
    "enforce_training_gate",
    "load_report",
    "material_partition_verdict",
    "synthetic_case",
    "write_report",
]
