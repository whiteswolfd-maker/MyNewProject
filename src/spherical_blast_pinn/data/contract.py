"""数据契约（提示词 §15）。

这一层回答一个问题：**这批数据能不能拿来做正式训练？** 它不回答"数据在物理上
对不对"——那是初场验收与物理依据的事，两者分开记录、分别阻塞。

设计原则（§15）：

* **缺失必须显式**。布尔有效性不得由 ``value != 0`` 推断——零速度是合法值。缺什么
  就报什么，不补零、不默认为通过。
* **数据版本覆盖提取器、材料映射、几何、前沿检测、字段有效性和观测语义**。只把
  ``scheme`` 名字改成 B、却继续用旧数组语义，是配置迁移最常见的假完成。
* 契约状态**不是数值代码能自己填的**：``evaluate_data_contract`` 只说数据这一侧，
  它把结论交给调用方，由调用方与初场、物理适用性**分开**判定。

一个重要后果：用启发式探测器（密度最陡下降）定出来的材料分区**不可靠**——它给出
的接触面位置有量化跳变，且大量帧落在装药半径以内（物理上不可能）。这类数据在契约
上是 **not ready**，理由要写清楚：权威分区必须从 d3plot 的材料体积分数重新提取。
这不是"门槛太严"，是那份数据确实还没准备好。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: 契约结构的版本。**字段含义变了就要 +1**——提取器、材料映射、几何、前沿检测、
#: 字段有效性、观测语义，任何一项的语义变化都会让旧数据不再满足新契约。
CONTRACT_VERSION = 2

#: 需要的球对称份额。本项目的模型是八分之一模型，展开到整球乘 8。
DEFAULT_SYMMETRY_FACTOR = 8.0


@dataclass
class ContractVerdict:
    """数据契约的判定结果。``ready`` 为假时 ``reasons`` 必须非空。"""

    ready: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "ready": self.ready,
            "reasons": list(self.reasons),
            "reason": "就绪" if self.ready else "；".join(self.reasons),
            **self.detail,
        }


#: 逐字段有效性必须覆盖的字段。缺一个就不是"逐字段"，是"部分字段"。
REQUIRED_VALIDITY_FIELDS = ("density", "velocity", "specific_energy")

#: 前沿资格来源里表示"什么都没记录"的那一档。契约必须拒绝它。
UNVERIFIED_ELIGIBILITY = "none recorded — UNVERIFIED"


def _symmetry_factor(source: Mapping[str, Any]) -> tuple[float | None, list[str]]:
    """球对称份额。**没有明确声明就不算有**（§8：覆盖率分母用实际对称份额）。

    返回值是 ``(份额, 问题列表)``。份额必须是**有限正数**——以前只判"真值"，
    于是 ``-8`` 与 ``nan`` 都能通过（两者都是真值）。同时，两个键都给出来时必须
    自洽：``symmetry_factor`` 与 ``symmetry_share`` 互为倒数，1/8 与 8 说的是同一件
    事，而 8 与 8 说的不是。
    """
    problems: list[str] = []
    factor = None
    share = source.get("symmetry_share")

    # 别名全部收集起来，**互相核对**。以前是"找到第一个就 break"：同时写着
    # ``symmetry_factor: 8`` 与 ``full_sphere_factor: 4`` 时，读第一个的人算出 8，
    # 读第二个的人算出 4，而契约只说"有"。
    aliases: dict[str, float] = {}
    for key in ("symmetry_factor", "symmetry_share_inverse", "full_sphere_factor"):
        value = source.get(key)
        if value is None:
            continue
        try:
            aliases[key] = float(value)
        except (TypeError, ValueError):
            # 非数字直接抛裸 ``ValueError`` 会让契约这个"报告工具"变成崩溃源；
            # 契约的职责是把问题写成理由。
            problems.append(
                f"{key} = {value!r} 不是数字，无法解析成球对称份额"
            )
            return None, problems
    if aliases:
        distinct = sorted(set(aliases.values()))
        if len(distinct) > 1:
            problems.append(
                f"球对称份额的别名互相矛盾：{aliases}。它们说的是同一件事，"
                "不一致时读数与折算会各按自己看到的那个来"
            )
            return None, problems
        factor = distinct[0]
    if share is not None:
        try:
            declared_share = float(share)
        except (TypeError, ValueError):
            problems.append(f"symmetry_share = {share!r} 不是数字")
            return None, problems
    else:
        declared_share = None

    if factor is None and declared_share is not None:
        if declared_share == 0.0:
            problems.append("symmetry_share 是 0，取倒数得不到球对称份额")
            return None, problems
        factor = 1.0 / declared_share

    if factor is None:
        return None, problems
    if not math.isfinite(factor) or factor <= 0.0:
        problems.append(
            f"球对称份额是 {factor!r}，不是有限正数；体积等效半径与覆盖率的分母"
            "用它，负值或 NaN 会让折算出来的每一个半径都失去意义"
        )
        return None, problems
    if declared_share is not None:
        if not math.isfinite(declared_share) or declared_share <= 0.0:
            problems.append(
                f"symmetry_share 是 {declared_share!r}，不是有限正数（它与 "
                "symmetry_factor 互为倒数）"
            )
        elif abs(factor * declared_share - 1.0) > 1.0e-9:
            problems.append(
                f"symmetry_factor = {factor:g} 与 symmetry_share = {declared_share:g} "
                "互相矛盾：互为倒数的两个数应当相乘得 1。两个键都在时它们必须指同一"
                "件事，否则读数的人会按先看到的那个去折算"
            )
    return factor, problems


def evaluate_case_contract(case: Any) -> ContractVerdict:
    """单个算例的契约检查。"""
    reasons: list[str] = []
    detail: dict[str, Any] = {"case_id": getattr(case, "case_id", None)}
    source = dict(getattr(case, "source", {}) or {})

    # -- 算例标识与几何 ---------------------------------------------------
    if not getattr(case, "case_id", None):
        reasons.append("算例缺少 case_id")
    radius = float(getattr(case, "charge_radius", 0.0) or 0.0)
    if not (radius > 0.0):
        reasons.append("算例缺少有效的装药半径 R_0")
    detail["charge_radius_m"] = radius

    symmetry, symmetry_problems = _symmetry_factor(source)
    detail["symmetry_factor"] = symmetry
    if symmetry is None and not symmetry_problems:
        reasons.append(
            "没有声明球对称份额（symmetry_factor / symmetry_share）："
            "体积等效半径与覆盖率的分母都依赖它，缺了就无从核对"
        )
    reasons.extend(symmetry_problems)

    # -- 时间语义（方案 B）------------------------------------------------
    # ``T_h = R_0/D + epsilon`` 需要卡片爆速；缺了它算例本身就会抛错，这里要
    # 把它变成一个可读的契约理由，而不是让调用方去吃一个异常。
    try:
        detonation_velocity = float(case.detonation_velocity)
        detail["detonation_velocity_m_s"] = detonation_velocity
    except Exception as exc:  # noqa: BLE001 - 想把它变成理由而不是崩溃
        detonation_velocity = None
        reasons.append(f"时间原点无法形成（缺卡片爆速）：{exc}")

    if detonation_velocity is not None:
        try:
            detail["start_time_s"] = float(case.start_time)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"交接时刻 T_h 算不出来：{exc}")

    times = np.asarray(getattr(case, "times", np.zeros(0)), dtype=np.float64)
    if times.size == 0:
        reasons.append("算例没有任何时间帧")
    elif not np.all(np.isfinite(times)):
        # 非有限的时间轴不能进任何插值或分层：``nan`` 会让排序、``searchsorted``
        # 与时间窗口判断全部给出无意义的结果，而它们都不报错。
        reasons.append(
            f"时间轴含 {int(np.count_nonzero(~np.isfinite(times)))} 个非有限值"
        )
    elif detonation_velocity is None:
        # 时间轴建不起来时 ``pinn_times`` 会抛 ``DataError``。契约检查是**报告**
        # 的工具，不能自己崩掉——把那句异常变成一条可读的理由（上面已经记过
        # "时间原点无法形成"），这里只补上帧数。
        detail["n_frames"] = int(times.size)
    else:
        try:
            pinn = np.asarray(case.pinn_times, dtype=np.float64)
            usable = np.asarray(case.supervisable_frames())
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"无法判断可监督帧：{exc}")
        else:
            detail["n_frames"] = int(times.size)
            detail["n_frames_supervisable"] = int(np.count_nonzero(usable))
            if not np.any(usable):
                reasons.append(
                    "启动时刻之后没有任何可监督帧（前沿检测没有产出可靠的前沿）"
                )
            else:
                detail["t_pinn_last_valid_s"] = float(pinn[usable].max())
                if float(pinn[usable].max()) <= 0.0:
                    reasons.append("可监督时间范围非正，训练窗口为空")

    # -- 材料来源：分区必须是**权威**的，不能是启发式探测器 ---------------
    material_source = str(source.get("material_masks_from", "") or "")
    detail["material_masks_from"] = material_source or "未记录"
    # **按机器可判的标志判断，不按字符串匹配。** 以前这里是
    # ``"volume fraction" in material_source.lower()``：一句"not from volume
    # fraction"里也含 "volume fraction"，于是**否认**会被读成**肯定**。说明文字
    # 是给人看的，闸门必须看布尔量。
    declared = source.get("material_partition_is_volume_fraction_based")
    authoritative = declared is True
    detail["material_partition_is_volume_fraction_based"] = declared
    # **标志之外还要有来源记录。** 只有一个布尔量时，那个 True 是谁写的、依据是什么
    # 都无从核对——上一轮的教训正是"判据从信任说明文字换成信任读取器"。标志与来源
    # 记录必须同时存在，且各说各的：标志说"是不是"，来源记录说"从哪来的"。
    recorded_source = str(source.get("material_partition_source_recorded") or "").strip()
    detail["material_partition_source_recorded"] = recorded_source or None
    if not authoritative:
        recorded = (
            "提取器没有留下这个标志（未核实）" if declared is None
            else f"提取器明确记为 {declared!r}"
        )
        reasons.append(
            "材料分区不是从 d3plot 的材料体积分数读出来的"
            f"（{recorded}；说明文字：{material_source or '未记录'}）：启发式探测器"
            "给出的接触面位置有量化跳变、大量帧落在装药半径以内（物理上不可能）。"
            "契约上判为未就绪——权威分区必须重新提取"
        )
    elif not recorded_source:
        reasons.append(
            "材料分区声明为体积分数来源，但**没有留下结构化来源记录**"
            "（material_partition_source）：只有一个是/否标志时，那个 True 是谁写的、"
            "依据是什么都无从核对"
        )

    # -- 每个场量各自的有效性（§11.1）-------------------------------------
    # ``valid`` 目前是一张整帧掩码，rho/u/e 共用。契约要求分别记录。
    #
    # 判据是 ``is None``，**不是** ``hasattr``：这两个属性是带默认值的字段，
    # ``hasattr`` 恒为真。用 ``hasattr`` 会让"没记录"和"都有效"在契约上变成同一
    # 件事——一句永远通过的检查比没有检查更糟。
    field_validity = getattr(case, "field_validity", None)
    detail["field_validity_fields"] = (
        None if field_validity is None else sorted(field_validity)
    )
    if field_validity is None:
        reasons.append(
            "算例没有逐字段有效性（field_validity）：rho/u/e 共用一张有效掩码时，"
            "一个字段缺失会丢掉同一位置其他可靠的字段（§11.1）"
        )
    else:
        # ``is None`` 只堵住了"属性存在但没有值"。**空字典与半张表同样是没记录。**
        # ``field_validity={}`` 以前直接通过——一份什么都没说的记录被读成"都有效"。
        missing = [name for name in REQUIRED_VALIDITY_FIELDS if name not in field_validity]
        if missing:
            reasons.append(
                f"逐字段有效性缺 {missing}（只有 {sorted(field_validity) or '空'}）："
                "有键才有那一项的独立掩码，缺键的字段会悄悄退回联合掩码"
            )
        reference = getattr(case, "valid", None)
        for name, mask in sorted(field_validity.items()):
            if getattr(mask, "dtype", None) != np.bool_:
                reasons.append(
                    f"逐字段掩码 {name!r} 的 dtype 是 {getattr(mask, 'dtype', None)}，"
                    "不是 bool：非布尔量读成有效性会把 0 之外的一切都当成有效"
                )
            elif reference is not None and np.shape(mask) != np.shape(reference):
                reasons.append(
                    f"逐字段掩码 {name!r} 的形状 {np.shape(mask)} 与联合掩码 "
                    f"{np.shape(reference)} 不一致"
                )

    # -- 前沿监督资格（§10）-----------------------------------------------
    eligibility = getattr(case, "front_supervision_eligibility", None)
    if eligibility is None or not eligibility:
        reasons.append(
            "算例没有前沿监督资格的分项记录（检测状态与监督资格没有分开），"
            "confirmed/doubtful 无法区分"
        )
    else:
        # **逐侧判断。** 两条前沿各有自己的证据来源，一侧缺失或退化不得改变另一侧
        # 的结论——聚合起来看的话，"接触侧明确写了 False、激波侧没有记录"会被读成
        # "整体还有记录可用"。
        per_side: dict[str, Any] = {}
        for side in ("contact", "shock"):
            side_source = str(
                eligibility.get(f"{side}_source")
                or (eligibility.get("source") if isinstance(eligibility.get("source"), str) else "")
                or ""
            )
            side_level = str(eligibility.get(f"{side}_level") or eligibility.get("level") or "")
            per_side[side] = side_source or "未记录"
            if side_level == "none" or UNVERIFIED_ELIGIBILITY in side_source or "UNVERIFIED" in side_source:
                reasons.append(
                    f"{side} 前沿的监督资格是**未核实**档（既没有资格数组也没有"
                    "三态码，只有『半径有限即可』）：那是最弱的一档判据，不能拿来"
                    "当训练标签"
                )
            elif side_level == "legacy_state_code":
                # **旧格式可以诊断，不可以训练。** 三态码只说"分辨得出吗"，不说
                # "够不够格当标签"；把这一档判成 ready，等于把"数据还没迁移"读成
                # "数据已经就绪"。
                reasons.append(
                    f"{side} 前沿只有**旧的三态码**，没有当前契约要求的逐帧资格"
                    "数组（§10）：这一档可以说'检测到了什么'，说不了'够不够格当"
                    "标签'。可以读来诊断，不能当正式训练数据"
                )
            elif not side_source:
                reasons.append(f"{side} 前沿既没有资格数组，也没有说明来源")
        detail["front_eligibility_source"] = per_side
        detail["front_eligibility_level"] = eligibility.get("level")
        detail["front_supervision_eligibility"] = (
            (eligibility.get("metadata") or {}).get("contact", {}).get("status_counts")
            if isinstance(eligibility.get("metadata"), Mapping) else None
        )
        detail["front_eligibility_counts"] = {
            side: int(np.count_nonzero(eligibility[f"{side}_eligible"]))
            for side in ("contact", "shock")
            if eligibility.get(f"{side}_eligible") is not None
        }

    detail["reasons_count"] = len(reasons)
    return ContractVerdict(ready=not reasons, reasons=reasons, detail=detail)


def evaluate_data_contract(cases: Sequence[Any], **_: Any) -> ContractVerdict:
    """整批算例的数据契约判定。

    ``ready`` 只有当**每一个**算例都满足契约时才为真——一个算例没有权威分区，
    就不能用它做正式训练。
    """
    verdicts = [evaluate_case_contract(case) for case in cases]
    reasons: list[str] = []
    for verdict in verdicts:
        for reason in verdict.reasons:
            reasons.append(f"[{verdict.detail.get('case_id')}] {reason}")
    return ContractVerdict(
        ready=all(v.ready for v in verdicts) and bool(verdicts),
        reasons=reasons,
        detail={
            "cases": [v.as_dict() for v in verdicts],
            "n_cases": len(verdicts),
            "n_ready": sum(1 for v in verdicts if v.ready),
        },
    )


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_SYMMETRY_FACTOR",
    "ContractVerdict",
    "evaluate_case_contract",
    "evaluate_data_contract",
]
