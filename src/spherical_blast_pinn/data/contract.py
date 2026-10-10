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


def _symmetry_factor(source: Mapping[str, Any]) -> float | None:
    """球对称份额。**没有明确声明就不算有**（§8：覆盖率分母用实际对称份额）。"""
    for key in ("symmetry_factor", "symmetry_share_inverse", "full_sphere_factor"):
        value = source.get(key)
        if value:
            return float(value)
    share = source.get("symmetry_share")
    if share:
        return 1.0 / float(share)
    return None


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

    symmetry = _symmetry_factor(source)
    detail["symmetry_factor"] = symmetry
    if symmetry is None:
        reasons.append(
            "没有声明球对称份额（symmetry_factor / symmetry_share）："
            "体积等效半径与覆盖率的分母都依赖它，缺了就无从核对"
        )

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
    authoritative = "volume fraction" in material_source.lower()
    if not authoritative:
        reasons.append(
            "材料分区不是从 d3plot 的材料体积分数读出来的"
            f"（记录为 {material_source or '未记录'}）：启发式探测器给出的接触面"
            "位置有量化跳变、大量帧落在装药半径以内（物理上不可能）。"
            "契约上判为未就绪——权威分区必须重新提取"
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

    # -- 前沿监督资格（§10）-----------------------------------------------
    eligibility = getattr(case, "front_supervision_eligibility", None)
    if eligibility is None:
        reasons.append(
            "算例没有前沿监督资格的分项记录（检测状态与监督资格没有分开），"
            "confirmed/doubtful 无法区分"
        )
    else:
        detail["front_supervision_eligibility"] = {
            side: (record or {}).get("status_counts")
            for side, record in eligibility.items()
            if isinstance(record, Mapping)
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
