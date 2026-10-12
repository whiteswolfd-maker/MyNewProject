"""DYNA 真实物理初始化审查（提示词 §3）。

这一步**必须先做**，而且不能只看 d3plot 第一帧的压力云图就下结论。本命令做
三件事：

1. 从 ``.k`` 卡片读出材料、状态方程、起爆设置、控制卡与输出卡；
2. 把卡片里的值与提示词 §4 的表格逐项对照；
3. 用**数据本身**检验时间零点代表什么——起爆是整球瞬时完成，还是有限速度
   传播。做法是在 ``(r, t)`` 剖面上找反应前沿，和 ``r = D t`` 比较。

第 3 条是判据的核心。读取器的名字、文件的时间戳、第一帧的压力云图都不能回答
"时间零点代表什么"，只有前沿位置随时间的移动能回答。

输出一张审查表，每一项都带证据来源，并明确标注哪些结论**尚无证据**。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..constants import UNIT_SYSTEMS, UnitSystem
from ..data import compatibility as compat
from ..data.extracted_case import load_extracted_case
from ..data.kfile_parser import (
    _floats,
    _is_comment,
    iter_keyword_blocks,
    parse_deck,
    scan_geometry,
)
from ..physics import material as mat
from ._common import add_common_arguments, configure_stdout, load_config_from_args, write_json

CONSISTENT = "一致"
INCONSISTENT = "不一致"
UNKNOWN = "缺证据"


@dataclass
class Finding:
    """审查表的一行。"""

    item: str
    conclusion: str
    evidence: str
    scheme_consistency: str
    usable_range: str = ""
    still_missing: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "审查项": self.item,
            "结论": self.conclusion,
            "证据文件及位置": self.evidence,
            "与方案 B 是否一致": self.scheme_consistency,
            "可用时空范围": self.usable_range,
            "仍缺什么信息": self.still_missing,
        }


@dataclass
class DeckAudit:
    """一个算例卡片的审查结果。"""

    case_id: str
    deck_path: str
    findings: list[Finding] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "deck": self.deck_path,
            "findings": [f.as_dict() for f in self.findings],
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# 卡片审查
# --------------------------------------------------------------------------


def _relative(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-30)


def infer_unit_system(explosive_density: float | None) -> UnitSystem | None:
    """由炸药密度判定单位制。

    判据是密度，不是文件名，也不是标题：卡片里 ro 的数值只有配上单位制才有
    意义，而规范给了 1630 kg/m³ 这个确定的对照值。

    mm-tonne-s（``mm_s_t``）与 SI 是这里出现的两种。注意吨每立方毫米到
    kg/m³ 的因子是 ``1e12``，不是 ``1e9``——把 ``1e-9`` 的密度当成"差不多
    就是 1630"会让后面每一个材料参数都差三个数量级。
    """
    if explosive_density is None or explosive_density <= 0.0:
        return None
    for name in ("si", "mm_s_t", "cm_us_g", "mm_ms_kg"):
        system = UNIT_SYSTEMS.get(name)
        if system is None:
            continue
        converted = float(explosive_density) * system.density_to_si
        if _relative(converted, mat.PRODUCT_REFERENCE_DENSITY) < 1.0e-3:
            return system
    return None


def _to_si(value: float | None, factor: float) -> float | None:
    if value is None:
        return None
    return float(value) * float(factor)


def audit_boundary_conditions(deck_path: Path) -> Finding:
    """外边界与对称面：读 ``*BOUNDARY_SPC_SET`` 与 ``*BOUNDARY_NON_REFLECTING``。

    这一项直接决定两件事：外边界是否会反射（污染晚期的场），以及质量与能量
    统计要不要乘一个几何倍数（对称面把整球剪掉了几分之几）。
    """
    wanted = ["*BOUNDARY_SPC_SET", "*BOUNDARY_NON_REFLECTING", "*SET_NODE_LIST"]
    spc: list[list[float]] = []
    nonreflecting: list[list[float]] = []
    try:
        for keyword, lines in iter_keyword_blocks(deck_path, wanted):
            for line in lines:
                if _is_comment(line):
                    continue
                if keyword == "*BOUNDARY_SPC_SET":
                    values = _floats(line, minimum=3)
                    if values is not None:
                        spc.append(values[:8])
                elif keyword == "*BOUNDARY_NON_REFLECTING":
                    values = _floats(line, minimum=1)
                    if values is not None:
                        nonreflecting.append(values[:3])
    except Exception as exc:  # pragma: no cover - defensive
        return Finding(
            item="8. 外边界条件与对称面",
            conclusion=f"读取边界卡失败：{exc}",
            evidence=str(deck_path),
            scheme_consistency=UNKNOWN,
        )

    parts = []
    if spc:
        planes = []
        for values in spc:
            dof = values[2:5] if len(values) >= 5 else []
            planes.append("/".join(str(int(d)) for d in dof))
        parts.append(f"{len(spc)} 组 *BOUNDARY_SPC_SET，约束自由度（x/y/z）= {planes}")
    else:
        parts.append("没有 *BOUNDARY_SPC_SET")
    if nonreflecting:
        parts.append(f"{len(nonreflecting)} 组 *BOUNDARY_NON_REFLECTING")
    else:
        parts.append("没有 *BOUNDARY_NON_REFLECTING，外边界可能是反射边界")

    conclusion = "；".join(parts)
    if spc and nonreflecting:
        if len(spc) == 3:
            conclusion += (
                "。三组各约束一个方向的平动，与三个对称面一致；配合非反射外边界，"
                "计算域是整球的八分之一，质量与能量统计需要乘以 8"
            )
        else:
            dof_counts = {sum(int(d) for d in values[2:5]) for values in spc if len(values) >= 5}
            if dof_counts == {3}:
                conclusion += (
                    "。这些组同时约束三个方向，是固定壁面而不是对称面；"
                    "计算域不是靠对称面剪出来的，几何倍数取决于网格实际张角，需另行确认"
                )
            else:
                conclusion += "。约束模式不是三个各约束一个方向，也不是全约束；对称面需要逐组人工核对"
    return Finding(
        item="8. 外边界条件与对称面",
        conclusion=conclusion,
        evidence=f"{deck_path}: *BOUNDARY_SPC_SET / *BOUNDARY_NON_REFLECTING / *SET_NODE_LIST",
        scheme_consistency=UNKNOWN,
        still_missing=(
            "非反射边界在强爆炸下的实际吸收效果需要运行时数据核对；"
            "仅凭卡片不能断定晚期有无反射污染"
        ),
    )


def audit_deck(case_id: str, deck_path: Path) -> DeckAudit:
    audit = DeckAudit(case_id=case_id, deck_path=str(deck_path))
    if not Path(deck_path).is_file():
        audit.findings.append(
            Finding(
                item="卡片可读性",
                conclusion="卡片不存在",
                evidence=str(deck_path),
                scheme_consistency=UNKNOWN,
                still_missing="提供该算例的 input.k",
            )
        )
        return audit

    summary = parse_deck(deck_path, geometry=True)
    audit.detail["title"] = summary.title
    audit.detail["termination_time"] = summary.termination_time
    audit.detail["d3plot_interval"] = summary.d3plot_interval
    audit.detail["initial_detonation"] = summary.initial_detonation
    audit.detail["initial_detonation_time"] = summary.initial_detonation_time
    audit.detail["control_ale"] = summary.control_ale
    audit.detail["n_parts"] = len(summary.parts)
    audit.detail["part_names"] = summary.material_names()

    # -- 1. 起爆方式：方案 B 要的是中心起爆、有限爆速 ---------------------
    explosive = summary.explosive_materials()
    has_detonation_card = summary.initial_detonation is not None
    if explosive and has_detonation_card:
        audit.findings.append(
            Finding(
                item="1. 整球瞬时初始化还是有限速度爆轰",
                conclusion=(
                    f"有限速度爆轰（programmed burn）。卡片有 {len(explosive)} 张 "
                    f"*MAT_HIGH_EXPLOSIVE_BURN 与一张 *INITIAL_DETONATION，"
                    f"起爆点 {summary.initial_detonation}，lt = {summary.initial_detonation_time}"
                ),
                evidence=f"{deck_path}: *MAT_HIGH_EXPLOSIVE_BURN / *INITIAL_DETONATION",
                # 方案 B 要求的就是中心起爆的有限爆速传播，所以卡片里有起爆卡与
                # 爆速是**一致**。方案 A 把"整球瞬时转化"当基准，这一项的判据在
                # 那次迁移里整体反了过来——留着旧判据会把合格的 B 算例全部标成
                # 不一致，而"不一致"会阻止训练。
                scheme_consistency=CONSISTENT,
                usable_range=(
                    "t_phys >= T_b = R_0/D 的帧属于两介质演化；T_b 之前是爆轰阶段，"
                    "剖面上只有装药与前沿，没有接触面与空气激波"
                ),
                still_missing="起爆点是否恰好是球心，需要几何坐标核对（见第 7 项）",
            )
        )
        d = summary.require_explosive().detonation_velocity
        if d:
            # 单位制还没判定，所以这里只记**原始卡片值**；换算到 m/s 在函数末尾。
            audit.detail["detonation_velocity_raw_card_units"] = float(d)
    elif explosive:
        audit.findings.append(
            Finding(
                item="1. 整球瞬时初始化还是有限速度爆轰",
                conclusion="有炸药材料卡但没有 *INITIAL_DETONATION；起爆方式无法从卡片判定",
                evidence=f"{deck_path}: *MAT_HIGH_EXPLOSIVE_BURN（无起爆卡）",
                scheme_consistency=UNKNOWN,
                still_missing="起爆卡或关键字说明",
            )
        )
    else:
        audit.findings.append(
            Finding(
                item="1. 整球瞬时初始化还是有限速度爆轰",
                conclusion="卡片里没有 *MAT_HIGH_EXPLOSIVE_BURN",
                evidence=f"{deck_path}",
                scheme_consistency=UNKNOWN,
                still_missing="炸药材料卡",
            )
        )

    # -- 5. 单位制：先做，因为后面每一项比较都要用它换算 -----------------
    explosive_card = summary.require_explosive() if explosive else None
    raw_density = explosive_card.density if explosive_card else None
    system = infer_unit_system(raw_density)
    audit.detail["explosive_density_raw"] = raw_density
    if system is None:
        audit.findings.append(
            Finding(
                item="5. 单位制",
                conclusion=(
                    f"未识别；炸药密度卡片值 {raw_density!r} 在已知单位制下换算后"
                    f"都与规范的 {mat.PRODUCT_REFERENCE_DENSITY:g} kg/m^3 对不上"
                ),
                evidence=f"{deck_path}: *MAT_HIGH_EXPLOSIVE_BURN 的 ro",
                scheme_consistency=UNKNOWN,
                still_missing="该卡片使用的单位制说明",
            )
        )
    else:
        density_si = _to_si(raw_density, system.density_to_si)
        audit.detail["unit_system_inferred"] = system.name
        audit.detail["explosive_density_kg_m3"] = density_si
        audit.detail["density_to_si_factor"] = system.density_to_si
        audit.findings.append(
            Finding(
                item="5. 单位制",
                conclusion=(
                    f"{system.name}；判据是炸药密度而不是文件名：卡片 ro = {raw_density:g}，"
                    f"按该单位制换算（×{system.density_to_si:g}）得 {density_si:.6g} kg/m^3，"
                    f"与规范的 {mat.PRODUCT_REFERENCE_DENSITY:g} kg/m^3 一致"
                ),
                evidence=f"{deck_path}: *MAT_HIGH_EXPLOSIVE_BURN 的 ro = {raw_density:g}",
                scheme_consistency=CONSISTENT,
            )
        )

    # -- 3. 材料参数与规范的一致性（换算到 SI 之后比较） -----------------
    jwl_cards = {k: v for k, v in summary.eos.items() if v.kind == "jwl"}
    if jwl_cards:
        card = jwl_cards[sorted(jwl_cards)[0]]
        pressure_factor = system.pressure_to_si if system is not None else 1.0
        volume_energy_factor = (
            system.energy_per_volume_to_si if system is not None else 1.0
        )
        mismatches = []
        checks = {
            "A": (mat.JWL_A, card.parameters.get("A")),
            "B": (mat.JWL_B, card.parameters.get("B")),
            "R1": (mat.JWL_R1, card.parameters.get("R1")),
            "R2": (mat.JWL_R2, card.parameters.get("R2")),
            "omega": (mat.JWL_OMEGA, card.parameters.get("omeg")),
        }
        audit.detail["jwl_card_raw"] = dict(card.parameters)
        for name, (expected, got) in checks.items():
            if got is None:
                mismatches.append(f"{name} 未读到")
                continue
            factor = 1.0 if name in ("R1", "R2", "omega") else pressure_factor
            converted = float(got) * factor
            if _relative(converted, expected) > 1.0e-5:
                mismatches.append(
                    f"{name}: 卡片 {got:g} → {converted:.6g} Pa（SI）vs 规范 {expected:.6g} Pa"
                )
        audit.findings.append(
            Finding(
                item="3a. JWL 系数与规范 §4 是否一致",
                conclusion=(
                    "一致（已按识别出的单位制换算到 SI）"
                    if not mismatches
                    else "不一致：" + "；".join(mismatches)
                ),
                evidence=f"{deck_path}: *EOS_JWL 第 {sorted(jwl_cards)[0]} 号",
                scheme_consistency=CONSISTENT if not mismatches else INCONSISTENT,
            )
        )

        # E0 / V0 / 能量参考零点
        e0 = card.parameters.get("E0")
        v0 = card.parameters.get("V0")
        notes = []
        if e0 is not None:
            notes.append(f"E0 = {e0:g}（按参考体积的能量密度）")
        if v0 is not None:
            notes.append(f"V0 = {v0:g}")
        e0_si = _to_si(e0, volume_energy_factor)
        audit.detail["e0_volume_energy_raw"] = e0
        audit.detail["e0_volume_energy_si"] = e0_si
        audit.detail["v0"] = v0
        audit.detail["specific_energy_j_kg"] = (
            mat.JWLParameters.specific_energy_from_volume_energy(
                e0_si, mat.PRODUCT_REFERENCE_DENSITY
            )
            if e0_si
            else None
        )
        audit.findings.append(
            Finding(
                item="3b/c/d. E0、V0 与能量参考零点",
                conclusion=(
                    ("；".join(notes) + f"；换算到 SI 后 E0 = {e0_si:.6g} J/m^3")
                    if e0_si
                    else ("；".join(notes) or "未读到 E0/V0")
                ),
                evidence=f"{deck_path}: *EOS_JWL 第 2 行",
                scheme_consistency=(
                    CONSISTENT
                    if e0_si is not None
                    and _relative(e0_si, mat.PRODUCT_VOLUME_ENERGY) < 1.0e-5
                    else UNKNOWN
                ),
                still_missing=(
                    "能量参考零点需要 *DATABASE_EXTENT_BINARY 的 engflg 与运行时能量"
                    "统计共同确认；仅凭卡片不能断定"
                ),
            )
        )

    # -- 6. 终止时间与输出间隔 -------------------------------------------
    endtim = summary.termination_time
    dt_plot = summary.d3plot_interval
    detail = []
    if endtim is not None:
        detail.append(f"endtim = {endtim:g}（时间单位）")
    if dt_plot is not None:
        detail.append(f"d3plot 间隔 = {dt_plot:g}")
        if endtim:
            detail.append(f"→ 约 {endtim / dt_plot:.0f} 个输出帧")
    audit.findings.append(
        Finding(
            item="6. 终止时间与输出间隔",
            conclusion="；".join(detail) or "未读到 *CONTROL_TERMINATION / *DATABASE_BINARY_D3PLOT",
            evidence=f"{deck_path}",
            scheme_consistency=CONSISTENT if detail else UNKNOWN,
            still_missing="外边界条件需要 *BOUNDARY_* 卡片（见第 8 项）",
        )
    )

    # -- 7. 几何：是否 1/8 球、对称面 -------------------------------------
    try:
        geometry = scan_geometry(deck_path)
    except Exception as exc:  # pragma: no cover - defensive
        audit.findings.append(
            Finding(
                item="7. 计算域形状与对称面",
                conclusion=f"几何扫描失败：{exc}",
                evidence=str(deck_path),
                scheme_consistency=UNKNOWN,
                still_missing="可解析的 *NODE / *ELEMENT_SOLID",
            )
        )
    else:
        audit.detail["geometry"] = geometry.as_dict()
        audit.detail["n_nodes"] = geometry.n_nodes
        audit.detail["n_elements"] = geometry.n_solid_elements
        shape = _describe_shape(geometry.bounding_box, geometry.radius_max)
        audit.findings.append(
            Finding(
                item="7. 计算域形状与对称面",
                conclusion=shape,
                evidence=f"{deck_path}: *NODE 包围盒 {geometry.bounding_box}",
                scheme_consistency=UNKNOWN,
                still_missing=(
                    "对称面靠 *BOUNDARY_SPC_SET 定义，其自由度组合需要逐条核对；"
                    "质量与能量统计是否需要几何倍数取决于此"
                ),
            )
        )

    # -- 8. 外边界条件与对称面 -------------------------------------------
    audit.findings.append(audit_boundary_conditions(Path(deck_path)))

    # 爆速必须在单位制确定**之后**换算到 m/s，而且第 2 项的燃尽时刻直接由它
    # 得到。留一个 mm/s 的爆速在那儿，``R_0/D`` 会小三个数量级，于是"燃尽前"
    # 一帧都不剩——判据会安静地给出"无法判定"，而不是报错。
    raw_velocity = explosive_card.detonation_velocity if explosive_card else None
    if raw_velocity:
        audit.detail["detonation_velocity_raw"] = float(raw_velocity)
        audit.detail["detonation_velocity_m_s"] = float(raw_velocity) * (
            system.velocity_to_si if system is not None else 1.0
        )

    audit.detail["warnings"] = list(summary.warnings)
    return audit


def _describe_shape(bounding_box, radius_max: float) -> str:
    """由节点包围盒判断计算域形状。

    只看包围盒可以区分"整球/八分之一球"和"楔形体"，但**不足以**判断对称面：
    楔形体的包围盒同样可能是正方体。对称面必须逐条核对 ``*BOUNDARY_SPC_SET``
    的自由度组合，本函数不代替那一步。
    """
    if not bounding_box:
        return "未能读到节点坐标，无法判断"
    spans = [float(high) - float(low) for low, high in bounding_box]
    if any(span <= 0.0 for span in spans):
        return f"退化包围盒跨度 {spans}，无法判断"
    ratio = max(spans) / min(spans)
    extent = f"包围盒跨度 {[round(s, 6) for s in spans]}，最大半径 {radius_max:.6g}"
    if ratio > 1.5:
        return f"非等向计算域（{extent}），既不是整球也不是规则八分之一球"
    return (
        f"三个方向跨度相同（{extent}），与整球或八分之一球相符；"
        "但楔形体的包围盒也是正方体，所以对称面仍需核对 *BOUNDARY_SPC_SET"
    )


# --------------------------------------------------------------------------
# 时间零点判据：前沿是否跟随 r = D t
# --------------------------------------------------------------------------


def _charge_radius_from_metadata(directory: Path) -> float | None:
    meta = Path(directory) / "metadata.json"
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return None
    value = payload.get("R_0")
    return float(value) if value else None


def audit_time_origin(
    case_id: str,
    directory: Path,
    *,
    detonation_velocity: float,
    charge_radius: float | None = None,
) -> Finding:
    """用数据判断时间零点代表什么。

    做法：逐帧找出压力最高的位置（反应前沿），与 ``D t`` 比较。若前沿在早期
    帧上以 ``D`` 匀速外移，时间零点就是起爆开始，反应在输出帧**之间**持续，
    与方案 B 的"零点即起爆、反应在输出帧之间以有限爆速推进"一致；方案 A 的
    "零点即整球全部产物"才是与数据不符的那一个。

    只看第一帧压力云图无法得到这个结论：第一帧的压力分布对两种初始化都可能
    "看起来像产物"。
    """
    npz = directory / "raw_line.npz"
    if not npz.exists():
        return Finding(
            item="2. 时间零点代表什么",
            conclusion="没有可读的时空剖面，无法判定",
            evidence=str(npz),
            scheme_consistency=UNKNOWN,
            still_missing="raw_line.npz 或等价的 (t, r, p) 剖面",
        )

    # 用 with 释放句柄：NpzFile 持有文件句柄，不关会让同路径后续的 os.replace
    # 在 Windows 上 PermissionError。
    with np.load(npz, allow_pickle=True) as handle:
        data = {key: handle[key] for key in handle.files}
    times = np.asarray(data["t"], dtype=np.float64)
    radius = np.asarray(data["x"], dtype=np.float64)
    pressure = np.asarray(data["P"], dtype=np.float64)

    # 反应前沿的判据：压力明显高于环境态的最外侧半径。
    #
    # 用**固定**的绝对阈值，而不是"本帧峰压的某个比例"。峰压比例在燃尽前后指的
    # 是两件不同的事：燃尽前峰压就是爆轰前沿，燃尽后峰压落在产物中心，半峰值的
    # 等值线跑到产物内部去了。用比例阈值会把两种截然不同的前沿混进同一次拟合，
    # 斜率于是变成 ~3 km/s 的一个无意义的中间值。
    FRONT_PRESSURE_PA = 1.0e9
    if charge_radius is None:
        charge_radius = _charge_radius_from_metadata(directory)
    burn_time = (
        float(charge_radius) / float(detonation_velocity)
        if charge_radius and detonation_velocity > 0.0
        else None
    )

    rows: list[tuple[float, float, float]] = []
    for frame in range(times.size):
        t = float(times[frame])
        if burn_time is not None and t > burn_time:
            # 燃尽之后，超过阈值的区域是正在膨胀的产物球，不再是爆轰前沿。
            # 把它和燃尽前的帧放进同一次拟合，斜率就变成一个没有物理含义的中间值。
            continue
        peak = float(pressure[frame].max())
        if peak < FRONT_PRESSURE_PA:
            continue
        above = np.flatnonzero(pressure[frame] > FRONT_PRESSURE_PA)
        if above.size == 0:
            continue
        rows.append((t, float(radius[int(above[-1])]), peak))

    if len(rows) < 4:
        spacing = float(np.median(np.diff(times))) if times.size > 1 else float("nan")
        resolution = (
            f"输出间隔中位数 {spacing:.3e} s，而燃尽时刻 R_0/D = {burn_time:.3e} s；"
            f"燃尽前只剩 {len(rows)} 帧"
            if burn_time
            else f"输出间隔中位数 {spacing:.3e} s"
        )
        return Finding(
            item="2. 时间零点代表什么",
            conclusion=(
                f"剖面的时间分辨率不足以判定：{resolution}。"
                "**这不是'时间零点没有移动前沿'，而是'这份导出看不见它'**"
            ),
            evidence=(
                f"{npz}: 帧数 {times.size}，时间跨度 {times[0]:.3e}–{times[-1]:.3e} s，"
                f"满足 P > {FRONT_PRESSURE_PA:.0e} Pa 的燃尽前帧数 {len(rows)}"
            ),
            scheme_consistency=UNKNOWN,
            still_missing=(
                "燃尽之前（t < R_0/D）的稠密输出帧；当前导出在这些时刻上没有采样点"
            ),
        )

    fit_t = np.array([t for t, _, _ in rows])
    fit_r = np.array([r for _, r, _ in rows])
    slope = float(np.polyfit(fit_t, fit_r, 1)[0])
    intercept = float(np.polyfit(fit_t, fit_r, 1)[1])
    deviation = _relative(slope, detonation_velocity)

    first_t, first_r, first_peak = rows[0]
    last_t, last_r, _ = rows[-1]
    conclusion = (
        f"时间零点不是整球同时转化。逐帧的强压缩前沿是移动的："
        f"t = {first_t:.3e} s 时在 r = {first_r:.4g} m（P = {first_peak:.4g} Pa），"
        f"t = {last_t:.3e} s 时在 r = {last_r:.4g} m。"
        f"线性拟合斜率 {slope:.4g} m/s，与卡片爆速 {detonation_velocity:.4g} m/s "
        f"的相对偏差 {deviation:.3%}"
    )
    return Finding(
        item="2. 时间零点代表什么",
        conclusion=conclusion,
        evidence=(
            f"{npz}: 逐帧取 P > {FRONT_PRESSURE_PA:.0e} Pa 的最外侧半径，对 t 作线性"
            f"拟合（{len(rows)} 帧，截距 {intercept:.3e} m）"
        ),
        # 前沿以卡片爆速匀速外移，说明零点就是起爆开始——这正是方案 B 的定义。
        # 拟合不上的时候才是"缺证据"，而不是"不一致"。
        scheme_consistency=CONSISTENT if deviation < 0.15 else UNKNOWN,
        usable_range=(
            "t_phys >= t_b = R_0/D 之后才存在空气层；此前剖面上只有未反应的装药"
            "和爆轰前沿，没有接触面"
        ),
        still_missing=(
            "卡片没有输出反应进度字段（*DATABASE_EXTENT_BINARY 未开启 ialemat "
            "一类的材料状态输出），反应速率本身无法逐帧确认，只能由前沿位置间接判断"
        ),
    )


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--deck", action="append", default=[],
        help="case_id=path/to/input.k，可重复",
    )
    parser.add_argument(
        "--data", action="append", default=[],
        help="case_id=path/to/extracted_dir，可重复；用于时间零点判据",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    configure_stdout()
    config = load_config_from_args(args)

    decks = [entry.split("=", 1) for entry in args.deck]
    datas = {entry.split("=", 1)[0]: Path(entry.split("=", 1)[1]) for entry in args.data}
    if not decks:
        print("no --deck supplied; nothing to audit", file=sys.stderr)
        return 2

    audits = []
    for case_id, deck in decks:
        audit = audit_deck(case_id, Path(deck))
        data_dir = datas.get(case_id)
        if data_dir is not None:
            velocity = float(audit.detail.get("detonation_velocity_m_s", 6930.0))
            audit.findings.append(audit_time_origin(case_id, data_dir, detonation_velocity=velocity))
        else:
            audit.findings.append(
                Finding(
                    item="2. 时间零点代表什么",
                    conclusion="未提供数据目录，无法用数据判定",
                    evidence="--data 未给出",
                    scheme_consistency=UNKNOWN,
                    still_missing=f"用 --data {case_id}=<目录> 提供时空剖面",
                )
            )
        audits.append(audit)

    out_dir = Path(args.output_dir) if args.output_dir else config.path_of(
        "training.output_dir", default=Path("outputs/run")
    ) / "dyna_audit"
    payload = {"cases": [a.as_dict() for a in audits]}
    write_json(out_dir / "dyna_audit.json", payload)
    report = render_report(audits)
    (out_dir / "dyna_audit.md").write_text(report, encoding="utf-8")
    print(report)

    # -- 机器可读的兼容性状态 ---------------------------------------------
    # 审查表是给人看的；训练入口读的是这一份。只写 md 的话，"审查发现不兼容"
    # 与"训练照常开始"可以同时成立。
    statuses: dict[str, compat.CaseCompatibility] = {}
    for audit in audits:
        partition, partition_evidence = _partition_verdict(datas.get(audit.case_id))
        statuses[audit.case_id] = compat.compatibility_from_findings(
            audit.case_id,
            audit.findings,
            source=str(out_dir / "dyna_audit.json"),
            material_partition=partition,
            material_partition_evidence=partition_evidence,
        )
    report_path = compat.write_report(
        out_dir / compat.REPORT_FILENAME,
        compat.build_report(
            statuses,
            generated_by="spherical_blast_pinn.cli.audit_dyna",
            notes=[
                "training reads this file; a case whose status is not 'compatible' "
                "may not supervise a scheme-A model without an explicit override",
            ],
        ),
    )
    print("\n## 数据兼容性状态（训练入口读取这一份）")
    print("")
    for entry in statuses.values():
        print(f"- {entry.summary_line()}")
    print(f"\nwritten: {out_dir / 'dyna_audit.json'}")
    print(f"written: {report_path}")
    return 0


def _partition_verdict(data_dir: Path | None) -> tuple[str | None, dict[str, Any]]:
    """从已提取剖面判定材料分区的可靠性；没有数据目录时返回"无证据"。"""
    if data_dir is None or not (Path(data_dir) / "raw_line.npz").exists():
        return None, {"reason": "no extracted profile supplied via --data"}
    try:
        case = load_extracted_case(Path(data_dir))
    except Exception as exc:  # pragma: no cover - defensive
        return compat.UNKNOWN, {"reason": f"could not load the extracted case: {exc}"}
    return compat.material_partition_verdict(case)


def render_report(audits: Sequence[DeckAudit]) -> str:
    lines = [
        "# DYNA 真实物理初始化审查",
        "",
        "本表由 `python -m spherical_blast_pinn.cli.audit_dyna` 生成。每一项都带证据来源；",
        "没有证据的项标为“缺证据”，不编造结论。",
        "",
    ]
    for audit in audits:
        lines.append(f"## {audit.case_id}")
        lines.append("")
        lines.append(f"卡片：`{audit.deck_path}`")
        lines.append("")
        lines.append("| 审查项 | 结论 | 证据文件及位置 | 与方案 B 是否一致 | 仍缺什么信息 |")
        lines.append("|---|---|---|---|---|")
        for finding in audit.findings:
            lines.append(
                "| {item} | {conclusion} | {evidence} | {scheme_consistency} | {missing} |".format(
                    item=_cell(finding.item),
                    conclusion=_cell(finding.conclusion),
                    evidence=_cell(finding.evidence),
                    scheme_consistency=_cell(finding.scheme_consistency),
                    missing=_cell(finding.still_missing),
                )
            )
        lines.append("")
        if audit.detail:
            lines.append("<details><summary>卡片明细</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(audit.detail, indent=2, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines)


def _cell(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
