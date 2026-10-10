"""求解参考初始场，并输出提示词 §5 要求的检查报告。

§5 逐条要求：双材料球对称流动、网格加密与时间步减小检查、正值/质量/能量/
外边界未受扰动检查、用一个不同半径的独立小算例核查尺度映射、EOS 一致性、
标记近接触面与激波展宽区，以及导出材料参数、方法、网格、物理时间、Rc/Rs、
有效区域与守恒收敛报告。

本命令把这些检查全部实际执行，而不是只写进文档：每一条都跑出数字，任何一条
不通过都要在报告里看得见。

时间口径（方案 B）
------------------

参考场对外登记的是**物理时间** ``T``；采样时刻按**相对爆轰渡越时间**
``T_b = R_0/D`` 给出。所以"PINN 初值那一刻"（``tau = 0``，即
``T = T_h = T_b + epsilon``）对应的采样时间是 **``epsilon``**，不是方案 A 的
那个 1e-6 秒绝对时间。两者都是 1e-6 量级，混用不会报错，只会把整条曲线挪走。

本命令是**诊断入口**，不是正式初值入口：它显式传 ``allow_unaccepted=True``，
并把未验收的标记一起写进产物。正式入口与训练入口走
``cli._common.assert_reference_accepted``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..physics import material as mat
from ..physics.detonation import (
    DetonationConfig,
    detonation_front_state,
    ideal_cj_state,
)
from ..physics.precursor import PrecursorConfig, solve_reference_field
from ..physics.similitude import detonation_transit_time, handover_time, start_delay
from ._common import (
    REFERENCE_CONSTRUCTION,
    _field_content_hash,
    record_acceptance,
    add_common_arguments,
    build_reference,
    configure_stdout,
    handover_relative_time,
    load_config_from_args,
    reference_config,
    reference_detonation_config,
    reference_detonation_velocity,
    write_json,
)


def _precursor_settings(config, cells: int) -> PrecursorConfig:
    base = reference_config(config)
    return PrecursorConfig(
        cells_inside=int(cells),
        growth=base.growth,
        outer_radius_factor=base.outer_radius_factor,
        cfl=base.cfl,
        q1=base.q1,
        q2=base.q2,
    )


def _solve_at_handover(
    config,
    radius: float,
    cells: int,
    *,
    detonation_cells: int | None = None,
):
    """在**该半径自己的**交接时刻求解一次，返回 ``(field, 相对时间)``。

    这里**不再接受** ``time_scale``：``handover_relative_time`` 已经随半径线性缩放
    （``epsilon = k epsilon_ref``），再乘一次 ``k`` 就把缩放做了两遍。两遍之后
    两个算例采的是**不同的无量纲时刻**，而两个数都是秒、都为正，比较会给出
    量级为 1 的"尺度偏差"——那是在量和自己比，不是在检验相似性。

    （这正是 `ReferenceField` 的文档里那句"传两次会把长度和时间各缩放一遍，
    而结果看起来仍然'像那么回事'"的同一个坑。）
    """
    relative = handover_relative_time(config, radius)
    detonation = reference_detonation_config(config)
    if detonation_cells is not None:
        detonation = DetonationConfig(cells=int(detonation_cells))
    field = solve_reference_field(
        charge_radius=radius,
        config=_precursor_settings(config, cells),
        detonation_velocity=reference_detonation_velocity(config),
        detonation_config=detonation,
        sample_times=[relative],
    )
    return field, relative


def _convergence_check(config, radius: float, levels: Sequence[int]) -> dict[str, Any]:
    """在同一物理时刻加密网格，比较两条前沿与峰值。"""
    rows = []
    for cells in levels:
        # §5.1 要的是**整个构造**的加密，不只是前置求解器的网格：产物场本身也有
        # 一套网格，它被插值到前置网格上。只加密后者，测到的是插值误差，不是
        # 产物场的分辨率。
        field, _ = _solve_at_handover(
            config, radius, cells, detonation_cells=cells
        )
        contact = float(field.contact_radius[-1]) / radius
        shock = float(field.shock_radius[-1]) / radius
        rows.append(
            {
                "cells_inside": int(cells),
                "contact_over_R0": contact,
                "shock_over_R0": shock,
                # §5.2：空气层厚度 h = R_s - R_c 必须**以 h 本身**归一，不能拿
                # R_0 或 R_s 当分母——h 比它们小一个量级，用大尺度的分母会把
                # 一个真实的变化压成"看起来收敛"。
                "h_over_R0": shock - contact,
                "min_density": float(field.density[-1].min()),
                "mass_relative_drift": abs(
                    field.metadata["total_mass_final"] / field.metadata["total_mass_initial"] - 1.0
                ),
            }
        )
    # **相邻两档之间**求差，而不是每一档都去和最细的一档比。
    #
    # 与最细一档比有两个毛病：最细那一行按定义是 0，于是"最细网格与自己一致"会
    # 被读成收敛；而且最后一对永远算不出观测阶（分母是 0）。相邻差分没有这两个
    # 问题，代价是没有一个"参考值"——判据因此只能是**序列本身是否单调下降**。
    for coarse, fine in zip(rows, rows[1:]):
        ratio = float(fine["cells_inside"]) / float(coarse["cells_inside"])
        fine["refinement_ratio"] = ratio
        for key in ("contact_over_R0", "shock_over_R0", "h_over_R0"):
            delta = abs(float(fine[key]) - float(coarse[key]))
            fine[f"delta_{key}"] = delta
            fine[f"delta_{key}_relative"] = delta / max(abs(float(fine[key])), 1e-30)
    return {
        "levels": rows,
        "finest": max(levels),
        "normalisation": (
            "R_c、R_s 以自身归一；空气层厚度 h = R_s - R_c 以 h 本身归一（§5.2）。"
            "偏差是**相邻两档之差**，不是与最细一档之差"
        ),
        "within_tolerance": all(
            abs(float(rows[-1][key])) < CONVERGENCE_TOLERANCE
            for key in ("delta_contact_over_R0_relative", "delta_shock_over_R0_relative", "delta_h_over_R0_relative")
        ),
        "monotone": _is_monotone(
            rows,
            ("delta_contact_over_R0_relative", "delta_shock_over_R0_relative", "delta_h_over_R0_relative"),
        ),
        "observed_order": _observed_order(rows, "delta_h_over_R0_relative"),
        "convergence_caveat": (
            "判据是相邻差分序列是否随加密单调下降，以及观测阶是否为正。"
            "出现负阶或相邻差分变大即为**非单调**——那比不收敛更值得警惕："
            "它说明离散误差被别的效应盖住了（例如产物场网格与前置网格同时加密时，"
            "两套误差的相对大小在变）"
        ),
    }


def _is_monotone(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> bool:
    """**每一项**都必须单调下降，不是只看 h。

    只看 h 会漏掉真正的问题：实测 ``Δh/h`` 是 6.9e-2 → 5.7e-2（在降），而同时
    ``ΔR_c/R_c`` 是 7.2e-5 → 4.0e-3（**涨了 55 倍**）。只盯着一项就会把一次
    "接触面位置不收敛"读成通过。
    """
    for key in keys:
        values = [abs(float(row[key])) for row in rows if key in row]
        if not all(b <= a for a, b in zip(values, values[1:])):
            return False
    return True


#: 网格/时间步加密的**幅度**要求。只要求"差分在缩小"是不够的：一个下降极慢的
#: 序列也能满足单调性，却离收敛很远。这里规定**最细一档**的相邻相对变化必须
#: 低于这个值。2e-2 是一个声明的工程门槛，不是从格式阶数推出来的——写在这里
#: 是为了让它可被质疑，而不是让它隐形。
CONVERGENCE_TOLERANCE = 2.0e-2


def _observed_order(rows: Sequence[dict[str, Any]], key: str, level_key: str = "cells_inside"):
    """由**相邻差分的相邻差分**给出观测阶。

    差分挂在每一对的较细一档上，所以序列长度是 ``len(rows) - 1``。网格按
    ``cells`` 等比加密时 ``h ∝ 1/cells``；时间步那一组的分级量是 ``cfl``。
    """
    import math as _math

    pairs = [(float(row[key]), float(row[level_key])) for row in rows if key in row]
    order = []
    for (d_coarse, l_coarse), (d_fine, l_fine) in zip(pairs, pairs[1:]):
        # 分级量**变大**是加密（cells），**变小**也是加密（cfl），所以比值取绝对值：
        # 写死成 ``cells`` 的方向会让时间步那一组永远算不出阶。
        span = abs(_math.log(l_fine / l_coarse))
        if d_coarse <= 0.0 or d_fine <= 0.0 or span <= 0.0:
            order.append(None)
            continue
        order.append(_math.log(d_coarse / d_fine) / span)
    return order


def _timestep_check(config, radius: float) -> dict[str, Any]:
    """在同一网格上减小 CFL，比较同一时刻的解。

    **两段都要测**。旧版本只改前置求解器的 CFL，产物构造那一段一直用配置里的
    值；而产物构造正是能量不变量偏差的来源，而且它的偏差随 CFL **反向**变化
    （CFL 越小偏差越大），所以"只测前置段"恰好漏掉了唯一有 CFL 依赖的地方。
    """
    cells = int(config.section("reference_field").get("cells_inside", 2000))
    rows = []
    for cfl in (0.4, 0.2, 0.1):
        settings = _precursor_settings(config, cells)
        settings.cfl = cfl
        relative = handover_relative_time(config, radius)
        detonation = reference_detonation_config(config)
        detonation.cfl = cfl
        field = solve_reference_field(
            charge_radius=radius,
            config=settings,
            detonation_velocity=reference_detonation_velocity(config),
            detonation_config=detonation,
            sample_times=[relative],
        )
        contact = float(field.contact_radius[-1]) / radius
        shock = float(field.shock_radius[-1]) / radius
        rows.append(
            {"cfl": cfl, "contact_over_R0": contact, "shock_over_R0": shock,
             "h_over_R0": shock - contact}
        )
    for coarse, fine in zip(rows, rows[1:]):
        for key in ("contact_over_R0", "h_over_R0"):
            delta = abs(float(fine[key]) - float(coarse[key]))
            fine[f"delta_{key}"] = delta
            fine[f"delta_{key}_relative"] = delta / max(abs(float(fine[key])), 1e-30)
    return {
        "levels": rows,
        # 时间步这一组没有 R_s 的分级差（只算了 R_c 与 h），键表因此与网格那组不同。
        "within_tolerance": all(
            abs(float(rows[-1][key])) < CONVERGENCE_TOLERANCE
            for key in ("delta_contact_over_R0_relative", "delta_h_over_R0_relative")
        ),
        "monotone": _is_monotone(
            rows, ("delta_contact_over_R0_relative", "delta_h_over_R0_relative")
        ),
        "observed_order": _observed_order(rows, "delta_h_over_R0_relative", "cfl"),
    }


def _scaling_check(config, radius: float, other_radius: float) -> dict[str, Any]:
    """用一个不同半径的独立小算例核查 ``q_R0(r,t) = q_ref(r/k, t/k)``。

    只缩放长度与时间：对应状态的 ``rho/u/e/p`` 不乘 ``k``，质量与能量随 ``k^3``。
    """
    cells = max(int(config.section("reference_field").get("cells_inside", 2000)) // 4, 8)
    k = float(other_radius) / float(radius)
    # 交接时刻随半径同比例缩放，而 ``_solve_at_handover`` 用的就是**该半径自己的**
    # 交接时刻，所以这里不需要再乘一次 k。
    base, _ = _solve_at_handover(config, radius, cells, detonation_cells=cells)
    target, _ = _solve_at_handover(config, other_radius, cells, detonation_cells=cells)
    # 两个算例各求各的拉格朗日网格，所以比较的是**同一物理时刻**下各自的当前
    # 单元中心，而不是某一套固定的初始坐标。
    mapped = target.final_cell_radii / k
    deviations = {}
    for name in ("density", "velocity", "specific_energy"):
        reference_values = np.interp(mapped, base.final_cell_radii, getattr(base, name)[-1])
        actual = getattr(target, name)[-1]
        scale = max(float(np.abs(reference_values).max()), 1e-30)
        deviations[name] = float(np.max(np.abs(actual - reference_values)) / scale)
    return {
        "k": k,
        "reference_radius": float(radius),
        "other_radius": float(other_radius),
        "state_relative_deviation": deviations,
        "contact_over_R0_reference": float(base.contact_radius[-1]) / radius,
        "contact_over_R0_other": float(target.contact_radius[-1]) / other_radius,
        "contact_shape_deviation": abs(
            float(base.contact_radius[-1]) / radius
            - float(target.contact_radius[-1]) / other_radius
        ),
        "caveat": (
            "尺度映射一致只证明**缩放实现自洽**，不证明物理相似性成立，也不证明"
            "原始产物场本身收敛（§5.5）。"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-convergence", action="store_true")
    args = parser.parse_args(argv)
    configure_stdout()
    config = load_config_from_args(args)
    radius = float(config.get("reference_field.charge_radius_m", 0.05))
    velocity = reference_detonation_velocity(config)
    relative = handover_relative_time(config, radius)
    transit = detonation_transit_time(radius, detonation_velocity=velocity)
    handover = handover_time(radius, detonation_velocity=velocity)
    delay = start_delay(radius)

    # 诊断入口：允许拿到未验收的场，但产物会带着未验收的标记一起写出去。
    field = build_reference(config, verbose=True, allow_unaccepted=True)
    out_dir = Path(args.output_dir) if args.output_dir else config.path_of(
        "training.output_dir", default=Path("outputs/run")
    ) / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- §5.4 的正值与守恒检查 -------------------------------------------
    pressure = field.pressure()
    masks = {
        "all": np.ones_like(pressure, dtype=bool),
        "product": np.broadcast_to(field.material == 0, pressure.shape),
        "air": np.broadcast_to(field.material == 1, pressure.shape),
    }
    at = relative + transit  # 交接时刻的物理时间 T_h
    checks = {
        "min_density": float(field.density.min()),
        "min_specific_energy": float(field.specific_energy.min()),
        "min_pressure": float(min(float(pressure[m].min()) for m in masks.values())),
        "mass_relative_drift": abs(
            field.metadata["total_mass_final"] / field.metadata["total_mass_initial"] - 1.0
        ),
        "contact_over_R0_at_handover": float(np.interp(at, field.times, field.contact_radius)) / radius,
        "shock_over_R0_at_handover": float(np.interp(at, field.times, field.shock_radius)) / radius,
        "gap_m_at_handover": float(
            np.interp(at, field.times, field.shock_radius)
            - np.interp(at, field.times, field.contact_radius)
        ),
        "outer_boundary_undisturbed": _outer_boundary_check(field),
    }
    checks["gap_over_R0_at_handover"] = checks["gap_m_at_handover"] / radius
    checks["positive_ok"] = bool(
        checks["min_density"] > 0.0
        and checks["min_specific_energy"] > 0.0
        and checks["min_pressure"] > 0.0
    )

    metadata = field.metadata or {}
    report: dict[str, Any] = {
        "reference_radius_m": radius,
        "construction": REFERENCE_CONSTRUCTION,
        "acceptance_status": metadata.get("acceptance_status"),
        "acceptance_reason": metadata.get("acceptance_reason"),
        # 方案 B 的时钟。相对时间是"相对 T_b"，物理时间才是 LS-DYNA 的原始时间。
        "clock": {
            "detonation_velocity_m_s": velocity,
            "transit_time_Tb_s": transit,
            "start_delay_epsilon_s": delay,
            "handover_time_Th_s": handover,
            "handover_relative_to_Tb_s": relative,
            "note": (
                "PINN 的时间原点 tau = T - T_h；本命令的 sample_times 相对 T_b，"
                "所以初值那一刻对应 relative = epsilon"
            ),
        },
        "detonation_front_state": detonation_front_state(
            detonation_velocity=velocity
        ).as_dict(),
        "ideal_cj_state": ideal_cj_state().as_dict(),
        "derived": {
            "e_star_j_kg": mat.PRODUCT_SPECIFIC_ENERGY,
            "u_star_m_s": mat.REFERENCE_VELOCITY,
            "product_pressure_scale_pa": mat.PRODUCT_VOLUME_ENERGY,
            "air_pressure_scale_pa": mat.AIR_DENSITY * mat.REFERENCE_VELOCITY**2,
        },
        "field": field.as_dict(),
        "staged_budget": metadata.get("staged_budget"),
        "acceptance_metrics": metadata.get("acceptance_metrics"),
        "checks": checks,
        "material": {
            "jwl": mat.product_jwl().as_dict(),
            "rho0": mat.PRODUCT_REFERENCE_DENSITY,
            "E0_volume_energy": mat.PRODUCT_VOLUME_ENERGY,
            "gamma_air": mat.AIR_GAMMA,
            "air_density": mat.AIR_DENSITY,
            "air_pressure": mat.AIR_PRESSURE,
        },
    }
    if not args.skip_convergence:
        report["grid_convergence"] = _convergence_check(config, radius, (500, 1000, 2000))
        report["timestep_sensitivity"] = _timestep_check(config, radius)
        report["scaling_check"] = _scaling_check(config, radius, 2.0 * radius)

    # -- 把完整的 §5 判定写回参考场 ------------------------------------
    #
    # 单次求解给不出收敛证据，所以它自己的状态永远是"缺证据"（见
    # ``physics.precursor.assess_acceptance``）。本命令跑完了加密、时间步与尺度
    # 三项核查，才有资格给出完整判定；判定与证据一起写回缓存，正式入口读的就是
    # 这一份。
    evidence: dict[str, Any] = {}
    if "grid_convergence" in report:
        grid = report["grid_convergence"]
        finest = grid["levels"][-1]
        evidence["grid_convergence"] = bool(grid["monotone"]) and all(
            abs(float(finest[key])) < CONVERGENCE_TOLERANCE
            for key in ("delta_contact_over_R0_relative", "delta_shock_over_R0_relative", "delta_h_over_R0_relative")
        )
    if "timestep_sensitivity" in report:
        step = report["timestep_sensitivity"]
        finest = step["levels"][-1]
        evidence["timestep_sensitivity"] = bool(step["monotone"]) and all(
            abs(float(finest[key])) < CONVERGENCE_TOLERANCE
            for key in ("delta_contact_over_R0_relative", "delta_h_over_R0_relative")
        )
    if "scaling_check" in report:
        scaling = report["scaling_check"]
        evidence["scale_mapping"] = all(
            value < 1.0e-6 for value in scaling["state_relative_deviation"].values()
        )
    if evidence:
        refreshed = record_acceptance(field, evidence)
        report["acceptance"] = refreshed
        report["field"]["acceptance_status"] = refreshed["acceptance_status"]
        report["field"]["acceptance_reason"] = refreshed["acceptance_reason"]
        report["acceptance_status"] = refreshed["acceptance_status"]
        report["acceptance_reason"] = refreshed["acceptance_reason"]

    write_json(out_dir / "reference_field_report.json", report)
    _save_field(field, out_dir / "reference_field.npz")
    print(_render(report))
    print(f"\nwritten: {out_dir / 'reference_field_report.json'}")
    return 0 if checks["positive_ok"] else 1


def _outer_boundary_check(field) -> dict[str, Any]:
    """外边界是否受扰动：最外侧若干单元的压力与速度相对环境态的偏离。"""
    ambient_p = mat.AIR_PRESSURE
    tail = slice(-5, None)
    pressure = field.pressure()[-1][tail]
    velocity = field.velocity[-1][tail]
    return {
        "max_relative_pressure_excess": float(np.max(np.abs(pressure - ambient_p)) / ambient_p),
        "max_absolute_velocity": float(np.max(np.abs(velocity))),
        "n_cells_inspected": int(pressure.size),
    }


def _save_field(field, path: Path) -> None:
    # 把场**内容**哈希写进元数据（§7.2）：证据必须能核对到当前这份场。
    field.metadata["field_content_hash"] = _field_content_hash(field)
    # **原子写**（§7.3）：先写临时文件再 ``os.replace``，避免进程中断留下半份缓存。
    # 临时文件名必须以 ``.npz`` 结尾（NumPy 对不以 ``.npz`` 结尾的字符串会**自动补**
    # ``.npz``；若写 ``formal.npz.tmp`` 实际落盘的是 ``formal.npz.tmp.npz``，
    # 随后 ``os.replace`` 找不到目标直接 FileNotFoundError）。
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        times=field.times,
        cell_radii=field.cell_radii,
        density=field.density,
        velocity=field.velocity,
        specific_energy=field.specific_energy,
        material=field.material,
        contact_radius=field.contact_radius,
        shock_radius=field.shock_radius,
        reference_radius=np.array([field.reference_radius]),
        # 诊断产物也带**构造来源**：一个没有这个键的文件不可能被误当成方案 B
        # 的场，也不可能因为"文件名对得上"而被复用。
        construction=np.array([REFERENCE_CONSTRUCTION]),
        metadata=np.array([json.dumps(field.metadata, default=str)]),
    )
    os.replace(tmp, path)


def _render(report: dict[str, Any]) -> str:
    lines = ["# 参考初始场报告（方案 B）", ""]
    clock = report["clock"]
    checks = report["checks"]
    lines.append("## 时钟")
    lines.append("")
    lines.append(f"- D = {clock['detonation_velocity_m_s']:.6f} m/s（卡片值）")
    lines.append(f"- T_b = R_0/D = {clock['transit_time_Tb_s']:.9e} s")
    lines.append(f"- epsilon = {clock['start_delay_epsilon_s']:.9e} s")
    lines.append(f"- T_h = T_b + epsilon = {clock['handover_time_Th_s']:.9e} s")
    lines.append("")
    lines.append(f"## 初值场（交接时刻 T_h，相对 T_b 为 "
                 f"{clock['handover_relative_to_Tb_s']:.6e} s）")
    lines.append("")
    lines.append(f"- R_c = {checks['contact_over_R0_at_handover']:.6f} R_0")
    lines.append(f"- R_s = {checks['shock_over_R0_at_handover']:.6f} R_0")
    lines.append(f"- 空气层厚度 h = {checks['gap_m_at_handover']:.6e} m "
                 f"= {checks['gap_over_R0_at_handover']:.6e} R_0")
    lines.append("")
    lines.append("## 验收状态（§5 全项）")
    lines.append("")
    lines.append(f"- `{report['acceptance_status']}` — {report['acceptance_reason']}")
    supplied = (report.get("acceptance") or {}).get("acceptance_evidence_supplied")
    if supplied is not None:
        lines.append(f"- 已提供证据：{supplied}")
    lines.append("")
    budget = (report.get("field") or {}).get("staged_budget")
    if budget:
        lines.append("## 分阶段收支（产物构造 / 网格交接 / 前置演化）")
        lines.append("")
        lines.append(
            "三段**分开**记账：拉格朗日格式在前置阶段的单元质量是常量，那条"
            "守恒律**在构造上**成立，因此它证明不了交接（换网格重采样）没有损失。"
        )
        lines.append("")
        lines.append("| 阶段 | ΔM / M_exact | ΔE / E_exact |")
        lines.append("|---|---|---|")
        s1 = budget["stage1_product_construction"]
        s2 = budget["stage2_handover"]
        s3 = budget["stage3_precursor_evolution"]
        lines.append(
            f"| 1 产物构造 | {s1['mass_relative']:+.3e} | {s1['energy_relative']:+.3e} |"
        )
        lines.append(
            f"| 2 网格交接 | {s2['mass_relative']:+.3e} | {s2['energy_relative']:+.3e} |"
        )
        # 缓存里的元数据来自**写出它的那个版本**。渲染器不能因为少一个键就崩掉，
        # 那会把一次已经算完的运行变成一次 traceback，而且看起来像求解失败。
        total = s3.get("energy_relative_total_domain")
        lines.append(
            f"| 3 前置演化（全域，节点加权） | {s3['mass_relative']:+.3e} | "
            + ("n/a（该缓存早于这一项）" if total is None else f"{total:+.3e}")
            + " |"
        )
        lines.append("")
        cell_convention = s3.get("energy_relative_product_only_cell_convention")
        if cell_convention is not None:
            lines.append(
                f"（第 3 阶段若只看产物区、且用单元中心约定，会读到 "
                f"{cell_convention:+.3e}——那不是守恒读数：产物推动空气，能量是转出去的；"
                "而格式守恒的是节点加权约定。）"
            )
            lines.append("")
    lines.append("## §5.4 要求的检查")
    lines.append("")
    lines.append(f"- 正值：{'通过' if checks['positive_ok'] else '**未通过**'}"
                 f"（min rho = {checks['min_density']:.6g}，"
                 f"min e = {checks['min_specific_energy']:.6g}，"
                 f"min p = {checks['min_pressure']:.6g}）")
    lines.append(f"- 质量相对漂移：{checks['mass_relative_drift']:.3e}")
    boundary = checks["outer_boundary_undisturbed"]
    lines.append(
        f"- 外边界：最大相对压力偏离 {boundary['max_relative_pressure_excess']:.3e}，"
        f"最大速度 {boundary['max_absolute_velocity']:.3e} m/s"
    )
    if "grid_convergence" in report:
        grid = report["grid_convergence"]
        lines.append("")
        lines.append("## 网格加密")
        lines.append("")
        lines.append("| cells_inside | R_c/R_0 | R_s/R_0 | h/R_0 | ΔR_c/R_c | ΔR_s/R_s | Δh/h |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in grid["levels"]:
            if "delta_h_over_R0_relative" in row:
                cells_col = (
                    f"{grid['levels'][grid['levels'].index(row) - 1]['cells_inside']}"
                    f"→{row['cells_inside']}"
                )
                cols = (
                    f"{row['delta_contact_over_R0_relative']:.3e} | "
                    f"{row['delta_shock_over_R0_relative']:.3e} | "
                    f"{row['delta_h_over_R0_relative']:.3e}"
                )
            else:
                cells_col = f"{row['cells_inside']}"
                cols = "（基准档） | （基准档） | （基准档）"
            lines.append(
                f"| {cells_col} | {row['contact_over_R0']:.6f} | "
                f"{row['shock_over_R0']:.6f} | {row['h_over_R0']:.6e} | {cols} |"
            )
        lines.append("")
        lines.append(
            f"- 各项全部单调下降：{'是' if grid['monotone'] else '**否**'}"
            f"；最细一档的相对变化须 < {CONVERGENCE_TOLERANCE:.0e}"
            f"：{'是' if grid.get('within_tolerance') else '**否**'}"
        )
        lines.append(
            f"- 观测阶：{grid['observed_order']}（**三档只给得出一个估计值**，"
            "而且它建立在绝对差之上；不要把单个数字读成「格式是几阶」）"
        )
        lines.append(f"- {grid['convergence_caveat']}")
    if "timestep_sensitivity" in report:
        step = report["timestep_sensitivity"]
        lines.append("")
        lines.append("## 时间步敏感性")
        lines.append("")
        lines.append("| CFL | R_c/R_0 | R_s/R_0 | h/R_0 | ΔR_c/R_c | Δh/h |")
        lines.append("|---|---|---|---|---|---|")
        for row in step["levels"]:
            if "delta_h_over_R0_relative" in row:
                cols = (
                    f"{row['delta_contact_over_R0_relative']:.3e} | "
                    f"{row['delta_h_over_R0_relative']:.3e}"
                )
            else:
                cols = "（基准档） | （基准档）"
            lines.append(
                f"| {row['cfl']} | {row['contact_over_R0']:.6f} | "
                f"{row['shock_over_R0']:.6f} | {row['h_over_R0']:.6e} | {cols} |"
            )
        lines.append("")
        lines.append(
            f"- 各项全部单调下降：{'是' if step['monotone'] else '**否**'}"
            f"；最细一档的相对变化须 < {CONVERGENCE_TOLERANCE:.0e}"
            f"：{'是' if step.get('within_tolerance') else '**否**'}"
        )
        lines.append(
            f"- 观测阶：{step['observed_order']}（三档只给得出一个估计值）"
        )
    if "scaling_check" in report:
        scaling = report["scaling_check"]
        lines.append("")
        lines.append("## 尺度映射核查（§5.5）")
        lines.append("")
        lines.append(
            f"k = {scaling['k']:.4g}；独立算例 R_0 = {scaling['other_radius']:.4g} m"
        )
        for name, value in scaling["state_relative_deviation"].items():
            lines.append(f"- {name} 状态量相对偏差：{value:.3e}")
        lines.append(f"- R_c/R_0 形状偏差：{scaling['contact_shape_deviation']:.3e}")
        lines.append(f"- **注意**：{scaling['caveat']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
