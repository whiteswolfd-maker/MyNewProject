#!/usr/bin/env python3
"""Render the initial-error JSON into a compact Markdown report."""
from __future__ import annotations

import json
import math
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parents[1] / (
        "outputs/fullframes/time_input_experiment/initial_error/initial_error.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    out = path.parent / "report.md"
    lines = ["# 初始误差定位：A/B last 与 best", ""]
    lines.append(
        "A=legacy，B=similarity。理论 IC 与仿真数据比较分开；归一化误差除以"
        "全局 `rho_scale/u_scale/p_scale/length_scale`，不是相对局部真值的百分比。"
    )
    lines.append("")
    lines.append("## t0 理论初值（A_last/B_last）")
    lines.append("")
    lines.append(
        "理论目标：`R_c0=case.initial.Rc0`；空气 `rho_a_sep/u_a_sep/p_a_sep`"
        "（20 例相同）。Rc 误差已除以 `R_sep`；空气为归一化 RMSE。"
    )
    lines.append(
        "| case | role | Rc0 (m) | A Rc pred | B Rc pred | "
        "A Rc err/Rsep | B Rc err/Rsep | A rho RMSE | B rho RMSE | "
        "A u RMSE | B u RMSE | A p RMSE | B p RMSE |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid, a in data["theoretical_t0"]["A_last"].items():
        b = data["theoretical_t0"]["B_last"][cid]
        lines.append(
            f"| {cid} | {a['role']} | {a['Rc']['target']:.6g} | "
            f"{a['Rc']['prediction']:.6g} | {b['Rc']['prediction']:.6g} | "
            f"{a['Rc']['normalised_error']:.5f} | "
            f"{b['Rc']['normalised_error']:.5f} | "
            f"{a['air']['rho']['rmse']:.5f} | {b['air']['rho']['rmse']:.5f} | "
            f"{a['air']['u']['rmse']:.5f} | {b['air']['u']['rmse']:.5f} | "
            f"{a['air']['p']['rmse']:.5f} | {b['air']['p']['rmse']:.5f} |"
        )
    lines.append("")
    lines.append("## t0 聚合（归一化 RMSE）")
    lines.append("| label | split | Rc | rho | u | p |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for label, roles in data["theoretical_t0_aggregate"].items():
        for split, value in roles.items():
            lines.append(
                f"| {label} | {split} | {value['Rc_rmse_normalised']:.5f} | "
                f"{value['rho_rmse_normalised']:.5f} | "
                f"{value['u_rmse_normalised']:.5f} | "
                f"{value['p_rmse_normalised']:.5f} |"
            )
    lines.append("")
    lines.append("## 读表注意：不能概括为“所有验证半径所有字段都更差”")
    lines.append("")
    lines.append(
        "- t0 理论 IC：B 的验证组 Rc 和 rho 在全部验证半径都更差；"
        "p 和 u 并非所有半径都更差。"
    )
    lines.append(
        "- 1% 仿真窗口：B 的验证组 rho 和 p 在全部验证半径更差；"
        "u 并非所有半径都更差。"
    )
    lines.append(
        "- 10% 仿真窗口：B 的验证组 rho 在 10/10 个半径更好；"
        "聚合 u 和 p 更差，但不能外推成所有半径的所有字段。"
    )
    lines.append(
        "- rest（10%–100%）：B 的 rho 大幅更好；u 和 p 是个别半径互有胜负，"
        "不是所有字段都更差。"
    )
    lines.append(
        "- 这些是描述性统计；早期误差大不能单独证明“初值权重小”。"
        "R350 的中/晚段接触回归是独立现象，不能并入初值问题。"
    )
    lines.append("")
    lines.append("## 早期仿真窗口（实际观测行，A/B 共同域；归一化 RMSE）")
    lines.append("")
    for suffix in ("last", "best"):
        lines.append(f"### {suffix}")
        lines.append("| split | window | model | n | rho | u | p |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for role in ("train", "validation"):
            for window in ("0_1pct", "0_5pct", "0_10pct", "rest_10pct", "0_100pct"):
                for label in (f"A_{suffix}", f"B_{suffix}"):
                    rows = []
                    for entry in data["early_simulation"][label].values():
                        row = entry[window]
                        if row.get("status") != "ok" or row.get("role") != role:
                            continue
                        metrics = row.get("common_metrics") or row.get("own_metrics")
                        if metrics:
                            rows.append(metrics)
                    if not rows:
                        continue
                    values = []
                    for field in ("rho", "u", "p"):
                        n = sum(row[field]["n"] for row in rows)
                        mse = (
                            sum(row[field]["mse"] * row[field]["n"] for row in rows) / n
                            if n else None
                        )
                        values.append(math.sqrt(mse) if mse is not None else None)
                    lines.append(
                        f"| {role} | {window} | {label} | "
                        f"{sum(row['rho']['n'] for row in rows)} | "
                        f"{values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} |"
                    )
        lines.append("")
    lines.append("## 验证集逐例早期误差（last，A/B 共同域归一化 RMSE）")
    lines.append(
        "| case | 1% rho A/B | 1% u A/B | 1% p A/B | "
        "10% rho A/B | 10% p A/B | rest rho A/B | rest u A/B | rest p A/B |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    validation_ids = [
        cid for cid, row in data["early_simulation"]["A_last"].items()
        if next(iter(row.values())).get("role") == "validation"
    ]
    for cid in validation_ids:
        vals = {}
        for label in ("A_last", "B_last"):
            vals[label] = {}
            for window in ("0_1pct", "0_10pct", "rest_10pct"):
                row = data["early_simulation"][label][cid][window]
                metrics = row.get("common_metrics") or row.get("own_metrics")
                vals[label][window] = {
                    field: metrics[field]["rmse"] for field in ("rho", "u", "p")
                } if metrics else {field: None for field in ("rho", "u", "p")}

        def pair(window, field):
            a = vals["A_last"][window][field]
            b = vals["B_last"][window][field]
            return f"{a:.5f}/{b:.5f}"

        lines.append(
            f"| {cid} | {pair('0_1pct', 'rho')} | {pair('0_1pct', 'u')} | "
            f"{pair('0_1pct', 'p')} | {pair('0_10pct', 'rho')} | "
            f"{pair('0_10pct', 'p')} | {pair('rest_10pct', 'rho')} | "
            f"{pair('rest_10pct', 'u')} | {pair('rest_10pct', 'p')} |"
        )
    lines.append("")
    lines.append("## R350 接触误差分时段（实际标签，MSE/RMSE/覆盖）")
    lines.append("| label | third | MSE | RMSE | covered/n |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, thirds in data["R350_thirds"].items():
        for index in ("0", "1", "2"):
            value = thirds[index]
            lines.append(
                f"| {label} | {index} | {value.get('mse'):.6f} | "
                f"{value.get('rmse'):.6f} | "
                f"{value.get('covered_count')}/{value.get('n')} |"
            )
    lines.append("")
    lines.append("## 早期共同域排除计数")
    lines.append("")
    for suffix in ("last", "best"):
        lines.append(f"### {suffix}")
        lines.append(
            "| window | selected total | used | A inside | B inside | common | "
            "A excluded | B excluded | common excluded |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for window, row in data["pair_early_simulation"][suffix].items():
            ex = row["exclusions"]
            lines.append(
                f"| {window} | {sum(x['selected_total'] for x in ex)} | "
                f"{sum(x['used'] for x in ex)} | "
                f"{sum(x['A_inside'] for x in ex)} | "
                f"{sum(x['B_inside'] for x in ex)} | "
                f"{sum(x['common'] for x in ex)} | "
                f"{sum(x['A_excluded'] for x in ex)} | "
                f"{sum(x['B_excluded'] for x in ex)} | "
                f"{sum(x['common_excluded'] for x in ex)} |"
            )
        lines.append("")
    lines.append("## 说明")
    lines.append(
        "- 理论 IC 的 zeta 网格为 0.01–0.99；不是物理半径，也不使用接触区间中点。"
    )
    lines.append(
        "- 仿真窗口只选实际观测帧/行；`rest_10pct` 与 `0_100pct` 采用帧均衡子样"
        "（最多 200 帧、每帧 100 行），JSON 中记录 `selected_total/used`。"
    )
    lines.append(
        "- 接触窗口只使用实际标签时刻；没有插值标签。"
    )
    lines.append(
        "- 本分析不训练、不改采样/权重/网络。"
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
