"""数据质量审查：哪些时空范围真正可以用来监督。

输出每个算例的：有效帧数与有效时间范围、前沿识别的可靠比例、空气层最薄处、
材料分区方法与它的不确定性，以及**明确不可用**的部分。

这份报告存在的理由很直接：监督标签的材料归属、有效掩码和时间范围都由它给出，
而一个没有被审查过的分区，会在训练里变成一个没人看得见的假设。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from ._common import (
    add_common_arguments,
    configure_stdout,
    load_cases,
    load_config_from_args,
    write_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    configure_stdout()
    config = load_config_from_args(args)

    cases = load_cases(config, split=args.split)
    reports: list[dict[str, Any]] = []
    for case in cases:
        report = case.quality_report()
        report["split"] = case.source.get("split")
        report["source_directory"] = case.source.get("directory")
        report["field_map"] = case.source.get("field_map")
        reports.append(report)

    out_dir = Path(args.output_dir) if args.output_dir else config.path_of(
        "training.output_dir", default=Path("outputs/run")
    ) / "audit_data"
    payload = {"cases": reports}
    write_json(out_dir / "data_audit.json", payload)
    text = render(reports)
    (out_dir / "data_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_audit.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten: {out_dir / 'data_audit.json'}")
    return 0 if all(r["n_frames_supervisable"] > 0 for r in reports) else 1


def render(reports: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# 数据质量审查",
        "",
        "本表由 `python -m spherical_blast_pinn.cli.audit_data` 生成。",
        "",
        "| 算例 | R_0 (m) | 帧数 | 可监督帧 | t_PINN 终止 (s) | 接触面有效率 | 主激波有效率 | 最薄空气层单元数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            "| {case_id} | {r0:.3f} | {n} | {ok} | {tmax} | {cv:.2f} | {sv:.2f} | {gap} |".format(
                case_id=r["case_id"],
                r0=r["charge_radius_m"],
                n=r["n_frames"],
                ok=r["n_frames_supervisable"],
                tmax=(
                    f"{r['t_pinn_last_valid_s']:.4e}"
                    if r.get("t_pinn_last_valid_s") is not None
                    else "—"
                ),
                cv=r["contact_valid_fraction"],
                sv=r["shock_valid_fraction"],
                gap=r["min_air_cells"],
            )
        )
    lines.append("")
    lines.append("## 字段来源与已知限制")
    lines.append("")
    for r in reports:
        lines.append(f"### {r['case_id']}")
        lines.append("")
        lines.append(f"- 目录：`{r.get('source_directory')}`")
        lines.append(f"- 字段映射：`{r.get('field_map')}`")
        lines.append(f"- 比内能来源：{r['energy_source']}")
        lines.append(f"- 材料分区来源：{r['material_masks_from']}")
        lines.append(f"- 识别参数：`{r['detector']}`")
        lines.append(
            f"- 有效时间范围：物理 {r['t_phys_first_s']:.4e} – {r['t_phys_last_s']:.4e} s；"
            f"启动时刻 {r['time_start_s']:.4e} s"
        )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
