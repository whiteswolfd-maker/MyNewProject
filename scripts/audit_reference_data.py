#!/usr/bin/env python3
"""Thin entry point for the read-only data and formula audit.

Stages:
    manifest    file identity and solver metadata for all configured cases
    imports     independent raw-vs-import comparison for the focus cases
    metrics     fixed-radius pressure histories and Henrych comparison
    consistency RH jumps, constraint compatibility, implementation checks
    report      Markdown summary of the collected evidence
    all         run every stage in that order

Nothing here writes to a cache, checkpoint or training output.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import _case_specs, _make_case, _section
from spherical_blast_pinn.contact_air.reference_audit import (
    AuditError,
    blast_metrics,
    case_manifest,
    constraint_compatibility,
    implementation_checks,
    import_comparison,
    peak_preservation,
    rh_audit,
)


def _load_audit(path: Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload["audit"]


def _case_config(audit: dict, case_id: str) -> dict:
    for case in audit["cases"]:
        if case["case_id"] == case_id:
            return case
    raise AuditError(f"case {case_id!r} is not in the audit config")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _case_data(audit: dict, case_id: str):
    config = load_config(_REPO_ROOT / "configs/contact_air/split_train_r50_500_cw3.yaml")
    section = _section(config)
    spec = next(
        (item for item in _case_specs(section) if str(item["case_id"]) == case_id),
        None,
    )
    if spec is None:
        raise AuditError(f"{case_id}: not present in the training configuration")
    return _make_case(section, spec, config, allow_synthetic=False)


def _plot_histories(history_map: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(history_map), figsize=(4.3 * len(history_map), 4.2),
                             squeeze=False)
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
    for ax, (case_id, histories) in zip(axes[0], history_map.items()):
        for color, (label, history) in zip(colors, histories.items()):
            ax.plot(np.asarray(history["t"], float) * 1e6,
                    np.asarray(history["dp_mpa"], float),
                    color=color, linewidth=1.6, label=f"Z={label}")
        ax.set_title(case_id, fontsize=10)
        ax.set_xlabel("raw time (µs)")
        ax.set_ylabel("overpressure (MPa)")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed-radius incident overpressure histories", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pressure_histories.png", dpi=150)
    plt.close(fig)


def _plot_similarity(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"r50_full": "#1f77b4", "r500_full": "#d95f02", "r1000_full": "#2ca02c"}
    for case_id in colors:
        subset = [row for row in rows if row["case_id"] == case_id
                  and row.get("peak_mpa") is not None]
        if not subset:
            continue
        subset.sort(key=lambda row: row["z_m_kg13"])
        ax.plot([row["z_m_kg13"] for row in subset],
                [row["peak_mpa"] for row in subset],
                marker="o", markersize=5, linewidth=2.0,
                color=colors[case_id], label=case_id)
    ax.set_xlabel("scaled distance Z (m/kg$^{1/3}$)")
    ax.set_ylabel("peak incident overpressure (MPa)")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "peak_vs_scaled_distance.png", dpi=150)
    plt.close(fig)


def _report(audit: dict, root: Path, out_dir: Path, manifest, metrics_rows,
            import_rows, preservation_rows, consistency, compatibility) -> None:
    lines = [
        "# Reference data and implementation audit",
        "",
        f"Audit config version: {audit['version']}",
        "",
        "## Layer A - original simulation",
        "",
    ]
    if manifest:
        normal = sum(1 for case in manifest["cases"] if case["normal_termination"])
        lines.append(f"- cases inspected: {len(manifest['cases'])}; normal termination: {normal}")
        for case in manifest["cases"]:
            lines.append(
                f"- {case['case_id']}: raw frames {case['raw_frame_count']}, "
                f"raw time {case['raw_time_start_s']:.3e}..{case['raw_time_end_s']:.3e} s, "
                f"charge mass {case['charge_mass_kg']:.6g} kg (uniform sphere)"
            )
    else:
        lines.append("- manifest not run")
    lines += ["", "## Layer B - import chain", ""]
    if import_rows:
        worst = max(
            (row for row in import_rows if row.get("max_rel_diff") is not None),
            key=lambda row: row["max_rel_diff"], default=None,
        )
        lines.append(f"- raw-vs-import comparisons: {len(import_rows)} rows")
        if worst:
            lines.append(
                f"- worst relative difference: {float(worst['max_rel_diff']):.3e} "
                f"({worst['case_id']} {worst['field']} {worst['region']} frame {worst['frame']})"
            )
        statuses = sorted({row.get("status") for row in import_rows})
        lines.append(f"- statuses: {statuses}")
    else:
        lines.append("- import comparison not run")
    lines += ["", "## Layer C - empirical references and compatibility", ""]
    lines.append(f"- Henrych status: {audit['references']['henrych']['status']}")
    lines.append(f"- Kingery-Bulmash: {audit['references']['kingery_bulmash']['status']} "
                 f"({audit['references']['kingery_bulmash']['reason']})")
    if metrics_rows:
        comparable = [row for row in metrics_rows if row.get("henrych_mpa") is not None]
        lines.append(f"- peak-overpressure rows: {len(metrics_rows)}; "
                     f"within the Henrych range: {len(comparable)}")
        lines += [
            "",
            "| case | Z | peak MPa | Henrych MPa | relative | arrival ms | status | flags |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        def number(row, key):
            value = row.get(key)
            return None if value in (None, "", "None") else float(value)

        for row in metrics_rows:
            peak = number(row, "peak_mpa")
            henrych = number(row, "henrych_mpa")
            relative = number(row, "peak_relative_difference")
            arrival = number(row, "arrival_s")
            lines.append(
                f"| {row['case_id']} | {number(row, 'z_m_kg13'):.2f} | "
                f"{'n/a' if peak is None else f'{peak:.4f}'} | "
                f"{'n/a' if henrych is None else f'{henrych:.4f}'} | "
                f"{'n/a' if relative is None else f'{relative:+.3f}'} | "
                f"{'n/a' if arrival is None else f'{arrival*1e3:.3f}'} | "
                f"{row.get('status')} | {row.get('flags')} |"
            )
    if preservation_rows:
        lines += [
            "",
            "### Peak visible after import",
            "",
            "| case | Z | raw peak MPa | imported peak MPa | difference |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in preservation_rows:
            lines.append(
                f"| {row['case_id']} | {float(row['z_m_kg13']):.2f} | "
                f"{float(row['raw_peak_mpa']):.4f} | "
                f"{float(row['imported_peak_mpa']):.4f} | "
                f"{float(row['peak_difference_mpa']):+.4f} |"
            )
    if compatibility:
        lines.append(f"- constraint compatibility entries: {len(compatibility)}")
    if consistency:
        lines.append(f"- consistency entries: {len(consistency)}")
    lines += [
        "",
        "## Layer D - PINN",
        "",
        "See `outputs/contact_air_diagnostics_cw3_4000/` for the 20-radius "
        "diagnostics already produced for the current cw3 model.",
        "",
        "## Limitations",
        "",
        "- Henrych coefficients are provisional: no printed copy was checked in this repository.",
        "- Kingery-Bulmash is unavailable; no arrival-time, duration or impulse reference is claimed.",
        "- Only the focus cases were taken through the raw-history stage in this pass.",
        "- No mesh or output-frequency sensitivity study was run.",
        "- Empirical agreement is evidence about the data, not proof that the simulation is correct.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(
        _REPO_ROOT / "configs/contact_air/reference_audit.yaml"))
    parser.add_argument("--stage", default="all",
                        choices=("manifest", "imports", "metrics", "consistency",
                                 "report", "all"))
    parser.add_argument("--cases", default=None,
                        help="comma-separated audit case ids for the raw-history stages")
    args = parser.parse_args(argv)
    audit = _load_audit(Path(args.config))
    root = (_REPO_ROOT / audit.get("repo_root", ".")).resolve()
    out_dir = root / audit["output_dir"]
    focus = (
        [part.strip() for part in args.cases.split(",") if part.strip()]
        if args.cases else list(audit["focus_cases"])
    )

    manifest_payload = None
    import_rows: list[dict] = []
    metrics_rows: list[dict] = []
    preservation_rows: list[dict] = []
    consistency = []
    compatibility = []

    if args.stage in ("manifest", "all"):
        cases = []
        for case in audit["cases"]:
            cases.append(case_manifest(
                case, root, audit["units"], audit["ambient"], audit["charge"]
            ))
        manifest_payload = {"version": audit["version"], "cases": cases}
        _write_json(out_dir / "manifest.json", manifest_payload)
        print(f"manifest: {len(cases)} cases")

    if args.stage in ("imports", "all"):
        for case_id in focus:
            case = _case_config(audit, case_id)
            import_rows.extend(import_comparison(case, root, audit["units"]))
        _write_csv(out_dir / "raw_import_comparison.csv", import_rows)
        print(f"imports: {len(import_rows)} rows")

    if args.stage in ("metrics", "all"):
        from spherical_blast_pinn.contact_air.reference_audit import (
            fixed_radius_histories,
        )
        history_map = {}
        for case_id in focus:
            case = _case_config(audit, case_id)
            histories = fixed_radius_histories(
                case, audit["units"], audit["ambient"], audit["charge"],
                audit["measurement_scaled_distances_m_kg13"],
            )
            history_map[case_id] = histories
            metrics_rows.extend(blast_metrics(
                case, audit["units"], audit["ambient"], audit["charge"],
                audit["measurement_scaled_distances_m_kg13"], histories=histories,
            ))
            preservation_rows.extend(peak_preservation(
                case, root, audit["units"], audit["ambient"], audit["charge"],
                audit["measurement_scaled_distances_m_kg13"], histories=histories,
            ))
        _write_csv(out_dir / "blast_metrics.csv", metrics_rows)
        _write_csv(out_dir / "peak_preservation.csv", preservation_rows)
        _plot_histories(history_map, out_dir)
        _plot_similarity(metrics_rows, out_dir)
        print(f"metrics: {len(metrics_rows)} rows")

    if args.stage in ("consistency", "all"):
        for case_id in focus:
            case = _case_config(audit, case_id)
            consistency.append(rh_audit(case, root, audit["ambient"]))
            compatibility.append(constraint_compatibility(case, root, audit["ambient"]))
            case_data = _case_data(audit, case_id)
            consistency[-1]["implementation"] = implementation_checks(case_data)
        _write_json(out_dir / "data_consistency.json", {
            "version": audit["version"],
            "rh": consistency,
            "peak_preservation": preservation_rows,
            "similarity_note": "peak-vs-Z curves use the same fixed scaled distances",
        })
        _write_json(out_dir / "constraint_compatibility.json", {
            "version": audit["version"],
            "cases": compatibility,
        })
        print(f"consistency: {len(consistency)} cases")

    if args.stage in ("report", "all"):
        def read_csv(path: Path) -> list[dict]:
            if not path.is_file():
                return []
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

        if not metrics_rows:
            metrics_rows = read_csv(out_dir / "blast_metrics.csv")
        if not import_rows:
            import_rows = read_csv(out_dir / "raw_import_comparison.csv")
        if not preservation_rows:
            preservation_rows = read_csv(out_dir / "peak_preservation.csv")
        if manifest_payload is None and (out_dir / "manifest.json").is_file():
            manifest_payload = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
        _report(audit, root, out_dir, manifest_payload, metrics_rows,
                import_rows, preservation_rows, consistency, compatibility)
        print(f"report: {out_dir / 'report.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
