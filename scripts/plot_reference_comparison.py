"""Compare PINN and the data-only baseline against ALE, Henrych and KG.

Three outputs:

1. peak-overpressure curves for R50 and R100 zero-shot;
2. |model - reference| curves for ALE / Henrych / Kinney-Graham;
3. a bar summary of absolute RMSE and relative RMS.

The data-only baseline is the matched one (576 air labels, the same labelled
budget as the PINN run).  R100 is trained on R50 and evaluated zero-shot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

PINN = "#2a78d6"
DATA = "#eb6834"
HENRYCH = "#1baf7a"
KG = "#6250d6"
ALE = "#0b0b0b"

CASE_PATHS = {
    "R50": {
        "pinn": "outputs/empirical_comparison_r50/peak_overpressure_vs_Z.csv",
        "data": "outputs/empirical_comparison_r50_data_only_matched/peak_overpressure_vs_Z.csv",
    },
    "R100 zero-shot": {
        "pinn": "outputs/empirical_comparison_r100_zero_shot/peak_overpressure_vs_Z.csv",
        "data": "outputs/empirical_comparison_r100_data_only_matched/peak_overpressure_vs_Z.csv",
    },
}
REFERENCES = ("ALE", "Henrych", "Kinney–Graham")
REF_COLUMNS = {
    "ALE": "dp_data_Pa",
    "Henrych": "dp_Henrych_Pa",
    "Kinney–Graham": "dp_KinneyGraham_Pa",
}


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in (
            "Z_m_kg13", "dp_data_Pa", "dp_model_Pa",
            "dp_Henrych_Pa", "dp_KinneyGraham_Pa",
        )
    }


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    error = pred - ref
    relative = error / ref
    return {
        "rmse_MPa": float(np.sqrt(np.mean(error ** 2)) / 1.0e6),
        "mae_MPa": float(np.mean(np.abs(error)) / 1.0e6),
        "rel_rms_pct": float(np.sqrt(np.mean(relative ** 2)) * 100.0),
        "bias_MPa": float(np.mean(error) / 1.0e6),
        "max_abs_MPa": float(np.max(np.abs(error)) / 1.0e6),
    }


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)


def _x_ticks(ax) -> None:
    ax.set_xscale("log")
    ticks = [0.12, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.113, 1.37)


def _curves(data, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    for ax, (case, values) in zip(axes, data.items()):
        z = values["pinn"]["Z_m_kg13"]
        ale = values["pinn"]["dp_data_Pa"] / 1.0e6
        henrych = values["pinn"]["dp_Henrych_Pa"] / 1.0e6
        kg = values["pinn"]["dp_KinneyGraham_Pa"] / 1.0e6
        pinn = values["pinn"]["dp_model_Pa"] / 1.0e6
        datann = values["data"]["dp_model_Pa"] / 1.0e6

        ax.loglog(
            z, ale, color=ALE, linewidth=1.4, marker="o", markersize=3.4,
            markerfacecolor="none", markevery=4, label="ALE 数据",
        )
        ax.loglog(
            z, henrych, color=HENRYCH, linewidth=1.6, linestyle="--",
            label="Henrych（1979）",
        )
        ax.loglog(
            z, kg, color=KG, linewidth=1.6, linestyle="-.",
            label="Kinney–Graham（1985）",
        )
        ax.loglog(
            z, pinn, color=PINN, linewidth=2.2, label="PINN",
        )
        ax.loglog(
            z, datann, color=DATA, linewidth=2.2, label="DataNN（576 标签）",
        )
        _x_ticks(ax)
        ax.set_xlabel(r"比例距离 $Z=R_s/W^{1/3}$  (m·kg$^{-1/3}$)", color=SECONDARY)
        ax.set_title(case, loc="left", fontsize=12, color=INK, pad=2)
        _style_axis(ax)
    axes[0].set_ylabel(r"峰值超压 $\Delta p$ (MPa)", color=SECONDARY)
    axes[1].tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", ncol=5, frameon=True,
        framealpha=0.95, edgecolor=GRID, fontsize=8.8,
        bbox_to_anchor=(0.5, 0.905),
    )
    legend.get_frame().set_facecolor(SURFACE)
    fig.text(
        0.04, 0.985,
        "峰值超压–比例距离：PINN 与普通网络对照 ALE / Henrych / Kinney–Graham",
        ha="left", va="top", fontsize=14.5, color=INK,
    )
    fig.text(
        0.04, 0.945,
        "左：R50 内插；右：R100 零样本。DataNN 与 PINN 使用相同的 576 个空气标签。",
        ha="left", va="top", fontsize=9.2, color=SECONDARY,
    )
    fig.subplots_adjust(top=0.77, bottom=0.16, left=0.07, right=0.98, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"written: {output}")
    plt.close(fig)


def _absolute_errors(data, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.4), sharex=True, sharey=True)
    for row, (case, values) in enumerate(data.items()):
        z = values["pinn"]["Z_m_kg13"]
        for col, ref_name in enumerate(REFERENCES):
            ax = axes[row, col]
            ref = values["pinn"][REF_COLUMNS[ref_name]] / 1.0e6
            for model_name, key, color in (
                ("PINN", "pinn", PINN),
                ("DataNN", "data", DATA),
            ):
                pred = values[key]["dp_model_Pa"] / 1.0e6
                error = np.abs(pred - ref)
                ax.semilogy(
                    z, error, color=color, linewidth=1.9,
                    marker="o", markersize=2.4, markevery=6,
                    markerfacecolor=color, markeredgewidth=0.0,
                    label=model_name if row == 0 and col == 0 else None,
                )
            _x_ticks(ax)
            ax.set_ylim(5.0e-4, 30.0)
            ax.set_title(
                f"{case}：与 {ref_name} 的绝对误差",
                loc="left", fontsize=10.5, color=INK, pad=6,
            )
            _style_axis(ax)
            if row == 1:
                ax.set_xlabel(r"$Z$ (m·kg$^{-1/3}$)", color=SECONDARY)
            if col == 0:
                ax.set_ylabel("|Δp_model − Δp_ref| (MPa)", color=SECONDARY)

    axes[0, 0].legend(
        loc="upper right", frameon=True, framealpha=0.95,
        edgecolor=GRID, fontsize=8.8,
    )
    fig.text(
        0.04, 0.995,
        "PINN 与普通网络相对 ALE / Henrych / Kinney–Graham 的绝对误差",
        ha="left", va="top", fontsize=14.5, color=INK,
    )
    fig.text(
        0.04, 0.955,
        "纵轴为对数轴，越低越贴近参考；上排 R50，下排 R100 零样本。",
        ha="left", va="top", fontsize=9.0, color=SECONDARY,
    )
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.075, right=0.985,
                        hspace=0.34, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"written: {output}")
    plt.close(fig)


def _summary_bars(metrics, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    colors = {"PINN": PINN, "DataNN": DATA}
    for row, case in enumerate(metrics):
        for col, (metric, fmt, ylabel) in enumerate((
            ("rmse_MPa", "{:.3f}", "绝对 RMSE (MPa)"),
            ("rel_rms_pct", "{:.1f}%", "相对 RMS (%)"),
        )):
            ax = axes[row, col]
            x = np.arange(len(REFERENCES), dtype=float)
            width = 0.34
            for offset, model in ((-width / 2, "PINN"), (width / 2, "DataNN")):
                values = [metrics[case][model][ref][metric] for ref in REFERENCES]
                bars = ax.bar(
                    x + offset, values, width=width, color=colors[model],
                    edgecolor="none", label=model, zorder=2,
                )
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        value + 0.03 * max(values),
                        fmt.format(value), ha="center", va="bottom",
                        fontsize=8.4, color=SECONDARY,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(REFERENCES)
            ax.set_title(
                f"{case}：{ylabel}", loc="left", fontsize=11.0,
                color=INK, pad=8,
            )
            ax.set_ylabel(ylabel, color=SECONDARY)
            ax.set_ylim(0.0, 1.22 * max(
                metrics[case][m][r][metric]
                for m in ("PINN", "DataNN") for r in REFERENCES
            ))
            _style_axis(ax)
            ax.grid(False, axis="x")
            ax.grid(True, axis="y", color=GRID, linewidth=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=True,
        framealpha=0.95, edgecolor=GRID, fontsize=9.5,
        bbox_to_anchor=(0.5, 0.965),
    )
    legend.get_frame().set_facecolor(SURFACE)
    fig.suptitle(
        "PINN 与普通网络对各参考量的整体误差",
        x=0.04, y=0.995, ha="left", fontsize=14.5, color=INK,
    )
    fig.text(
        0.04, 0.95,
        "每格越低越好；左列绝对误差，右列相对误差。",
        ha="left", va="top", fontsize=9.0, color=SECONDARY,
    )
    fig.subplots_adjust(top=0.85, bottom=0.09, left=0.09, right=0.98,
                        hspace=0.42, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"written: {output}")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", default="outputs/model_comparison/reference_figures"
    )
    args = parser.parse_args(argv)

    data = {
        case: {
            "pinn": _read(Path(paths["pinn"])),
            "data": _read(Path(paths["data"])),
        }
        for case, paths in CASE_PATHS.items()
    }

    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    rows: list[dict[str, object]] = []
    for case, values in data.items():
        metrics[case] = {"PINN": {}, "DataNN": {}}
        for model_name, key in (("PINN", "pinn"), ("DataNN", "data")):
            pred = values[key]["dp_model_Pa"]
            for ref_name in REFERENCES:
                ref = values[key][REF_COLUMNS[ref_name]]
                values_metrics = _metrics(pred, ref)
                metrics[case][model_name][ref_name] = values_metrics
                rows.append({
                    "case": case, "model": model_name, "reference": ref_name,
                    **values_metrics,
                })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / "model_reference_metrics.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {table}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "figure.facecolor": SURFACE,
    })

    _curves(data, output_dir / "reference_peak_curves.png")
    _absolute_errors(data, output_dir / "reference_absolute_errors.png")
    _summary_bars(metrics, output_dir / "reference_error_summary.png")

    print("\nOverall metrics (RMSE / relative RMS):")
    for case in metrics:
        for ref in REFERENCES:
            print(
                f"{case:16s} {ref:14s} "
                f"PINN {metrics[case]['PINN'][ref]['rmse_MPa']:.3f} MPa / "
                f"{metrics[case]['PINN'][ref]['rel_rms_pct']:.2f}%   "
                f"DataNN {metrics[case]['DataNN'][ref]['rmse_MPa']:.3f} MPa / "
                f"{metrics[case]['DataNN'][ref]['rel_rms_pct']:.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
