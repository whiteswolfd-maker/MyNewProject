"""Plot the PINN vs data-only comparison from summary.csv.

Three panels:

1. peak-overpressure relative RMS [%]
2. peak-overpressure RMSE [MPa]
3. normalized PDE residual [log scale]

The first two show data accuracy; the third shows physical consistency.  Lower
is better in all three.  Direct value labels are present on every bar so the
plot does not rely on color alone.
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
DATA576 = "#eb6834"
DATA2112 = "#1baf7a"

MODELS = ("PINN", "DataNN-576", "DataNN-2112")
COLORS = {"PINN": PINN, "DataNN-576": DATA576, "DataNN-2112": DATA2112}
LABELS = {
    "PINN": "PINN（物理约束）",
    "DataNN-576": "DataNN（576 标签）",
    "DataNN-2112": "DataNN（2112 标签）",
}


def _read(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["case"], row["model"]): {
        key: float(value) for key, value in row.items()
        if key not in ("case", "model")
    } for row in rows}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)


def _bar_panel(ax, data, cases, metric, title, ylabel, fmt, *, log=False):
    x = np.arange(len(cases), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    values_all = []
    for offset, model in zip(offsets, MODELS):
        values = np.asarray([
            data[(case, model)][metric] for case in cases
        ], dtype=float)
        values_all.append(values)
        bottom = 1.0e-7 if log else 0.0
        bars = ax.bar(
            x + offset, values - bottom if log else values, width=width,
            bottom=bottom, color=COLORS[model], edgecolor="none",
            label=LABELS[model], zorder=2,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * (1.12 if log else 1.0) if log else value + 0.02 * max(values_all[0].max(), 1.0e-12),
                fmt(value),
                ha="center", va="bottom" if not log else "bottom",
                fontsize=8.3, color=SECONDARY,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=SECONDARY)
    ax.text(
        1.0, 1.02, "越低越好", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5, color=MUTED,
    )
    _style(ax)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1.0e-4, 0.08)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary", default="outputs/model_comparison/summary.csv"
    )
    parser.add_argument(
        "--output", default="outputs/model_comparison/pinn_vs_data_only_summary.png"
    )
    args = parser.parse_args(argv)

    data = _read(Path(args.summary))
    cases = ["R50", "R100 zero-shot"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 10.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": SURFACE,
    })

    fig = plt.figure(figsize=(10.0, 8.2))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(1.0, 1.05), hspace=0.42, wspace=0.28,
        top=0.80, bottom=0.10, left=0.08, right=0.97,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    _bar_panel(
        ax1, data, cases, "peak_rel_rms_pct",
        "峰值超压相对 RMS（%）", "相对 RMS (%)",
        lambda value: f"{value:.1f}%",
    )
    ax1.set_ylim(0.0, 18.0)

    _bar_panel(
        ax2, data, cases, "peak_rmse_MPa",
        "峰值超压 RMSE（MPa）", "RMSE (MPa)",
        lambda value: f"{value:.3f}",
    )
    ax2.set_ylim(0.0, 1.15)

    _bar_panel(
        ax3, data, cases, "pde_total",
        "归一化 PDE 残差（越低越接近 Euler 方程）", "PDE total（对数轴）",
        lambda value: f"{value:.1e}", log=True,
    )

    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=True,
        framealpha=0.95, edgecolor=GRID, fontsize=9.5,
        bbox_to_anchor=(0.5, 0.895),
    )
    legend.get_frame().set_facecolor(SURFACE)

    fig.text(
        0.06, 0.975,
        "PINN 与普通数据网络：峰值超压与物理一致性对比",
        ha="left", va="top", fontsize=15, color=INK,
    )
    fig.text(
        0.06, 0.938,
        "R50 训练；R100 为零样本。DataNN-576 与 PINN 使用相同的 576 个空气标签；"
        "DataNN-2112 使用 2112 个标签，是更强的数据基线。",
        ha="left", va="top", fontsize=9.2, color=SECONDARY,
    )
    fig.text(
        0.06, 0.025,
        "前两栏是数据精度，第三栏是物理一致性。普通网络拟合数据更好，但 PDE 残差明显更大。",
        ha="left", va="bottom", fontsize=8.8, color=MUTED,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
