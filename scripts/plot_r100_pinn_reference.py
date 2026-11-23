"""Plot the R100 zero-shot PINN peak overpressure against ALE and formulas.

The model was trained on R50 only; R100 was never used for training,
validation, early stopping or hyperparameter selection.
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
ALE = "#0b0b0b"
HENRYCH = "#1baf7a"
KG = "#6250d6"


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


def _metrics(pred, ref):
    error = pred - ref
    return (
        float(np.sqrt(np.mean(error ** 2))),
        float(np.sqrt(np.mean((error / ref) ** 2)) * 100.0),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        default=(
            "outputs/empirical_comparison_r100_pinn_20261123/"
            "peak_overpressure_vs_Z.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/empirical_comparison_r100_pinn_20261123/"
            "r100_pinn_vs_references.png"
        ),
    )
    args = parser.parse_args(argv)

    data = _read(Path(args.csv))
    z = data["Z_m_kg13"]
    ale = data["dp_data_Pa"] / 1.0e6
    pinn = data["dp_model_Pa"] / 1.0e6
    henrych = data["dp_Henrych_Pa"] / 1.0e6
    kg = data["dp_KinneyGraham_Pa"] / 1.0e6

    rms_a, rel_a = _metrics(pinn, ale)
    rms_h, rel_h = _metrics(pinn, henrych)
    rms_k, rel_k = _metrics(pinn, kg)

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

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.9),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    ax.loglog(
        z, ale, color=ALE, linewidth=1.4, marker="o", markersize=3.4,
        markerfacecolor="none", markevery=4, label="ALE 数据",
    )
    ax.loglog(z, henrych, color=HENRYCH, linewidth=1.7, linestyle="--",
              label="Henrych（1979）")
    ax.loglog(z, kg, color=KG, linewidth=1.7, linestyle="-.",
              label="Kinney–Graham（1985）")
    ax.loglog(z, pinn, color=PINN, linewidth=2.4, label="PINN（R100 零样本）")
    ax.set_xscale("log")
    ticks = [0.12, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.113, 1.37)
    ax.set_xlabel(r"比例距离 $Z=R_s/W^{1/3}$  (m·kg$^{-1/3}$)", color=SECONDARY)
    ax.set_ylabel(r"峰值超压 $\Delta p$ (MPa)", color=SECONDARY)
    ax.set_title("峰值超压–比例距离", loc="left", fontsize=12, color=INK, pad=8)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95,
              edgecolor=GRID, fontsize=8.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)

    for values, color, label, rms, rel in (
        (np.abs(pinn - ale), ALE, "PINN − ALE", rms_a, rel_a),
        (np.abs(pinn - henrych), HENRYCH, "PINN − Henrych", rms_h, rel_h),
        (np.abs(pinn - kg), KG, "PINN − Kinney–Graham", rms_k, rel_k),
    ):
        ax2.semilogy(
            z, values, color=color, linewidth=1.9,
            marker="o", markersize=2.6, markevery=6,
            markerfacecolor=color, markeredgewidth=0.0,
            label=f"{label}（RMS {rms:.3f} MPa / {rel:.1f}%）",
        )
    ax2.set_xscale("log")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{value:g}" for value in ticks])
    ax2.minorticks_off()
    ax2.set_xlim(0.113, 1.37)
    ax2.set_ylim(5.0e-4, 30.0)
    ax2.set_xlabel(r"比例距离 $Z$ (m·kg$^{-1/3}$)", color=SECONDARY)
    ax2.set_ylabel("|Δp_PINN − Δp_ref| (MPa)", color=SECONDARY)
    ax2.set_title("PINN 与各参考的绝对误差", loc="left", fontsize=12,
                  color=INK, pad=8)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.95,
               edgecolor=GRID, fontsize=8.0)
    ax2.grid(True, which="major", color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax2.spines[spine].set_color(GRID)
    ax2.tick_params(colors=MUTED, labelcolor=SECONDARY)

    fig.text(
        0.04, 0.965,
        "R100 零样本：PINN 与 ALE / Henrych / Kinney–Graham 的峰值超压对比",
        ha="left", va="top", fontsize=14.0, color=INK,
    )
    fig.text(
        0.04, 0.925,
        "PINN 只用 R50 训练；R100 未参与训练、验证或调参。"
        "右图纵轴为对数轴，越低越贴近参考。",
        ha="left", va="top", fontsize=9.0, color=SECONDARY,
    )
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.075, right=0.98,
                        wspace=0.28)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"written: {output}")
    print(
        f"PINN vs ALE: {rms_a:.4f} MPa / {rel_a:.2f}%\n"
        f"PINN vs Henrych: {rms_h:.4f} MPa / {rel_h:.2f}%\n"
        f"PINN vs Kinney-Graham: {rms_k:.4f} MPa / {rel_k:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
