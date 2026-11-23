"""Split the empirical peak-overpressure comparison into three readable charts.

The combined figure in ``compare_empirical.py`` puts four absolute curves on one
log-log panel and three relative-deviation curves on a second panel.  This
script keeps the same data but makes three separate deviation charts:

1. PINN against Henrych and Kinney-Graham
2. ALE data against Henrych and Kinney-Graham
3. PINN against the ALE data

Every y value is a signed relative deviation of *peak incident overpressure*
(``dp = p_peak - p_ambient``):

    (a - b) / b * 100 %

The denominator is always the reference named second in the label:

    PINN vs Henrych       -> (PINN - Henrych) / Henrych
    PINN vs Kinney-Graham -> (PINN - Kinney-Graham) / Kinney-Graham
    ALE vs Henrych        -> (ALE - Henrych) / Henrych
    ALE vs Kinney-Graham  -> (ALE - Kinney-Graham) / Kinney-Graham
    PINN vs ALE           -> (PINN - ALE) / ALE

Negative means the first quantity is lower than the reference.  The script also
writes one table with all five deviation columns, so the chart values are not
only readable through color.

Input is the CSV written by ``compare_empirical.py``:

    outputs/empirical_comparison_r50/peak_overpressure_vs_Z.csv
    outputs/empirical_comparison_r100_zero_shot/peak_overpressure_vs_Z.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Validated categorical slots from the dataviz reference palette (light mode).
# Slots 1 and 2 pass the adjacent-pair CVD and normal-vision checks.
BLUE = "#2a78d6"      # Henrych reference
ORANGE = "#eb6834"    # Kinney-Graham reference
VIOLET = "#6250d6"    # single-series PINN vs ALE
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
ZERO = "#52514e"


def _load(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    columns = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in (
            "R_s_m", "Z_m_kg13", "dp_data_Pa", "dp_model_Pa",
            "dp_Henrych_Pa", "dp_KinneyGraham_Pa",
        )
    }
    return columns


def _relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / b * 100.0


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _stats(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    return (
        float(np.sqrt(np.mean(finite ** 2))),
        float(np.max(np.abs(finite))),
    )


def _setup_axes(ax, z: np.ndarray, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(max(0.10, float(z.min()) * 0.92), float(z.max()) * 1.08)
    ax.set_xlabel(r"比例距离 $Z = R_s/W^{1/3}$  (m·kg$^{-1/3}$)", color=SECONDARY)
    ax.set_ylabel(ylabel, color=SECONDARY)
    ticks = [0.12, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    ax.axhline(0.0, color=ZERO, linewidth=1.0, zorder=1)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)


def _plot_two(
    ax, z, curves, *, title, subtitle, footnote, colors, labels, ylabel,
) -> None:
    _setup_axes(ax, z, ylabel)
    for values, color, label in zip(curves, colors, labels):
        mask = _finite_mask(z, values)
        ax.plot(
            z[mask], values[mask], color=color, linewidth=2.0,
            marker="o", markersize=3.0, markevery=5,
            markerfacecolor=color, markeredgewidth=0.0,
            label=label, zorder=3,
        )
    all_values = np.concatenate([v[_finite_mask(z, v)] for v in curves])
    low = min(0.0, float(all_values.min()))
    high = max(0.0, float(all_values.max()))
    pad = 0.08 * max(high - low, 1.0e-12)
    ax.set_ylim(low - pad, high + pad)
    ax.set_title(title, loc="left", fontsize=13.5, color=INK, pad=18)
    ax.text(
        0.0, 1.015, subtitle, transform=ax.transAxes,
        fontsize=9.5, color=SECONDARY, va="bottom",
    )
    ax.text(
        0.0, -0.19, footnote, transform=ax.transAxes,
        fontsize=9.0, color=MUTED, va="top",
    )
    legend = ax.legend(
        loc="best", frameon=True, framealpha=0.95, edgecolor=GRID,
        fontsize=9.2, labelcolor=SECONDARY,
    )
    legend.get_frame().set_facecolor(SURFACE)


def _plot_one(ax, z, values, *, title, subtitle, footnote, label, color, ylabel) -> None:
    _setup_axes(ax, z, ylabel)
    mask = _finite_mask(z, values)
    ax.plot(
        z[mask], values[mask], color=color, linewidth=2.0,
        marker="o", markersize=3.2, markevery=5,
        markerfacecolor=color, markeredgewidth=0.0,
        zorder=3,
    )
    finite = values[mask]
    low = min(0.0, float(finite.min()))
    high = max(0.0, float(finite.max()))
    pad = 0.08 * max(high - low, 1.0e-12)
    ax.set_ylim(low - pad, high + pad)
    ax.set_title(title, loc="left", fontsize=13.5, color=INK, pad=18)
    ax.text(
        0.0, 1.015, subtitle, transform=ax.transAxes,
        fontsize=9.5, color=SECONDARY, va="bottom",
    )
    ax.text(
        0.0, -0.19, footnote, transform=ax.transAxes,
        fontsize=9.0, color=MUTED, va="top",
    )
    ax.text(
        0.98, 0.96, label, transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, color=SECONDARY,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                  edgecolor=GRID, alpha=0.95),
    )


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"written: {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, help="peak_overpressure_vs_Z.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="R50", help="case label, e.g. R50 or R100 零样本")
    parser.add_argument(
        "--quantity", choices=("relative", "absolute"), default="relative",
        help=(
            "relative: (a-b)/b*100 [%%]; absolute: signed a-b [MPa]. "
            "Both compare peak overpressure, not the full pressure field."
        ),
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load(csv_path)
    z = data["Z_m_kg13"]
    model = data["dp_model_Pa"]
    ale = data["dp_data_Pa"]
    henrych = data["dp_Henrych_Pa"]
    kg = data["dp_KinneyGraham_Pa"]

    if args.quantity == "relative":
        convert = _relative
        unit = "%"
        title_word = "相对偏差"
        ylabel = "相对偏差 (%)"
        prefix = "deviation"
        column_suffix = "_pct"
        def number(value: float) -> str:
            return f"{value:.1f}%"
    else:
        def convert(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (a - b) / 1.0e6
        unit = "MPa"
        title_word = "绝对误差"
        ylabel = "峰值超压之差 (MPa)"
        prefix = "error"
        column_suffix = "_MPa"
        def number(value: float) -> str:
            return f"{value:.3f} MPa"

    pinn_h = convert(model, henrych)
    pinn_kg = convert(model, kg)
    ale_h = convert(ale, henrych)
    ale_kg = convert(ale, kg)
    pinn_ale = convert(model, ale)

    rms_ph, max_ph = _stats(pinn_h)
    rms_pk, max_pk = _stats(pinn_kg)
    rms_ah, max_ah = _stats(ale_h)
    rms_ak, max_ak = _stats(ale_kg)
    rms_pa, max_pa = _stats(pinn_ale)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": SURFACE,
    })

    common = (
        r"$\Delta p=p_{\mathrm{peak}}-p_{\mathrm{ambient}}$；"
        "负值表示前者低于后者"
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    _plot_two(
        ax, z, [pinn_h, pinn_kg],
        title=f"{args.label}：PINN 与经验公式的峰值超压{title_word}",
        subtitle=f"参考值分别为 Henrych 与 Kinney–Graham；纵轴为有符号{title_word}",
        footnote=(
            f"PINN vs Henrych: RMS={number(rms_ph)}, max={number(max_ph)}    "
            f"PINN vs Kinney–Graham: RMS={number(rms_pk)}, max={number(max_pk)}    "
            + common
        ),
        colors=[BLUE, ORANGE],
        labels=[
            f"PINN − Henrych（RMS {number(rms_ph)}）",
            f"PINN − Kinney–Graham（RMS {number(rms_pk)}）",
        ],
        ylabel=ylabel,
    )
    _save(fig, output_dir / f"{prefix}_pinn_vs_empirical.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    _plot_two(
        ax, z, [ale_h, ale_kg],
        title=f"{args.label}：ALE 数据与经验公式的峰值超压{title_word}",
        subtitle=f"参考值分别为 Henrych 与 Kinney–Graham；纵轴为有符号{title_word}",
        footnote=(
            f"ALE vs Henrych: RMS={number(rms_ah)}, max={number(max_ah)}    "
            f"ALE vs Kinney–Graham: RMS={number(rms_ak)}, max={number(max_ak)}    "
            + common
        ),
        colors=[BLUE, ORANGE],
        labels=[
            f"ALE − Henrych（RMS {number(rms_ah)}）",
            f"ALE − Kinney–Graham（RMS {number(rms_ak)}）",
        ],
        ylabel=ylabel,
    )
    _save(fig, output_dir / f"{prefix}_ale_vs_empirical.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    _plot_one(
        ax, z, pinn_ale,
        title=f"{args.label}：PINN 与 ALE 数据的峰值超压{title_word}",
        subtitle=f"参考值为 ALE 激波后峰值超压；纵轴为有符号{title_word}",
        footnote=(
            f"PINN vs ALE: RMS={number(rms_pa)}, max={number(max_pa)}    " + common
        ),
        label=f"PINN − ALE   RMS {number(rms_pa)}",
        color=VIOLET,
        ylabel=ylabel,
    )
    _save(fig, output_dir / f"{prefix}_pinn_vs_ale.png")
    plt.close(fig)

    table_path = output_dir / (
        "relative_deviation_curves.csv" if args.quantity == "relative"
        else "absolute_error_curves.csv"
    )
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Z_m_kg13", "R_s_m",
            f"PINN_vs_Henrych{column_suffix}",
            f"PINN_vs_KinneyGraham{column_suffix}",
            f"ALE_vs_Henrych{column_suffix}",
            f"ALE_vs_KinneyGraham{column_suffix}",
            f"PINN_vs_ALE{column_suffix}",
            "dp_data_Pa", "dp_model_Pa",
            "dp_Henrych_Pa", "dp_KinneyGraham_Pa",
        ])
        for index in range(z.size):
            writer.writerow([
                f"{value:.12g}" for value in (
                    z[index],
                    data["R_s_m"][index],
                    pinn_h[index], pinn_kg[index], ale_h[index], ale_kg[index],
                    pinn_ale[index],
                    ale[index], model[index], henrych[index], kg[index],
                )
            ])
    print(f"written: {table_path}")

    print(
        f"RMS ({unit}): PINN-H {rms_ph:.3f}  PINN-KG {rms_pk:.3f}  "
        f"ALE-H {rms_ah:.3f}  ALE-KG {rms_ak:.3f}  PINN-ALE {rms_pa:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
