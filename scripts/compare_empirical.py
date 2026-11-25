"""Compare the R50 PINN shock-peak overpressure with free-air TNT formulas.

What is compared
----------------
Henrych and Kinney-Graham are *peak incident overpressure* correlations for a
free-air TNT burst.  They are functions of the scaled distance

    Z = R_s / W^(1/3),        W = TNT-equivalent mass [kg]

They do not provide the full pressure profile.  Therefore this script compares
one point per shock position/time:

    dp_data(t)  = p_inner(t) - p_amb        (ALE post-shock state)
    dp_model(t) = p(t, zeta=1) - p_amb      (PINN at the shock boundary)
    dp_emp(Z)   = Henrych(Z) or Kinney-Graham(Z)

The top panel is the paper-style log-log peak-overpressure curve.  The bottom
panel shows relative gaps so that model error and data/formula discrepancy are
not confused.

Formulas, units and validity
----------------------------
Henrych (1979), *The Dynamics of Explosion and Its Use*, free-air TNT,
Z in m/kg^(1/3), overpressure in MPa:

    0.05 <= Z <= 0.3:
        dp = 1.40717/Z + 0.55397/Z^2 - 0.03572/Z^3 + 0.000625/Z^4
    0.3 < Z <= 1:
        dp = 0.61938/Z - 0.03262/Z^2 + 0.21324/Z^3
    1 < Z <= 10:
        dp = 0.0662/Z + 0.405/Z^2 + 0.3288/Z^3

Kinney & Graham (1985) closed-form fit, often called "Kinney-Bulmash" in the
blast literature (the true Kingery-Bulmash curve is a piecewise log-polynomial
fit).  P0 is ambient pressure; the result has the same pressure unit as P0:

    dp = P0 * 808 * (1 + (Z/4.5)^2)
         / [ sqrt(1 + (Z/0.048)^2)
             * sqrt(1 + (Z/0.32)^2)
             * sqrt(1 + (Z/1.35)^2) ]

Here P0 = 101325 Pa, so dp is in Pa.

Both are free-air (spherical) burst correlations.  Do not apply the
surface-burst variants to this 1D spherical case without changing the formulas
and the TNT-equivalent assumption.  Points outside a formula's Z range are
reported as NaN, not extrapolated.

Data definitions
----------------
- `dp_data` is the maximum ALE pressure in a narrow window just behind the
  detected shock (default: 5 cells inside the shock, pure air).  This is the
  post-shock peak, not the contact/JWL pressure and not the single-cell
  `p_inner` state.  If the standard profile is unavailable or a frame has no
  valid window point, the value is recorded as missing; it is never replaced
  by `p_inner_pa` or another metric.
- `dp_model` uses the PINN pressure at zeta=1, i.e. the shock boundary.  The
  model's right boundary is constrained by the finite-Mach RH state from the
  measured D_s, so this comparison is "model given measured shock trajectory",
  not an autonomous shock prediction.

The script writes:
    peak_overpressure_vs_Z.csv
    peak_overpressure_vs_Z.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spherical_blast_pinn.contact_air.data import load_case  # noqa: E402


P_AMB = 101325.0


def henrych_dp_pa(z: np.ndarray) -> np.ndarray:
    """Henrych (1979) free-air TNT peak overpressure, returned in Pa."""
    z = np.asarray(z, dtype=np.float64)
    out = np.full_like(z, np.nan)
    m1 = (z >= 0.05) & (z <= 0.3)
    m2 = (z > 0.3) & (z <= 1.0)
    m3 = (z > 1.0) & (z <= 10.0)
    out[m1] = (
        1.40717 / z[m1]
        + 0.55397 / z[m1] ** 2
        - 0.03572 / z[m1] ** 3
        + 0.000625 / z[m1] ** 4
    )
    out[m2] = (
        0.61938 / z[m2]
        - 0.03262 / z[m2] ** 2
        + 0.21324 / z[m2] ** 3
    )
    out[m3] = (
        0.0662 / z[m3]
        + 0.405 / z[m3] ** 2
        + 0.3288 / z[m3] ** 3
    )
    return out * 1.0e6


def kinney_graham_dp_pa(z: np.ndarray, p0: float = P_AMB) -> np.ndarray:
    """Kinney & Graham (1985) free-air TNT peak overpressure, returned in Pa."""
    z = np.asarray(z, dtype=np.float64)
    numerator = 808.0 * (1.0 + (z / 4.5) ** 2)
    denominator = (
        np.sqrt(1.0 + (z / 0.048) ** 2)
        * np.sqrt(1.0 + (z / 0.32) ** 2)
        * np.sqrt(1.0 + (z / 1.35) ** 2)
    )
    return p0 * numerator / denominator


def _relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / b


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prediction",
        default="outputs/single_r50_bounded_cheb_logp_6600/pred_r50_full.npz",
    )
    parser.add_argument(
        "--cache",
        default="outputs/fullframes/cache/r50_full.npz",
    )
    parser.add_argument(
        "--standard",
        default="outputs/fullframes/standard/r50_full.npz",
        help="standard 1-D profile used to extract the ALE post-shock peak",
    )
    parser.add_argument(
        "--peak-cells", type=float, default=5.0,
        help="number of cells inside the shock used for the ALE peak",
    )
    parser.add_argument(
        "--peak-stride", type=int, default=10,
        help=(
            "use every Nth frame for the ALE peak (the standard npz is 3.2 GB; "
            "stride=1 reads all 983 frames and is slow)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/empirical_comparison_r50",
    )
    parser.add_argument("--ambient", type=float, default=P_AMB)
    parser.add_argument(
        "--model-zeta-index", type=int, default=-1,
        help="zeta index used as the model shock peak (default: last, zeta=1)",
    )
    args = parser.parse_args(argv)

    prediction = np.load(args.prediction)
    times = np.asarray(prediction["times"], dtype=np.float64)
    zeta = np.asarray(prediction["zeta"], dtype=np.float64)
    pressure = np.asarray(prediction["pressure"], dtype=np.float64)
    shock_radius_pred = np.asarray(prediction["shock_radius"], dtype=np.float64)[:, 0]

    case = load_case(Path(args.cache), role="train")
    r0 = float(case.charge_radius_m)
    rho_tnt = float(case.precursor["jwl"]["rho_tnt"])
    w_tnt = (4.0 / 3.0) * math.pi * r0 ** 3 * rho_tnt
    w13 = w_tnt ** (1.0 / 3.0)

    frame_indices = np.asarray(case.contact.indices(), dtype=np.int64)
    frame_times = np.asarray(case.contact.times[frame_indices], dtype=np.float64)
    keep = (
        (frame_times >= case.t0 - 1.0e-15)
        & (frame_times <= case.t_end + 1.0e-15)
        & np.isfinite(np.asarray(case.contact.low[frame_indices], dtype=np.float64))
        & np.isfinite(np.asarray(case.contact.up[frame_indices], dtype=np.float64))
    )
    frame_indices = frame_indices[keep]
    frame_times = frame_times[keep]
    stride = max(1, int(args.peak_stride))
    if stride > 1:
        frame_indices = frame_indices[::stride]
        frame_times = frame_times[::stride]

    standard_path = Path(args.standard)
    standard = None
    standard_offsets = None
    if standard_path.is_file():
        standard = np.load(standard_path, mmap_mode="r")
        standard_offsets = np.asarray(standard["frame_offsets"], dtype=np.int64)
        print(f"ALE peak source: {standard_path} ({args.peak_cells:g} cells inside)")
    else:
        # Do not silently substitute a different metric.  A missing standard
        # profile means the shock-window peak is unavailable for every frame.
        print(
            f"error: standard profile {standard_path} not found; "
            "the shock-window peak is unavailable and will be recorded as "
            "missing, not replaced by the single-cell p_inner state"
        )

    cell = r0 / 500.0
    p_data = np.full(frame_times.size, np.nan, dtype=np.float64)
    rs_data = np.full(frame_times.size, np.nan, dtype=np.float64)
    n_window_missing = 0
    for j, index in enumerate(frame_indices):
        index = int(index)
        shock = case.shock.radii[index]
        if not np.isfinite(shock):
            continue
        rs_data[j] = float(shock)
        if standard is None:
            continue
        start = int(standard_offsets[index])
        end = int(standard_offsets[index + 1])
        standard_radius = np.asarray(
            standard["radius"][start:end], dtype=np.float64
        )
        standard_pressure = np.asarray(
            standard["pressure"][start:end], dtype=np.float64
        )
        standard_alpha_air = np.asarray(
            standard["alpha_air"][start:end], dtype=np.float64
        )
        standard_valid_pressure = (
            np.asarray(standard["valid_pressure"][start:end]).astype(bool)
            if "valid_pressure" in standard.files
            else np.ones_like(standard_pressure, dtype=bool)
        )
        window = (
            (standard_radius >= shock - float(args.peak_cells) * cell)
            & (standard_radius <= shock + 0.25 * cell)
            & (standard_alpha_air >= 0.95)
            & standard_valid_pressure
            & np.isfinite(standard_pressure)
        )
        if np.any(window):
            p_data[j] = float(np.max(standard_pressure[window]))
        else:
            n_window_missing += 1
    if n_window_missing:
        print(
            f"warning: {n_window_missing}/{frame_times.size} frames have no "
            "valid shock-window peak; they are recorded as missing, not "
            "replaced by p_inner"
        )

    model_shock = pressure[:, int(args.model_zeta_index)]
    in_model_range = (frame_times >= times[0]) & (frame_times <= times[-1])
    p_model_at_frames = np.full(frame_times.size, np.nan, dtype=np.float64)
    p_model_at_frames[in_model_range] = np.interp(
        frame_times[in_model_range], times, model_shock
    )
    if np.any(~in_model_range):
        print(
            f"warning: {int((~in_model_range).sum())} frames lie outside the "
            "prediction time range; they are excluded, not extrapolated"
        )

    good = (
        np.isfinite(p_data)
        & np.isfinite(p_model_at_frames)
        & np.isfinite(rs_data)
        & (rs_data > 0.0)
    )
    frame_times = frame_times[good]
    p_data = p_data[good]
    p_model = p_model_at_frames[good]
    rs_data = rs_data[good]

    z = rs_data / w13
    dp_data = p_data - float(args.ambient)
    dp_model = p_model - float(args.ambient)
    dp_henrych = henrych_dp_pa(z)
    dp_kg = kinney_graham_dp_pa(z, p0=float(args.ambient))

    order = np.argsort(z)
    frame_times = frame_times[order]
    rs_data = rs_data[order]
    z = z[order]
    dp_data = dp_data[order]
    dp_model = dp_model[order]
    dp_henrych = dp_henrych[order]
    dp_kg = dp_kg[order]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "peak_overpressure_vs_Z.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_s", "R_s_m", "Z_m_kg13",
            "dp_data_Pa", "dp_model_Pa",
            "dp_Henrych_Pa", "dp_KinneyGraham_Pa",
            "model_minus_data_Pa",
            "data_minus_Henrych_Pa", "data_minus_KG_Pa",
            "model_minus_Henrych_Pa", "model_minus_KG_Pa",
        ])
        for row in zip(
            frame_times, rs_data, z,
            dp_data, dp_model, dp_henrych, dp_kg,
            dp_model - dp_data,
            dp_data - dp_henrych, dp_data - dp_kg,
            dp_model - dp_henrych, dp_model - dp_kg,
        ):
            writer.writerow([f"{value:.12g}" for value in row])

    # Summary on the Henrych-valid subset, so the two formulas are not mixed
    # with out-of-range extrapolation.
    valid = np.isfinite(dp_henrych) & np.isfinite(dp_kg)
    def stats(a, b):
        rel = _relative(a[valid], b[valid])
        rel = rel[np.isfinite(rel)]
        if not rel.size:
            return float("nan"), float("nan")
        return float(np.sqrt(np.mean(rel ** 2))), float(np.max(np.abs(rel)))
    print(f"TNT mass W = {w_tnt:.6g} kg, W^(1/3) = {w13:.6g} kg^(1/3)")
    print(f"Z range = {z.min():.4f} .. {z.max():.4f} m/kg^(1/3), n = {z.size}")
    for label, a, b in (
        ("model vs data", dp_model, dp_data),
        ("data vs Henrych", dp_data, dp_henrych),
        ("data vs Kinney-Graham", dp_data, dp_kg),
        ("model vs Henrych", dp_model, dp_henrych),
        ("model vs Kinney-Graham", dp_model, dp_kg),
    ):
        rms, mx = stats(a, b)
        print(f"{label}: rel RMS = {rms:.4%}, rel max = {mx:.4%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )

    marker_step = max(1, z.size // 28)
    ax.loglog(
        z, dp_data / 1.0e6, color="black", marker="o", markersize=3.2,
        markerfacecolor="none", markevery=marker_step, linewidth=1.2,
        label="ALE 数据（激波后峰值）",
    )
    ax.loglog(
        z, dp_model / 1.0e6, color="#d62728", linewidth=2.2, label="PINN 模型",
    )
    ax.loglog(
        z[np.isfinite(dp_henrych)], dp_henrych[np.isfinite(dp_henrych)] / 1.0e6,
        color="#1f77b4", linestyle="--", linewidth=1.8,
        label="Henrych（1979）",
    )
    ax.loglog(
        z, dp_kg / 1.0e6, color="#2ca02c", linestyle="-.", linewidth=1.8,
        label="Kinney–Graham（1985）",
    )
    ax.set_ylabel(r"峰值超压 $\Delta p$ (MPa)")
    ax.set_xlabel(r"比例距离 $Z = R_s/W^{1/3}$ (m·kg$^{-1/3}$)")
    ax.set_xlim(max(0.08, z.min() * 0.92), z.max() * 1.08)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    ax.text(
        0.03, 0.05,
        f"$W={w_tnt:.3f}$ kg TNT\n自由空气爆炸\n"
        f"$Z={z.min():.2f}$–${z.max():.2f}$",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                  edgecolor="0.7"),
    )

    rel_model_data = 100.0 * _relative(dp_model, dp_data)
    rel_data_h = 100.0 * _relative(dp_data, dp_henrych)
    rel_data_kg = 100.0 * _relative(dp_data, dp_kg)
    ax2.axhline(0.0, color="0.4", linewidth=0.9)
    ax2.plot(z, rel_model_data, color="#d62728", linewidth=1.8,
             label="PINN − ALE 数据")
    ax2.plot(z, rel_data_h, color="#1f77b4", linestyle="--", linewidth=1.4,
             label="ALE 数据 − Henrych")
    ax2.plot(z, rel_data_kg, color="#2ca02c", linestyle="-.", linewidth=1.4,
             label="ALE 数据 − Kinney–Graham")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"比例距离 $Z$ (m·kg$^{-1/3}$)")
    ax2.set_ylabel("相对偏差 (%)")
    ax2.legend(loc="best", ncol=1, frameon=True)

    # Keep the bottom panel readable without hiding genuine large deviations.
    finite = np.concatenate([
        rel_model_data[np.isfinite(rel_model_data)],
        rel_data_h[np.isfinite(rel_data_h)],
        rel_data_kg[np.isfinite(rel_data_kg)],
    ])
    if finite.size:
        lo, hi = np.percentile(finite, [2.0, 98.0])
        pad = max(2.0, 0.15 * max(abs(lo), abs(hi)))
        ax2.set_ylim(min(lo - pad, -5.0), max(hi + pad, 5.0))

    fig.tight_layout()
    png_path = output_dir / "peak_overpressure_vs_Z.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"written: {csv_path}")
    print(f"written: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
