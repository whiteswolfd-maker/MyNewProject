"""Compare the PINN and data-only baseline peak-overpressure predictions.

Both CSVs are written by ``compare_empirical.py`` and therefore use the same
ALE frames, the same shock radius and the same post-shock peak definition
(maximum pressure within five pure-air cells behind the shock).  The only
difference is the model column.

Reported metrics compare ``dp_model`` against ``dp_data``:

    RMSE, MAE, bias, relative RMS, max |error|, R^2

in MPa (except relative RMS and R^2).  Metrics are also split into near,
mid and far Z ranges so a model that only wins at one distance is visible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in ("Z_m_kg13", "dp_data_Pa", "dp_model_Pa")
    }


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred - target
    relative = error / target
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": int(target.size),
        "rmse_MPa": float(np.sqrt(np.mean(error ** 2)) / 1.0e6),
        "mae_MPa": float(np.mean(np.abs(error)) / 1.0e6),
        "bias_MPa": float(np.mean(error) / 1.0e6),
        "rel_rms_pct": float(np.sqrt(np.mean(relative ** 2)) * 100.0),
        "max_abs_MPa": float(np.max(np.abs(error)) / 1.0e6),
        "r2": float(1.0 - np.sum(error ** 2) / denominator),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pinn-csv", required=True)
    parser.add_argument("--data-only-csv", required=True)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    pinn = _read(Path(args.pinn_csv))
    data_only = _read(Path(args.data_only_csv))
    if pinn["Z_m_kg13"].shape != data_only["Z_m_kg13"].shape:
        raise ValueError("the two CSVs have different row counts")
    if not np.allclose(pinn["Z_m_kg13"], data_only["Z_m_kg13"], rtol=0.0, atol=1.0e-12):
        raise ValueError("the two CSVs do not use the same Z samples")
    if not np.allclose(pinn["dp_data_Pa"], data_only["dp_data_Pa"], rtol=0.0, atol=1.0e-6):
        raise ValueError("the two CSVs do not use the same ALE peak data")

    z = pinn["Z_m_kg13"]
    target = pinn["dp_data_Pa"]
    bins = (
        ("all", np.ones_like(z, dtype=bool)),
        ("near_Z_lt_0.3", z < 0.3),
        ("mid_0.3_to_1.0", (z >= 0.3) & (z <= 1.0)),
        ("far_Z_gt_1.0", z > 1.0),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_name, values in (
        ("PINN", pinn["dp_model_Pa"]),
        ("Data-only", data_only["dp_model_Pa"]),
    ):
        for bin_name, mask in bins:
            metrics = _metrics(values[mask], target[mask])
            rows.append({"case": args.case_label, "model": model_name,
                         "range": bin_name, **metrics})

    path = output_dir / f"pinn_vs_data_only_{args.case_label}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {path}")

    header = (
        f"{'model':<10} {'range':<16} {'n':>3} {'RMSE':>9} {'MAE':>9} "
        f"{'bias':>9} {'relRMS':>9} {'max':>9} {'R2':>7}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['model']:<10} {row['range']:<16} {row['n']:>3} "
            f"{row['rmse_MPa']:>9.4f} {row['mae_MPa']:>9.4f} "
            f"{row['bias_MPa']:>9.4f} {row['rel_rms_pct']:>8.2f}% "
            f"{row['max_abs_MPa']:>9.4f} {row['r2']:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
