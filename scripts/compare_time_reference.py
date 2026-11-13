"""Compare two prediction sets (same split, different time-input scaling).

Plots the signed contact deviation at the measured frame times, in model time,
for the best checkpoint of each run.  The deviation is
``R_c,pred - measured interval`` in units of ``R_sep``: negative means the
prediction is inside (below R_c,low), positive means beyond R_c,up, zero means
inside the interval.  The interval midpoint is never used.

Both prediction directories must come from the same configuration and the same
case set; only the trained weights differ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import _case_specs, _make_case, _section

from scripts.diagnose_radius_split import _load_prediction


def _deviation(case, prediction: dict[str, np.ndarray]):
    times = prediction["times"]
    radius = prediction["contact_radius"][:, 0]
    contact = case.contact
    keep = (
        np.asarray(contact.valid, dtype=bool)
        & (contact.times >= times[0])
        & (contact.times <= times[-1])
    )
    model_us = contact.times[keep] * 1.0e6
    at_frames = np.interp(contact.times[keep], times, radius)
    scale = float(case.length_scale)
    deviation = np.where(
        at_frames < contact.low[keep], (at_frames - contact.low[keep]) / scale,
        np.where(at_frames > contact.up[keep],
                 (at_frames - contact.up[keep]) / scale, 0.0),
    )
    return model_us, deviation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-pred-dir", required=True)
    parser.add_argument("--tref-pred-dir", required=True)
    parser.add_argument(
        "--cases", default="r50_full,r250_full,r500_full,r550_full,r750_full,r1000_full",
    )
    parser.add_argument("--label-a", default="t_ref = 1 s (baseline)")
    parser.add_argument("--label-b", default="t_ref = 0.01 s")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config(args.config)
    section = _section(config)
    wanted = [part.strip() for part in args.cases.split(",") if part.strip()]
    specs = {str(spec["case_id"]): spec for spec in _case_specs(section)}

    baseline_color = "#1f77b4"
    tref_color = "#d95f02"
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), sharex=False, sharey=True)
    for ax, case_id in zip(axes.ravel(), wanted):
        spec = specs.get(case_id)
        if spec is None:
            raise SystemExit(f"no case {case_id!r} in {args.config}")
        case = _make_case(section, spec, config, allow_synthetic=False)
        baseline = _load_prediction(
            Path(args.baseline_pred_dir) / f"pred_best_{case_id}.npz"
        )
        tref = _load_prediction(
            Path(args.tref_pred_dir) / f"pred_best_{case_id}.npz"
        )
        x_base, dev_base = _deviation(case, baseline)
        x_tref, dev_tref = _deviation(case, tref)
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.plot(x_base, dev_base, color=baseline_color, linewidth=2.0,
                label=args.label_a)
        ax.plot(x_tref, dev_tref, color=tref_color, linewidth=2.0,
                label=args.label_b)
        ax.set_title(f"{case_id} ({case.charge_radius_m * 1.0e3:.0f} mm)",
                     fontsize=10)
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("aligned model time (µs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("R_c,pred − interval (R_sep)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Contact deviation at measured frames, best checkpoints",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
