"""Where does the remaining error sit in time and in the air layer?

Bins use fixed reference data only:
  time   : first 20%, middle 60%, last 20% of the prediction time segment
  space  : position between the measured contact interval upper bound and the
           measured shock, thirds: near contact / middle / near shock

The predicted contact radius only decides whether a point is inside the model
domain; it never defines the bins.  Plots density profiles for three radii and
writes the binned statistics as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import _case_specs, _make_case, _section

from scripts.diagnose_radius_split import _evaluate_case, _load_prediction

FIELDS = ("rho", "u", "p")
TIME_LABELS = ("first20", "middle60", "last20")
SPACE_LABELS = ("near_contact", "middle", "near_shock")


def _bins(case, prediction, evaluation):
    obs = case.observations
    times = prediction["times"]
    span = float(times[-1] - times[0])
    valid_t = (obs.time >= times[0]) & (obs.time <= times[-1])
    rc = prediction["contact_radius"][:, 0]
    rc_at = np.interp(obs.time, times, rc)
    rs_at = np.asarray(case.shock.position(obs.time), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = (obs.radius - rc_at) / (rs_at - rc_at)
    in_domain = valid_t & np.isfinite(zeta) & (zeta >= 0.0) & (zeta <= 1.0)

    contact = case.contact
    valid_contact = np.asarray(contact.valid, dtype=bool)
    if np.any(valid_contact):
        up_at = np.interp(
            obs.time, contact.times[valid_contact], contact.up[valid_contact]
        )
    else:
        up_at = np.full(obs.time.shape, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        q_ref = (obs.radius - up_at) / (rs_at - up_at)
        q_ref = np.where(np.isfinite(q_ref), np.clip(q_ref, 0.0, 0.999999), 0.0)

    t_fraction = (obs.time - times[0]) / span if span > 0.0 else np.zeros(obs.time.shape)
    t_bin = np.where(
        t_fraction < 0.2, 0, np.where(t_fraction < 0.8, 1, 2)
    )
    q_bin = np.minimum((q_ref * 3.0).astype(np.int64), 2)

    return {
        "obs": obs,
        "valid_t": valid_t,
        "in_domain": in_domain,
        "t_bin": t_bin,
        "q_bin": q_bin,
        "times": times,
        "contact": contact,
    }


def _bin_stats(case, prediction, evaluation, bins) -> dict:
    obs = bins["obs"]
    valid_t = bins["valid_t"]
    in_domain = bins["in_domain"]
    t_bin = bins["t_bin"]
    q_bin = bins["q_bin"]

    stats: dict[str, dict] = {}
    for ti, tlabel in enumerate(TIME_LABELS):
        for qi, qlabel in enumerate(SPACE_LABELS):
            key = f"{tlabel}|{qlabel}"
            sel = valid_t & (t_bin == ti) & (q_bin == qi)
            entry = {
                "excluded": int(np.count_nonzero(sel & ~in_domain)),
                "n": {f: 0 for f in FIELDS},
                "sum_sq": {f: 0.0 for f in FIELDS},
            }
            for name in FIELDS:
                field = evaluation["fields"][name]
                use = np.asarray(field["use"], dtype=bool)
                error = (field["predicted"] - field["target"]) / field["scale"]
                used = sel & use
                count = int(np.count_nonzero(used))
                entry["n"][name] = count
                if count:
                    entry["sum_sq"][name] = float(np.sum(error[used] ** 2))
            entry["rmse"] = {
                f: (float(np.sqrt(entry["sum_sq"][f] / entry["n"][f]))
                    if entry["n"][f] else None)
                for f in FIELDS
            }
            entry.pop("sum_sq", None)
            stats[key] = entry

    contact = bins["contact"]
    times = bins["times"]
    keep = (
        np.asarray(contact.valid, dtype=bool)
        & (contact.times >= times[0])
        & (contact.times <= times[-1])
    )
    rc = prediction["contact_radius"][:, 0]
    rc_contact = np.interp(contact.times[keep], times, rc)
    below = np.maximum(contact.low[keep] - rc_contact, 0.0)
    above = np.maximum(rc_contact - contact.up[keep], 0.0)
    violation = (below + above) / float(case.length_scale)
    frac = (contact.times[keep] - times[0]) / float(times[-1] - times[0])
    contact_bins: dict[str, dict] = {}
    for ti, label in enumerate(TIME_LABELS):
        sel = (
            (frac < 0.2) if ti == 0 else
            ((frac >= 0.2) & (frac < 0.8)) if ti == 1 else (frac >= 0.8)
        )
        contact_bins[label] = {
            "n_frames": int(np.count_nonzero(sel)),
            "mean_violation_over_Rsep": (
                float(np.mean(violation[sel])) if np.any(sel) else None
            ),
        }
    return {"bins": stats, "contact": contact_bins}


def _aggregate(cases: list[dict]) -> dict:
    bins: dict[str, dict] = {}
    for case in cases:
        for key, entry in case["bins"].items():
            target = bins.setdefault(key, {
                "excluded": 0,
                "n": {f: 0 for f in FIELDS},
                "sum_sq": {f: 0.0 for f in FIELDS},
            })
            target["excluded"] += entry["excluded"]
            for f in FIELDS:
                target["n"][f] += entry["n"][f]
                if entry["rmse"][f] is not None:
                    target["sum_sq"][f] += entry["rmse"][f] ** 2 * entry["n"][f]
    for entry in bins.values():
        entry["rmse"] = {
            f: (float(np.sqrt(entry["sum_sq"][f] / entry["n"][f]))
                if entry["n"][f] else None)
            for f in FIELDS
        }
        entry.pop("sum_sq", None)
    contact: dict[str, dict] = {}
    for label in TIME_LABELS:
        frames = sum(
            case["contact"][label]["n_frames"] for case in cases
            if case["contact"][label]["mean_violation_over_Rsep"] is not None
        )
        if frames:
            value = sum(
                case["contact"][label]["mean_violation_over_Rsep"]
                * case["contact"][label]["n_frames"]
                for case in cases
                if case["contact"][label]["mean_violation_over_Rsep"] is not None
            ) / frames
        else:
            value = None
        contact[label] = {"n_frames": frames, "mean_violation_over_Rsep": value}
    return {"bins": bins, "contact": contact}


def _plot_density(case, prediction_a, prediction_b, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = prediction_a["times"]
    zetas = prediction_a["zeta"]
    targets = [0.25, 0.50, 0.75, 0.98]
    fractions = np.linspace(times[0], times[-1], len(times))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    obs = case.observations
    unique_times = np.unique(obs.time)

    baseline_color = "#1f77b4"
    extended_color = "#d95f02"
    for ax, fraction in zip(axes.ravel(), targets):
        target_t = times[0] + fraction * (times[-1] - times[0])
        index = int(np.argmin(np.abs(times - target_t)))
        rc_a = prediction_a["contact_radius"][index, 0]
        rs_a = prediction_a["shock_radius"][index, 0]
        rc_b = prediction_b["contact_radius"][index, 0]
        rs_b = prediction_b["shock_radius"][index, 0]
        radius_a = rc_a + zetas * (rs_a - rc_a)
        radius_b = rc_b + zetas * (rs_b - rc_b)
        ax.plot(radius_a, prediction_a["density"][index], color=baseline_color,
                linewidth=2.0, label="2000 loops")
        ax.plot(radius_b, prediction_b["density"][index], color=extended_color,
                linewidth=2.0, label="4000 loops")
        frame_time = unique_times[int(np.argmin(np.abs(unique_times - times[index])))]
        frame = obs.time == frame_time
        use = frame & np.asarray(obs.valid_density, bool) & np.isfinite(obs.density)
        if np.any(use):
            ax.scatter(obs.radius[use], obs.density[use], s=4, color="#777777",
                       alpha=0.45, linewidth=0, label="reference frame")
        ax.set_title(f"t_model = {times[index] * 1e6:.1f} µs", fontsize=9)
        ax.set_xlabel("radius (m)")
        ax.set_ylabel("density (kg/m³)")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{case.case_id}: density profiles ({case.charge_radius_m*1e3:.0f} mm charge)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir-a", required=True,
                        help="earlier run, e.g. 2000-loop cw3")
    parser.add_argument("--pred-dir-b", required=True,
                        help="current run, e.g. 4000-loop cw3")
    parser.add_argument("--label-a", default="2000 loops")
    parser.add_argument("--label-b", default="4000 loops")
    parser.add_argument("--focus-cases", default="r50_full,r500_full,r1000_full")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--plots-dir", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    section = _section(config)
    focus = {part.strip() for part in args.focus_cases.split(",") if part.strip()}
    plots = Path(args.plots_dir)
    cases_out: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = {"train": [], "validation": []}

    for spec in _case_specs(section):
        case = _make_case(section, spec, config, allow_synthetic=False)
        pred_a = _load_prediction(Path(args.pred_dir_a) / f"pred_best_{case.case_id}.npz")
        pred_b = _load_prediction(Path(args.pred_dir_b) / f"pred_best_{case.case_id}.npz")
        eval_b = _evaluate_case(case, pred_b)
        bins = _bins(case, pred_b, eval_b)
        stats = _bin_stats(case, pred_b, eval_b, bins)
        entry = {
            "role": case.role,
            "charge_radius_m": float(case.charge_radius_m),
            **stats,
        }
        cases_out[case.case_id] = entry
        grouped[case.role].append(entry)
        if case.case_id in focus:
            _plot_density(case, pred_a, pred_b,
                          plots / f"density_{case.case_id}.png")

    payload = {
        "config": str(args.config),
        "pred_dir_b": str(args.pred_dir_b),
        "protocol": {
            "time_bins": "first 20% / middle 60% / last 20% of prediction time segment",
            "space_bins": "thirds of (measured contact upper bound .. measured shock); "
                          "fixed reference data, not predicted R_c",
            "exclusion": "observations with valid time but outside the predicted [R_c, R_s] domain",
        },
        "groups": {role: _aggregate(rows) for role, rows in grouped.items()},
        "cases": cases_out,
        "focus_cases": sorted(focus),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(value):
        return " n/a" if value is None else f"{value:.3f}"

    for role, agg in payload["groups"].items():
        print(f"== {role}")
        print(f"{'bin':28s} {'rho':>6s} {'u':>6s} {'p':>6s} {'n_rho':>8s} {'excluded':>8s}")
        for key, entry in agg["bins"].items():
            print(f"{key:28s} {fmt(entry['rmse']['rho']):>6s} "
                  f"{fmt(entry['rmse']['u']):>6s} {fmt(entry['rmse']['p']):>6s} "
                  f"{entry['n']['rho']:8d} {entry['excluded']:8d}")
        print("contact violation by time bin:",
              {k: (None if v['mean_violation_over_Rsep'] is None
                   else round(v['mean_violation_over_Rsep'], 3))
               for k, v in agg["contact"].items()})
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
