"""Fixed-checkpoint diagnostic for the train/validation radius split.

Reads predictions already written by ``predict`` (one file per case and
checkpoint), recomputes the same metrics as ``evaluate``, and adds the
observation exclusion counts that ``evaluate`` does not report.  It also
compares two checkpoints on the points that are valid for both, so a change in
the scored point set cannot masquerade as an accuracy change.

This script never trains, never modifies a cache, and never uses the contact
interval midpoint as a target: the measured band is kept as ``low``/``up``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from spherical_blast_pinn.config import load_config
from spherical_blast_pinn.contact_air.cli import _case_specs, _make_case, _section

FIELDS = ("rho", "u", "p")
FIELD_ATTR = {
    "rho": ("density", "valid_density", "rho_scale"),
    "u": ("velocity", "valid_velocity", "u_scale"),
    "p": ("pressure", "valid_pressure", "p_scale"),
}


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {
            "times": np.asarray(archive["times"], dtype=np.float64),
            "zeta": np.asarray(archive["zeta"], dtype=np.float64),
            "density": np.asarray(archive["density"], dtype=np.float64),
            "velocity": np.asarray(archive["velocity"], dtype=np.float64),
            "pressure": np.asarray(archive["pressure"], dtype=np.float64),
            "contact_radius": np.asarray(archive["contact_radius"], dtype=np.float64),
            "shock_radius": np.asarray(archive["shock_radius"], dtype=np.float64),
        }
        if "checkpoint_loop" in archive.files:
            payload["checkpoint_loop"] = int(
                np.asarray(archive["checkpoint_loop"]).reshape(-1)[0]
            )
        else:
            payload["checkpoint_loop"] = -1
        return payload


def _interpolator(prediction: dict[str, np.ndarray], name: str):
    return RegularGridInterpolator(
        (prediction["times"], prediction["zeta"]), prediction[name],
        bounds_error=False, fill_value=np.nan,
    )


def _evaluate_case(case, prediction: dict[str, np.ndarray]) -> dict:
    times = prediction["times"]
    zeta_grid = prediction["zeta"]
    rc = prediction["contact_radius"][:, 0]
    obs = case.observations

    valid_t = (obs.time >= times[0]) & (obs.time <= times[-1])
    rc_at_obs = np.interp(obs.time, times, rc)
    rs_at_obs = np.asarray(case.shock.position(obs.time), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta_pred = (obs.radius - rc_at_obs) / (rs_at_obs - rc_at_obs)
    finite = np.isfinite(zeta_pred)
    in_domain = valid_t & finite & (zeta_pred >= 0.0) & (zeta_pred <= 1.0)
    points = np.column_stack([obs.time, np.clip(zeta_pred, 0.0, 1.0)])

    shared_excluded = {
        "time_out_of_model_range": int(np.count_nonzero(~valid_t)),
        "nonfinite_coordinate": int(np.count_nonzero(valid_t & ~finite)),
        "inside_predicted_contact": int(np.count_nonzero(valid_t & finite & (zeta_pred < 0.0))),
        "beyond_measured_shock": int(np.count_nonzero(valid_t & finite & (zeta_pred > 1.0))),
    }

    fields: dict[str, dict] = {}
    for name in FIELDS:
        attr, mask_attr, scale_attr = FIELD_ATTR[name]
        target = np.asarray(getattr(obs, attr), dtype=np.float64)
        mask = np.asarray(getattr(obs, mask_attr), dtype=bool)
        scale = float(getattr(case, scale_attr))
        predicted = np.asarray(_interpolator(prediction, attr)(points), dtype=np.float64)
        finite_pred = np.isfinite(predicted)
        use = in_domain & mask & finite_pred & np.isfinite(target)
        error = (predicted[use] - target[use]) / scale
        rmse = float(np.sqrt(np.mean(error * error))) if use.any() else None
        max_abs = float(np.max(np.abs(error))) if use.any() else None
        excluded = dict(shared_excluded)
        excluded["invalid_observation_mask"] = int(np.count_nonzero(in_domain & ~mask))
        excluded["nonfinite_prediction"] = int(
            np.count_nonzero(in_domain & mask & ~finite_pred)
        )
        excluded["nonfinite_target"] = int(
            np.count_nonzero(in_domain & mask & finite_pred & ~np.isfinite(target))
        )
        excluded["used"] = int(np.count_nonzero(use))
        if sum(excluded.values()) != obs.size:
            raise RuntimeError(
                f"{case.case_id} {name}: exclusion counts do not partition "
                f"{obs.size} observations: {excluded}"
            )
        fields[name] = {
            "rmse": rmse, "max_abs": max_abs, "n_used": int(np.count_nonzero(use)),
            "excluded": excluded,
            "use": use, "predicted": predicted, "target": target, "scale": scale,
        }

    valid_contact = (
        np.asarray(case.contact.valid, dtype=bool)
        & (case.contact.times >= times[0])
        & (case.contact.times <= times[-1])
    )
    rc_contact = np.interp(case.contact.times[valid_contact], times, rc)
    below = np.maximum(case.contact.low[valid_contact] - rc_contact, 0.0)
    above = np.maximum(rc_contact - case.contact.up[valid_contact], 0.0)
    violation = (below + above) / float(case.length_scale)
    contact = {
        "n_frames": int(np.count_nonzero(valid_contact)),
        "covered_frames": int(np.count_nonzero((below == 0.0) & (above == 0.0))),
        "mean_violation_over_Rsep": float(np.mean(violation)) if violation.size else None,
        "max_violation_over_Rsep": float(np.max(violation)) if violation.size else None,
        "coverage": (
            float(np.count_nonzero((below == 0.0) & (above == 0.0))
                  / np.count_nonzero(valid_contact))
            if np.count_nonzero(valid_contact) else None
        ),
    }

    return {
        "case_id": case.case_id,
        "role": case.role,
        "charge_radius_m": float(case.charge_radius_m),
        "n_observations": int(obs.size),
        "n_in_domain": int(np.count_nonzero(in_domain)),
        "fields": fields,
        "contact": contact,
        "times": times,
        "contact_radius": rc,
        "shock_radius": prediction["shock_radius"][:, 0],
        "zeta_grid": zeta_grid,
    }


def _pooled_rmse(items: list[tuple[float, int]]) -> float | None:
    count = sum(n for _, n in items)
    if count == 0:
        return None
    return float(np.sqrt(sum(rmse * rmse * n for rmse, n in items) / count))


def _aggregate(cases: list[dict]) -> dict:
    n_obs = sum(case["n_observations"] for case in cases)
    n_domain = sum(case["n_in_domain"] for case in cases)
    fields: dict[str, dict] = {}
    for name in FIELDS:
        pooled = _pooled_rmse([
            (case["fields"][name]["rmse"], case["fields"][name]["n_used"])
            for case in cases if case["fields"][name]["rmse"] is not None
        ])
        excluded_total: dict[str, int] = {}
        for case in cases:
            for key, value in case["fields"][name]["excluded"].items():
                excluded_total[key] = excluded_total.get(key, 0) + int(value)
        fields[name] = {
            "pooled_rmse": pooled,
            "n_used": sum(case["fields"][name]["n_used"] for case in cases),
            "excluded_total": excluded_total,
        }
    frames = sum(case["contact"]["n_frames"] for case in cases)
    covered = sum(case["contact"]["covered_frames"] for case in cases)
    violations = [
        case["contact"]["mean_violation_over_Rsep"] * case["contact"]["n_frames"]
        for case in cases if case["contact"]["n_frames"]
    ]
    contact = {
        "n_frames": frames,
        "covered_frames": covered,
        "coverage": (covered / frames) if frames else None,
        "mean_violation_over_Rsep": (sum(violations) / frames) if frames else None,
    }
    return {
        "n_cases": len(cases),
        "n_observations": n_obs,
        "n_in_domain": n_domain,
        "in_domain_fraction": (n_domain / n_obs) if n_obs else None,
        "fields": fields,
        "contact": contact,
    }


def _common_point_comparison(case, best: dict, last: dict) -> dict:
    out: dict[str, dict] = {}
    for name in FIELDS:
        best_field = best["fields"][name]
        last_field = last["fields"][name]
        common = best_field["use"] & last_field["use"]
        scale = best_field["scale"]
        target = best_field["target"]
        result = {"n_common": int(np.count_nonzero(common))}
        for tag, field in (("best", best_field), ("last", last_field)):
            if common.any():
                error = (field["predicted"][common] - target[common]) / scale
                result[f"{tag}_rmse"] = float(np.sqrt(np.mean(error * error)))
            else:
                result[f"{tag}_rmse"] = None
        out[name] = result
    return out


def _aggregate_common(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for name in FIELDS:
        n = sum(row[name]["n_common"] for row in rows)
        entry = {"n_common": n}
        for tag in ("best", "last"):
            if n:
                pooled = np.sqrt(sum(
                    row[name][f"{tag}_rmse"] ** 2 * row[name]["n_common"]
                    for row in rows if row[name]["n_common"]
                ) / n)
                entry[f"{tag}_rmse"] = float(pooled)
            else:
                entry[f"{tag}_rmse"] = None
        out[name] = entry
    return out


def _plot_case(case, best: dict, last: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alignment = case.event_alignment
    t0 = float(alignment["t0_model_s"])
    t_event_raw = float(alignment["event_time_raw_s"])
    t_sep_raw = float(alignment["t_separation_theoretical_s"])

    times = best["times"]
    # The cache stores contact times already event-aligned to model time
    # (t_model = t0 + (t_raw - t_event_raw)).  Predictions use the same model
    # time.  Filter and compare in model time; convert to raw time exactly once
    # for the raw-time panel.
    raw_times = times - t0 + t_event_raw

    contact = case.contact
    keep = (
        np.asarray(contact.valid, dtype=bool)
        & (contact.times >= times[0])
        & (contact.times <= times[-1])
    )
    contact_model_us = contact.times[keep] * 1.0e6
    contact_raw_us = (contact.times[keep] - t0 + t_event_raw) * 1.0e6
    low_mm = contact.low[keep] * 1.0e3
    up_mm = contact.up[keep] * 1.0e3

    raw_us = raw_times * 1.0e6
    model_us = times * 1.0e6
    best_mm = best["contact_radius"] * 1.0e3
    last_mm = last["contact_radius"] * 1.0e3

    # Signed contact deviation at the measured frame times.  Negative means the
    # prediction is inside (below R_c,low); positive means beyond R_c,up; zero
    # means the prediction is inside the measured interval.  The interval
    # midpoint is never used.
    def deviation(prediction_radius: np.ndarray) -> np.ndarray:
        at_frames = np.interp(contact.times[keep], times, prediction_radius)
        low = contact.low[keep]
        up = contact.up[keep]
        scale = float(case.length_scale)
        return np.where(
            at_frames < low, (at_frames - low) / scale,
            np.where(at_frames > up, (at_frames - up) / scale, 0.0),
        )

    dev_best = deviation(best["contact_radius"])
    dev_last = deviation(last["contact_radius"])

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 7.4), sharey="row", sharex="col",
    )
    (ax_raw, ax_model), (ax_dev_raw, ax_dev_model) = axes
    band_color = "#b8b8b8"
    best_color = "#1f77b4"
    last_color = "#d95f02"

    for ax, x_band, x_best, x_last, xlabel, event_x in (
        (ax_raw, contact_raw_us, raw_us, raw_us, "", t_sep_raw * 1.0e6),
        (ax_model, contact_model_us, model_us, model_us, "", t0 * 1.0e6),
    ):
        ax.fill_between(
            x_band, low_mm, up_mm, color=band_color, alpha=0.55, linewidth=0,
            label="measured contact interval [R_c,low, R_c,up]",
        )
        ax.plot(x_best, best_mm, color=best_color, linewidth=2.0,
                label="predicted R_c — best.pt")
        ax.plot(x_last, last_mm, color=last_color, linewidth=2.0, linestyle="--",
                label="predicted R_c — last.pt")
        ax.axvline(event_x, color="#555555", linewidth=1.0, linestyle=":")
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax, x, xlabel, event_x in (
        (ax_dev_raw, contact_raw_us, "raw frame time (µs)", t_sep_raw * 1.0e6),
        (ax_dev_model, contact_model_us, "aligned model time t_model (µs)", t0 * 1.0e6),
    ):
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.plot(x, dev_best, color=best_color, linewidth=2.0)
        ax.plot(x, dev_last, color=last_color, linewidth=2.0, linestyle="--")
        ax.axvline(event_x, color="#555555", linewidth=1.0, linestyle=":")
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax_raw.set_ylabel("radius (mm)")
    ax_dev_raw.set_ylabel("R_c,pred − interval (R_sep)")
    ax_raw.set_title("raw LS-DYNA frame time", fontsize=10)
    ax_model.set_title("event-aligned model time", fontsize=10)
    handles, labels = ax_raw.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{case.case_id}: predicted contact trajectory vs measured interval "
        f"({case.charge_radius_m * 1.0e3:.0f} mm charge)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--plots-dir", required=True)
    parser.add_argument(
        "--plot-cases", default="r50_full,r250_full,r500_full,r550_full,r750_full,r1000_full",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    section = _section(config)
    specs = _case_specs(section)
    predictions_dir = Path(args.predictions_dir)
    plot_cases = {part.strip() for part in args.plot_cases.split(",") if part.strip()}
    plots_dir = Path(args.plots_dir)

    cases_out: dict[str, dict] = {}
    common_rows: dict[str, list[dict]] = {"train": [], "validation": []}
    plotted = []

    for spec in specs:
        case = _make_case(section, spec, config, allow_synthetic=False)
        case_id = case.case_id
        best = _load_prediction(predictions_dir / f"pred_best_{case_id}.npz")
        last = _load_prediction(predictions_dir / f"pred_last_{case_id}.npz")
        best_eval = _evaluate_case(case, best)
        last_eval = _evaluate_case(case, last)
        common = _common_point_comparison(case, best_eval, last_eval)
        common_rows[case.role].append(common)
        if case_id in plot_cases:
            _plot_case(case, best_eval, last_eval,
                       plots_dir / f"contact_{case_id}.png")
            plotted.append(case_id)
        for result in (best_eval, last_eval):
            for name in FIELDS:
                result["fields"][name].pop("use", None)
                result["fields"][name].pop("predicted", None)
                result["fields"][name].pop("target", None)
                result["fields"][name].pop("scale", None)
            for key in ("times", "contact_radius", "shock_radius", "zeta_grid"):
                result.pop(key, None)
        cases_out[case_id] = {
            "role": case.role,
            "charge_radius_m": float(case.charge_radius_m),
            "best": best_eval,
            "last": last_eval,
            "common_points": common,
        }

    grouped: dict[str, dict] = {}
    for role in ("train", "validation"):
        cases = [cases_out[key] for key in cases_out if cases_out[key]["role"] == role]
        grouped[role] = {
            "best": _aggregate([case["best"] for case in cases]),
            "last": _aggregate([case["last"] for case in cases]),
            "common_points": _aggregate_common(common_rows[role]),
        }

    payload = {
        "config": str(args.config),
        "checkpoints": {"best": str(args.best), "last": str(args.last)},
        "protocol": {
            "observation_time": "raw obs.time compared to prediction model times, same as evaluate",
            "domain": "time inside prediction range, zeta in [0,1] with predicted R_c and measured R_s",
            "contact_target": "measured interval [R_c,low, R_c,up]; midpoint is never used",
            "plots": "left panel raw frame time, right panel event-aligned model time; both limited to the prediction time range",
        },
        "cases": cases_out,
        "groups": grouped,
        "plotted_cases": plotted,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(value):
        return "  n/a" if value is None else f"{value:.3f}"

    print(f"{'group':10s} {'ckpt':5s} {'rho':>6s} {'u':>6s} {'p':>6s} "
          f"{'viol':>6s} {'cov':>5s} {'in_dom':>14s}")
    for role, entry in grouped.items():
        for tag in ("best", "last"):
            agg = entry[tag]
            print(
                f"{role:10s} {tag:5s} {fmt(agg['fields']['rho']['pooled_rmse']):>6s} "
                f"{fmt(agg['fields']['u']['pooled_rmse']):>6s} "
                f"{fmt(agg['fields']['p']['pooled_rmse']):>6s} "
                f"{fmt(agg['contact']['mean_violation_over_Rsep']):>6s} "
                f"{fmt(agg['contact']['coverage']):>5s} "
                f"{agg['n_in_domain']:6d}/{agg['n_observations']:<7d}"
            )
    print("common-point RMSE:")
    for role, entry in grouped.items():
        common = entry["common_points"]
        print(
            f"{role:10s} rho {fmt(common['rho']['best_rmse'])}/{fmt(common['rho']['last_rmse'])} "
            f"u {fmt(common['u']['best_rmse'])}/{fmt(common['u']['last_rmse'])} "
            f"p {fmt(common['p']['best_rmse'])}/{fmt(common['p']['last_rmse'])}"
        )
    print(f"plots: {', '.join(plotted) if plotted else 'none'}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
