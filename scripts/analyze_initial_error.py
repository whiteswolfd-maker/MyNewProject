#!/usr/bin/env python3
"""Read-only initial-error localization for the A/B time-input checkpoints.

* Theoretical IC at t0: ContactNet R_c against case.initial.Rc0, AirNet rho/u/p
  against the separated-air state on a fixed zeta grid.
* Simulation comparison: actual valid observation rows in the first 1%, 5% and
  10% of each case's [t0,t_end] interval, evaluated at their physical (t,r).
  The reported air metrics use the A/B common domain; own-domain counts and
  exclusions are recorded separately.
* Contact labels are actual label times; no midpoint and no interpolation.

No training, no sampling, no checkpoint writes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _air_prediction(trainer, case, t, r):
    t = torch.as_tensor(t, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64)
    r0 = torch.full_like(t, float(case.charge_radius_m))
    with torch.enable_grad():
        r0c, rs, ds = trainer._contact_inputs(case, t)
        contact = trainer.contact_net(
            t.reshape(-1, 1), r0c, rs, ds, trainer._closures[case.case_id],
            time_input=case.time_input, create_graph=False,
        )
    contact = _detach_contact(contact)
    length = rs.reshape(-1) - contact.radius.reshape(-1)
    positive = torch.isfinite(length) & (length > 0.0)
    safe = torch.where(positive, length, torch.ones_like(length))
    zeta = (r - contact.radius.reshape(-1)) / safe
    inside = positive & (zeta >= 0.0) & (zeta <= 1.0)
    with torch.no_grad():
        fields = trainer.air_net(
            zeta.reshape(-1, 1), t.reshape(-1, 1), r0.reshape(-1, 1),
            trainer._scales[case.case_id], time_input=case.time_input,
        )
    return {
        "inside": inside.numpy(),
        "pred": {
            "rho": fields.density.detach().numpy().reshape(-1),
            "u": fields.velocity.detach().numpy().reshape(-1),
            "p": fields.pressure.detach().numpy().reshape(-1),
        },
    }


def _detach_contact(fields):
    return type(fields)(
        radius=fields.radius.detach(), speed=fields.speed.detach(),
        sigmoid=fields.sigmoid.detach(), raw=fields.raw.detach(),
        pressure=fields.pressure.detach(),
        relative_volume=fields.relative_volume.detach(),
    )


def _metrics(pred, target, mask, scale):
    use = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(target)
    n = int(use.sum())
    if n == 0:
        return {
            "status": "not_evaluable", "n": 0, "n_candidates": int(mask.sum()),
            "mse": None, "rmse": None, "n_nonfinite": int(mask.sum()),
        }
    error = (pred[use] - target[use]) / float(scale)
    mse = float(np.mean(error * error))
    return {
        "status": "ok", "n": n, "n_candidates": int(mask.sum()),
        "mse": mse, "rmse": math.sqrt(mse),
        "n_nonfinite": int(mask.sum() - n),
    }


def _t0_case(trainer, case):
    t = torch.tensor([float(case.t0)], dtype=torch.float64)
    r0, rs, ds = trainer._contact_inputs(case, t)
    contact = trainer.contact_net(
        t.reshape(-1, 1), r0, rs, ds, trainer._closures[case.case_id],
        time_input=case.time_input, create_graph=False,
    )
    rc_pred = float(contact.radius.detach().reshape(-1)[0])
    rc_target = float(case.initial.Rc0)
    zetas = np.linspace(0.01, 0.99, 99)
    zeta = torch.as_tensor(zetas, dtype=torch.float64).reshape(-1, 1)
    tt = torch.full_like(zeta, float(case.t0))
    rr0 = torch.full_like(zeta, float(case.charge_radius_m))
    with torch.no_grad():
        fields = trainer.air_net(
            zeta, tt, rr0, trainer._scales[case.case_id],
            time_input=case.time_input,
        )
    rho = fields.density.detach().numpy().reshape(-1)
    u = fields.velocity.detach().numpy().reshape(-1)
    p = fields.pressure.detach().numpy().reshape(-1)
    const_rho = np.full_like(rho, float(case.rho_scale))
    const_u = np.full_like(u, float(case.u_scale))
    const_p = np.full_like(p, float(case.p_scale))
    return {
        "case_id": case.case_id,
        "role": case.role,
        "Rc": {
            "prediction": rc_pred, "target": rc_target,
            "abs_error": rc_pred - rc_target,
            "normalised_error": (rc_pred - rc_target) / float(case.length_scale),
        },
        "air": {
            field: _metrics(pred, target, np.ones_like(pred, dtype=bool), scale)
            for field, pred, target, scale in (
                ("rho", rho, const_rho, case.rho_scale),
                ("u", u, const_u, case.u_scale),
                ("p", p, const_p, case.p_scale),
            )
        },
    }


def _window_case(trainer, case, lo_fraction, hi_fraction, max_rows=None):
    """Actual observations in the fractional window, evaluated at physical t,r."""
    t0, tend = float(case.t0), float(case.t_end)
    lo = t0 + float(lo_fraction) * (tend - t0)
    hi = t0 + float(hi_fraction) * (tend - t0)
    obs = case.observations
    time = np.asarray(obs.time, dtype=np.float64)
    radius = np.asarray(obs.radius, dtype=np.float64)
    valid = (
        np.asarray(obs.valid_density, dtype=bool)
        & np.asarray(obs.valid_velocity, dtype=bool)
        & np.asarray(obs.valid_pressure, dtype=bool)
        & np.isfinite(obs.density) & np.isfinite(obs.velocity)
        & np.isfinite(obs.pressure)
    )
    selected = valid & (time >= lo) & (time <= hi)
    if not selected.any():
        return {
            "case_id": case.case_id, "role": case.role,
            "lo_fraction": lo_fraction, "hi_fraction": hi_fraction,
            "status": "no_reference_rows", "selected": 0, "used": 0,
        }
    index = np.flatnonzero(selected)
    total = int(index.size)
    frame_used = None
    if max_rows is not None and total > int(max_rows):
        # Frame-balanced subsample: actual frames and actual rows only.  This
        # avoids letting frames with many rows dominate a late-time window.
        frame_all = np.asarray(obs.frame_index, dtype=np.int64)[index]
        unique_frames = np.unique(frame_all)
        if unique_frames.size > 200:
            unique_frames = unique_frames[
                np.linspace(0, unique_frames.size - 1, 200).astype(np.int64)
            ]
        kept = []
        for frame in unique_frames:
            rows = index[frame_all == frame]
            if rows.size > 100:
                rows = rows[np.linspace(0, rows.size - 1, 100).astype(np.int64)]
            kept.append(rows)
        index = np.unique(np.concatenate(kept)) if kept else index[:0]
        frame_used = int(unique_frames.size)
    prediction = _air_prediction(trainer, case, time[index], radius[index])
    target = {
        "rho": np.asarray(obs.density, dtype=np.float64)[index],
        "u": np.asarray(obs.velocity, dtype=np.float64)[index],
        "p": np.asarray(obs.pressure, dtype=np.float64)[index],
    }
    inside = prediction["inside"]
    frame_ids = np.unique(np.asarray(obs.frame_index)[index])
    own_metrics = {
        field: _metrics(
            prediction["pred"][field], target[field], inside,
            getattr(case, f"{field}_scale"),
        )
        for field in ("rho", "u", "p")
    }
    return {
        "case_id": case.case_id, "role": case.role,
        "lo_fraction": lo_fraction, "hi_fraction": hi_fraction,
        "lo_time": lo, "hi_time": hi,
        "status": "ok", "selected": total, "used": int(index.size),
        "n_frames": int(frame_ids.size),
        "own_metrics": own_metrics,
        "inside": inside,
        "pred": prediction["pred"],
        "target": target,
        "time": time[index],
        "radius": radius[index],
        "frame_index": np.asarray(obs.frame_index)[index],
    }


def _contact_window(trainer, case, upper):
    indices = case.contact.indices()
    times_all = np.asarray(case.contact.times[indices], dtype=np.float64)
    keep = (
        (times_all >= float(case.t0))
        & (times_all <= float(upper))
        & np.asarray([case.shock.has_slope(float(t)) for t in times_all])
    )
    times = times_all[keep]
    low = np.asarray(case.contact.low[indices], dtype=np.float64)[keep]
    up = np.asarray(case.contact.up[indices], dtype=np.float64)[keep]
    if times.size == 0:
        return {"status": "no_labels", "n": 0}
    return {
        "status": "ok",
        **diag._contact_metrics(trainer, case, times, low, up),
        "n_frames": int(times.size),
    }


def _aggregate_air(rows, model_key, common_mask):
    out = {}
    for field, scale_key in (("rho", "rho"), ("u", "u"), ("p", "p")):
        if common_mask == "common":
            masks = [row.get("common_mask") for row in rows]
        elif common_mask == "own":
            masks = [row.get("inside") for row in rows]
        else:
            masks = [row.get("inside") for row in rows]
        values = []
        for row, mask in zip(rows, masks):
            if mask is None:
                continue
            values.append(_metrics(
                row["pred"][field], row["target"][field], mask,
                row["scales"][field],
            ))
        out[field] = {
            "cases": len(values),
            "n": int(sum(v["n"] for v in values)),
            "mse": (
                sum(v["mse"] * v["n"] for v in values) / sum(v["n"] for v in values)
                if sum(v["n"] for v in values) else None
            ),
            "rmse": None,
        }
        if out[field]["mse"] is not None:
            out[field]["rmse"] = math.sqrt(out[field]["mse"])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-legacy", required=True)
    parser.add_argument("--config-similarity", required=True)
    parser.add_argument("--a-last", required=True)
    parser.add_argument("--a-best", required=True)
    parser.add_argument("--b-last", required=True)
    parser.add_argument("--b-best", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    legacy, _ = diag._build_trainer(args.config_legacy)
    similarity, _ = diag._build_trainer(args.config_similarity)
    cases = list(legacy.cases) + list(legacy.validation_cases)
    models = {
        "A_last": (legacy, Path(args.a_last)),
        "A_best": (legacy, Path(args.a_best)),
        "B_last": (similarity, Path(args.b_last)),
        "B_best": (similarity, Path(args.b_best)),
    }
    early_fractions = {"0_1pct": 0.01, "0_5pct": 0.05, "0_10pct": 0.10}
    windows = {
        "0_1pct": (0.0, 0.01, None),
        "0_5pct": (0.0, 0.05, None),
        "0_10pct": (0.0, 0.10, None),
        "rest_10pct": (0.10, 1.0, 20000),
        "0_100pct": (0.0, 1.0, 20000),
    }
    result = {
        "config_legacy": args.config_legacy,
        "config_similarity": args.config_similarity,
        "checkpoints": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, (_, path) in models.items()
        },
        "theoretical_t0": {},
        "early_simulation": {},
        "contact_windows": {},
        "R350_thirds": {},
        "notes": {
            "theoretical_t0": (
                "t0 IC: ContactNet R_c vs case.initial.Rc0; AirNet rho/u/p vs "
                "separated-air constants on zeta in [0.01,0.99]."
            ),
            "early_simulation": (
                "actual valid observation rows in the listed similarity window; "
                "model evaluated at the same physical (t,r); reported air metrics "
                "use the A/B common domain, own counts and exclusions are separate."
            ),
            "contact": (
                "actual contact label times only; no interval midpoint and no "
                "interpolation."
            ),
            "normalisation": (
                "normalised MSE/RMSE divides by case rho_scale/u_scale/p_scale "
                "or length_scale; it is not a relative-to-local-truth percentage."
            ),
        },
    }

    for label, (trainer, path) in models.items():
        trainer.load(path)
        print(label, "loop", trainer.loop)
        t0_rows = {}
        for case in cases:
            t0_rows[case.case_id] = _t0_case(trainer, case)
        result["theoretical_t0"][label] = t0_rows

        early = {}
        for case in cases:
            case_windows = {}
            for name, (lo_fraction, hi_fraction, max_rows) in windows.items():
                row = _window_case(
                    trainer, case, lo_fraction, hi_fraction, max_rows
                )
                if row.get("status") == "ok":
                    row["scales"] = {
                        "rho": case.rho_scale, "u": case.u_scale, "p": case.p_scale,
                    }
                case_windows[name] = row
            early[case.case_id] = case_windows
        result["early_simulation"][label] = early

        contact = {}
        for case in cases:
            entry = {}
            for name, fraction in early_fractions.items():
                upper = float(case.t0) + fraction * (
                    float(case.t_end) - float(case.t0)
                )
                entry[name] = _contact_window(trainer, case, upper)
            contact[case.case_id] = entry
        result["contact_windows"][label] = contact

    # R350 contact by thirds, last and best.
    for label, (trainer, path) in models.items():
        trainer.load(path)
        case = next(c for c in cases if c.case_id == "r350_full")
        indices = case.contact.indices()
        times = np.asarray(case.contact.times[indices], dtype=np.float64)
        low = np.asarray(case.contact.low[indices], dtype=np.float64)
        up = np.asarray(case.contact.up[indices], dtype=np.float64)
        masks = diag._third_masks(times, case.t0, case.t_end)
        result["R350_thirds"][label] = {
            str(i): {
                "status": "ok",
                **diag._contact_metrics(
                    trainer, case, times[m], low[m], up[m]
                ),
                "n_labels": int(m.sum()),
            }
            for i, m in enumerate(masks)
        }

    # Common-domain air metrics for A/B pairs.
    for suffix in ("last", "best"):
        a_label, b_label = f"A_{suffix}", f"B_{suffix}"
        pair = {}
        for name, (lo_fraction, hi_fraction, max_rows) in windows.items():
            rows_a = []
            rows_b = []
            common_masks = []
            exclusions = []
            for case in cases:
                arow = result["early_simulation"][a_label][case.case_id][name]
                brow = result["early_simulation"][b_label][case.case_id][name]
                if arow.get("status") != "ok" or brow.get("status") != "ok":
                    continue
                assert np.array_equal(arow["time"], brow["time"])
                assert np.array_equal(arow["radius"], brow["radius"])
                common = arow["inside"] & brow["inside"]
                arow["common_mask"] = common
                brow["common_mask"] = common
                arow["common_metrics"] = {
                    field: _metrics(
                        arow["pred"][field], arow["target"][field], common,
                        arow["scales"][field],
                    )
                    for field in ("rho", "u", "p")
                }
                brow["common_metrics"] = {
                    field: _metrics(
                        brow["pred"][field], brow["target"][field], common,
                        brow["scales"][field],
                    )
                    for field in ("rho", "u", "p")
                }
                rows_a.append(arow)
                rows_b.append(brow)
                common_masks.append(common)
                exclusions.append({
                    "case_id": case.case_id,
                    "selected_total": int(arow["selected"]),
                    "used": int(arow["used"]),
                    "A_inside": int(arow["inside"].sum()),
                    "B_inside": int(brow["inside"].sum()),
                    "common": int(common.sum()),
                    "A_excluded": int(arow["used"] - arow["inside"].sum()),
                    "B_excluded": int(brow["used"] - brow["inside"].sum()),
                    "common_excluded": int(arow["used"] - common.sum()),
                })
            pair[name] = {
                "A_own": _aggregate_air(rows_a, "A", "own"),
                "B_own": _aggregate_air(rows_b, "B", "own"),
                "A_common": _aggregate_air(rows_a, "A", "common"),
                "B_common": _aggregate_air(rows_b, "B", "common"),
                "exclusions": exclusions,
            }
        result.setdefault("pair_early_simulation", {})[suffix] = pair

    # Drop the in-memory arrays; only metrics/counts are written.
    raw_keys = (
        "inside", "pred", "target", "time", "radius", "frame_index",
        "common_mask", "scales",
    )
    for label_rows in result["early_simulation"].values():
        for case_windows in label_rows.values():
            for row in case_windows.values():
                for key in raw_keys:
                    row.pop(key, None)

    # Theoretical t0 aggregate per label and role.
    for label, rows in result["theoretical_t0"].items():
        for role in ("train", "validation"):
            subset = [r for r in rows.values() if r["role"] == role]
            result.setdefault("theoretical_t0_aggregate", {}).setdefault(label, {})[role] = {
                "Rc_rmse_normalised": math.sqrt(sum(
                    (r["Rc"]["normalised_error"] ** 2) for r in subset
                ) / len(subset)),
                "rho_rmse_normalised": math.sqrt(sum(
                    r["air"]["rho"]["mse"] for r in subset
                ) / len(subset)),
                "u_rmse_normalised": math.sqrt(sum(
                    r["air"]["u"]["mse"] for r in subset
                ) / len(subset)),
                "p_rmse_normalised": math.sqrt(sum(
                    r["air"]["p"]["mse"] for r in subset
                ) / len(subset)),
                "n_cases": len(subset),
            }

    (output / "initial_error.json").write_text(
        json.dumps(result, indent=2, default=diag._jsonable), encoding="utf-8"
    )
    print("wrote", output / "initial_error.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
