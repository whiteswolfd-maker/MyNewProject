#!/usr/bin/env python3
"""Evaluate the ContactNet learning-rate resume experiment (corrected).

The contact metrics use every legal contact label, per-case mean then
case-equal mean, exact integer coverage and fixed early/middle/late bins.

Air metrics use three point sets:
* each model's own valid observations;
* the A/B common valid observations;
* the common valid observations of the start-4000, A-4500 and B-4500 models.

Physics residuals are reported in two coordinate systems:
* ``own_mapped_t_zeta``: the old fixed t/zeta grid, each model mapped through
  its own R_c (kept for comparison; not a fixed physical point set);
* ``fixed_physical_common``: one saved data-derived (case_id, t, r) reference
  set, mapped to each model's zeta, with residuals only on the common valid
  points.  The contact-interval midpoint is never used.

This script only reads checkpoints; it never trains or writes model state.
"""
from __future__ import annotations

import argparse
import hashlib
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
import physical_residuals as phys  # noqa: E402
from spherical_blast_pinn.contact_air.losses import (  # noqa: E402
    data_coordinates, pde_loss,
)
from spherical_blast_pinn.contact_air.mapping import build_geometry  # noqa: E402
from spherical_blast_pinn.contact_air.models import (  # noqa: E402
    air_fields_with_derivatives,
)
from spherical_blast_pinn.contact_air.trainer import _detach_contact  # noqa: E402

GROUPS = {
    "A_contact_lr_1e-4": "contact_lr_1e-4_resume4000_to4500",
    "B_contact_lr_2p5e-5": "contact_lr_2p5e-5_resume4000_to4500",
}
LABELS = ("start_4000", "A_4500", "B_4500")
PAIR_LABELS = ("A_4500", "B_4500")
PLOT_CASES = (
    "r50_full", "r100_full", "r250_full", "r500_full",
    "r550_full", "r750_full", "r1000_full",
)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _points(trainer, cases) -> list[dict]:
    result = []
    for case in cases:
        indices = case.contact.indices()
        times = np.asarray(case.contact.times[indices], dtype=np.float64)
        keep = (
            (times >= float(case.t0))
            & (times <= float(case.t_end))
            & np.asarray([case.shock.has_slope(float(t)) for t in times])
        )
        indices = indices[keep]
        result.append({
            "case_id": case.case_id,
            "times": np.asarray(case.contact.times[indices], dtype=np.float64),
            "low": np.asarray(case.contact.low[indices], dtype=np.float64),
            "up": np.asarray(case.contact.up[indices], dtype=np.float64),
            "t0": float(case.t0),
            "t_end": float(case.t_end),
        })
    return result


def _reference_metrics(trainer, cases, points) -> dict:
    by_id = {case.case_id: case for case in cases}
    rows = []
    for point in points:
        case = by_id[point["case_id"]]
        metrics = diag._contact_metrics(
            trainer, case, point["times"], point["low"], point["up"]
        )
        masks = diag._third_masks(point["times"], point["t0"], point["t_end"])
        thirds = [
            diag._contact_metrics(
                trainer, case, point["times"][mask],
                point["low"][mask], point["up"][mask],
            )
            for mask in masks
        ]
        rows.append({"case_id": case.case_id, **metrics, "thirds": thirds})
    aggregate = diag._aggregate_metrics(rows, expected_cases=len(points))
    segment = []
    for which in range(3):
        segment.append(diag._aggregate_metrics(
            [{"case_id": row["case_id"], **row["thirds"][which]} for row in rows],
            expected_cases=len(points),
        ))
    aggregate.update({
        "cases": rows,
        "thirds": segment,
        "definition": {
            "macro": "sum(M_k)/K over cases with a valid label",
            "pooled": "sum(n_k*M_k)/sum(n_k)",
            "coverage": "sum(covered_count_k)/sum(n_k), exact integers",
        },
    })
    return aggregate


def _air_predictions(trainer, case) -> dict:
    obs = case.observations
    time = np.asarray(obs.time, dtype=np.float64)
    radius = np.asarray(obs.radius, dtype=np.float64)
    n = time.size
    selected = np.isfinite(time) & np.isfinite(radius)
    selected &= (time >= float(case.t0)) & (time <= float(case.t_end))
    result = {
        "case_id": case.case_id,
        "n_total": int(n),
        "n_time_valid": int(selected.sum()),
        "time": time,
        "radius": radius,
        "target": {
            "rho": np.asarray(obs.density, dtype=np.float64),
            "u": np.asarray(obs.velocity, dtype=np.float64),
            "p": np.asarray(obs.pressure, dtype=np.float64),
        },
        "valid": {
            "rho": np.asarray(obs.valid_density, bool),
            "u": np.asarray(obs.valid_velocity, bool),
            "p": np.asarray(obs.valid_pressure, bool),
        },
        "pred": {
            name: np.full(n, np.nan, dtype=np.float64)
            for name in ("rho", "u", "p")
        },
        "inside": np.zeros(n, dtype=bool),
    }
    if not selected.any():
        return result
    index = np.nonzero(selected)[0]
    t = torch.as_tensor(time[index], dtype=torch.float64)
    r = torch.as_tensor(radius[index], dtype=torch.float64)
    with torch.enable_grad():
        r0, rs, ds = trainer._contact_inputs(case, t)
        contact = trainer.contact_net(
            t.reshape(-1, 1), r0, rs, ds, trainer._closures[case.case_id],
            create_graph=False,
        )
    contact = _detach_contact(contact)
    zeta, inside, _ = data_coordinates(
        radius=r, t=t, contact=contact, case=case
    )
    inside = inside.reshape(-1)
    r0 = torch.full_like(t, float(case.charge_radius_m))
    with torch.no_grad():
        fields = trainer.air_net(
            zeta.detach(), t.reshape(-1, 1), r0.reshape(-1, 1),
            trainer._scales[case.case_id],
        )
    result["inside"][index] = inside.detach().numpy()
    for name, value in (
        ("rho", fields.density), ("u", fields.velocity), ("p", fields.pressure)
    ):
        values = value.detach().reshape(-1).numpy()
        values = np.where(result["inside"][index], values, np.nan)
        result["pred"][name][index] = values
    return result


def _field_metrics(pred, target, mask, scale) -> dict:
    """MSE/RMSE on one explicit mask; empty is not an error of zero."""
    use = mask & np.isfinite(pred) & np.isfinite(target)
    n = int(use.sum())
    candidates = int(mask.sum())
    if n == 0:
        return {
            "status": "not_evaluable", "n": 0, "n_candidates": candidates,
            "mse": None, "rmse": None, "n_nonfinite": 0,
        }
    error = (pred[use] - target[use]) / float(scale)
    mse = float(np.mean(error * error))
    return {
        "status": "ok", "n": n, "n_candidates": candidates,
        "mse": mse, "rmse": math.sqrt(mse),
        "n_nonfinite": int(candidates - n),
    }


def _common_mask(models, labels, name) -> np.ndarray:
    mask = np.ones_like(models[labels[0]]["inside"], dtype=bool)
    for label in labels:
        model = models[label]
        mask &= model["valid"][name] & model["inside"]
        mask &= np.isfinite(model["pred"][name])
    return mask


def _common_metrics(models, labels, name, mask, scale) -> dict:
    return {
        label: _field_metrics(
            models[label]["pred"][name], models[label]["target"][name],
            mask, scale,
        )
        for label in labels
    }


def _air_case_metrics(models: dict, case) -> dict:
    scales = {"rho": case.rho_scale, "u": case.u_scale, "p": case.p_scale}
    own = {label: {} for label in models}
    common_two = {name: {} for name in ("rho", "u", "p")}
    common_three = {name: {} for name in ("rho", "u", "p")}
    counts = {}
    for label, model in models.items():
        counts[label] = {
            "n_total": model["n_total"],
            "n_time_valid": model["n_time_valid"],
            "n_inside": int(model["inside"].sum()),
            "n_outside": int(model["n_total"] - model["inside"].sum()),
            "inside_fraction": (
                float(model["inside"].sum() / model["n_total"])
                if model["n_total"] else None
            ),
        }
    for name in ("rho", "u", "p"):
        for label, model in models.items():
            own[label][name] = _field_metrics(
                model["pred"][name], model["target"][name],
                model["valid"][name] & model["inside"], scales[name],
            )
        pair_mask = _common_mask(models, PAIR_LABELS, name)
        common_two[name] = _common_metrics(
            models, PAIR_LABELS, name, pair_mask, scales[name]
        )
        three_mask = _common_mask(models, LABELS, name)
        common_three[name] = _common_metrics(
            models, LABELS, name, three_mask, scales[name]
        )
    return {
        "case_id": case.case_id,
        "counts": counts,
        "own": own,
        "common_two": common_two,
        "common_three": common_three,
    }


def _aggregate_metric_rows(rows: list[dict]) -> dict:
    """Aggregate explicit per-case metrics without hiding missing cases."""
    values = [
        row for row in rows
        if row is not None and row.get("status") == "ok"
    ]
    missing = [
        row for row in rows
        if row is None or row.get("status") != "ok"
    ]
    result = {
        "n_cases": len(rows),
        "n_cases_evaluable": len(values),
        "n_cases_missing": len(missing),
        "complete": len(values) == len(rows),
        "status": "ok" if values else "not_evaluable",
        "mean_case_rmse": (
            sum(row["rmse"] for row in values) / len(values)
            if values else None
        ),
        "macro_mse": (
            sum(row["mse"] for row in values) / len(values)
            if values else None
        ),
        "n_points": int(sum(row["n"] for row in values)),
    }
    if values:
        result["macro_rmse"] = math.sqrt(result["macro_mse"])
        total = result["n_points"]
        result["pooled_mse"] = sum(
            row["mse"] * row["n"] for row in values
        ) / total
        result["pooled_rmse"] = math.sqrt(result["pooled_mse"])
    else:
        result["macro_rmse"] = None
        result["pooled_mse"] = None
        result["pooled_rmse"] = None
    return result


def _aggregate_block(per_case: dict, getter) -> dict:
    return _aggregate_metric_rows([getter(item) for item in per_case.values()])


def _old_own_mapping_physics(trainer, case) -> dict:
    """Old metric: fixed t/zeta, each model mapped through its own R_c."""
    times = np.linspace(float(case.t0), float(case.t_end), 9)
    zetas = np.linspace(0.0, 1.0, 17)
    tt, zz = np.meshgrid(times, zetas, indexing="ij")
    t = torch.as_tensor(tt.reshape(-1), dtype=torch.float64)
    zeta = torch.as_tensor(zz.reshape(-1), dtype=torch.float64)
    with torch.enable_grad():
        r0, rs, ds = trainer._contact_inputs(case, t)
        contact = trainer.contact_net(
            t.reshape(-1, 1), r0, rs, ds, trainer._closures[case.case_id],
            create_graph=True,
        )
        geometry = build_geometry(contact.radius, contact.speed, rs, ds, zeta)
        fields, derivatives = air_fields_with_derivatives(
            trainer.air_net, zeta, t, r0, trainer._scales[case.case_id],
            create_graph=True,
        )
        pde = pde_loss(
            fields=fields, derivatives=derivatives, geometry=geometry, case=case
        )
        air0 = trainer.air_net(
            torch.zeros_like(zeta).reshape(-1, 1), t.reshape(-1, 1),
            r0.reshape(-1, 1), trainer._scales[case.case_id],
        )
        u_error = (
            air0.velocity.reshape(-1) - contact.speed.reshape(-1)
        ) / case.u_scale
        p_error = (
            air0.pressure.reshape(-1) - contact.pressure.reshape(-1)
        ) / case.p_scale
    return {
        "case_id": case.case_id,
        "label": "own_mapped_t_zeta",
        "n_points": int(t.numel()),
        "pde_mass_normalised": pde.parts["pde_mass_normalised"],
        "pde_momentum_normalised": pde.parts["pde_momentum_normalised"],
        "pde_energy_normalised": pde.parts["pde_energy_normalised"],
        "interface_velocity_rmse": float(torch.sqrt(torch.mean(u_error ** 2))),
        "interface_pressure_rmse": float(torch.sqrt(torch.mean(p_error ** 2))),
    }


def _initial_condition_metrics(trainer, case) -> dict:
    """AirNet initial-time mismatch against the separated-air state."""
    zetas = np.linspace(0.01, 0.99, 33)
    zeta = torch.as_tensor(zetas, dtype=torch.float64).reshape(-1, 1)
    t = torch.full_like(zeta, float(case.t0))
    r0 = torch.full_like(zeta, float(case.charge_radius_m))
    with torch.no_grad():
        fields = trainer.air_net(
            zeta, t, r0, trainer._scales[case.case_id]
        )
    result = {"case_id": case.case_id, "n": int(zeta.numel()),
              "status": "ok"}
    for name, values, scale in (
        ("rho", fields.density, case.rho_scale),
        ("u", fields.velocity, case.u_scale),
        ("p", fields.pressure, case.p_scale),
    ):
        error = (
            values.detach().reshape(-1).numpy() - float(scale)
        ) / float(scale)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error * error)))
    return result


def _aggregate_initial(rows: list[dict]) -> dict:
    n = sum(row["n"] for row in rows)
    result = {"n": n, "status": "ok" if n else "not_evaluable"}
    for name in ("rho", "u", "p"):
        result[f"{name}_rmse"] = (
            math.sqrt(sum(
                row[f"{name}_rmse"] ** 2 * row["n"] for row in rows
            ) / n) if n else None
        )
    return result


def _reference_physical_points(trainer, cases, limit: int) -> dict:
    payload = {"case_ids": [case.case_id for case in cases], "cases": {}}
    arrays = {}
    for case in cases:
        t, r = phys._subsample(case, limit)
        t_np = t.detach().numpy()
        r_np = r.detach().numpy()
        slope = np.asarray(
            [case.shock.has_slope(float(value)) for value in t_np], dtype=bool
        )
        payload["cases"][case.case_id] = {
            "n": int(t_np.size),
            "files": {"t": f"t_{case.case_id}", "r": f"r_{case.case_id}",
                      "slope": f"slope_{case.case_id}"},
            "n_slope_invalid": int((~slope).sum()),
        }
        arrays[f"t_{case.case_id}"] = t_np
        arrays[f"r_{case.case_id}"] = r_np
        arrays[f"slope_{case.case_id}"] = slope
    payload["points_per_case_limit"] = int(limit)
    payload["protocol"] = (
        "deterministic subsample of valid observations within [t0,t_end]; "
        "shared (case_id,t,r) for every model; no contact-interval midpoint"
    )
    return {"metadata": payload, "arrays": arrays}


def _fixed_physical_physics(
    trainer, cases, checkpoints: dict, reference: dict,
) -> dict:
    """PDE residuals on one data-derived (t,r) set, common to three models."""
    domains = {}
    for label, path in checkpoints.items():
        trainer.load(path)
        domains[label] = {}
        for case in cases:
            item = reference["metadata"]["cases"][case.case_id]
            t = torch.as_tensor(reference["arrays"][f"t_{case.case_id}"],
                                dtype=torch.float64)
            r = torch.as_tensor(reference["arrays"][f"r_{case.case_id}"],
                                dtype=torch.float64)
            slope = reference["arrays"][f"slope_{case.case_id}"]
            inside, rc, rc_dot, rs, ds, _, scales = phys._domain(
                case, trainer.contact_net, t, r
            )
            finite = (
                torch.isfinite(rc) & torch.isfinite(rc_dot)
                & torch.isfinite(rs) & torch.isfinite(ds)
            ).numpy()
            length = rs - rc
            positive = torch.isfinite(length) & (length > 0.0)
            zeta = torch.where(
                positive, (r - rc) / torch.where(
                    positive, length, torch.ones_like(length)
                ), torch.zeros_like(length),
            )
            domains[label][case.case_id] = {
                "t": t, "r": r, "slope": slope,
                "inside": inside.numpy(),
                "finite": finite,
                "zeta": zeta.numpy(),
                "rc": rc, "rc_dot": rc_dot,
                "rs": rs, "ds": ds,
                "scales": scales,
                "n": int(t.numel()),
                "n_slope_invalid": int((~slope).sum()),
                "n_nonfinite": int((slope & ~finite).sum()),
                "n_outside": int((slope & finite & ~inside.numpy()).sum()),
                "n_layer_invalid": int((
                    slope & finite & ((rs - rc).numpy() <= 0.0)
                ).sum()),
            }
    rows = {}
    for case in cases:
        item = domains[LABELS[0]][case.case_id]
        common = item["slope"].copy()
        for label in LABELS:
            common &= domains[label][case.case_id]["inside"]
        interior = common.copy()
        for label in LABELS:
            zeta = domains[label][case.case_id]["zeta"]
            interior &= (zeta > 1.0e-12) & (zeta < 1.0 - 1.0e-12)
        counts = {
            label: {
                "candidates": domains[label][case.case_id]["n"],
                "slope_invalid": domains[label][case.case_id]["n_slope_invalid"],
                "nonfinite": domains[label][case.case_id]["n_nonfinite"],
                "outside": domains[label][case.case_id]["n_outside"],
                "layer_invalid": domains[label][case.case_id]["n_layer_invalid"],
                "common": int(common.sum()),
                "common_interior": int(interior.sum()),
                "boundary_excluded": int(common.sum() - interior.sum()),
            }
            for label in LABELS
        }
        residuals = {}
        for label in LABELS:
            trainer.load(checkpoints[label])
            domain = domains[label][case.case_id]
            residuals[label] = phys._residual_rms(
                case, trainer.air_net,
                domain["t"], domain["r"], domain["rc"], domain["rc_dot"],
                domain["rs"], domain["ds"], domain["scales"],
                torch.as_tensor(interior, dtype=torch.bool),
            )
            residuals[label]["status"] = (
                "ok" if residuals[label]["n"] > 0 else "not_evaluable"
            )
        rows[case.case_id] = {
            "case_id": case.case_id,
            "role": case.role,
            "common": int(common.sum()),
            "counts": counts,
            "residuals": residuals,
        }
    return rows


def _aggregate_physics_rows(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"status": "not_evaluable", "n": 0}
    fields = ("mass", "momentum", "energy")
    result = {"n_candidates": 0, "n": 0, "n_nonfinite": 0}
    for field in fields:
        valid = [
            row["residuals"][label] for row in rows
            if row["residuals"][label].get(field) is not None
        ]
        n = sum(item["n"] for item in valid)
        result[field] = (
            math.sqrt(sum(item[field] ** 2 * item["n"] for item in valid) / n)
            if n else None
        )
    result["n"] = sum(
        row["residuals"][label]["n"] for row in rows
    )
    result["n_nonfinite"] = sum(
        row["residuals"][label].get("n_nonfinite", 0) for row in rows
    )
    result["n_candidates"] = sum(
        row["counts"][label]["common_interior"] for row in rows
    )
    result["status"] = "ok" if result["n"] else "not_evaluable"
    return result


def _aggregate_interface(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "not_evaluable", "n": 0}
    n = sum(row["n"] for row in rows)
    result = {"n": n, "status": "ok" if n else "not_evaluable"}
    for key in ("velocity_rms", "pressure_rms"):
        result[key] = (
            math.sqrt(sum(row[key] ** 2 * row["n"] for row in rows) / n)
            if n else None
        )
    return result


def _contact_trajectory(trainer, case, points: int = 200):
    times = np.linspace(float(case.t0), float(case.t_end), int(points))
    t = torch.as_tensor(times, dtype=torch.float64)
    with torch.enable_grad():
        contact = diag._contact_at(trainer, case, t, create_graph=False)
    return times, contact.radius.detach().numpy().reshape(-1)


def _contact_at_label_times(trainer, case, times):
    t = torch.as_tensor(np.asarray(times), dtype=torch.float64)
    with torch.enable_grad():
        contact = diag._contact_at(trainer, case, t, create_graph=False)
    return contact.radius.detach().numpy().reshape(-1)


def _write_plots(output: Path, metrics: dict, trainer, cases, checkpoints: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loops = metrics["loops"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for group, entry in metrics["groups"].items():
        train = [entry[str(loop)]["train"]["macro_mse"] for loop in loops]
        validation = [entry[str(loop)]["validation"]["macro_mse"] for loop in loops]
        axes[0].plot(loops, train, marker="o", label=f"{group} train")
        axes[1].plot(loops, validation, marker="o", label=f"{group} validation")
    axes[0].set_ylabel("train full-label macro MSE")
    axes[1].set_ylabel("validation full-label macro MSE")
    axes[1].set_xlabel("loop")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "contact_error_vs_loop.png", dpi=160)
    plt.close(fig)

    by_id = {case.case_id: case for case in cases}
    fig, axes = plt.subplots(len(PLOT_CASES), 1, figsize=(9, 3.0 * len(PLOT_CASES)))
    for axis, case_id in zip(axes, PLOT_CASES):
        case = by_id[case_id]
        indices = case.contact.indices()
        times = np.asarray(case.contact.times[indices], dtype=np.float64)
        keep = (
            (times >= float(case.t0)) & (times <= float(case.t_end))
            & np.asarray([case.shock.has_slope(float(value)) for value in times])
        )
        times = times[keep]
        low = np.asarray(case.contact.low[indices][keep]) * 1000.0
        up = np.asarray(case.contact.up[indices][keep]) * 1000.0
        axis.fill_between(times * 1000.0, low, up, color="0.8",
                          label="measured interval")
        for label, (loop, checkpoint) in checkpoints.items():
            trainer.load(checkpoint)
            grid, radius = _contact_trajectory(trainer, case)
            axis.plot(grid * 1000.0, radius * 1000.0,
                      label=f"{label} loop {loop}")
        axis.set_title(case_id)
        axis.set_ylabel("R_c (mm)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    axes[-1].set_xlabel("t (ms)")
    fig.tight_layout()
    fig.savefig(output / "contact_trajectories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(PLOT_CASES), 1, figsize=(9, 2.6 * len(PLOT_CASES)))
    for axis, case_id in zip(axes, PLOT_CASES):
        case = by_id[case_id]
        indices = case.contact.indices()
        times = np.asarray(case.contact.times[indices], dtype=np.float64)
        keep = (
            (times >= float(case.t0)) & (times <= float(case.t_end))
            & np.asarray([case.shock.has_slope(float(value)) for value in times])
        )
        truth_t = times[keep]
        low = np.asarray(case.contact.low[indices][keep])
        up = np.asarray(case.contact.up[indices][keep])
        for label, (loop, checkpoint) in checkpoints.items():
            trainer.load(checkpoint)
            # Direct model call at the label times, not an interpolation of a
            # coarse display grid.
            radius = _contact_at_label_times(trainer, case, truth_t)
            gap = np.where(radius < low, radius - low,
                           np.where(radius > up, radius - up, 0.0))
            axis.plot(truth_t * 1000.0, gap / case.length_scale,
                      label=f"{label} loop {loop}")
        axis.axhline(0.0, color="0.5", linewidth=1)
        axis.set_title(f"{case_id}: contact outside measured interval")
        axis.set_ylabel("signed gap / R_sep")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    axes[-1].set_xlabel("t (ms)")
    fig.tight_layout()
    fig.savefig(output / "contact_interval_residuals.png", dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points-per-case", type=int, default=20000)
    parser.add_argument("--interface-points", type=int, default=128)
    args = parser.parse_args(argv)
    experiment = Path(args.experiment_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer, _ = diag._build_trainer(args.config)
    train_cases = list(trainer.cases)
    validation_cases = list(trainer.validation_cases)
    all_cases = train_cases + validation_cases
    train_points = _points(trainer, train_cases)
    validation_points = _points(trainer, validation_cases)
    start = experiment / "checkpoints" / "loop_4000.pt"
    checkpoints = {
        "start_4000": start,
        "A_4500": experiment / GROUPS["A_contact_lr_1e-4"] / "checkpoints" / "loop_4500.pt",
        "B_4500": experiment / GROUPS["B_contact_lr_2p5e-5"] / "checkpoints" / "loop_4500.pt",
    }
    loops = [4000]
    for group, relative in GROUPS.items():
        directory = experiment / relative / "checkpoints"
        found = sorted(
            (int(path.stem.split("_")[-1]), path)
            for path in directory.glob("loop_*.pt")
            if int(path.stem.split("_")[-1]) > 4000
        )
        loops.extend(loop for loop, _ in found)
    loops = sorted(set(loops))

    metrics = {
        "config": args.config,
        "experiment_dir": str(experiment),
        "checkpoint_start": str(start),
        "checkpoint_start_sha256": _sha256(start),
        "loops": loops,
        "groups": {},
        "endpoints": {},
        "physics_protocol": {
            "old": "own_mapped_t_zeta: fixed t/zeta, r=R_c+zeta(R_s-R_c); "
                   "this is not a fixed physical point set",
            "new": "fixed_physical_common: one saved data-derived (t,r) set, "
                   "mapped through each model's R_c; common valid points only",
            "aggregation": "PDE RMS pooled by point count over common points; "
                           "interface on each model's own R_c",
        },
    }
    for group, relative in GROUPS.items():
        metrics["groups"][group] = {}
        for loop in loops:
            path = (
                start if loop == 4000
                else experiment / relative / "checkpoints" / f"loop_{loop}.pt"
            )
            if not path.exists():
                metrics["groups"][group][str(loop)] = {"missing": True}
                continue
            trainer.load(path)
            metrics["groups"][group][str(loop)] = {
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
                "train": _reference_metrics(trainer, train_cases, train_points),
                "validation": _reference_metrics(
                    trainer, validation_cases, validation_points
                ),
                "effective_learning_rates": trainer.effective_learning_rates(),
                "loop": int(trainer.loop),
            }

    # Contact references are complete; air predictions are evaluated separately
    # at the common start and both endpoints.
    air_predictions = {}
    for label, path in checkpoints.items():
        trainer.load(path)
        air_predictions[label] = {
            case.case_id: _air_predictions(trainer, case) for case in all_cases
        }

    endpoint_metrics = {}
    for case in all_cases:
        endpoint_metrics[case.case_id] = _air_case_metrics(
            {label: air_predictions[label][case.case_id] for label in LABELS},
            case,
        )

    # Old own-mapped t/zeta physics, kept only as a labelled comparison.
    old_physics = {}
    for label, path in checkpoints.items():
        trainer.load(path)
        old_physics[label] = {
            case.case_id: _old_own_mapping_physics(trainer, case)
            for case in all_cases
        }

    reference = _reference_physical_points(
        trainer, all_cases, int(args.points_per_case)
    )
    np.savez_compressed(
        output / "physical_reference_points.npz", **reference["arrays"]
    )
    (output / "physical_reference_metadata.json").write_text(
        json.dumps(reference["metadata"], indent=2, default=diag._jsonable),
        encoding="utf-8",
    )
    fixed_physics = _fixed_physical_physics(
        trainer, all_cases, checkpoints, reference
    )

    # Own-R_c interface errors at fixed physical times, one per model.
    interface = {}
    initial_condition = {}
    for label, path in checkpoints.items():
        trainer.load(path)
        interface[label] = {
            case.case_id: phys._interface_errors(
                case, trainer.contact_net, trainer.air_net,
                points=int(args.interface_points),
            )
            for case in all_cases
        }
        initial_condition[label] = {
            case.case_id: _initial_condition_metrics(trainer, case)
            for case in all_cases
        }

    for label, path in checkpoints.items():
        entry = {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256(path),
            "train": {},
            "validation": {},
            "air_own": {},
            "air_common_two": {},
            "air_common_three": {},
            "physics_own_mapped": {},
            "physics_fixed_common": {},
            "interface": {},
            "initial_condition": {},
        }
        for case in all_cases:
            case_metrics = endpoint_metrics[case.case_id]
            split = "train" if case.role == "train" else "validation"
            entry[split][case.case_id] = case_metrics
            entry["air_own"][case.case_id] = case_metrics["own"][label]
            entry["air_common_two"][case.case_id] = (
                case_metrics["common_two"] if label in PAIR_LABELS else None
            )
            entry["air_common_three"][case.case_id] = case_metrics["common_three"]
            entry["physics_own_mapped"][case.case_id] = old_physics[label][case.case_id]
            entry["physics_fixed_common"][case.case_id] = fixed_physics[case.case_id]["residuals"][label]
            entry["interface"][case.case_id] = interface[label][case.case_id]
            entry["initial_condition"][case.case_id] = initial_condition[label][case.case_id]
        for split in ("train", "validation"):
            subset = {
                case.case_id: entry[split][case.case_id]
                for case in all_cases if case.role == split
            }
            entry[f"{split}_air_own"] = {
                field: _aggregate_block(
                    subset,
                    lambda item, f=field: item["own"][label][f],
                )
                for field in ("rho", "u", "p")
            }
            entry[f"{split}_air_common_two"] = {
                field: _aggregate_block(
                    subset,
                    lambda item, f=field: item["common_two"][f][label],
                ) if label in PAIR_LABELS else None
                for field in ("rho", "u", "p")
            }
            entry[f"{split}_air_common_three"] = {
                field: _aggregate_block(
                    subset,
                    lambda item, f=field: item["common_three"][f][label],
                )
                for field in ("rho", "u", "p")
            }
            physics_rows = [
                fixed_physics[case.case_id]
                for case in all_cases if case.role == split
            ]
            entry[f"{split}_physics_fixed_common"] = _aggregate_physics_rows(
                physics_rows, label
            )
            entry[f"{split}_physics_own_mapped"] = _aggregate_old_physics(
                [old_physics[label][case.case_id]
                 for case in all_cases if case.role == split]
            )
            entry[f"{split}_interface"] = _aggregate_interface(
                [interface[label][case.case_id]
                 for case in all_cases if case.role == split]
            )
            entry[f"{split}_initial_condition"] = _aggregate_initial(
                [initial_condition[label][case.case_id]
                 for case in all_cases if case.role == split]
            )
        metrics["endpoints"][label] = entry

    metrics["physical_fixed_rows"] = fixed_physics
    metrics["physical_reference"] = {
        "npz": "physical_reference_points.npz",
        "metadata": "physical_reference_metadata.json",
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=diag._jsonable), encoding="utf-8"
    )
    _write_plots(
        output, metrics, trainer, all_cases,
        {
            "original": (4000, start),
            "A lr=1e-4": (4500, checkpoints["A_4500"]),
            "B lr=2.5e-5": (4500, checkpoints["B_4500"]),
        },
    )
    print(f"written: {output}")
    return 0


def _aggregate_old_physics(rows: list[dict]) -> dict:
    keys = (
        "pde_mass_normalised", "pde_momentum_normalised",
        "pde_energy_normalised", "interface_velocity_rmse",
        "interface_pressure_rmse",
    )
    return {
        key: float(np.mean([item[key] for item in rows])) if rows else None
        for key in keys
    }


if __name__ == "__main__":
    raise SystemExit(main())
