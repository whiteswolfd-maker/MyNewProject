#!/usr/bin/env python3
"""Evaluate the ContactNet learning-rate resume experiment.

Reads the per-50-loop checkpoints produced by the two independent resume runs.
Contact metrics use every legal contact label (per case mean, then case-equal
mean), exact integer coverage and fixed early/middle/late bins.  Air metrics are
reported on each model's own valid points and on the common valid points of the
A/B endpoints.  Physics residuals use a fixed t/zeta grid per case.

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
from spherical_blast_pinn.contact_air.losses import (  # noqa: E402
    data_coordinates, pde_loss,
)
from spherical_blast_pinn.contact_air.mapping import build_geometry  # noqa: E402
from spherical_blast_pinn.contact_air.models import (  # noqa: E402
    air_fields_with_derivatives,
)

GROUPS = {
    "A_contact_lr_1e-4": "contact_lr_1e-4_resume4000_to4500",
    "B_contact_lr_2p5e-5": "contact_lr_2p5e-5_resume4000_to4500",
}
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
    contact = diag._detach_contact(contact) if hasattr(diag, "_detach_contact") else contact
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


def _field_rmse(pred, target, mask, scale):
    use = mask & np.isfinite(pred) & np.isfinite(target)
    n = int(use.sum())
    if n == 0:
        return {"n": 0, "rmse": None}
    error = (pred[use] - target[use]) / float(scale)
    return {"n": n, "rmse": float(np.sqrt(np.mean(error * error)))}


def _air_case_metrics(model: dict, case, *, common: dict | None = None,
                      label: str | None = None,
                      common_label: str | None = None) -> dict:
    scales = {"rho": case.rho_scale, "u": case.u_scale, "p": case.p_scale}
    own = {}
    shared = {}
    for name in ("rho", "u", "p"):
        base = model["valid"][name] & model["inside"]
        own[name] = _field_rmse(
            model["pred"][name], model["target"][name], base, scales[name]
        )
        if common is not None:
            mask = (
                base & common["inside"]
                & np.isfinite(model["pred"][name])
                & np.isfinite(common["pred"][name])
            )
            shared[name] = {
                str(label): _field_rmse(
                    model["pred"][name], model["target"][name], mask, scales[name]
                ),
                str(common_label): _field_rmse(
                    common["pred"][name], common["target"][name], mask,
                    scales[name],
                ),
            }
    return {
        "case_id": case.case_id,
        "n_total": model["n_total"],
        "n_time_valid": model["n_time_valid"],
        "n_inside": int(model["inside"].sum()),
        "n_outside": int(model["n_total"] - model["inside"].sum()),
        "inside_fraction": (
            float(model["inside"].sum() / model["n_total"])
            if model["n_total"] else None
        ),
        "own": own,
        "common": shared if common is not None else None,
    }


def _aggregate_air(per_case: dict) -> dict:
    fields = ("rho", "u", "p")
    result = {}
    for name in fields:
        values = [
            item["own"][name]["rmse"] for item in per_case.values()
            if item["own"][name]["rmse"] is not None
        ]
        pooled_n = sum(item["own"][name]["n"] for item in per_case.values())
        result[name] = {
            "cases_evaluable": len(values),
            "macro_rmse": (sum(values) / len(values)) if values else None,
            "n_points": int(pooled_n),
        }
    return result


def _aggregate_common(per_case: dict, label: str) -> dict:
    result = {}
    for name in ("rho", "u", "p"):
        values = [
            item["common"][name][label]["rmse"]
            for item in per_case.values()
            if item.get("common") and item["common"][name][label]["rmse"] is not None
        ]
        n_points = sum(
            item["common"][name][label]["n"] for item in per_case.values()
            if item.get("common")
        )
        result[name] = {
            "cases_evaluable": len(values),
            "macro_rmse": (sum(values) / len(values)) if values else None,
            "n_points": int(n_points),
        }
    return result


def _physics_metrics(trainer, case) -> dict:
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
        u_error = (air0.velocity.reshape(-1) - contact.speed.reshape(-1)) / case.u_scale
        p_error = (air0.pressure.reshape(-1) - contact.pressure.reshape(-1)) / case.p_scale
    return {
        "case_id": case.case_id,
        "n_points": int(t.numel()),
        "pde_mass_normalised": pde.parts["pde_mass_normalised"],
        "pde_momentum_normalised": pde.parts["pde_momentum_normalised"],
        "pde_energy_normalised": pde.parts["pde_energy_normalised"],
        "interface_velocity_rmse": float(torch.sqrt(torch.mean(u_error ** 2))),
        "interface_pressure_rmse": float(torch.sqrt(torch.mean(p_error ** 2))),
    }


def _aggregate_physics(per_case: dict) -> dict:
    keys = (
        "pde_mass_normalised", "pde_momentum_normalised",
        "pde_energy_normalised", "interface_velocity_rmse",
        "interface_pressure_rmse",
    )
    return {
        key: float(np.mean([item[key] for item in per_case.values()]))
        for key in keys
    }


def _contact_trajectory(trainer, case):
    times = np.linspace(float(case.t0), float(case.t_end), 200)
    t = torch.as_tensor(times, dtype=torch.float64)
    with torch.enable_grad():
        contact = diag._contact_at(trainer, case, t, create_graph=False)
    return times, contact.radius.detach().numpy().reshape(-1)


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
            grid, radius = _contact_trajectory(trainer, case)
            rc = np.interp(truth_t, grid, radius)
            gap = np.where(rc < low, rc - low, np.where(rc > up, rc - up, 0.0))
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
    args = parser.parse_args(argv)
    experiment = Path(args.experiment_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer, _ = diag._build_trainer(args.config)
    train_cases = list(trainer.cases)
    validation_cases = list(trainer.validation_cases)
    train_points = _points(trainer, train_cases)
    validation_points = _points(trainer, validation_cases)
    start = experiment / "checkpoints" / "loop_4000.pt"
    checkpoints = {"A": start, "B": start}
    loops = [4000]
    for group, relative in GROUPS.items():
        directory = experiment / relative / "checkpoints"
        found = sorted(
            (int(path.stem.split("_")[-1]), path)
            for path in directory.glob("loop_*.pt")
            if int(path.stem.split("_")[-1]) > 4000
        )
        loops.extend(loop for loop, _ in found)
        checkpoints[group] = found[-1][1]
    loops = sorted(set(loops))

    metrics = {
        "config": args.config,
        "experiment_dir": str(experiment),
        "checkpoint_start": str(start),
        "checkpoint_start_sha256": _sha256(start),
        "loops": loops,
        "groups": {},
        "endpoints": {},
    }
    for group, relative in GROUPS.items():
        metrics["groups"][group] = {}
        for loop in loops:
            if loop == 4000:
                path = start
            else:
                path = experiment / relative / "checkpoints" / f"loop_{loop}.pt"
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

    # Air fields at start and both endpoints, including common valid points.
    endpoint_paths = {
        "start_4000": start,
        "A_4500": experiment / GROUPS["A_contact_lr_1e-4"] / "checkpoints" / "loop_4500.pt",
        "B_4500": experiment / GROUPS["B_contact_lr_2p5e-5"] / "checkpoints" / "loop_4500.pt",
    }
    air_predictions = {}
    physics = {}
    for label, path in endpoint_paths.items():
        trainer.load(path)
        air_predictions[label] = {}
        physics[label] = {}
        for case in train_cases + validation_cases:
            air_predictions[label][case.case_id] = _air_predictions(trainer, case)
            physics[label][case.case_id] = _physics_metrics(trainer, case)
    for label, path in endpoint_paths.items():
        entry = {"checkpoint": str(path), "checkpoint_sha256": _sha256(path),
                 "train": {}, "validation": {}, "physics": {}}
        for case in train_cases + validation_cases:
            model = air_predictions[label][case.case_id]
            entry["physics"][case.case_id] = physics[label][case.case_id]
            if label in ("A_4500", "B_4500"):
                other_label = "B_4500" if label == "A_4500" else "A_4500"
                other = air_predictions[other_label][case.case_id]
                case_metrics = _air_case_metrics(
                    model, case, common=other,
                    label=label, common_label=other_label,
                )
            else:
                case_metrics = _air_case_metrics(model, case)
            target_group = "train" if any(
                item.case_id == case.case_id for item in train_cases
            ) else "validation"
            entry[target_group][case.case_id] = case_metrics
        entry["train_aggregate"] = _aggregate_air(entry["train"])
        entry["validation_aggregate"] = _aggregate_air(entry["validation"])
        if label in ("A_4500", "B_4500"):
            entry["train_common_aggregate"] = _aggregate_common(
                entry["train"], label
            )
            entry["validation_common_aggregate"] = _aggregate_common(
                entry["validation"], label
            )
        entry["physics_aggregate_train"] = _aggregate_physics(
            {key: value for key, value in entry["physics"].items()
             if key in {item.case_id for item in train_cases}}
        )
        entry["physics_aggregate_validation"] = _aggregate_physics(
            {key: value for key, value in entry["physics"].items()
             if key in {item.case_id for item in validation_cases}}
        )
        metrics["endpoints"][label] = entry

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=diag._jsonable), encoding="utf-8"
    )
    _write_plots(
        output, metrics, trainer, train_cases + validation_cases,
        {
            "original": (4000, start),
            "A lr=1e-4": (4500, endpoint_paths["A_4500"]),
            "B lr=2.5e-5": (4500, endpoint_paths["B_4500"]),
        },
    )
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
