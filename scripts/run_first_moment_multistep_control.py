#!/usr/bin/env python3
"""Paired 100-step first-moment reset control for the current C baseline.

For each of C_best and C_last, and for each of two pre-fixed sampling seeds,
two arms start from the same checkpoint, optimizer state and per-step batch
sequence:

A: normal Adam continuation.
B: ContactNet's stored ``exp_avg`` is zeroed once at the start; ``exp_avg_sq``,
   ``step``, betas, eps, lr and all network parameters are retained.  After
   that, standard Adam is used; there is no per-step or periodic reset.

The run is exactly 100 official alternating updates per arm.  It does not
write checkpoints, does not change the formal default, and never uses
validation labels for training, gradients, resets, stopping, extension or
model selection.

Evaluation is on fixed label/physics point sets and consumes no training RNG.
The primary comparison is the fixed 100-step endpoint; no best-step checkpoint
is selected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import contact_fullcontact_pair as cf  # noqa: E402
import diagnose_contact_gradients as diag  # noqa: E402
import physical_residuals as phys  # noqa: E402
from spherical_blast_pinn.contact_air.cli import _networks  # noqa: E402

C_CONFIG = (
    "configs/contact_air/"
    "split_train_r50_500_fullframes_similarity.yaml"
)
CHECKPOINTS = {
    "best": Path("outputs/contact_air_time_input_pair_c_none_4000/best.pt"),
    "last": Path("outputs/contact_air_time_input_pair_c_none_4000/last.pt"),
}
SAMPLE_SEEDS = (20261115, 20261116)
BATCH_SEED_STRIDE = 104729
EVAL_STEPS = (0, 10, 25, 50, 100)
DEFAULT_STEPS = 100
PHYSICS_POINTS = 1024
INTERFACE_POINTS = 128


def _sha256(path: Path) -> str:
    return diag._sha256(Path(path))


def _state_hash(module) -> str:
    return cf._state_hash(module)


def _optimizer_hash(optimizer) -> str:
    return cf._optimizer_hash(optimizer)


def _batch_digest(batches) -> str:
    hasher = hashlib.sha256()
    for case, batch in batches:
        hasher.update(case.case_id.encode("utf-8"))
        for name in (
            "t_pde", "zeta_pde", "t_ic", "zeta_ic", "t_right", "t_data",
            "radius_data", "rho_data", "u_data", "p_data", "valid_rho",
            "valid_u", "valid_p", "t_contact", "low_contact", "up_contact",
        ):
            value = getattr(batch, name)
            hasher.update(name.encode("utf-8"))
            hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _state_dict(net) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in net.state_dict().items()
    }


def _clone_metrics(metrics):
    return json.loads(json.dumps(metrics, default=diag._jsonable))


def _log(log_path: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _full_points(cases) -> list[dict]:
    points = []
    for case in cases:
        labels = cf._full_labels(case)
        if int(labels["n"]) <= 0:
            raise RuntimeError(
                f"case {case.case_id}: no legal contact labels for evaluation"
            )
        points.append({
            "case_id": case.case_id,
            "times": labels["times"],
            "low": labels["low"],
            "up": labels["up"],
            "t0": float(case.t0),
            "t_end": float(case.t_end),
            "role": case.role,
        })
    return points


def _case_map(trainer) -> dict:
    return {
        case.case_id: case
        for case in list(trainer.cases) + list(trainer.validation_cases)
    }


def _contact_metrics_for(trainer, points, case_map) -> dict:
    cases = []
    for point in points:
        case = case_map[point["case_id"]]
        metrics = diag._contact_metrics(
            trainer, case, point["times"], point["low"], point["up"]
        )
        masks = diag._third_masks(
            point["times"], point["t0"], point["t_end"]
        )
        thirds = [
            diag._contact_metrics(
                trainer, case, point["times"][mask],
                point["low"][mask], point["up"][mask],
            )
            for mask in masks
        ]
        cases.append({
            "case_id": case.case_id,
            "role": case.role,
            **metrics,
            "thirds": thirds,
        })
    expected = len(points)
    aggregate = diag._aggregate_metrics(cases, expected_cases=expected)
    segments = [
        diag._aggregate_metrics(
            [
                {"case_id": row["case_id"], **row["thirds"][which]}
                for row in cases
            ],
            expected_cases=expected,
        )
        for which in range(3)
    ]
    return {
        **aggregate,
        "cases": cases,
        "thirds": segments,
    }


def _contact_macro(trainer, points, case_map) -> dict:
    cases = []
    for point in points:
        case = case_map[point["case_id"]]
        metrics = diag._contact_metrics(
            trainer, case, point["times"], point["low"], point["up"]
        )
        cases.append({"case_id": case.case_id, **metrics})
    aggregate = diag._aggregate_metrics(
        cases, expected_cases=len(points)
    )
    return {
        "n": aggregate["n"],
        "macro_mse": aggregate["macro_mse"],
        "pooled_mse": aggregate["pooled_mse"],
        "coverage": aggregate["coverage"],
        "mean_violation_over_Rsep_pooled": aggregate[
            "mean_violation_over_Rsep_pooled"
        ],
    }


def _observation_subset(case, limit: int) -> dict:
    obs = case.observations
    mask = (
        np.asarray(obs.valid_density, dtype=bool)
        & np.asarray(obs.valid_velocity, dtype=bool)
        & np.asarray(obs.valid_pressure, dtype=bool)
        & np.isfinite(obs.density)
        & np.isfinite(obs.velocity)
        & np.isfinite(obs.pressure)
        & (obs.time >= float(case.t0))
        & (obs.time <= float(case.t_end))
    )
    index = np.flatnonzero(mask)
    if index.size > limit:
        pick = np.linspace(0, index.size - 1, limit).astype(np.int64)
        index = index[np.unique(pick)]
    return {
        "t": torch.as_tensor(obs.time[index], dtype=torch.float64),
        "r": torch.as_tensor(obs.radius[index], dtype=torch.float64),
        "rho": torch.as_tensor(obs.density[index], dtype=torch.float64),
        "u": torch.as_tensor(obs.velocity[index], dtype=torch.float64),
        "p": torch.as_tensor(obs.pressure[index], dtype=torch.float64),
    }


def _physics_reference(cases, contact_a, air_a, contact_b, air_b, limit):
    rows = {}
    for case in cases:
        data = _observation_subset(case, limit)
        inside_a, *_ = phys._domain(case, contact_a, data["t"], data["r"])
        inside_b, *_ = phys._domain(case, contact_b, data["t"], data["r"])
        fixed = inside_a & inside_b
        rows[case.case_id] = {
            "case": case,
            **data,
            "fixed_common": fixed,
            "n_fixed_common": int(fixed.sum()),
            "n_candidates": int(data["t"].numel()),
        }
    return rows


def _air_data_errors(case, air, row, keep, rc, rs, ds, scales) -> dict:
    n = int(keep.sum())
    if n == 0:
        return {
            "n": 0, "n_nonfinite": 0,
            "rho": None, "u": None, "p": None,
        }
    t = row["t"][keep]
    r = row["r"]
    length = (rs - rc)[keep]
    if not bool(torch.all(length > 0.0)):
        raise RuntimeError(f"{case.case_id}: non-positive layer thickness")
    zeta = ((r - rc)[keep] / length).reshape(-1, 1)
    fields = air(
        zeta, t.reshape(-1, 1),
        torch.full_like(t, float(case.charge_radius_m)).reshape(-1, 1),
        scales, time_input=case.time_input,
    )
    values = {
        "rho": (fields.density.reshape(-1) - row["rho"][keep])
        / float(case.rho_scale),
        "u": (fields.velocity.reshape(-1) - row["u"][keep])
        / float(case.u_scale),
        "p": (fields.pressure.reshape(-1) - row["p"][keep])
        / float(case.p_scale),
    }
    finite = torch.ones_like(values["rho"], dtype=torch.bool)
    for value in values.values():
        finite &= torch.isfinite(value)
    out = {
        "n": int(finite.sum()),
        "n_nonfinite": int(n - finite.sum()),
    }
    for name, value in values.items():
        out[name] = (
            float(torch.sqrt(torch.mean(value[finite] * value[finite])).detach())
            if int(finite.sum()) else None
        )
    return out


def _pool_rms(rows, getter, count_getter) -> float | None:
    total = sum(int(count_getter(row)) for row in rows)
    if total == 0:
        return None
    values = [
        (float(getter(row)), int(count_getter(row)))
        for row in rows
        if getter(row) is not None and int(count_getter(row)) > 0
    ]
    if not values:
        return None
    return math.sqrt(sum(value * value * n for value, n in values) / total)


def _physics_pair_metrics(
    contact_a, air_a, contact_b, air_b, refs, interface_points: int,
) -> dict:
    rows = []
    for case_id, ref in refs.items():
        case = ref["case"]
        data = ref
        inside_a, rc_a, rcd_a, rs, ds, _, scales = phys._domain(
            case, contact_a, data["t"], data["r"]
        )
        inside_b, rc_b, rcd_b, _, _, _, _ = phys._domain(
            case, contact_b, data["t"], data["r"]
        )
        keep = data["fixed_common"] & inside_a & inside_b
        res_a = phys._residual_rms(
            case, air_a, data["t"], data["r"], rc_a, rcd_a, rs, ds,
            scales, keep,
        )
        res_b = phys._residual_rms(
            case, air_b, data["t"], data["r"], rc_b, rcd_b, rs, ds,
            scales, keep,
        )
        air_a_errors = _air_data_errors(
            case, air_a, data, keep, rc_a, rs, ds, scales
        )
        air_b_errors = _air_data_errors(
            case, air_b, data, keep, rc_b, rs, ds, scales
        )
        inter_a = phys._interface_errors(
            case, contact_a, air_a, points=interface_points
        )
        inter_b = phys._interface_errors(
            case, contact_b, air_b, points=interface_points
        )
        rows.append({
            "case_id": case_id,
            "role": case.role,
            "n_candidates": ref["n_candidates"],
            "n_fixed_common": ref["n_fixed_common"],
            "n_common_now": int(keep.sum()),
            "pde_a": res_a,
            "pde_b": res_b,
            "air_data_a": air_a_errors,
            "air_data_b": air_b_errors,
            "interface_a": inter_a,
            "interface_b": inter_b,
        })

    def pde_pool(key, model):
        return _pool_rms(
            rows,
            lambda row: row[f"pde_{model}"][key],
            lambda row: row[f"pde_{model}"]["n"],
        )

    def data_pool(key, model):
        return _pool_rms(
            rows,
            lambda row: row[f"air_data_{model}"][key],
            lambda row: row[f"air_data_{model}"]["n"],
        )

    def interface_pool(key, model):
        return _pool_rms(
            rows,
            lambda row: row[f"interface_{model}"][key],
            lambda row: row[f"interface_{model}"]["n"],
        )

    return {
        "n_candidates": int(sum(row["n_candidates"] for row in rows)),
        "n_fixed_common": int(sum(row["n_fixed_common"] for row in rows)),
        "n_common_now": int(sum(row["n_common_now"] for row in rows)),
        "a": {
            "pde": {
                key: pde_pool(key, "a")
                for key in ("mass", "momentum", "energy")
            },
            "air_data": {
                key: data_pool(key, "a")
                for key in ("rho", "u", "p")
            },
            "interface": {
                key: interface_pool(key, "a")
                for key in ("velocity_rms", "pressure_rms")
            },
        },
        "b": {
            "pde": {
                key: pde_pool(key, "b")
                for key in ("mass", "momentum", "energy")
            },
            "air_data": {
                key: data_pool(key, "b")
                for key in ("rho", "u", "p")
            },
            "interface": {
                key: interface_pool(key, "b")
                for key in ("velocity_rms", "pressure_rms")
            },
        },
        "cases": rows,
    }


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _step_record(trainer, batches, sample_seed, step, train_points, case_map):
    air_total, lambda_pde, lambda_data = trainer._air_update(batches)
    contact_total = trainer.contact_update(
        batches, lambda_pde=lambda_pde, lambda_data=lambda_data
    )
    trainer.loop += 1
    macro = _contact_macro(trainer, train_points, case_map)
    exp_avg_norms = [
        float(state["exp_avg"].norm())
        for state in trainer.opt_contact.state.values()
        if "exp_avg" in state
    ]
    record = {
        "step": int(step),
        "loop": int(trainer.loop),
        "sample_seed": int(sample_seed),
        "batch_sha256": _batch_digest(batches),
        "air_loss": float(air_total.total.detach()),
        "contact_loss": float(contact_total.total.detach()),
        "lambda_pde": float(lambda_pde),
        "lambda_data": float(lambda_data),
        "air_parts": dict(air_total.parts),
        "contact_parts": dict(contact_total.parts),
        "contact_learning_rate": float(
            trainer.opt_contact.param_groups[0]["lr"]
        ),
        "air_learning_rate": float(
            trainer.opt_air.param_groups[0]["lr"]
        ),
        "exp_avg_norm_max": max(exp_avg_norms) if exp_avg_norms else 0.0,
        "contact_optimizer_steps": diag._adam_summary(
            trainer.opt_contact
        )["step_values"],
        "train_contact_macro": macro,
    }
    trainer.history.loops.append(record)
    if not _finite_tree(record):
        raise RuntimeError(f"non-finite step record at step {step}")
    if not all(
        bool(torch.isfinite(parameter).all())
        for parameter in trainer.contact_net.parameters()
    ) or not all(
        bool(torch.isfinite(parameter).all())
        for parameter in trainer.air_net.parameters()
    ):
        raise RuntimeError(f"non-finite network parameter at step {step}")
    return record


def _zero_first_moment_once(trainer):
    params = cf._contact_parameters(trainer)
    before = cf._adam_snapshot(trainer.opt_contact, params)
    for parameter in params:
        state = trainer.opt_contact.state.get(parameter)
        if not isinstance(state, dict) or "exp_avg" not in state:
            raise RuntimeError("ContactNet Adam state is missing exp_avg")
        state["exp_avg"].zero_()
    after = cf._adam_snapshot(trainer.opt_contact, params)
    return before, after


def _snapshot_diff(first, second, key):
    differences = [
        float((a[key] - b[key]).abs().max())
        for a, b in zip(first, second)
    ]
    return max(differences) if differences else 0.0


def _run_arm(
    trainer, checkpoint, arm, sample_seed, steps, eval_steps,
    train_points, val_points, case_map, log_path,
):
    trainer.load(checkpoint)
    start_loop = int(trainer.loop)
    initial_contact = _state_hash(trainer.contact_net)
    initial_air = _state_hash(trainer.air_net)
    initial_opt = _optimizer_hash(trainer.opt_contact)
    initial_air_opt = _optimizer_hash(trainer.opt_air)
    params = cf._contact_parameters(trainer)
    snapshots_before = cf._adam_snapshot(trainer.opt_contact, params)
    zero_events = 0
    zero_before = zero_after = None
    if arm == "B":
        zero_before, zero_after = _zero_first_moment_once(trainer)
        zero_events = 1
    zero_evidence = {
        "zero_events": zero_events,
        "exp_avg_norm_before": (
            max((float(item["exp_avg"].norm())
                 for item in snapshots_before), default=0.0)
        ),
        "exp_avg_norm_after": (
            max((float(item["exp_avg"].norm())
                 for item in zero_after), default=0.0)
            if zero_after is not None else None
        ),
        "exp_avg_sq_before": (
            _snapshot_diff(snapshots_before, zero_before, "exp_avg_sq")
            if zero_before is not None else None
        ),
        "step_before_equal": (
            sorted(item["step"] for item in snapshots_before)
            == sorted(item["step"] for item in zero_before)
            if zero_before is not None else None
        ),
        "betas": list(trainer.opt_contact.param_groups[0]["betas"]),
        "eps": float(trainer.opt_contact.param_groups[0]["eps"]),
        "lr": float(trainer.opt_contact.param_groups[0]["lr"]),
    }
    steps_out = []
    evaluations = {}
    batch_hashes = []
    for step in range(1, int(steps) + 1):
        sample = int(sample_seed) + (step - 1) * BATCH_SEED_STRIDE
        generator = torch.Generator().manual_seed(sample)
        batches = trainer._batches(generator=generator)
        record = _step_record(
            trainer, batches, sample, step, train_points, case_map
        )
        if step > 1 and record["exp_avg_norm_max"] == 0.0:
            raise RuntimeError(
                f"{arm}: exp_avg is zero at step {step}; a second reset occurred"
            )
        batch_hashes.append(record["batch_sha256"])
        steps_out.append(record)
        if step in eval_steps and step > 0:
            train_full = _contact_metrics_for(
                trainer, train_points, case_map
            )
            evaluation = {"step": step, "train_contact": train_full}
            if step == int(steps):
                evaluation["validation_contact"] = _contact_metrics_for(
                    trainer, val_points, case_map
                )
            evaluations[str(step)] = evaluation
            _log(
                log_path,
                f"{arm} seed={sample_seed} step={step} "
                f"train_macro={train_full['macro_mse']:.6e}",
            )
    end_loop = int(trainer.loop)
    endpoint = {
        "loop": end_loop,
        "contact_state": _state_hash(trainer.contact_net),
        "air_state": _state_hash(trainer.air_net),
        "contact_optimizer": _optimizer_hash(trainer.opt_contact),
        "air_optimizer": _optimizer_hash(trainer.opt_air),
        "contact_state_dict": _state_dict(trainer.contact_net),
        "air_state_dict": _state_dict(trainer.air_net),
        "contact_step_values": diag._adam_summary(
            trainer.opt_contact
        )["step_values"],
        "air_step_values": diag._adam_summary(
            trainer.opt_air
        )["step_values"],
    }
    _log(
        log_path,
        f"{arm} seed={sample_seed} finished loop {end_loop}",
    )
    return {
        "arm": arm,
        "start_loop": start_loop,
        "end_loop": end_loop,
        "initial_contact_state": initial_contact,
        "initial_air_state": initial_air,
        "initial_contact_optimizer": initial_opt,
        "initial_air_optimizer": initial_air_opt,
        "zero_evidence": zero_evidence,
        "steps": steps_out,
        "evaluations": evaluations,
        "batch_hashes": batch_hashes,
        "endpoint": {
            key: value for key, value in endpoint.items()
            if not key.endswith("_state_dict")
        },
        "_endpoint_state": endpoint,
    }


def _pair_summary(pair, steps) -> dict:
    a_end = pair["A"]["evaluations"][str(steps)]["train_contact"]
    b_end = pair["B"]["evaluations"][str(steps)]["train_contact"]
    a_val = pair["A"]["evaluations"][str(steps)]["validation_contact"]
    b_val = pair["B"]["evaluations"][str(steps)]["validation_contact"]
    return {
        "endpoint_step": int(steps),
        "selection": (
            f"fixed {int(steps)}-step endpoint; no best-step selection"
        ),
        "train_contact": {
            "A": {
                "macro_mse": a_end["macro_mse"],
                "pooled_mse": a_end["pooled_mse"],
                "coverage": a_end["coverage"],
                "mean_violation_over_Rsep_pooled": a_end[
                    "mean_violation_over_Rsep_pooled"
                ],
            },
            "B": {
                "macro_mse": b_end["macro_mse"],
                "pooled_mse": b_end["pooled_mse"],
                "coverage": b_end["coverage"],
                "mean_violation_over_Rsep_pooled": b_end[
                    "mean_violation_over_Rsep_pooled"
                ],
            },
        },
        "validation_contact": {
            "A": {
                "macro_mse": a_val["macro_mse"],
                "pooled_mse": a_val["pooled_mse"],
                "coverage": a_val["coverage"],
                "mean_violation_over_Rsep_pooled": a_val[
                    "mean_violation_over_Rsep_pooled"
                ],
            },
            "B": {
                "macro_mse": b_val["macro_mse"],
                "pooled_mse": b_val["pooled_mse"],
                "coverage": b_val["coverage"],
                "mean_violation_over_Rsep_pooled": b_val[
                    "mean_violation_over_Rsep_pooled"
                ],
            },
        },
    }


def _write_curves(path: Path, pairs) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "checkpoint", "sample_seed", "arm", "step", "loop",
            "sample_seed_step", "train_contact_macro_mse",
            "contact_loss", "air_loss", "exp_avg_norm_max",
        ])
        for pair in pairs:
            for arm in ("A", "B"):
                for row in pair[arm]["steps"]:
                    writer.writerow([
                        pair["checkpoint"], pair["sample_seed"], arm,
                        row["step"], row["loop"], row["sample_seed"],
                        row["train_contact_macro"]["macro_mse"],
                        row["contact_loss"], row["air_loss"],
                        row["exp_avg_norm_max"],
                    ])


def _status_reasons(payload) -> list[str]:
    reasons = []
    for pair in payload.get("pairs", []):
        if pair.get("status") != "ok":
            reasons.append(
                f"{pair.get('checkpoint')}/{pair.get('sample_seed')}: "
                f"{pair.get('error', 'pair status failed')}"
            )
    if payload.get("checkpoint_hashes", {}).get("unchanged") is not True:
        reasons.append("a source checkpoint SHA-256 changed")
    return reasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--checkpoints", default="best,last")
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in SAMPLE_SEEDS)
    )
    parser.add_argument("--physics-points", type=int, default=PHYSICS_POINTS)
    parser.add_argument("--interface-points", type=int, default=INTERFACE_POINTS)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log_path = output / "run.log"
    log_path.write_text("", encoding="utf-8")
    _log(log_path, f"start; steps={args.steps}")

    checkpoints = [
        item.strip() for item in args.checkpoints.split(",") if item.strip()
    ]
    unknown = [item for item in checkpoints if item not in CHECKPOINTS]
    if unknown:
        parser.error(f"unknown checkpoints: {unknown}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        parser.error("at least one sampling seed is required")
    steps = int(args.steps)
    if steps <= 0:
        parser.error("--steps must be positive")
    eval_steps = [step for step in EVAL_STEPS if step <= steps]
    if steps not in eval_steps:
        eval_steps.append(steps)
    eval_steps = sorted(set(eval_steps))

    checkpoint_before = {
        tag: _sha256(CHECKPOINTS[tag]) for tag in checkpoints
    }
    payload = {
        "format": "spherical_blast_pinn.first_moment_multistep_control",
        "version": 1,
        "config": C_CONFIG,
        "config_sha256": _sha256(Path(C_CONFIG)),
        "steps": steps,
        "eval_steps": eval_steps,
        "sample_seeds": seeds,
        "checkpoints": {
            tag: str(CHECKPOINTS[tag]) for tag in checkpoints
        },
        "checkpoint_hashes": {
            "before": checkpoint_before,
            "after": {},
            "unchanged": None,
        },
        "protocol": {
            "A": "normal ContactNet Adam continuation",
            "B": "ContactNet exp_avg zeroed once at start; exp_avg_sq, step, "
                 "betas, eps, lr and parameters retained; standard Adam after",
            "no_periodic_reset": True,
            "batch_seed_rule": (
                f"sample_seed + (step - 1) * {BATCH_SEED_STRIDE}"
            ),
            "evaluation_rng": (
                "fixed full-label sets and deterministic physics points; "
                "evaluation consumes no training RNG"
            ),
            "validation_used_for_decisions": False,
            "primary_comparison": "fixed endpoint at the requested final step",
            "selection": "none",
        },
        "pairs": [],
        "status": None,
        "failure_reasons": [],
    }

    trainer, meta = diag._build_trainer(C_CONFIG)
    train_points = _full_points(trainer.cases)
    val_points = _full_points(trainer.validation_cases)
    case_map = _case_map(trainer)
    _log(
        log_path,
        f"reference labels: train={sum(len(p['times']) for p in train_points)} "
        f"validation={sum(len(p['times']) for p in val_points)}",
    )

    try:
        for tag in checkpoints:
            checkpoint = CHECKPOINTS[tag]
            _log(log_path, f"checkpoint {tag}: {checkpoint}")
            for sample_seed in seeds:
                pair = {
                    "checkpoint": tag,
                    "sample_seed": int(sample_seed),
                    "status": "running",
                }
                _log(
                    log_path,
                    f"pair {tag}/{sample_seed}: start",
                )
                trainer.load(checkpoint)
                start_contact = _contact_metrics_for(
                    trainer, train_points, case_map
                )
                start_validation = _contact_metrics_for(
                    trainer, val_points, case_map
                )
                start_contact_state = _state_hash(trainer.contact_net)
                start_air_state = _state_hash(trainer.air_net)
                start_optimizer = _optimizer_hash(trainer.opt_contact)
                train_ref = _physics_reference(
                    trainer.cases, trainer.contact_net, trainer.air_net,
                    trainer.contact_net, trainer.air_net,
                    int(args.physics_points),
                )
                val_ref = _physics_reference(
                    trainer.validation_cases, trainer.contact_net,
                    trainer.air_net, trainer.contact_net, trainer.air_net,
                    int(args.physics_points),
                )
                start_physics = {
                    "train": _physics_pair_metrics(
                        trainer.contact_net, trainer.air_net,
                        trainer.contact_net, trainer.air_net,
                        train_ref, int(args.interface_points),
                    ),
                    "validation": _physics_pair_metrics(
                        trainer.contact_net, trainer.air_net,
                        trainer.contact_net, trainer.air_net,
                        val_ref, int(args.interface_points),
                    ),
                }
                pair["start"] = {
                    "loop": int(trainer.loop),
                    "contact_state": start_contact_state,
                    "air_state": start_air_state,
                    "contact_optimizer": start_optimizer,
                    "train_contact": start_contact,
                    "validation_contact": start_validation,
                    "physics": start_physics,
                }
                # A first, then B from the same checkpoint.
                arm_a = _run_arm(
                    trainer, checkpoint, "A", sample_seed, steps,
                    eval_steps, train_points, val_points, case_map,
                    log_path,
                )
                state_a = arm_a.pop("_endpoint_state")
                arm_b = _run_arm(
                    trainer, checkpoint, "B", sample_seed, steps,
                    eval_steps, train_points, val_points, case_map,
                    log_path,
                )
                state_b = arm_b.pop("_endpoint_state")
                pair["A"] = arm_a
                pair["B"] = arm_b
                pair["batch_sequence_equal"] = (
                    arm_a["batch_hashes"] == arm_b["batch_hashes"]
                )
                pair["start_state_equal"] = (
                    arm_a["initial_contact_state"]
                    == arm_b["initial_contact_state"]
                    == start_contact_state
                    and arm_a["initial_air_state"]
                    == arm_b["initial_air_state"]
                    == start_air_state
                    and arm_a["initial_contact_optimizer"]
                    == arm_b["initial_contact_optimizer"]
                    == start_optimizer
                )
                pair["zero_evidence"] = arm_b["zero_evidence"]
                if not pair["batch_sequence_equal"]:
                    raise RuntimeError(
                        f"{tag}/{sample_seed}: A/B batch sequence differs"
                    )
                if not pair["start_state_equal"]:
                    raise RuntimeError(
                        f"{tag}/{sample_seed}: A/B start state differs"
                    )
                if arm_b["zero_evidence"]["zero_events"] != 1:
                    raise RuntimeError(
                        f"{tag}/{sample_seed}: B did not have exactly one reset"
                    )
                if arm_b["zero_evidence"]["exp_avg_norm_after"] != 0.0:
                    raise RuntimeError(
                        f"{tag}/{sample_seed}: B exp_avg was not zero"
                    )
                if arm_b["zero_evidence"]["exp_avg_sq_before"] != 0.0:
                    raise RuntimeError(
                        f"{tag}/{sample_seed}: B exp_avg_sq changed at reset"
                    )
                if not arm_b["zero_evidence"]["step_before_equal"]:
                    raise RuntimeError(
                        f"{tag}/{sample_seed}: B step changed at reset"
                    )

                # Endpoint physics is evaluated jointly on fixed common points.
                contact_a, air_a = _networks(meta["section"])
                contact_a.load_state_dict(state_a["contact_state_dict"])
                air_a.load_state_dict(state_a["air_state_dict"])
                contact_b, air_b = _networks(meta["section"])
                contact_b.load_state_dict(state_b["contact_state_dict"])
                air_b.load_state_dict(state_b["air_state_dict"])
                end_physics = {
                    "train": _physics_pair_metrics(
                        contact_a, air_a, contact_b, air_b, train_ref,
                        int(args.interface_points),
                    ),
                    "validation": _physics_pair_metrics(
                        contact_a, air_a, contact_b, air_b, val_ref,
                        int(args.interface_points),
                    ),
                }
                pair["endpoint_physics"] = end_physics
                pair["summary"] = _pair_summary(pair, steps)
                pair["checks"] = {
                    "batch_sequence_equal": pair["batch_sequence_equal"],
                    "start_state_equal": pair["start_state_equal"],
                    "one_reset_only": (
                        pair["zero_evidence"]["zero_events"] == 1
                    ),
                    "exp_avg_zero_after_reset": (
                        pair["zero_evidence"]["exp_avg_norm_after"] == 0.0
                    ),
                    "exp_avg_sq_preserved_at_reset": (
                        pair["zero_evidence"]["exp_avg_sq_before"] == 0.0
                    ),
                    "step_preserved_at_reset": bool(
                        pair["zero_evidence"]["step_before_equal"]
                    ),
                    "A_loop_advanced_requested": (
                        pair["A"]["end_loop"] - pair["A"]["start_loop"]
                        == steps
                    ),
                    "B_loop_advanced_requested": (
                        pair["B"]["end_loop"] - pair["B"]["start_loop"]
                        == steps
                    ),
                    "A_steps_recorded": len(pair["A"]["steps"]) == steps,
                    "B_steps_recorded": len(pair["B"]["steps"]) == steps,
                    "start_physics_finite": _finite_tree(start_physics),
                    "physics_finite": _finite_tree(end_physics),
                    "contact_finite": _finite_tree(
                        pair["summary"]
                    ) and _finite_tree(start_contact)
                    and _finite_tree(start_validation),
                }
                pair["status"] = (
                    "ok" if all(pair["checks"].values()) else "failed"
                )
                _log(
                    log_path,
                    f"pair {tag}/{sample_seed}: status={pair['status']} "
                    f"train C A/B={pair['summary']['train_contact']['A']['macro_mse']:.6e}/"
                    f"{pair['summary']['train_contact']['B']['macro_mse']:.6e} "
                    f"val C A/B={pair['summary']['validation_contact']['A']['macro_mse']:.6e}/"
                    f"{pair['summary']['validation_contact']['B']['macro_mse']:.6e}",
                )
                payload["pairs"].append(pair)
                if pair["status"] != "ok":
                    raise RuntimeError(
                        f"{tag}/{sample_seed}: pair checks failed"
                    )
        payload["status"] = "ok"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        _log(log_path, f"FAILED: {payload['error']}")

    checkpoint_after = {
        tag: _sha256(CHECKPOINTS[tag]) for tag in checkpoints
    }
    payload["checkpoint_hashes"]["after"] = checkpoint_after
    payload["checkpoint_hashes"]["unchanged"] = (
        checkpoint_before == checkpoint_after
    )
    if not payload["checkpoint_hashes"]["unchanged"]:
        payload["status"] = "failed"
        payload["error"] = "source checkpoint SHA-256 changed"
    payload["failure_reasons"] = _status_reasons(payload)
    if payload.get("error") and payload["error"] not in payload["failure_reasons"]:
        payload["failure_reasons"].append(payload["error"])
    if payload["failure_reasons"]:
        payload["status"] = "failed"
    payload["elapsed_s"] = time.perf_counter() - started
    source_paths = [
        Path(__file__).resolve(),
        _REPO_ROOT / "scripts" / "diagnose_current_contact_update_decomposition.py",
        _REPO_ROOT / "scripts" / "diagnose_current_contact_first_moment_control.py",
        _REPO_ROOT / "scripts" / "diagnose_current_contact_step.py",
        _REPO_ROOT / "tests" / "test_current_contact_status.py",
        Path(C_CONFIG).resolve(),
    ]
    payload["provenance"] = {
        "source_files": {
            str(path.relative_to(_REPO_ROOT)): _sha256(path)
            for path in source_paths if path.exists()
        },
        "config_path": C_CONFIG,
        "config_sha256": _sha256(Path(C_CONFIG)),
        "config": meta["config"].to_dict(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "run_log": str(log_path),
        "physics_points_per_case": int(args.physics_points),
        "interface_points": int(args.interface_points),
    }
    control_path = output / "control.json"
    control_path.write_text(
        json.dumps(payload, indent=2, default=diag._jsonable),
        encoding="utf-8",
    )
    _write_curves(output / "curves.csv", payload["pairs"])
    _log(
        log_path,
        f"written {control_path}; status={payload['status']}",
    )
    print(json.dumps({
        "wrote": str(control_path),
        "status": payload["status"],
        "failure_reasons": payload["failure_reasons"],
        "pairs": len(payload["pairs"]),
    }, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
