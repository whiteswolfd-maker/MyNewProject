#!/usr/bin/env python3
"""Read-only diagnosis of ContactNet training behaviour.

Two modes:

* ``training`` - batches use the official 10-case training budget and the
  official per-case sampling distribution/order.  This is the only mode used
  to judge the aggregate update.
* ``stratified`` - a supplementary diagnostic that restricts PDE/interface and
  contact labels to an early/middle/late window.  It never reverts to the full
  window when a slice is empty; such a slice is marked not evaluable.

The script never updates a checkpoint on disk.  It can
(1) compute aggregate, per-term gradients over the mean loss of all ten
training cases and (2) run one real Adam step on an in-memory copy of a
checkpoint, including the AirNet stage, and compare the current batch with a
fixed set of all valid training contact labels before/after that step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from spherical_blast_pinn.config import load_config  # noqa: E402
from spherical_blast_pinn.contact_air.cli import (  # noqa: E402
    _case_specs,
    _load_checkpoint,
    _make_case,
    _networks,
    _section,
)
from spherical_blast_pinn.contact_air.losses import (  # noqa: E402
    schedule_weights,
)
from spherical_blast_pinn.contact_air.trainer import (  # noqa: E402
    AlternatingTrainer,
    _summarise,
    seed_everything,
)
from spherical_blast_pinn.contact_air.cli import _train_config  # noqa: E402

_TERM_NAMES = ("pde", "contact_ic", "interface", "contact_data")

def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _coefficients(trainer: AlternatingTrainer) -> dict[str, float]:
    lambda_pde, lambda_data = schedule_weights(
        trainer.loop, trainer.config.n_schedule
    )
    return {
        "pde": float(lambda_pde),
        "contact_ic": 1.0,
        "interface": 1.0,
        "contact_data": float(lambda_data) * float(
            trainer.config.contact_data_weight
        ),
    }


def _contact_at(trainer, case, times: torch.Tensor, *, create_graph: bool):
    t = times.reshape(-1, 1)
    r0 = torch.full_like(t, float(case.charge_radius_m))
    rs, ds = case.shock_tensors(times.reshape(-1))
    return trainer.contact_net(
        t, r0, rs.reshape(-1, 1).to(torch.float64),
        ds.reshape(-1, 1).to(torch.float64),
        trainer._closures[case.case_id], create_graph=create_graph,
    )


def _cosine(a: torch.Tensor, b: torch.Tensor, threshold: float) -> dict:
    na, nb = float(a.norm()), float(b.norm())
    if not (math.isfinite(na) and math.isfinite(nb)):
        return {"status": "non_finite", "cosine": None, "angle_deg": None}
    if na <= threshold or nb <= threshold:
        return {
            "status": "direction_undefined", "cosine": None, "angle_deg": None,
            "norm_a": na, "norm_b": nb,
        }
    value = float(torch.dot(a, b) / (a.norm() * b.norm()))
    value = max(-1.0, min(1.0, value))
    return {
        "status": "ok", "cosine": value,
        "angle_deg": math.degrees(math.acos(value)),
        "norm_a": na, "norm_b": nb,
    }


def _grad(term: torch.Tensor, params) -> tuple[torch.Tensor, bool]:
    if not term.requires_grad:
        vector = torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
        return vector, True
    grads = torch.autograd.grad(
        term, params, retain_graph=True, allow_unused=True, materialize_grads=True
    )
    vector = torch.cat([g.reshape(-1) for g in grads]) if grads else torch.zeros(0)
    return vector, bool(torch.isfinite(vector).all())


def _range(values: torch.Tensor) -> dict:
    if values.numel() == 0:
        return {"count": 0, "min_s": None, "max_s": None}
    return {
        "count": int(values.numel()),
        "min_s": float(values.min()),
        "max_s": float(values.max()),
    }


def _per_case_terms(trainer, batches) -> list[dict]:
    out = []
    for case, batch in batches:
        terms = trainer.contact_stage_terms(case, batch)
        out.append({
            "case_id": case.case_id,
            "terms": terms,
            "counts": {name: int(terms[name].counts.get(name, 0))
                       for name in _TERM_NAMES},
            "losses": {name: float(terms[name].total.detach())
                       for name in _TERM_NAMES},
            "time_ranges": {
                "pde": _range(batch.t_pde),
                "ic": _range(batch.t_ic),
                "contact": _range(batch.t_contact),
            },
            "contact_evaluable": bool(batch.t_contact.numel()),
        })
    return out


def _aggregate_terms(per_case: list[dict]) -> dict[str, Any]:
    like = torch.zeros((), dtype=torch.float64)
    return {
        name: _summarise([entry["terms"][name] for entry in per_case], like)
        for name in _TERM_NAMES
    }


def _static_record(trainer, batches, *, compute_gradients: bool,
                   threshold: float) -> dict:
    per_case = _per_case_terms(trainer, batches)
    aggregate = _aggregate_terms(per_case)
    coeffs = _coefficients(trainer)
    record: dict[str, Any] = {
        "coefficients": coeffs,
        "terms": {},
        "per_case": [
            {
                "case_id": entry["case_id"],
                "losses": entry["losses"],
                "counts": entry["counts"],
                "time_ranges": entry["time_ranges"],
                "contact_evaluable": entry["contact_evaluable"],
            }
            for entry in per_case
        ],
    }
    vectors: dict[str, torch.Tensor] = {}
    finite_all = True
    if compute_gradients:
        params = [p for p in trainer.contact_net.parameters() if p.requires_grad]
        for name in _TERM_NAMES:
            term = aggregate[name]
            raw_loss = float(term.total.detach())
            weighted_loss = coeffs[name] * term.total
            vector, finite = _grad(weighted_loss, params)
            finite_all = finite_all and finite
            vectors[name] = vector
            record["terms"][name] = {
                "raw_loss": raw_loss,
                "coefficient": coeffs[name],
                "weighted_loss": coeffs[name] * raw_loss,
                "grad_norm_weighted": float(vector.norm()),
                "grad_norm_raw": (
                    float(vector.norm()) / abs(coeffs[name])
                    if coeffs[name] != 0.0 else None
                ),
                "finite_gradient": finite,
                "counts": {k: int(v) for k, v in term.counts.items()},
            }
        g_d = vectors["contact_data"]
        g_o = vectors["pde"] + vectors["contact_ic"] + vectors["interface"]
        g_total = g_d + g_o
        record["aggregate_gradients"] = {
            "g_data_norm": float(g_d.norm()),
            "g_other_norm": float(g_o.norm()),
            "g_total_norm": float(g_total.norm()),
            "cos_data_vs_pde": _cosine(vectors["contact_data"], vectors["pde"], threshold),
            "cos_data_vs_ic": _cosine(vectors["contact_data"], vectors["contact_ic"], threshold),
            "cos_data_vs_interface": _cosine(vectors["contact_data"], vectors["interface"], threshold),
            "cos_data_vs_other": _cosine(g_d, g_o, threshold),
            "cos_data_vs_total": _cosine(g_d, g_total, threshold),
            "finite_all": finite_all,
        }
        record["_vector_g_data"] = g_d
        record["_vectors"] = vectors
    else:
        for name in _TERM_NAMES:
            record["terms"][name] = {
                "raw_loss": float(aggregate[name].total.detach()),
                "coefficient": coeffs[name],
                "weighted_loss": coeffs[name] * float(aggregate[name].total.detach()),
                "counts": {k: int(v) for k, v in aggregate[name].counts.items()},
            }
    record["losses_finite"] = all(
        math.isfinite(record["terms"][name]["raw_loss"])
        for name in _TERM_NAMES
    )
    record["contact_evaluable"] = any(
        entry["contact_evaluable"] for entry in record["per_case"]
    )
    record["pde_evaluable"] = all(
        entry["time_ranges"]["pde"]["count"] > 0 for entry in record["per_case"]
    )
    return record


def _contact_metrics(trainer, case, times: np.ndarray, low: np.ndarray, up: np.ndarray) -> dict:
    if times.size == 0:
        return {
            "n": 0, "mean_squared_violation": None, "rmse": None,
            "mean_violation_over_Rsep": None, "max_violation_over_Rsep": None,
            "coverage": None,
        }
    t = torch.as_tensor(times, dtype=torch.float64)
    with torch.enable_grad():
        fields = _contact_at(trainer, case, t, create_graph=False)
    radius = fields.radius.detach().reshape(-1)
    low_t = torch.as_tensor(low, dtype=torch.float64)
    up_t = torch.as_tensor(up, dtype=torch.float64)
    scale = float(case.length_scale)
    below = torch.clamp(low_t - radius, min=0.0)
    above = torch.clamp(radius - up_t, min=0.0)
    squared = (below / scale) ** 2 + (above / scale) ** 2
    violation = (below + above) / scale
    return {
        "n": int(times.size),
        "mean_squared_violation": float(squared.mean()),
        "rmse": float(torch.sqrt(torch.mean(squared))),
        "mean_violation_over_Rsep": float(violation.mean()),
        "max_violation_over_Rsep": float(violation.max()),
        "coverage": float(torch.count_nonzero((below == 0.0) & (above == 0.0))
                          / times.size),
    }


def _reference_points(trainer) -> list[dict]:
    points = []
    for case in trainer.cases:
        indices = case.contact.indices()
        times = case.contact.times[indices]
        keep = (
            (times >= float(case.t0))
            & (times <= float(case.t_end))
            & np.asarray([case.shock.has_slope(float(t)) for t in times])
        )
        indices = indices[keep]
        points.append({
            "case_id": case.case_id,
            "times": np.asarray(case.contact.times[indices]),
            "low": np.asarray(case.contact.low[indices]),
            "up": np.asarray(case.contact.up[indices]),
            "t0": float(case.t0),
            "t_end": float(case.t_end),
        })
    return points


def _reference_metrics(trainer, points: list[dict]) -> dict:
    cases = []
    total_sq = 0.0
    total_n = 0
    total_violation = 0.0
    total_covered = 0
    for point in points:
        case = next(c for c in trainer.cases if c.case_id == point["case_id"])
        metrics = _contact_metrics(
            trainer, case, point["times"], point["low"], point["up"]
        )
        thirds = []
        span = point["t_end"] - point["t0"]
        for which in range(3):
            lo = point["t0"] + (which / 3.0) * span
            hi = point["t0"] + ((which + 1) / 3.0) * span
            mask = ((point["times"] >= lo) & (point["times"] <= hi))
            thirds.append(_contact_metrics(
                trainer, case, point["times"][mask],
                point["low"][mask], point["up"][mask],
            ))
        cases.append({"case_id": point["case_id"], **metrics, "thirds": thirds})
        if metrics["n"]:
            total_sq += metrics["mean_squared_violation"] * metrics["n"]
            total_violation += metrics["mean_violation_over_Rsep"] * metrics["n"]
            total_covered += int(metrics["coverage"] * metrics["n"])
            total_n += metrics["n"]
    return {
        "n": total_n,
        "mean_squared_violation": (total_sq / total_n) if total_n else None,
        "rmse": math.sqrt(total_sq / total_n) if total_n else None,
        "mean_violation_over_Rsep": (total_violation / total_n) if total_n else None,
        "coverage": (total_covered / total_n) if total_n else None,
        "cases": cases,
    }


def _batch_metrics(trainer, batches) -> dict:
    cases = []
    total_sq = 0.0
    total_n = 0
    total_violation = 0.0
    total_covered = 0
    for case, batch in batches:
        times = batch.t_contact.detach().numpy()
        low = batch.low_contact.detach().numpy()
        up = batch.up_contact.detach().numpy()
        metrics = _contact_metrics(trainer, case, times, low, up)
        cases.append({"case_id": case.case_id, **metrics})
        if metrics["n"]:
            total_sq += metrics["mean_squared_violation"] * metrics["n"]
            total_violation += metrics["mean_violation_over_Rsep"] * metrics["n"]
            total_covered += int(metrics["coverage"] * metrics["n"])
            total_n += metrics["n"]
    return {
        "n": total_n,
        "mean_squared_violation": (total_sq / total_n) if total_n else None,
        "rmse": math.sqrt(total_sq / total_n) if total_n else None,
        "mean_violation_over_Rsep": (total_violation / total_n) if total_n else None,
        "coverage": (total_covered / total_n) if total_n else None,
        "cases": cases,
    }


def _flatten(params) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _adam_summary(optimizer) -> dict:
    steps = []
    for state in optimizer.state.values():
        if "step" in state:
            value = state["step"]
            steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return {
        "n_state_entries": len(optimizer.state),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "step_values": sorted(set(steps))[:10],
    }


def _optimizer_order_ok(optimizer, module) -> dict:
    groups = optimizer.param_groups
    parameters = list(module.parameters())
    ordered = (
        len(groups) == 1
        and len(groups[0].get("params", [])) == len(parameters)
        and all(a is b for a, b in zip(groups[0]["params"], parameters))
    )
    return {
        "n_groups": len(groups),
        "n_params": len(parameters),
        "order_matches_module": bool(ordered),
    }


def _build_trainer(config_path: str) -> tuple[AlternatingTrainer, dict]:
    config = load_config(config_path)
    section = _section(config)
    specs = _case_specs(section)
    train = [
        _make_case(section, spec, config, allow_synthetic=False)
        for spec in specs if str(spec.get("role", "train")) == "train"
    ]
    validation = [
        _make_case(section, spec, config, allow_synthetic=False)
        for spec in specs if str(spec.get("role", "train")) == "validation"
    ]
    seed = int(section.get("training", {}).get("seed", 0))
    seed_everything(seed)
    contact, air = _networks(section)
    trainer = AlternatingTrainer(
        cases=train, contact_net=contact, air_net=air,
        config=_train_config(section, None),
        output_dir=Path("outputs/fullframes/contact_training_diagnostics/_trainer"),
        external_validation_cases=validation,
    )
    return trainer, {"config": config, "section": section}


def _slice_ranges(case, which: int) -> tuple[float, float]:
    span = float(case.t_end) - float(case.t0)
    lo = float(case.t0) + (which / 3.0) * span
    hi = float(case.t0) + ((which + 1) / 3.0) * span
    return lo, hi


def _make_batches(trainer, mode: str, index: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed) + index * 7919)
    if mode == "training":
        return trainer._batches(generator=generator), None
    # A single time range is shared by all training cases in stratified mode.
    # The ranges are naturally different per case because t0/t_end differ, so
    # distribute slice indices over the requested batches.
    which = index % 3
    ranges = [
        (max(float(case.t0), float(case.t0) + (which / 3.0) *
             (float(case.t_end) - float(case.t0))),
         min(float(case.t_end), float(case.t0) + ((which + 1) / 3.0) *
             (float(case.t_end) - float(case.t0))))
        for case in trainer.cases
    ]
    # sample_case_batch takes one range; stratify per case by constructing the
    # batches directly with the official per-case counts.
    n = len(trainer.cases)
    counts = {
        "pde": _split(trainer.config.n_pde, n),
        "ic": _split(trainer.config.n_ic, n),
        "right": _split(trainer.config.n_right, n),
        "data": _split(trainer.config.n_data, n),
        "contact": _split(trainer.config.n_contact, n),
    }
    from spherical_blast_pinn.contact_air.sampling import sample_case_batch
    batches = []
    for i, case in enumerate(trainer.cases):
        lo, hi = ranges[i]
        batches.append((
            case,
            sample_case_batch(
                case, n_pde=counts["pde"][i], n_ic=counts["ic"][i],
                n_right=counts["right"][i], n_data=counts["data"][i],
                n_contact=counts["contact"][i], generator=generator,
                pde_time_range=(lo, hi), contact_time_range=(lo, hi),
            ),
        ))
    return batches, which


def _split(total: int, n: int) -> list[int]:
    base, extra = divmod(int(total), int(n))
    return [base + (1 if i < extra else 0) for i in range(n)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("training", "stratified"),
                        default="training")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch-seed", type=int, default=20261114)
    parser.add_argument("--adam-trials", type=int, default=0)
    parser.add_argument("--adam-seed", type=int, default=20261115)
    parser.add_argument("--zero-threshold", type=float, default=1.0e-12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if int(args.batches) <= 0:
        parser.error("--batches must be > 0")
    if int(args.adam_trials) < 0:
        parser.error("--adam-trials must be >= 0")
    if args.mode == "stratified" and int(args.adam_trials) > 0:
        parser.error("Adam single-step checks are defined for --mode training only")

    checkpoint = Path(args.checkpoint)
    checkpoint_hash_before = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adam_available = bool((payload.get("opt_contact") or {}).get("state"))
    trainer, meta = _build_trainer(args.config)
    # Static diagnostics must start from the checkpoint parameters, not from
    # the trainer's freshly initialised networks.
    if adam_available:
        trainer.load(checkpoint)
    else:
        # Networks only: an old checkpoint without optimizer state must still
        # support the static gradient diagnostic.
        _load_checkpoint(checkpoint, trainer.contact_net, trainer.air_net)
        trainer.loop = int(payload.get("loop", 0))
        trainer.stage_state = str(payload.get("stage_state", "alternating"))

    requested = int(args.batches)
    completed = 0
    failed = 0
    non_finite_any = False
    batches_out = []
    for index in range(requested):
        try:
            batches, which = _make_batches(
                trainer, args.mode, index, int(args.batch_seed)
            )
            record = _static_record(
                trainer, batches, compute_gradients=True,
                threshold=float(args.zero_threshold),
            )
            record["batch"] = index
            record["mode"] = args.mode
            record["slice"] = which
            record["batch_seed"] = int(args.batch_seed)
            record["contact_evaluable"] = any(
                entry["contact_evaluable"] for entry in record["per_case"]
            )
            record["pde_evaluable"] = all(
                entry["time_ranges"]["pde"]["count"] > 0
                for entry in record["per_case"]
            )
            g = record["aggregate_gradients"]
            if not g["finite_all"]:
                non_finite_any = True
            # Drop the graph-only vector before JSON serialisation.
            record.pop("_vector_g_data", None)
            record.pop("_vectors", None)
            batches_out.append(record)
            completed += 1
            print(f"batch {index + 1}/{requested} mode={args.mode} "
                  f"slice={which} done")
        except Exception as exc:  # record the failure and continue
            failed += 1
            batches_out.append({
                "batch": index, "mode": args.mode,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"batch {index + 1}/{requested} FAILED: {exc}", file=sys.stderr)

    adam_trials = []
    reference = _reference_points(trainer)
    if int(args.adam_trials) > 0:
        if not adam_available:
            print(
                "checkpoint has no optimizer state; Adam single-step checks "
                "are not available", file=sys.stderr,
            )
        else:
            for trial in range(int(args.adam_trials)):
                try:
                    record = _adam_trial(
                        trainer, checkpoint, reference, args, trial
                    )
                    adam_trials.append(record)
                    if record.get("non_finite"):
                        non_finite_any = True
                    print(f"adam trial {trial + 1}/{args.adam_trials} done")
                except Exception as exc:
                    adam_trials.append({
                        "trial": trial, "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"adam trial {trial + 1} FAILED: {exc}",
                          file=sys.stderr)
    elif int(args.adam_trials) == 0:
        # Static diagnostic still reports the fixed reference state once.
        pass

    checkpoint_hash_after = _sha256(checkpoint)
    output = {
        "format": "spherical_blast_pinn.contact_gradient_diagnostic",
        "version": 2,
        "config": args.config,
        "checkpoint": str(checkpoint),
        "checkpoint_loop": int(payload.get("loop", 0)),
        "mode": args.mode,
        "batch_seed": int(args.batch_seed),
        "requested_batches": requested,
        "completed_batches": completed,
        "failed_batches": failed,
        "adam_available": adam_available,
        "adam_requested": int(args.adam_trials),
        "adam_completed": sum(1 for entry in adam_trials
                              if entry.get("status") != "failed"),
        "adam_failed": sum(1 for entry in adam_trials
                           if entry.get("status") == "failed"),
        "checkpoint_sha256": {
            "before": checkpoint_hash_before,
            "after": checkpoint_hash_after,
            "unchanged": checkpoint_hash_before == checkpoint_hash_after,
        },
        "non_finite": bool(non_finite_any),
        "training_cases": [case.case_id for case in trainer.cases],
        "validation_cases": [case.case_id for case in trainer.validation_cases],
        "batches": batches_out,
        "adam_trials": adam_trials,
        "note": (
            "static batches use the official 10-case budget; stratified "
            "batches are supplementary and never fall back to the full window; "
            "Adam trials are independent one-step updates from the checkpoint "
            "and never modify the file"
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, default=_jsonable),
                   encoding="utf-8")
    print(f"written: {out}")
    if non_finite_any or failed:
        return 1
    return 0


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _adam_trial(trainer, checkpoint: Path, reference, args, trial: int) -> dict:
    trainer.load(checkpoint)
    optimizer_order = _optimizer_order_ok(
        trainer.opt_contact, trainer.contact_net
    )
    if not optimizer_order["order_matches_module"] or not optimizer_order["n_groups"] == 1:
        raise RuntimeError(f"ContactNet optimizer parameter order is not safe: {optimizer_order}")
    seed = int(args.adam_seed) + trial * 104729
    batches = trainer._batches(generator=torch.Generator().manual_seed(seed))
    lambda_pde, lambda_data = schedule_weights(
        trainer.loop, trainer.config.n_schedule
    )
    params = [p for p in trainer.contact_net.parameters() if p.requires_grad]
    theta_before = _flatten(params)
    adam_before = _adam_summary(trainer.opt_contact)

    air_total, lambda_pde, lambda_data = trainer._air_update(batches)
    before_static = _static_record(
        trainer, batches, compute_gradients=True,
        threshold=float(args.zero_threshold),
    )
    vectors = before_static.pop("_vectors")
    before_static.pop("_vector_g_data", None)
    g_data = vectors["contact_data"]
    ref_before = _reference_metrics(trainer, reference)
    batch_before = _batch_metrics(trainer, batches)

    trainer.contact_update(
        batches, lambda_pde=lambda_pde, lambda_data=lambda_data
    )
    after_static = _static_record(
        trainer, batches, compute_gradients=False,
        threshold=float(args.zero_threshold),
    )
    ref_after = _reference_metrics(trainer, reference)
    batch_after = _batch_metrics(trainer, batches)
    theta_after = _flatten(params)
    delta = theta_after - theta_before
    adam_after = _adam_summary(trainer.opt_contact)
    dot = float(torch.dot(g_data, delta))
    contact_data_before = before_static["terms"]["contact_data"]["weighted_loss"]
    contact_data_after = after_static["terms"]["contact_data"]["weighted_loss"]
    total_before = sum(
        before_static["terms"][name]["weighted_loss"] for name in _TERM_NAMES
    )
    total_after = sum(
        after_static["terms"][name]["weighted_loss"] for name in _TERM_NAMES
    )
    return {
        "trial": trial,
        "status": "ok",
        "checkpoint_loop": int(trainer.loop),
        "stage_state": str(trainer.stage_state),
        "simulated_update_from_loop": int(trainer.loop),
        "simulated_update_to_loop": int(trainer.loop) + 1,
        "batch_seed": seed,
        "lambda_pde": float(lambda_pde),
        "lambda_data": float(lambda_data),
        "contact_data_weight": float(trainer.config.contact_data_weight),
        "optimizer_order": optimizer_order,
        "adam_before": adam_before,
        "adam_after": adam_after,
        "air_loss": float(air_total.total.detach()),
        "before": before_static,
        "after": after_static,
        "batch_contact_before": batch_before,
        "batch_contact_after": batch_after,
        "reference_contact_before": ref_before,
        "reference_contact_after": ref_after,
        "parameter_delta": {
            "l2": float(delta.norm()),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "n": int(delta.numel()),
        },
        "parameter_after_sha256": hashlib.sha256(
            theta_after.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "contact_data_loss_change": {
            "weighted_current_batch": contact_data_after - contact_data_before,
            "weighted_total_current_batch": total_after - total_before,
            "first_order_prediction_g_d_dot_delta": dot,
        },
        "non_finite": bool(
            not before_static.get("losses_finite", True)
            or not after_static.get("losses_finite", True)
            or not before_static.get("aggregate_gradients", {}).get("finite_all", True)
        ),
        "reference_loss_change": {
            "mean_squared_violation": (
                None if ref_before["mean_squared_violation"] is None
                or ref_after["mean_squared_violation"] is None else
                ref_after["mean_squared_violation"]
                - ref_before["mean_squared_violation"]
            ),
            "mean_violation_over_Rsep": (
                None if ref_before["mean_violation_over_Rsep"] is None
                or ref_after["mean_violation_over_Rsep"] is None else
                ref_after["mean_violation_over_Rsep"]
                - ref_before["mean_violation_over_Rsep"]
            ),
            "coverage": (
                None if ref_before["coverage"] is None
                or ref_after["coverage"] is None else
                ref_after["coverage"] - ref_before["coverage"]
            ),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
