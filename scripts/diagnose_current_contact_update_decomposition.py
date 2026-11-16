#!/usr/bin/env python3
"""Read-only decomposition of the current C ContactNet updates.

This reuses the exact 80 single-step trials of
``diagnose_current_contact_step.py`` (2 checkpoints x 2 seeds x 20 trials) and
adds, for every trial:

* the weighted gradients of the four ContactNet-stage terms on the official
  random batch;
* the gradient of the full-training-label contact reference C;
* the actual Adam candidate delta, reconstructed under the second-moment
  denominator of that same step, split into stored Adam history and the four
  current weighted terms;
* projections of gC and gJ on the delta components and on the full delta.

The random contact labels drive the update.  The full training labels are used
only to evaluate C and gC.  Validation labels are never loaded.  No checkpoint
is written and no training loop is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import contact_fullcontact_pair as cf  # noqa: E402
import diagnose_contact_gradients as diag  # noqa: E402
import diagnose_current_contact_step as step  # noqa: E402

SEEDS = step.SEEDS
ALPHAS = step.ALPHAS
TERMS = step.TERMS
C_CONFIG = step.C_CONFIG
CHECKPOINTS = step.CHECKPOINTS
_TOL = 1.0e-12


def _sha256(path: Path) -> str:
    return step._sha256(path)


def _status_reasons(payload) -> list[str]:
    reasons: list[str] = []
    for tag, entry in payload.get("checkpoints", {}).items():
        if not entry.get("unchanged", False):
            reasons.append(f"{tag}: checkpoint SHA-256 changed")
        trials = entry.get("trials", [])
        if not trials:
            reasons.append(f"{tag}: no trials recorded")
        failed = [
            row for row in trials
            if row.get("status") != "ok"
            or not all(row.get("checks", {}).values())
        ]
        if failed:
            reasons.append(
                f"{tag}: {len(failed)}/{len(trials)} trial status failed"
            )
    return reasons


def _median(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    return (
        values[n // 2]
        if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
    )


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _stats(values) -> dict:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {"n": 0, "median": None, "mean": None}
    return {"n": len(values), "median": _median(values), "mean": _mean(values)}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> dict:
    return diag._cosine(a, b, _TOL)


def _objective_graph(trainer, batches, coefficients):
    """Four equal-case term means and the weighted joint objective J."""
    rows = [
        trainer.contact_stage_terms(case, batch)
        for case, batch in batches
    ]
    n = len(rows)
    like = torch.zeros((), dtype=torch.float64)
    aggregate = {
        name: sum((row[name].total for row in rows), like) / float(n)
        for name in TERMS
    }
    total = sum(coefficients[name] * aggregate[name] for name in TERMS)
    raw = {
        name: float(aggregate[name].detach()) for name in TERMS
    }
    return aggregate, total, raw


def _gradient_bundle(trainer, random_batches, full_batches, coefficients):
    """gJ, gC, and every weighted current term gradient."""
    params = cf._contact_parameters(trainer)
    aggregate, total, raw = _objective_graph(
        trainer, random_batches, coefficients
    )
    gradients: dict[str, torch.Tensor] = {}
    finite = True
    for name in TERMS:
        vector, ok = diag._grad(
            coefficients[name] * aggregate[name], params
        )
        gradients[name] = vector
        finite = finite and ok
    g_total, total_ok = diag._grad(total, params)
    full_contact = step._full_contact_value(trainer, full_batches)
    g_contact, contact_ok = diag._grad(full_contact, params)
    finite = (
        finite and total_ok and contact_ok
        and bool(torch.isfinite(g_total).all())
        and bool(torch.isfinite(g_contact).all())
    )
    contribution_sum = sum(gradients.values(), torch.zeros_like(g_total))
    return {
        "params": params,
        "aggregate": aggregate,
        "raw": raw,
        "gradients": gradients,
        "g_total": g_total,
        "g_contact": g_contact,
        "full_contact": full_contact,
        "finite": bool(finite),
        "term_sum_abs_error": float((contribution_sum - g_total).norm()),
        "term_sum_relative_error": (
            float((contribution_sum - g_total).norm() / g_total.norm())
            if float(g_total.norm()) > 0.0 else None
        ),
    }


def _flag_dict(c_at_one: dict, base_j: float, base_c: float,
               reduced_c: list[tuple[float, float]]) -> dict:
    return {
        "J_improves": bool(c_at_one["J_change"] < 0.0),
        "C_improves": bool(c_at_one["C_change"] < 0.0),
        "J_and_C_improve": bool(
            c_at_one["J_change"] < 0.0 and c_at_one["C_change"] < 0.0
        ),
        "J_improves_C_worse": bool(
            c_at_one["J_change"] < 0.0 and c_at_one["C_change"] > 0.0
        ),
        "full_C_worse": bool(c_at_one["C_change"] > 0.0),
        "full_J_worse": bool(c_at_one["J_change"] > 0.0),
        "reduced_C_improves": bool(
            min((value for _, value in reduced_c), default=base_c) < base_c
        ),
        "full_C_worse_reduced_C_improves": bool(
            c_at_one["C_change"] > 0.0
            and min((value for _, value in reduced_c), default=base_c) < base_c
        ),
    }


def _component_projections(gC, gJ, delta_parts: dict, delta) -> dict:
    out = {
        "gC_dot_history": float(torch.dot(gC, delta_parts["history"])),
        "gC_dot_terms": {
            name: float(torch.dot(gC, delta_parts["terms"][name]))
            for name in TERMS
        },
        "gJ_dot_history": float(torch.dot(gJ, delta_parts["history"])),
        "gJ_dot_terms": {
            name: float(torch.dot(gJ, delta_parts["terms"][name]))
            for name in TERMS
        },
        "gC_dot_delta": float(torch.dot(gC, delta)),
        "gJ_dot_delta": float(torch.dot(gJ, delta)),
    }
    out["gC_dot_terms_sum"] = float(sum(out["gC_dot_terms"].values()))
    out["gJ_dot_terms_sum"] = float(sum(out["gJ_dot_terms"].values()))
    out["gC_component_sum"] = (
        out["gC_dot_history"] + out["gC_dot_terms_sum"]
    )
    out["gJ_component_sum"] = (
        out["gJ_dot_history"] + out["gJ_dot_terms_sum"]
    )
    out["gC_component_error"] = out["gC_component_sum"] - out["gC_dot_delta"]
    out["gJ_component_error"] = out["gJ_component_sum"] - out["gJ_dot_delta"]
    return out


def _trial(trainer, checkpoint, labels, seed, trial):
    trial_seed = int(seed) + int(trial) * 104729
    generator = torch.Generator().manual_seed(trial_seed)
    trainer.load(checkpoint)
    checkpoint_loop = int(trainer.loop)
    contact_before = step._state_hash(trainer.contact_net)
    opt_before = cf._optimizer_hash(trainer.opt_contact)
    air_before = step._state_hash(trainer.air_net)

    random_batches = trainer._batches(generator=generator)
    full_batches = step._full_batches(trainer, random_batches, labels)
    air_total, lambda_pde, lambda_data = trainer._air_update(random_batches)
    air_after = step._state_hash(trainer.air_net)
    coefficients = diag._coefficients(trainer)
    theta0 = cf._flatten(cf._contact_parameters(trainer))
    c0, j0, raw0 = step._values(trainer, random_batches, coefficients)
    c_ref0 = float(step._full_contact_value(trainer, full_batches).detach())

    params = cf._contact_parameters(trainer)
    optimizer_order = diag._optimizer_order_ok(
        trainer.opt_contact, trainer.contact_net
    )
    if not optimizer_order["order_matches_module"]:
        raise RuntimeError(f"unsafe ContactNet optimizer order: {optimizer_order}")
    if float(trainer.config.gradient_clip) > 0.0:
        raise RuntimeError("decomposition requires gradient_clip == 0")
    snapshots = cf._adam_snapshot(trainer.opt_contact, params)
    bundle = _gradient_bundle(
        trainer, random_batches, full_batches, coefficients
    )
    gradients = bundle["gradients"]
    g_total = bundle["g_total"]
    g_contact = bundle["g_contact"]

    captured: dict[str, torch.Tensor] = {}

    def _capture_step(_optimizer, _args, _kwargs):
        captured["gradient"] = cf._flatten([
            parameter.grad for parameter in params
        ])

    handle = trainer.opt_contact.register_step_pre_hook(_capture_step)
    started = time.perf_counter()
    try:
        trainer.contact_update(
            random_batches, lambda_pde=lambda_pde, lambda_data=lambda_data
        )
    finally:
        handle.remove()
    update_s = time.perf_counter() - started
    theta1 = cf._flatten(cf._contact_parameters(trainer))
    delta = theta1 - theta0
    delta_norm = float(delta.norm())
    captured_gradient = captured.get("gradient")
    capture_error = (
        None if captured_gradient is None
        else float((captured_gradient - g_total).norm())
    )
    capture_rel = (
        None if capture_error is None or float(g_total.norm()) == 0.0
        else capture_error / float(g_total.norm())
    )
    reconstruction = cf._adam_decomposition(
        trainer.opt_contact, params, snapshots, gradients, g_total, delta
    )
    history = reconstruction["_history"]
    term_deltas = reconstruction["_terms"]
    component = _component_projections(
        g_contact, g_total,
        {"history": history, "terms": term_deltas}, delta,
    )
    delta_terms = sum(
        term_deltas.values(), torch.zeros_like(history)
    )
    history_norm = float(history.norm())
    terms_norm = float(delta_terms.norm())
    shares = {
        "history_share": (
            history_norm / delta_norm if delta_norm > 0.0 else None
        ),
        "terms_share": (
            terms_norm / delta_norm if delta_norm > 0.0 else None
        ),
    }

    term_grad_norms = {
        name: float(gradients[name].norm()) for name in TERMS
    }
    term_delta_norms = {
        name: float(term_deltas[name].norm()) for name in TERMS
    }
    grad_norm_total = sum(term_grad_norms.values())

    records = []
    base_j, base_c = j0, c_ref0
    reduced_c = []
    for alpha in ALPHAS:
        cf._set_flat(
            cf._contact_parameters(trainer), theta0 + float(alpha) * delta
        )
        c_random, j_value, raw = step._values(
            trainer, random_batches, coefficients
        )
        c_full = float(
            step._full_contact_value(trainer, full_batches).detach()
        )
        records.append({
            "alpha": float(alpha),
            "J": j_value,
            "J_change": j_value - base_j,
            "J_terms_raw": raw,
            "J_terms_weighted": {
                name: coefficients[name] * raw[name] for name in TERMS
            },
            "C": c_full,
            "C_change": c_full - base_c,
            "gJ_dot_delta": component["gJ_dot_delta"],
            "gC_dot_delta": component["gC_dot_delta"],
        })
        if alpha < 1.0:
            reduced_c.append((float(alpha), c_full))
    cf._set_flat(cf._contact_parameters(trainer), theta0)
    restored_contact = step._state_hash(trainer.contact_net) == contact_before
    c_at_one = records[0]
    flags = _flag_dict(c_at_one, base_j, base_c, reduced_c)
    finite_values = all(
        math.isfinite(float(record[key]))
        for record in records for key in ("J", "C", "J_change", "C_change")
    )
    reconstruction_rel = reconstruction.get(
        "reconstruction_relative_error"
    )
    contribution_rel = reconstruction.get(
        "contribution_sum_relative_error"
    )
    reconstruction_ok = bool(
        reconstruction.get("supported")
        and (
            reconstruction_rel is None
            or (math.isfinite(reconstruction_rel)
                and abs(reconstruction_rel) <= 1.0e-8)
        )
        and (
            contribution_rel is None
            or (math.isfinite(contribution_rel)
                and abs(contribution_rel) <= 1.0e-8)
        )
    )
    capture_ok = bool(
        capture_error is not None
        and (
            capture_rel is None
            or (math.isfinite(capture_rel) and abs(capture_rel) <= 1.0e-8)
        )
    )
    checks = {
        "finite_gradients": bundle["finite"],
        "finite_values": finite_values,
        "air_frozen_during_contact": step._state_hash(trainer.air_net) == air_after,
        "contact_restored_after_evaluation": bool(restored_contact),
        "optimizer_order_matches_module": bool(
            optimizer_order["order_matches_module"]
        ),
        "captured_gradient_matches_autograd": capture_ok,
        "adam_reconstruction": reconstruction_ok,
    }
    status = "ok" if all(checks.values()) else "failed"

    record = {
        "trial": int(trial),
        "seed": trial_seed,
        "checkpoint_loop": checkpoint_loop,
        "initial_contact_state": contact_before,
        "initial_air_state": air_before,
        "initial_contact_optimizer": opt_before,
        "air_after_state": air_after,
        "air_loss": float(air_total.total.detach()),
        "lambda_pde": float(lambda_pde),
        "lambda_data": float(lambda_data),
        "contact_data_weight": float(trainer.config.contact_data_weight),
        "coefficients": dict(coefficients),
        "delta_norm": delta_norm,
        "update_s": update_s,
        "J_base": base_j,
        "C_base": base_c,
        "J_terms_base_raw": raw0,
        "J_terms_base_weighted": {
            name: coefficients[name] * raw0[name] for name in TERMS
        },
        "term_grad_norm_weighted": term_grad_norms,
        "term_grad_norm_share": {
            name: (term_grad_norms[name] / grad_norm_total
                   if grad_norm_total > 0.0 else None)
            for name in TERMS
        },
        "term_delta_norm": term_delta_norms,
        "term_delta_share": {
            name: (term_delta_norms[name] / delta_norm
                   if delta_norm > 0.0 else None)
            for name in TERMS
        },
        "history_delta_norm": history_norm,
        "current_terms_delta_norm": terms_norm,
        "history_share": shares["history_share"],
        "current_terms_share": shares["terms_share"],
        "cos_data_fullC": _cosine(gradients["contact_data"], g_contact),
        "cos_data_J": _cosine(gradients["contact_data"], g_total),
        "cos_J_C": _cosine(g_total, g_contact),
        "cos_data_other": _cosine(
            gradients["contact_data"],
            sum(
                (gradients[name] for name in ("pde", "contact_ic", "interface")),
                torch.zeros_like(g_total),
            ),
        ),
        "cos_history_delta": _cosine(history, delta),
        "cos_terms_delta": _cosine(delta_terms, delta),
        "cos_history_terms": _cosine(history, delta_terms),
        "grad_norm_data_weighted": term_grad_norms["contact_data"],
        "grad_norm_C_full": float(g_contact.norm()),
        "grad_norm_J": float(g_total.norm()),
        "norm_ratio_data_to_C": (
            term_grad_norms["contact_data"] / float(g_contact.norm())
            if float(g_contact.norm()) > 0.0 else None
        ),
        "dot_data_fullC": float(
            torch.dot(gradients["contact_data"], g_contact)
        ),
        "dot_data_J": float(torch.dot(gradients["contact_data"], g_total)),
        "adam": {
            "history_norm": history_norm,
            "term_norm": {
                name: term_delta_norms[name] for name in TERMS
            },
            "lr": reconstruction["lr"],
            "betas": reconstruction["betas"],
            "eps": reconstruction["eps"],
            "step_values": reconstruction["step_values"],
            "denominator_median": reconstruction["denominator_median"],
            "reconstructed_norm": reconstruction["reconstructed_norm"],
            "reconstruction_abs_error": reconstruction[
                "reconstruction_abs_error"
            ],
            "reconstruction_relative_error": reconstruction_rel,
            "contribution_sum_abs_error": reconstruction[
                "contribution_sum_abs_error"
            ],
            "contribution_sum_relative_error": contribution_rel,
        },
        "projections": component,
        "gradient_capture_abs_error": capture_error,
        "gradient_capture_relative_error": capture_rel,
        "term_gradient_sum_abs_error": bundle["term_sum_abs_error"],
        "term_gradient_sum_relative_error": bundle[
            "term_sum_relative_error"
        ],
        "records": records,
        "flags": flags,
        "checks": checks,
        "status": status,
    }
    for key in ("_history", "_terms", "_reconstruction"):
        reconstruction.pop(key, None)
    return record


def _summary(rows) -> dict:
    def count(key):
        return int(sum(bool(row["flags"][key]) for row in rows))

    def values(key):
        return [row[key] for row in rows]

    def nested(key, name):
        return [row[key].get(name) for row in rows]

    history_dominates = sum(
        1 for row in rows
        if row["history_delta_norm"] > row["current_terms_delta_norm"]
    )
    sign_consistent = sum(
        1 for row in rows
        if (row["projections"]["gC_dot_delta"] < 0.0)
        == (row["records"][0]["C_change"] < 0.0)
    )
    dominant_grad = {
        name: sum(
            1 for row in rows
            if max(TERMS, key=lambda key: row["term_grad_norm_weighted"][key])
            == name
        )
        for name in TERMS
    }
    dominant_delta = {
        name: sum(
            1 for row in rows
            if max(TERMS, key=lambda key: row["term_delta_norm"][key])
            == name
        )
        for name in TERMS
    }
    return {
        "n": len(rows),
        "counts": {
            "J_improves": count("J_improves"),
            "C_improves": count("C_improves"),
            "J_and_C_improve": count("J_and_C_improve"),
            "J_improves_C_worse": count("J_improves_C_worse"),
            "full_C_worse": count("full_C_worse"),
            "full_J_worse": count("full_J_worse"),
            "reduced_C_improves": count("reduced_C_improves"),
            "full_C_worse_reduced_C_improves": count(
                "full_C_worse_reduced_C_improves"
            ),
            "history_dominates": history_dominates,
            "gC_dot_delta_sign_consistent": sign_consistent,
        },
        "median": {
            "C_change": _median([
                row["records"][0]["C_change"] for row in rows
            ]),
            "J_change": _median([
                row["records"][0]["J_change"] for row in rows
            ]),
            "delta_norm": _median(values("delta_norm")),
            "cos_data_fullC": _median([
                row["cos_data_fullC"].get("cosine") for row in rows
            ]),
            "cos_data_J": _median([
                row["cos_data_J"].get("cosine") for row in rows
            ]),
            "cos_J_C": _median([
                row["cos_J_C"].get("cosine") for row in rows
            ]),
            "cos_data_other": _median([
                row["cos_data_other"].get("cosine") for row in rows
            ]),
            "cos_history_delta": _median([
                row["cos_history_delta"].get("cosine") for row in rows
            ]),
            "cos_terms_delta": _median([
                row["cos_terms_delta"].get("cosine") for row in rows
            ]),
            "cos_history_terms": _median([
                row["cos_history_terms"].get("cosine") for row in rows
            ]),
            "norm_ratio_data_to_C": _median(
                values("norm_ratio_data_to_C")
            ),
            "history_share": _median(values("history_share")),
            "current_terms_share": _median(values("current_terms_share")),
            "gC_dot_delta": _median([
                row["projections"]["gC_dot_delta"] for row in rows
            ]),
            "gC_dot_history": _median([
                row["projections"]["gC_dot_history"] for row in rows
            ]),
            "gC_dot_terms_sum": _median([
                row["projections"]["gC_dot_terms_sum"] for row in rows
            ]),
            "gJ_dot_delta": _median([
                row["projections"]["gJ_dot_delta"] for row in rows
            ]),
            "gJ_dot_history": _median([
                row["projections"]["gJ_dot_history"] for row in rows
            ]),
            "gJ_dot_terms_sum": _median([
                row["projections"]["gJ_dot_terms_sum"] for row in rows
            ]),
            "C_first_order_error": _median([
                row["projections"]["gC_dot_delta"]
                - row["records"][0]["C_change"] for row in rows
            ]),
            "J_first_order_error": _median([
                row["projections"]["gJ_dot_delta"]
                - row["records"][0]["J_change"] for row in rows
            ]),
        },
        "term_grad_norm_weighted": {
            name: _stats(nested("term_grad_norm_weighted", name))
            for name in TERMS
        },
        "term_delta_norm": {
            name: _stats(nested("term_delta_norm", name))
            for name in TERMS
        },
        "gC_dot_terms": {
            name: _stats([
                row["projections"]["gC_dot_terms"][name] for row in rows
            ])
            for name in TERMS
        },
        "gJ_dot_terms": {
            name: _stats([
                row["projections"]["gJ_dot_terms"][name] for row in rows
            ])
            for name in TERMS
        },
        "dominant_gradient_term": dominant_grad,
        "dominant_delta_term": dominant_delta,
        "reconstruction_relative_error": _stats([
            row["adam"]["reconstruction_relative_error"] for row in rows
        ]),
        "gradient_capture_relative_error": _stats([
            row["gradient_capture_relative_error"] for row in rows
        ]),
        "term_gradient_sum_relative_error": _stats([
            row["term_gradient_sum_relative_error"] for row in rows
        ]),
    }


def _summaries(rows) -> dict:
    worse = [row for row in rows if row["flags"]["full_C_worse"]]
    return {
        "all": _summary(rows),
        "contact_worse": _summary(worse),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    trainer, _ = diag._build_trainer(C_CONFIG)
    labels = {
        case.case_id: cf._full_labels(case) for case in trainer.cases
    }
    expected = [case.case_id for case in trainer.cases]
    issues = cf._label_issues(labels, expected)
    if issues:
        raise RuntimeError("; ".join(issues))

    payload = {
        "format": "spherical_blast_pinn.current_contact_update_decomposition",
        "version": 1,
        "config": C_CONFIG,
        "checkpoints": {},
        "seeds": list(SEEDS),
        "trials_per_checkpoint_seed": int(args.trials),
        "alphas": list(ALPHAS),
        "full_label_counts": {
            case_id: int(item["n"]) for case_id, item in labels.items()
        },
        "full_label_total": int(sum(item["n"] for item in labels.values())),
        "settings": {
            "update_contact_labels": "official random batch",
            "reference_contact_labels": "all legal training labels",
            "validation_labels_used": False,
            "air_update": "official _air_update on the same random batch",
            "adam_decomposition": (
                "fixed this step's second-moment denominator; stored first "
                "moment plus the four current weighted gradients"
            ),
        },
    }
    started = time.perf_counter()
    for tag, checkpoint in CHECKPOINTS.items():
        before = _sha256(checkpoint)
        rows = []
        for seed in SEEDS:
            for trial in range(int(args.trials)):
                rows.append((seed, _trial(
                    trainer, checkpoint, labels, seed, trial
                )))
        after = _sha256(checkpoint)
        payload["checkpoints"][tag] = {
            "path": str(checkpoint),
            "sha256": before,
            "sha256_after": after,
            "unchanged": before == after,
            "loops": {
                seed: sorted({record["checkpoint_loop"]
                              for s, record in rows if s == seed})
                for seed in SEEDS
            },
            "seeds": {
                str(seed): _summaries([
                    record for s, record in rows if s == seed
                ]) for seed in SEEDS
            },
            "combined": _summaries([record for _, record in rows]),
            "status_counts": {
                status: sum(1 for _, record in rows
                            if record.get("status") == status)
                for status in ("ok", "failed")
            },
            "trials": [
                {"seed": seed, **record} for seed, record in rows
            ],
        }
        if before != after:
            raise RuntimeError(f"{checkpoint}: checkpoint changed")
    payload["elapsed_s"] = time.perf_counter() - started
    payload["failure_reasons"] = _status_reasons(payload)
    payload["status"] = "ok" if not payload["failure_reasons"] else "failed"
    output_file = output / "decomposition.json"
    output_file.write_text(
        json.dumps(payload, indent=2, default=diag._jsonable),
        encoding="utf-8",
    )
    print(json.dumps({
        "wrote": str(output_file),
        "sha256": _sha256(output_file),
        "status": payload["status"],
        "failure_reasons": payload["failure_reasons"],
        "checkpoints": {
            tag: {
                "loop": entry["loops"],
                "status_counts": entry["status_counts"],
                "combined": entry["combined"],
            }
            for tag, entry in payload["checkpoints"].items()
        },
    }, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
