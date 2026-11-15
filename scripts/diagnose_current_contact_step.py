#!/usr/bin/env python3
"""Read-only single-step diagnosis of the current C checkpoints.

For each checkpoint and seed:
  * restore the checkpoint;
  * draw the official random contact batch;
  * run the official AirNet update;
  * compute J/gJ on that same random batch;
  * compute C/gC on all legal training contact labels (reference only);
  * take one real ContactNet Adam candidate step on the random batch;
  * evaluate J and C at alpha=1,1/2,1/4,1/8 on in-memory parameter copies;
  * restore for the next trial.

The full-label set is never used for the update direction or step choice.
Validation cases are never loaded.
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
from spherical_blast_pinn.contact_air.losses import contact_data_loss  # noqa: E402

SEEDS = (20261115, 20261116)
ALPHAS = (1.0, 0.5, 0.25, 0.125)
TERMS = ("pde", "contact_ic", "interface", "contact_data")
C_CONFIG = (
    "configs/contact_air/"
    "split_train_r50_500_fullframes_similarity_none_pair.yaml"
)
CHECKPOINTS = {
    "best": Path(
        "outputs/contact_air_time_input_pair_c_none_4000/best.pt"
    ),
    "last": Path(
        "outputs/contact_air_time_input_pair_c_none_4000/last.pt"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_hash(module) -> str:
    return cf._state_hash(module)


def _full_batches(trainer, random_batches, labels):
    return cf._all_label_batches(trainer, random_batches, labels)


def _full_contact_value(trainer, full_batches):
    """Equal-case full-label contact interval MSE; no PDE/interface terms."""
    values = []
    for case, batch in full_batches:
        r0, rs, ds = trainer._contact_inputs(case, batch.t_contact)
        fields = trainer.contact_net(
            batch.t_contact.reshape(-1, 1), r0, rs, ds,
            trainer._closures[case.case_id],
            time_input=case.time_input, create_graph=True,
        )
        values.append(contact_data_loss(fields, batch, case).total)
    return sum(values, torch.zeros((), dtype=torch.float64)) / float(
        len(values)
    )


def _values(trainer, batches, coefficients):
    rows = [trainer.contact_stage_terms(case, batch) for case, batch in batches]
    n = len(rows)
    like = torch.zeros((), dtype=torch.float64)
    raw = {
        name: float(sum((row[name].total for row in rows), like).detach())
        / float(n)
        for name in TERMS
    }
    total = sum(coefficients[name] * raw[name] for name in TERMS)
    return float(raw["contact_data"]), float(total), raw


def _gradients(trainer, random_batches, full_batches, coefficients):
    params = cf._contact_parameters(trainer)
    rows = [
        trainer.contact_stage_terms(case, batch)
        for case, batch in random_batches
    ]
    n = len(rows)
    like = torch.zeros((), dtype=torch.float64)
    aggregate = {
        name: sum((row[name].total for row in rows), like) / float(n)
        for name in TERMS
    }
    total = sum(coefficients[name] * aggregate[name] for name in TERMS)
    full_contact = _full_contact_value(trainer, full_batches)
    g_total, total_finite = diag._grad(total, params)
    g_contact, contact_finite = diag._grad(full_contact, params)
    return g_total, g_contact, bool(total_finite and contact_finite)


def _trial(trainer, checkpoint, labels, seed, trial):
    trial_seed = int(seed) + int(trial) * 104729
    generator = torch.Generator().manual_seed(trial_seed)
    trainer.load(checkpoint)
    checkpoint_loop = int(trainer.loop)
    contact_before = _state_hash(trainer.contact_net)
    opt_before = cf._optimizer_hash(trainer.opt_contact)
    air_before = _state_hash(trainer.air_net)

    random_batches = trainer._batches(generator=generator)
    full_batches = _full_batches(trainer, random_batches, labels)
    air_total, lambda_pde, lambda_data = trainer._air_update(random_batches)
    air_after = _state_hash(trainer.air_net)
    coefficients = diag._coefficients(trainer)
    theta0 = cf._flatten(cf._contact_parameters(trainer))
    c0, j0, raw0 = _values(trainer, random_batches, coefficients)
    c_ref0 = float(_full_contact_value(trainer, full_batches).detach())
    g_total, g_contact, finite = _gradients(
        trainer, random_batches, full_batches, coefficients
    )

    # One real Adam candidate step on the official random contact batch.
    trainer.contact_update(
        random_batches, lambda_pde=lambda_pde, lambda_data=lambda_data
    )
    theta1 = cf._flatten(cf._contact_parameters(trainer))
    delta = theta1 - theta0
    delta_norm = float(delta.norm())
    dot_total = float(torch.dot(g_total, delta))
    dot_contact = float(torch.dot(g_contact, delta))
    air_fixed = _state_hash(trainer.air_net) == air_after

    records = []
    base_j, base_c = j0, c_ref0
    for alpha in ALPHAS:
        cf._set_flat(
            cf._contact_parameters(trainer), theta0 + float(alpha) * delta
        )
        c_ref, j_value, raw = _values(
            trainer, random_batches, coefficients
        )
        c_full = float(_full_contact_value(trainer, full_batches).detach())
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
            "gJ_dot_delta": dot_total,
            "gC_dot_delta": dot_contact,
        })
    # The interpolation is not committed; restore the starting contact state.
    cf._set_flat(cf._contact_parameters(trainer), theta0)
    restored_contact = _state_hash(trainer.contact_net) == contact_before
    c_at_one = next(item for item in records if item["alpha"] == 1.0)
    c_vals = [item["C"] for item in records if item["alpha"] < 1.0]
    j_vals = [item["J"] for item in records if item["alpha"] < 1.0]
    full_worse_c = c_at_one["C_change"] > 0.0
    reduced_better_c = min(c_vals) < base_c
    reduced_better_j = min(j_vals) < base_j
    flags = {
        "J_improves": c_at_one["J_change"] < 0.0,
        "C_improves": c_at_one["C_change"] < 0.0,
        "J_and_C_improve": (
            c_at_one["J_change"] < 0.0
            and c_at_one["C_change"] < 0.0
        ),
        "J_improves_C_worse": (
            c_at_one["J_change"] < 0.0
            and c_at_one["C_change"] > 0.0
        ),
        "full_C_worse": full_worse_c,
        "reduced_C_improves": reduced_better_c,
        "full_C_worse_reduced_C_improves": (
            full_worse_c and reduced_better_c
        ),
        "reduced_C_no_improvement": not reduced_better_c,
        "full_J_worse": c_at_one["J_change"] > 0.0,
        "reduced_J_improves": reduced_better_j,
    }
    return {
        "trial": int(trial),
        "seed": trial_seed,
        "checkpoint_loop": checkpoint_loop,
        "air_loss": float(air_total.total.detach()),
        "lambda_pde": float(lambda_pde),
        "lambda_data": float(lambda_data),
        "contact_data_weight": float(trainer.config.contact_data_weight),
        "initial_contact_state": contact_before,
        "initial_air_state": air_before,
        "initial_contact_optimizer": opt_before,
        "air_after_state": air_after,
        "air_frozen_during_contact": bool(air_fixed),
        "contact_restored_after_evaluation": bool(restored_contact),
        "finite_gradients": bool(finite),
        "delta_norm": delta_norm,
        "J_base": base_j,
        "C_base": base_c,
        "J_terms_base_raw": raw0,
        "J_terms_base_weighted": {
            name: coefficients[name] * raw0[name] for name in TERMS
        },
        "records": records,
        "flags": flags,
        "status": "ok" if finite and air_fixed and restored_contact else "failed",
    }


def _summary(rows):
    keys = (
        "J_improves", "C_improves", "J_and_C_improve",
        "J_improves_C_worse", "full_C_worse", "reduced_C_improves",
        "full_C_worse_reduced_C_improves", "reduced_C_no_improvement",
        "full_J_worse", "reduced_J_improves",
    )
    return {
        "n": len(rows),
        "counts": {
            key: int(sum(bool(row["flags"][key]) for row in rows))
            for key in keys
        },
        "median_full_J_change": _median(
            [row["records"][0]["J_change"] for row in rows]
        ),
        "median_full_C_change": _median(
            [row["records"][0]["C_change"] for row in rows]
        ),
        "median_delta_norm": _median([row["delta_norm"] for row in rows]),
    }


def _median(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    n = len(values)
    return (
        values[n // 2]
        if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
    )


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
        "format": "spherical_blast_pinn.current_contact_step",
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
        },
    }
    all_rows = {}
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
                str(seed): _summary([
                    record for s, record in rows if s == seed
                ]) for seed in SEEDS
            },
            "combined": _summary([record for _, record in rows]),
            "trials": [
                {"seed": seed, **record} for seed, record in rows
            ],
        }
        all_rows[tag] = rows
        if before != after:
            raise RuntimeError(f"{checkpoint}: checkpoint changed")
    payload["elapsed_s"] = time.perf_counter() - started
    payload["status"] = "ok" if all(
        entry["unchanged"] for entry in payload["checkpoints"].values()
    ) else "failed"
    output_file = output / "diagnostic.json"
    output_file.write_text(
        json.dumps(payload, indent=2, default=diag._jsonable),
        encoding="utf-8",
    )
    print(json.dumps({
        "wrote": str(output_file),
        "sha256": _sha256(output_file),
        "status": payload["status"],
        "checkpoints": {
            tag: {
                "loop": entry["loops"],
                "combined": entry["combined"],
            }
            for tag, entry in payload["checkpoints"].items()
        },
    }, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
