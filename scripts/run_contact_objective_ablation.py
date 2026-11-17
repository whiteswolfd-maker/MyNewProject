#!/usr/bin/env python3
"""Diagnostic-only ContactNet objective ablation.

The formal two-stage loop is kept.  Only the ContactNet backward target is
changed, from the full weighted objective to
``lambda_data * contact_data_weight * L_contact_data``.  The PDE, IC and
interface terms are still evaluated and recorded, but do not enter backward.

This script is a diagnostic entry point: it never changes the formal default,
never writes into the formal checkpoint directories, and never selects a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402
from spherical_blast_pinn.contact_air.losses import schedule_weights  # noqa: E402
from spherical_blast_pinn.contact_air.trainer import TrainingError  # noqa: E402

_TERMS = ("pde", "contact_ic", "interface", "contact_data")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _tensor_hash(tensors) -> str:
    hasher = hashlib.sha256()
    for value in tensors:
        hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _optimizer_hash(optimizer) -> str:
    hasher = hashlib.sha256()
    for parameter in optimizer.param_groups[0]["params"]:
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            value = state[key]
            if torch.is_tensor(value):
                hasher.update(key.encode("utf-8"))
                hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


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


def _batch_from_copy(trainer) -> list:
    generator = torch.Generator()
    generator.set_state(trainer.generator.get_state())
    return trainer._batches(generator=generator)


def _term_values(terms_by_case: list[dict]) -> dict:
    result = {}
    n = len(terms_by_case)
    for name in _TERMS:
        values = [
            float(entry["terms"][name].total.detach())
            for entry in terms_by_case
        ]
        result[name] = {
            "raw_mean": sum(values) / n,
            "mean_semantics": (
                "theta_cell_weighted" if name == "contact_data"
                else "unweighted"
            ),
            "per_case": {
                entry["case_id"]: value
                for entry, value in zip(terms_by_case, values)
            },
        }
    return result


def _weighted_terms(raw: dict, lambda_pde: float, lambda_data: float,
                    contact_data_weight: float) -> dict:
    return {
        "pde": lambda_pde * raw["pde"]["raw_mean"],
        "contact_ic": raw["contact_ic"]["raw_mean"],
        "interface": raw["interface"]["raw_mean"],
        "contact_data": (
            lambda_data * contact_data_weight * raw["contact_data"]["raw_mean"]
        ),
    }


def _preflight(trainer, checkpoint: Path, output: Path) -> dict:
    start_hash = _sha256(checkpoint)
    trainer.load(checkpoint)
    initial_contact = _tensor_hash(trainer.contact_net.parameters())
    initial_air = _tensor_hash(trainer.air_net.parameters())
    initial_opt_contact = _optimizer_hash(trainer.opt_contact)
    initial_opt_air = _optimizer_hash(trainer.opt_air)
    initial_contact_step = int(next(iter(trainer.opt_contact.state.values()))["step"])
    initial_air_step = int(next(iter(trainer.opt_air.state.values()))["step"])
    initial_generator = trainer.generator.get_state().clone()
    initial_torch_rng = torch.get_rng_state().clone()
    first_batch = _batch_from_copy(trainer)
    first_batch_hash = _batch_digest(first_batch)

    # Standard AirNet stage from a fresh restore.
    trainer.load(checkpoint)
    batches_standard = _batch_from_copy(trainer)
    air_standard, lambda_pde, lambda_data = trainer._air_update(batches_standard)
    air_state_standard = _tensor_hash(trainer.air_net.parameters())
    air_loss_standard = float(air_standard.total.detach())

    # Diagnostic objective from a fresh restore.
    trainer.load(checkpoint)
    batches_d = _batch_from_copy(trainer)
    air_d, lambda_pde_d, lambda_data_d = trainer._air_update(batches_d)
    air_state_d = _tensor_hash(trainer.air_net.parameters())
    air_loss_d = float(air_d.total.detach())
    contact_params = [p for p in trainer.contact_net.parameters() if p.requires_grad]
    contact_terms = [
        trainer.contact_stage_terms(case, batch)
        for case, batch in batches_d
    ]
    aggregate_data = torch.zeros((), dtype=torch.float64)
    for terms in contact_terms:
        aggregate_data = aggregate_data + terms["contact_data"].total
    aggregate_data = aggregate_data / len(contact_terms)
    coefficients = diag._coefficients(trainer)
    expected_vector, expected_finite = diag._grad(
        coefficients["contact_data"] * aggregate_data, contact_params
    )
    captured = {}

    def _capture(_optimizer, _args, _kwargs):
        captured["gradient"] = torch.cat([
            parameter.grad.detach().reshape(-1)
            for parameter in contact_params
        ])

    air_before_contact = _tensor_hash(trainer.air_net.parameters())
    handle = trainer.opt_contact.register_step_pre_hook(_capture)
    try:
        contact_total, terms_by_case = trainer.contact_update(
            batches_d, lambda_pde=lambda_pde_d, lambda_data=lambda_data_d,
            objective="contact_data_only", return_terms=True,
        )
    finally:
        handle.remove()
    air_after_contact = _tensor_hash(trainer.air_net.parameters())
    actual_vector = captured.get("gradient")
    gradient_error = (
        None if actual_vector is None
        else float((actual_vector - expected_vector).norm())
    )
    raw = _term_values(terms_by_case)
    weighted = _weighted_terms(
        raw, lambda_pde_d, lambda_data_d,
        float(trainer.config.contact_data_weight),
    )
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": start_hash,
        "initial_contact_param_hash": initial_contact,
        "initial_air_param_hash": initial_air,
        "initial_contact_optimizer_hash": initial_opt_contact,
        "initial_air_optimizer_hash": initial_opt_air,
        "initial_contact_step": initial_contact_step,
        "initial_air_step": initial_air_step,
        "initial_generator_hash": hashlib.sha256(
            initial_generator.numpy().tobytes()
        ).hexdigest(),
        "initial_torch_rng_hash": hashlib.sha256(
            initial_torch_rng.numpy().tobytes()
        ).hexdigest(),
        "first_batch_sha256": first_batch_hash,
        "air_stage_standard": {
            "loss": air_loss_standard,
            "state_hash": air_state_standard,
            "lambda_pde": float(lambda_pde),
            "lambda_data": float(lambda_data),
        },
        "air_stage_diagnostic": {
            "loss": air_loss_d,
            "state_hash": air_state_d,
            "lambda_pde": float(lambda_pde_d),
            "lambda_data": float(lambda_data_d),
        },
        "air_stage_equal": (
            air_loss_standard == air_loss_d
            and air_state_standard == air_state_d
            and float(lambda_pde) == float(lambda_pde_d)
            and float(lambda_data) == float(lambda_data_d)
        ),
        "first_batch_equal_to_standard": first_batch_hash == _batch_digest(
            batches_standard
        ),
        "contact_data_coefficient": coefficients["contact_data"],
        "expected_contact_data_coefficient": (
            float(lambda_data) * float(trainer.config.contact_data_weight)
        ),
        "contact_data_loss": {
            "raw_mean": raw["contact_data"]["raw_mean"],
            "weighted_mean": weighted["contact_data"],
        },
        "other_term_values": {
            name: raw[name]["raw_mean"] for name in _TERMS
        },
        "diagnostic_contact_total": float(contact_total.total.detach()),
        "gradient_abs_error": gradient_error,
        "gradient_relative_error": (
            None if gradient_error is None
            else gradient_error / max(1.0, float(expected_vector.norm()))
        ),
        "airnet_frozen_during_contact": air_before_contact == air_after_contact,
        "contact_step_after": int(
            next(iter(trainer.opt_contact.state.values()))["step"]
        ),
        "air_step_after": int(next(iter(trainer.opt_air.state.values()))["step"]),
    }
    payload["checks"] = {
        "air_stage_equal": payload["air_stage_equal"],
        "first_batch_equal": payload["first_batch_equal_to_standard"],
        "coefficient_09": abs(
            payload["contact_data_coefficient"] - 0.9
        ) <= 1.0e-12,
        "contact_only_gradient": (
            gradient_error is not None
            and gradient_error <= 1.0e-9 * max(1.0, float(expected_vector.norm()))
        ),
        "airnet_frozen": payload["airnet_frozen_during_contact"],
        "adam_steps_advanced_once": (
            payload["contact_step_after"] == initial_contact_step + 1
            and payload["air_step_after"] == initial_air_step + 1
        ),
    }
    (output / "preflight.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _train(trainer, checkpoint: Path, target: int, output: Path,
           checkpoint_every: int, objective: str) -> dict:
    trainer.load(checkpoint)
    start_loop = int(trainer.loop)
    if target < start_loop:
        raise TrainingError(f"target {target} is below start {start_loop}")
    if objective != "contact_data_only":
        raise TrainingError("this diagnostic only supports contact_data_only")
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, output / "checkpoints" / f"loop_{start_loop}.pt")
    history = []
    while trainer.loop < target:
        batches = trainer._batches()
        air_total, lambda_pde, lambda_data = trainer._air_update(batches)
        contact_total, terms_by_case = trainer.contact_update(
            batches, lambda_pde=lambda_pde, lambda_data=lambda_data,
            objective=objective, return_terms=True,
        )
        trainer.loop += 1
        raw = _term_values(terms_by_case)
        weighted = _weighted_terms(
            raw, lambda_pde, lambda_data,
            float(trainer.config.contact_data_weight),
        )
        record = {
            "loop": int(trainer.loop),
            "objective": objective,
            "phase_weighting": "always_on",
            "air_loss": float(air_total.total.detach()),
            "contact_target": float(contact_total.total.detach()),
            "lambda_pde": float(lambda_pde),
            "lambda_data": float(lambda_data),
            "contact_data_weight": float(trainer.config.contact_data_weight),
            "contact_data_coefficient": (
                float(lambda_data) * float(trainer.config.contact_data_weight)
            ),
            "contact_learning_rate": float(
                trainer.opt_contact.param_groups[0]["lr"]
            ),
            "air_learning_rate": float(trainer.opt_air.param_groups[0]["lr"]),
            "terms_raw_mean": {
                name: raw[name]["raw_mean"] for name in _TERMS
            },
            "terms_weighted_mean": weighted,
            "terms_per_case": {
                name: raw[name]["per_case"] for name in _TERMS
            },
            "air_parts": dict(air_total.parts),
            "counts": {**{f"air.{k}": v for k, v in air_total.counts.items()},
                       **{f"contact.{k}": v for k, v in contact_total.counts.items()}},
        }
        history.append(record)
        trainer.history.loops.append(record)
        if checkpoint_every > 0 and trainer.loop % checkpoint_every == 0:
            trainer.save(output / "checkpoints" / f"loop_{trainer.loop}.pt")
            print(f"loop {trainer.loop} checkpoint written")
    trainer.save(output / "last.pt")
    result = {
        "loops": int(trainer.loop),
        "start_loop": start_loop,
        "target_loop": target,
        "objective": objective,
        "phase_weighting": "always_on",
        "effective_learning_rates": trainer.effective_learning_rates(),
        "history": history,
        "last_validation": trainer.evaluate(),
    }
    (output / "train_result.json").write_text(
        json.dumps(result, indent=2, default=diag._jsonable), encoding="utf-8"
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-loops", type=int, default=4500)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--objective", choices=("contact_data_only",), default="contact_data_only"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer, meta = diag._build_trainer(args.config)
    preflight = _preflight(trainer, Path(args.checkpoint), output)
    print(json.dumps(preflight["checks"], indent=2))
    if not all(preflight["checks"].values()):
        return 1
    if args.preflight_only:
        return 0
    result = _train(
        trainer, Path(args.checkpoint), int(args.max_loops), output,
        int(args.checkpoint_every), args.objective,
    )
    manifest = {
        "format": "spherical_blast_pinn.contact_objective_ablation",
        "version": 1,
        "config": args.config,
        "config_sha256": _sha256(Path(args.config)),
        "start_checkpoint": str(args.checkpoint),
        "start_checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "start_loop": int(result["start_loop"]),
        "target_loop": int(result["target_loop"]),
        "objective": args.objective,
        "contact_learning_rate": result["effective_learning_rates"]["contact"],
        "air_learning_rate": result["effective_learning_rates"]["air"],
        "sampling": dict(meta["section"].get("sampling", {})),
        "preflight_checks": preflight["checks"],
        "diagnostic_only": True,
        "formal_default_changed": False,
    }
    (output / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "loops": result["loops"],
        "output": str(output),
        "effective_learning_rates": result["effective_learning_rates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
