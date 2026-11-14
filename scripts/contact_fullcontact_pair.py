#!/usr/bin/env python3
"""Paired A/B one-step experiment: random contact sampling vs all legal labels.

A uses the official training batch, including its random contact-label subset.
B starts from the same checkpoint, receives the same non-contact batch and the
same AirNet update, but its ContactNet update replaces only the contact labels
with every legal training label (per-case mean, then equal-case mean).

Each arm is an independent one-step update from the checkpoint; the checkpoint
file is never written.  The script records the fixed reference metrics, the
current objective terms, exact coverage counts and paired A/B differences.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402


def _sha256(path: Path) -> str:
    return diag._sha256(path)


def _full_labels(case) -> dict:
    """All legal contact labels for a training case, with exclusions."""
    indices = case.contact.indices()
    times = np.asarray(case.contact.times[indices], dtype=np.float64)
    in_time = (times >= float(case.t0)) & (times <= float(case.t_end))
    has_slope = np.asarray(
        [case.shock.has_slope(float(t)) for t in times], dtype=bool
    ) if times.size else np.zeros(0, dtype=bool)
    keep = in_time & has_slope
    chosen = indices[keep]
    return {
        "times": np.asarray(case.contact.times[chosen], dtype=np.float64),
        "low": np.asarray(case.contact.low[chosen], dtype=np.float64),
        "up": np.asarray(case.contact.up[chosen], dtype=np.float64),
        "n": int(chosen.size),
        "excluded_outside_time": int(np.count_nonzero(~in_time)),
        "excluded_no_slope": int(np.count_nonzero(in_time & ~has_slope)),
        "total_frame_candidates": int(indices.size),
    }


def _all_label_batches(trainer, batches, labels: dict):
    result = []
    for case, batch in batches:
        current = labels[case.case_id]
        result.append((
            case,
            dataclasses.replace(
                batch,
                t_contact=torch.as_tensor(current["times"], dtype=torch.float64),
                low_contact=torch.as_tensor(current["low"], dtype=torch.float64),
                up_contact=torch.as_tensor(current["up"], dtype=torch.float64),
            ),
        ))
    return result


def _state_hash(module) -> str:
    hasher = hashlib.sha256()
    for parameter in module.parameters():
        hasher.update(parameter.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _contact_parameters(trainer) -> list[torch.nn.Parameter]:
    return [p for p in trainer.contact_net.parameters() if p.requires_grad]


def _flatten(parameters) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def _rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def _terms_before_after(trainer, batches) -> dict:
    return diag._static_record(
        trainer, batches, compute_gradients=False, threshold=1.0e-12
    )


def _optimizer_hash(optimizer) -> str:
    hasher = hashlib.sha256()
    for state in optimizer.state.values():
        for key in sorted(state):
            value = state[key]
            if torch.is_tensor(value):
                hasher.update(key.encode("utf-8"))
                hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _finite_reference(reference: dict) -> bool:
    return all(
        reference.get(key) is None or math.isfinite(float(reference[key]))
        for key in ("macro_mse", "pooled_mse", "coverage")
    )


def _total_weighted(record: dict) -> float | None:
    values = [
        record["terms"][name].get("weighted_loss")
        for name in diag._TERM_NAMES
    ]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return float(sum(values))


def _trial(trainer, checkpoint: Path, reference, labels, args, trial: int) -> dict:
    seed = int(args.seed) + trial * 104729
    generator = torch.Generator().manual_seed(seed)
    # One original training batch; B reuses the same non-contact tensors.
    batches_a = trainer._batches(generator=generator)
    batches_b = _all_label_batches(trainer, batches_a, labels)
    full_total = sum(item["n"] for item in labels.values())
    if full_total == 0:
        raise RuntimeError("all legal training contact-label sets are empty")

    # ---- A: official random contact labels --------------------------------
    trainer.load(checkpoint)
    theta_start = _flatten(_contact_parameters(trainer))
    start_state = _state_hash(trainer.contact_net)
    start_param_initial = _state_hash(trainer.contact_net)
    start_opt_initial = _optimizer_hash(trainer.opt_contact)
    air_a, lambda_pde, lambda_data = trainer._air_update(batches_a)
    air_state_a = _state_hash(trainer.air_net)
    before_a = _terms_before_after(trainer, batches_a)
    ref_before_a = diag._reference_metrics(trainer, reference)
    common_before_a = diag._batch_metrics(trainer, batches_a)
    full_before_a = diag._batch_metrics(trainer, batches_b)
    pre_state_a = _state_hash(trainer.contact_net)
    pre_opt_a = _optimizer_hash(trainer.opt_contact)
    started = time.perf_counter()
    trainer.contact_update(
        batches_a, lambda_pde=lambda_pde, lambda_data=lambda_data
    )
    time_a = time.perf_counter() - started
    after_a = _terms_before_after(trainer, batches_a)
    ref_after_a = diag._reference_metrics(trainer, reference)
    common_after_a = diag._batch_metrics(trainer, batches_a)
    full_after_a = diag._batch_metrics(trainer, batches_b)
    theta_after_a = _flatten(_contact_parameters(trainer))
    delta_a = theta_after_a - theta_start

    # ---- B: same non-contact batch, all legal contact labels -------------
    trainer.load(checkpoint)
    if _state_hash(trainer.contact_net) != start_state:
        raise RuntimeError("B did not restore the same initial ContactNet state")
    start_param_initial_b = _state_hash(trainer.contact_net)
    start_opt_initial_b = _optimizer_hash(trainer.opt_contact)
    air_b, lambda_pde_b, lambda_data_b = trainer._air_update(batches_a)
    air_state_b = _state_hash(trainer.air_net)
    before_b = _terms_before_after(trainer, batches_b)
    ref_before_b = diag._reference_metrics(trainer, reference)
    common_before_b = diag._batch_metrics(trainer, batches_a)
    full_before_b = diag._batch_metrics(trainer, batches_b)
    pre_state_b = _state_hash(trainer.contact_net)
    pre_opt_b = _optimizer_hash(trainer.opt_contact)
    started = time.perf_counter()
    trainer.contact_update(
        batches_b, lambda_pde=lambda_pde_b, lambda_data=lambda_data_b
    )
    time_b = time.perf_counter() - started
    after_b = _terms_before_after(trainer, batches_b)
    ref_after_b = diag._reference_metrics(trainer, reference)
    common_after_b = diag._batch_metrics(trainer, batches_a)
    full_after_b = diag._batch_metrics(trainer, batches_b)
    theta_after_b = _flatten(_contact_parameters(trainer))
    delta_b = theta_after_b - theta_start

    def change(before, after):
        return diag._reference_change(before, after)

    return {
        "trial": trial,
        "seed": seed,
        "checkpoint_loop": int(trainer.loop),
        "lambda_pde": float(lambda_pde),
        "lambda_data": float(lambda_data),
        "contact_data_weight": float(trainer.config.contact_data_weight),
        "full_contact_labels": int(full_total),
        "full_contact_labels_per_case": {
            key: item["n"] for key, item in labels.items()
        },
        "full_contact_exclusions": {
            key: {
                "out_of_time": item["excluded_outside_time"],
                "no_shock_slope": item["excluded_no_slope"],
                "candidate_frames": item["total_frame_candidates"],
            }
            for key, item in labels.items()
        },
        "A": {
            "air_loss": float(air_a.total.detach()),
            "before": before_a,
            "after": after_a,
            "reference_change": change(ref_before_a, ref_after_a),
            "reference_before": ref_before_a,
            "reference_after": ref_after_a,
            "common_A_labels_before": common_before_a,
            "common_A_labels_after": common_after_a,
            "full_labels_before": full_before_a,
            "full_labels_after": full_after_a,
            "parameter_delta_l2": float(delta_a.norm()),
            "time_s": time_a,
        },
        "B": {
            "air_loss": float(air_b.total.detach()),
            "before": before_b,
            "after": after_b,
            "reference_change": change(ref_before_b, ref_after_b),
            "reference_before": ref_before_b,
            "reference_after": ref_after_b,
            "common_A_labels_before": common_before_b,
            "common_A_labels_after": common_after_b,
            "full_labels_before": full_before_b,
            "full_labels_after": full_after_b,
            "parameter_delta_l2": float(delta_b.norm()),
            "time_s": time_b,
        },
        "paired": {
            "initial_param_equal": bool(start_param_initial == start_param_initial_b),
            "initial_optimizer_equal": bool(start_opt_initial == start_opt_initial_b),
            "pre_contact_param_equal": bool(pre_state_a == pre_state_b),
            "pre_contact_optimizer_equal": bool(pre_opt_a == pre_opt_b),
            "air_state_equal": bool(air_state_a == air_state_b),
            "air_loss_equal": bool(
                torch.equal(air_a.total.detach(), air_b.total.detach())
            ),
            "initial_state_equal": bool(pre_state_a == pre_state_b),
            "initial_state_a": pre_state_a,
            "initial_state_b": pre_state_b,
            "pre_optimizer_a": pre_opt_a,
            "pre_optimizer_b": pre_opt_b,
            "reference_macro_mse_difference_B_minus_A": (
                None if ref_before_a["macro_mse"] is None
                or ref_before_b["macro_mse"] is None
                or ref_after_a["macro_mse"] is None
                or ref_after_b["macro_mse"] is None else
                (ref_after_b["macro_mse"] - ref_before_b["macro_mse"])
                - (ref_after_a["macro_mse"] - ref_before_a["macro_mse"])
            ),
            "reference_pooled_mse_difference_B_minus_A": (
                None if ref_before_a["pooled_mse"] is None
                or ref_before_b["pooled_mse"] is None
                or ref_after_a["pooled_mse"] is None
                or ref_after_b["pooled_mse"] is None else
                (ref_after_b["pooled_mse"] - ref_before_b["pooled_mse"])
                - (ref_after_a["pooled_mse"] - ref_before_a["pooled_mse"])
            ),
            "reference_coverage_difference_B_minus_A": (
                None if ref_before_a["coverage"] is None
                or ref_before_b["coverage"] is None
                or ref_after_a["coverage"] is None
                or ref_after_b["coverage"] is None else
                (ref_after_b["coverage"] - ref_before_b["coverage"])
                - (ref_after_a["coverage"] - ref_before_a["coverage"])
            ),
            "full_label_macro_mse_difference_B_minus_A": (
                None if full_before_a["macro_mse"] is None
                or full_before_b["macro_mse"] is None
                or full_after_a["macro_mse"] is None
                or full_after_b["macro_mse"] is None else
                (full_after_b["macro_mse"] - full_before_b["macro_mse"])
                - (full_after_a["macro_mse"] - full_before_a["macro_mse"])
            ),
            "common_A_macro_mse_difference_B_minus_A": (
                None if common_before_a["macro_mse"] is None
                or common_before_b["macro_mse"] is None
                or common_after_a["macro_mse"] is None
                or common_after_b["macro_mse"] is None else
                (common_after_b["macro_mse"] - common_before_b["macro_mse"])
                - (common_after_a["macro_mse"] - common_before_a["macro_mse"])
            ),
            "parameter_delta_l2_difference_B_minus_A": (
                float(delta_b.norm()) - float(delta_a.norm())
            ),
        },
        "non_contact_sampling_equal": _non_contact_equal(batches_a, batches_b),
        "non_finite": bool(
            not _finite_reference(ref_before_a) or not _finite_reference(ref_after_a)
            or not _finite_reference(ref_before_b) or not _finite_reference(ref_after_b)
            or not _finite_record(before_a) or not _finite_record(after_a)
            or not _finite_record(before_b) or not _finite_record(after_b)
        ),
    }


def _non_contact_equal(batches_a, batches_b) -> bool:
    for (case_a, batch_a), (case_b, batch_b) in zip(batches_a, batches_b):
        if case_a.case_id != case_b.case_id:
            return False
        for field in ("t_pde", "zeta_pde", "t_ic", "zeta_ic", "t_right",
                      "t_data", "radius_data", "rho_data", "u_data", "p_data",
                      "valid_rho", "valid_u", "valid_p"):
            if not torch.equal(getattr(batch_a, field), getattr(batch_b, field)):
                return False
    return True


def _finite_record(record: dict) -> bool:
    for term in record.get("terms", {}).values():
        value = term.get("raw_loss")
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--tag", default="best")
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be > 0")
    checkpoint = Path(args.checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    trainer, _ = diag._build_trainer(args.config)
    trainer.load(checkpoint)
    reference = diag._reference_points(trainer)
    labels = {
        case.case_id: _full_labels(case)
        for case in trainer.cases
    }
    label_total = sum(item["n"] for item in labels.values())
    records = []
    failures = 0
    non_finite = False
    started = time.perf_counter()
    for trial in range(int(args.trials)):
        try:
            trial_started = time.perf_counter()
            record = _trial(
                trainer, checkpoint, reference, labels, args, trial
            )
            record["elapsed_s"] = time.perf_counter() - trial_started
            records.append(record)
            if record.get("non_finite"):
                non_finite = True
            print(f"trial {trial + 1}/{args.trials} done")
        except Exception as exc:
            failures += 1
            records.append({
                "trial": trial, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"trial {trial + 1} FAILED: {exc}", file=sys.stderr)
    elapsed = time.perf_counter() - started
    checkpoint_after = _sha256(checkpoint)
    output = {
        "format": "spherical_blast_pinn.contact_fullcontact_pair",
        "version": 1,
        "config": args.config,
        "checkpoint": str(checkpoint),
        "tag": args.tag,
        "checkpoint_loop": int(trainer.loop),
        "checkpoint_sha256": {
            "before": checkpoint_hash,
            "after": checkpoint_after,
            "unchanged": checkpoint_hash == checkpoint_after,
        },
        "requested_trials": int(args.trials),
        "completed_trials": len(records) - failures,
        "failed_trials": failures,
        "non_finite": bool(non_finite),
        "seed": int(args.seed),
        "full_contact_labels": int(label_total),
        "reference_frames_per_trial": int(
            sum(len(point["times"]) for point in reference)
        ),
        "labels": labels,
        "trials": records,
        "elapsed_s": elapsed,
        "rss_mb": _rss_mb(),
        "note": (
            "A uses the official random contact subset; B uses all legal training "
            "contact labels with a per-case mean then equal-case mean. Both start "
            "from the same checkpoint and use the same non-contact batch and AirNet "
            "update. Each arm is an independent one-step update, not continuous "
            "training."
        ),
    }
    output["ok"] = (
        failures == 0
        and not non_finite
        and checkpoint_hash == checkpoint_after
        and label_total > 0
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, default=diag._jsonable),
                   encoding="utf-8")
    print(f"written: {out}")
    if not output["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
