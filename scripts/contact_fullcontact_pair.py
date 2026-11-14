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


def _label_issues(labels: dict, expected_cases: list[str]) -> list[str]:
    """Every expected training case must have a non-empty legal label set."""
    issues: list[str] = []
    missing = [case_id for case_id in expected_cases if case_id not in labels]
    if missing:
        issues.append(f"missing legal label sets for cases: {missing}")
    for case_id in expected_cases:
        item = labels.get(case_id)
        if not isinstance(item, dict):
            continue
        n = item.get("n")
        if not isinstance(n, int) or n <= 0:
            issues.append(f"case {case_id}: legal contact-label set is empty")
        times = item.get("times")
        low = item.get("low")
        up = item.get("up")
        if not all(hasattr(value, "shape") for value in (times, low, up)):
            issues.append(f"case {case_id}: label arrays are missing")
        elif not (len(times) == len(low) == len(up) == int(n or 0)):
            issues.append(f"case {case_id}: label array lengths disagree with n")
    return issues


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


def _pairing_issues(record: dict) -> list[str]:
    """Runtime conditions that make an A/B pair invalid, not merely noisy."""
    issues: list[str] = []
    paired = record.get("paired", {})
    for flag in (
        "initial_param_equal", "initial_optimizer_equal",
        "pre_contact_param_equal", "pre_contact_optimizer_equal",
        "air_state_equal", "air_loss_equal", "initial_state_equal",
    ):
        if paired.get(flag) is not True:
            issues.append(f"pairing condition {flag} is not true")
    if record.get("non_contact_sampling_equal") is not True:
        issues.append("non-contact sampling differs between A and B")
    per_case = record.get("full_contact_labels_per_case") or {}
    empty = [case_id for case_id, n in per_case.items() if int(n) <= 0]
    if empty:
        issues.append(f"empty legal contact-label sets: {empty}")
    return issues


def _trial(trainer, checkpoint: Path, reference, labels, args, trial: int) -> dict:
    seed = int(args.seed) + trial * 104729
    generator = torch.Generator().manual_seed(seed)
    # One original training batch; B reuses the same non-contact tensors.
    batches_a = trainer._batches(generator=generator)
    batches_b = _all_label_batches(trainer, batches_a, labels)
    full_total = sum(item["n"] for item in labels.values())
    empty_cases = [
        case.case_id for case in trainer.cases
        if int(labels.get(case.case_id, {}).get("n", 0)) <= 0
    ]
    if empty_cases:
        raise RuntimeError(
            f"legal contact-label set is empty for training cases: {empty_cases}"
        )
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
    issues = _pairing_issues(record)
    if record["non_finite"]:
        issues.append("non-finite loss or reference metric")
    record["status"] = "failed" if issues else "ok"
    if issues:
        record["error"] = "; ".join(issues)
    return record


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


# ---------------------------------------------------------------------------
# Actual Adam-direction analysis
# ---------------------------------------------------------------------------

_DIRECTION_STEPS = (1.0, 0.5, 0.25, 0.125)
_FD_STEPS = (1.0e-3, 1.0e-4)
_GRAD_TOL = 1.0e-10
_ADAM_TOL = 1.0e-10
_NEAR_ZERO_REL = 1.0e-10


def _set_flat(parameters, vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            n = parameter.numel()
            parameter.copy_(vector[offset:offset + n].view_as(parameter))
            offset += n


def _objective_graph(trainer, batches):
    """Per-case terms with graphs, plus equal-case C and weighted J."""
    rows = []
    for case, batch in batches:
        rows.append((case, batch, trainer.contact_stage_terms(case, batch)))
    coefficients = diag._coefficients(trainer)
    n = len(rows)
    like = torch.zeros((), dtype=torch.float64)
    aggregate = {
        name: sum((row[2][name].total for row in rows), like) / float(n)
        for name in diag._TERM_NAMES
    }
    contact = sum((row[2]["contact_data"].total for row in rows), like) / float(n)
    total = sum(
        coefficients[name] * aggregate[name] for name in diag._TERM_NAMES
    )
    return rows, coefficients, aggregate, contact, total


def _objective_values(trainer, batches, coefficients) -> tuple[float, float, dict]:
    """C(theta), J(theta) and the four raw equal-case term means."""
    rows = []
    for case, batch in batches:
        rows.append(trainer.contact_stage_terms(case, batch))
    n = len(rows)
    like = torch.zeros((), dtype=torch.float64)
    raw = {
        name: float(sum((row[name].total for row in rows), like).detach() / float(n))
        for name in diag._TERM_NAMES
    }
    contact = raw["contact_data"]
    total = sum(coefficients[name] * raw[name] for name in diag._TERM_NAMES)
    return contact, total, raw


def _adam_snapshot(optimizer, parameters) -> list[dict]:
    entries = []
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict):
            raise RuntimeError("ContactNet optimizer state is missing")
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key not in state:
                raise RuntimeError(f"ContactNet optimizer state is missing {key}")
        step = state["step"]
        step_value = int(step.item()) if torch.is_tensor(step) else int(step)
        entries.append({
            "step": step_value,
            "exp_avg": state["exp_avg"].detach().clone(),
            "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
        })
    return entries


def _adam_decomposition(
    optimizer, parameters, snapshots, gradients, gradient_total, delta_actual,
) -> dict:
    """Algebraic Adam reconstruction under the actual second-moment scaling.

    The decomposition fixes the denominator from this step's ``v_t`` and
    splits the numerator into the stored historical first moment and the four
    current weighted term gradients.  It is not a causal "remove one loss"
    experiment.
    """
    group = optimizer.param_groups[0]
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("Adam reconstruction requires exactly one param group")
    unsupported = []
    if float(group.get("weight_decay", 0.0)) != 0.0:
        unsupported.append("nonzero weight_decay")
    if bool(group.get("amsgrad", False)):
        unsupported.append("AMSGrad")
    if bool(group.get("maximize", False)):
        unsupported.append("maximize")
    if unsupported:
        raise RuntimeError(
            "Adam reconstruction does not support: " + ", ".join(unsupported)
        )
    beta1, beta2 = (float(value) for value in group.get("betas", (0.9, 0.999)))
    eps = float(group.get("eps", 1.0e-8))
    lr = float(group["lr"])

    history_parts: list[torch.Tensor] = []
    term_parts = {name: [] for name in diag._TERM_NAMES}
    reconstruction: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []
    bias1_values: list[float] = []
    bias2_values: list[float] = []
    steps: list[int] = []
    offset = 0
    for parameter, snapshot in zip(parameters, snapshots):
        n = parameter.numel()
        index = slice(offset, offset + n)
        offset += n
        current = gradient_total[index]
        step = snapshot["step"] + 1
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        exp_avg = beta1 * snapshot["exp_avg"].reshape(-1) + (1.0 - beta1) * current
        exp_avg_sq = (
            beta2 * snapshot["exp_avg_sq"].reshape(-1)
            + (1.0 - beta2) * current * current
        )
        denominator = torch.sqrt(exp_avg_sq / bias2) + eps
        scale = -lr / bias1
        reconstruction.append(scale * exp_avg / denominator)
        history_parts.append(
            scale * beta1 * snapshot["exp_avg"].reshape(-1) / denominator
        )
        for name in diag._TERM_NAMES:
            term_parts[name].append(
                scale * (1.0 - beta1) * gradients[name][index] / denominator
            )
        denominators.append(denominator)
        bias1_values.append(bias1)
        bias2_values.append(bias2)
        steps.append(step)
    reconstructed = torch.cat(reconstruction)
    history = torch.cat(history_parts)
    terms = {name: torch.cat(parts) for name, parts in term_parts.items()}
    contribution_sum = history + sum(terms.values())
    delta_norm = float(delta_actual.norm())
    recon_error = float((reconstructed - delta_actual).norm())
    contribution_error = float((contribution_sum - delta_actual).norm())
    denom = torch.cat(denominators)
    return {
        "supported": True,
        "lr": lr,
        "betas": [beta1, beta2],
        "eps": eps,
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "amsgrad": bool(group.get("amsgrad", False)),
        "maximize": bool(group.get("maximize", False)),
        "step_values": sorted(set(steps)),
        "bias_correction1_values": sorted(set(round(v, 15) for v in bias1_values)),
        "bias_correction2_values": sorted(set(round(v, 15) for v in bias2_values)),
        "denominator_min": float(denom.min()),
        "denominator_max": float(denom.max()),
        "denominator_median": float(denom.median()),
        "history_norm": float(history.norm()),
        "term_norms": {name: float(terms[name].norm()) for name in diag._TERM_NAMES},
        "reconstructed_norm": float(reconstructed.norm()),
        "delta_norm": delta_norm,
        "reconstruction_abs_error": recon_error,
        "reconstruction_relative_error": (
            recon_error / delta_norm if delta_norm > 0.0 else None
        ),
        "contribution_sum_abs_error": contribution_error,
        "contribution_sum_relative_error": (
            contribution_error / delta_norm if delta_norm > 0.0 else None
        ),
        "_history": history,
        "_terms": terms,
        "_reconstruction": reconstructed,
    }


def _direction_marker(value: float, scale: float) -> str:
    threshold = _NEAR_ZERO_REL * max(1.0, abs(scale))
    if abs(value) <= threshold:
        return "near_zero"
    return "positive" if value > 0.0 else "negative"


def _direction_issues(checks: dict, non_finite: bool) -> list[str]:
    issues = [f"{key}=false" for key, value in checks.items() if not value]
    if non_finite:
        issues.append("non-finite direction quantity")
    return issues


def _direction_trial(trainer, checkpoint: Path, reference, labels, args, trial: int) -> dict:
    seed = int(args.seed) + trial * 104729
    batches_a = trainer._batches(generator=torch.Generator().manual_seed(seed))
    batches_b = _all_label_batches(trainer, batches_a, labels)
    if not _non_contact_equal(batches_a, batches_b):
        raise RuntimeError("B changed a non-contact tensor")
    expected_cases = [case.case_id for case in trainer.cases]
    issues_labels = _label_issues(labels, expected_cases)
    if issues_labels:
        raise RuntimeError("; ".join(issues_labels))

    # ---- A: official random contact labels, kept as the paired reference ----
    trainer.load(checkpoint)
    initial_state_a = _state_hash(trainer.contact_net)
    initial_opt_a = _optimizer_hash(trainer.opt_contact)
    air_a, lambda_pde, lambda_data = trainer._air_update(batches_a)
    air_state_a = _state_hash(trainer.air_net)
    pre_state_a = _state_hash(trainer.contact_net)
    pre_opt_a = _optimizer_hash(trainer.opt_contact)
    a_before = _terms_before_after(trainer, batches_a)
    ref_before_a = diag._reference_metrics(trainer, reference)
    theta_a_before = _flatten(_contact_parameters(trainer))
    started = time.perf_counter()
    trainer.contact_update(
        batches_a, lambda_pde=lambda_pde, lambda_data=lambda_data
    )
    time_a = time.perf_counter() - started
    theta_a_after = _flatten(_contact_parameters(trainer))
    delta_a = theta_a_after - theta_a_before
    a_after = _terms_before_after(trainer, batches_a)
    ref_after_a = diag._reference_metrics(trainer, reference)

    # ---- B: same checkpoint / non-contact batch / AirNet update ------------
    trainer.load(checkpoint)
    initial_state_b = _state_hash(trainer.contact_net)
    initial_opt_b = _optimizer_hash(trainer.opt_contact)
    air_b, lambda_pde_b, lambda_data_b = trainer._air_update(batches_a)
    air_state_b = _state_hash(trainer.air_net)
    pre_state_b = _state_hash(trainer.contact_net)
    pre_opt_b = _optimizer_hash(trainer.opt_contact)
    before_b = _terms_before_after(trainer, batches_b)
    ref_before_b = diag._reference_metrics(trainer, reference)
    params = _contact_parameters(trainer)
    optimizer_order = diag._optimizer_order_ok(
        trainer.opt_contact, trainer.contact_net
    )
    if not optimizer_order["order_matches_module"]:
        raise RuntimeError(f"unsafe ContactNet optimizer order: {optimizer_order}")
    theta_before = _flatten(params)
    snapshots = _adam_snapshot(trainer.opt_contact, params)
    adam_before = diag._adam_summary(trainer.opt_contact)

    rows, coefficients, aggregate, contact_before, total_before = _objective_graph(
        trainer, batches_b
    )
    gradients = {}
    finite_gradients = True
    for name in diag._TERM_NAMES:
        vector, finite = diag._grad(coefficients[name] * aggregate[name], params)
        gradients[name] = vector
        finite_gradients = finite_gradients and finite
    gradient_total, total_finite = diag._grad(total_before, params)
    gradient_contact, contact_finite = diag._grad(contact_before, params)
    finite_gradients = (
        finite_gradients and total_finite and contact_finite
        and bool(torch.isfinite(gradient_total).all())
        and bool(torch.isfinite(gradient_contact).all())
    )
    gradient_sum = sum(gradients.values())
    gradient_sum_error = float((gradient_sum - gradient_total).norm())
    gradient_total_norm = float(gradient_total.norm())
    del rows, aggregate, contact_before, total_before

    captured: dict[str, torch.Tensor] = {}

    def _capture_step(_optimizer, _args, _kwargs):
        captured["gradient"] = _flatten([
            parameter.grad for parameter in params
        ])

    handle = trainer.opt_contact.register_step_pre_hook(_capture_step)
    started = time.perf_counter()
    try:
        trainer.contact_update(
            batches_b, lambda_pde=lambda_pde_b, lambda_data=lambda_data_b
        )
    finally:
        handle.remove()
    time_b = time.perf_counter() - started
    theta_after = _flatten(params)
    delta = theta_after - theta_before
    delta_norm = float(delta.norm())
    adam_after = diag._adam_summary(trainer.opt_contact)
    optimizer_after = _optimizer_hash(trainer.opt_contact)
    after_b = _terms_before_after(trainer, batches_b)
    ref_after_b = diag._reference_metrics(trainer, reference)
    contact_after, total_after, raw_after = _objective_values(
        trainer, batches_b, coefficients
    )
    captured_gradient = captured.get("gradient")
    captured_error = (
        None if captured_gradient is None
        else float((captured_gradient - gradient_total).norm())
    )

    dot_contact = float(torch.dot(gradient_contact, delta))
    dot_total = float(torch.dot(gradient_total, delta))
    contact_change = contact_after - float(
        before_b["terms"]["contact_data"]["raw_loss"]
    )
    total_change = total_after - sum(
        before_b["terms"][name]["weighted_loss"] for name in diag._TERM_NAMES
    )
    dot_contact_a = float(torch.dot(gradient_contact, delta_a))
    dot_total_a = float(torch.dot(gradient_total, delta_a))

    if float(trainer.config.gradient_clip) > 0.0:
        raise RuntimeError(
            "Adam reconstruction does not support gradient_clip > 0"
        )
    decomposition = _adam_decomposition(
        trainer.opt_contact, params, snapshots, gradients, gradient_total, delta
    )
    decomposition["gC_dot_history"] = float(
        torch.dot(gradient_contact, decomposition["_history"])
    )
    decomposition["gC_dot_terms"] = {
        name: float(torch.dot(gradient_contact, decomposition["_terms"][name]))
        for name in diag._TERM_NAMES
    }
    decomposition["gC_dot_reconstruction"] = float(
        torch.dot(gradient_contact, decomposition["_reconstruction"])
    )
    decomposition["gJ_dot_history"] = float(
        torch.dot(gradient_total, decomposition["_history"])
    )
    decomposition["gJ_dot_terms"] = {
        name: float(torch.dot(gradient_total, decomposition["_terms"][name]))
        for name in diag._TERM_NAMES
    }
    decomposition["term_contribution_sum"] = float(sum(
        decomposition["gC_dot_terms"].values()
    ) + decomposition["gC_dot_history"])
    for key in ("_history", "_terms", "_reconstruction"):
        decomposition.pop(key, None)

    post_state_hash = _state_hash(trainer.contact_net)
    post_opt_hash = _optimizer_hash(trainer.opt_contact)
    air_post_step = _state_hash(trainer.air_net)
    rng_post = torch.get_rng_state().clone()

    # ---- same-direction step scaling, on in-memory copies only -------------
    scale_records = []
    for step_scale in _DIRECTION_STEPS:
        theta = theta_before + float(step_scale) * delta
        _set_flat(params, theta)
        value_contact, value_total, _ = _objective_values(
            trainer, batches_b, coefficients
        )
        scale_records.append({
            "scale": float(step_scale),
            "contact": value_contact,
            "total": value_total,
            "contact_change": value_contact - float(
                before_b["terms"]["contact_data"]["raw_loss"]
            ),
            "total_change": value_total - sum(
                before_b["terms"][name]["weighted_loss"]
                for name in diag._TERM_NAMES
            ),
            "contact_first_order": float(step_scale) * dot_contact,
            "total_first_order": float(step_scale) * dot_total,
        })

    fd_records = []
    base_contact = float(before_b["terms"]["contact_data"]["raw_loss"])
    base_total = sum(
        before_b["terms"][name]["weighted_loss"] for name in diag._TERM_NAMES
    )
    for h in _FD_STEPS:
        _set_flat(params, theta_before + h * delta)
        plus_contact, plus_total, _ = _objective_values(
            trainer, batches_b, coefficients
        )
        _set_flat(params, theta_before - h * delta)
        minus_contact, minus_total, _ = _objective_values(
            trainer, batches_b, coefficients
        )
        fd_contact = (plus_contact - minus_contact) / (2.0 * h)
        fd_total = (plus_total - minus_total) / (2.0 * h)
        fd_records.append({
            "h": float(h),
            "contact_fd": fd_contact,
            "total_fd": fd_total,
            "contact_first_order": dot_contact,
            "total_first_order": dot_total,
            "contact_abs_error": abs(fd_contact - dot_contact),
            "total_abs_error": abs(fd_total - dot_total),
            "contact_relative_error": (
                abs(fd_contact - dot_contact) / max(abs(dot_contact), 1.0e-30)
            ),
            "total_relative_error": (
                abs(fd_total - dot_total) / max(abs(dot_total), 1.0e-30)
            ),
        })

    # Restore the actual post-step state; interpolation must not leak.
    _set_flat(params, theta_after)
    restored_state = _state_hash(trainer.contact_net)
    restored_opt = _optimizer_hash(trainer.opt_contact)
    rng_restored = torch.get_rng_state().clone()
    air_restored = _state_hash(trainer.air_net)

    scale_one = next(
        (item for item in scale_records if item["scale"] == 1.0), None
    )
    near_zero_contact = _direction_marker(
        dot_contact, float(gradient_contact.norm() * delta_norm)
    ) == "near_zero"
    checks = {
        "initial_param_equal": initial_state_a == initial_state_b,
        "initial_optimizer_equal": initial_opt_a == initial_opt_b,
        "pre_contact_param_equal": pre_state_a == pre_state_b,
        "pre_contact_optimizer_equal": pre_opt_a == pre_opt_b,
        "air_state_equal": air_state_a == air_state_b,
        "air_loss_equal": bool(
            torch.equal(air_a.total.detach(), air_b.total.detach())
        ),
        "non_contact_sampling_equal": True,
        "optimizer_order_matches_module": optimizer_order["order_matches_module"],
        "air_frozen_during_contact_step": air_post_step == air_state_b,
        "parameters_restored": restored_state == post_state_hash,
        "optimizer_state_unchanged_by_interpolation": restored_opt == post_opt_hash,
        "rng_unchanged_by_interpolation": bool(torch.equal(rng_restored, rng_post)),
        "air_state_unchanged_by_interpolation": air_restored == air_post_step,
        "gradients_finite": bool(finite_gradients),
        "gradient_sum_matches_total": bool(
            gradient_sum_error
            <= _GRAD_TOL * max(1.0, gradient_total_norm)
        ),
        "captured_gradient_matches_total": bool(
            captured_error is not None
            and captured_error <= _GRAD_TOL * max(1.0, gradient_total_norm)
        ),
        "adam_reconstruction_matches_delta": bool(
            decomposition["reconstruction_relative_error"] is not None
            and decomposition["reconstruction_relative_error"] <= _ADAM_TOL
        ),
        "contribution_sum_matches_delta": bool(
            decomposition["contribution_sum_relative_error"] is not None
            and decomposition["contribution_sum_relative_error"] <= _ADAM_TOL
        ),
        "gC_decomposition_matches_dot_delta": bool(
            abs(decomposition["gC_dot_reconstruction"] - dot_contact)
            <= _ADAM_TOL * max(1.0, abs(dot_contact))
        ),
        "gC_contribution_sum_matches": bool(
            abs(decomposition["term_contribution_sum"]
                - decomposition["gC_dot_reconstruction"])
            <= _ADAM_TOL * max(1.0, abs(dot_contact))
        ),
        "step_scale_one_matches_actual": bool(
            scale_one is not None
            and abs(scale_one["contact"] - contact_after) <= 1.0e-12
            and abs(scale_one["total"] - total_after) <= 1.0e-12
        ),
    }
    non_finite = bool(
        not finite_gradients
        or not math.isfinite(dot_contact)
        or not math.isfinite(dot_total)
        or not math.isfinite(contact_change)
        or not math.isfinite(total_change)
        or not math.isfinite(delta_norm)
    )
    record = {
        "trial": trial,
        "seed": seed,
        "checkpoint_loop": int(trainer.loop),
        "lambda_pde": float(lambda_pde_b),
        "lambda_data": float(lambda_data_b),
        "contact_data_weight": float(trainer.config.contact_data_weight),
        "full_contact_labels": int(sum(item["n"] for item in labels.values())),
        "full_contact_labels_per_case": {
            case_id: int(item["n"]) for case_id, item in labels.items()
        },
        "A": {
            "air_loss": float(air_a.total.detach()),
            "before": a_before,
            "after": a_after,
            "reference_before": ref_before_a,
            "reference_after": ref_after_a,
            "reference_change": diag._reference_change(
                ref_before_a, ref_after_a
            ),
            "parameter_delta_l2": float(delta_a.norm()),
            "time_s": time_a,
        },
        "B": {
            "air_loss": float(air_b.total.detach()),
            "before": before_b,
            "after": after_b,
            "reference_before": ref_before_b,
            "reference_after": ref_after_b,
            "reference_change": diag._reference_change(
                ref_before_b, ref_after_b
            ),
            "contact_after": contact_after,
            "total_after": total_after,
            "raw_after": raw_after,
            "parameter_delta_l2": delta_norm,
            "parameter_delta_max_abs": (
                float(delta.abs().max()) if delta.numel() else 0.0
            ),
            "parameter_count": int(delta.numel()),
            "parameter_hash_before": hashlib.sha256(
                theta_before.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "parameter_hash_after": hashlib.sha256(
                theta_after.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "optimizer_hash_before": pre_opt_b,
            "optimizer_hash_after": optimizer_after,
            "time_s": time_b,
        },
        "pairing": checks,
        "non_contact_sampling_equal": True,
        "gradients": {
            "coefficients": coefficients,
            "weighted_term_norms": {
                name: float(gradients[name].norm()) for name in diag._TERM_NAMES
            },
            "term_gradient_sum_norm": float(gradient_sum.norm()),
            "total_gradient_norm": gradient_total_norm,
            "contact_gradient_norm": float(gradient_contact.norm()),
            "term_sum_abs_error": gradient_sum_error,
            "captured_gradient_abs_error": captured_error,
            "finite": bool(finite_gradients),
        },
        "adam": {
            "before": adam_before,
            "after": adam_after,
            "decomposition": decomposition,
        },
        "objectives": {
            "definition": {
                "C": "equal-case mean of the full-label interval MSE",
                "J": "current weighted ContactNet objective on the full-label batch",
            },
            "contact_before": base_contact,
            "total_before": base_total,
            "contact_after": contact_after,
            "total_after": total_after,
            "contact_change": contact_change,
            "total_change": total_change,
            "gC_dot_delta": dot_contact,
            "gJ_dot_delta": dot_total,
            "gC_direction": _direction_marker(
                dot_contact, float(gradient_contact.norm() * delta_norm)
            ),
            "gJ_direction": _direction_marker(
                dot_total, float(gradient_total.norm() * delta_norm)
            ),
            "near_zero": near_zero_contact,
            "gC_dot_delta_A": dot_contact_a,
            "gJ_dot_delta_A": dot_total_a,
            "A_parameter_delta_l2": float(delta_a.norm()),
        },
        "step_scaling": scale_records,
        "finite_difference": fd_records,
        "checks": checks,
        "non_finite": non_finite,
    }
    issues = _direction_issues(checks, non_finite)
    record["status"] = "failed" if issues else "ok"
    if issues:
        record["error"] = "; ".join(issues)
    return record


def _run_direction(trainer, checkpoint, checkpoint_hash, reference, labels, args) -> tuple[int, dict]:
    expected_cases = [case.case_id for case in trainer.cases]
    label_issues = _label_issues(labels, expected_cases)
    label_total = sum(int(item["n"]) for item in labels.values())
    records = []
    failures = 0
    non_finite = False
    started = time.perf_counter()
    for trial in range(int(args.trials)):
        if label_issues:
            failures += 1
            records.append({
                "trial": trial, "status": "failed",
                "error": "; ".join(label_issues),
            })
            continue
        try:
            trial_started = time.perf_counter()
            record = _direction_trial(
                trainer, checkpoint, reference, labels, args, trial
            )
            record["elapsed_s"] = time.perf_counter() - trial_started
            records.append(record)
            if record.get("status") == "failed":
                failures += 1
            if record.get("non_finite"):
                non_finite = True
            print(f"direction trial {trial + 1}/{args.trials} "
                  f"{'done' if record.get('status') != 'failed' else 'FAILED'}")
        except Exception as exc:
            failures += 1
            records.append({
                "trial": trial, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"direction trial {trial + 1} FAILED: {exc}", file=sys.stderr)
    elapsed = time.perf_counter() - started
    checkpoint_after = _sha256(checkpoint)
    reference_frames = int(sum(len(point["times"]) for point in reference))
    output = {
        "format": "spherical_blast_pinn.contact_adam_direction",
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
        "expected_train_cases": expected_cases,
        "label_issues": label_issues,
        "full_contact_labels": int(label_total),
        "full_contact_labels_per_case": {
            key: int(item["n"]) for key, item in labels.items()
        },
        "reference_frames_per_trial": reference_frames,
        "labels": labels,
        "trials": records,
        "elapsed_s": elapsed,
        "rss_mb": _rss_mb(),
        "note": (
            "Actual Adam direction analysis on the full-label B path.  Each trial "
            "starts from the original checkpoint, runs the same AirNet update, "
            "records the four weighted ContactNet gradients and the real Adam "
            "step, then evaluates C/J along the same direction on in-memory "
            "parameter copies.  No continuous training, no learning-rate or "
            "optimizer-state changes."
        ),
    }
    failure_reasons = []
    if failures:
        failure_reasons.append(f"failed trials: {failures}/{int(args.trials)}")
    if non_finite:
        failure_reasons.append("non-finite value encountered")
    if checkpoint_hash != checkpoint_after:
        failure_reasons.append("checkpoint SHA-256 changed during the diagnostic")
    failure_reasons.extend(label_issues)
    output["failure_reasons"] = list(dict.fromkeys(failure_reasons))
    output["ok"] = not output["failure_reasons"] and label_total > 0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, indent=2, default=diag._jsonable), encoding="utf-8"
    )
    print(f"written: {out}")
    return (0 if output["ok"] else 1), output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--tag", default="best")
    parser.add_argument(
        "--analysis", choices=("pair", "direction"), default="pair",
        help="pair: A/B one-step comparison; direction: actual Adam direction",
    )
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
    expected_cases = [case.case_id for case in trainer.cases]
    label_issues = _label_issues(labels, expected_cases)
    if args.analysis == "direction":
        code, _ = _run_direction(
            trainer, checkpoint, checkpoint_hash, reference, labels, args
        )
        return code
    records = []
    failures = 0
    non_finite = False
    started = time.perf_counter()
    for trial in range(int(args.trials)):
        if label_issues:
            failures += 1
            records.append({
                "trial": trial, "status": "failed",
                "error": "; ".join(label_issues),
            })
            continue
        try:
            trial_started = time.perf_counter()
            record = _trial(
                trainer, checkpoint, reference, labels, args, trial
            )
            record["elapsed_s"] = time.perf_counter() - trial_started
            records.append(record)
            if record.get("status") == "failed":
                failures += 1
            if record.get("non_finite"):
                non_finite = True
            print(f"trial {trial + 1}/{args.trials} "
                  f"{'done' if record.get('status') != 'failed' else 'FAILED'}")
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
        "full_contact_labels_per_case": {
            key: int(item["n"]) for key, item in labels.items()
        },
        "expected_train_cases": expected_cases,
        "label_issues": label_issues,
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
    failure_reasons = []
    if failures:
        failure_reasons.append(f"failed trials: {failures}/{int(args.trials)}")
    if non_finite:
        failure_reasons.append("non-finite value encountered")
    if checkpoint_hash != checkpoint_after:
        failure_reasons.append("checkpoint SHA-256 changed during the diagnostic")
    failure_reasons.extend(label_issues)
    output["failure_reasons"] = list(dict.fromkeys(failure_reasons))
    output["ok"] = (
        failures == 0
        and not non_finite
        and checkpoint_hash == checkpoint_after
        and not label_issues
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
