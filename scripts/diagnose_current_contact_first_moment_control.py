#!/usr/bin/env python3
"""Paired A/B one-step control for the current C ContactNet Adam history.

The 80 trials are exactly those of
``diagnose_current_contact_update_decomposition.py``: two checkpoints, two
fixed seeds, twenty independent batches per checkpoint-seed.  Both arms restore
the checkpoint, sample the same official random batch, run the same AirNet
update, and evaluate the same full-training-label C.

A: the original ContactNet Adam step.
B: the same step after zeroing **only** the stored first moment ``exp_avg``.
   ``exp_avg_sq``, ``step``, betas, eps, lr and all network parameters stay
   unchanged; the same current gradient is used.

The script verifies that B's actual delta equals the current-term sum of the
original fixed-denominator decomposition and that ``delta_A - delta_B`` equals
its stored-history component.  It never writes a checkpoint, never trains
continuously, and never loads validation labels.  Any non-finite value or
failed state check makes the process exit nonzero.
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
import diagnose_current_contact_update_decomposition as decomp  # noqa: E402

SEEDS = step.SEEDS
TERMS = step.TERMS
C_CONFIG = step.C_CONFIG
CHECKPOINTS = step.CHECKPOINTS
_TOL = 1.0e-12
_MATCH_TOL = 1.0e-8


def _sha256(path: Path) -> str:
    return step._sha256(path)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    difference = (a - b).abs()
    return float(difference.max()) if difference.numel() else 0.0


def _relative_error(numerator: float, denominator: float):
    if denominator == 0.0:
        return None
    return numerator / denominator


def _snapshot_diff(first, second, key: str) -> float:
    return max(
        (_max_abs_diff(a[key], b[key]) for a, b in zip(first, second)),
        default=0.0,
    )


def _snapshot_steps(snapshots) -> list[int]:
    return sorted({int(item["step"]) for item in snapshots})


def _group_values(optimizer) -> dict:
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("expected exactly one ContactNet parameter group")
    group = optimizer.param_groups[0]
    return {
        "lr": float(group["lr"]),
        "betas": tuple(float(value) for value in group.get("betas", (0.9, 0.999))),
        "eps": float(group.get("eps", 1.0e-8)),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "amsgrad": bool(group.get("amsgrad", False)),
        "maximize": bool(group.get("maximize", False)),
    }


def _optimizer_supported(values: dict) -> tuple[bool, list[str]]:
    unsupported = []
    if values["weight_decay"] != 0.0:
        unsupported.append("weight_decay")
    if values["amsgrad"]:
        unsupported.append("amsgrad")
    if values["maximize"]:
        unsupported.append("maximize")
    return not unsupported, unsupported


def _values(trainer, random_batches, full_batches, coefficients):
    c_random, j_value, raw = step._values(
        trainer, random_batches, coefficients
    )
    c_full = float(step._full_contact_value(trainer, full_batches).detach())
    return {
        "C_random": c_random,
        "C": c_full,
        "J": j_value,
        "terms_raw": dict(raw),
        "terms_weighted": {
            name: coefficients[name] * raw[name] for name in TERMS
        },
    }


def _prepare(trainer, checkpoint, random_batches, full_batches):
    trainer.load(checkpoint)
    coefficients = diag._coefficients(trainer)
    checkpoint_loop = int(trainer.loop)
    initial_contact = step._state_hash(trainer.contact_net)
    initial_air = step._state_hash(trainer.air_net)
    initial_opt = cf._optimizer_hash(trainer.opt_contact)
    air_total, lambda_pde, lambda_data = trainer._air_update(random_batches)
    air_after = step._state_hash(trainer.air_net)
    air_opt_after = cf._optimizer_hash(trainer.opt_air)
    params = cf._contact_parameters(trainer)
    theta0 = cf._flatten(params)
    pre_contact = step._state_hash(trainer.contact_net)
    pre_opt = cf._optimizer_hash(trainer.opt_contact)
    snapshots = cf._adam_snapshot(trainer.opt_contact, params)
    values = _values(trainer, random_batches, full_batches, coefficients)
    return {
        "checkpoint_loop": checkpoint_loop,
        "initial_contact_state": initial_contact,
        "initial_air_state": initial_air,
        "initial_contact_optimizer": initial_opt,
        "air_after_state": air_after,
        "air_after_optimizer": air_opt_after,
        "air_loss": float(air_total.total.detach()),
        "lambda_pde": float(lambda_pde),
        "lambda_data": float(lambda_data),
        "contact_state_before_update": pre_contact,
        "contact_optimizer_before_update": pre_opt,
        "snapshots_before": snapshots,
        "theta0": theta0,
        "params": params,
        "coefficients": coefficients,
        "base": values,
        "group": _group_values(trainer.opt_contact),
    }


def _capture_update(trainer, random_batches, lambda_pde, lambda_data, params):
    captured: dict[str, torch.Tensor] = {}

    def _hook(_optimizer, _args, _kwargs):
        captured["gradient"] = cf._flatten([
            parameter.grad for parameter in params
        ])

    handle = trainer.opt_contact.register_step_pre_hook(_hook)
    started = time.perf_counter()
    try:
        trainer.contact_update(
            random_batches, lambda_pde=lambda_pde, lambda_data=lambda_data
        )
    finally:
        handle.remove()
    return captured, time.perf_counter() - started


def _post_update(trainer, prep, random_batches, full_batches, coefficients):
    theta = cf._flatten(prep["params"])
    delta = theta - prep["theta0"]
    values = _values(trainer, random_batches, full_batches, coefficients)
    base = prep["base"]
    return {
        "theta": theta,
        "delta": delta,
        "values": values,
        "C_change": values["C"] - base["C"],
        "J_change": values["J"] - base["J"],
        "delta_norm": float(delta.norm()),
        "post_contact_state": step._state_hash(trainer.contact_net),
        "post_air_state": step._state_hash(trainer.air_net),
        "post_optimizer_hash": cf._optimizer_hash(trainer.opt_contact),
        "snapshots_after": cf._adam_snapshot(
            trainer.opt_contact, prep["params"]
        ),
    }


def _zero_first_moment(trainer, params):
    before = cf._adam_snapshot(trainer.opt_contact, params)
    for parameter in params:
        state = trainer.opt_contact.state.get(parameter)
        if not isinstance(state, dict) or "exp_avg" not in state:
            raise RuntimeError("ContactNet Adam state is missing exp_avg")
        state["exp_avg"].zero_()
    after = cf._adam_snapshot(trainer.opt_contact, params)
    return before, after


def _trial(trainer, checkpoint, labels, seed, trial):
    trial_seed = int(seed) + int(trial) * 104729
    generator = torch.Generator().manual_seed(trial_seed)
    random_batches = trainer._batches(generator=generator)
    full_batches = step._full_batches(trainer, random_batches, labels)

    # ---- A: original ContactNet Adam step ---------------------------------
    prep_a = _prepare(
        trainer, checkpoint, random_batches, full_batches
    )
    coefficients = prep_a["coefficients"]
    supported, unsupported = _optimizer_supported(prep_a["group"])
    if not supported:
        raise RuntimeError(
            "first-moment control does not support: "
            + ", ".join(unsupported)
        )
    if float(trainer.config.gradient_clip) > 0.0:
        raise RuntimeError("first-moment control requires gradient_clip == 0")
    bundle = decomp._gradient_bundle(
        trainer, random_batches, full_batches, coefficients
    )
    captured_a, time_a = _capture_update(
        trainer, random_batches, prep_a["lambda_pde"],
        prep_a["lambda_data"], prep_a["params"],
    )
    post_a = _post_update(
        trainer, prep_a, random_batches, full_batches, coefficients
    )
    decomposition = cf._adam_decomposition(
        trainer.opt_contact, prep_a["params"], prep_a["snapshots_before"],
        bundle["gradients"], bundle["g_total"], post_a["delta"],
    )
    history = decomposition["_history"]
    term_deltas = decomposition["_terms"]
    current_terms = sum(
        term_deltas.values(), torch.zeros_like(history)
    )

    # ---- B: same checkpoint, batch, AirNet update, gradient; exp_avg=0 ----
    prep_b = _prepare(
        trainer, checkpoint, random_batches, full_batches
    )
    snapshots_b_before_zero = cf._adam_snapshot(
        trainer.opt_contact, prep_b["params"]
    )
    snapshots_b_after_zero = None
    _, snapshots_b_after_zero = _zero_first_moment(
        trainer, prep_b["params"]
    )
    captured_b, time_b = _capture_update(
        trainer, random_batches, prep_b["lambda_pde"],
        prep_b["lambda_data"], prep_b["params"],
    )
    post_b = _post_update(
        trainer, prep_b, random_batches, full_batches,
        prep_b["coefficients"],
    )

    # ---- exact consistency checks -----------------------------------------
    g_contact = bundle["g_contact"]
    g_total = bundle["g_total"]
    delta_a = post_a["delta"]
    delta_b = post_b["delta"]
    gradient_a = captured_a.get("gradient")
    gradient_b = captured_b.get("gradient")
    gradient_diff = (
        None if gradient_a is None or gradient_b is None
        else _max_abs_diff(gradient_a, gradient_b)
    )
    gradient_scale = (
        None if gradient_a is None else float(gradient_a.norm())
    )
    gradient_rel = (
        None if gradient_diff is None or not gradient_scale
        else gradient_diff / gradient_scale
    )
    capture_a_error = (
        None if gradient_a is None
        else float((gradient_a - g_total).norm())
    )
    capture_b_error = (
        None if gradient_b is None
        else float((gradient_b - g_total).norm())
    )
    capture_a_rel = (
        None if capture_a_error is None or not gradient_scale
        else capture_a_error / gradient_scale
    )
    capture_b_rel = (
        None if capture_b_error is None or not gradient_scale
        else capture_b_error / gradient_scale
    )
    theta_diff = _max_abs_diff(prep_a["theta0"], prep_b["theta0"])
    base_c_diff = abs(
        prep_a["base"]["C"] - prep_b["base"]["C"]
    )
    base_j_diff = abs(
        prep_a["base"]["J"] - prep_b["base"]["J"]
    )
    air_loss_diff = abs(prep_a["air_loss"] - prep_b["air_loss"])
    lambda_diff = max(
        abs(prep_a["lambda_pde"] - prep_b["lambda_pde"]),
        abs(prep_a["lambda_data"] - prep_b["lambda_data"]),
    )
    b_delta_error = float((delta_b - current_terms).norm())
    b_delta_rel = _relative_error(b_delta_error, float(delta_b.norm()))
    history_error = float((delta_a - delta_b - history).norm())
    history_rel = _relative_error(
        history_error, float(history.norm())
    )
    exp_avg_sq_pre_diff = _snapshot_diff(
        prep_a["snapshots_before"], snapshots_b_before_zero,
        "exp_avg_sq",
    )
    step_pre_equal = (
        _snapshot_steps(prep_a["snapshots_before"])
        == _snapshot_steps(snapshots_b_before_zero)
    )
    exp_avg_sq_post_diff = _snapshot_diff(
        post_a["snapshots_after"], post_b["snapshots_after"],
        "exp_avg_sq",
    )
    step_post_equal = (
        _snapshot_steps(post_a["snapshots_after"])
        == _snapshot_steps(post_b["snapshots_after"])
    )
    zero_norm = max(
        (float(item["exp_avg"].norm()) for item in snapshots_b_after_zero),
        default=0.0,
    )
    group_a = prep_a["group"]
    group_pre_equal = prep_a["group"] == prep_b["group"]
    trainer.load(checkpoint)
    group_b = _group_values(trainer.opt_contact)
    betas_equal = tuple(group_a["betas"]) == tuple(group_b["betas"])
    eps_equal = group_a["eps"] == group_b["eps"]
    lr_equal = group_a["lr"] == group_b["lr"]
    group_restored_equal = group_a == group_b

    checks = {
        "initial_contact_state_equal": (
            prep_a["initial_contact_state"] == prep_b["initial_contact_state"]
        ),
        "initial_air_state_equal": (
            prep_a["initial_air_state"] == prep_b["initial_air_state"]
        ),
        "initial_contact_optimizer_equal": (
            prep_a["initial_contact_optimizer"]
            == prep_b["initial_contact_optimizer"]
        ),
        "pre_contact_state_equal": (
            prep_a["contact_state_before_update"]
            == prep_b["contact_state_before_update"]
        ),
        "pre_contact_optimizer_equal": (
            prep_a["contact_optimizer_before_update"]
            == prep_b["contact_optimizer_before_update"]
        ),
        "pre_theta_equal": theta_diff == 0.0,
        "air_state_equal": (
            prep_a["air_after_state"] == prep_b["air_after_state"]
        ),
        "air_optimizer_equal": (
            prep_a["air_after_optimizer"] == prep_b["air_after_optimizer"]
        ),
        "air_loss_equal": air_loss_diff == 0.0,
        "lambda_equal": lambda_diff == 0.0,
        "group_pre_equal": group_pre_equal,
        "base_C_equal": base_c_diff == 0.0,
        "base_J_equal": base_j_diff == 0.0,
        "gradient_equal": gradient_diff == 0.0,
        "A_gradient_matches_decomposition": (
            capture_a_rel is not None and capture_a_rel <= _MATCH_TOL
        ),
        "B_gradient_matches_decomposition": (
            capture_b_rel is not None and capture_b_rel <= _MATCH_TOL
        ),
        "exp_avg_sq_before_equal": exp_avg_sq_pre_diff == 0.0,
        "step_before_equal": step_pre_equal,
        "exp_avg_sq_after_equal": exp_avg_sq_post_diff == 0.0,
        "step_after_equal": step_post_equal,
        "betas_unchanged": betas_equal,
        "eps_unchanged": eps_equal,
        "lr_unchanged": lr_equal,
        "group_restored_unchanged": group_restored_equal,
        "b_exp_avg_zero": zero_norm == 0.0,
        "B_delta_matches_current_terms": (
            b_delta_rel is not None and b_delta_rel <= _MATCH_TOL
        ),
        "A_minus_B_matches_history": (
            history_rel is not None and history_rel <= _MATCH_TOL
        ),
        "A_decomposition_reconstructs": (
            decomposition.get("reconstruction_relative_error") is not None
            and abs(decomposition["reconstruction_relative_error"])
            <= _MATCH_TOL
        ),
        "A_decomposition_contribution": (
            decomposition.get("contribution_sum_relative_error") is not None
            and abs(decomposition["contribution_sum_relative_error"])
            <= _MATCH_TOL
        ),
        "A_air_frozen_during_contact": (
            post_a["post_air_state"] == prep_a["air_after_state"]
        ),
        "B_air_frozen_during_contact": (
            post_b["post_air_state"] == prep_b["air_after_state"]
        ),
        "finite_gradients": bool(bundle["finite"]),
        "finite_values": all(
            math.isfinite(float(record[key]))
            for record in (prep_a["base"], post_a["values"],
                           prep_b["base"], post_b["values"])
            for key in ("C", "J")
        ) and all(
            math.isfinite(float(value))
            for record in (prep_a["base"], post_a["values"],
                           prep_b["base"], post_b["values"])
            for value in record["terms_raw"].values()
        ),
        "finite_delta": all(
            bool(torch.isfinite(value).all())
            for value in (delta_a, delta_b, history, current_terms)
        ) and all(
            math.isfinite(float(value.norm()))
            for value in (delta_a, delta_b, history, current_terms)
        ),
    }
    status = "ok" if all(checks.values()) else "failed"

    def projection(gradient, delta):
        return {
            "gC_dot_delta": float(torch.dot(g_contact, delta)),
            "gJ_dot_delta": float(torch.dot(g_total, delta)),
            "cos_delta_gC": decomp._cosine(delta, g_contact),
            "cos_delta_gJ": decomp._cosine(delta, g_total),
            "delta_norm": float(delta.norm()),
        }

    def arm_record(prep, post, captured, elapsed, extra=None):
        values = post["values"]
        base = prep["base"]
        record = {
            "checkpoint_loop": prep["checkpoint_loop"],
            "initial_contact_state": prep["initial_contact_state"],
            "initial_air_state": prep["initial_air_state"],
            "initial_contact_optimizer": prep["initial_contact_optimizer"],
            "air_after_state": prep["air_after_state"],
            "air_after_optimizer": prep["air_after_optimizer"],
            "air_loss": prep["air_loss"],
            "lambda_pde": prep["lambda_pde"],
            "lambda_data": prep["lambda_data"],
            "contact_state_before_update": prep["contact_state_before_update"],
            "contact_optimizer_before_update":
                prep["contact_optimizer_before_update"],
            "group": prep["group"],
            "base": base,
            "C": values["C"],
            "J": values["J"],
            "terms_raw": values["terms_raw"],
            "terms_weighted": values["terms_weighted"],
            "C_change": post["C_change"],
            "J_change": post["J_change"],
            "delta_norm": post["delta_norm"],
            "projection": projection(g_contact, post["delta"]),
            "post_contact_state": post["post_contact_state"],
            "post_air_state": post["post_air_state"],
            "post_optimizer_hash": post["post_optimizer_hash"],
            "step_values_after": _snapshot_steps(
                post["snapshots_after"]
            ),
            "time_s": elapsed,
            "gradient_capture_abs_error": (
                None if captured.get("gradient") is None
                else float((captured["gradient"] - g_total).norm())
            ),
        }
        if extra:
            record.update(extra)
        return record

    record = {
        "trial": int(trial),
        "seed": trial_seed,
        "A": arm_record(
            prep_a, post_a, captured_a, time_a,
            extra={"exp_avg_norm_before": max(
                (float(item["exp_avg"].norm())
                 for item in prep_a["snapshots_before"]), default=0.0
            )},
        ),
        "B": arm_record(
            prep_b, post_b, captured_b, time_b,
            extra={
                "exp_avg_norm_before_zero": max(
                    (float(item["exp_avg"].norm())
                     for item in snapshots_b_before_zero), default=0.0
                ),
                "exp_avg_norm_after_zero": zero_norm,
                "exp_avg_sq_before_equal_A": exp_avg_sq_pre_diff,
                "step_before_equal_A": step_pre_equal,
                "exp_avg_sq_after_equal_A": exp_avg_sq_post_diff,
                "step_after_equal_A": step_post_equal,
            },
        ),
        "consistency": {
            "theta_max_abs_diff": theta_diff,
            "base_C_abs_diff": base_c_diff,
            "base_J_abs_diff": base_j_diff,
            "air_loss_abs_diff": air_loss_diff,
            "lambda_abs_diff": lambda_diff,
            "group_pre_equal": group_pre_equal,
            "gradient_max_abs_diff": gradient_diff,
            "gradient_relative_diff": gradient_rel,
            "exp_avg_sq_before_max_abs_diff": exp_avg_sq_pre_diff,
            "exp_avg_sq_after_max_abs_diff": exp_avg_sq_post_diff,
            "step_before_equal": step_pre_equal,
            "step_after_equal": step_post_equal,
            "B_delta_current_terms_abs_error": b_delta_error,
            "B_delta_current_terms_relative_error": b_delta_rel,
            "A_minus_B_history_abs_error": history_error,
            "A_minus_B_history_relative_error": history_rel,
            "A_decomposition_reconstruction_relative_error":
                decomposition.get("reconstruction_relative_error"),
            "A_decomposition_contribution_relative_error":
                decomposition.get("contribution_sum_relative_error"),
            "A_history_norm": float(history.norm()),
            "A_current_terms_norm": float(current_terms.norm()),
            "delta_A_norm": float(delta_a.norm()),
            "delta_B_norm": float(delta_b.norm()),
            "delta_B_over_A": (
                float(delta_b.norm() / delta_a.norm())
                if float(delta_a.norm()) > 0.0 else None
            ),
            "cos_delta_A_B": decomp._cosine(delta_a, delta_b),
            "beta1_unchanged": betas_equal,
            "beta2_unchanged": betas_equal,
            "eps_unchanged": eps_equal,
            "lr_unchanged": lr_equal,
            "group_restored_unchanged": group_restored_equal,
        },
        "flags": {
            "A_C_worse": bool(post_a["C_change"] > 0.0),
            "B_C_worse": bool(post_b["C_change"] > 0.0),
            "B_C_improves": bool(post_b["C_change"] < 0.0),
            "B_C_better_than_A": bool(
                post_b["values"]["C"] < post_a["values"]["C"]
            ),
            "B_J_improves": bool(post_b["J_change"] < 0.0),
            "A_current_terms_favorable": bool(
                projection(g_contact, current_terms)["gC_dot_delta"] < 0.0
            ),
        },
        "checks": checks,
        "status": status,
    }
    for key in ("_history", "_terms", "_reconstruction"):
        decomposition.pop(key, None)
    return record


def _median(values):
    return decomp._median(values)


def _stats(values):
    return decomp._stats(values)


def _summary(rows) -> dict:
    def count(key):
        return int(sum(bool(row["flags"][key]) for row in rows))

    def med(path):
        out = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            out.append(value)
        return _median(out)

    def stat(path):
        out = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            out.append(value)
        return _stats(out)

    def med_diff(path_b, path_a):
        out = []
        for row in rows:
            value_b = row
            value_a = row
            for key in path_b:
                value_b = value_b[key]
            for key in path_a:
                value_a = value_a[key]
            out.append(value_b - value_a)
        return _median(out)

    a_worse = [row for row in rows if row["flags"]["A_C_worse"]]
    return {
        "n": len(rows),
        "counts": {
            "A_C_worse": count("A_C_worse"),
            "B_C_worse": count("B_C_worse"),
            "B_C_improves": count("B_C_improves"),
            "B_C_better_than_A": count("B_C_better_than_A"),
            "B_J_improves": count("B_J_improves"),
            "A_C_worse_current_favorable": sum(
                1 for row in rows
                if row["flags"]["A_C_worse"]
                and row["flags"]["A_current_terms_favorable"]
            ),
            "A_C_worse_current_unfavorable": sum(
                1 for row in rows
                if row["flags"]["A_C_worse"]
                and not row["flags"]["A_current_terms_favorable"]
            ),
            "B_C_improves_among_A_worse": sum(
                1 for row in a_worse if row["flags"]["B_C_improves"]
            ),
            "B_C_better_than_A_among_A_worse": sum(
                1 for row in a_worse if row["flags"]["B_C_better_than_A"]
            ),
            "B_J_improves_among_A_worse": sum(
                1 for row in a_worse if row["flags"]["B_J_improves"]
            ),
        },
        "median": {
            "A_C_change": med(("A", "C_change")),
            "B_C_change": med(("B", "C_change")),
            "A_J_change": med(("A", "J_change")),
            "B_J_change": med(("B", "J_change")),
            "C_B_minus_A": med_diff(("B", "C"), ("A", "C")),
            "J_B_minus_A": med_diff(("B", "J"), ("A", "J")),
            "delta_A_norm": med(("A", "delta_norm")),
            "delta_B_norm": med(("B", "delta_norm")),
            "delta_ratio": med(("consistency", "delta_B_over_A")),
            "history_norm": med(("consistency", "A_history_norm")),
            "current_terms_norm": med(
                ("consistency", "A_current_terms_norm")
            ),
            "cos_delta_A_B": med(
                ("consistency", "cos_delta_A_B", "cosine")
            ),
            "A_gC_dot_delta": med(("A", "projection", "gC_dot_delta")),
            "B_gC_dot_delta": med(("B", "projection", "gC_dot_delta")),
            "A_gJ_dot_delta": med(("A", "projection", "gJ_dot_delta")),
            "B_gJ_dot_delta": med(("B", "projection", "gJ_dot_delta")),
            "A_cos_gC": med(("A", "projection", "cos_delta_gC", "cosine")),
            "B_cos_gC": med(("B", "projection", "cos_delta_gC", "cosine")),
            "A_cos_gJ": med(("A", "projection", "cos_delta_gJ", "cosine")),
            "B_cos_gJ": med(("B", "projection", "cos_delta_gJ", "cosine")),
            "B_delta_current_terms_relative_error": med(
                ("consistency", "B_delta_current_terms_relative_error")
            ),
            "A_minus_B_history_relative_error": med(
                ("consistency", "A_minus_B_history_relative_error")
            ),
        },
        "term_raw_A": {
            name: _stats([row["A"]["terms_raw"][name] for row in rows])
            for name in TERMS
        },
        "term_raw_B": {
            name: _stats([row["B"]["terms_raw"][name] for row in rows])
            for name in TERMS
        },
        "term_raw_B_minus_A": {
            name: _stats([
                row["B"]["terms_raw"][name] - row["A"]["terms_raw"][name]
                for row in rows
            ])
            for name in TERMS
        },
        "term_weighted_A": {
            name: _stats([
                row["A"]["terms_weighted"][name] for row in rows
            ])
            for name in TERMS
        },
        "term_weighted_B": {
            name: _stats([
                row["B"]["terms_weighted"][name] for row in rows
            ])
            for name in TERMS
        },
        "term_weighted_B_minus_A": {
            name: _stats([
                row["B"]["terms_weighted"][name]
                - row["A"]["terms_weighted"][name]
                for row in rows
            ])
            for name in TERMS
        },
        "B_gC_dot_delta_stats": stat(
            ("B", "projection", "gC_dot_delta")
        ),
        "B_J_dot_delta_stats": stat(
            ("B", "projection", "gJ_dot_delta")
        ),
    }


def _subsets(rows) -> dict:
    a_worse = [row for row in rows if row["flags"]["A_C_worse"]]
    favorable = [
        row for row in a_worse if row["flags"]["A_current_terms_favorable"]
    ]
    unfavorable = [
        row for row in a_worse
        if not row["flags"]["A_current_terms_favorable"]
    ]
    return {
        "all": _summary(rows),
        "A_C_worse": _summary(a_worse),
        "A_C_worse_current_favorable": _summary(favorable),
        "A_C_worse_current_unfavorable": _summary(unfavorable),
    }


def _status_reasons(payload) -> list[str]:
    reasons: list[str] = []
    for tag, entry in payload.get("checkpoints", {}).items():
        if not entry.get("unchanged", False):
            reasons.append(f"{tag}: checkpoint SHA-256 changed")
        trials = entry.get("trials", [])
        if not trials:
            reasons.append(f"{tag}: no trials recorded")
        failed = [row for row in trials if row.get("status") != "ok"]
        if failed:
            reasons.append(
                f"{tag}: {len(failed)}/{len(trials)} trial status failed"
            )
    return reasons


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
        "format": "spherical_blast_pinn.current_contact_first_moment_control",
        "version": 1,
        "config": C_CONFIG,
        "checkpoints": {},
        "seeds": list(SEEDS),
        "trials_per_checkpoint_seed": int(args.trials),
        "full_label_counts": {
            case_id: int(item["n"]) for case_id, item in labels.items()
        },
        "full_label_total": int(sum(item["n"] for item in labels.values())),
        "settings": {
            "update_contact_labels": "official random batch",
            "reference_contact_labels": "all legal training labels",
            "validation_labels_used": False,
            "air_update": "same official _air_update on the same random batch",
            "A": "original ContactNet Adam step",
            "B": "exp_avg zeroed only; exp_avg_sq, step, betas, eps, lr and "
                 "network parameters retained; same current gradient",
            "no_continuous_training": True,
        },
    }
    started = time.perf_counter()
    for tag, checkpoint in CHECKPOINTS.items():
        before = _sha256(checkpoint)
        rows = []
        for seed in SEEDS:
            for trial in range(int(args.trials)):
                try:
                    record = _trial(
                        trainer, checkpoint, labels, seed, trial
                    )
                except Exception as exc:
                    trial_seed = int(seed) + int(trial) * 104729
                    record = {
                        "trial": int(trial),
                        "seed": trial_seed,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "checks": {},
                        "flags": {},
                    }
                rows.append((seed, record))
        after = _sha256(checkpoint)
        payload["checkpoints"][tag] = {
            "path": str(checkpoint),
            "sha256": before,
            "sha256_after": after,
            "unchanged": before == after,
            "loops": {
                seed: sorted({record.get("checkpoint_loop", record.get(
                    "A", {}).get("checkpoint_loop")
                ) for s, record in rows if s == seed
                    and record.get("status") == "ok"})
                for seed in SEEDS
            },
            "seeds": {
                str(seed): _subsets([
                    record for s, record in rows
                    if s == seed and record.get("status") == "ok"
                ]) for seed in SEEDS
            },
            "combined": _subsets([
                record for _, record in rows
                if record.get("status") == "ok"
            ]),
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
    output_file = output / "control.json"
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
                "loops": entry["loops"],
                "status_counts": entry["status_counts"],
                "combined": entry["combined"],
            }
            for tag, entry in payload["checkpoints"].items()
        },
    }, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
