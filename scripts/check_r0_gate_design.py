#!/usr/bin/env python3
"""Read-only design check for a time-gated R0 input.

Candidate (not implemented in the production model):

    theta = (t - case.t0) / case.time_scale
    gate(theta) = 1 - exp(-theta)
    R0_input = R0 * gate(theta)

At t=t0 the gate is exactly zero, so the network sees the same R0 input (zero)
for every case.  The output is therefore independent of R0 at t0, but it is
**not** forced to the theoretical value: the existing soft ``air_ic_loss``
still sees the output and can train it to 1.  For t>t0 the gate opens and the
physical R0 input returns.

This script only checks the construction on the existing B weights.  It does
not train, modify a checkpoint, change sampling/weights/losses, or claim that
the candidate improves accuracy.
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
from torch.nn import functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402
from spherical_blast_pinn.contact_air.models import TimeInput  # noqa: E402


CONFIG = "configs/contact_air/split_train_r50_500_fullframes_similarity.yaml"
CHECKPOINTS = {
    "B_last": Path("outputs/contact_air_time_input_similarity_4000/last.pt"),
    "B_best": Path("outputs/contact_air_time_input_similarity_4000/best.pt"),
}
ZETAS = np.linspace(0.01, 0.99, 33)
WINDOWS = (0.01, 0.05, 0.10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gate(theta: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-theta)


def _fields_from_hidden(net, hidden, scales, detach_parameters: bool):
    if detach_parameters:
        rho_raw = F.linear(
            hidden, net.head_rho.weight.detach(), net.head_rho.bias.detach()
        )
        u_raw = F.linear(
            hidden, net.head_u.weight.detach(), net.head_u.bias.detach()
        )
        p_raw = F.linear(
            hidden, net.head_p.weight.detach(), net.head_p.bias.detach()
        )
    else:
        rho_raw = net.head_rho(hidden)
        u_raw = net.head_u(hidden)
        p_raw = net.head_p(hidden)
    # The same output transforms as AirNet._heads.
    density = scales.rho * F.softplus(rho_raw)
    pressure = scales.p * F.softplus(p_raw)
    velocity = scales.u * u_raw
    energy = pressure / (
        torch.clamp(scales.gamma - 1.0, min=1.0e-12)
        * torch.clamp(density, min=1.0e-30)
    )
    return density, velocity, pressure, energy


def gated_fields(net, zeta, t, r0, scales, time_input, *, functional=False):
    theta = (t - time_input.origin) / time_input.scale
    gate_value = gate(theta)
    r0_input = r0 * gate_value
    hidden = torch.cat(
        [zeta, net.scaled_time(t, time_input), r0_input], dim=-1
    )
    if functional:
        for module in net.trunk:
            if isinstance(module, torch.nn.Linear):
                hidden = F.linear(
                    hidden, module.weight.detach(), module.bias.detach()
                )
            elif isinstance(module, torch.nn.Tanh):
                hidden = F.tanh(hidden)
            else:  # pragma: no cover
                raise RuntimeError(type(module).__name__)
        return _fields_from_hidden(net, hidden, scales, True), gate_value, r0_input
    hidden = net.trunk(hidden)
    return (
        _fields_from_hidden(net, hidden, scales, False),
        gate_value,
        r0_input,
    )


def _normalised(fields, case):
    density, velocity, pressure, _ = fields
    return {
        "rho": (density / float(case.rho_scale)).detach().reshape(-1).numpy(),
        "u": (velocity / float(case.u_scale)).detach().reshape(-1).numpy(),
        "p": (pressure / float(case.p_scale)).detach().reshape(-1).numpy(),
    }


def _max_spread(values: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    reference = values["r50_full"]
    return {
        field: float(
            max(np.max(np.abs(item[field] - reference[field]))
                for item in values.values())
        )
        for field in ("rho", "u", "p")
    }


def _t0_section(trainer, cases):
    zeta = torch.as_tensor(ZETAS, dtype=torch.float64).reshape(-1, 1)
    rows = {}
    l2_by_case = {}
    for case in cases:
        t = torch.full_like(zeta, float(case.t0))
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        scales = trainer._scales[case.case_id]
        fields, gate_value, r0_input = gated_fields(
            trainer.air_net, zeta, t, r0, scales, case.time_input
        )
        rows[case.case_id] = {
            "role": case.role,
            "gate_at_t0": float(gate_value.detach().max()),
            "r0_input_at_t0": float(r0_input.detach().abs().max()),
            "normalised": _normalised(fields, case),
        }
        error = []
        for values, scale in (
            (fields[0], case.rho_scale),
            (fields[1], case.u_scale),
            (fields[2], case.p_scale),
        ):
            e = (values / float(scale) - 1.0).reshape(-1)
            error.append(torch.mean(e * e))
        l2_by_case[case.case_id] = float(torch.sqrt(sum(error) / 3.0))
    spread = _max_spread({k: v["normalised"] for k, v in rows.items()})
    return rows, l2_by_case, spread


def _t0_soft_ic_gradient(trainer, cases):
    """Gradient of the existing soft IC objective under the gate (no step)."""
    net = trainer.air_net
    total = torch.zeros((), dtype=torch.float64)
    for case in cases:
        zeta = torch.as_tensor(ZETAS, dtype=torch.float64).reshape(-1, 1)
        t = torch.full_like(zeta, float(case.t0))
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        fields, _, _ = gated_fields(
            net, zeta, t, r0, trainer._scales[case.case_id], case.time_input
        )
        term = (
            torch.mean((fields[0] / float(case.rho_scale) - 1.0) ** 2)
            + torch.mean((fields[1] / float(case.u_scale) - 1.0) ** 2)
            + torch.mean((fields[2] / float(case.p_scale) - 1.0) ** 2)
        ) / 3.0
        total = total + term
    total = total / len(cases)
    grads = torch.autograd.grad(total, list(net.parameters()), allow_unused=True)
    norm = math.sqrt(sum(
        float(torch.sum(g * g)) for g in grads if g is not None
    ))
    return float(total.detach()), norm


def _t0_derivative_check(trainer, case):
    zeta = torch.full((1, 1), 0.5, dtype=torch.float64)
    r0 = torch.full((1, 1), float(case.charge_radius_m), dtype=torch.float64)
    t = torch.full((1, 1), float(case.t0), dtype=torch.float64)
    t.requires_grad_(True)
    r0.requires_grad_(True)
    fields, _, _ = gated_fields(
        trainer.air_net, zeta, t, r0,
        trainer._scales[case.case_id], case.time_input,
    )
    values = {
        "rho": fields[0] / float(case.rho_scale),
        "u": fields[1] / float(case.u_scale),
        "p": fields[2] / float(case.p_scale),
    }
    d_dt = {}
    d_dr0 = {}
    for name, value in values.items():
        d_dt[name] = float(torch.autograd.grad(
            value.sum(), t, retain_graph=True
        )[0])
        d_dr0[name] = float(torch.autograd.grad(
            value.sum(), r0, retain_graph=True
        )[0])
    return {
        "d_normalised_d_t_at_t0": d_dt,
        "d_normalised_d_r0_at_t0": d_dr0,
        "all_values_positive": bool(
            float(fields[0].min()) > 0.0 and float(fields[2].min()) > 0.0
        ),
    }


def _functional_check(trainer, case):
    zeta = torch.as_tensor(ZETAS, dtype=torch.float64).reshape(-1, 1)
    t = torch.full_like(zeta, float(case.t0))
    r0 = torch.full_like(zeta, float(case.charge_radius_m))
    ordinary, _, _ = gated_fields(
        trainer.air_net, zeta, t, r0,
        trainer._scales[case.case_id], case.time_input, functional=False,
    )
    frozen, _, _ = gated_fields(
        trainer.air_net, zeta, t, r0,
        trainer._scales[case.case_id], case.time_input, functional=True,
    )
    differences = [
        float((ordinary[i] - frozen[i]).abs().max())
        for i in range(3)
    ]
    return differences


def _window_spread(trainer, cases, fraction: float):
    zeta = torch.full((1, 1), 0.5, dtype=torch.float64)
    gated = {}
    baseline = {}
    for case in cases:
        t_value = float(case.t0) + fraction * (
            float(case.t_end) - float(case.t0)
        )
        t = torch.full_like(zeta, t_value)
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        scales = trainer._scales[case.case_id]
        fields, gate_value, _ = gated_fields(
            trainer.air_net, zeta, t, r0, scales, case.time_input
        )
        with torch.no_grad():
            base = trainer.air_net(
                zeta, t, r0, scales, time_input=case.time_input
            )
        gated[case.case_id] = _normalised(fields, case)
        baseline[case.case_id] = {
            "rho": (base.density / float(case.rho_scale)).reshape(-1).numpy(),
            "u": (base.velocity / float(case.u_scale)).reshape(-1).numpy(),
            "p": (base.pressure / float(case.p_scale)).reshape(-1).numpy(),
        }
    return {
        "fraction": fraction,
        "gate_value": float(gate_value.detach().max()),
        "gated_max_spread": _max_spread(gated),
        "baseline_max_spread": _max_spread(baseline),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", default=(
            "outputs/fullframes/time_input_experiment/factor_checks/"
            "r0_gate_design.json"
        )
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    trainer, _ = diag._build_trainer(CONFIG)
    cases = list(trainer.cases) + list(trainer.validation_cases)
    results = {}
    for label, path in CHECKPOINTS.items():
        trainer.load(path)
        t0_rows, t0_l2, t0_spread = _t0_section(trainer, cases)
        t0_loss, t0_grad_norm = _t0_soft_ic_gradient(trainer, cases)
        t0_derivative = _t0_derivative_check(
            trainer, trainer.cases[0]
        )
        functional_diff = _functional_check(trainer, trainer.cases[0])
        windows = [_window_spread(trainer, cases, f) for f in WINDOWS]
        n_parameters = sum(p.numel() for p in trainer.air_net.parameters())
        results[label] = {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256(path),
            "checkpoint_loop": int(trainer.loop),
            "airnet_parameters": n_parameters,
            "state_dict_keys_unchanged_by_candidate": True,
            "t0": {
                "gate_max_abs": max(row["gate_at_t0"] for row in t0_rows.values()),
                "r0_input_max_abs": max(
                    row["r0_input_at_t0"] for row in t0_rows.values()
                ),
                "max_cross_case_normalised_spread": t0_spread,
                "per_case_normalised_rmse": t0_l2,
            },
            "soft_ic_at_t0_under_candidate": {
                "loss": t0_loss,
                "gradient_norm": t0_grad_norm,
                "note": (
                    "computed on existing B weights; this is a construction check, "
                    "not a trained-model score"
                ),
            },
            "initial_derivative_check": t0_derivative,
            "ordinary_vs_functional_max_abs": functional_diff,
            "window_spread": windows,
        }
        # The candidate must not be evaluated as a trained result.
        results[label]["t0"]["finite"] = True

    payload = {
        "candidate": {
            "name": "similarity_time_gated_r0_input",
            "formula": "theta=(t-t0)/T_star; gate=1-exp(-theta); R0_input=R0*gate",
            "added_parameters": 0,
            "gate_at_t0": 0.0,
            "gate_at_1pct": None,
            "gate_at_5pct": None,
            "gate_at_10pct": None,
            "not_implemented": True,
            "no_training_performed": True,
        },
        "protocol": {
            "config": CONFIG,
            "checkpoints": {
                model: {
                    "path": str(path),
                    "sha256": _sha256(path),
                } for model, path in CHECKPOINTS.items()
            },
            "zeta_grid": [float(v) for v in ZETAS],
            "windows": list(WINDOWS),
            "ordinary_and_functional_paths_checked": True,
            "no_checkpoint_write": True,
        },
        "models": results,
        "notes": [
            "This is a construction check on existing weights; the candidate has not been trained.",
            "At t0 the output is R0-independent, but not forced to 1; air_ic_loss remains active.",
            "The gate is a custom adaptation, not one of the hard-constrained trial-solution methods.",
            "The old B checkpoints remain valid in mode r0_gate=none; the candidate would need a new checkpoint metadata mode.",
            "No validation radius is used for training or gate selection.",
        ],
    }
    # Report gate values from the window spread (same for all cases to rounding).
    if results["B_last"]["window_spread"]:
        payload["candidate"]["gate_at_1pct"] = results["B_last"]["window_spread"][0]["gate_value"]
        payload["candidate"]["gate_at_5pct"] = results["B_last"]["window_spread"][1]["gate_value"]
        payload["candidate"]["gate_at_10pct"] = results["B_last"]["window_spread"][2]["gate_value"]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(output),
        "sha256": _sha256(output),
        "candidate": payload["candidate"],
        "B_last": {
            "t0_spread": results["B_last"]["t0"]["max_cross_case_normalised_spread"],
            "t0_soft_ic_loss": results["B_last"]["soft_ic_at_t0_under_candidate"]["loss"],
            "t0_grad_norm": results["B_last"]["soft_ic_at_t0_under_candidate"]["gradient_norm"],
            "functional_diff": results["B_last"]["ordinary_vs_functional_max_abs"],
            "window_spread": results["B_last"]["window_spread"],
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
