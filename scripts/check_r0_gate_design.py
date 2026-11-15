#!/usr/bin/env python3
"""Read-only checks for the production AirNet ``r0_input_gate`` option.

This script exercises the actual model code, not a private copy of the gate
formula.  It loads existing B checkpoints only for inspection, builds a clone
with ``r0_input_gate='similarity_time'``, and never trains or writes a
checkpoint.

The candidate is a hard structural statement about the *input radius*:

    theta=(t-case.t0)/case.time_scale
    gate=1-exp(-theta)
    R0_input=R0*gate

At t0, R0_input=0 for every case, so the initial normalized output is
independent of R0.  The output is not forced to the theoretical value; the
existing soft ``air_ic_loss`` remains active.  For t>t0 the gate opens.

The current data have R0/T_star constant across cases.  Therefore the gate
derivative at t0 is the same magnitude for every case; this script reports that
property explicitly and does not claim a radius-dependent gate derivative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_contact_gradients as diag  # noqa: E402
from spherical_blast_pinn.contact_air.losses import (  # noqa: E402
    air_ic_loss,
    finite_mach_rh_state,
    right_loss,
)
from spherical_blast_pinn.contact_air.models import (  # noqa: E402
    AirNet,
    TimeInput,
    air_fields_with_derivatives,
)
from spherical_blast_pinn.contact_air.trainer import describe_matches  # noqa: E402


CONFIG = "configs/contact_air/split_train_r50_500_fullframes_similarity.yaml"
CHECKPOINTS = {
    "B_last": Path("outputs/contact_air_time_input_similarity_4000/last.pt"),
    "B_best": Path("outputs/contact_air_time_input_similarity_4000/best.pt"),
}
ZETAS = np.linspace(0.01, 0.99, 33)
WINDOWS = (0.01, 0.05, 0.10)
DERIVATIVE_DELTAS = (0.0, 0.25, 1.0)
FIELDS = ("rho", "u", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_gate(net: AirNet) -> AirNet:
    clone = AirNet(
        width=int(net.trunk[0].out_features),
        depth=sum(isinstance(m, torch.nn.Linear) for m in net.trunk),
        dtype=net.dtype,
        device=net.device,
        time_reference_s=net.time_reference_s,
        time_input_mode=net.time_input_mode,
        similarity_input_divisor=net.similarity_input_divisor,
        r0_input_gate="similarity_time",
    )
    clone.load_state_dict(net.state_dict())
    return clone


def _norm(fields, case):
    return {
        "rho": (fields.density / float(case.rho_scale)).detach().reshape(-1).numpy(),
        "u": (fields.velocity / float(case.u_scale)).detach().reshape(-1).numpy(),
        "p": (fields.pressure / float(case.p_scale)).detach().reshape(-1).numpy(),
    }


def _spread(values):
    ref = values["r50_full"]
    return {
        field: float(max(np.max(np.abs(v[field] - ref[field]))
                          for v in values.values()))
        for field in FIELDS
    }


def _field_loss(fields):
    return fields.density.sum() + fields.velocity.sum() + fields.pressure.sum()


def _flatten_grads(params, grads):
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).reshape(-1)
        for p, g in zip(params, grads)
    ])


def _none_baseline_check(net, case, zeta):
    """Compare the none path with the historical three-input construction."""
    scales = net._check_scales
    params = list(net.parameters())
    t_values = (
        float(case.t0),
        float(case.t0) + 0.01 * (float(case.t_end) - float(case.t0)),
    )
    output_max = 0.0
    grad_max = 0.0
    for t_value in t_values:
        t = torch.full_like(zeta, t_value)
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        ordinary = net(zeta, t, r0, scales, time_input=case.time_input)
        hidden = net.trunk(torch.cat(
            [zeta, net.scaled_time(t, case.time_input), r0], dim=-1
        ))
        manual = net._heads(hidden, scales)
        output_max = max(
            output_max,
            float((ordinary.density - manual.density).abs().max()),
            float((ordinary.velocity - manual.velocity).abs().max()),
            float((ordinary.pressure - manual.pressure).abs().max()),
        )
        grads_a = torch.autograd.grad(
            _field_loss(ordinary), params, retain_graph=True, allow_unused=True
        )
        grads_b = torch.autograd.grad(
            _field_loss(manual), params, allow_unused=True
        )
        grad_max = max(
            grad_max,
            float((_flatten_grads(params, grads_a)
                   - _flatten_grads(params, grads_b)).abs().max()),
        )
    return {
        "t_values_s": list(t_values),
        "max_abs_output_difference": output_max,
        "max_abs_gradient_difference": grad_max,
    }


def _t0_section(net, cases, zeta):
    fields = {}
    rmse = {}
    for case in cases:
        t = torch.full_like(zeta, float(case.t0))
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        scales = net._check_scales_by_case[case.case_id]
        value = net(zeta, t, r0, scales, time_input=case.time_input)
        fields[case.case_id] = _norm(value, case)
        errors = []
        for values, scale in (
            (value.density, case.rho_scale),
            (value.velocity, case.u_scale),
            (value.pressure, case.p_scale),
        ):
            e = (values / float(scale) - 1.0).reshape(-1)
            errors.append(torch.mean(e * e))
        rmse[case.case_id] = float(torch.sqrt(sum(errors) / 3.0))
    return fields, rmse


def _soft_ic_gradient(net, cases):
    params = list(net.parameters())
    total = torch.zeros((), dtype=torch.float64)
    for case in cases:
        zeta = torch.as_tensor(ZETAS, dtype=torch.float64).reshape(-1, 1)
        batch = SimpleNamespace(
            zeta_ic=zeta,
            t_ic=torch.full_like(zeta, float(case.t0)),
        )
        term = air_ic_loss(
            net, batch, case, net._check_scales_by_case[case.case_id]
        )
        total = total + term.total
    total = total / len(cases)
    grads = torch.autograd.grad(total, params, allow_unused=True)
    return float(total.detach()), float(torch.linalg.vector_norm(
        _flatten_grads(params, grads)
    ))


def _derivative_check(net, case):
    zeta = torch.tensor([[0.5]], dtype=torch.float64)
    r0 = torch.tensor([[float(case.charge_radius_m)]], dtype=torch.float64)
    scales = net._check_scales_by_case[case.case_id]
    records = []
    for delta in DERIVATIVE_DELTAS:
        t_value = float(case.t0) + delta * float(case.time_scale)
        t = torch.tensor([[t_value]], dtype=torch.float64, requires_grad=True)
        fields, derivatives = air_fields_with_derivatives(
            net, zeta, t, r0, scales, create_graph=False,
            time_input=case.time_input,
        )
        h = 1.0e-8 * float(case.time_scale)
        with torch.no_grad():
            plus = net(
                zeta, torch.tensor([[t_value + h]], dtype=torch.float64),
                r0, scales, time_input=case.time_input,
            )
            if delta == 0.0:
                minus = net(
                    zeta, torch.tensor([[t_value]], dtype=torch.float64),
                    r0, scales, time_input=case.time_input,
                )
                denominator = h
            else:
                minus = net(
                    zeta, torch.tensor([[t_value - h]], dtype=torch.float64),
                    r0, scales, time_input=case.time_input,
                )
                denominator = 2.0 * h
        per_field = {}
        for name, pred in (
            ("rho", fields.density), ("u", fields.velocity),
            ("p", fields.pressure),
        ):
            plus_v = plus.density if name == "rho" else (
                plus.velocity if name == "u" else plus.pressure
            )
            minus_v = minus.density if name == "rho" else (
                minus.velocity if name == "u" else minus.pressure
            )
            fd = (plus_v - minus_v) / denominator
            auto = derivatives[f"{name}_t"]
            per_field[name] = {
                "autograd": float(auto),
                "finite_difference": float(fd),
                "max_abs_difference": float((auto - fd).abs().max()),
                "max_relative_difference": float(
                    ((auto - fd).abs() / torch.clamp(fd.abs(), min=1.0e-30)).max()
                ),
            }
        records.append({"theta": float(delta), "fields": per_field})
    return records


def _functional_check(net, cases):
    max_output = {name: 0.0 for name in FIELDS}
    max_derivative = {name: 0.0 for name in FIELDS}
    parameter_grads_all_none = True
    for case in cases:
        zeta = torch.as_tensor(ZETAS, dtype=torch.float64).reshape(-1, 1)
        t = torch.full_like(zeta, float(case.t0))
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        scales = net._check_scales_by_case[case.case_id]
        fields, derivatives = air_fields_with_derivatives(
            net, zeta, t, r0, scales, create_graph=False,
            functional=False, time_input=case.time_input,
        )
        func_fields, func_derivatives = air_fields_with_derivatives(
            net, zeta, t, r0, scales, create_graph=False,
            functional=True, time_input=case.time_input,
        )
        for name, a, b in (
            ("rho", fields.density, func_fields.density),
            ("u", fields.velocity, func_fields.velocity),
            ("p", fields.pressure, func_fields.pressure),
        ):
            max_output[name] = max(max_output[name], float((a - b).abs().max()))
            max_derivative[name] = max(
                max_derivative[name],
                float((derivatives[f"{name}_t"]
                       - func_derivatives[f"{name}_t"]).abs().max()),
                float((derivatives[f"{name}_z"]
                       - func_derivatives[f"{name}_z"]).abs().max()),
            )
        grads = torch.autograd.grad(
            _field_loss(func_fields), list(net.parameters()), allow_unused=True
        )
        if any(g is not None and bool(g.abs().sum() > 0) for g in grads):
            parameter_grads_all_none = False
    return {
        "max_abs_field_difference": max_output,
        "max_abs_t_or_z_derivative_difference": max_derivative,
        "functional_parameter_gradients_zero_or_none": parameter_grads_all_none,
    }


def _window_spread(baseline, gate_net, cases, fraction):
    zeta = torch.full((1, 1), 0.5, dtype=torch.float64)
    base = {}
    gated = {}
    for case in cases:
        t_value = float(case.t0) + fraction * (
            float(case.t_end) - float(case.t0)
        )
        t = torch.full_like(zeta, t_value)
        r0 = torch.full_like(zeta, float(case.charge_radius_m))
        scales = gate_net._check_scales_by_case[case.case_id]
        with torch.no_grad():
            b = baseline(zeta, t, r0, scales, time_input=case.time_input)
            g = gate_net(zeta, t, r0, scales, time_input=case.time_input)
        base[case.case_id] = _norm(b, case)
        gated[case.case_id] = _norm(g, case)
    theta = fraction * (float(cases[0].t_end) - float(cases[0].t0)) / float(
        cases[0].time_scale
    )
    # The normalized theta is the same for every case under similarity.
    gate_value = float(-math.expm1(-theta))
    return {
        "fraction": float(fraction),
        "gate_value": gate_value,
        "baseline_max_spread": _spread(base),
        "gated_max_spread": _spread(gated),
    }


def _right_loss_check(net, case):
    scales = net._check_scales_by_case[case.case_id]
    span = float(case.t_end) - float(case.t0)
    t_right = torch.tensor(
        [float(case.t0) + 0.25 * span, float(case.t0) + 0.50 * span],
        dtype=torch.float64,
    )
    batch = SimpleNamespace(t_right=t_right, t_ic=torch.zeros(0, dtype=torch.float64))
    term = right_loss(
        net=net, batch=batch, case=case, scales=scales
    )
    _, shock_speed = case.shock_tensors(t_right.reshape(-1))
    state = finite_mach_rh_state(
        shock_speed.reshape(-1, 1),
        gamma=case.gamma_air,
        ambient_density=case.ambient_density,
        ambient_pressure=case.ambient_pressure,
    )
    zeta = torch.ones_like(t_right).reshape(-1, 1)
    r0 = torch.full_like(zeta, float(case.charge_radius_m))
    fields = net(zeta, t_right.reshape(-1, 1), r0, scales,
                 time_input=case.time_input)
    manual = (
        ((fields.density - state["density"]) / float(case.rho_scale)).pow(2).mean()
        + ((fields.velocity - state["velocity"]) / float(case.u_scale)).pow(2).mean()
        + ((fields.pressure - state["pressure"]) / float(case.p_scale)).pow(2).mean()
    )
    return {
        "right_loss_total": float(term.total.detach()),
        "manual_total_from_gated_net": float(manual.detach()),
        "max_abs_difference": float((term.total - manual).abs().detach()),
    }


def _ratio_section(cases):
    ratios = []
    for case in cases:
        ratio = float(case.charge_radius_m) / float(case.time_scale)
        ratios.append({
            "case_id": case.case_id,
            "r0_over_time_scale_m_per_s": ratio,
            "time_scale_over_r0_s_per_m": 1.0 / ratio,
            "d_r0_input_dt_at_t0_m_per_s": ratio,
            "d_r0_input_dt_at_theta_1_m_per_s": ratio * math.exp(-1.0),
            "theta_at_1pct": 0.01 * (
                float(case.t_end) - float(case.t0)
            ) / float(case.time_scale),
            "theta_at_5pct": 0.05 * (
                float(case.t_end) - float(case.t0)
            ) / float(case.time_scale),
            "theta_at_10pct": 0.10 * (
                float(case.t_end) - float(case.t0)
            ) / float(case.time_scale),
        })
    values = [row["r0_over_time_scale_m_per_s"] for row in ratios]
    return {
        "cases": ratios,
        "min": min(values),
        "max": max(values),
        "is_constant": math.isclose(min(values), max(values), rel_tol=1.0e-12),
        "interpretation": (
            "The gate derivative dR0_input/dt = (R0/T_star)*exp(-theta) has "
            "the same magnitude for every case at the same theta because R0/T_star "
            "is constant. Do not claim a radius-dependent gate derivative."
        ),
    }


def _checkpoint_section(net, gate_net):
    old_without_gate = dict(net.describe())
    old_without_gate.pop("r0_input_gate", None)
    old_without_gate.pop("r0_gate_rule", None)
    return {
        "none_description": net.describe(),
        "gate_description": gate_net.describe(),
        "old_description_without_gate_matches_none": describe_matches(
            old_without_gate, net.describe()
        ),
        "gate_description_matches_gate": describe_matches(
            gate_net.describe(), gate_net.describe()
        ),
        "gate_description_rejected_by_none": not describe_matches(
            gate_net.describe(), net.describe()
        ),
        "state_dict_keys_identical": list(gate_net.state_dict()) == list(net.state_dict()),
        "parameter_count_identical": sum(p.numel() for p in gate_net.parameters())
        == sum(p.numel() for p in net.parameters()),
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
    zeta = torch.as_tensor(ZETAS, dtype=torch.float64).reshape(-1, 1)
    results = {}
    for label, path in CHECKPOINTS.items():
        trainer.load(path)
        none_net = trainer.air_net
        # Temporary, read-only lookup tables for this diagnostic.  They are not
        # part of the model or checkpoint and are removed before returning.
        none_net._check_scales_by_case = dict(trainer._scales)
        gate_net = _clone_gate(none_net)
        gate_net._check_scales_by_case = dict(trainer._scales)
        first_case = trainer.cases[0]
        none_net._check_scales = trainer._scales[first_case.case_id]

        t0_gate, t0_rmse = _t0_section(gate_net, cases, zeta)
        soft_loss, soft_grad = _soft_ic_gradient(gate_net, cases)
        derivative = _derivative_check(gate_net, first_case)
        functional = _functional_check(gate_net, cases)
        windows = [
            _window_spread(none_net, gate_net, cases, fraction)
            for fraction in WINDOWS
        ]
        right = _right_loss_check(gate_net, first_case)
        baseline = _none_baseline_check(none_net, first_case, zeta)
        ratio = _ratio_section(cases)
        checkpoint = _checkpoint_section(none_net, gate_net)
        results[label] = {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256(path),
            "checkpoint_loop": int(trainer.loop),
            "none_mode": {
                "baseline_reconstruction": baseline,
                "description": none_net.describe(),
            },
            "gate_mode": {
                "t0": {
                    "max_cross_case_normalised_spread": _spread(t0_gate),
                    "per_case_normalised_rmse": t0_rmse,
                },
                "soft_ic_at_t0": {
                    "loss": soft_loss,
                    "gradient_norm": soft_grad,
                    "note": "existing B weights; construction check, not a trained score",
                },
                "physical_time_derivative": derivative,
                "ordinary_functional": functional,
                "window_spread": windows,
                "right_loss": right,
                "description": gate_net.describe(),
            },
            "checkpoint": checkpoint,
        }
        for net in (none_net, gate_net):
            for name in ("_check_scales", "_check_scales_by_case"):
                if hasattr(net, name):
                    delattr(net, name)

    payload = {
        "candidate": {
            "name": "similarity_time_gated_r0_input",
            "formula": "theta=(t-t0)/T_star; gate=1-exp(-theta); R0_input=R0*gate",
            "added_parameters": 0,
            "soft_ic_preserved": True,
            "hard_value_constraint": False,
            "no_training_performed": True,
            "not_claimed_all_t_gate_near_one": True,
        },
        "protocol": {
            "config": CONFIG,
            "checkpoints": {
                model: {"path": str(path), "sha256": _sha256(path)}
                for model, path in CHECKPOINTS.items()
            },
            "zeta_points": [float(v) for v in ZETAS],
            "derivative_deltas_in_T_star": list(DERIVATIVE_DELTAS),
            "windows": list(WINDOWS),
            "no_checkpoint_write": True,
            "no_optimizer_step": True,
        },
        "ratio": ratio,
        "models": results,
        "notes": [
            "The gate is the production AirNet r0_input_gate='similarity_time'; this script does not duplicate the formula.",
            "At t0 the R0 input is zero for all cases; the output is not forced to 1.",
            "The old checkpoints load in gate='none'; gate mode is a different semantics and must be recorded.",
            "R0/T_star is constant in the current data, so gate-derivative magnitudes are not radius-dependent.",
            "Right-boundary is checked through right_loss; interface uses the same AirNet object in the trainer.",
            "These are construction checks on existing weights, not accuracy results.",
        ],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(output),
        "sha256": _sha256(output),
        "ratio_is_constant": ratio["is_constant"],
        "models": {
            label: {
                "t0_spread": item["gate_mode"]["t0"]["max_cross_case_normalised_spread"],
                "soft_ic_loss": item["gate_mode"]["soft_ic_at_t0"]["loss"],
                "soft_ic_grad_norm": item["gate_mode"]["soft_ic_at_t0"]["gradient_norm"],
                "functional": item["gate_mode"]["ordinary_functional"],
                "window_spread": item["gate_mode"]["window_spread"],
                "none_baseline": item["none_mode"]["baseline_reconstruction"],
                "checkpoint": item["checkpoint"],
            }
            for label, item in results.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
