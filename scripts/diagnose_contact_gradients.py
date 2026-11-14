#!/usr/bin/env python3
"""Read-only diagnosis of the ContactNet loss terms and their gradients.

It does **not** update the networks or the optimizers.  For the exact contact
stage of the alternating trainer it:

1. evaluates ``R_c`` at the contact-label times and compares it with the
   measured ``[R_c,low, R_c,up]`` band and with an existing prediction file;
2. independently recomputes the interval violation and compares it with the
   program's ``contact_data_loss`` / ``contact_violation_rmse``;
3. builds fixed batches (early/middle/late contact times) with an independent
   generator and computes, for every ContactNet term, the raw loss, its actual
   coefficient, the weighted gradient norm on ContactNet parameters, and the
   angle between the contact-data gradient and the other terms.

A zero gradient has no direction.  Such a term is reported as
``direction: undefined``; it is never divided by an epsilon to manufacture an
angle.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
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
    _contact_violation_values,
    contact_data_loss,
    contact_ic_loss,
    contact_violation_rmse,
    interface_loss,
    pde_loss,
    schedule_weights,
)
from spherical_blast_pinn.contact_air.mapping import build_geometry  # noqa: E402
from spherical_blast_pinn.contact_air.models import (  # noqa: E402
    air_fields_with_derivatives,
    closure_from_precursor,
    scales_from_precursor,
)
from spherical_blast_pinn.contact_air.sampling import (  # noqa: E402
    sample_case_batch,
)

_TERM_NAMES = ("pde", "contact_ic", "interface", "contact_data")


def _detach_contact(fields):
    from spherical_blast_pinn.contact_air.models import ContactFields
    return ContactFields(
        radius=fields.radius.detach(), speed=fields.speed.detach(),
        sigmoid=fields.sigmoid.detach(), raw=fields.raw.detach(),
        pressure=fields.pressure.detach(),
        relative_volume=fields.relative_volume.detach(),
    )


def _contact_at(net, case, closure, times: torch.Tensor, *, create_graph: bool):
    t = times.reshape(-1, 1)
    r0 = torch.full_like(t, float(case.charge_radius_m))
    rs, ds = case.shock_tensors(times.reshape(-1))
    rs = rs.reshape(-1, 1).to(torch.float64)
    ds = ds.reshape(-1, 1).to(torch.float64)
    return net(t, r0, rs, ds, closure, create_graph=create_graph)


def _stratified_contact_batch(case, batch, generator, which: int, count: int):
    """Replace the contact labels with an early/middle/late fixed slice."""
    indices = case.contact.indices()
    times = case.contact.times[indices]
    keep = (
        (times >= float(case.t0))
        & (times <= float(case.t_end))
        & np.asarray([case.shock.has_slope(float(value)) for value in times])
    )
    indices = indices[keep]
    times = times[keep]
    if indices.size == 0:
        return batch, {"status": "no_contact_frames"}
    span = float(case.t_end) - float(case.t0)
    lo = float(case.t0) + (which / 3.0) * span
    hi = float(case.t0) + ((which + 1) / 3.0) * span
    mask = (times >= lo) & (times <= hi)
    local = np.flatnonzero(mask)
    if local.size == 0:
        local = np.arange(indices.size)
    picks = torch.randint(
        0, int(local.size), (int(count),), generator=generator
    ).cpu().numpy()
    chosen = indices[local[picks]]
    t_contact = torch.as_tensor(case.contact.times[chosen], dtype=torch.float64)
    low = torch.as_tensor(case.contact.low[chosen], dtype=torch.float64)
    up = torch.as_tensor(case.contact.up[chosen], dtype=torch.float64)
    return (
        replace(batch, t_contact=t_contact, low_contact=low, up_contact=up),
        {
            "status": "ok",
            "time_lo_s": lo,
            "time_hi_s": hi,
            "available_frames": int(local.size),
        },
    )


def _flat_grads(term: torch.Tensor, params):
    grads = torch.autograd.grad(
        term, params, retain_graph=True, allow_unused=True,
    )
    parts = []
    for parameter, grad in zip(params, grads):
        parts.append(
            torch.zeros_like(parameter).reshape(-1)
            if grad is None else grad.reshape(-1)
        )
    vector = torch.cat(parts) if parts else torch.zeros(0)
    return vector, bool(torch.isfinite(vector).all())


def _angle_deg(a: torch.Tensor, b: torch.Tensor, threshold: float) -> dict:
    na = float(a.norm())
    nb = float(b.norm())
    if not (math.isfinite(na) and math.isfinite(nb)):
        return {"status": "non_finite"}
    if na <= threshold or nb <= threshold:
        return {
            "status": "direction_undefined",
            "norm_a": na,
            "norm_b": nb,
        }
    cosine = float(torch.dot(a, b) / (a.norm() * b.norm()))
    cosine = max(-1.0, min(1.0, cosine))
    return {
        "status": "ok",
        "angle_deg": math.degrees(math.acos(cosine)),
        "cosine": cosine,
        "norm_a": na,
        "norm_b": nb,
    }


def _label_check(case, contact, closure, prediction_dir: Path | None,
                 prediction_tag: str = "best"):
    indices = case.contact.indices()
    times = case.contact.times[indices]
    keep = (
        (times >= float(case.t0))
        & (times <= float(case.t_end))
        & np.asarray([case.shock.has_slope(float(value)) for value in times])
    )
    indices = indices[keep]
    times = times[keep]
    if indices.size == 0:
        return {"status": "no_valid_contact_frames"}
    t = torch.as_tensor(times, dtype=torch.float64)
    with torch.enable_grad():
        fields = _contact_at(contact, case, closure, t, create_graph=False)
    radius = fields.radius.detach().reshape(-1)
    low = torch.as_tensor(case.contact.low[indices], dtype=torch.float64)
    up = torch.as_tensor(case.contact.up[indices], dtype=torch.float64)
    scale = float(case.length_scale)
    below = torch.clamp(low - radius, min=0.0)
    above = torch.clamp(radius - up, min=0.0)
    values = (below / scale) ** 2 + (above / scale) ** 2
    independent_mean = float(values.mean())
    independent_rmse = float(torch.sqrt(torch.mean(values)))

    batch = replace(
        sample_case_batch(
            case, n_pde=0, n_ic=0, n_right=0, n_data=0, n_contact=0,
            generator=torch.Generator().manual_seed(1),
        ),
        t_contact=t,
        low_contact=low,
        up_contact=up,
    )
    with torch.enable_grad():
        fields2 = _contact_at(contact, case, closure, t, create_graph=True)
        program = contact_data_loss(fields2, batch, case)
        program_rmse, program_count = contact_violation_rmse(fields2, batch, case)
        program_values, _ = _contact_violation_values(fields2, batch, case)
    per_value_max = float((program_values.detach() - values).abs().max())
    result = {
        "status": "ok",
        "n_frames": int(times.size),
        "independent_mean_violation_squared": independent_mean,
        "program_mean_violation_squared": float(program.total.detach()),
        "mean_abs_difference": abs(independent_mean - float(program.total.detach())),
        "per_value_max_abs_difference": per_value_max,
        "independent_rmse": independent_rmse,
        "program_rmse": program_rmse,
        "program_count": int(program_count),
        "covered_frames": int(torch.count_nonzero((below == 0.0) & (above == 0.0))),
        "radius_min_m": float(radius.min()),
        "radius_max_m": float(radius.max()),
        "band_low_min_m": float(low.min()),
        "band_up_max_m": float(up.max()),
    }
    if prediction_dir is not None:
        path = prediction_dir / f"pred_{prediction_tag}_{case.case_id}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                pred_times = np.asarray(archive["times"], dtype=np.float64)
                pred_rc = np.asarray(archive["contact_radius"])[:, 0]
            pt = torch.as_tensor(pred_times, dtype=torch.float64)
            with torch.enable_grad():
                direct = _contact_at(contact, case, closure, pt, create_graph=False)
            difference = np.abs(
                direct.radius.detach().reshape(-1).numpy() - pred_rc
            )
            result["prediction_file"] = {
                "path": str(path),
                "n_times": int(pred_times.size),
                "max_abs_difference_m": float(difference.max()),
                "mean_abs_difference_m": float(difference.mean()),
            }
    return result


def _contact_term_batch(
    *, case, batch, contact, air, closure, scales, coefficient_data: float,
    lambda_pde: float, contact_data_weight: float, threshold: float,
):
    r0, rs, ds = _contact_inputs(case, batch.t_pde)
    c_fields = contact(
        batch.t_pde.reshape(-1, 1), r0, rs, ds, closure, create_graph=True
    )
    geometry = build_geometry(
        c_fields.radius, c_fields.speed, rs, ds, batch.zeta_pde.reshape(-1, 1)
    )
    a_fields, a_derivs = air_fields_with_derivatives(
        air, batch.zeta_pde, batch.t_pde, r0, scales,
        create_graph=True, functional=True,
    )
    pde = pde_loss(fields=a_fields, derivatives=a_derivs, geometry=geometry,
                   case=case)

    t0 = torch.full_like(batch.t_pde, float(case.t0))
    r00, rs0, ds0 = _contact_inputs(case, t0)
    c0 = contact(
        t0.reshape(-1, 1), r00, rs0, ds0, closure, create_graph=True
    )
    ic = contact_ic_loss(c0, case)

    if batch.t_pde.numel():
        air0 = air.forward_functional(
            torch.zeros_like(batch.t_pde).reshape(-1, 1),
            batch.t_pde.reshape(-1, 1), r0.reshape(-1, 1), scales,
            detach_parameters=True,
        )
        interface = interface_loss(air_at_contact=air0, contact=c_fields, case=case)
    else:
        interface = None

    if batch.t_contact.numel():
        c_contact = _contact_at(contact, case, closure, batch.t_contact,
                                create_graph=True)
        data = contact_data_loss(c_contact, batch, case)
    else:
        data = None

    coefficients = {
        "pde": float(lambda_pde),
        "contact_ic": 1.0,
        "interface": 1.0,
        "contact_data": float(coefficient_data),
    }
    terms = {
        "pde": pde,
        "contact_ic": ic,
        "interface": interface,
        "contact_data": data,
    }
    params = [p for p in contact.parameters() if p.requires_grad]
    vectors = {}
    entries = {}
    for name in _TERM_NAMES:
        term = terms[name]
        if term is None:
            entries[name] = {"status": "not_present"}
            continue
        loss = float(term.total.detach())
        coeff = coefficients[name]
        vector, finite = _flat_grads(term.total, params)
        weighted = vector * coeff
        vectors[name] = weighted
        entries[name] = {
            "status": "ok",
            "raw_loss": loss,
            "coefficient": coeff,
            "weighted_loss": coeff * loss,
            "grad_norm_raw": float(vector.norm()),
            "grad_norm_weighted": float(weighted.norm()),
            "finite_gradient": finite,
            "count": {k: int(v) for k, v in term.counts.items()},
        }
    angles = {}
    contact_vector = vectors.get("contact_data")
    for name in ("pde", "contact_ic", "interface"):
        other = vectors.get(name)
        if contact_vector is None or other is None:
            angles[f"contact_data_vs_{name}"] = {"status": "not_present"}
        else:
            angles[f"contact_data_vs_{name}"] = _angle_deg(
                contact_vector, other, threshold
            )
    others = [vectors[name] for name in ("pde", "contact_ic", "interface")
              if name in vectors]
    if contact_vector is not None and others:
        combined_other = torch.stack(others).sum(dim=0)
        total = combined_other + contact_vector
        angles["contact_data_vs_other_terms"] = _angle_deg(
            contact_vector, combined_other, threshold
        )
        angles["contact_data_vs_total"] = _angle_deg(
            contact_vector, total, threshold
        )
    return entries, angles


def _contact_inputs(case, t: torch.Tensor):
    t_flat = t.reshape(-1)
    r0 = torch.full_like(t_flat, float(case.charge_radius_m)).reshape(-1, 1)
    rs, ds = case.shock_tensors(t_flat)
    return r0, rs.reshape(-1, 1).to(torch.float64), ds.reshape(-1, 1).to(torch.float64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--cases", default="r50_full,r250_full,r500_full,r550_full,r750_full,r1000_full",
    )
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--prediction-tag", default="best",
                        choices=("best", "last"))
    parser.add_argument("--n-pde", type=int, default=205)
    parser.add_argument("--n-ic", type=int, default=51)
    parser.add_argument("--n-contact", type=int, default=51)
    parser.add_argument("--batch-seed", type=int, default=20261114)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--zero-threshold", type=float, default=1.0e-12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    section = _section(config)
    specs = {str(spec["case_id"]): spec for spec in _case_specs(section)}
    selected = [part.strip() for part in args.cases.split(",") if part.strip()]
    missing = [case_id for case_id in selected if case_id not in specs]
    if missing:
        raise SystemExit(f"cases not in config: {missing}")
    contact, air = _networks(section)
    checkpoint = _load_checkpoint(Path(args.checkpoint), contact, air)
    loop = int(checkpoint.get("loop", 0))
    weight = float(
        checkpoint.get("config", {}).get("contact_data_weight", 1.0)
    )
    n_schedule = int(checkpoint.get("config", {}).get("n_schedule", 2000))
    lambda_pde, lambda_data = schedule_weights(loop, n_schedule)
    coefficient_data = float(lambda_data) * weight

    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else None
    records = {}
    for case_id in selected:
        spec = specs[case_id]
        case = _make_case(section, spec, config, allow_synthetic=False)
        closure = closure_from_precursor(case.precursor, like=torch.zeros((), dtype=torch.float64))
        scales = scales_from_precursor(case.precursor, like=torch.zeros((), dtype=torch.float64))
        generator = torch.Generator().manual_seed(
            int(args.batch_seed) + sum(ord(ch) for ch in case_id)
        )
        base = sample_case_batch(
            case, n_pde=int(args.n_pde), n_ic=int(args.n_ic), n_right=0,
            n_data=0, n_contact=0, generator=generator,
        )
        label = _label_check(
            case, contact, closure, prediction_dir,
            prediction_tag=str(args.prediction_tag),
        )
        batches = []
        for which in range(max(1, min(int(args.batches), 3))):
            batch, info = _stratified_contact_batch(
                case, base, generator, which, int(args.n_contact)
            )
            with torch.enable_grad():
                entries, angles = _contact_term_batch(
                    case=case, batch=batch, contact=contact, air=air,
                    closure=closure, scales=scales,
                    coefficient_data=coefficient_data,
                    lambda_pde=lambda_pde, contact_data_weight=weight,
                    threshold=float(args.zero_threshold),
                )
            batches.append({
                "slice": which,
                "contact_selection": info,
                "terms": entries,
                "angles": angles,
            })
        records[case_id] = {
            "role": str(spec.get("role", "train")),
            "charge_radius_m": float(case.charge_radius_m),
            "label_check": label,
            "batches": batches,
        }
        print(case_id, "label:", label.get("status"), "batches:", len(batches))

    summary = {
        "checkpoint": str(args.checkpoint),
        "prediction_tag": str(args.prediction_tag),
        "loop": loop,
        "contact_data_weight": weight,
        "lambda_pde": float(lambda_pde),
        "lambda_data": float(lambda_data),
        "contact_data_coefficient": coefficient_data,
        "zero_threshold": float(args.zero_threshold),
        "terms_in_contact_stage": {
            "pde": "lambda_pde * pde_loss(air functional, geometry from ContactNet)",
            "contact_ic": "1.0 * contact_ic_loss",
            "interface": "1.0 * interface_loss",
            "contact_data": "lambda_data * contact_data_weight * contact_data_loss",
        },
        "cases": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
