"""Build the PINN vs data-only summary table used by the comparison chart.

For each model and case this script reports:

* peak-overpressure RMSE and relative RMS from ``compare_empirical.py``;
* full-field rho/u/p normalized RMSE from ``evaluate``;
* one normalized PDE residual on a fixed 2048-point sample.

The PDE sample and seed are identical for every model, so the last column is a
like-for-like measure of physical consistency, not a training-loss value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from spherical_blast_pinn.contact_air.data import load_case
from spherical_blast_pinn.contact_air.losses import pde_loss
from spherical_blast_pinn.contact_air.mapping import build_geometry
from spherical_blast_pinn.contact_air.models import (
    AirNet,
    ContactNet,
    air_fields_with_derivatives,
    closure_from_precursor,
    scales_from_precursor,
)
from spherical_blast_pinn.contact_air.sampling import sample_case_batch
from spherical_blast_pinn.contact_air.trainer import _detach_contact


MODELS = {
    "PINN": "outputs/single_r50_bounded_cheb_logp_6600/last.pt",
    "DataNN-576": "outputs/single_r50_data_only_matched/last.pt",
    "DataNN-2112": "outputs/single_r50_data_only/last.pt",
}

CASES = {
    "R50": {
        "cache": "outputs/fullframes/cache/r50_full.npz",
        "eval": "outputs/single_r50_bounded_cheb_logp_6600/eval_r50_full.json",
        "evals": {
            "PINN": "outputs/single_r50_bounded_cheb_logp_6600/eval_r50_full.json",
            "DataNN-576": "outputs/single_r50_data_only_matched/eval_r50_full.json",
            "DataNN-2112": "outputs/single_r50_data_only/eval_r50_full.json",
        },
        "peaks": {
            "PINN": "outputs/empirical_comparison_r50/peak_overpressure_vs_Z.csv",
            "DataNN-576": "outputs/empirical_comparison_r50_data_only_matched/peak_overpressure_vs_Z.csv",
            "DataNN-2112": "outputs/empirical_comparison_r50_data_only/peak_overpressure_vs_Z.csv",
        },
    },
    "R100 zero-shot": {
        "cache": "outputs/fullframes/cache/r100_full.npz",
        "eval": "outputs/r100_zero_shot_from_r50/eval_r100_full.json",
        "evals": {
            "PINN": "outputs/r100_zero_shot_from_r50/eval_r100_full.json",
            "DataNN-576": "outputs/r100_data_only_matched/eval_r100_full.json",
            "DataNN-2112": "outputs/r100_data_only/eval_r100_full.json",
        },
        "peaks": {
            "PINN": "outputs/empirical_comparison_r100_zero_shot/peak_overpressure_vs_Z.csv",
            "DataNN-576": "outputs/empirical_comparison_r100_data_only_matched/peak_overpressure_vs_Z.csv",
            "DataNN-2112": "outputs/empirical_comparison_r100_data_only/peak_overpressure_vs_Z.csv",
        },
    },
}


def _build_networks(describe_contact: dict, describe_air: dict):
    contact = ContactNet(
        width=int(describe_contact["width"]),
        depth=int(describe_contact["depth"]),
        time_input_mode=str(describe_contact["time_input"]),
        similarity_input_divisor=float(
            describe_contact.get("similarity_input_divisor", 64.0)
        ),
        contact_output=str(describe_contact.get("contact_output", "sigmoid_rs")),
        time_features=str(describe_contact.get("time_features", "linear")),
        chebyshev_degree=int(describe_contact.get("chebyshev_degree", 0)),
    )
    air = AirNet(
        width=int(describe_air["width"]),
        depth=int(describe_air["depth"]),
        time_input_mode=str(describe_air["time_input"]),
        similarity_input_divisor=float(
            describe_air.get("similarity_input_divisor", 64.0)
        ),
    )
    return contact, air


def _pde_residual(checkpoint: Path, cache: Path) -> float:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contact, air = _build_networks(
        payload["contact_net_describe"], payload["air_net_describe"]
    )
    contact.load_state_dict(payload["contact_net"])
    air.load_state_dict(payload["air_net"])
    contact.eval()
    air.eval()

    case = load_case(cache, role="train")
    like = torch.zeros(1, 1, dtype=torch.float64)
    closure = closure_from_precursor(case.precursor, like=like)
    scales = scales_from_precursor(case.precursor, like=like)
    batch = sample_case_batch(
        case, n_pde=2048, n_ic=0, n_right=0, n_data=0, n_contact=0,
        generator=torch.Generator().manual_seed(20261120),
    )
    t = batch.t_pde.reshape(-1)
    r0 = torch.full_like(t, float(case.charge_radius_m)).reshape(-1, 1)
    rs, ds = case.shock_tensors(t)
    rs = rs.reshape(-1, 1)
    ds = ds.reshape(-1, 1)
    c_fields = contact(
        t.reshape(-1, 1), r0, rs, ds, closure,
        time_input=case.time_input, create_graph=False,
    )
    c_fields = _detach_contact(c_fields)
    geometry = build_geometry(
        c_fields.radius, c_fields.speed, rs, ds, batch.zeta_pde.reshape(-1, 1)
    )
    fields, derivatives = air_fields_with_derivatives(
        air, batch.zeta_pde, t, r0, scales,
        create_graph=True, time_input=case.time_input,
    )
    term = pde_loss(fields=fields, derivatives=derivatives, geometry=geometry, case=case)
    return float(term.total.detach())


def _peak_metrics(path: Path) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data = np.asarray([float(row["dp_data_Pa"]) for row in rows])
    model = np.asarray([float(row["dp_model_Pa"]) for row in rows])
    error = model - data
    return (
        float(np.sqrt(np.mean(error ** 2)) / 1.0e6),
        float(np.sqrt(np.mean((error / data) ** 2)) * 100.0),
    )


def _eval_metrics(path: Path) -> tuple[float, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    air = payload["air"]
    return float(air["rho_rmse"]), float(air["u_rmse"]), float(air["p_rmse"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", default="outputs/model_comparison/summary.csv"
    )
    parser.add_argument(
        "--skip-pde", action="store_true",
        help="reuse pde_total from an existing summary instead of recomputing",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case_name, spec in CASES.items():
        pde_cache = {}
        if not args.skip_pde:
            for model, checkpoint in MODELS.items():
                print(f"PDE residual: {case_name} / {model}")
                pde_cache[model] = _pde_residual(Path(checkpoint), Path(spec["cache"]))
        for model in MODELS:
            peak_rmse, peak_rel = _peak_metrics(Path(spec["peaks"][model]))
            rho, u, p = _eval_metrics(Path(spec["evals"][model]))
            rows.append({
                "case": case_name,
                "model": model,
                "peak_rmse_MPa": peak_rmse,
                "peak_rel_rms_pct": peak_rel,
                "rho_rmse": rho,
                "u_rmse": u,
                "p_rmse": p,
                "pde_total": pde_cache.get(model, float("nan")),
            })

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
