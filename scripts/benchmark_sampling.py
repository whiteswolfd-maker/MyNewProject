#!/usr/bin/env python3
"""Measure the indexed sampler and, optionally, a real training loop.

This is an engineering benchmark, not a convergence result.  It compares the
indexed sampler with its own scan fallback (the pre-index code path) on the
same observations and random seeds.  The fallback is used only for timing.

    PYTHONPATH=src python scripts/benchmark_sampling.py \
        --config configs/contact_air/split_train_r50_500_fullframes.yaml \
        --case r550_full --calls 30 \
        --output outputs/fullframes/sampling_benchmark.json

Add ``--full-step`` to build the configured 10/10 split and time one complete
alternating step (fast vs scan), the contact-stage forward/backward on one
case, and validation scoring.  The trainer selects CPU explicitly; CUDA
synchronisation is therefore a no-op, but the helper still synchronises when a
CUDA device is actually used.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from spherical_blast_pinn.config import load_config  # noqa: E402
from spherical_blast_pinn.contact_air.cli import (  # noqa: E402
    _case_specs,
    _make_case,
    _networks,
    _section,
    _train_config,
)
from spherical_blast_pinn.contact_air.data import AirObservationStore  # noqa: E402
from spherical_blast_pinn.contact_air.losses import (  # noqa: E402
    contact_data_loss,
    contact_ic_loss,
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
from spherical_blast_pinn.contact_air.trainer import (  # noqa: E402
    AlternatingTrainer,
    seed_everything,
)


def _sync() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _timed(function, repeats: int) -> dict:
    _sync()
    started = time.perf_counter()
    for _ in range(int(repeats)):
        function()
    _sync()
    elapsed = time.perf_counter() - started
    return {
        "repeats": int(repeats),
        "total_s": elapsed,
        "mean_s": elapsed / float(repeats),
    }


def _contact_inputs(case, t):
    t_flat = t.reshape(-1)
    r0 = torch.full_like(t_flat, float(case.charge_radius_m)).reshape(-1, 1)
    rs, ds = case.shock_tensors(t_flat)
    return r0, rs.reshape(-1, 1).to(torch.float64), ds.reshape(-1, 1).to(torch.float64)


def _contact_stage_forward(trainer, case, batch):
    closure = trainer._closures[case.case_id]
    scales = trainer._scales[case.case_id]
    r0, rs, ds = _contact_inputs(case, batch.t_pde)
    contact = trainer.contact_net
    air = trainer.air_net
    c_fields = contact(
        batch.t_pde.reshape(-1, 1), r0, rs, ds, closure, create_graph=True
    )
    geometry = build_geometry(
        c_fields.radius, c_fields.speed, rs, ds, batch.zeta_pde.reshape(-1, 1)
    )
    fields, derivatives = air_fields_with_derivatives(
        air, batch.zeta_pde, batch.t_pde, r0, scales,
        create_graph=True, functional=True,
    )
    pde = pde_loss(fields=fields, derivatives=derivatives, geometry=geometry,
                   case=case)
    t0 = torch.full_like(batch.t_pde, float(case.t0))
    r00, rs0, ds0 = _contact_inputs(case, t0)
    c0 = contact(
        t0.reshape(-1, 1), r00, rs0, ds0, closure, create_graph=True
    )
    ic = contact_ic_loss(c0, case)
    air0 = air.forward_functional(
        torch.zeros_like(batch.t_pde).reshape(-1, 1),
        batch.t_pde.reshape(-1, 1), r0.reshape(-1, 1), scales,
        detach_parameters=True,
    )
    interface = interface_loss(air_at_contact=air0, contact=c_fields, case=case)
    if batch.t_contact.numel():
        r0c, rsc, dsc = _contact_inputs(case, batch.t_contact)
        c_contact = contact(
            batch.t_contact.reshape(-1, 1), r0c, rsc, dsc, closure,
            create_graph=True,
        )
        data = contact_data_loss(c_contact, batch, case)
    else:
        data = None
    lambda_pde, lambda_data = schedule_weights(
        trainer.loop, trainer.config.n_schedule
    )
    total = lambda_pde * pde.total + ic.total + interface.total
    if data is not None:
        total = total + lambda_data * trainer.config.contact_data_weight * data.total
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", default="r550_full")
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--scan-calls", type=int, default=5)
    parser.add_argument("--full-step", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    section = _section(config)
    specs = {str(spec["case_id"]): spec for spec in _case_specs(section)}
    case = _make_case(section, specs[args.case], config, allow_synthetic=False)

    def draw():
        sample_case_batch(
            case, n_pde=205, n_ic=51, n_right=51, n_data=51, n_contact=51,
            generator=torch.Generator().manual_seed(7),
        )

    # Warm the page cache and torch kernels.
    draw()
    fast_sample = _timed(draw, args.calls)

    original_sample = AirObservationStore.sample
    AirObservationStore.sample = AirObservationStore._sample_scan
    try:
        draw()
        scan_sample = _timed(draw, args.scan_calls)
    finally:
        AirObservationStore.sample = original_sample

    payload = {
        "config": args.config,
        "case": args.case,
        "observations": case.observations.size,
        "frames": int(case.observations._frame_ids.size),
        "sampling": {
            "fast": fast_sample,
            "scan": scan_sample,
            "speedup": (
                scan_sample["mean_s"] / fast_sample["mean_s"]
                if fast_sample["mean_s"] > 0.0 else None
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "note": "the trainer constructs networks on CPU; GPU is not used",
        },
    }

    if args.full_step:
        train_specs = [
            spec for spec in _case_specs(section)
            if str(spec.get("role", "train")) == "train"
        ]
        validation_specs = [
            spec for spec in _case_specs(section)
            if str(spec.get("role", "train")) == "validation"
        ]
        train_cases = [
            _make_case(section, spec, config, allow_synthetic=False)
            for spec in train_specs
        ]
        validation_cases = [
            _make_case(section, spec, config, allow_synthetic=False)
            for spec in validation_specs
        ]
        seed_everything(int(section.get("training", {}).get("seed", 0)))
        contact, air = _networks(section)
        trainer = AlternatingTrainer(
            cases=train_cases, contact_net=contact, air_net=air,
            config=_train_config(section, None),
            output_dir=Path(args.output).parent / "_benchmark_trainer_run",
            external_validation_cases=validation_cases,
        )
        def step():
            trainer.step()
        step()
        fast_step = _timed(step, 2)
        AirObservationStore.sample = AirObservationStore._sample_scan
        try:
            step()
            scan_step = _timed(step, 1)
        finally:
            AirObservationStore.sample = original_sample
        trainer.evaluate()
        evaluate = _timed(trainer.evaluate, 1)

        # Contact-stage forward and backward on one case, same batch.
        def contact_batch():
            return sample_case_batch(
                case, n_pde=205, n_ic=51, n_right=0, n_data=0, n_contact=51,
                generator=torch.Generator().manual_seed(31),
            )

        # Warm up graph construction and the backward kernels.
        trainer.opt_contact.zero_grad(set_to_none=True)
        with torch.enable_grad():
            warm = _contact_stage_forward(trainer, case, contact_batch())
        warm.backward()
        batch = contact_batch()
        trainer.opt_contact.zero_grad(set_to_none=True)
        _sync()
        started = time.perf_counter()
        with torch.enable_grad():
            total = _contact_stage_forward(trainer, case, batch)
        _sync()
        forward_s = time.perf_counter() - started
        started = time.perf_counter()
        total.backward()
        _sync()
        backward_s = time.perf_counter() - started
        payload["step"] = {
            "fast": fast_step,
            "scan": scan_step,
            "speedup": (
                scan_step["mean_s"] / fast_step["mean_s"]
                if fast_step["mean_s"] > 0.0 else None
            ),
            "contact_stage_forward_s": forward_s,
            "contact_stage_backward_s": backward_s,
            "validation_evaluate_s": evaluate["mean_s"],
            "note": "contact-stage timings are one case on the configured CPU device",
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
