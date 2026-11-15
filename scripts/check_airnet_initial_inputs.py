#!/usr/bin/env python3
"""Read-only audit of AirNet inputs at each case's t=t0.

For every case this script checks:

* the similarity time argument ``(t0-t0)/(T_star*64)`` is exactly zero;
* the legacy argument ``t0/time_reference_s`` is not zero and differs by case;
* the theoretical separated-air target and the three output scales are the
  same for all cases;
* with a common ``R0`` (and the case's own ``t0``/``T_star``), the ordinary and
  functional paths return the same fields, so at fixed ``zeta`` the remaining
  cross-case input difference is ``R0`` only;
* the ordinary AirNet path, ``air_fields_with_derivatives`` and the frozen- or
  live-parameter functional path agree numerically at t0;
* the official training IC path and the evaluation initial-condition path
  receive the same t0 input.

No checkpoint, sample, weight or physical formula is changed.
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
import evaluate_contact_lr_resume as base  # noqa: E402
from spherical_blast_pinn.contact_air.losses import air_ic_loss  # noqa: E402
from spherical_blast_pinn.contact_air.models import (  # noqa: E402
    air_fields_with_derivatives,
)


ZETAS = np.linspace(0.01, 0.99, 33)
CONFIG = "configs/contact_air/split_train_r50_500_fullframes_similarity.yaml"
B_LAST = "outputs/contact_air_time_input_similarity_4000/last.pt"
B_BEST = "outputs/contact_air_time_input_similarity_4000/best.pt"
LEGACY_CONFIG = "configs/contact_air/split_train_r50_500_fullframes_legacy_4000.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def _norm_fields(fields, case) -> dict[str, np.ndarray]:
    return {
        "rho": (fields.density / case.rho_scale).detach().reshape(-1).numpy(),
        "u": (fields.velocity / case.u_scale).detach().reshape(-1).numpy(),
        "p": (fields.pressure / case.p_scale).detach().reshape(-1).numpy(),
    }


def common_r0_control(
    net,
    cases,
    zeta: torch.Tensor,
    common_r0: torch.Tensor,
    scales_by_case: dict,
) -> tuple[dict, dict]:
    """Evaluate every case at its own t0 with a common physical R0.

    The previous diagnostic used t=0 globally.  That is wrong: in similarity
    mode the network time is (t-case.t0)/(T_star*64), so t=0 is not t0 unless
    the case origin is zero.  Here each case keeps its own t0 and time_input;
    only R0 is replaced.  The returned record includes the actual t, origin,
    scale and scaled time, and the function hard-fails if scaled time is not
    exactly zero.
    """
    records = {}
    fields = {}
    for case in cases:
        t = torch.full_like(zeta, float(case.t0))
        xt = net.scaled_time(t, case.time_input)
        max_abs = float(xt.abs().max())
        if max_abs != 0.0:
            raise RuntimeError(
                f"{case.case_id}: common-R0 control scaled_time is {max_abs}, "
                "not zero; t must be the case t0"
            )
        records[case.case_id] = {
            "t_s": float(case.t0),
            "origin_s": float(case.time_input.origin),
            "scale_s": float(case.time_input.scale),
            "scaled_time_max_abs": max_abs,
        }
        with torch.no_grad():
            fields[case.case_id] = net(
                zeta, t, common_r0, scales_by_case[case.case_id],
                time_input=case.time_input,
            )
    return records, fields


def _case_record(trainer, case, zeta: torch.Tensor) -> dict:
    t = torch.full_like(zeta, float(case.t0))
    r0 = torch.full_like(zeta, float(case.charge_radius_m))
    scales = trainer._scales[case.case_id]
    net = trainer.air_net

    with torch.no_grad():
        ordinary = net(zeta, t, r0, scales, time_input=case.time_input)
        functional_frozen = net.forward_functional(
            zeta, t, r0, scales, detach_parameters=True,
            time_input=case.time_input,
        )
        functional_live = net.forward_functional(
            zeta, t, r0, scales, detach_parameters=False,
            time_input=case.time_input,
        )
        # The same time input used by the ordinary evaluation/prediction path.
        xt = net.scaled_time(t, case.time_input)

    fields_from_helper, _ = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False,
        functional=False, time_input=case.time_input,
    )
    fields_from_helper_functional, _ = air_fields_with_derivatives(
        net, zeta, t, r0, scales, create_graph=False,
        functional=True, time_input=case.time_input,
    )

    # Exercise the exact training IC helper.  It only reads zeta_ic/t_ic.
    ic_batch = SimpleNamespace(zeta_ic=zeta, t_ic=t)
    ic_term = air_ic_loss(net, ic_batch, case, scales)

    # The evaluation initial-condition helper uses the same case.time_input.
    initial_eval = base._initial_condition_metrics(trainer, case)
    direct_rmse = {}
    fields = ordinary
    for name, values, scale in (
        ("rho", fields.density, case.rho_scale),
        ("u", fields.velocity, case.u_scale),
        ("p", fields.pressure, case.p_scale),
    ):
        error = (values / float(scale) - 1.0).reshape(-1).numpy()
        direct_rmse[name] = float(np.sqrt(np.mean(error * error)))

    return {
        "case_id": case.case_id,
        "role": case.role,
        "charge_radius_m": float(case.charge_radius_m),
        "t0_s": float(case.t0),
        "time_scale_s": float(case.time_scale),
        "x_t_similarity_at_t0_max_abs": float(xt.abs().max()),
        "output_scales": {
            "rho": float(case.rho_scale),
            "u": float(case.u_scale),
            "p": float(case.p_scale),
        },
        "targets_normalised": {"rho": 1.0, "u": 1.0, "p": 1.0},
        "ordinary_minus_functional_frozen_max_abs": {
            "rho": _max_abs(ordinary.density, functional_frozen.density),
            "u": _max_abs(ordinary.velocity, functional_frozen.velocity),
            "p": _max_abs(ordinary.pressure, functional_frozen.pressure),
        },
        "ordinary_minus_functional_live_max_abs": {
            "rho": _max_abs(ordinary.density, functional_live.density),
            "u": _max_abs(ordinary.velocity, functional_live.velocity),
            "p": _max_abs(ordinary.pressure, functional_live.pressure),
        },
        "ordinary_minus_helper_max_abs": {
            "rho": _max_abs(ordinary.density, fields_from_helper.density),
            "u": _max_abs(ordinary.velocity, fields_from_helper.velocity),
            "p": _max_abs(ordinary.pressure, fields_from_helper.pressure),
        },
        "helper_functional_minus_ordinary_max_abs": {
            "rho": _max_abs(
                fields_from_helper_functional.density, ordinary.density
            ),
            "u": _max_abs(
                fields_from_helper_functional.velocity, ordinary.velocity
            ),
            "p": _max_abs(
                fields_from_helper_functional.pressure, ordinary.pressure
            ),
        },
        "training_ic_loss": float(ic_term.total.detach()),
        "training_ic_count": int(ic_term.counts.get("ic", 0)),
        "evaluation_initial_rmse": {
            "rho": float(initial_eval["rho_rmse"]),
            "u": float(initial_eval["u_rmse"]),
            "p": float(initial_eval["p_rmse"]),
        },
        "direct_t0_rmse": direct_rmse,
        "evaluation_minus_direct_rmse": {
            name: float(initial_eval[f"{name}_rmse"] - direct_rmse[name])
            for name in ("rho", "u", "p")
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=(
        "outputs/fullframes/time_input_experiment/factor_checks/"
        "initial_input_checks.json"
    ))
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    trainer, _ = diag._build_trainer(CONFIG)
    legacy, _ = diag._build_trainer(LEGACY_CONFIG)
    cases = list(trainer.cases) + list(trainer.validation_cases)
    zeta = torch.as_tensor(ZETAS, dtype=torch.float64).reshape(-1, 1)

    records = {}
    for label, path in (("B_last", Path(B_LAST)), ("B_best", Path(B_BEST))):
        trainer.load(path)
        per_case = {
            case.case_id: _case_record(trainer, case, zeta)
            for case in cases
        }

        # At each case's own t0 and fixed zeta, replace only the physical R0.
        # t must be case.t0, not zero: t=0 would change scaled_time whenever
        # the case origin is not zero.
        common_r0 = torch.full_like(zeta, 0.5)
        control_records, common_fields_raw = common_r0_control(
            trainer.air_net, cases, zeta, common_r0, trainer._scales
        )
        common_fields = {
            case.case_id: _norm_fields(common_fields_raw[case.case_id], case)
            for case in cases
        }
        reference = common_fields[cases[0].case_id]
        common_max = {
            field: max(
                float(np.max(np.abs(values[field] - reference[field])))
                for values in common_fields.values()
            )
            for field in ("rho", "u", "p")
        }

        records[label] = {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256(path),
            "time_input_mode": trainer.air_net.time_input_mode,
            "r0_input_gate": trainer.air_net.r0_input_gate,
            "similarity_input_divisor": float(
                trainer.air_net.similarity_input_divisor
            ),
            "cases": per_case,
            "common_r0_control": control_records,
            "summary": {
                "n_cases": len(cases),
                "all_similarity_x_t_zero": all(
                    row["x_t_similarity_at_t0_max_abs"] == 0.0
                    for row in per_case.values()
                ),
                "max_similarity_x_t_at_t0": max(
                    row["x_t_similarity_at_t0_max_abs"]
                    for row in per_case.values()
                ),
                "common_r0_control_all_scaled_time_zero": all(
                    row["scaled_time_max_abs"] == 0.0
                    for row in control_records.values()
                ),
                "common_r0_control_actual_t_matches_case_t0": all(
                    row["t_s"] == row["origin_s"]
                    for row in control_records.values()
                ),
                "common_r0_control_max_scaled_time": max(
                    row["scaled_time_max_abs"]
                    for row in control_records.values()
                ),
                "all_output_scales_equal": len({
                    tuple(row["output_scales"].values())
                    for row in per_case.values()
                }) == 1,
                "all_targets_equal": len({
                    tuple(row["targets_normalised"].values())
                    for row in per_case.values()
                }) == 1,
                "max_ordinary_minus_functional_frozen": max(
                    max(row["ordinary_minus_functional_frozen_max_abs"].values())
                    for row in per_case.values()
                ),
                "max_ordinary_minus_functional_live": max(
                    max(row["ordinary_minus_functional_live_max_abs"].values())
                    for row in per_case.values()
                ),
                "max_ordinary_minus_helper": max(
                    max(row["ordinary_minus_helper_max_abs"].values())
                    for row in per_case.values()
                ),
                "max_helper_functional_minus_ordinary": max(
                    max(row["helper_functional_minus_ordinary_max_abs"].values())
                    for row in per_case.values()
                ),
                "max_evaluation_minus_direct_rmse": max(
                    abs(value)
                    for row in per_case.values()
                    for value in row["evaluation_minus_direct_rmse"].values()
                ),
                "common_r0_max_normalised_difference": common_max,
            },
        }

    # The legacy A model is the contrast case: its t0 input is not zero and
    # changes with the case's physical t0.
    legacy_records = {}
    for case in cases:
        t0 = torch.tensor([[float(case.t0)]], dtype=torch.float64)
        legacy_records[case.case_id] = {
            "t0_s": float(case.t0),
            "x_t_legacy_at_t0": float(
                legacy.air_net.scaled_time(t0).reshape(-1)[0]
            ),
            "time_reference_s": float(legacy.air_net.time_reference_s),
        }

    payload = {
        "protocol": {
            "config": CONFIG,
            "legacy_config": LEGACY_CONFIG,
            "zeta_grid": [float(v) for v in ZETAS],
            "zeta_grid_kind": "fixed internal normalised coordinate; not a physical radius",
            "t": "each case's physical t0",
            "r0": "each case's actual charge radius, except the common-R0 control",
            "common_r0_m": 0.5,
            "common_r0_t": "each case's own t0; recorded and asserted scaled_time=0",
            "non_destructive": True,
            "supersedes": (
                "initial_input_checks_wrong_t0_control.json, which used t=0 "
                "for the common-R0 control"
            ),
        },
        "models": records,
        "legacy_A_t0_time_argument": {
            "cases": legacy_records,
            "min": min(r["x_t_legacy_at_t0"] for r in legacy_records.values()),
            "max": max(r["x_t_legacy_at_t0"] for r in legacy_records.values()),
        },
        "notes": [
            "The similarity path is x_t=(t-t0)/(T_star*64); at t=t0 it is zero.",
            "The legacy path is x_t=t/time_reference_s; at t=t0 it is not zero.",
            "The common-R0 control holds each case's own t=t0, zeta and time_input fixed; only R0 is replaced by 0.5 m.",
            "The old control used t=0 and is preserved with an explicit error marker in initial_input_checks_wrong_t0_control.json.",
            "Ordinary, helper and functional fields are compared directly, not through a loss.",
            "No gradient, checkpoint or optimizer state is written.",
        ],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(output),
        "sha256": _sha256(output),
        "summary": {
            label: item["summary"] for label, item in records.items()
        },
        "legacy_x_t_range": [
            payload["legacy_A_t0_time_argument"]["min"],
            payload["legacy_A_t0_time_argument"]["max"],
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
