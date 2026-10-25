"""Diagnostics for the local-planar-Riemann startup at several ``delta`` values.

This is a **sensitivity report**, not a time-convergence study: different
``delta`` values correspond to different physical times ``epsilon``.  The
purpose is to check the self-similar scalings, the wave ordering, the EOS
consistency and the independently integrated mass/energy residuals.

It does not start training and does not call the old Lagrangian startup.

Usage::

    PYTHONPATH=src python -m spherical_blast_pinn.cli.validate_startup_planar \
        --charge-radius 0.05 --deltas 0.005,0.0025,0.00125
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..physics.two_medium_initial import build_two_medium_initial_field
from ..physics.two_medium_startup_planar import (
    PLANAR_STARTUP_MODEL_ID,
    PlanarStartupConfig,
    build_planar_startup_field,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge-radius", type=float, default=0.05)
    parser.add_argument(
        "--deltas",
        type=str,
        default="0.005,0.0025,0.00125",
        help="comma-separated candidate delta=(Rs-Rc)/R0 values",
    )
    parser.add_argument("--outer-radius-factor", type=float, default=40.0)
    parser.add_argument("--inner-cells", type=int, default=32)
    parser.add_argument("--fan-cells", type=int, default=64)
    parser.add_argument("--plateau-cells", type=int, default=4)
    parser.add_argument("--air-cells", type=int, default=8)
    parser.add_argument("--ambient-cells", type=int, default=32)
    parser.add_argument("--dense-samples", type=int, default=4096)
    parser.add_argument("--integration-step-factor", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/planar_startup_validation"))
    return parser


def _parse_deltas(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("at least one delta is required")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deltas = _parse_deltas(args.deltas)
    initial = build_two_medium_initial_field(
        charge_radius=args.charge_radius,
        outer_radius=args.outer_radius_factor * args.charge_radius,
    )
    records = []
    for delta in deltas:
        config = PlanarStartupConfig(
            delta=delta,
            inner_cells=args.inner_cells,
            fan_cells=args.fan_cells,
            plateau_cells=args.plateau_cells,
            air_cells=args.air_cells,
            ambient_cells=args.ambient_cells,
            dense_samples=args.dense_samples,
            integration_step_factor=args.integration_step_factor,
        )
        field = build_planar_startup_field(initial, config=config)
        checks = field.checks
        records.append(
            {
                "delta": float(delta),
                "epsilon_s": float(field.epsilon),
                "contact_radius_m": float(field.contact_radius),
                "shock_radius_m": float(field.shock_radius),
                "air_layer_width_m": float(field.air_layer_width),
                "rarefaction_head_radius_m": float(field.rarefaction_head_radius),
                "rarefaction_tail_radius_m": float(field.rarefaction_tail_radius),
                "product_disturbed_length_m": float(field.product_disturbed_length),
                "product_disturbed_length_over_r0": float(
                    field.product_disturbed_length_over_r0
                ),
                "contact_speed_m_s": float(field.contact_speed),
                "shock_speed_m_s": float(field.shock_speed),
                "p_star_pa": float(field.solution.p_star),
                "u_star_m_s": float(field.solution.u_star),
                "shock_speed_minus_u_star_m_s": float(
                    field.solution.shock_speed - field.solution.u_star
                ),
                "checks_passed": bool(checks["passed"]),
                "contact_pressure_relative": float(checks["contact_pressure_relative"]),
                "contact_velocity_relative": float(checks["contact_velocity_relative"]),
                "spherical_mass_residual": float(checks["spherical_mass_residual"]),
                "spherical_product_mass_residual": float(
                    checks["spherical_product_mass_residual"]
                ),
                "spherical_energy_residual": float(checks["spherical_energy_residual"]),
                "planar_mass_residual": float(checks["planar_mass_residual"]),
                "planar_energy_residual": float(checks["planar_energy_residual"]),
                "grid_vs_dense_spherical_mass_residual": float(
                    checks["grid_vs_dense_spherical_mass_residual"]
                ),
                "grid_vs_dense_spherical_energy_residual": float(
                    checks["grid_vs_dense_spherical_energy_residual"]
                ),
                "riemann_invariant_residual": float(
                    checks["riemann_checks"]["product_riemann_invariant_residual"]
                ),
                "cache_key": field.cache_key(),
            }
        )

    base = records[0]
    for record in records:
        record["epsilon_ratio_to_first"] = record["epsilon_s"] / base["epsilon_s"]
        record["contact_offset_ratio_to_first"] = (
            record["contact_radius_m"] - args.charge_radius
        ) / (base["contact_radius_m"] - args.charge_radius)
        record["shock_offset_ratio_to_first"] = (
            record["shock_radius_m"] - args.charge_radius
        ) / (base["shock_radius_m"] - args.charge_radius)
        record["disturbed_length_ratio_to_first"] = (
            record["product_disturbed_length_m"] / base["product_disturbed_length_m"]
        )
        record["star_state_matches_first"] = all(
            abs(record[key] - base[key]) <= 1.0e-9 * max(abs(base[key]), 1.0)
            for key in ("p_star_pa", "u_star_m_s", "shock_speed_minus_u_star_m_s")
        )
        if not record["checks_passed"]:
            raise RuntimeError(f"planar startup checks failed for delta={record['delta']}")
        if not record["star_state_matches_first"]:
            raise RuntimeError(
                f"star state drifted with delta={record['delta']}; this is not expected"
            )
        if record["rarefaction_head_radius_m"] <= 0.0:
            raise RuntimeError(f"release wave reached the centre for delta={record['delta']}")

    report = {
        "model_id": PLANAR_STARTUP_MODEL_ID,
        "purpose": (
            "delta sensitivity / self-similar scaling diagnostics; NOT a time-"
            "convergence study (different delta means different physical epsilon)"
        ),
        "charge_radius_m": float(args.charge_radius),
        "initial_state": {
            "product_density": float(initial.product_state["density"]),
            "product_specific_energy": float(initial.product_state["specific_energy"]),
            "air_density": float(initial.air_state["density"]),
            "air_pressure": float(initial.air_state["pressure"]),
            "air_gamma": float(initial.air_gamma),
        },
        "records": records,
        "notes": [
            "Rc-R0, Rs-R0 and the disturbed product length scale linearly with delta",
            "p_star, u_star and D_s are properties of the initial pressure jump, not of delta",
            "spherical mass/energy residuals are the local-planar geometric approximation",
            "planar mass/energy residuals check the Riemann fan itself",
            "no training is started and training_qualified remains false",
        ],
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "planar_startup_validation.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"model_id : {PLANAR_STARTUP_MODEL_ID}")
    print("delta sensitivity (different delta = different physical time):")
    for record in records:
        print(
            f"  delta={record['delta']:.9g}  eps={record['epsilon_s']:.6e}  "
            f"Rc={record['contact_radius_m']:.9g}  Rs={record['shock_radius_m']:.9g}  "
            f"head={record['rarefaction_head_radius_m']:.9g}  "
            f"disturbed/R0={record['product_disturbed_length_over_r0']:.6g}  "
            f"spherical mass={record['spherical_mass_residual']:.3e} "
            f"energy={record['spherical_energy_residual']:.3e}  "
            f"planar mass={record['planar_mass_residual']:.3e} "
            f"energy={record['planar_energy_residual']:.3e}"
        )
    print(f"written  : {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
