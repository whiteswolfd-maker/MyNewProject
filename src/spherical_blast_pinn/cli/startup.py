"""Build the small-positive-time planar-Riemann startup / handoff field.

This is the explicit entry point for the 2026-10-25 initialization decision.  It

1. constructs the exact ``t_phys = 0`` uniform product sphere + stationary air
   field with :mod:`spherical_blast_pinn.physics.two_medium_initial`;
2. solves the local planar JWL-product/air Riemann problem once;
3. embeds the self-similar fan in the spherical domain at
   ``t_phys = epsilon`` and writes the handoff arrays and diagnostics.

It does **not** call ``ProductsDetonation``, ``SphericalPrecursor``, the old
``TwoMediumStartup.run_to`` or any finite-volume/Lagrangian time stepping.  It
does not start training.

Usage::

    PYTHONPATH=src python -m spherical_blast_pinn.cli.startup \
        --charge-radius 0.05 --delta 0.005
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..physics.two_medium_initial import build_two_medium_initial_field
from ..physics.two_medium_startup_handoff import write_planar_startup_handoff
from ..physics.two_medium_startup_planar import (
    PLANAR_STARTUP_MODEL_ID,
    PlanarStartupConfig,
    build_planar_startup_field,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge-radius", type=float, default=0.05, help="R0, m")
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0 / 200.0,
        help="candidate air-layer ratio (Rs-Rc)/R0; not a validated constant",
    )
    parser.add_argument(
        "--outer-radius-factor", type=float, default=40.0, help="outer radius / R0"
    )
    parser.add_argument("--inner-cells", type=int, default=64)
    parser.add_argument("--fan-cells", type=int, default=96)
    parser.add_argument("--plateau-cells", type=int, default=8)
    parser.add_argument("--air-cells", type=int, default=16)
    parser.add_argument("--ambient-cells", type=int, default=64)
    parser.add_argument("--dense-samples", type=int, default=8192)
    parser.add_argument("--integration-step-factor", type=float, default=0.005)
    parser.add_argument("--root-tolerance", type=float, default=1.0e-11)
    parser.add_argument(
        "--product-density",
        type=float,
        default=None,
        help="optional initial product density in kg/m^3 (not the EOS rho0)",
    )
    parser.add_argument(
        "--product-specific-energy",
        type=float,
        default=None,
        help="optional product specific energy in J/kg (not divided by density again)",
    )
    parser.add_argument("--air-density", type=float, default=None)
    parser.add_argument("--air-pressure", type=float, default=None)
    parser.add_argument("--air-gamma", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/planar_startup"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    initial = build_two_medium_initial_field(
        charge_radius=args.charge_radius,
        product_density=args.product_density,
        product_specific_energy=args.product_specific_energy,
        air_density=args.air_density,
        air_pressure=args.air_pressure,
        air_gamma=args.air_gamma,
        outer_radius=args.outer_radius_factor * args.charge_radius,
    )
    config = PlanarStartupConfig(
        delta=args.delta,
        inner_cells=args.inner_cells,
        fan_cells=args.fan_cells,
        plateau_cells=args.plateau_cells,
        air_cells=args.air_cells,
        ambient_cells=args.ambient_cells,
        dense_samples=args.dense_samples,
        integration_step_factor=args.integration_step_factor,
        root_relative_tolerance=args.root_tolerance,
    )
    field = build_planar_startup_field(initial, config=config)
    out_dir = Path(args.output_dir) / PLANAR_STARTUP_MODEL_ID / field.cache_key()[:16]
    json_path, npz_path = write_planar_startup_handoff(field, out_dir)

    checks = field.checks
    print(f"model_id          : {field.model_id}")
    print(f"method            : {field.method_version}")
    print(f"riemann method    : {field.solution.method_version}")
    print(f"t_phys / tau      : {field.time:.9e} s / {field.pin_time:.1f} s")
    print(f"epsilon / delta   : {field.epsilon:.9e} s / {field.delta:.9e}")
    print(f"R0 / outer        : {field.charge_radius:.9g} / {field.radii[-1]:.9g} m")
    print(f"Rc / Rs           : {field.contact_radius:.9g} / {field.shock_radius:.9g} m")
    print(f"u* / D_s          : {field.contact_speed:.9g} / {field.shock_speed:.9g} m/s")
    print(
        "rarefaction       : "
        f"head r={field.rarefaction_head_radius:.9g} m, "
        f"tail r={field.rarefaction_tail_radius:.9g} m, "
        f"head speed={field.rarefaction_head_speed:.6g} m/s, "
        f"tail speed={field.rarefaction_tail_speed:.6g} m/s"
    )
    print(
        "disturbed / air   : "
        f"{field.product_disturbed_length_over_r0:.6g} R0 / "
        f"{field.air_layer_width / field.charge_radius:.6g} R0"
    )
    print(
        "contact residuals  : "
        f"p={checks['contact_pressure_relative']:.3e}, "
        f"u={checks['contact_velocity_relative']:.3e}"
    )
    print(
        "spherical residuals: "
        f"mass={checks['spherical_mass_residual']:.3e}, "
        f"product mass={checks['spherical_product_mass_residual']:.3e}, "
        f"energy={checks['spherical_energy_residual']:.3e} "
        "(local planar geometry approximation; reported, not rescaled)"
    )
    print(
        "planar residuals   : "
        f"mass={checks['planar_mass_residual']:.3e}, "
        f"energy={checks['planar_energy_residual']:.3e}"
    )
    print(
        "energy metric      : "
        f"dE={checks['spherical_energy_change_signed_j_per_sr']:.6g} J/sr, "
        f"|dE|/U_product={checks['spherical_energy_change_abs_over_initial_product_internal']:.3e} "
        f"(U_product={checks['initial_product_internal_energy_per_sr']:.6g} J/sr); "
        f"global residual={checks['spherical_energy_residual']:.3e} "
        "(global denominator includes ambient air; supplemental diagnostic only)"
    )
    print(f"cells              : {field.n_cells} ({field.n_inside} product)")
    print(f"checks             : {'PASS' if checks['passed'] else 'FAIL'}")
    print(f"training_qualified: {field.training_qualified} (provenance only; not a run gate)")
    print(f"cache_key          : {field.cache_key()}")
    print(f"written            : {npz_path}")
    print(f"                     {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
