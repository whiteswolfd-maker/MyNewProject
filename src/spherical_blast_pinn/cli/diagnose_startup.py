"""定位启动验证失败来源的诊断脚本（只诊断，不放行）。

不改推进公式、物理参数或验收阈值，也不接入训练。输出两份 JSON：

* ``startup_diagnosis_evidence.json`` —— 各运行的接触一致性、前沿/空气层敏感性、
  物理坐标与前沿对齐坐标下的场差、共同域排除范围；
* ``startup_diagnosis_profiles.json`` —— 接触面附近逐单元的局部剖面。

用法::

    PYTHONPATH=src python -m spherical_blast_pinn.cli.diagnose_startup
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..physics.eos import jwl_pressure_numpy
from ..physics.two_medium_initial import build_two_medium_initial_field
from ..physics.two_medium_startup import METHOD_VERSION
from .validate_startup import (
    _contact_metrics,
    _field_errors,
    _front_relative,
    _run,
)

_SAMPLES = 400
_PROFILE_HALF_WIDTH = 6


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge-radius", type=float, default=0.05)
    parser.add_argument("--handoff-time", type=float, default=1.0e-6)
    parser.add_argument("--coarse-cells", type=int, default=800)
    parser.add_argument("--fine-cells", type=int, default=1600)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--fine-cfl", type=float, default=0.1)
    parser.add_argument("--outside-growth", type=float, default=1.01)
    parser.add_argument("--outer-radius-factor", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/startup_diagnosis"))
    return parser


def _centres(solver) -> np.ndarray:
    return (0.5 * (solver.radii[1:] ** 3 + solver.radii[:-1] ** 3)) ** (1.0 / 3.0)


def _contact_profile(solver) -> dict[str, Any]:
    centres = _centres(solver)
    contact_cell = int(solver._contact_cell)
    low = max(0, contact_cell - _PROFILE_HALF_WIDTH)
    high = min(solver.mass.size - 1, contact_cell + _PROFILE_HALF_WIDTH + 1)
    cells = []
    for index in range(low, high + 1):
        cells.append(
            {
                "index": index,
                "material": int(solver.material[index]),
                "centre_m": float(centres[index]),
                "width_m": float(solver.radii[index + 1] - solver.radii[index]),
                "density": float(solver.density[index]),
                "specific_energy": float(solver.specific_energy[index]),
                "pressure": float(solver.pressure[index]),
                "artificial_viscosity": float(solver.artificial_viscosity[index]),
                "total_pressure": float(
                    solver.pressure[index] + solver.artificial_viscosity[index]
                ),
                "node_velocity_left": float(solver.velocity[index]),
                "node_velocity_right": float(solver.velocity[index + 1]),
            }
        )
    return {
        "contact_cell": contact_cell,
        "contact_radius_m": float(solver._contact_radius()),
        "contact_node_velocity": float(solver.velocity[contact_cell + 1]),
        "cells": cells,
    }


def _one_sided_face_pressure(solver, order: int) -> dict[str, float]:
    """单侧外推到接触面的压力。``order=1`` 线性，``order=2`` 二次。

    只作为诊断：产物稀疏波在最后几个单元很陡，外推值对阶数敏感；不用两侧平均。
    """
    centres = _centres(solver)
    contact_cell = int(solver._contact_cell)
    face = float(solver._contact_radius())

    def side(indices):
        cs = centres[indices]
        rho = np.polyval(np.polyfit(cs, solver.density[indices], order), face)
        energy = np.polyval(
            np.polyfit(cs, solver.specific_energy[indices], order), face
        )
        return float(rho), float(energy)

    product_indices = np.arange(contact_cell - order, contact_cell + 1)
    air_indices = np.arange(contact_cell + 1, contact_cell + 2 + order)
    rho_p, e_p = side(product_indices)
    rho_a, e_a = side(air_indices)
    pressure_p = float(
        jwl_pressure_numpy(
            np.array([rho_p]), np.array([e_p]), solver.product_eos
        )[0]
    )
    pressure_a = float((solver.air_gamma - 1.0) * rho_a * e_a)
    return {
        "product_pressure": pressure_p,
        "air_pressure": pressure_a,
        "pressure_relative": abs(pressure_p - pressure_a)
        / max(abs(pressure_p), abs(pressure_a), 1.0e-30),
        "product_density": rho_p,
        "air_density": rho_a,
    }


def _aligned_errors(coarse, fine) -> dict[str, float]:
    """按各自前沿位置对齐后的场差。

    产物用 ``xi = r / Rc``，空气用 ``eta = (r - Rc) / (Rs - Rc)``，两端都映射到
    ``[0, 1]``。权重取各自坐标下的球体积权重（产物 ``xi^2``，空气
    ``(Rc + eta L)^2``），``L`` 因子在相对范数里约掉。
    """
    errors: dict[str, float] = {}
    xi = np.linspace(0.0, 1.0, _SAMPLES)
    product_coarse = coarse.sample_product(xi * coarse.contact_radius)
    product_fine = fine.sample_product(xi * fine.contact_radius)
    weight_product = xi**2
    for key in ("density", "velocity", "specific_energy"):
        numerator = float(
            np.sum(weight_product * (product_coarse[key] - product_fine[key]) ** 2)
        )
        denominator = float(np.sum(weight_product * product_fine[key] ** 2))
        errors[f"product_{key}"] = (
            float(np.sqrt(numerator / denominator)) if denominator > 0.0 else float("inf")
        )

    eta = np.linspace(0.0, 1.0, _SAMPLES)
    length_coarse = coarse.shock_radius - coarse.contact_radius
    length_fine = fine.shock_radius - fine.contact_radius
    air_coarse = coarse.sample_air(coarse.contact_radius + eta * length_coarse)
    air_fine = fine.sample_air(fine.contact_radius + eta * length_fine)
    weight_air = (fine.contact_radius + eta * length_fine) ** 2
    for key in ("density", "velocity", "specific_energy"):
        numerator = float(np.sum(weight_air * (air_coarse[key] - air_fine[key]) ** 2))
        denominator = float(np.sum(weight_air * air_fine[key] ** 2))
        errors[f"air_{key}"] = (
            float(np.sqrt(numerator / denominator)) if denominator > 0.0 else float("inf")
        )
    return errors


def _front_record(handoff) -> dict[str, float]:
    return {
        "contact_radius_m": float(handoff.contact_radius),
        "shock_radius_m": float(handoff.shock_radius),
        "air_layer_m": float(handoff.shock_radius - handoff.contact_radius),
        "contact_speed_m_s": float(handoff.contact_speed),
        "shock_speed_m_s": float(handoff.shock_speed),
    }


def _relative_change(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-30)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    field = build_two_medium_initial_field(charge_radius=args.charge_radius)
    common = dict(
        growth=args.outside_growth, outer=args.outer_radius_factor, end_time=args.handoff_time
    )
    coarse_solver, coarse = _run(field, cells=args.coarse_cells, cfl=args.cfl, **common)
    fine_solver, fine = _run(field, cells=args.fine_cells, cfl=args.cfl, **common)
    small_solver, small = _run(
        field, cells=args.fine_cells, cfl=args.fine_cfl, **common
    )

    physical = _field_errors(coarse, fine)
    aligned = _aligned_errors(coarse, fine)
    timestep_physical = _field_errors(fine, small)

    fronts = {
        f"n{args.coarse_cells}_cfl{args.cfl:g}": _front_record(coarse),
        f"n{args.fine_cells}_cfl{args.cfl:g}": _front_record(fine),
        f"n{args.fine_cells}_cfl{args.fine_cfl:g}": _front_record(small),
    }
    coarse_id = f"n{args.coarse_cells}_cfl{args.cfl:g}"
    fine_id = f"n{args.fine_cells}_cfl{args.cfl:g}"
    small_id = f"n{args.fine_cells}_cfl{args.fine_cfl:g}"

    delta_contact = abs(coarse.contact_radius - fine.contact_radius)
    delta_shock = abs(coarse.shock_radius - fine.shock_radius)
    layer = fine.shock_radius - fine.contact_radius
    excluded = {
        "delta_contact_m": float(delta_contact),
        "delta_shock_m": float(delta_shock),
        "fine_air_layer_m": float(layer),
        "delta_contact_over_fine_layer": float(delta_contact / layer) if layer > 0 else float("inf"),
        "delta_shock_over_fine_layer": float(delta_shock / layer) if layer > 0 else float("inf"),
        "physical_product_excluded_band_m": float(delta_contact),
        "physical_air_excluded_bands_m": [float(delta_contact), float(delta_shock)],
    }

    contact_by_run = {
        coarse_id: _contact_metrics(coarse),
        fine_id: _contact_metrics(fine),
        small_id: _contact_metrics(small),
    }
    face_estimates = {
        coarse_id: {
            "linear": _one_sided_face_pressure(coarse_solver, 1),
            "quadratic": _one_sided_face_pressure(coarse_solver, 2),
        },
        fine_id: {
            "linear": _one_sided_face_pressure(fine_solver, 1),
            "quadratic": _one_sided_face_pressure(fine_solver, 2),
        },
    }

    grid_sensitivity = {
        "contact": _relative_change(coarse.contact_radius, fine.contact_radius),
        "shock": _relative_change(coarse.shock_radius, fine.shock_radius),
        "air_layer": _relative_change(
            coarse.shock_radius - coarse.contact_radius,
            fine.shock_radius - fine.contact_radius,
        ),
        "contact_speed": _relative_change(coarse.contact_speed, fine.contact_speed),
        "shock_speed": _relative_change(coarse.shock_speed, fine.shock_speed),
        "front_relative_combined": _front_relative(coarse, fine),
    }
    timestep_sensitivity = {
        "contact": _relative_change(fine.contact_radius, small.contact_radius),
        "shock": _relative_change(fine.shock_radius, small.shock_radius),
        "air_layer": _relative_change(
            fine.shock_radius - fine.contact_radius,
            small.shock_radius - small.contact_radius,
        ),
        "contact_speed": _relative_change(fine.contact_speed, small.contact_speed),
        "shock_speed": _relative_change(fine.shock_speed, small.shock_speed),
        "front_relative_combined": _front_relative(fine, small),
    }

    evidence = {
        "usage": "diagnostic only; not an acceptance gate",
        "method_version": METHOD_VERSION,
        "charge_radius": float(args.charge_radius),
        "handoff_time_s": float(args.handoff_time),
        "handoff_run": coarse_id,
        "runs": [coarse_id, fine_id, small_id],
        "fronts": fronts,
        "grid_sensitivity": grid_sensitivity,
        "timestep_sensitivity": timestep_sensitivity,
        "contact_by_run": contact_by_run,
        "face_pressure_estimates": face_estimates,
        "field_errors_physical_coordinates": {
            "per_variable": physical,
            "max": float(max(physical.values())),
            "definition": (
                "volume-weighted (r^2 dr) relative L2 on the common physical domain: "
                "product [0, min Rc], air [max Rc, min Rs]"
            ),
        },
        "field_errors_front_aligned": {
            "per_variable": aligned,
            "max": float(max(aligned.values())),
            "definition": (
                "same relative L2 after mapping product xi=r/Rc and air "
                "eta=(r-Rc)/(Rs-Rc) to [0,1]; weights are the spherical volume "
                "weights in each coordinate (the layer-width factor cancels)"
            ),
        },
        "timestep_field_errors_physical": {
            "per_variable": timestep_physical,
            "max": float(max(timestep_physical.values())),
        },
        "excluded_regions": excluded,
        "aligned_over_physical_ratio": {
            key: (
                float(aligned[key] / physical[key])
                if physical[key] > 0.0
                else float("inf")
            )
            for key in physical
        },
    }

    profiles = {
        "usage": "diagnostic only; local contact profiles",
        "method_version": METHOD_VERSION,
        "handoff_time_s": float(args.handoff_time),
        "profiles": {
            coarse_id: _contact_profile(coarse_solver),
            fine_id: _contact_profile(fine_solver),
            small_id: _contact_profile(small_solver),
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "startup_diagnosis_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    (out_dir / "startup_diagnosis_profiles.json").write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )

    print(f"method        : {METHOD_VERSION}")
    print(f"handoff run   : {coarse_id}  (contact metric must come from this run)")
    print("fronts:")
    for run_id, record in fronts.items():
        print(
            f"  {run_id:<14} Rc={record['contact_radius_m']:.8f} "
            f"Rs={record['shock_radius_m']:.8f} L={record['air_layer_m']*1e3:.4f} mm "
            f"Rc_dot={record['contact_speed_m_s']:.1f} Rs_dot={record['shock_speed_m_s']:.1f}"
        )
    print(
        "grid sensitivity   : "
        + " ".join(f"{k}={v:.3e}" for k, v in grid_sensitivity.items())
    )
    print(
        "dt   sensitivity   : "
        + " ".join(f"{k}={v:.3e}" for k, v in timestep_sensitivity.items())
    )
    for run_id, metrics in contact_by_run.items():
        print(
            f"contact {run_id:<14} adjacent p rel={metrics['pressure_relative']:.4f} "
            f"(p_prod={metrics['contact_pressure_product']:.4g} "
            f"p_air={metrics['contact_pressure_air']:.4g})"
        )
    for run_id, estimates in face_estimates.items():
        for order, values in estimates.items():
            print(
                f"face {order:<9} {run_id:<14} p_prod={values['product_pressure']:.4g} "
                f"p_air={values['air_pressure']:.4g} rel={values['pressure_relative']:.4f}"
            )
    print("physical errors:")
    for key, value in sorted(physical.items()):
        print(f"  {key:<26} {value:.3e}   aligned {aligned[key]:.3e}")
    print(
        "excluded       : dRc={:.3e} m ({:.1f}% of fine layer)  dRs={:.3e} m ({:.1f}%)".format(
            excluded["delta_contact_m"],
            100.0 * excluded["delta_contact_over_fine_layer"],
            excluded["delta_shock_m"],
            100.0 * excluded["delta_shock_over_fine_layer"],
        )
    )
    print(f"written        : {out_dir / 'startup_diagnosis_evidence.json'}")
    print(f"                 {out_dir / 'startup_diagnosis_profiles.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
