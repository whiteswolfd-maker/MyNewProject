"""生成并核对启动链路的**外部**验证证据。

固定的一组计算（物理参数、外边界与网格增长规则保持一致）：

* ``N=800,  CFL=0.2``  —— 待交接运行；
* ``N=1600, CFL=0.2``  —— 网格敏感性；
* ``N=1600, CFL=0.1``  —— 时间步敏感性。

只用两组网格，所以结果是「网格敏感性」，**不是**渐近收敛阶。证据由本脚本计算
并写出，不接受手工填写；阈值由 ``assess_validation_evidence`` 判定。

场量比较的口径：在两侧共同的材料域上按体积权重 ``r^2 dr`` 求相对 L2 误差，
产物侧取 ``[0, min(R_c)]``，空气侧取 ``[max(R_c), min(R_s)]``（只比较压缩层，
不让未扰动空气稀释误差），分别对 ``rho, u, e`` 计算后取最大值。插值始终按材料
分开，不跨界面。

用法::

    PYTHONPATH=src python -m spherical_blast_pinn.cli.validate_startup
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from ..physics.two_medium_initial import build_two_medium_initial_field
from ..physics.two_medium_startup import (
    METHOD_VERSION,
    StartupConfig,
    TwoMediumStartup,
    build_handoff,
)

_SAMPLES = 400


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/startup_validation"))
    return parser


def _run(field, *, cells: int, cfl: float, growth: float, outer: float, end_time: float):
    solver = TwoMediumStartup(
        field,
        config=StartupConfig(
            cells_inside=cells,
            outside_growth=growth,
            outer_radius_factor=outer,
            cfl=cfl,
        ),
    )
    solver.run_to(end_time)
    return solver, build_handoff(solver)


def _field_errors(coarse, fine) -> dict[str, float]:
    """共同材料域上的体积加权相对 L2 误差；逐变量返回。"""
    errors: dict[str, float] = {}
    product_upper = min(coarse.contact_radius, fine.contact_radius) * (1.0 - 1.0e-6)
    if product_upper > 0.0:
        radii = np.linspace(0.0, product_upper, _SAMPLES)
        left = coarse.sample_product(radii)
        right = fine.sample_product(radii)
        weights = radii**2
        for key in ("density", "velocity", "specific_energy"):
            numerator = float(np.sum(weights * (left[key] - right[key]) ** 2))
            denominator = float(np.sum(weights * right[key] ** 2))
            errors[f"product_{key}"] = (
                math.sqrt(numerator / denominator) if denominator > 0.0 else math.inf
            )
    else:
        for key in ("density", "velocity", "specific_energy"):
            errors[f"product_{key}"] = math.inf

    air_lower = max(coarse.contact_radius, fine.contact_radius) * (1.0 + 1.0e-6)
    air_upper = min(coarse.shock_radius, fine.shock_radius) * (1.0 - 1.0e-6)
    if air_upper > air_lower:
        radii = np.linspace(air_lower, air_upper, _SAMPLES)
        left = coarse.sample_air(radii)
        right = fine.sample_air(radii)
        weights = radii**2
        for key in ("density", "velocity", "specific_energy"):
            numerator = float(np.sum(weights * (left[key] - right[key]) ** 2))
            denominator = float(np.sum(weights * right[key] ** 2))
            errors[f"air_{key}"] = (
                math.sqrt(numerator / denominator) if denominator > 0.0 else math.inf
            )
    else:
        for key in ("density", "velocity", "specific_energy"):
            errors[f"air_{key}"] = math.inf
    return errors


def _front_relative(coarse, fine) -> float:
    contact = abs(coarse.contact_radius - fine.contact_radius) / max(
        abs(fine.contact_radius), 1.0e-30
    )
    shock = abs(coarse.shock_radius - fine.shock_radius) / max(
        abs(fine.shock_radius), 1.0e-30
    )
    return float(max(contact, shock))


def _contact_metrics(handoff) -> dict:
    states = handoff.contact_states()
    return {
        "pressure_relative": float(states["pressure_relative"]),
        "velocity_relative": float(states["velocity_relative"]),
        "contact_pressure_product": float(states["product"]["pressure"]),
        "contact_pressure_air": float(states["air"]["pressure"]),
        "product_cell_radius_m": float(states["product"]["radius_m"]),
        "air_cell_radius_m": float(states["air"]["radius_m"]),
        "face_linear_diagnostic": states["face_linear"],
        "metric": (
            "pressure difference between the two cells adjacent to the contact, "
            "normalized by their maximum; it is a discrete upper bound on the face "
            "jump and includes the half-cell gradient"
        ),
    }


def _run_record(run_id: str, handoff, config: dict) -> dict:
    return {
        "id": run_id,
        "cells_inside": int(config["cells_inside"]),
        "outside_growth": float(config["outside_growth"]),
        "outer_radius_factor": float(config["outer_radius_factor"]),
        "cfl": float(config["cfl"]),
        "charge_radius": float(handoff.charge_radius),
        "handoff_time_s": float(handoff.time),
    }


def _diagnostics(handoff) -> dict:
    diagnostics = handoff.provenance["front_diagnostics"]
    return {
        "contact_radius_m": handoff.contact_radius,
        "shock_radius_m": handoff.shock_radius,
        "contact_speed_m_s": handoff.contact_speed,
        "shock_speed_m_s": handoff.shock_speed,
        "shock_speed_trajectory_m_s": diagnostics.get("shock_speed_trajectory"),
        "shock_speed_deviation": diagnostics.get("shock_speed_deviation"),
        "shock_rh_mass_rel": diagnostics.get("shock_rh_mass_rel"),
        "shock_rh_momentum_rel": diagnostics.get("shock_rh_momentum_rel"),
        "shock_rh_energy_rel": diagnostics.get("shock_rh_energy_rel"),
        "shock_diagnostics_valid": diagnostics.get("shock_diagnostics_valid"),
        "energy": handoff.provenance["energy"],
        "volume_work": handoff.provenance["volume_work"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    field = build_two_medium_initial_field(charge_radius=args.charge_radius)
    common = dict(
        growth=args.outside_growth, outer=args.outer_radius_factor, end_time=args.handoff_time
    )
    coarse_solver, coarse = _run(field, cells=args.coarse_cells, cfl=args.cfl, **common)
    fine_solver, fine = _run(field, cells=args.fine_cells, cfl=args.cfl, **common)
    small_cfl_solver, small_cfl = _run(
        field, cells=args.fine_cells, cfl=args.fine_cfl, **common
    )

    grid_errors = _field_errors(coarse, fine)
    timestep_errors = _field_errors(fine, small_cfl)
    grid_front = _front_relative(coarse, fine)
    timestep_front = _front_relative(fine, small_cfl)
    # 接触一致性必须取**待交接运行**本身；细网格只作对照，不能代替它。
    contact = _contact_metrics(coarse)
    contact_comparison = _contact_metrics(fine)

    ids = {
        "coarse": f"n{args.coarse_cells}_cfl{args.cfl:g}",
        "fine": f"n{args.fine_cells}_cfl{args.cfl:g}",
        "fine_small_cfl": f"n{args.fine_cells}_cfl{args.fine_cfl:g}",
    }
    configs = {
        ids["coarse"]: coarse_solver.config,
        ids["fine"]: fine_solver.config,
        ids["fine_small_cfl"]: small_cfl_solver.config,
    }
    evidence = {
        "source": (
            "python -m spherical_blast_pinn.cli.validate_startup "
            f"--charge-radius {args.charge_radius:g} --handoff-time {args.handoff_time:g} "
            f"--coarse-cells {args.coarse_cells} --fine-cells {args.fine_cells} "
            f"--cfl {args.cfl:g} --fine-cfl {args.fine_cfl:g} "
            f"(generated {datetime.now(timezone.utc).isoformat(timespec='seconds')})"
        ),
        "method_version": METHOD_VERSION,
        "charge_radius": float(args.charge_radius),
        "handoff_time_s": float(args.handoff_time),
        "initial_state": dict(fine.provenance["initial_state"]),
        "eos": {key: float(value) for key, value in fine.product_eos.as_dict().items()},
        "artificial_viscosity": {
            "q1": float(fine_solver.config.q1),
            "q2": float(fine_solver.config.q2),
        },
        "runs": [
            _run_record(ids["coarse"], coarse, {
                "cells_inside": coarse_solver.config.cells_inside,
                "outside_growth": coarse_solver.config.outside_growth,
                "outer_radius_factor": coarse_solver.config.outer_radius_factor,
                "cfl": coarse_solver.config.cfl,
            }),
            _run_record(ids["fine"], fine, {
                "cells_inside": fine_solver.config.cells_inside,
                "outside_growth": fine_solver.config.outside_growth,
                "outer_radius_factor": fine_solver.config.outer_radius_factor,
                "cfl": fine_solver.config.cfl,
            }),
            _run_record(ids["fine_small_cfl"], small_cfl, {
                "cells_inside": small_cfl_solver.config.cells_inside,
                "outside_growth": small_cfl_solver.config.outside_growth,
                "outer_radius_factor": small_cfl_solver.config.outer_radius_factor,
                "cfl": small_cfl_solver.config.cfl,
            }),
        ],
        "handoff_run": ids["coarse"],
        "grid_convergence": {
            "coarse_run": ids["coarse"],
            "fine_run": ids["fine"],
            "front_relative": float(grid_front),
            "field_relative": float(max(grid_errors.values())),
            "per_variable": grid_errors,
        },
        "timestep_sensitivity": {
            "coarse_run": ids["fine"],
            "fine_run": ids["fine_small_cfl"],
            "front_relative": float(timestep_front),
            "field_relative": float(max(timestep_errors.values())),
            "per_variable": timestep_errors,
        },
        "contact_consistency": contact,
        "contact_consistency_comparison_run": contact_comparison,
        "metric_definition": (
            "volume-weighted (r^2 dr) relative L2 on the common material-side domain: "
            "product [0, min Rc], air [max Rc, min Rs]; max over rho, u, e; "
            "two grids only, so this is a sensitivity result, not an observed order"
        ),
        "run_diagnostics": {
            ids["coarse"]: _diagnostics(coarse),
            ids["fine"]: _diagnostics(fine),
            ids["fine_small_cfl"]: _diagnostics(small_cfl),
        },
    }

    verdict_handoff = build_handoff(coarse_solver, validation_evidence=evidence)
    verdict = verdict_handoff.external_validation

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "startup_validation_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    (out_dir / "startup_validation_verdict.json").write_text(
        json.dumps(
            {
                "accepted": verdict["accepted"],
                "reasons": verdict["reasons"],
                "thresholds": verdict["thresholds"],
                "training_qualified": verdict_handoff.training_qualified,
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        ),
        encoding="utf-8",
    )

    print(f"method        : {METHOD_VERSION}")
    print(f"t_h           : {args.handoff_time:.6e} s   R0 = {args.charge_radius:.6g} m")
    print(f"runs          : {ids['coarse']}, {ids['fine']}, {ids['fine_small_cfl']}")
    print(
        f"grid          : front {grid_front:.3e}  field {max(grid_errors.values()):.3e}"
    )
    print(
        f"timestep      : front {timestep_front:.3e}  field {max(timestep_errors.values()):.3e}"
    )
    print(
        f"contact       : pressure {contact['pressure_relative']:.3e}  "
        f"velocity {contact['velocity_relative']:.3e}"
    )
    for name, value in sorted(grid_errors.items()):
        print(f"  grid {name:<22} {value:.3e}")
    for name, value in sorted(timestep_errors.items()):
        print(f"  dt   {name:<22} {value:.3e}")
    print(f"accepted      : {verdict['accepted']}")
    for reason in verdict["reasons"]:
        print(f"  REJECTED: {reason}")
    print(f"training      : {'QUALIFIED' if verdict_handoff.training_qualified else 'NOT READY'}")
    print(f"written       : {out_dir / 'startup_validation_evidence.json'}")
    print(f"                {out_dir / 'startup_validation_verdict.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
