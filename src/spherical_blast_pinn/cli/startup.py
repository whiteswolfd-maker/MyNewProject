"""构造「分段常数初值 → 短时两介质推进 → 交接场」并写出文件。

这是新链路的显式入口。它**不读**旧参考场缓存、不调用旧爆轰求解器，也不启动
训练。输出目录绑定模型标识、方法版本与缓存键，避免和旧 ``reference_field``
的产物混在一起。

用法::

    PYTHONPATH=src python -m spherical_blast_pinn.cli.startup \
        --charge-radius 0.05 --handoff-time 1e-6 --cells-inside 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..physics.two_medium_initial import build_two_medium_initial_field
from ..physics.two_medium_startup import (
    STARTUP_MODEL_ID,
    StartupConfig,
    TwoMediumStartup,
    build_handoff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge-radius", type=float, default=0.05, help="R0, m")
    parser.add_argument(
        "--handoff-time", type=float, default=1.0e-6, help="t_h > 0 in physical seconds"
    )
    parser.add_argument("--cells-inside", type=int, default=2000)
    parser.add_argument("--outside-growth", type=float, default=1.005)
    parser.add_argument("--outer-radius-factor", type=float, default=40.0)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--q1", type=float, default=1.5)
    parser.add_argument("--q2", type=float, default=0.06)
    parser.add_argument(
        "--validation-evidence",
        type=Path,
        default=None,
        help="JSON with grid/time-step/contact validation bound to this run",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/startup"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    field = build_two_medium_initial_field(charge_radius=args.charge_radius)
    solver = TwoMediumStartup(
        field,
        config=StartupConfig(
            cells_inside=args.cells_inside,
            outside_growth=args.outside_growth,
            outer_radius_factor=args.outer_radius_factor,
            cfl=args.cfl,
            q1=args.q1,
            q2=args.q2,
        ),
    )
    solver.run_to(args.handoff_time)
    evidence = None
    if args.validation_evidence is not None:
        evidence = json.loads(
            Path(args.validation_evidence).read_text(encoding="utf-8")
        )
    handoff = build_handoff(solver, validation_evidence=evidence)

    out_dir = Path(args.output_dir) / STARTUP_MODEL_ID / handoff.cache_key()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "handoff.npz",
        radii=handoff.radii,
        velocity=handoff.velocity,
        density=handoff.density,
        specific_energy=handoff.specific_energy,
        pressure=handoff.pressure,
        sound_speed=handoff.sound_speed,
        material=handoff.material,
        mass=handoff.mass,
        cell_volume=handoff.cell_volume,
    )
    report = handoff.as_dict()
    (out_dir / "handoff.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )

    print(f"model_id      : {handoff.model_id}")
    print(f"method        : {handoff.method_version}")
    print(f"t_phys        : {handoff.time:.6e} s   (tau = t_phys - t_h)")
    print(f"R0 / outer    : {handoff.charge_radius:.6g} / {handoff.radii[-1]:.6g} m")
    print(f"Rc / Rs       : {handoff.contact_radius:.6g} / {handoff.shock_radius:.6g} m")
    print(f"Rc_dot/Rs_dot : {handoff.contact_speed:.6g} / {handoff.shock_speed:.6g} m/s")
    print(
        f"time levels   : all state at t^n = {handoff.time:.6e} s "
        "(no half-step velocity)"
    )
    print(f"cells         : {handoff.material.size} ({handoff.n_inside} product)")
    print(
        "energy        : "
        f"internal={report['energy']['internal']:.6e} "
        f"kinetic={report['energy']['kinetic']:.6e} "
        f"boundary_work={report['energy']['boundary_work']:.6e} "
        f"drift={report['energy']['drift']:.2e}"
    )
    print(
        f"volume/work   : last={report['volume_work']['last_relative_mismatch']:.2e} "
        f"max={report['volume_work']['max_relative_mismatch']:.2e}"
    )
    failed = handoff.qualification["internal_checks"].get("failed", [])
    print(
        "internal      : "
        + ("PASS" if handoff.internal_checks_passed else "FAIL")
        + (f" (failed: {', '.join(failed)})" if failed else "")
    )
    external = handoff.external_validation
    print(
        "external      : "
        + ("ACCEPTED" if external["accepted"] else "MISSING/REJECTED")
        + (f" ({'; '.join(external['reasons'])})" if external.get("reasons") else "")
    )
    if handoff.training_qualified:
        print("training      : QUALIFIED by the handoff checks (training entry is not wired)")
    else:
        print("training      : NOT READY - diagnostic snapshot only")
    print(f"cache_key     : {handoff.cache_key()}")
    print(f"written       : {out_dir / 'handoff.npz'}")
    print(f"                {out_dir / 'handoff.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
