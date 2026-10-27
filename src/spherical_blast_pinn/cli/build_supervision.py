"""Build a supervision file from an existing extracted DYNA profile.

This is a conversion command, not a training command.  It reads
``raw_line.npz`` + ``material.npz`` + ``metadata.json`` through the existing
:func:`load_extracted_case`, binds the profile to the startup handoff's
``epsilon``, and writes the new versioned supervision contract.

The DYNA time origin is explicit::

    T_origin = LT + R0 / D
    t_phys   = T_DYNA - T_origin
    tau      = t_phys - epsilon_case

``LT`` and ``D`` come from the extraction metadata unless explicitly supplied.
An unknown origin may be written for audit, but such a file is refused by
supervision and same-time comparison until the caller supplies it.

Example::

    python -m spherical_blast_pinn.cli.build_supervision \
        --profile-dir F:/2/extracted_geom/50mm_d \
        --handoff-json outputs/planar_startup_diagnostic/<...>/handoff.json \
        --output outputs/supervision/tnt_50mm.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data.dyna_supervision import (
    DynaSupervisionError,
    origin_parameters_from_metadata,
    pressure_gauge_series,
    supervision_from_extracted_case,
)
from ..data.extracted_case import DataError, load_extracted_case
from ..data.startup_supervision import (
    SUPERVISION_FORMAT,
    SUPERVISION_VERSION,
    SupervisionError,
    TimeOrigin,
)
from ..physics import material as mat
from ..physics.two_medium_startup_handoff import load_planar_startup_handoff
from ._common import SetupError, configure_stdout, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--handoff-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--time-origin-status",
        choices=("formula", "unknown"),
        default="formula",
        help="formula uses LT + R0/D; unknown writes a review-only file",
    )
    parser.add_argument(
        "--detonation-velocity",
        type=float,
        default=None,
        help="D in m/s; defaults to the extraction metadata",
    )
    parser.add_argument(
        "--lighting-time-s",
        type=float,
        default=None,
        help="LT in seconds; defaults to *INITIAL_DETONATION time in the metadata",
    )
    parser.add_argument("--contact-delta-m", type=float, default=None)
    parser.add_argument("--shock-delta-m", type=float, default=None)
    parser.add_argument(
        "--gauge-radius-m",
        action="append",
        type=float,
        default=[],
        help="fixed physical radius for a reference pressure history; repeatable",
    )
    parser.add_argument(
        "--ambient-pressure-pa",
        type=float,
        default=mat.AIR_PRESSURE,
        help="ambient pressure used only for gauge overpressure",
    )
    parser.add_argument("--origin-method", default="caller declared unknown")
    parser.add_argument(
        "--origin-evidence",
        default="no confirmed DYNA-to-model time origin was supplied",
    )
    return parser


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-30)


def _field_counts(dataset) -> dict[str, int]:
    return {
        "density": int(np.count_nonzero(dataset.valid_density)),
        "velocity": int(np.count_nonzero(dataset.valid_velocity)),
        "specific_energy": int(np.count_nonzero(dataset.valid_specific_energy)),
        "pressure": int(np.count_nonzero(dataset.valid_pressure)),
    }


def _medium_counts(dataset) -> dict[str, int]:
    codes, counts = np.unique(dataset.medium, return_counts=True)
    names = {-1: "unclassified", 0: "product", 1: "air"}
    return {
        names.get(int(code), f"code_{int(code)}"): int(count)
        for code, count in zip(codes, counts)
    }


def _report(
    *,
    dataset,
    profile_dir: Path,
    handoff_json: Path,
    epsilon_s: float,
    origin_parameters,
    gauges: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    ready = not dataset.supervision_blockers()
    tau = None
    if dataset.time_origin.ready:
        tau = dataset.tau_time(epsilon_s)
    return {
        "format": SUPERVISION_FORMAT,
        "version": SUPERVISION_VERSION,
        "case_id": dataset.case_id,
        "synthetic": bool(dataset.synthetic),
        "source_kind": dataset.source_kind,
        "profile_dir": str(profile_dir),
        "handoff_json": str(handoff_json),
        "charge_radius_m": float(dataset.charge_radius_m),
        "epsilon_case_s": float(epsilon_s),
        "time_origin": dataset.time_origin.as_dict(),
        "origin_parameters": None if origin_parameters is None else origin_parameters.as_dict(),
        "n_frames": int(dataset.n_frames),
        "n_radial_points": int(dataset.radius.shape[1]),
        "times_dyna_s": [float(v) for v in dataset.times_dyna],
        "tau_s": None if tau is None else [float(v) for v in tau],
        "medium_counts": _medium_counts(dataset),
        "valid_field_counts": _field_counts(dataset),
        "front_valid_counts": {
            "contact": int(np.count_nonzero(dataset.contact_valid)),
            "shock": int(np.count_nonzero(dataset.shock_valid)),
        },
        "front_delta_m": {
            "contact": float(dataset.contact_delta_m),
            "shock": float(dataset.shock_delta_m),
        },
        "supervision_ready": ready,
        "supervision_blockers": [] if ready else dataset.supervision_blockers(),
        "gauges": [dict(gauge) for gauge in gauges],
        "field_provenance": dataset.source.get("extraction", {}).get("field_provenance"),
        "source": dataset.source,
        "notes": list(dataset.notes or []),
        "output": str(output),
    }


def _run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_stdout()
    try:
        case = load_extracted_case(args.profile_dir)
    except DataError as exc:
        raise SetupError(f"cannot load extracted profile {args.profile_dir}: {exc}") from exc

    try:
        field = load_planar_startup_handoff(args.handoff_json)
    except ValueError as exc:
        raise SetupError(f"cannot load startup handoff {args.handoff_json}: {exc}") from exc
    if _relative_error(float(case.charge_radius), float(field.charge_radius)) > 1.0e-12:
        raise SetupError(
            f"profile R0={case.charge_radius!r} m and handoff "
            f"R0={field.charge_radius!r} m disagree"
        )

    epsilon = float(field.epsilon)
    if args.time_origin_status == "formula":
        origin_parameters = origin_parameters_from_metadata(
            case,
            epsilon_s=epsilon,
            detonation_velocity_m_s=args.detonation_velocity,
            lighting_time_s=args.lighting_time_s,
        )
        time_origin = origin_parameters.as_time_origin()
    else:
        origin_parameters = None
        time_origin = TimeOrigin(
            status="unknown",
            origin_time_s=None,
            method=str(args.origin_method),
            evidence=str(args.origin_evidence),
        )

    dataset = supervision_from_extracted_case(
        case,
        time_origin=time_origin,
        epsilon_s=epsilon,
        synthetic=False,
        contact_delta_m=args.contact_delta_m,
        shock_delta_m=args.shock_delta_m,
        notes=(
            "converted by cli.build_supervision; no optimizer or trainer was invoked",
            f"handoff cache_key={field.cache_key()}",
        ),
    )
    dataset.save(args.output)
    gauges = [
        pressure_gauge_series(
            dataset,
            radius_m=float(radius),
            ambient_pressure_pa=float(args.ambient_pressure_pa),
            epsilon_s=epsilon,
        )
        for radius in args.gauge_radius_m
    ]
    report = _report(
        dataset=dataset,
        profile_dir=args.profile_dir,
        handoff_json=args.handoff_json,
        epsilon_s=epsilon,
        origin_parameters=origin_parameters,
        gauges=gauges,
        output=args.output,
    )
    report_path = args.report or args.output.with_suffix(".json")
    write_json(report_path, report)

    print(f"case              : {dataset.case_id}")
    print(f"DYNA states       : {dataset.n_frames}")
    print(f"T_origin          : {dataset.time_origin.origin_time_s!r} s")
    print(f"epsilon_case      : {epsilon!r} s")
    print(f"contact/shock ok  : {report['front_valid_counts']}")
    print(f"supervision ready : {report['supervision_ready']}")
    if not report["supervision_ready"]:
        for blocker in report["supervision_blockers"]:
            print(f"  blocked: {blocker}")
    for gauge in gauges:
        print(
            f"gauge {gauge['radius_m']:.6g} m : "
            f"{gauge['n_valid']}/{gauge['n_frames']} frames available"
        )
    print(f"written           : {args.output}")
    print(f"report            : {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (SetupError, DynaSupervisionError, SupervisionError, DataError) as exc:
        print(f"\n** {exc} **\n", file=sys.stderr)
        return 6


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
