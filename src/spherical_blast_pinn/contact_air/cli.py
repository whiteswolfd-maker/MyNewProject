"""Command line for the ContactNet/AirNet scheme.

Commands (run from the repository root):

.. code-block:: text

    python -m spherical_blast_pinn.contact_air.cli precheck \
        --config configs/contact_air/base.yaml --case tnt_50mm

    python -m spherical_blast_pinn.contact_air.cli build-cache \
        --config configs/contact_air/base.yaml --case tnt_50mm \
        --profile-dir <extracted profile directory> --out <cache.npz>

    python -m spherical_blast_pinn.contact_air.cli train \
        --config configs/contact_air/smoke.yaml --cases three \
        --max-loops 4 --output-dir outputs/contact_air_smoke

    python -m spherical_blast_pinn.contact_air.cli predict-500 \
        --config configs/contact_air/base.yaml --checkpoint <last.pt> \
        --output <prediction.npz>

    python -m spherical_blast_pinn.contact_air.cli evaluate-500 \
        --config configs/contact_air/base.yaml --prediction <prediction.npz> \
        --output <evaluation.json>

No path is baked into the repository; cache and output paths come from the
configuration or the command line.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from scipy.interpolate import RegularGridInterpolator

from ..config import Config, load_config
from .data import (
    CaseData,
    CaseDataError,
    build_case_from_profile,
    load_case,
    save_case,
    synthetic_case,
)
from .models import AirNet, ContactNet
from .precursor import (
    PrecursorConfig,
    PrecursorError,
    compute_precursor,
)
from .trainer import AlternatingTrainer, TrainConfig, TrainingError


class CommandError(RuntimeError):
    """A command's inputs are missing or inconsistent."""


def _section(config: Config) -> dict[str, Any]:
    section = config.section("contact_air")
    if not section:
        raise CommandError("configuration has no contact_air section")
    return section


def _resolve(config: Config, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config.root / path)


def _precursor_config(section: Mapping[str, Any], radius: float) -> PrecursorConfig:
    payload = copy.deepcopy(dict(section["precursor"]))
    payload["charge_radius_m"] = float(radius)
    return PrecursorConfig.from_mapping(payload)


def _case_specs(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = list(section.get("cases", []) or [])
    if not specs:
        raise CommandError("contact_air.cases is empty")
    return [dict(spec) for spec in specs]


def _select_specs(section: Mapping[str, Any], which: str) -> list[dict[str, Any]]:
    specs = _case_specs(section)
    role = "train"
    if which == "three":
        selected = [spec for spec in specs if str(spec.get("role", "train")) == role]
    else:
        selected = [
            spec for spec in specs
            if str(spec.get("role", "train")) == role and which in str(spec.get("case_id", ""))
        ]
    if not selected:
        raise CommandError(f"no training case matches {which!r}")
    return selected


def _make_case(
    section: Mapping[str, Any], spec: Mapping[str, Any], config: Config, *,
    allow_synthetic: bool,
) -> CaseData:
    radius = float(spec["charge_radius_m"])
    case_id = str(spec["case_id"])
    role = str(spec.get("role", "train"))
    if role == "train" and abs(radius * 1000.0 - 500.0) <= 1.0e-9:
        raise CommandError(
            f"case {case_id!r}: R0=500 mm is the final-validation radius and "
            "must not be a training case"
        )
    cached = spec.get("cache")
    if cached:
        path = _resolve(config, cached)
        if not path.is_file():
            raise CommandError(f"case {case_id!r}: cache {path} does not exist")
        case = load_case(path, role=role)
        if abs(case.charge_radius_m - radius) > 1.0e-12 * max(radius, 1.0):
            raise CommandError(
                f"case {case_id!r}: cache R0={case.charge_radius_m!r}, config says {radius!r}"
            )
        return case
    if not bool(spec.get("synthetic", False)) or not allow_synthetic:
        raise CommandError(
            f"case {case_id!r}: no cache is configured and synthetic data is not allowed. "
            "Run build-cache on an extracted profile directory first."
        )
    precursor = compute_precursor(_precursor_config(section, radius))
    tolerances = dict(section["tolerances"])
    return synthetic_case(
        case_id=case_id, charge_radius_m=radius, precursor=precursor, role=role,
        n_frames=int(spec.get("n_frames", 41)),
        n_radial=int(spec.get("n_radial", 400)),
        duration_s=float(spec.get("duration_s", 2.0e-5)),
        tolerances=tolerances,
    )


def _networks(section: Mapping[str, Any]):
    if torch is None:  # pragma: no cover
        raise CommandError("PyTorch is required")
    net_section = section.get("networks", {})
    dtype_name = str(net_section.get("dtype", "float64"))
    dtype = getattr(torch, dtype_name, torch.float64)
    contact_section = net_section.get("contact", {})
    air_section = net_section.get("air", {})
    contact = ContactNet(
        width=int(contact_section.get("width", 50)),
        depth=int(contact_section.get("depth", 5)),
        dtype=dtype, device="cpu",
    )
    air = AirNet(
        width=int(air_section.get("width", 32)),
        depth=int(air_section.get("depth", 10)),
        dtype=dtype, device="cpu",
    )
    return contact, air


def _train_config(section: Mapping[str, Any], loops: int | None) -> TrainConfig:
    training = dict(section.get("training", {}))
    sampling = dict(section.get("sampling", {}))
    values = {
        "n_pde": int(sampling.get("n_pde", 512)),
        "n_ic_shock": int(sampling.get("n_ic_shock", 128)),
        "n_ic_ambient": int(sampling.get("n_ic_ambient", 128)),
        "n_right": int(sampling.get("n_right", 128)),
        "n_data": int(sampling.get("n_data", 128)),
        "n_contact": int(sampling.get("n_contact", 128)),
        "loops": int(training.get("loops", 2000)),
        "learning_rate": float(training.get("learning_rate", 1.0e-4)),
        "n_schedule": int(training.get("n_schedule", 2000)),
        "log_every": int(training.get("log_every", 50)),
        "checkpoint_every": int(training.get("checkpoint_every", 100)),
        "validate_every": int(training.get("validate_every", 100)),
        "seed": int(training.get("seed", 0)),
        "gradient_clip": float(training.get("gradient_clip", 0.0)),
        "validation_holdout_fraction": float(
            section.get("validation", {}).get("holdout_fraction", 0.3)
        ),
        "validation_data_points": int(
            section.get("validation", {}).get("data_points", 256)
        ),
        "validation_contact_points": int(
            section.get("validation", {}).get("contact_points", 256)
        ),
    }
    if loops is not None:
        values["loops"] = int(loops)
    return TrainConfig(**values)


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "detach"):
        return float(value.detach())
    return str(value)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def command_precheck(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    specs = _case_specs(section)
    if getattr(args, "all", False):
        rows = []
        for item in specs:
            if str(item.get("role", "train")) != "train":
                continue
            radius = float(item["charge_radius_m"])
            result = compute_precursor(_precursor_config(section, radius))
            rows.append({
                "case_id": item["case_id"],
                "charge_radius_m": radius,
                "R_sep": result.R_sep,
                "V_sep": result.V_sep,
                "p_sep": result.p_sep,
                "u_c_sep": result.u_c_sep,
                "D_s_sep": result.D_s_sep,
                "t_sep": result.t_sep,
                "t0": result.t0,
                "Rc0": result.Rc0,
                "Rs0": result.Rs0,
                "checks": dict(result.checks),
            })
        payload = {
            "status": "precursor_table",
            "note": "all three training radii; R_b(t0) is a data-side missing item",
            "rows": rows,
        }
        if args.output:
            _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, default=_jsonable))
        return 0
    wanted = args.case or "tnt_50mm"
    spec = next((item for item in specs if str(item.get("case_id")) == wanted), None)
    if spec is None:
        raise CommandError(f"no case {wanted!r} in contact_air.cases")
    radius = float(spec["charge_radius_m"])
    precursor_config = _precursor_config(section, radius)
    result = compute_precursor(precursor_config)
    report: dict[str, Any] = {
        "case_id": wanted,
        "charge_radius_m": radius,
        "precursor": result.as_dict(),
        "precursor_checks": dict(result.checks),
        "status": "precursor_ready_ic_unverified",
        "initial_condition_verified": False,
    }
    cached = spec.get("cache")
    if cached:
        path = _resolve(config, cached)
        if path.is_file():
            case = load_case(path, role=str(spec.get("role", "train")))
            report.update({
                "cache": str(path),
                "t0": case.t0,
                "t_end": case.t_end,
                "R_b_at_t0": case.initial.Rb0,
                "initial": case.as_dict()["initial"],
                "r_b_domain": case.boundary.domain,
                "observations": case.observations.size,
                "status": "ic_ready",
                "initial_condition_verified": True,
            })
        else:
            report["missing"] = [f"cache {path} does not exist"]
    else:
        report["missing"] = [
            "R_b(t0) trajectory: no cache is available; the full initial field "
            "cannot be generated for this case"
        ]
    if args.output:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, default=_jsonable))
    return 0


def command_build_cache(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    spec = next(
        (item for item in _case_specs(section) if str(item.get("case_id")) == args.case),
        None,
    )
    if spec is None:
        raise CommandError(f"no case {args.case!r} in contact_air.cases")
    radius = float(spec["charge_radius_m"])
    precursor = compute_precursor(_precursor_config(section, radius))
    case = build_case_from_profile(
        args.profile_dir, precursor=precursor,
        role=str(spec.get("role", "train")), tolerances=dict(section["tolerances"]),
        source={"profile_dir": str(args.profile_dir), "case_id": args.case},
    )
    save_case(case, args.out)
    print(json.dumps(case.as_dict(), indent=2, default=_jsonable))
    return 0


def command_train(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    specs = _select_specs(section, args.cases)
    cases = [_make_case(section, spec, config, allow_synthetic=not args.no_synthetic)
             for spec in specs]
    contact, air = _networks(section)
    train_config = _train_config(section, args.max_loops)
    output_dir = Path(args.output_dir) if args.output_dir else _resolve(
        config, section.get("output_dir", "outputs/contact_air")
    )
    trainer = AlternatingTrainer(
        cases=cases, contact_net=contact, air_net=air,
        config=train_config, output_dir=output_dir,
    )
    if args.resume:
        info = trainer.load(args.resume)
        print(f"resumed from {args.resume} at loop {info['loop']}")
    result = trainer.train()
    _write_json(output_dir / "train_result.json", result)
    print(json.dumps({
        "loops": result["loops"],
        "last_validation": result["last_validation"],
        "output_dir": str(output_dir),
    }, indent=2, default=_jsonable))
    return 0


def _load_checkpoint(path: Path, contact: ContactNet, air: AirNet) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spherical_blast_pinn.contact_air_checkpoint":
        raise CommandError(f"{path}: not a contact-air checkpoint")
    contact.load_state_dict(payload["contact_net"])
    air.load_state_dict(payload["air_net"])
    return payload


def command_predict_500(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    specs = _case_specs(section)
    wanted = args.case or "tnt_500mm"
    spec = next((item for item in specs if str(item.get("case_id")) == wanted), None)
    if spec is None:
        raise CommandError(f"no test case {wanted!r} in contact_air.cases")
    if str(spec.get("role", "test")) != "test":
        raise CommandError(f"case {wanted!r} is not role=test")
    case = _make_case(section, spec, config, allow_synthetic=True)
    contact, air = _networks(section)
    payload = _load_checkpoint(Path(args.checkpoint), contact, air)
    n_t = int(args.n_times)
    n_z = int(args.n_zeta)
    times = np.linspace(case.t0, case.t_end, n_t)
    zetas = np.linspace(0.0, 1.0, n_z)
    t_grid, z_grid = np.meshgrid(times, zetas, indexing="ij")
    t_tensor = torch.as_tensor(t_grid.reshape(-1, 1), dtype=torch.float64)
    z_tensor = torch.as_tensor(z_grid.reshape(-1, 1), dtype=torch.float64)
    r0 = torch.full_like(t_tensor, float(case.charge_radius_m))
    rb, rb_dot = case.boundary_tensors(t_grid.reshape(-1))
    rb = rb.reshape(-1, 1).to(torch.float64)
    rb_dot = rb_dot.reshape(-1, 1).to(torch.float64)
    from .models import closure_from_precursor, scales_from_precursor
    closure = closure_from_precursor(case.precursor, like=t_tensor)
    scales = scales_from_precursor(case.precursor, like=t_tensor)
    with torch.enable_grad():
        c_fields = contact(t_tensor, r0, rb, rb_dot, closure, create_graph=False)
    c_fields = type(c_fields)(
        radius=c_fields.radius.detach(), speed=c_fields.speed.detach(),
        sigmoid=c_fields.sigmoid.detach(), raw=c_fields.raw.detach(),
        pressure=c_fields.pressure.detach(),
        relative_volume=c_fields.relative_volume.detach(),
    )
    with torch.no_grad():
        air_fields = air(z_tensor, t_tensor, r0, scales)
    length = rb - c_fields.radius
    radius = c_fields.radius + z_tensor * length
    payload_out = {
        "format": "spherical_blast_pinn.contact_air_prediction",
        "version": 1,
        "case_id": case.case_id,
        "synthetic": bool(case.source.get("synthetic", False)),
        "checkpoint": str(args.checkpoint),
        "checkpoint_loop": payload.get("loop"),
        "times": times,
        "zeta": zetas,
        "radius": radius.detach().numpy().reshape(n_t, n_z),
        "contact_radius": c_fields.radius.detach().numpy().reshape(n_t, n_z),
        "contact_speed": c_fields.speed.detach().numpy().reshape(n_t, n_z),
        "rb": rb.detach().numpy().reshape(n_t, n_z),
        "rb_dot": rb_dot.detach().numpy().reshape(n_t, n_z),
        "density": air_fields.density.detach().numpy().reshape(n_t, n_z),
        "velocity": air_fields.velocity.detach().numpy().reshape(n_t, n_z),
        "pressure": air_fields.pressure.detach().numpy().reshape(n_t, n_z),
        "specific_energy": air_fields.specific_energy.detach().numpy().reshape(n_t, n_z),
        "note": (
            "prediction uses only the measured R_b trajectory, R_b dot, the case "
            "physics and the trained networks; no reference R_raw, contact interval "
            "or internal field was read"
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload_out)
    print(f"written: {out}")
    return 0


def command_evaluate_500(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    specs = _case_specs(section)
    wanted = args.case or "tnt_500mm"
    spec = next((item for item in specs if str(item.get("case_id")) == wanted), None)
    if spec is None:
        raise CommandError(f"no test case {wanted!r} in contact_air.cases")
    case = _make_case(section, spec, config, allow_synthetic=True)
    with np.load(args.prediction, allow_pickle=False) as archive:
        times = np.asarray(archive["times"])
        zetas = np.asarray(archive["zeta"])
        rho = np.asarray(archive["density"])
        u = np.asarray(archive["velocity"])
        p = np.asarray(archive["pressure"])
        rc = np.asarray(archive["contact_radius"])[:, 0]
        rb_pred = np.asarray(archive["rb"])[:, 0]
    if times.ndim != 1 or zetas.ndim != 1 or rho.shape != (times.size, zetas.size):
        raise CommandError("prediction arrays have unexpected shapes")
    interp = {
        "rho": RegularGridInterpolator((times, zetas), rho, bounds_error=False, fill_value=np.nan),
        "u": RegularGridInterpolator((times, zetas), u, bounds_error=False, fill_value=np.nan),
        "p": RegularGridInterpolator((times, zetas), p, bounds_error=False, fill_value=np.nan),
    }
    obs = case.observations
    valid_t = (obs.time >= times[0]) & (obs.time <= times[-1]) & obs.valid_domain
    rc_at_obs = np.interp(obs.time, times, rc)
    rb_at_obs = np.interp(obs.time, times, rb_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta_pred = (obs.radius - rc_at_obs) / (rb_at_obs - rc_at_obs)
    in_domain = valid_t & np.isfinite(zeta_pred) & (zeta_pred >= 0.0) & (zeta_pred <= 1.0)
    points = np.column_stack([obs.time, zeta_pred])
    predicted = {
        name: np.asarray(interp[name](points), dtype=np.float64)
        for name in ("rho", "u", "p")
    }
    metrics: dict[str, Any] = {"n_observations": int(obs.size), "n_in_domain": int(in_domain.sum())}
    for name, target, mask, scale in (
        ("rho", obs.density, obs.valid_density, case.rho_scale),
        ("u", obs.velocity, obs.valid_velocity, case.u_scale),
        ("p", obs.pressure, obs.valid_pressure, case.p_scale),
    ):
        use = in_domain & np.asarray(mask, bool) & np.isfinite(predicted[name]) & np.isfinite(target)
        if np.any(use):
            error = (predicted[name][use] - target[use]) / scale
            metrics[f"{name}_n"] = int(use.sum())
            metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(error * error)))
            metrics[f"{name}_max_abs"] = float(np.max(np.abs(error)))
        else:
            metrics[f"{name}_n"] = 0
            metrics[f"{name}_rmse"] = None
            metrics[f"{name}_max_abs"] = None
    valid_contact = case.contact.valid & (case.contact.times >= times[0]) & (
        case.contact.times <= times[-1]
    )
    contact_metrics: dict[str, Any] = {"n_contact_frames": int(valid_contact.sum())}
    if np.any(valid_contact):
        rc_contact = np.interp(
            case.contact.times[valid_contact], times, rc
        )
        below = np.maximum(case.contact.low[valid_contact] - rc_contact, 0.0)
        above = np.maximum(rc_contact - case.contact.up[valid_contact], 0.0)
        contact_metrics["mean_violation_over_Rsep"] = float(
            np.mean((below + above) / case.length_scale)
        )
        contact_metrics["covered_frames"] = int(np.count_nonzero(
            (below == 0.0) & (above == 0.0)
        ))
        contact_metrics["coverage"] = float(contact_metrics["covered_frames"] / valid_contact.sum())
    report = {
        "case_id": case.case_id,
        "synthetic": bool(case.source.get("synthetic", False)),
        "prediction": str(args.prediction),
        "air": metrics,
        "contact": contact_metrics,
        "note": (
            "evaluation uses the saved prediction first, then the 500 mm internal "
            "observations; the observation zeta is recomputed with predicted R_c/R_b, "
            "not with R_raw"
        ),
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, default=_jsonable))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("precheck")
    pre.add_argument("--config", required=True)
    pre.add_argument("--case", default="tnt_50mm")
    pre.add_argument("--all", action="store_true", help="print the 50/80/100 precursor table")
    pre.add_argument("--output", default=None)
    cache = sub.add_parser("build-cache")
    cache.add_argument("--config", required=True)
    cache.add_argument("--case", required=True)
    cache.add_argument("--profile-dir", required=True)
    cache.add_argument("--out", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--cases", default="three", choices=("50", "80", "100", "three"))
    train.add_argument("--max-loops", type=int, default=None)
    train.add_argument("--output-dir", default=None)
    train.add_argument("--resume", default=None)
    train.add_argument("--no-synthetic", action="store_true")
    pred = sub.add_parser("predict-500")
    pred.add_argument("--config", required=True)
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--case", default="tnt_500mm")
    pred.add_argument("--n-times", type=int, default=64)
    pred.add_argument("--n-zeta", type=int, default=64)
    ev = sub.add_parser("evaluate-500")
    ev.add_argument("--config", required=True)
    ev.add_argument("--prediction", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--case", default="tnt_500mm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "precheck":
            return command_precheck(args)
        if args.command == "build-cache":
            return command_build_cache(args)
        if args.command == "train":
            return command_train(args)
        if args.command == "predict-500":
            return command_predict_500(args)
        if args.command == "evaluate-500":
            return command_evaluate_500(args)
        raise CommandError(f"unknown command {args.command!r}")
    except (CommandError, CaseDataError, PrecursorError, TrainingError, ValueError) as exc:
        print(f"\n** {exc} **\n", file=sys.stderr)
        return 6


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
