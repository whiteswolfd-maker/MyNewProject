"""Command line for the one-dimensional ContactNet/AirNet scheme.

Commands (run from the repository root):

.. code-block:: text

    python -m spherical_blast_pinn.contact_air.cli precheck \
        --config configs/contact_air/base.yaml --all

    python -m spherical_blast_pinn.contact_air.cli build-cache \
        --config configs/contact_air/base.yaml --case tnt_50mm \
        --radius 0.05 --source <standard_1d_source.npz> --out <cache.npz>

    python -m spherical_blast_pinn.contact_air.cli train \
        --config configs/contact_air/smoke.yaml --cases all \
        --max-loops 4 --output-dir outputs/contact_air_smoke

    python -m spherical_blast_pinn.contact_air.cli predict \
        --config configs/contact_air/smoke.yaml --case tnt_50mm \
        --checkpoint <last.pt> --output <prediction.npz>

    python -m spherical_blast_pinn.contact_air.cli evaluate \
        --config configs/contact_air/smoke.yaml --case tnt_50mm \
        --prediction <prediction.npz> --output <evaluation.json>

No path is baked into the repository; cache and output paths come from the
configuration or the command line.  There is no fixed-radius training ban:
the data split is whatever the configured ``cases`` entries explicitly say.
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
    ContactLabelConfig,
    PredictionCaseData,
    build_case,
    load_case,
    load_prediction_case,
    save_case,
    synthetic_case,
)
from .geometry import GeometryError
from .losses import LossError, data_coordinates, masked_rmse
from .models import AirNet, ContactNet
from .one_d import OneDDataError, ShockDetectionConfig, read_standard_1d
from .precursor import PrecursorConfig, PrecursorError, compute_precursor
from .sampling import SamplingError
from .trainer import (
    AlternatingTrainer,
    TrainConfig,
    TrainingError,
    describe_matches,
    seed_everything,
)


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


def _precursor_payload(section: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(section["precursor"]))
    return payload


def _precursor_config(section: Mapping[str, Any], radius: float) -> PrecursorConfig:
    payload = _precursor_payload(section)
    payload["charge_radius_m"] = float(radius)
    return PrecursorConfig.from_mapping(payload)


def _contact_config(section: Mapping[str, Any]) -> ContactLabelConfig:
    return ContactLabelConfig.from_mapping(section.get("contact_interval"))


def _shock_config(section: Mapping[str, Any]) -> ShockDetectionConfig:
    return ShockDetectionConfig.from_mapping(section.get("shock_detection"))


def _case_specs(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = list(section.get("cases", []) or [])
    if not specs:
        raise CommandError(
            "contact_air.cases is empty. Define an explicit data split: which "
            "radii are train/validation/test is a research decision, not a "
            "default. The smoke configuration is the only pre-filled example."
        )
    return [dict(spec) for spec in specs]


def _select_specs(section: Mapping[str, Any], which: str) -> list[dict[str, Any]]:
    specs = _case_specs(section)
    if str(which).strip().lower() in ("all", "train"):
        selected = [spec for spec in specs if str(spec.get("role", "train")) == "train"]
    else:
        wanted = {part.strip() for part in str(which).split(",") if part.strip()}
        selected = [
            spec for spec in specs
            if str(spec.get("case_id")) in wanted
            and str(spec.get("role", "train")) == "train"
        ]
        missing = wanted - {str(spec.get("case_id")) for spec in selected}
        if missing:
            raise CommandError(
                f"no train case matches {sorted(missing)}; training only accepts "
                "cases explicitly configured with role: train"
            )
    if not selected:
        raise CommandError(
            f"no training case matches {which!r}; define role: train explicitly"
        )
    return selected


#: Physics constants that must match between training and prediction.  Radius-
#: dependent derived values (R_sep, t0, Rc0, Rs0) intentionally differ.
_SHARED_PRECURSOR_SCALARS = (
    "epsilon_denominator_factor", "shock_model", "air_density", "air_pressure",
    "gamma_air", "material_source", "material_source_verified",
)
_SHARED_DIMENSIONLESS = (
    "V_cj", "V_sep", "C", "p_sep", "u_c_sep", "D_s_sep",
)


def _value_mismatch(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) != bool(b)
    if isinstance(a, str) or isinstance(b, str):
        return str(a) != str(b)
    try:
        left, right = float(a), float(b)
    except (TypeError, ValueError):
        return a != b
    return not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-18)


def _compare_shared_precursor(
    cached: Mapping[str, Any], expected: Mapping[str, Any], *, label: str,
) -> None:
    problems: list[str] = []
    for key in _SHARED_PRECURSOR_SCALARS:
        if key not in cached or key not in expected:
            problems.append(f"{key}: missing")
        elif _value_mismatch(cached[key], expected[key]):
            problems.append(f"{key}: cache={cached[key]!r}, config={expected[key]!r}")
    for block in ("jwl", "detonation"):
        left = cached.get(block, {})
        right = expected.get(block, {})
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            problems.append(f"{block}: missing or malformed")
            continue
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                problems.append(f"{block}.{key}: missing")
            elif _value_mismatch(left[key], right[key]):
                problems.append(
                    f"{block}.{key}: cache={left[key]!r}, config={right[key]!r}"
                )
    if problems:
        raise CommandError(
            f"{label}: cached precursor does not match the current configuration "
            f"({', '.join(problems[:8])}). Re-run build-cache for this case."
        )


def _compare_dimensionless(cached: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    problems = [
        key for key in _SHARED_DIMENSIONLESS
        if key not in cached or key not in expected
        or _value_mismatch(cached[key], expected[key])
    ]
    if problems:
        raise CommandError(
            f"{label}: dimensionless precursor values differ ({problems}); the "
            "shared physics/card is not the same"
        )


def _compare_cached_precursor(
    cached: Mapping[str, Any], expected: Mapping[str, Any], *, label: str,
) -> None:
    _compare_shared_precursor(cached, expected, label=label)
    _compare_dimensionless(cached, expected, label=label)
    for key in ("t0", "R_sep", "Rc0", "Rs0"):
        if key not in cached or key not in expected or _value_mismatch(cached[key], expected[key]):
            raise CommandError(
                f"{label}: cached {key}={cached.get(key)!r} does not match the "
                f"configured {expected.get(key)!r}; re-run build-cache"
            )


def _cache_metadata(path: Path) -> Mapping[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise CommandError(f"{path}: cache has no metadata")
        return json.loads(str(archive["metadata"]))


def _cache_is_synthetic(path: Path) -> bool:
    try:
        metadata = _cache_metadata(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    case_meta = metadata.get("case", {})
    source = case_meta.get("source", {}) if isinstance(case_meta, Mapping) else {}
    return bool(isinstance(source, Mapping) and source.get("synthetic", False))


def _reject_synthetic_cache(
    case: CaseData | PredictionCaseData, spec: Mapping[str, Any], *, allow_synthetic: bool,
) -> None:
    if not bool(case.source.get("synthetic", False)):
        return
    if not bool(spec.get("synthetic", False)):
        raise CommandError(
            f"case {case.case_id!r}: a synthetic cache was loaded while the case "
            "is configured with synthetic: false; regenerate a real cache or set "
            "synthetic: true deliberately"
        )
    if not allow_synthetic:
        raise CommandError(
            f"case {case.case_id!r}: --no-synthetic forbids loading a synthetic cache"
        )


def _check_cached_case(
    case: CaseData | PredictionCaseData, section: Mapping[str, Any], *,
    case_id: str, radius: float, label: str,
) -> None:
    if str(case.case_id) != str(case_id):
        raise CommandError(
            f"{label}: cache case_id={case.case_id!r}, config says {case_id!r}"
        )
    if abs(float(case.charge_radius_m) - float(radius)) > 1.0e-12 * max(radius, 1.0):
        raise CommandError(
            f"{label}: cache R0={case.charge_radius_m!r}, config says {radius!r}"
        )
    expected = compute_precursor(_precursor_config(section, radius)).as_dict()
    _compare_cached_precursor(case.precursor, expected, label=label)


def _make_case(
    section: Mapping[str, Any], spec: Mapping[str, Any], config: Config, *,
    allow_synthetic: bool,
) -> CaseData:
    radius = float(spec["charge_radius_m"])
    case_id = str(spec["case_id"])
    role = str(spec.get("role", "train"))
    cached = spec.get("cache")
    if cached:
        path = _resolve(config, cached)
        if not path.is_file():
            raise CommandError(f"case {case_id!r}: cache {path} does not exist")
        if (not bool(spec.get("synthetic", False)) or not allow_synthetic) and _cache_is_synthetic(path):
            raise CommandError(
                f"case {case_id!r}: cache {path} is synthetic, but synthetic:false "
                "or --no-synthetic forbids loading it"
            )
        case = load_case(path, role=role)
        _reject_synthetic_cache(case, spec, allow_synthetic=allow_synthetic)
        _check_cached_case(
            case, section, case_id=case_id, radius=radius,
            label=f"case {case_id!r} cache {path}",
        )
        return case
    if not bool(spec.get("synthetic", False)) or not allow_synthetic:
        raise CommandError(
            f"case {case_id!r}: no cache is configured and synthetic data is not "
            "allowed. Run build-cache on a standard 1-D source first."
        )
    precursor = compute_precursor(_precursor_config(section, radius))
    return synthetic_case(
        case_id=case_id, charge_radius_m=radius, precursor=precursor, role=role,
        n_frames=int(spec.get("n_frames", 21)),
        n_radial=int(spec.get("n_radial", 2400)),
        duration_s=spec.get("duration_s"),
        tolerances=dict(section.get("tolerances", {})),
        contact_config=_contact_config(section),
        shock_config=_shock_config(section),
    )


def _make_prediction_case(
    section: Mapping[str, Any], spec: Mapping[str, Any], config: Config, *,
    allow_synthetic: bool,
) -> PredictionCaseData:
    radius = float(spec["charge_radius_m"])
    case_id = str(spec["case_id"])
    role = str(spec.get("role", "test"))
    cached = spec.get("cache")
    if cached:
        path = _resolve(config, cached)
        if not path.is_file():
            raise CommandError(f"case {case_id!r}: cache {path} does not exist")
        if (not bool(spec.get("synthetic", False)) or not allow_synthetic) and _cache_is_synthetic(path):
            raise CommandError(
                f"case {case_id!r}: prediction cache {path} is synthetic, but "
                "synthetic:false or --no-synthetic forbids loading it"
            )
        case = load_prediction_case(path, role=role)
        _reject_synthetic_cache(case, spec, allow_synthetic=allow_synthetic)
        _check_cached_case(
            case, section, case_id=case_id, radius=radius,
            label=f"case {case_id!r} prediction cache {path}",
        )
        return case
    if not bool(spec.get("synthetic", False)) or not allow_synthetic:
        raise CommandError(
            f"case {case_id!r}: no cache is configured and synthetic data is not allowed."
        )
    precursor = compute_precursor(_precursor_config(section, radius))
    case = synthetic_case(
        case_id=case_id, charge_radius_m=radius, precursor=precursor, role=role,
        n_frames=int(spec.get("n_frames", 21)),
        n_radial=int(spec.get("n_radial", 2400)),
        duration_s=spec.get("duration_s"),
        tolerances=dict(section.get("tolerances", {})),
        contact_config=_contact_config(section),
        shock_config=_shock_config(section),
    )
    return case.prediction_view()


def _networks(section: Mapping[str, Any]):
    if torch is None:  # pragma: no cover
        raise CommandError("PyTorch is required")
    net_section = section.get("networks", {})
    dtype_name = str(net_section.get("dtype", "float64"))
    dtype = getattr(torch, dtype_name, torch.float64)
    contact_section = net_section.get("contact", {})
    air_section = net_section.get("air", {})
    time_reference_s = float(net_section.get("time_reference_s", 1.0))
    contact = ContactNet(
        width=int(contact_section.get("width", 50)),
        depth=int(contact_section.get("depth", 5)),
        dtype=dtype, device="cpu",
        time_reference_s=time_reference_s,
    )
    air = AirNet(
        width=int(air_section.get("width", 32)),
        depth=int(air_section.get("depth", 10)),
        dtype=dtype, device="cpu",
        time_reference_s=time_reference_s,
    )
    return contact, air


def _train_config(
    section: Mapping[str, Any], loops: int | None,
    contact_data_weight: float | None = None,
) -> TrainConfig:
    training = dict(section.get("training", {}))
    sampling = dict(section.get("sampling", {}))
    values = {
        "n_pde": int(sampling.get("n_pde", 512)),
        "n_ic": int(sampling.get("n_ic", 128)),
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
        "contact_data_weight": float(
            training.get("contact_data_weight", 1.0)
        ),
    }
    if loops is not None:
        values["loops"] = int(loops)
    if contact_data_weight is not None:
        values["contact_data_weight"] = float(contact_data_weight)
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
    raw_specs = list(section.get("cases", []) or [])
    if not raw_specs:
        if args.case is not None:
            raise CommandError(
                f"contact_air.cases is empty; no case {args.case!r} is configured"
            )
        payload = {
            "status": "no_cases_configured",
            "n_cases": 0,
            "rows": [],
            "missing": [
                "no explicit train/validation/test split is configured",
                "a standard 1-D source (or a cache built from one) is required "
                "before any geometry, shock trajectory or contact label exists",
                "no real 50 mm ALE1D data is present; the binary reader is not "
                "implemented and must not be guessed",
            ],
            "note": (
                "base.yaml deliberately has cases: []. Define the split in a run "
                "configuration; no radius is a fixed default."
            ),
        }
        if args.output:
            _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, default=_jsonable))
        return 0
    specs = [dict(spec) for spec in raw_specs]
    wanted = args.case
    selected = [
        spec for spec in specs
        if wanted is None or str(spec.get("case_id")) == wanted
    ]
    if not selected:
        raise CommandError(f"no case {wanted!r} in contact_air.cases")
    rows = []
    for spec in selected:
        radius = float(spec["charge_radius_m"])
        result = compute_precursor(_precursor_config(section, radius))
        row = {
            "case_id": spec.get("case_id"),
            "role": spec.get("role", "train"),
            "charge_radius_m": radius,
            "precursor": result.as_dict(),
            "precursor_checks": dict(result.checks),
            "initial_condition_verified": False,
            "status": "precursor_ready_ic_unverified",
            "missing": [
                "a standard 1-D source or cache is required for R_s(t_event) and "
                "the contact labels; the precursor alone does not define them"
            ],
        }
        cached = spec.get("cache")
        if cached:
            path = _resolve(config, cached)
            if path.is_file():
                case = load_case(path, role=str(spec.get("role", "train")))
                row.update({
                    "cache": str(path),
                    "t0": case.t0,
                    "t_end": case.t_end,
                    "R_s_at_t0": case.initial.Rs_data_at_t0,
                    "initial": case.as_dict()["initial"],
                    "shock_continuous_intervals": case.shock.continuous_intervals,
                    "event_alignment": case.event_alignment,
                    "observations": case.observations.size,
                    "initial_compatibility": case.initial_compatibility(),
                    "separation": case.separation,
                    "status": "ic_geometry_ready_physics_unverified",
                    "initial_condition_verified": False,
                    "missing": [],
                })
            else:
                row["missing"] = [f"cache {path} does not exist"]
        rows.append(row)
    payload = {
        "status": "precursor_table",
        "n_cases": len(rows),
        "rows": rows,
        "note": (
            "role assignments are exactly those in the configuration; no radius "
            "is reserved as a fixed blind test"
        ),
    }
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, default=_jsonable))
    return 0


def command_build_cache(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    spec = next(
        (item for item in list(section.get("cases", []) or [])
         if str(item.get("case_id")) == args.case),
        None,
    )
    radius = args.radius if args.radius is not None else (
        float(spec["charge_radius_m"]) if spec is not None else None
    )
    if radius is None:
        raise CommandError(
            f"case {args.case!r} is not in contact_air.cases; pass --radius to "
            "build its cache"
        )
    role = args.role or (str(spec.get("role", "train")) if spec is not None else "train")
    try:
        source = read_standard_1d(args.source)
    except OneDDataError as exc:
        raise CommandError(str(exc)) from exc
    if source.case_id != args.case:
        raise CommandError(
            f"source case_id={source.case_id!r} does not match --case {args.case!r}"
        )
    if abs(source.charge_radius_m - float(radius)) > 1.0e-12:
        raise CommandError(
            f"source charge_radius_m={source.charge_radius_m!r} does not match "
            f"the configured {radius!r}"
        )
    precursor = compute_precursor(_precursor_config(section, float(radius)))
    case = build_case(
        source, precursor=precursor, role=role,
        tolerances=dict(section.get("tolerances", {})),
        contact_config=_contact_config(section),
        shock_config=_shock_config(section),
        source_meta={"case_id": args.case, "source_path": str(args.source)},
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
    # Explicit role: validation cases are scored every validate_every loops and
    # can select best.pt, but they are never sampled for training.
    validation_specs = [
        spec for spec in _case_specs(section)
        if str(spec.get("role", "train")) == "validation"
    ]
    validation_cases = [
        _make_case(section, spec, config, allow_synthetic=not args.no_synthetic)
        for spec in validation_specs
    ]
    # Seed before constructing the networks: torch's default generator draws
    # the initial weights, so seeding only inside the trainer would leave the
    # initial parameters different between otherwise identical runs.
    seed = int(section.get("training", {}).get("seed", 0))
    seed_everything(seed)
    contact, air = _networks(section)
    train_config = _train_config(
        section, args.max_loops,
        contact_data_weight=args.contact_data_weight,
    )
    output_dir = Path(args.output_dir) if args.output_dir else _resolve(
        config, section.get("output_dir", "outputs/contact_air")
    )
    trainer = AlternatingTrainer(
        cases=cases, contact_net=contact, air_net=air,
        config=train_config, output_dir=output_dir,
        external_validation_cases=validation_cases,
    )
    if args.resume:
        info = trainer.load(args.resume)
        print(f"resumed from {args.resume} at loop {info['loop']}")
    if args.learning_rate is not None:
        rate = float(args.learning_rate)
        if rate <= 0.0:
            raise CommandError("--learning-rate must be positive")
        # Applied after a resume because load() restores the Adam state,
        # including the learning rate stored in it.
        for optimizer in (trainer.opt_contact, trainer.opt_air):
            for group in optimizer.param_groups:
                group["lr"] = rate
        print(f"learning rate overridden to {rate:.3e} for both optimizers")
    result = trainer.train()
    _write_json(output_dir / "train_result.json", result)
    print(json.dumps({
        "loops": result["loops"],
        "last_validation": result["last_validation"],
        "output_dir": str(output_dir),
    }, indent=2, default=_jsonable))
    return 0


def _check_checkpoint_state(module, state, name: str, path: Path) -> None:
    current = module.state_dict()
    if set(state) != set(current):
        raise CommandError(f"{path}: {name} keys do not match the configured architecture")
    for key, value in current.items():
        saved = state[key]
        if tuple(saved.shape) != tuple(value.shape) or saved.dtype != value.dtype:
            raise CommandError(
                f"{path}: {name} architecture mismatch at {key!r}: "
                f"{tuple(saved.shape)} vs {tuple(value.shape)}"
            )


def _load_checkpoint(path: Path, contact: ContactNet, air: AirNet) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spherical_blast_pinn.contact_air_checkpoint":
        raise CommandError(f"{path}: not a contact-air checkpoint")
    if payload.get("version") != 4:
        raise CommandError(
            f"{path}: checkpoint version {payload.get('version')!r} is not supported; "
            "retrain: version 4 carries the R_s trajectory contract and the "
            "two-optimizer stage state"
        )
    _check_checkpoint_state(contact, payload["contact_net"], "contact_net", Path(path))
    _check_checkpoint_state(air, payload["air_net"], "air_net", Path(path))
    for key, module in (("contact_net", contact), ("air_net", air)):
        if not describe_matches(payload.get(f"{key}_describe"), module.describe()):
            raise CommandError(
                f"{path}: {key} model convention/architecture metadata does not "
                "match the configured network"
            )
    contact.load_state_dict(payload["contact_net"])
    air.load_state_dict(payload["air_net"])
    contact.eval()
    air.eval()
    return payload


def _check_prediction_compatibility(
    case: PredictionCaseData, payload: Mapping[str, Any],
    contact: ContactNet, air: AirNet,
) -> None:
    metadata = list(payload.get("case_metadata", []) or [])
    references = [
        (str(item.get("case_id", "<unknown>")), item.get("precursor"))
        for item in metadata
        if isinstance(item, Mapping) and isinstance(item.get("precursor"), Mapping)
    ]
    if not references:
        raise CommandError(
            "checkpoint has no training-case precursor metadata; retrain with the "
            "current version before predicting"
        )
    for case_id, reference in references:
        _compare_shared_precursor(
            case.precursor, reference,
            label=f"prediction physics vs training case {case_id!r}",
        )
        _compare_dimensionless(
            case.precursor, reference,
            label=f"prediction dimensionless physics vs training case {case_id!r}",
        )
    for key, module in (("contact_net", contact), ("air_net", air)):
        if not describe_matches(payload.get(f"{key}_describe"), module.describe()):
            raise CommandError(
                f"checkpoint {key} model convention does not match the prediction model"
            )


def command_predict(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    specs = _case_specs(section)
    spec = next((item for item in specs if str(item.get("case_id")) == args.case), None)
    if spec is None:
        raise CommandError(f"no case {args.case!r} in contact_air.cases")
    case = _make_prediction_case(section, spec, config, allow_synthetic=True)
    contact, air = _networks(section)
    payload = _load_checkpoint(Path(args.checkpoint), contact, air)
    _check_prediction_compatibility(case, payload, contact, air)
    n_t = int(args.n_times)
    n_z = int(args.n_zeta)
    times = np.linspace(case.t0, case.t_end, n_t)
    zetas = np.linspace(0.0, 1.0, n_z)
    t_grid, z_grid = np.meshgrid(times, zetas, indexing="ij")
    t_tensor = torch.as_tensor(t_grid.reshape(-1, 1), dtype=torch.float64)
    z_tensor = torch.as_tensor(z_grid.reshape(-1, 1), dtype=torch.float64)
    r0 = torch.full_like(t_tensor, float(case.charge_radius_m))
    rs, ds = case.shock_tensors(t_grid.reshape(-1))
    rs = rs.reshape(-1, 1).to(torch.float64)
    ds = ds.reshape(-1, 1).to(torch.float64)
    from .models import closure_from_precursor, scales_from_precursor
    closure = closure_from_precursor(case.precursor, like=t_tensor)
    scales = scales_from_precursor(case.precursor, like=t_tensor)
    with torch.enable_grad():
        c_fields = contact(t_tensor, r0, rs, ds, closure, create_graph=False)
    c_fields = type(c_fields)(
        radius=c_fields.radius.detach(), speed=c_fields.speed.detach(),
        sigmoid=c_fields.sigmoid.detach(), raw=c_fields.raw.detach(),
        pressure=c_fields.pressure.detach(),
        relative_volume=c_fields.relative_volume.detach(),
    )
    with torch.no_grad():
        air_fields = air(z_tensor, t_tensor, r0, scales)
    length = rs - c_fields.radius
    radius = c_fields.radius + z_tensor * length
    payload_out = {
        "format": "spherical_blast_pinn.contact_air_prediction",
        "version": 2,
        "case_id": case.case_id,
        "synthetic": bool(case.source.get("synthetic", False)),
        "checkpoint": str(args.checkpoint),
        "checkpoint_loop": payload.get("loop"),
        "times": times,
        "zeta": zetas,
        "radius": radius.detach().numpy().reshape(n_t, n_z),
        "contact_radius": c_fields.radius.detach().numpy().reshape(n_t, n_z),
        "contact_speed": c_fields.speed.detach().numpy().reshape(n_t, n_z),
        "shock_radius": rs.detach().numpy().reshape(n_t, n_z),
        "shock_speed": ds.detach().numpy().reshape(n_t, n_z),
        "density": air_fields.density.detach().numpy().reshape(n_t, n_z),
        "velocity": air_fields.velocity.detach().numpy().reshape(n_t, n_z),
        "pressure": air_fields.pressure.detach().numpy().reshape(n_t, n_z),
        "specific_energy": air_fields.specific_energy.detach().numpy().reshape(n_t, n_z),
        "prediction_data_scope": np.asarray(json.dumps({
            "loaded_fields": ["metadata", "shock_times", "shock_radius", "shock_valid"],
            "not_loaded": ["internal observations", "contact interval"],
            "time_range_source": "continuous R_s segment containing t0",
            "internal_supervision_used_for_time_range": False,
        })),
        "note": (
            "prediction uses the identified external shock trajectory, dR_s/dt, "
            "the case physics and the trained networks; it is 'prediction given "
            "the measured shock trajectory', not autonomous cross-radius shock "
            "prediction"
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload_out)
    print(f"written: {out}")
    return 0


def command_evaluate(args) -> int:
    config = load_config(args.config)
    section = _section(config)
    specs = _case_specs(section)
    spec = next((item for item in specs if str(item.get("case_id")) == args.case), None)
    if spec is None:
        raise CommandError(f"no case {args.case!r} in contact_air.cases")
    case = _make_case(section, spec, config, allow_synthetic=True)
    with np.load(args.prediction, allow_pickle=False) as archive:
        prediction_case_id = str(archive["case_id"]) if "case_id" in archive.files else None
        times = np.asarray(archive["times"])
        zetas = np.asarray(archive["zeta"])
        rho = np.asarray(archive["density"])
        u = np.asarray(archive["velocity"])
        p = np.asarray(archive["pressure"])
        rc = np.asarray(archive["contact_radius"])[:, 0]
    if prediction_case_id is not None and prediction_case_id != case.case_id:
        raise CommandError(
            f"prediction is for case {prediction_case_id!r}, evaluation case is "
            f"{case.case_id!r}"
        )
    if times.ndim != 1 or zetas.ndim != 1 or rho.shape != (times.size, zetas.size):
        raise CommandError("prediction arrays have unexpected shapes")
    case.shock.require_interval(float(times[0]), float(times[-1]))
    interp = {
        "rho": RegularGridInterpolator((times, zetas), rho, bounds_error=False, fill_value=np.nan),
        "u": RegularGridInterpolator((times, zetas), u, bounds_error=False, fill_value=np.nan),
        "p": RegularGridInterpolator((times, zetas), p, bounds_error=False, fill_value=np.nan),
    }
    obs = case.observations
    valid_t = (obs.time >= times[0]) & (obs.time <= times[-1])
    rc_at_obs = np.interp(obs.time, times, rc)
    rs_at_obs = np.asarray(case.shock.position(obs.time), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta_pred = (obs.radius - rc_at_obs) / (rs_at_obs - rc_at_obs)
    in_domain = valid_t & np.isfinite(zeta_pred) & (zeta_pred >= 0.0) & (zeta_pred <= 1.0)
    points = np.column_stack([obs.time, np.clip(zeta_pred, 0.0, 1.0)])
    predicted = {
        name: np.asarray(interp[name](points), dtype=np.float64)
        for name in ("rho", "u", "p")
    }
    metrics: dict[str, Any] = {
        "n_observations": int(obs.size), "n_in_domain": int(in_domain.sum()),
        "n_out_of_domain": int(obs.size - in_domain.sum()),
    }
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
        rc_contact = np.interp(case.contact.times[valid_contact], times, rc)
        below = np.maximum(case.contact.low[valid_contact] - rc_contact, 0.0)
        above = np.maximum(rc_contact - case.contact.up[valid_contact], 0.0)
        contact_metrics["mean_violation_over_Rsep"] = float(
            np.mean((below + above) / case.length_scale)
        )
        contact_metrics["covered_frames"] = int(np.count_nonzero(
            (below == 0.0) & (above == 0.0)
        ))
        contact_metrics["coverage"] = float(
            contact_metrics["covered_frames"] / valid_contact.sum()
        )
    report = {
        "case_id": case.case_id,
        "synthetic": bool(case.source.get("synthetic", False)),
        "prediction": str(args.prediction),
        "air": metrics,
        "contact": contact_metrics,
        "evaluation_data_scope": {
            "loaded_fields": ["saved prediction", "internal observations", "contact interval"],
            "note": "this command, not predict, is allowed to read internal reference data",
        },
        "note": (
            "the observation coordinate is recomputed with the predicted R_c and "
            "the measured shock R_s, not with the contact-interval midpoint"
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
    pre.add_argument("--case", default=None)
    pre.add_argument("--all", action="store_true", help="print every configured case")
    pre.add_argument("--output", default=None)
    cache = sub.add_parser("build-cache")
    cache.add_argument("--config", required=True)
    cache.add_argument("--case", required=True)
    cache.add_argument("--radius", type=float, default=None)
    cache.add_argument("--role", choices=("train", "validation", "test"), default=None)
    cache.add_argument("--source", required=True, help="standard 1-D source (.npz or directory)")
    cache.add_argument("--out", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--cases", default="all", help="'all' or comma-separated train case ids")
    train.add_argument("--max-loops", type=int, default=None)
    train.add_argument("--output-dir", default=None)
    train.add_argument("--resume", default=None)
    train.add_argument(
        "--learning-rate", type=float, default=None,
        help=(
            "override the configured learning rate for both optimizers.  "
            "Applied after --resume restores the Adam state, which otherwise "
            "brings back the old learning rate"
        ),
    )
    train.add_argument(
        "--contact-data-weight", type=float, default=None,
        help=(
            "multiply the contact-interval supervision term (lambda_data * "
            "contact_data) by this factor; every other loss weight is unchanged"
        ),
    )
    train.add_argument("--no-synthetic", action="store_true")
    pred = sub.add_parser("predict")
    pred.add_argument("--config", required=True)
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--case", required=True)
    pred.add_argument("--n-times", type=int, default=64)
    pred.add_argument("--n-zeta", type=int, default=64)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--config", required=True)
    ev.add_argument("--prediction", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--case", required=True)
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
        if args.command == "predict":
            return command_predict(args)
        if args.command == "evaluate":
            return command_evaluate(args)
        raise CommandError(f"unknown command {args.command!r}")
    except (
        CommandError, CaseDataError, PrecursorError, TrainingError, LossError,
        SamplingError, GeometryError, OneDDataError, ValueError,
    ) as exc:
        print(f"\n** {exc} **\n", file=sys.stderr)
        return 6


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
