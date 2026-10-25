"""Versioned save/load round trip for the planar-startup handoff.

The handoff is two files:

* ``handoff.npz`` — the physical cell/node arrays plus the scalar wave and
  contact data;
* ``handoff.json`` — the report, the EOS/initial parameters, the Riemann
  solver settings, and the metadata needed to reproduce the analytic sampler.

The loader is deliberately separate from the old ``ReferenceField`` cache
reader.  It reconstructs the local planar Riemann solution from the saved
initial state and solver settings, checks that reconstruction against every
saved solution scalar/table summary, and only then builds a
:class:`~spherical_blast_pinn.physics.two_medium_startup_planar.PlanarStartupField`.
The field constructor then replays the normal consistency validation against
the saved arrays, and the loader recomputes the mass/energy checks and
integrals from those arrays, the analytic sampler and the initial parameters.
Saved ``checks``, ``integrals`` and their top-level copies must match the
recomputation; missing, non-finite or inconsistent diagnostics are rejected.
A mismatch is rejected; reconstructed values are never silently substituted
for conflicting saved content.

Nothing here starts training, writes a ``ReferenceField`` cache, or changes
the startup physics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .eos import EOSError, JWLParameters
from .two_medium_riemann import (
    RIEMANN_METHOD_VERSION,
    RiemannError,
    solve_jwl_air_riemann,
)
from .two_medium_startup_planar import (
    PLANAR_STARTUP_METHOD_VERSION,
    PLANAR_STARTUP_MODEL_ID,
    PlanarStartupError,
    PlanarStartupField,
    compute_planar_startup_diagnostics,
)

#: Serialization schema, independent of the physics method version.
PLANAR_STARTUP_HANDOFF_FORMAT = "planar_startup_handoff"
PLANAR_STARTUP_HANDOFF_FORMAT_VERSION = 1

HANDOFF_JSON_NAME = "handoff.json"
HANDOFF_NPZ_NAME = "handoff.npz"

_JSON_REL_TOL = 1.0e-12
_JSON_ABS_TOL = 1.0e-15
_SCALAR_MATCH_REL_TOL = 1.0e-14
_SCALAR_MATCH_ABS_TOL = 0.0

#: Saved diagnostics are recomputed from the same arrays and sampler.  A new
#: legal file is therefore bitwise equal after the JSON round trip; the only
#: known legal difference is the last-bit change from the initial-field mass
#: sums to the closed-form sphere integrals (about 2e-16 relative).  This
#: tolerance is far below any meaningful change in a reported mass/energy
#: residual, while absorbing that roundoff and cross-platform libm differences.
_DIAGNOSTIC_REL_TOL = 1.0e-11
_DIAGNOSTIC_ABS_TOL = 1.0e-14

#: Top-level copies of the diagnostics, checked against the independently
#: recomputed ``checks``/``integrals`` rather than trusted from the report.
_TOP_LEVEL_DIAGNOSTICS = (
    ("spherical_energy_residual", "checks", "spherical_energy_residual"),
    ("initial_product_internal_energy_per_sr", "checks", "initial_product_internal_energy_per_sr"),
    ("spherical_energy_change_signed_j_per_sr", "checks", "spherical_energy_change_signed_j_per_sr"),
    ("spherical_energy_change_abs_j_per_sr", "checks", "spherical_energy_change_abs_j_per_sr"),
    (
        "spherical_energy_change_abs_over_initial_product_internal",
        "checks",
        "spherical_energy_change_abs_over_initial_product_internal",
    ),
    ("grid_integrals", "integrals", "grid"),
    ("dense_integrals", "integrals", "dense"),
)

_REQUIRED_REPORT_KEYS = (
    "handoff_format",
    "handoff_format_version",
    "model_id",
    "method_version",
    "riemann_method_version",
    "physical_time_s",
    "pin_time_s",
    "charge_radius_m",
    "epsilon_s",
    "delta",
    "contact_radius_m",
    "shock_radius_m",
    "contact_speed_m_s",
    "shock_speed_m_s",
    "rarefaction_head_radius_m",
    "rarefaction_tail_radius_m",
    "rarefaction_head_speed_m_s",
    "rarefaction_tail_speed_m_s",
    "spherical_energy_residual",
    "initial_product_internal_energy_per_sr",
    "spherical_energy_change_signed_j_per_sr",
    "spherical_energy_change_abs_j_per_sr",
    "spherical_energy_change_abs_over_initial_product_internal",
    "grid_integrals",
    "dense_integrals",
    "cache_key",
    "product_eos",
    "initial_state",
    "air_gamma",
    "riemann",
    "checks",
    "integrals",
    "metadata",
    "units",
    "reference_field",
    "training_qualified",
)

_ARRAY_KEYS = (
    "radii",
    "cell_volume",
    "density",
    "velocity",
    "specific_energy",
    "pressure",
    "sound_speed",
    "material",
    "region",
    "mass",
    "node_velocity",
    "epsilon",
    "delta",
    "contact_radius",
    "shock_radius",
    "contact_speed",
    "shock_speed",
)
_REQUIRED_NPZ_KEYS = _ARRAY_KEYS + ("handoff_format_version",)

#: JSON report scalar -> scalar array in ``handoff.npz``.
_SCALAR_CROSS_CHECKS = (
    ("epsilon_s", "epsilon"),
    ("delta", "delta"),
    ("contact_radius_m", "contact_radius"),
    ("shock_radius_m", "shock_radius"),
    ("contact_speed_m_s", "contact_speed"),
    ("shock_speed_m_s", "shock_speed"),
)

_INITIAL_STATE_KEYS = (
    "charge_radius",
    "outer_radius",
    "product_density",
    "product_specific_energy",
    "air_density",
    "air_specific_energy",
    "air_pressure",
    "air_gamma",
    "product_pressure",
    "product_sound_speed",
    "air_sound_speed",
)


class PlanarStartupHandoffError(PlanarStartupError):
    """The saved planar-startup handoff is missing, inconsistent or unsupported."""


def _required_mapping(container: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    if key not in container:
        raise PlanarStartupHandoffError(f"{where} is missing required field {key!r}")
    value = container[key]
    if not isinstance(value, Mapping):
        raise PlanarStartupHandoffError(f"{where}.{key} must be a JSON object")
    return value


def _required_string(container: Mapping[str, Any], key: str, where: str) -> str:
    if key not in container:
        raise PlanarStartupHandoffError(f"{where} is missing required field {key!r}")
    value = container[key]
    if not isinstance(value, str) or not value:
        raise PlanarStartupHandoffError(f"{where}.{key} must be a non-empty string")
    return value


def _required_number(container: Mapping[str, Any], key: str, where: str) -> float:
    if key not in container:
        raise PlanarStartupHandoffError(f"{where} is missing required field {key!r}")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanarStartupHandoffError(f"{where}.{key} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise PlanarStartupHandoffError(f"{where}.{key} must be finite")
    return number


def _required_integer(container: Mapping[str, Any], key: str, where: str) -> int:
    if key not in container:
        raise PlanarStartupHandoffError(f"{where} is missing required field {key!r}")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanarStartupHandoffError(f"{where}.{key} must be an integer")
    return int(value)


def _array_scalar(payload: Mapping[str, np.ndarray], key: str) -> float:
    array = np.asarray(payload[key])
    if not np.issubdtype(array.dtype, np.number):
        raise PlanarStartupHandoffError(f"handoff.npz array {key!r} must be numeric")
    if array.size != 1:
        raise PlanarStartupHandoffError(
            f"handoff.npz array {key!r} must be a scalar, got shape {array.shape}"
        )
    number = float(array.reshape(-1)[0])
    if not math.isfinite(number):
        raise PlanarStartupHandoffError(f"handoff.npz array {key!r} must be finite")
    return number


def _array_integer(payload: Mapping[str, np.ndarray], key: str) -> int:
    number = _array_scalar(payload, key)
    integer = int(number)
    if float(integer) != number:
        raise PlanarStartupHandoffError(f"handoff.npz array {key!r} must be an integer")
    return integer


def _require_close_scalar(
    name: str,
    actual: float,
    expected: float,
    *,
    rel_tol: float,
    abs_tol: float,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise PlanarStartupHandoffError(
            f"{name} is inconsistent: got {actual!r}, expected {expected!r}"
        )


def _require_tree_close(name: str, expected: Any, actual: Any) -> None:
    """Require a saved JSON tree to match the independently reconstructed tree."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise PlanarStartupHandoffError(f"{name} must be a JSON object")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise PlanarStartupHandoffError(
                f"{name} fields do not match the reconstructed solution: "
                f"missing={missing}, unexpected={extra}"
            )
        for key in sorted(expected_keys):
            _require_tree_close(f"{name}.{key}", expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            raise PlanarStartupHandoffError(f"{name} must be a JSON array")
        if len(expected) != len(actual):
            raise PlanarStartupHandoffError(
                f"{name} length is {len(actual)}, expected {len(expected)}"
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            _require_tree_close(f"{name}[{index}]", left, right)
        return
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual is not expected:
            raise PlanarStartupHandoffError(
                f"{name} is {actual!r}, expected {expected!r}"
            )
        return
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise PlanarStartupHandoffError(f"{name} must be a real number")
        value = float(actual)
        if not math.isfinite(value):
            raise PlanarStartupHandoffError(f"{name} must be finite")
        _require_close_scalar(
            name,
            value,
            float(expected),
            rel_tol=_JSON_REL_TOL,
            abs_tol=_JSON_ABS_TOL,
        )
        return
    if isinstance(expected, str):
        if actual != expected:
            raise PlanarStartupHandoffError(
                f"{name} is {actual!r}, expected {expected!r}"
            )
        return
    if expected is None:
        if actual is not None:
            raise PlanarStartupHandoffError(f"{name} must be null")
        return
    raise PlanarStartupHandoffError(  # pragma: no cover - internal schema error
        f"{name} has unsupported saved type {type(expected).__name__}"
    )


def _require_diagnostics_match(name: str, saved: Any, recomputed: Any) -> None:
    """Compare saved diagnostics with an independent recomputation.

    Required fields must exist.  Numeric fields are finite and equal within
    ``_DIAGNOSTIC_REL_TOL``/``_DIAGNOSTIC_ABS_TOL``; booleans and counts are
    exact.  Descriptive strings must be present and have the right type, but
    their wording is not a numerical acceptance condition.
    """
    if isinstance(recomputed, Mapping):
        if not isinstance(saved, Mapping):
            raise PlanarStartupHandoffError(f"{name} must be a JSON object")
        unexpected = sorted(set(saved) - set(recomputed))
        if unexpected:
            raise PlanarStartupHandoffError(
                f"{name} has unexpected diagnostic field(s): {unexpected}"
            )
        for key in sorted(recomputed):
            path = f"{name}.{key}"
            if key not in saved:
                raise PlanarStartupHandoffError(
                    f"{name} is missing required diagnostic field {key!r}"
                )
            _require_diagnostics_match(path, saved[key], recomputed[key])
        return
    if isinstance(recomputed, bool):
        if not isinstance(saved, bool) or saved is not recomputed:
            raise PlanarStartupHandoffError(
                f"{name} is inconsistent: got {saved!r}, recomputed {recomputed!r}"
            )
        return
    if isinstance(recomputed, (int, np.integer)):
        if isinstance(saved, bool) or not isinstance(saved, int):
            raise PlanarStartupHandoffError(f"{name} must be an integer")
        if int(saved) != int(recomputed):
            raise PlanarStartupHandoffError(
                f"{name} is inconsistent: got {saved!r}, recomputed {int(recomputed)!r}"
            )
        return
    if isinstance(recomputed, (float, np.floating)):
        expected = float(recomputed)
        if not math.isfinite(expected):
            raise PlanarStartupHandoffError(
                f"{name} cannot be certified: recomputed value is not finite"
            )
        if isinstance(saved, bool) or not isinstance(saved, (int, float)):
            raise PlanarStartupHandoffError(f"{name} must be a real number")
        actual = float(saved)
        if not math.isfinite(actual):
            raise PlanarStartupHandoffError(f"{name} must be finite")
        if not math.isclose(
            actual,
            expected,
            rel_tol=_DIAGNOSTIC_REL_TOL,
            abs_tol=_DIAGNOSTIC_ABS_TOL,
        ):
            raise PlanarStartupHandoffError(
                f"{name} is inconsistent: got {actual!r}, recomputed {expected!r}"
            )
        return
    if isinstance(recomputed, str):
        if not isinstance(saved, str):
            raise PlanarStartupHandoffError(f"{name} must be a string")
        return
    if isinstance(recomputed, (list, tuple)):
        if not isinstance(saved, (list, tuple)):
            raise PlanarStartupHandoffError(f"{name} must be a JSON array")
        if len(saved) != len(recomputed):
            raise PlanarStartupHandoffError(
                f"{name} length is {len(saved)}, recomputed {len(recomputed)}"
            )
        for index, (left, right) in enumerate(zip(saved, recomputed)):
            _require_diagnostics_match(f"{name}[{index}]", left, right)
        return
    if recomputed is None:
        if saved is not None:
            raise PlanarStartupHandoffError(f"{name} must be null")
        return
    raise PlanarStartupHandoffError(  # pragma: no cover - internal schema error
        f"{name} has unsupported recomputed type {type(recomputed).__name__}"
    )


def _resolve_handoff_paths(path: Path | str) -> tuple[Path, Path]:
    target = Path(path)
    if target.is_dir():
        return target / HANDOFF_JSON_NAME, target / HANDOFF_NPZ_NAME
    if target.name == HANDOFF_JSON_NAME:
        return target, target.with_name(HANDOFF_NPZ_NAME)
    if target.name == HANDOFF_NPZ_NAME:
        return target.with_name(HANDOFF_JSON_NAME), target
    raise PlanarStartupHandoffError(
        "handoff path must be the output directory, handoff.json or handoff.npz; "
        f"got {target}"
    )


def _read_report(json_path: Path) -> Mapping[str, Any]:
    if not json_path.is_file():
        raise PlanarStartupHandoffError(f"missing handoff report: {json_path}")
    try:
        report = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanarStartupHandoffError(
            f"could not read JSON handoff report {json_path}: {exc}"
        ) from exc
    if not isinstance(report, Mapping):
        raise PlanarStartupHandoffError(f"{json_path} must contain a JSON object")
    return report


def _read_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.is_file():
        raise PlanarStartupHandoffError(f"missing handoff arrays: {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            files = set(payload.files)
            missing = sorted(set(_REQUIRED_NPZ_KEYS) - files)
            if missing:
                raise PlanarStartupHandoffError(
                    f"{npz_path} is missing required array(s): {missing}"
                )
            arrays: dict[str, np.ndarray] = {}
            for key in _REQUIRED_NPZ_KEYS:
                array = np.array(payload[key], copy=True)
                if array.dtype.hasobject:
                    raise PlanarStartupHandoffError(
                        f"{npz_path} array {key!r} has object dtype"
                    )
                arrays[key] = array
            return arrays
    except PlanarStartupHandoffError:
        raise
    except Exception as exc:  # zip, pickle-disabled or dtype failure
        raise PlanarStartupHandoffError(
            f"could not read NPZ handoff arrays {npz_path}: {exc}"
        ) from exc


def _validate_format(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    if "handoff_format" not in report:
        raise PlanarStartupHandoffError(
            "handoff.json is missing required field 'handoff_format'"
        )
    if report["handoff_format"] != PLANAR_STARTUP_HANDOFF_FORMAT:
        raise PlanarStartupHandoffError(
            f"unsupported handoff format {report['handoff_format']!r}; "
            f"expected {PLANAR_STARTUP_HANDOFF_FORMAT!r}"
        )
    if "handoff_format_version" not in report:
        raise PlanarStartupHandoffError(
            "handoff.json is missing required field 'handoff_format_version'"
        )
    version = report["handoff_format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise PlanarStartupHandoffError("handoff_format_version must be an integer")
    if version != PLANAR_STARTUP_HANDOFF_FORMAT_VERSION:
        raise PlanarStartupHandoffError(
            f"unsupported handoff format version {version!r}; "
            f"expected {PLANAR_STARTUP_HANDOFF_FORMAT_VERSION}"
        )
    saved_version = _array_integer(arrays, "handoff_format_version")
    if saved_version != version:
        raise PlanarStartupHandoffError(
            "handoff.json and handoff.npz disagree on handoff_format_version: "
            f"{version!r} vs {saved_version!r}"
        )


def _validate_versions(report: Mapping[str, Any]) -> None:
    model_id = _required_string(report, "model_id", "handoff.json")
    if model_id != PLANAR_STARTUP_MODEL_ID:
        raise PlanarStartupHandoffError(
            f"unsupported handoff model {model_id!r}; expected {PLANAR_STARTUP_MODEL_ID!r}"
        )
    method_version = _required_string(report, "method_version", "handoff.json")
    if method_version != PLANAR_STARTUP_METHOD_VERSION:
        raise PlanarStartupHandoffError(
            f"unsupported startup method {method_version!r}; "
            f"expected {PLANAR_STARTUP_METHOD_VERSION!r}"
        )
    riemann_version = _required_string(report, "riemann_method_version", "handoff.json")
    if riemann_version != RIEMANN_METHOD_VERSION:
        raise PlanarStartupHandoffError(
            f"unsupported Riemann method {riemann_version!r}; "
            f"expected {RIEMANN_METHOD_VERSION!r}"
        )


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    model_id: str,
    method_version: str,
    riemann_version: str,
) -> tuple[float, float]:
    for key, expected in (
        ("model_id", model_id),
        ("method_version", method_version),
        ("riemann_method_version", riemann_version),
    ):
        actual = _required_string(metadata, key, "handoff.json.metadata")
        if actual != expected:
            raise PlanarStartupHandoffError(
                f"handoff.json.metadata.{key} is {actual!r}, expected {expected!r}"
            )
    if metadata.get("reference_field") is not False:
        raise PlanarStartupHandoffError(
            "handoff metadata must mark reference_field=false"
        )
    if metadata.get("training_qualified") is not False:
        raise PlanarStartupHandoffError(
            "handoff metadata must mark training_qualified=false"
        )
    grid = _required_mapping(metadata, "grid_config", "handoff.json.metadata")
    for key in (
        "inner_cells",
        "fan_cells",
        "plateau_cells",
        "air_cells",
        "ambient_cells",
        "dense_samples",
    ):
        value = _required_integer(grid, key, "handoff.json.metadata.grid_config")
        if value < 1:
            raise PlanarStartupHandoffError(
                f"handoff.json.metadata.grid_config.{key} must be >= 1"
            )
    step = _required_number(
        grid, "integration_step_factor", "handoff.json.metadata.grid_config"
    )
    if step <= 0.0:
        raise PlanarStartupHandoffError(
            "handoff.json.metadata.grid_config.integration_step_factor must be positive"
        )
    root_tolerance = _required_number(
        grid, "root_relative_tolerance", "handoff.json.metadata.grid_config"
    )
    if root_tolerance <= 0.0:
        raise PlanarStartupHandoffError(
            "handoff.json.metadata.grid_config.root_relative_tolerance must be positive"
        )
    return step, root_tolerance


def _initial_state(report: Mapping[str, Any]) -> dict[str, float]:
    raw = _required_mapping(report, "initial_state", "handoff.json")
    missing = sorted(set(_INITIAL_STATE_KEYS) - set(raw))
    if missing:
        raise PlanarStartupHandoffError(
            f"handoff.json.initial_state is missing required field(s): {missing}"
        )
    state: dict[str, float] = {}
    for key in _INITIAL_STATE_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanarStartupHandoffError(
                f"handoff.json.initial_state.{key} must be a real number"
            )
        number = float(value)
        if not math.isfinite(number):
            raise PlanarStartupHandoffError(
                f"handoff.json.initial_state.{key} must be finite"
            )
        state[key] = number
    return state


def write_planar_startup_handoff(
    field: PlanarStartupField, directory: Path | str
) -> tuple[Path, Path]:
    """Write ``handoff.json`` and ``handoff.npz`` for a validated startup field.

    The pair is the only supported round-trip format for this startup.  It is
    versioned and keeps both the saved physical arrays and enough parameters to
    reconstruct the analytic sampler; no legacy ``ReferenceField`` cache is
    written.
    """
    if not isinstance(field, PlanarStartupField):
        raise PlanarStartupHandoffError(
            "field must be a PlanarStartupField built by build_planar_startup_field"
        )
    out_dir = Path(directory)
    if out_dir.exists() and not out_dir.is_dir():
        raise PlanarStartupHandoffError(f"handoff output path is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / HANDOFF_JSON_NAME
    npz_path = out_dir / HANDOFF_NPZ_NAME

    report = field.as_dict()
    report["handoff_format"] = PLANAR_STARTUP_HANDOFF_FORMAT
    report["handoff_format_version"] = PLANAR_STARTUP_HANDOFF_FORMAT_VERSION
    report["metadata"] = dict(field.metadata)

    arrays: dict[str, np.ndarray] = {
        "radii": field.radii,
        "cell_volume": field.cell_volume,
        "density": field.density,
        "velocity": field.velocity,
        "specific_energy": field.specific_energy,
        "pressure": field.pressure,
        "sound_speed": field.sound_speed,
        "material": field.material,
        "region": field.region,
        "mass": field.mass,
        "node_velocity": field.node_velocity,
        "epsilon": np.asarray(field.epsilon, dtype=np.float64),
        "delta": np.asarray(field.delta, dtype=np.float64),
        "contact_radius": np.asarray(field.contact_radius, dtype=np.float64),
        "shock_radius": np.asarray(field.shock_radius, dtype=np.float64),
        "contact_speed": np.asarray(field.contact_speed, dtype=np.float64),
        "shock_speed": np.asarray(field.shock_speed, dtype=np.float64),
        "handoff_format_version": np.asarray(
            PLANAR_STARTUP_HANDOFF_FORMAT_VERSION, dtype=np.int64
        ),
    }
    try:
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=float),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PlanarStartupHandoffError(
            f"could not write handoff pair in {out_dir}: {exc}"
        ) from exc
    return json_path, npz_path


def load_planar_startup_handoff(path: Path | str) -> PlanarStartupField:
    """Load and validate a planar-startup handoff pair.

    ``path`` may be the output directory, ``handoff.json`` or ``handoff.npz``.
    Missing fields, an unsupported version, inconsistent JSON/NPZ scalars, a
    Riemann reconstruction that disagrees with the saved solution, any saved
    array that disagrees with the common analytic sampler, and any saved
    diagnostic that disagrees with an independent recomputation are rejected
    explicitly.
    """
    json_path, npz_path = _resolve_handoff_paths(path)
    report = _read_report(json_path)
    arrays = _read_arrays(npz_path)
    _validate_format(report, arrays)

    missing = sorted(set(_REQUIRED_REPORT_KEYS) - set(report))
    if missing:
        raise PlanarStartupHandoffError(
            f"{json_path} is missing required field(s): {missing}"
        )
    _validate_versions(report)

    if report["reference_field"] is not False:
        raise PlanarStartupHandoffError("handoff.json must mark reference_field=false")
    if report["training_qualified"] is not False:
        raise PlanarStartupHandoffError(
            "handoff.json must mark training_qualified=false"
        )
    if not isinstance(report["units"], str) or not report["units"]:
        raise PlanarStartupHandoffError("handoff.json.units must be a non-empty string")

    for json_key, array_key in _SCALAR_CROSS_CHECKS:
        reported = _required_number(report, json_key, "handoff.json")
        stored = _array_scalar(arrays, array_key)
        _require_close_scalar(
            f"handoff.json.{json_key} vs handoff.npz.{array_key}",
            reported,
            stored,
            rel_tol=_SCALAR_MATCH_REL_TOL,
            abs_tol=_SCALAR_MATCH_ABS_TOL,
        )

    eos_payload = _required_mapping(report, "product_eos", "handoff.json")
    try:
        product_eos = JWLParameters.from_mapping(eos_payload)
    except (EOSError, TypeError, ValueError) as exc:
        raise PlanarStartupHandoffError(
            f"handoff.json.product_eos is invalid: {exc}"
        ) from exc
    air_gamma = _required_number(report, "air_gamma", "handoff.json")
    initial_state = _initial_state(report)

    metadata = dict(_required_mapping(report, "metadata", "handoff.json"))
    step, root_tolerance = _validate_metadata(
        metadata,
        model_id=_required_string(report, "model_id", "handoff.json"),
        method_version=_required_string(report, "method_version", "handoff.json"),
        riemann_version=_required_string(
            report, "riemann_method_version", "handoff.json"
        ),
    )

    try:
        solution = solve_jwl_air_riemann(
            product_eos=product_eos,
            product_density=initial_state["product_density"],
            product_specific_energy=initial_state["product_specific_energy"],
            air_density=initial_state["air_density"],
            air_pressure=initial_state["air_pressure"],
            air_gamma=air_gamma,
            integration_step_factor=step,
            root_relative_tolerance=root_tolerance,
        )
    except RiemannError as exc:
        raise PlanarStartupHandoffError(
            f"saved parameters do not reproduce the Riemann solution: {exc}"
        ) from exc

    saved_riemann = _required_mapping(report, "riemann", "handoff.json")
    _require_tree_close("handoff.json.riemann", solution.as_dict(), saved_riemann)

    checks = dict(_required_mapping(report, "checks", "handoff.json"))
    integrals = dict(_required_mapping(report, "integrals", "handoff.json"))

    try:
        field = PlanarStartupField(
            model_id=_required_string(report, "model_id", "handoff.json"),
            method_version=_required_string(report, "method_version", "handoff.json"),
            time=_required_number(report, "physical_time_s", "handoff.json"),
            pin_time=_required_number(report, "pin_time_s", "handoff.json"),
            charge_radius=_required_number(report, "charge_radius_m", "handoff.json"),
            epsilon=_required_number(report, "epsilon_s", "handoff.json"),
            delta=_required_number(report, "delta", "handoff.json"),
            contact_radius=_required_number(report, "contact_radius_m", "handoff.json"),
            shock_radius=_required_number(report, "shock_radius_m", "handoff.json"),
            contact_speed=_required_number(report, "contact_speed_m_s", "handoff.json"),
            shock_speed=_required_number(report, "shock_speed_m_s", "handoff.json"),
            rarefaction_head_radius=_required_number(
                report, "rarefaction_head_radius_m", "handoff.json"
            ),
            rarefaction_tail_radius=_required_number(
                report, "rarefaction_tail_radius_m", "handoff.json"
            ),
            rarefaction_head_speed=_required_number(
                report, "rarefaction_head_speed_m_s", "handoff.json"
            ),
            rarefaction_tail_speed=_required_number(
                report, "rarefaction_tail_speed_m_s", "handoff.json"
            ),
            radii=arrays["radii"],
            cell_volume=arrays["cell_volume"],
            density=arrays["density"],
            velocity=arrays["velocity"],
            specific_energy=arrays["specific_energy"],
            pressure=arrays["pressure"],
            sound_speed=arrays["sound_speed"],
            material=arrays["material"],
            region=arrays["region"],
            mass=arrays["mass"],
            node_velocity=arrays["node_velocity"],
            product_eos=product_eos,
            air_gamma=air_gamma,
            initial_state=initial_state,
            solution=solution,
            checks=checks,
            integrals=integrals,
            metadata=metadata,
        )
    except PlanarStartupError as exc:
        raise PlanarStartupHandoffError(
            f"handoff arrays fail the startup consistency validation: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise PlanarStartupHandoffError(
            f"handoff arrays are not a valid startup field: {exc}"
        ) from exc

    try:
        recomputed_checks, recomputed_integrals = compute_planar_startup_diagnostics(
            field
        )
    except PlanarStartupError as exc:
        raise PlanarStartupHandoffError(
            f"could not recompute the handoff diagnostics: {exc}"
        ) from exc
    if recomputed_checks.get("passed") is not True:
        raise PlanarStartupHandoffError(
            "recomputed startup diagnostics do not pass; refusing the handoff"
        )
    _require_diagnostics_match(
        "handoff.json.checks", checks, recomputed_checks
    )
    _require_diagnostics_match(
        "handoff.json.integrals", integrals, recomputed_integrals
    )
    recomputed_groups = {
        "checks": recomputed_checks,
        "integrals": recomputed_integrals,
    }
    for report_key, group, key in _TOP_LEVEL_DIAGNOSTICS:
        _require_diagnostics_match(
            f"handoff.json.{report_key}",
            report[report_key],
            recomputed_groups[group][key],
        )

    saved_cache_key = _required_string(report, "cache_key", "handoff.json")
    if field.cache_key() != saved_cache_key:
        raise PlanarStartupHandoffError(
            "handoff.cache_key does not match the loaded parameters and arrays"
        )
    return field


__all__ = [
    "HANDOFF_JSON_NAME",
    "HANDOFF_NPZ_NAME",
    "PLANAR_STARTUP_HANDOFF_FORMAT",
    "PLANAR_STARTUP_HANDOFF_FORMAT_VERSION",
    "PlanarStartupHandoffError",
    "load_planar_startup_handoff",
    "write_planar_startup_handoff",
]
