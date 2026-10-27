"""The material and ambient states a startup field actually carries.

A :class:`~...two_medium_startup_planar.PlanarStartupField` already stores the
JWL parameter set, the air ``gamma`` and the initial product/air
thermodynamic state that were used to build it.  Those saved values are the
authoritative physical conditions for a run.  This module is the single
definition used by:

* training assembly (field vs. YAML, field vs. field, field vs. model);
* checkpoint save/load (the ``material`` case record);
* frozen prediction (checkpoint vs. a new-radius field).

The distinction between a *physical condition* and a *normalization scale* is
deliberate.  ``rho_*``, ``U_*`` and ``e_*`` are numerical scales; they are not
forced to equal the actual initial state.  The JWL EOS, air ``gamma``, ambient
density/pressure and the initial product/air state are physical and must be
bound exactly (within round-off).

Only the initial-state keys that describe thermodynamics are compared.  The
charge radius and the outer radius are geometry: they legitimately differ
between radii and are checked elsewhere.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Mapping

from . import material as mat
from .eos import EOSError, JWLParameters

#: Keys of ``PlanarStartupField.initial_state`` that are physical material /
#: environment conditions rather than geometry.
INITIAL_STATE_KEYS = (
    "product_density",
    "product_specific_energy",
    "product_pressure",
    "product_sound_speed",
    "air_density",
    "air_specific_energy",
    "air_pressure",
    "air_sound_speed",
)

#: Full JWL parameter set.  ``E0`` is not part of the EOS object; it is checked
#: through the initial product specific energy, ``e_p0 = E0 / rho0``.
JWL_KEYS = ("A", "B", "R1", "R2", "omega", "rho0", "energy_shift")

#: Configuration keys that describe physical conditions.  The base file also
#: carries explicitly reference-only diagnostics (``detonation_velocity_reference``
#: and ``p_cj_reference``); those are not used by the field or the model and are
#: deliberately not compared here.
CONFIG_JWL_KEYS = JWL_KEYS + ("E0_volume_energy", "V0")
CONFIG_AIR_KEYS = ("gamma", "rho", "pressure")

#: Explicitly reference-only diagnostics accepted in the base file.  They are
#: not physical inputs and are not compared.
REFERENCE_ONLY_JWL_KEYS = ("detonation_velocity_reference", "p_cj_reference")

#: Relative tolerance for physical-condition comparisons.  Values produced by the
#: same numerical path are bitwise equal; this only absorbs platform libm
#: differences and the E0 <-> e_p0 round trip.
CONDITION_RELATIVE_TOLERANCE = 1.0e-9


class MaterialConditionsError(ValueError):
    """A field, configuration or model carries inconsistent physical conditions."""


def _finite(name: str, value: Any) -> float:
    """A real scalar from a field/model record (strings are not accepted)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise MaterialConditionsError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise MaterialConditionsError(f"{name} must be finite, got {value!r}")
    return number


def _config_number(name: str, value: Any) -> float:
    """A YAML scalar.

    The base file writes values such as ``3.7120e11``; PyYAML's YAML 1.1
    resolver returns that exponent form as a string.  Coercing it is not
    "accepting garbage": the string must parse as a finite float, and bools are
    still rejected.  The field/model side continues to use :func:`_finite` and
    does not accept strings.
    """
    if isinstance(value, bool):
        raise MaterialConditionsError(f"{name} must be a real number, got {value!r}")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise MaterialConditionsError(
                f"{name} must be a real number, got {value!r}"
            ) from exc
    return _finite(name, value)


def _positive(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise MaterialConditionsError(f"{name} must be positive, got {value!r}")
    return number


def _config_positive(name: str, value: Any) -> float:
    number = _config_number(name, value)
    if number <= 0.0:
        raise MaterialConditionsError(f"{name} must be positive, got {value!r}")
    return number


def _same(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= CONDITION_RELATIVE_TOLERANCE * max(
        abs(float(left)), abs(float(right)), 1.0e-30
    )


def field_material_conditions(field: Any) -> dict[str, Any]:
    """The authoritative conditions stored with a startup field.

    The record has the same shape wherever it is compared::

        {"jwl": {...}, "air_gamma": ..., "initial_state": {...}}

    ``JWLParameters.as_dict()`` is used so that a checkpoint and a freshly
    loaded handoff define the EOS in exactly the same terms.
    """
    eos = getattr(field, "product_eos", None)
    if not isinstance(eos, JWLParameters):
        raise MaterialConditionsError(
            f"startup field product_eos is {type(eos).__name__}, not JWLParameters"
        )
    try:
        eos.validate()
    except EOSError as exc:
        raise MaterialConditionsError(f"startup field product_eos is invalid: {exc}") from exc
    state = getattr(field, "initial_state", None)
    if not isinstance(state, Mapping):
        raise MaterialConditionsError("startup field has no initial_state mapping")
    missing = sorted(set(INITIAL_STATE_KEYS) - set(state))
    if missing:
        raise MaterialConditionsError(
            f"startup field initial_state is missing physical key(s): {missing}"
        )
    gamma = _finite("startup field air_gamma", getattr(field, "air_gamma", None))
    if gamma <= 1.0:
        raise MaterialConditionsError(
            f"startup field air_gamma must exceed 1, got {gamma!r}"
        )
    return {
        "jwl": {key: _finite(f"jwl.{key}", eos.as_dict()[key]) for key in JWL_KEYS},
        "air_gamma": gamma,
        "initial_state": {
            key: _finite(f"initial_state.{key}", state[key])
            for key in INITIAL_STATE_KEYS
        },
    }


def model_material_conditions(model: Any) -> dict[str, Any]:
    """The EOS and ambient state a coupled model actually uses.

    The model does not carry the initial product state (that belongs to the
    startup field).  Its record therefore contains only the JWL set and the
    air state; :func:`condition_mismatches` compares the fields that both
    records define.
    """
    product_net = getattr(model, "product_net", None)
    air_net = getattr(model, "air_net", None)
    ambient = getattr(model, "undisturbed", None)
    if product_net is None or air_net is None or ambient is None:
        raise MaterialConditionsError(
            "model does not expose product_net, air_net and undisturbed"
        )
    eos = getattr(product_net, "jwl", None)
    if not isinstance(eos, JWLParameters):
        raise MaterialConditionsError(
            f"model ProductNet.jwl is {type(eos).__name__}, not JWLParameters"
        )
    gamma = _finite("model air gamma", getattr(air_net, "gamma_air", None))
    density = _positive("model ambient density", getattr(ambient, "density", None))
    pressure = _positive("model ambient pressure", getattr(ambient, "pressure", None))
    if gamma <= 1.0:
        raise MaterialConditionsError(f"model air gamma must exceed 1, got {gamma!r}")
    specific_energy = pressure / ((gamma - 1.0) * density)
    sound_speed = math.sqrt(gamma * pressure / density)
    return {
        "jwl": {key: _finite(f"jwl.{key}", eos.as_dict()[key]) for key in JWL_KEYS},
        "air_gamma": gamma,
        "initial_state": {
            "air_density": density,
            "air_specific_energy": specific_energy,
            "air_pressure": pressure,
            "air_sound_speed": sound_speed,
        },
    }


def validate_material_record(record: Any, *, where: str) -> None:
    """Require a **complete** material record (checkpoint / frozen prediction).

    ``condition_mismatches`` deliberately compares only keys present on both
    sides, which is what lets an air-only model record be checked.  That also
    means a truncated checkpoint record would silently skip the missing
    product-state keys.  Checkpoint records therefore go through this stricter
    validator before any comparison.
    """
    if not isinstance(record, Mapping):
        raise MaterialConditionsError(f"{where}: material record is not a mapping")
    jwl = record.get("jwl")
    if not isinstance(jwl, Mapping):
        raise MaterialConditionsError(f"{where}: material record has no JWL mapping")
    missing_jwl = sorted(set(JWL_KEYS) - set(jwl))
    if missing_jwl:
        raise MaterialConditionsError(
            f"{where}: material record JWL is missing {missing_jwl}"
        )
    for key in JWL_KEYS:
        _finite(f"{where} JWL.{key}", jwl[key])
    if "air_gamma" not in record:
        raise MaterialConditionsError(f"{where}: material record has no air_gamma")
    gamma = _finite(f"{where} air_gamma", record["air_gamma"])
    if gamma <= 1.0:
        raise MaterialConditionsError(
            f"{where}: material record air_gamma must exceed 1, got {gamma!r}"
        )
    state = record.get("initial_state")
    if not isinstance(state, Mapping):
        raise MaterialConditionsError(
            f"{where}: material record has no initial_state mapping"
        )
    missing_state = sorted(set(INITIAL_STATE_KEYS) - set(state))
    if missing_state:
        raise MaterialConditionsError(
            f"{where}: material record initial_state is missing {missing_state}"
        )
    for key in INITIAL_STATE_KEYS:
        _finite(f"{where} initial_state.{key}", state[key])


def condition_mismatches(
    reference: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    reference_label: str = "reference",
    actual_label: str = "actual",
) -> list[str]:
    """Compare two condition records; return human-readable differences.

    The comparison is deliberately key-driven rather than "the initial-front
    ratios happen to match".  Only keys present in both ``initial_state``
    mappings are compared, which lets a model record (air-only) be checked
    against a full field/checkpoint record.
    """
    problems: list[str] = []
    ref_jwl = reference.get("jwl")
    act_jwl = actual.get("jwl")
    if not isinstance(ref_jwl, Mapping) or not isinstance(act_jwl, Mapping):
        problems.append(f"{reference_label} or {actual_label} has no JWL mapping")
    else:
        for key in JWL_KEYS:
            if key not in ref_jwl or key not in act_jwl:
                problems.append(
                    f"JWL.{key} is missing from {reference_label if key not in ref_jwl else actual_label}"
                )
                continue
            left = _finite(f"{reference_label} JWL.{key}", ref_jwl[key])
            right = _finite(f"{actual_label} JWL.{key}", act_jwl[key])
            if not _same(left, right):
                problems.append(
                    f"JWL.{key}: {reference_label}={left!r}, {actual_label}={right!r}"
                )
    for key in ("air_gamma",):
        if key not in reference or key not in actual:
            problems.append(f"{key} is missing from one condition record")
            continue
        left = _finite(f"{reference_label} {key}", reference[key])
        right = _finite(f"{actual_label} {key}", actual[key])
        if not _same(left, right):
            problems.append(
                f"{key}: {reference_label}={left!r}, {actual_label}={right!r}"
            )
    ref_state = reference.get("initial_state")
    act_state = actual.get("initial_state")
    if not isinstance(ref_state, Mapping) or not isinstance(act_state, Mapping):
        problems.append(f"{reference_label} or {actual_label} has no initial_state mapping")
        return problems
    for key in INITIAL_STATE_KEYS:
        if key not in ref_state or key not in act_state:
            continue
        left = _finite(f"{reference_label} initial_state.{key}", ref_state[key])
        right = _finite(f"{actual_label} initial_state.{key}", act_state[key])
        if not _same(left, right):
            problems.append(
                f"initial_state.{key}: {reference_label}={left!r}, "
                f"{actual_label}={right!r}"
            )
    return problems


def _validate_config_sections(config: Any) -> None:
    """Reject unrecognised physical keys instead of silently ignoring them.

    A typo such as ``jwl.A_custom`` would otherwise leave the default EOS in
    force while the run looks configured.  Reference-only diagnostics that the
    base file documents are accepted and deliberately not compared.
    """
    known_jwl = set(JWL_KEYS) | set(CONFIG_JWL_KEYS) | set(REFERENCE_ONLY_JWL_KEYS)
    unknown_jwl = sorted(set(config.section("jwl")) - known_jwl)
    if unknown_jwl:
        raise MaterialConditionsError(
            f"configuration jwl has unknown physical key(s) {unknown_jwl}; refusing "
            "to ignore them"
        )
    unknown_air = sorted(set(config.section("air")) - set(CONFIG_AIR_KEYS))
    if unknown_air:
        raise MaterialConditionsError(
            f"configuration air has unknown physical key(s) {unknown_air}; refusing "
            "to ignore them"
        )


def explicit_config_conditions(config: Any) -> dict[str, Any]:
    """Physical values explicitly present in the YAML.

    Absent keys mean "the handoff is authoritative", not "use a default and
    compare".  This distinction matters when a handoff carries a custom JWL:
    a configuration that simply omits ``jwl`` must not force it back to TNT.
    """
    _validate_config_sections(config)
    result: dict[str, Any] = {}
    jwl = config.section("jwl")
    for key in CONFIG_JWL_KEYS:
        if key in jwl:
            result[f"jwl.{key}"] = _config_number(f"jwl.{key}", jwl[key])
    air = config.section("air")
    for key in CONFIG_AIR_KEYS:
        if key in air:
            result[f"air.{key}"] = _config_number(f"air.{key}", air[key])
    return result


def configured_jwl_and_energy(config: Any) -> tuple[JWLParameters, float]:
    """Build the JWL set and ``E0`` that a freshly built initial field must use.

    Missing JWL keys fall back to :func:`material.product_jwl`; explicitly
    present keys override them.  ``E0`` is the LS-DYNA per-reference-volume
    energy and is used through ``e_p0 = E0 / rho0``.  ``V0`` is not carried by
    :class:`JWLParameters` and is only supported at 1; a non-default value is
    refused instead of silently ignored.
    """
    _validate_config_sections(config)
    section = config.section("jwl")
    base = mat.product_jwl().as_dict()
    merged = dict(base)
    for key in JWL_KEYS:
        if key in section:
            merged[key] = _config_number(f"jwl.{key}", section[key])
    try:
        params = JWLParameters.from_mapping(merged)
    except (EOSError, TypeError, ValueError) as exc:
        raise MaterialConditionsError(f"configuration jwl is invalid: {exc}") from exc

    if "V0" in section:
        v0 = _config_number("jwl.V0", section["V0"])
        if not _same(v0, 1.0):
            raise MaterialConditionsError(
                f"configuration jwl.V0={v0!r} is not supported: the JWL implementation "
                "uses the LS-DYNA V0=1 reference volume and the startup field does not "
                "carry V0. Refusing to ignore a non-default value"
            )
    energy = (
        _config_positive("jwl.E0_volume_energy", section["E0_volume_energy"])
        if "E0_volume_energy" in section
        else float(mat.PRODUCT_VOLUME_ENERGY)
    )
    return params, energy


def _snap_to_card_default(value: float, default: float) -> float:
    """Keep the existing default startup field bitwise unchanged.

    ``base.yaml`` writes the card pressure as ``101320.0`` while
    ``material.AIR_PRESSURE`` is the card-derived
    ``101319.99999999999`` Pa.  They agree to 1e-16 relative and denote the
    same physical state.  When an explicit configuration value is numerically
    the card default, use the card's exact internal value so the default field
    arrays and Riemann state remain bitwise identical.  A genuinely different
    value is used as written.
    """
    return float(default) if _same(value, default) else float(value)


def configured_air_state(config: Any) -> tuple[float, float, float]:
    """``(gamma, density, pressure)`` for a freshly built initial field."""
    _validate_config_sections(config)
    section = config.section("air")
    gamma = _config_positive(
        "air.gamma", section.get("gamma", mat.AIR_GAMMA)
    )
    if gamma <= 1.0:
        raise MaterialConditionsError(
            f"configuration air.gamma must exceed 1, got {gamma!r}"
        )
    density = _config_positive("air.rho", section.get("rho", mat.AIR_DENSITY))
    pressure = _config_positive(
        "air.pressure", section.get("pressure", mat.AIR_PRESSURE)
    )
    return (
        _snap_to_card_default(gamma, mat.AIR_GAMMA),
        _snap_to_card_default(density, mat.AIR_DENSITY),
        _snap_to_card_default(pressure, mat.AIR_PRESSURE),
    )


def config_condition_mismatches(
    field: Any, config: Any, *, where: str = "startup field"
) -> list[str]:
    """Compare an authoritative field with explicitly configured physical values.

    The field is authoritative.  A configuration value that disagrees is an
    error naming both values, never a silent override in either direction.
    ``E0`` is compared through ``product_specific_energy * rho0``.
    """
    conditions = field_material_conditions(field)
    values = explicit_config_conditions(config)
    problems: list[str] = []
    for key in JWL_KEYS:
        path = f"jwl.{key}"
        if path not in values:
            continue
        expected = values[path]
        actual = float(conditions["jwl"][key])
        if not _same(expected, actual):
            problems.append(
                f"configuration {path}={expected!r}, but {where} product_eos {key}={actual!r}"
            )
    if "jwl.E0_volume_energy" in values:
        expected = values["jwl.E0_volume_energy"]
        actual = (
            float(conditions["initial_state"]["product_specific_energy"])
            * float(conditions["jwl"]["rho0"])
        )
        if not _same(expected, actual):
            problems.append(
                "configuration jwl.E0_volume_energy="
                f"{expected!r}, but {where} implies "
                f"e_p0*rho0={actual!r} "
                "(initial_state.product_specific_energy * product_eos.rho0)"
            )
    if "jwl.V0" in values:
        v0 = values["jwl.V0"]
        if not _same(v0, 1.0):
            problems.append(
                f"configuration jwl.V0={v0!r} is not supported by the current JWL "
                "implementation (LS-DYNA V0=1); refusing to ignore it"
            )
    if "air.gamma" in values:
        expected = values["air.gamma"]
        actual = float(conditions["air_gamma"])
        if not _same(expected, actual):
            problems.append(
                f"configuration air.gamma={expected!r}, but {where} air_gamma={actual!r}"
            )
    for config_key, state_key in (
        ("air.rho", "air_density"),
        ("air.pressure", "air_pressure"),
    ):
        if config_key not in values:
            continue
        expected = values[config_key]
        actual = float(conditions["initial_state"][state_key])
        if not _same(expected, actual):
            problems.append(
                f"configuration {config_key}={expected!r}, but {where} "
                f"initial_state.{state_key}={actual!r}"
            )
    return problems


__all__ = [
    "CONDITION_RELATIVE_TOLERANCE",
    "CONFIG_AIR_KEYS",
    "CONFIG_JWL_KEYS",
    "INITIAL_STATE_KEYS",
    "JWL_KEYS",
    "MaterialConditionsError",
    "condition_mismatches",
    "config_condition_mismatches",
    "configured_air_state",
    "configured_jwl_and_energy",
    "explicit_config_conditions",
    "field_material_conditions",
    "model_material_conditions",
    "validate_material_record",
]
